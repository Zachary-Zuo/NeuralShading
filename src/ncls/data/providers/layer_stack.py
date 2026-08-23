from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ncls.core.material import material_program_from_layer_stack
from ncls.data.collector import CollectionConfig
from ncls.data.contract import EvaluatedBlock, PositionKind, QueryPlan, ReferenceDescriptor, SourceState, SurfaceSample, make_state_id
from ncls.data.priors import (
    E0_LAYER_STACK_BOUNDARY_CASE_IDS,
    E0_LAYER_STACK_BOUNDARY_PROFILE_ID,
    E1_LAYER_STACK_MULTI_INTERFACE_PROFILE_ID,
    E1_LAYER_STACK_NARROW_CONDUCTOR_PROFILE_ID,
    E2_LAYER_STACK_SHARED_DECODER_FAMILY_COUNT,
    E2_LAYER_STACK_SHARED_DECODER_LOCAL_STATE_COUNT,
    E2_LAYER_STACK_SHARED_DECODER_PROFILE_ID,
    LAYER_STACK_RESEARCH_PRIOR_ID,
    LAYER_STACK_STATE_PROFILE_IDS,
    e0_layer_stack_boundary_cases,
    e1_layer_stack_multi_interface_cases,
    e1_layer_stack_narrow_conductor_cases,
    e2_layer_stack_shared_decoder_families,
    sample_stack_families,
)
from ncls.data.reference import FalcorReferenceEvaluator, evaluate_reference_adaptive, evaluate_reference_fixed

from .base import BaseProvider, PROJECT_ROOT, assign_group_splits, implementation_hash


@dataclass(frozen=True)
class LayerStackProviderConfig:
    family_count: int = 8
    local_state_count: int = 4
    samples_per_replica: int = 64
    query_group_batch: int = 64
    max_depth: int = 64
    adaptive: bool = False
    batch_samples: int = 256
    min_samples: int = 512
    max_samples: int = 16384
    relative_standard_error: float = 0.03
    state_profile_id: str = LAYER_STACK_RESEARCH_PRIOR_ID

    def __post_init__(self) -> None:
        values = (
            self.family_count, self.local_state_count, self.samples_per_replica, self.query_group_batch,
            self.max_depth, self.batch_samples, self.min_samples, self.max_samples,
        )
        if min(values) < 1 or self.max_samples < self.min_samples:
            raise ValueError("LayerStack provider sizes must be positive and ordered")
        if self.adaptive and (self.min_samples % self.batch_samples or self.max_samples % self.batch_samples):
            raise ValueError("adaptive sample limits must be multiples of batch_samples")
        if not 0.0 < self.relative_standard_error < 1.0:
            raise ValueError("relative_standard_error must lie in (0, 1)")
        if self.state_profile_id not in LAYER_STACK_STATE_PROFILE_IDS:
            raise ValueError(f"unknown LayerStack state profile {self.state_profile_id!r}")
        fixed_profile_counts = {
            E0_LAYER_STACK_BOUNDARY_PROFILE_ID: len(E0_LAYER_STACK_BOUNDARY_CASE_IDS),
            E1_LAYER_STACK_NARROW_CONDUCTOR_PROFILE_ID: 1,
            E1_LAYER_STACK_MULTI_INTERFACE_PROFILE_ID: 1,
            E2_LAYER_STACK_SHARED_DECODER_PROFILE_ID: E2_LAYER_STACK_SHARED_DECODER_FAMILY_COUNT,
        }
        if self.state_profile_id in fixed_profile_counts:
            expected = fixed_profile_counts[self.state_profile_id]
            if self.family_count != expected:
                raise ValueError(
                    f"{self.state_profile_id} requires family_count={expected}"
                )
            expected_local_states = (
                E2_LAYER_STACK_SHARED_DECODER_LOCAL_STATE_COUNT
                if self.state_profile_id == E2_LAYER_STACK_SHARED_DECODER_PROFILE_ID
                else 1
            )
            if self.local_state_count != expected_local_states:
                raise ValueError(
                    f"{self.state_profile_id} requires local_state_count={expected_local_states}"
                )


class LayerStackProvider(BaseProvider):
    def __init__(
        self,
        collection: CollectionConfig,
        config: LayerStackProviderConfig = LayerStackProviderConfig(),
        *,
        evaluator: Any | None = None,
    ) -> None:
        super().__init__(collection)
        self.provider_config = config
        source_paths = (
            Path(__file__),
            PROJECT_ROOT / "src/ncls/data/providers/base.py",
            PROJECT_ROOT / "src/ncls/data/directions.py",
            PROJECT_ROOT / "src/ncls/data/priors.py",
            PROJECT_ROOT / "src/ncls/data/reference.py",
            PROJECT_ROOT / "shaders/ncls/contracts/layer_stack_ir.slang",
            PROJECT_ROOT / "shaders/ncls/reference/sampling.slang",
            PROJECT_ROOT / "shaders/ncls/reference/interfaces.slang",
            PROJECT_ROOT / "shaders/ncls/reference/random_walk_reference.slang",
            PROJECT_ROOT / "shaders/ncls/data/reference_layer_stack.cs.slang",
        )
        self.descriptor = ReferenceDescriptor(
            "ncls.layer-stack@1",
            "ncls.layer-stack-random-walk@1",
            "ncls.material-program@1",
            incident_domain="upper-hemisphere",
            position_kind=PositionKind.CONSTANT,
            deterministic=False,
            capabilities=("evaluate", "monte-carlo-moments"),
            implementation_sha256=implementation_hash(source_paths),
        )
        if config.state_profile_id == E0_LAYER_STACK_BOUNDARY_PROFILE_ID:
            cases = e0_layer_stack_boundary_cases()
        elif config.state_profile_id == E1_LAYER_STACK_NARROW_CONDUCTOR_PROFILE_ID:
            cases = e1_layer_stack_narrow_conductor_cases()
        elif config.state_profile_id == E1_LAYER_STACK_MULTI_INTERFACE_PROFILE_ID:
            cases = e1_layer_stack_multi_interface_cases()
        else:
            cases = ()
        if config.state_profile_id == E2_LAYER_STACK_SHARED_DECODER_PROFILE_ID:
            families = e2_layer_stack_shared_decoder_families(collection.seed)
            case_ids = [
                f"shared-decoder-family-{index:04d}"
                for index in range(E2_LAYER_STACK_SHARED_DECODER_FAMILY_COUNT)
            ]
            group_ids = [
                f"layer-stack-e2-family-{index:04d}"
                for index in range(E2_LAYER_STACK_SHARED_DECODER_FAMILY_COUNT)
            ]
        elif cases:
            case_ids = [case_id for case_id, _ in cases]
            families = [[stack] for _, stack in cases]
            group_prefix = (
                "layer-stack-boundary"
                if config.state_profile_id == E0_LAYER_STACK_BOUNDARY_PROFILE_ID
                else "layer-stack-e1"
            )
            group_ids = [f"{group_prefix}-{case_id}" for case_id in case_ids]
        else:
            families = sample_stack_families(config.family_count, config.local_state_count, collection.seed)
            case_ids = [f"sampled-family-{index:06d}" for index in range(config.family_count)]
            group_ids = [f"layer-stack-family-{index:06d}" for index in range(config.family_count)]
        splits = assign_group_splits(group_ids, collection.seed)
        states = []
        for family_index, family in enumerate(families):
            group_id = group_ids[family_index]
            for local_index, stack in enumerate(family):
                program = material_program_from_layer_stack(
                    stack,
                    metadata={
                        "family_index": family_index,
                        "local_state_index": local_index,
                        "state_profile_id": config.state_profile_id,
                        "state_profile_case_id": case_ids[family_index],
                    },
                )
                payload = program.to_json().encode("utf-8")
                source_hash = hashlib.sha256(payload).hexdigest()
                states.append(
                    SourceState(
                        make_state_id(self.descriptor.family_id, self.descriptor.native_schema_id, payload, source_hash),
                        self.descriptor.family_id,
                        self.descriptor.reference_id,
                        f"{group_id}/state-{local_index:04d}",
                        group_id,
                        self.descriptor.native_schema_id,
                        payload,
                        "",
                        source_hash,
                        splits[group_id],
                        stack,
                    )
                )
        self._states = tuple(states)
        self._evaluator = evaluator

    def source_states(self) -> Sequence[SourceState]:
        return self._states

    def _active_evaluator(self, plan: QueryPlan):
        if self._evaluator is None:
            self._evaluator = FalcorReferenceEvaluator(
                plan.light_directions,
                max_depth=self.provider_config.max_depth,
                max_query_group_batch=self.provider_config.query_group_batch,
            )
        if int(self._evaluator.light_count) != plan.direction_count:
            raise ValueError("LayerStack evaluator direction count disagrees with the persisted QueryPlan")
        return self._evaluator

    def evaluate(
        self,
        state: SourceState,
        surfaces: Sequence[SurfaceSample],
        plan: QueryPlan,
    ) -> EvaluatedBlock:
        if len(surfaces) != 1:
            raise ValueError("constant LayerStack states require exactly one surface sample")
        evaluator = self._active_evaluator(plan)
        materials = [state.runtime_state] * len(plan.view_directions)
        seeds = np.asarray(
            [
                (self.config.seed ^ int(state.state_id[:8], 16) ^ (index * 0x9E3779B1)) & 0xFFFFFFFF
                for index in range(len(materials))
            ],
            dtype=np.uint32,
        )
        evaluated_parts = []
        for start in range(0, len(materials), self.provider_config.query_group_batch):
            stop = min(start + self.provider_config.query_group_batch, len(materials))
            lights = (
                plan.light_directions
                if plan.light_directions.ndim == 2
                else plan.light_directions[start:stop]
            )
            if self.provider_config.adaptive:
                evaluated_parts.append(evaluate_reference_adaptive(
                    evaluator,
                    materials[start:stop],
                    plan.view_directions[start:stop],
                    query_group_seeds=seeds[start:stop],
                    light_directions=lights,
                    batch_samples=self.provider_config.batch_samples,
                    min_samples=self.provider_config.min_samples,
                    max_samples=self.provider_config.max_samples,
                    relative_standard_error=self.provider_config.relative_standard_error,
                ))
            else:
                evaluated_parts.append(evaluate_reference_fixed(
                    evaluator,
                    materials[start:stop],
                    plan.view_directions[start:stop],
                    query_group_seeds=seeds[start:stop],
                    light_directions=lights,
                    samples_per_replica=self.provider_config.samples_per_replica,
                ))

        def concatenate(name: str) -> np.ndarray:
            return np.concatenate([getattr(part, name) for part in evaluated_parts], axis=0)

        shape = (1, len(plan.view_directions), plan.direction_count)
        return EvaluatedBlock(
            concatenate("mean")[None].astype(np.float32),
            concatenate("variance")[None].astype(np.float32),
            concatenate("replica_mean_a")[None].astype(np.float32),
            concatenate("replica_mean_b")[None].astype(np.float32),
            np.broadcast_to(concatenate("sample_count")[None, :, None], shape).copy().astype(np.uint32),
            np.ones(shape, dtype=np.uint8),
            np.ones(shape, dtype=np.uint32),
            np.zeros(shape, dtype=np.float32),
            np.broadcast_to(seeds[None, :, None], shape).copy().astype(np.uint64),
        )

    def metadata(self):
        return {**super().metadata(), "provider_config": self.provider_config.__dict__}

    def close(self) -> None:
        self._evaluator = None
