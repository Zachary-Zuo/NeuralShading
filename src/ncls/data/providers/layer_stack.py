from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from ncls.core.material import SheenInterface, material_program_from_layer_stack
from ncls.data.collector import CollectionConfig
from ncls.data.contract import (
    EvaluatedBlock,
    PositionKind,
    QueryPlan,
    QueryRole,
    ReferenceDescriptor,
    SourceState,
    SurfaceSample,
    make_state_id,
)
from ncls.data.directions import (
    equal_area_hemisphere,
    peak_grazing_mixture_query,
)
from ncls.data.priors import (
    LAYER_STACK_PROFILE,
    layer_stack_difficulty,
    layer_stack_v1_families,
    layer_stack_v1_splits,
)
from ncls.data.reference import (
    FalcorReferenceEvaluator,
    evaluate_reference_adaptive,
    evaluate_reference_batched_fixed,
    evaluate_reference_fixed,
)

from .base import BaseProvider, PROJECT_ROOT, implementation_hash


@dataclass(frozen=True)
class LayerStackProviderConfig:
    family_count: int = 28
    states_per_family: int = 10
    heldout_family_count: int = 4
    fixed_samples_per_replica: int = 64
    max_dispatch_queries: int = 4096
    max_depth: int = 64
    adaptive: bool = False
    batch_samples_per_replica: int = 256
    min_combined_samples: int = 1024
    max_combined_samples: int = 262144
    relative_standard_error: float = 0.04
    maximum_group_relative_standard_error: float = 0.10
    enforce_maximum_group_relative_standard_error: bool = True
    peak_calibration_directions: int = 4096
    peak_calibration_samples_per_replica: int = 64
    selected_state_ids: tuple[str, ...] = ()
    state_profile: str = LAYER_STACK_PROFILE

    def __post_init__(self) -> None:
        values = (
            self.family_count, self.states_per_family, self.fixed_samples_per_replica,
            self.max_dispatch_queries, self.max_depth, self.batch_samples_per_replica,
            self.min_combined_samples, self.max_combined_samples,
            self.peak_calibration_directions, self.peak_calibration_samples_per_replica,
        )
        if min(values) < 1 or self.max_combined_samples < self.min_combined_samples:
            raise ValueError("LayerStack provider sizes must be positive and ordered")
        if self.max_dispatch_queries > 4096:
            raise ValueError("LayerStack reference dispatch may not exceed 4096 queries")
        combined_batch = 2 * self.batch_samples_per_replica
        if self.adaptive and (
            self.min_combined_samples % combined_batch
            or self.max_combined_samples % combined_batch
        ):
            raise ValueError("combined adaptive limits must be multiples of both replica batches")
        if not 0.0 < self.relative_standard_error < 1.0:
            raise ValueError("relative_standard_error must lie in (0, 1)")
        if not self.relative_standard_error <= self.maximum_group_relative_standard_error < 1.0:
            raise ValueError("maximum group relative SE must be at least the target and below one")
        if not isinstance(self.enforce_maximum_group_relative_standard_error, bool):
            raise ValueError("maximum group relative SE enforcement flag must be boolean")
        if not 1 <= self.heldout_family_count < self.family_count:
            raise ValueError("heldout family count must leave fitted families")
        if self.state_profile != LAYER_STACK_PROFILE:
            raise ValueError(f"unsupported LayerStack state profile {self.state_profile!r}")
        if len(set(self.selected_state_ids)) != len(self.selected_state_ids):
            raise ValueError("selected_state_ids must be unique")
        if self.peak_calibration_directions < 512:
            raise ValueError("moving-peak calibration requires at least 512 directions")


class LayerStackProvider(BaseProvider):
    def __init__(
        self,
        collection: CollectionConfig,
        config: LayerStackProviderConfig = LayerStackProviderConfig(),
    ) -> None:
        super().__init__(collection)
        self.provider_config = config
        source_paths = (
            Path(__file__),
            PROJECT_ROOT / "src/ncls/data/providers/base.py",
            PROJECT_ROOT / "src/ncls/data/directions.py",
            PROJECT_ROOT / "src/ncls/data/falcor.py",
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
        families = layer_stack_v1_families(
            config.family_count, config.states_per_family, collection.seed
        )
        splits = layer_stack_v1_splits(
            config.family_count,
            config.states_per_family,
            config.heldout_family_count,
            collection.seed,
        )
        states = []
        for family_index, (structure_family_id, family) in enumerate(families):
            for local_index, stack in enumerate(family):
                split, cohort = splits[(family_index, local_index)]
                difficulty, tags = layer_stack_difficulty(stack)
                group_id = f"{structure_family_id}/state-{local_index:04d}"
                program = material_program_from_layer_stack(
                    stack,
                    metadata={
                        "family_index": family_index,
                        "local_state_index": local_index,
                        "state_profile": config.state_profile,
                        "structure_family_id": structure_family_id,
                        "difficulty_class": difficulty,
                        "difficulty_tags": list(tags),
                        "evaluation_cohort": cohort,
                    },
                )
                payload = program.to_json().encode("utf-8")
                source_hash = hashlib.sha256(payload).hexdigest()
                states.append(
                    SourceState(
                        state_id=make_state_id(
                            self.descriptor.family_id,
                            self.descriptor.native_schema_id,
                            payload,
                            source_hash,
                        ),
                        family_id=self.descriptor.family_id,
                        reference_id=self.descriptor.reference_id,
                        asset_id=group_id,
                        split_group_id=group_id,
                        native_schema_id=self.descriptor.native_schema_id,
                        native_payload=payload,
                        source_uri="",
                        source_sha256=source_hash,
                        split=split,
                        structure_family_id=structure_family_id,
                        difficulty_class=difficulty,
                        difficulty_tags=tags,
                        evaluation_cohort=cohort,
                        runtime_state=stack,
                    )
                )
        if config.selected_state_ids:
            available = {state.state_id: state for state in states}
            missing = sorted(set(config.selected_state_ids) - set(available))
            if missing:
                raise ValueError(f"selected LayerStack states are absent: {missing}")
            states = [available[state_id] for state_id in config.selected_state_ids]
        self._states = tuple(states)
        self._evaluator_cache: dict[tuple[int, int, int], FalcorReferenceEvaluator] = {}

    def source_states(self) -> Sequence[SourceState]:
        return self._states

    @staticmethod
    def _sheen_peak_centers(
        views: np.ndarray,
        roughness: float,
    ) -> np.ndarray:
        z = np.linspace(1e-3, 1.0, 4096, dtype=np.float64)
        inverse_alpha = 1.0 / max(roughness, 1e-3)
        r = (1.0 - roughness) ** 2
        one_minus_r = 1.0 - r
        a = 25.3245 * r + 21.5473 * one_minus_r
        b = 3.32435 * r + 3.82987 * one_minus_r
        c = 0.16801 * r + 0.19823 * one_minus_r
        d = -1.27393 * r - 1.97760 * one_minus_r
        e = -4.85967 * r - 4.32054 * one_minus_r

        def sheen_lambda(cosine: np.ndarray) -> np.ndarray:
            cosine = np.clip(cosine, 0.0, 1.0)
            value = a / (1.0 + b * np.power(cosine, c)) + d * cosine + e
            complement = 1.0 - cosine
            complement_value = (
                a / (1.0 + b * np.power(complement, c)) + d * complement + e
            )
            mid = a / (1.0 + b * np.power(0.5, c)) + d * 0.5 + e
            return np.where(
                cosine < 0.5,
                np.exp(value),
                np.exp(2.0 * mid - complement_value),
            )

        result = np.empty_like(views, dtype=np.float64)
        for index, view in enumerate(np.asarray(views, dtype=np.float64)):
            xy = view[:2] / np.linalg.norm(view[:2])
            wi = np.column_stack((np.sqrt(1.0 - z * z)[:, None] * xy[None, :], z))
            half = wi + view[None, :]
            half /= np.linalg.norm(half, axis=1, keepdims=True)
            sin2 = np.maximum(1.0 - half[:, 2] * half[:, 2], 0.0078125)
            distribution = (
                (2.0 + inverse_alpha)
                * np.power(sin2, 0.5 * inverse_alpha)
                / (2.0 * np.pi)
            )
            # 单界面入口按 (viewDirection, lightDirection) 传入 shader 的
            # (wi, wo)；softened 项属于候选 light cosine，顺序不可互换。
            softened = np.power(
                sheen_lambda(z),
                1.0 + 2.0 * np.power(1.0 - z, 8),
            )
            masking = 1.0 / (
                1.0 + softened + sheen_lambda(np.asarray([view[2]]))[0]
            )
            response = distribution * masking / (4.0 * view[2])
            result[index] = wi[int(np.argmax(response))]
        return result.astype(np.float32)

    def _calibrated_peak_centers(
        self,
        state: SourceState,
        base: QueryPlan,
    ) -> np.ndarray:
        """用独立高分辨率 reference probe 测出每个 `wo` 的 response 峰位。"""

        direction_rows = []
        for view_index in range(len(base.view_directions)):
            directions, _ = equal_area_hemisphere(
                self.provider_config.peak_calibration_directions,
                azimuth_offset=(view_index + 1) * 0.173 * np.pi,
            )
            direction_rows.append(directions)
        lights = np.stack(direction_rows)
        measure = 2.0 * np.pi
        probe = QueryPlan(
            base.view_directions,
            lights,
            np.full(
                lights.shape[:2],
                measure / self.provider_config.peak_calibration_directions,
                dtype=np.float32,
            ),
            np.full(
                lights.shape[:2],
                1.0 / measure,
                dtype=np.float32,
            ),
            "layer-stack-peak-calibration-v1",
            base.seed ^ 0xC411B4A7,
            base.query_roles,
        )
        evaluated = self._evaluate_queries(
            state,
            (SurfaceSample(),),
            probe,
            adaptive=False,
            fixed_samples_per_replica=self.provider_config.peak_calibration_samples_per_replica,
        )
        magnitude = np.sum(np.abs(evaluated.mean[0]), axis=-1)
        peak_index = np.argmax(magnitude, axis=1)
        return lights[np.arange(len(lights)), peak_index]

    def query_plan(
        self,
        state: SourceState,
        surfaces: Sequence[SurfaceSample] = (),
    ) -> QueryPlan:
        base = super().query_plan(state, surfaces)
        stack = state.runtime_state
        if self.config.proposal == "uniform":
            return base
        if len(stack.interfaces) == 1 and isinstance(stack.interfaces[0], SheenInterface):
            centers = self._sheen_peak_centers(
                base.view_directions,
                stack.interfaces[0].roughness,
            )
            proposal_tag = "layer-stack-sheen"
        else:
            if "M" in state.difficulty_tags:
                centers = self._calibrated_peak_centers(state, base)
                proposal_tag = "layer-stack-calibrated-response"
            else:
                centers = base.view_directions.copy()
                centers[:, :2] *= -1.0
                proposal_tag = "layer-stack-reflection"
        lights = base.light_directions.copy()
        weights = base.solid_angle_weights.copy()
        pdf = base.proposal_pdf.copy()
        proposal_ids = list(base.proposal_id)
        for role in QueryRole:
            selected = base.query_roles == int(role)
            if not np.any(selected):
                continue
            role_lights, role_weights, role_pdf = peak_grazing_mixture_query(
                base.view_directions[selected],
                base.direction_count,
                full_sphere=False,
                seed=base.seed ^ ((int(role) + 1) * 0x9E3779B1),
                reflection_centers=centers[selected],
                component_weights=self.config.mixture_weights,
                critical_band_abs_cosine=(
                    self.config.critical_wi_abs_cosine_min,
                    self.config.critical_wi_abs_cosine_max,
                ),
            )
            lights[selected] = role_lights
            weights[selected] = role_weights
            pdf[selected] = role_pdf
            for index in np.flatnonzero(selected).tolist():
                proposal_ids[index] = proposal_ids[index].replace(
                    "-peak-aware-", f"-{proposal_tag}-peak-aware-"
                )
        return QueryPlan(
            base.view_directions,
            lights,
            weights,
            pdf,
            proposal_ids,
            base.seed,
            base.query_roles,
        )

    def _reference_evaluator(
        self,
        initial_light_directions: np.ndarray,
        *,
        query_group_batch: int,
        light_index_offset: int,
    ) -> FalcorReferenceEvaluator:
        tile_count = int(initial_light_directions.shape[-2])
        key = (tile_count, query_group_batch, light_index_offset)
        evaluator = self._evaluator_cache.get(key)
        if evaluator is None:
            evaluator = FalcorReferenceEvaluator(
                initial_light_directions,
                max_depth=self.provider_config.max_depth,
                max_query_group_batch=query_group_batch,
                light_index_offset=light_index_offset,
            )
            self._evaluator_cache[key] = evaluator
        return evaluator

    def _evaluate_queries(
        self,
        state: SourceState,
        surfaces: Sequence[SurfaceSample],
        plan: QueryPlan,
        *,
        adaptive: bool,
        fixed_samples_per_replica: int,
        batched_fixed_samples_per_replica: int | None = None,
    ) -> EvaluatedBlock:
        if len(surfaces) != 1:
            raise ValueError("constant LayerStack states require exactly one surface sample")
        materials = [state.runtime_state] * len(plan.view_directions)
        seeds = np.asarray(
            [
                (plan.seed ^ (index * 0x9E3779B1)) & 0xFFFFFFFF
                for index in range(len(materials))
            ],
            dtype=np.uint32,
        )
        group_count = len(materials)
        direction_count = plan.direction_count
        rgb_shape = (group_count, direction_count, 3)
        mean = np.empty(rgb_shape, dtype=np.float32)
        variance = np.empty(rgb_shape, dtype=np.float32)
        replica_a = np.empty(rgb_shape, dtype=np.float32)
        replica_b = np.empty(rgb_shape, dtype=np.float32)
        sample_count = np.empty((group_count, direction_count), dtype=np.uint32)
        direction_tile = min(direction_count, self.provider_config.max_dispatch_queries)
        for light_start in range(0, direction_count, direction_tile):
            light_stop = min(light_start + direction_tile, direction_count)
            tile_count = light_stop - light_start
            query_group_batch = max(
                1,
                self.provider_config.max_dispatch_queries // tile_count,
            )
            evaluator = self._reference_evaluator(
                plan.light_directions[0, light_start:light_stop],
                query_group_batch=query_group_batch,
                light_index_offset=light_start,
            )
            for group_start in range(0, group_count, query_group_batch):
                group_stop = min(group_start + query_group_batch, group_count)
                lights = plan.light_directions[
                    group_start:group_stop, light_start:light_stop
                ]
                if int(evaluator.light_count) != tile_count:
                    raise ValueError("LayerStack evaluator light tile disagrees with QueryPlan")
                arguments = dict(
                    query_group_seeds=seeds[group_start:group_stop],
                    light_directions=lights,
                )
                if batched_fixed_samples_per_replica is not None:
                    batch_samples = min(
                        self.provider_config.batch_samples_per_replica,
                        batched_fixed_samples_per_replica,
                    )
                    if batched_fixed_samples_per_replica % batch_samples:
                        raise ValueError("batched fixed reference budget must divide its batch size")
                    evaluated = evaluate_reference_batched_fixed(
                        evaluator,
                        materials[group_start:group_stop],
                        plan.view_directions[group_start:group_stop],
                        samples_per_replica=batched_fixed_samples_per_replica,
                        batch_samples_per_replica=batch_samples,
                        **arguments,
                    )
                elif adaptive:
                    evaluated = evaluate_reference_adaptive(
                        evaluator,
                        materials[group_start:group_stop],
                        plan.view_directions[group_start:group_stop],
                        batch_samples=self.provider_config.batch_samples_per_replica,
                        min_samples=self.provider_config.min_combined_samples // 2,
                        max_samples=self.provider_config.max_combined_samples // 2,
                        relative_standard_error=self.provider_config.relative_standard_error,
                        **arguments,
                    )
                    peak = np.max(np.abs(evaluated.mean), axis=(1, 2), keepdims=True)
                    denominator = np.maximum(
                        np.abs(evaluated.mean),
                        np.maximum(0.005 * peak, 1e-8),
                    )
                    standard_error = np.sqrt(
                        np.maximum(evaluated.variance, 0.0)
                        / np.maximum(evaluated.sample_count[:, None, None], 1)
                    )
                    group_p95 = np.quantile(
                        standard_error / denominator,
                        0.95,
                        axis=(1, 2),
                    )
                    if self.provider_config.enforce_maximum_group_relative_standard_error and np.any(
                        group_p95
                        > self.provider_config.maximum_group_relative_standard_error
                    ):
                        failed = float(np.max(group_p95))
                        raise RuntimeError(
                            "adaptive-v1 exhausted its budget above the maximum "
                            f"query-group relative SE: {failed:.6g}"
                        )
                else:
                    evaluated = evaluate_reference_fixed(
                        evaluator,
                        materials[group_start:group_stop],
                        plan.view_directions[group_start:group_stop],
                        samples_per_replica=fixed_samples_per_replica,
                        **arguments,
                    )
                region = np.s_[group_start:group_stop, light_start:light_stop]
                mean[region] = evaluated.mean
                variance[region] = evaluated.variance
                replica_a[region] = evaluated.replica_mean_a
                replica_b[region] = evaluated.replica_mean_b
                sample_count[region] = evaluated.sample_count[:, None]
        shape = (1, len(plan.view_directions), plan.direction_count)
        return EvaluatedBlock(
            mean[None],
            variance[None],
            replica_a[None],
            replica_b[None],
            sample_count[None],
            np.ones(shape, dtype=np.uint8),
            np.ones(shape, dtype=np.uint32),
            np.zeros(shape, dtype=np.float32),
            np.broadcast_to(seeds[None, :, None], shape).copy().astype(np.uint64),
        )

    def evaluate(
        self,
        state: SourceState,
        surfaces: Sequence[SurfaceSample],
        plan: QueryPlan,
    ) -> EvaluatedBlock:
        return self._evaluate_queries(
            state,
            surfaces,
            plan,
            adaptive=self.provider_config.adaptive,
            fixed_samples_per_replica=self.provider_config.fixed_samples_per_replica,
        )

    def evaluate_fixed(
        self,
        state: SourceState,
        surfaces: Sequence[SurfaceSample],
        plan: QueryPlan,
        *,
        samples_per_replica: int,
    ) -> EvaluatedBlock:
        """用调用方冻结的固定双 replica 预算执行同一 reference。"""

        if samples_per_replica < 1:
            raise ValueError("samples_per_replica must be positive")
        return self._evaluate_queries(
            state,
            surfaces,
            plan,
            adaptive=False,
            fixed_samples_per_replica=samples_per_replica,
        )

    def evaluate_batched_fixed(
        self,
        state: SourceState,
        surfaces: Sequence[SurfaceSample],
        plan: QueryPlan,
        *,
        samples_per_replica: int,
    ) -> EvaluatedBlock:
        """用 GPU 小 batch 与 CPU float64 moments 合并执行高预算固定采样。"""

        if samples_per_replica < 1:
            raise ValueError("samples_per_replica must be positive")
        return self._evaluate_queries(
            state,
            surfaces,
            plan,
            adaptive=False,
            fixed_samples_per_replica=samples_per_replica,
            batched_fixed_samples_per_replica=samples_per_replica,
        )

    def metadata(self):
        return {**super().metadata(), "provider_config": self.provider_config.__dict__}

    def close(self) -> None:
        self._evaluator_cache.clear()
