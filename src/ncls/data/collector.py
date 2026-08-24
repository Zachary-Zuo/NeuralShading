from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from .contract import ReferenceProvider
from .dataset import ReferenceDatasetManifest, ReferenceDatasetWriter
from .directions import (
    E1_MIXTURE_QUERY_PROFILE_ID,
    E2_LAYER_STACK_MIXTURE_QUERY_PROFILE_ID,
    MIXTURE_QUERY_PROFILE_ID,
)
from .surfaces import CONSTANT_FOOTPRINT_PROFILE_ID, E0_FOOTPRINT_MINIMUM_SAMPLE_COUNT, E0_FOOTPRINT_PROFILE_ID, SURFACE_PROFILE_IDS


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class CollectionConfig:
    view_count: int = 16
    validation_view_count: int = 0
    test_view_count: int = 0
    adversarial_view_count: int = 0
    light_count: int = 128
    spatial_sample_count: int = 1
    footprint_width: float = 1.0 / 4096.0
    surface_profile_id: str = CONSTANT_FOOTPRINT_PROFILE_ID
    seed: int = 20260824
    split_direction_scramble: bool = True
    query_profile_id: str = "ncls.uniform-split-independent@1"

    def __post_init__(self) -> None:
        if min(self.view_count, self.light_count, self.spatial_sample_count) < 1:
            raise ValueError("train query counts must be positive")
        if min(self.validation_view_count, self.test_view_count, self.adversarial_view_count) < 0:
            raise ValueError("non-train query role counts must be nonnegative")
        if self.footprint_width < 0.0 or self.seed < 0:
            raise ValueError("footprint width and seed must be nonnegative")
        if self.surface_profile_id not in SURFACE_PROFILE_IDS:
            raise ValueError("unsupported surface profile")
        if self.surface_profile_id == E0_FOOTPRINT_PROFILE_ID and (
            self.spatial_sample_count < E0_FOOTPRINT_MINIMUM_SAMPLE_COUNT
            or self.footprint_width <= 0.0
        ):
            raise ValueError(
                f"{E0_FOOTPRINT_PROFILE_ID} requires at least "
                f"{E0_FOOTPRINT_MINIMUM_SAMPLE_COUNT} spatial samples and a positive footprint width"
            )
        if self.query_profile_id not in {
            "ncls.uniform-split-independent@1",
            MIXTURE_QUERY_PROFILE_ID,
            E1_MIXTURE_QUERY_PROFILE_ID,
            E2_LAYER_STACK_MIXTURE_QUERY_PROFILE_ID,
        }:
            raise ValueError("unsupported query profile")


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    result = completed.stdout.strip()
    return result if completed.returncode == 0 and result else "unknown"


def collect_reference_dataset(
    output: Path | str,
    providers: Sequence[ReferenceProvider],
    config: CollectionConfig,
    *,
    created_at: str | None = None,
    generator_git_commit: str | None = None,
) -> ReferenceDatasetManifest:
    """通过统一 provider 协议采集一个或多个原生材质族。"""

    if not providers:
        raise ValueError("at least one reference provider is required")
    state_records = []
    state_provider: list[ReferenceProvider] = []
    for provider in providers:
        states = tuple(provider.source_states())
        if not states:
            raise ValueError(f"provider {provider.descriptor.reference_id} produced no source states")
        for state in states:
            if state.family_id != provider.descriptor.family_id or state.reference_id != provider.descriptor.reference_id:
                raise ValueError("provider descriptor disagrees with one of its source states")
            if state.native_schema_id != provider.descriptor.native_schema_id:
                raise ValueError("provider native schema disagrees with one of its source states")
            state_records.append(state)
            state_provider.append(provider)
    writer = ReferenceDatasetWriter(
        output,
        state_records,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        generator_git_commit=generator_git_commit or _git_commit(),
        query_profile_ids=[config.query_profile_id, *(provider.descriptor.query_profile_id for provider in providers)],
        generation_config=asdict(config),
        provider_metadata=[provider.metadata() for provider in providers],
    )
    try:
        for state_index, (state, provider) in enumerate(zip(state_records, state_provider, strict=True)):
            surfaces = tuple(provider.surface_samples(state))
            if not surfaces or any(surface.position_kind != provider.descriptor.position_kind for surface in surfaces):
                raise ValueError("provider surface samples disagree with its declared position kind")
            plan = provider.query_plan(state, surfaces)
            if np.any(plan.view_directions[:, 2] <= 0.0):
                raise ValueError("surface reference view directions must lie above the local surface")
            if provider.descriptor.incident_domain == "upper-hemisphere" and np.any(plan.light_directions[..., 2] <= 0.0):
                raise ValueError("upper-hemisphere provider emitted an incident direction below the surface")
            evaluated = provider.evaluate(state, surfaces, plan)
            writer.append(state_index, surfaces, plan, evaluated)
            if state_index + 1 == len(state_provider) or state_provider[state_index + 1] is not provider:
                provider.close()
        return writer.finalize()
    except Exception:
        writer.abort()
        raise
    finally:
        for provider in providers:
            provider.close()
