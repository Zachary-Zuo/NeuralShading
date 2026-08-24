from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from .contract import EvaluatedBlock, QueryPlan, ReferenceProvider
from .dataset import ReferenceDatasetManifest, ReferenceDatasetWriter
from .contract import QUERY_ROLE_NAMES
from .surfaces import (
    CONSTANT_FOOTPRINT_PROFILE,
    FOOTPRINT_SWEEP_MINIMUM_SAMPLE_COUNT,
    FOOTPRINT_SWEEP_PROFILE,
    SURFACE_PROFILES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class CollectionConfig:
    name: str = "smoke"
    query_role: str = "train"
    view_count: int = 4
    light_count: int = 128
    proposal: str = "uniform"
    mixture_weights: tuple[float, ...] = (0.5, 0.35, 0.15)
    grazing_view_fraction: float = 0.2
    grazing_min_degrees: float = 75.0
    grazing_max_degrees: float = 89.0
    transmission_view_count: int = 0
    critical_view_min_degrees: float = 35.0
    critical_view_max_degrees: float = 55.0
    critical_wi_abs_cosine_min: float = 0.65
    critical_wi_abs_cosine_max: float = 0.85
    spatial_sample_count: int = 1
    footprint_width: float = 1.0 / 4096.0
    surface_profile: str = CONSTANT_FOOTPRINT_PROFILE
    seed: int = 20260824
    split_direction_scramble: bool = True

    def __post_init__(self) -> None:
        if min(self.view_count, self.light_count, self.spatial_sample_count) < 1:
            raise ValueError("query counts must be positive")
        if not 0 <= self.transmission_view_count < self.view_count:
            raise ValueError("transmission view count must be smaller than total view count")
        if not self.name or self.query_role not in QUERY_ROLE_NAMES:
            raise ValueError("collection requires a name and known query role")
        if self.proposal not in {"uniform", "peak-aware", "adversarial"}:
            raise ValueError("proposal must be uniform, peak-aware or adversarial")
        if not self.mixture_weights or any(weight < 0.0 for weight in self.mixture_weights):
            raise ValueError("mixture weights must be nonnegative")
        if self.proposal != "uniform" and not np.isclose(sum(self.mixture_weights), 1.0):
            raise ValueError("mixture weights must sum to one")
        if not 0.0 <= self.grazing_view_fraction <= 1.0:
            raise ValueError("grazing view fraction must lie in [0, 1]")
        if not 0.0 < self.grazing_min_degrees < self.grazing_max_degrees < 90.0:
            raise ValueError("grazing view band must lie inside (0, 90) degrees")
        if not 0.0 < self.critical_view_min_degrees < self.critical_view_max_degrees < 90.0:
            raise ValueError("critical view band must lie inside (0, 90) degrees")
        if not 0.0 < self.critical_wi_abs_cosine_min < self.critical_wi_abs_cosine_max < 1.0:
            raise ValueError("critical wi cosine band must lie inside (0, 1)")
        if self.footprint_width < 0.0 or self.seed < 0:
            raise ValueError("footprint width and seed must be nonnegative")
        if self.surface_profile not in SURFACE_PROFILES:
            raise ValueError("unsupported surface profile")
        if self.surface_profile == FOOTPRINT_SWEEP_PROFILE and (
            self.spatial_sample_count < FOOTPRINT_SWEEP_MINIMUM_SAMPLE_COUNT
            or self.footprint_width <= 0.0
        ):
            raise ValueError(
                f"{FOOTPRINT_SWEEP_PROFILE} requires at least "
                f"{FOOTPRINT_SWEEP_MINIMUM_SAMPLE_COUNT} spatial samples and a positive footprint width"
            )


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


def _reciprocal_block(
    provider: ReferenceProvider,
    state,
    surfaces,
    plan: QueryPlan,
) -> EvaluatedBlock:
    """对每条 `(wo, wi)` 采集 canonical reciprocal pair，供 source-aware 指标使用。"""

    view_count = len(plan.view_directions)
    direction_count = plan.direction_count
    original_views = np.repeat(plan.view_directions, direction_count, axis=0)
    blocks = []
    for surface_index, surface in enumerate(surfaces):
        lights = (
            plan.light_directions
            if plan.light_directions.ndim == 3
            else plan.light_directions[surface_index]
        ).reshape(view_count * direction_count, 3)
        transmission = lights[:, 2] < 0.0
        reciprocal_views = lights.copy()
        reciprocal_lights = original_views.copy()
        reciprocal_views[transmission] *= -1.0
        reciprocal_lights[transmission] *= -1.0
        reciprocal_plan = QueryPlan(
            reciprocal_views,
            reciprocal_lights[:, None, :],
            np.ones((len(reciprocal_views), 1), dtype=np.float32),
            np.ones((len(reciprocal_views), 1), dtype=np.float32),
            "reciprocal-paired-v1",
            plan.seed ^ ((surface_index + 1) * 0xA24BAED5),
            np.repeat(plan.query_roles, direction_count),
        )
        evaluated = provider.evaluate(state, (surface,), reciprocal_plan)
        fields = {}
        for name in (
            "mean", "variance", "replica_mean_a", "replica_mean_b",
            "sample_count", "valid", "event_flags", "reference_pdf", "rng_seed",
        ):
            value = np.asarray(getattr(evaluated, name))
            tail = value.shape[3:] if value.ndim == 4 else value.shape[3:]
            fields[name] = value.reshape(1, view_count, direction_count, *tail)
        blocks.append(fields)
    return EvaluatedBlock(**{
        name: np.concatenate([block[name] for block in blocks], axis=0)
        for name in blocks[0]
    })


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
        sampling_name=config.name,
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
            reciprocal = _reciprocal_block(provider, state, surfaces, plan)
            writer.append(state_index, surfaces, plan, evaluated, reciprocal)
            if state_index + 1 == len(state_provider) or state_provider[state_index + 1] is not provider:
                provider.close()
        return writer.finalize()
    except Exception:
        writer.abort()
        raise
    finally:
        for provider in providers:
            provider.close()
