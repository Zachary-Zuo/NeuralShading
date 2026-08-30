from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

from ncls.core.scattering import MaterialPayload, RuntimePayload

from .method import MethodDescriptor


@dataclass(frozen=True)
class MethodArtifactInventory:
    runtime_artifacts: frozenset[str]
    slang_entry_points: frozenset[str]

    @classmethod
    def from_payloads(
        cls,
        runtime: RuntimePayload,
        asset: MaterialPayload,
        *,
        checkpoint_model_state: bool,
    ) -> "MethodArtifactInventory":
        artifacts = {
            *(f"slang:{name}" for name in runtime.module_closure),
            *(f"program:{name}" for name in runtime.blobs),
            *(f"program-sampler:{name}" for name in runtime.sampler_descriptors),
            *(f"asset:{name}" for name in asset.blobs),
            *(f"asset:{name}" for name in asset.resources),
            *(f"asset-sampler:{name}" for name in asset.sampler_descriptors),
        }
        if checkpoint_model_state:
            artifacts.add("checkpoint:model_state")
        module_text = "\n".join(
            payload.decode("utf-8", errors="ignore")
            for payload in runtime.module_closure.values()
        )
        entries = frozenset(
            re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", module_text)
        )
        return cls(frozenset(artifacts), entries)


def validate_phase_execution(
    descriptor: MethodDescriptor,
    phases: Iterable[Mapping[str, Any]],
) -> None:
    phase_values = tuple(dict(phase) for phase in phases)
    names = {str(phase.get("name", "")) for phase in phase_values}
    for component in descriptor.components:
        if not component.required:
            continue
        if not set(component.active_phases).issubset(names):
            raise ValueError(f"required component {component.component_id!r} has an absent phase")
        for phase_name in component.active_phases:
            phase = next(value for value in phase_values if value.get("name") == phase_name)
            route_kinds = {str(route.get("kind", "")) for route in phase.get("routes", ())}
            active_groups = {str(group) for group in phase.get("parameter_groups", ())}
            if not set(component.batch_dependencies).issubset(route_kinds):
                raise ValueError(
                    f"required component {component.component_id!r} lacks a typed batch dependency"
                )
            if component.parameter_groups and not set(component.parameter_groups).intersection(active_groups):
                raise ValueError(
                    f"required component {component.component_id!r} is inactive in phase {phase_name!r}"
                )


def validate_objective_outputs(
    descriptor: MethodDescriptor,
    phase_name: str,
    outputs: Mapping[str, Any],
) -> None:
    required = {
        name
        for component in descriptor.components
        if component.required and phase_name in component.active_phases
        for name in component.python_outputs
    }
    missing = required - set(outputs)
    if missing:
        raise RuntimeError(
            f"training objective omitted required component outputs: {sorted(missing)}"
        )


def validate_gradient_coverage(
    descriptor: MethodDescriptor,
    coverage: Mapping[str, Mapping[str, Any]],
) -> None:
    required_groups = {
        group
        for component in descriptor.components
        if component.required
        for group in component.parameter_groups
    }
    missing = required_groups - set(coverage)
    failed = {
        group
        for group in required_groups & set(coverage)
        if not all(
            bool(coverage[group].get(field, False))
            for field in (
                "finite_observed", "nonzero_gradient_observed",
                "parameter_update_observed",
            )
        )
    }
    if missing or failed:
        raise RuntimeError(
            f"required component gradient coverage failed; missing={sorted(missing)}, "
            f"failed={sorted(failed)}"
        )


def validate_artifact_coverage(
    descriptor: MethodDescriptor,
    inventory: MethodArtifactInventory,
) -> None:
    required_artifacts = {
        name
        for component in descriptor.components
        if component.required
        for name in component.runtime_artifacts
    }
    required_entries = {
        name
        for component in descriptor.components
        if component.required
        for name in component.slang_entry_points
    }
    missing_artifacts = required_artifacts - inventory.runtime_artifacts
    missing_entries = required_entries - inventory.slang_entry_points
    if missing_artifacts or missing_entries:
        raise RuntimeError(
            f"required component artifact coverage failed; "
            f"artifacts={sorted(missing_artifacts)}, entries={sorted(missing_entries)}"
        )


__all__ = [
    "MethodArtifactInventory",
    "validate_artifact_coverage",
    "validate_gradient_coverage",
    "validate_objective_outputs",
    "validate_phase_execution",
]
