from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from ncls.core.identity import require_sha256, sha256_file, sha256_json
from ncls.learning.method import MethodDescriptor

from .config import TrainingConfig


@dataclass(frozen=True)
class TrainingCheckpoint:
    method_key: str
    method_descriptor_sha256: str
    implementation_identity: str
    component_manifest: Mapping[str, Any]
    training_config: Mapping[str, Any]
    training_config_sha256: str
    phase_graph_sha256: str
    reference_program_identity: str
    reference_execution_plan_identity: str
    native_asset_collection_identity: str
    query_stream_identity: str
    source_contracts: tuple[Mapping[str, Any], ...]
    source_snapshot_ids: tuple[str, ...]
    global_step: int
    phase_index: int
    phase_name: str
    phase_step: int
    selection_evidence: Mapping[str, Any]
    model_state: Mapping[str, torch.Tensor]
    phase_optimization_state: Mapping[str, Any]
    rng_state: Mapping[str, Any]
    query_stream_state: Mapping[str, Any]
    gradient_coverage: Mapping[str, Mapping[str, Any]]
    validation_state: Mapping[str, Any]
    format_name: str = "ncls.training-checkpoint"
    format_version: int = 4

    def __post_init__(self) -> None:
        if self.format_name != "ncls.training-checkpoint" or self.format_version != 4:
            raise ValueError("unsupported TrainingCheckpoint format")
        if not self.method_key or not self.phase_name:
            raise ValueError("TrainingCheckpoint method and phase identities are required")
        for name, value in (
            ("method_descriptor_sha256", self.method_descriptor_sha256),
            ("implementation_identity", self.implementation_identity),
            ("training_config_sha256", self.training_config_sha256),
            ("phase_graph_sha256", self.phase_graph_sha256),
            ("reference_program_identity", self.reference_program_identity),
            ("reference_execution_plan_identity", self.reference_execution_plan_identity),
            ("native_asset_collection_identity", self.native_asset_collection_identity),
            ("query_stream_identity", self.query_stream_identity),
        ):
            require_sha256(name, value)
        training_config = dict(self.training_config)
        if sha256_json(training_config) != self.training_config_sha256:
            raise ValueError("TrainingCheckpoint training config hash mismatch")
        config = TrainingConfig.from_dict(training_config)
        if config.method_key != self.method_key:
            raise ValueError("TrainingCheckpoint method disagrees with training config")
        if sha256_json([phase.to_dict() for phase in config.phases]) != self.phase_graph_sha256:
            raise ValueError("TrainingCheckpoint phase graph hash mismatch")
        expected_index, expected_step = config.locate_step(self.global_step)
        expected_name = (
            "complete" if expected_index == len(config.phases) else config.phases[expected_index].name
        )
        if (
            self.phase_index != expected_index
            or self.phase_step != expected_step
            or self.phase_name != expected_name
        ):
            raise ValueError("TrainingCheckpoint phase cursor disagrees with global_step")
        if not self.source_contracts or not self.source_snapshot_ids:
            raise ValueError("TrainingCheckpoint source contracts and snapshot IDs are required")
        for snapshot_id in self.source_snapshot_ids:
            require_sha256("source_snapshot_id", snapshot_id)
        component_manifest = dict(self.component_manifest)
        if set(component_manifest) != {"schema", "parameter_groups", "components"}:
            raise ValueError("TrainingCheckpoint component manifest fields are invalid")
        if component_manifest["schema"] != "ncls.method-components@1":
            raise ValueError("TrainingCheckpoint component manifest schema is invalid")
        model_state = dict(self.model_state)
        if not model_state or any(not isinstance(value, torch.Tensor) for value in model_state.values()):
            raise ValueError("TrainingCheckpoint model_state must be a nonempty tensor mapping")
        if any(
            not bool(torch.isfinite(value).all())
            for value in model_state.values()
            if value.is_floating_point()
        ):
            raise ValueError("TrainingCheckpoint model tensors must be finite")
        optimization = dict(self.phase_optimization_state)
        if self.phase_name == "complete":
            if optimization:
                raise ValueError("complete TrainingCheckpoint must not retain optimizer state")
        elif set(optimization) != {"phase_name", "optimizer", "scheduler", "precision"}:
            raise ValueError("TrainingCheckpoint phase optimization fields are invalid")
        elif optimization["phase_name"] != self.phase_name:
            raise ValueError("TrainingCheckpoint optimizer belongs to another phase")
        coverage = {
            str(group): dict(value) for group, value in self.gradient_coverage.items()
        }
        if set(coverage) != set(component_manifest["parameter_groups"]):
            raise ValueError("TrainingCheckpoint gradient coverage must cover every parameter group")
        required_coverage_fields = {
            "finite_observed", "nonzero_gradient_observed", "parameter_update_observed",
            "last_audit_step",
        }
        for value in coverage.values():
            if set(value) != required_coverage_fields:
                raise ValueError("TrainingCheckpoint gradient coverage fields are invalid")
            if any(
                not isinstance(value[field], bool)
                for field in (
                    "finite_observed",
                    "nonzero_gradient_observed",
                    "parameter_update_observed",
                )
            ):
                raise ValueError("TrainingCheckpoint gradient coverage flags must be boolean")
            if int(value["last_audit_step"]) < -1:
                raise ValueError("TrainingCheckpoint gradient audit cursor is invalid")
        if self.phase_name == "complete":
            required_groups = {
                str(group)
                for component in component_manifest["components"]
                if bool(component.get("required", False))
                for group in component.get("parameter_groups", ())
            }
            failed_groups = {
                group
                for group in required_groups
                if group not in coverage
                or not all(
                    coverage[group][field]
                    for field in (
                        "finite_observed",
                        "nonzero_gradient_observed",
                        "parameter_update_observed",
                    )
                )
            }
            if failed_groups:
                raise ValueError(
                    "complete TrainingCheckpoint has incomplete required gradient coverage: "
                    f"{sorted(failed_groups)}"
                )
        object.__setattr__(self, "component_manifest", component_manifest)
        object.__setattr__(self, "training_config", training_config)
        object.__setattr__(
            self, "source_contracts", tuple(dict(value) for value in self.source_contracts)
        )
        object.__setattr__(self, "source_snapshot_ids", tuple(self.source_snapshot_ids))
        object.__setattr__(self, "selection_evidence", dict(self.selection_evidence))
        object.__setattr__(self, "model_state", model_state)
        object.__setattr__(self, "phase_optimization_state", optimization)
        object.__setattr__(self, "rng_state", dict(self.rng_state))
        object.__setattr__(self, "query_stream_state", dict(self.query_stream_state))
        object.__setattr__(self, "gradient_coverage", coverage)
        object.__setattr__(self, "validation_state", dict(self.validation_state))

    def validate_method(self, descriptor: MethodDescriptor) -> None:
        if (
            descriptor.method_key != self.method_key
            or descriptor.descriptor_sha256 != self.method_descriptor_sha256
        ):
            raise ValueError("TrainingCheckpoint method descriptor identity mismatch")
        expected_manifest = {
            "schema": "ncls.method-components@1",
            "parameter_groups": {
                name: list(values) for name, values in descriptor.parameter_groups.items()
            },
            "components": [component.to_dict() for component in descriptor.components],
        }
        if self.component_manifest != expected_manifest:
            raise ValueError("TrainingCheckpoint component manifest disagrees with method")
        fields = {field.name: field for field in descriptor.tensor_state_schema}
        if set(self.model_state) != set(fields):
            raise ValueError("TrainingCheckpoint tensor keys disagree with method schema")
        symbols: dict[str, int] = {}
        for name, tensor in self.model_state.items():
            field = fields[name]
            if field.dtype != str(tensor.dtype).removeprefix("torch."):
                raise ValueError(f"TrainingCheckpoint tensor {name!r} dtype mismatch")
            if len(field.shape) != tensor.ndim:
                raise ValueError(f"TrainingCheckpoint tensor {name!r} rank mismatch")
            for expected, actual in zip(field.shape, tensor.shape, strict=True):
                if isinstance(expected, int) and expected != actual:
                    raise ValueError(f"TrainingCheckpoint tensor {name!r} shape mismatch")
                if isinstance(expected, str):
                    previous = symbols.setdefault(expected, int(actual))
                    if previous != actual:
                        raise ValueError(
                            f"TrainingCheckpoint symbolic dimension {expected!r} mismatch"
                        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "method_key": self.method_key,
            "method_descriptor_sha256": self.method_descriptor_sha256,
            "implementation_identity": self.implementation_identity,
            "component_manifest": dict(self.component_manifest),
            "training_config": dict(self.training_config),
            "training_config_sha256": self.training_config_sha256,
            "phase_graph_sha256": self.phase_graph_sha256,
            "reference_program_identity": self.reference_program_identity,
            "reference_execution_plan_identity": self.reference_execution_plan_identity,
            "native_asset_collection_identity": self.native_asset_collection_identity,
            "query_stream_identity": self.query_stream_identity,
            "source_contracts": [dict(value) for value in self.source_contracts],
            "source_snapshot_ids": list(self.source_snapshot_ids),
            "global_step": self.global_step,
            "phase_index": self.phase_index,
            "phase_name": self.phase_name,
            "phase_step": self.phase_step,
            "selection_evidence": dict(self.selection_evidence),
            "model_state": dict(self.model_state),
            "phase_optimization_state": dict(self.phase_optimization_state),
            "rng_state": dict(self.rng_state),
            "query_stream_state": dict(self.query_stream_state),
            "gradient_coverage": {
                group: dict(value) for group, value in self.gradient_coverage.items()
            },
            "validation_state": dict(self.validation_state),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "TrainingCheckpoint":
        required = {
            "format_name", "format_version", "method_key", "method_descriptor_sha256",
            "implementation_identity", "component_manifest", "training_config",
            "training_config_sha256", "phase_graph_sha256", "reference_program_identity",
            "reference_execution_plan_identity", "native_asset_collection_identity",
            "query_stream_identity", "source_contracts", "source_snapshot_ids",
            "global_step", "phase_index", "phase_name", "phase_step",
            "selection_evidence", "model_state", "phase_optimization_state",
            "rng_state", "query_stream_state", "gradient_coverage", "validation_state",
        }
        if set(value) != required:
            raise ValueError(f"TrainingCheckpoint fields must be exactly {sorted(required)}")
        return cls(
            str(value["method_key"]),
            str(value["method_descriptor_sha256"]),
            str(value["implementation_identity"]),
            value["component_manifest"],
            value["training_config"],
            str(value["training_config_sha256"]),
            str(value["phase_graph_sha256"]),
            str(value["reference_program_identity"]),
            str(value["reference_execution_plan_identity"]),
            str(value["native_asset_collection_identity"]),
            str(value["query_stream_identity"]),
            tuple(value["source_contracts"]),
            tuple(str(item) for item in value["source_snapshot_ids"]),
            int(value["global_step"]),
            int(value["phase_index"]),
            str(value["phase_name"]),
            int(value["phase_step"]),
            value["selection_evidence"],
            value["model_state"],
            value["phase_optimization_state"],
            value["rng_state"],
            value["query_stream_state"],
            value["gradient_coverage"],
            value["validation_state"],
            str(value["format_name"]),
            int(value["format_version"]),
        )


def save_checkpoint(path: Path | str, checkpoint: TrainingCheckpoint) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(checkpoint.to_payload(), temporary)
    os.replace(temporary, target)
    digest = sha256_file(target)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


def load_checkpoint(
    path: Path | str,
    *,
    descriptor: MethodDescriptor | None = None,
    map_location: str | torch.device = "cpu",
) -> TrainingCheckpoint:
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError("TrainingCheckpoint SHA-256 sidecar is missing")
    expected = sidecar.read_text(encoding="ascii").strip()
    require_sha256("TrainingCheckpoint sidecar", expected)
    if sha256_file(target) != expected:
        raise ValueError("TrainingCheckpoint file hash mismatch")
    value = torch.load(target, map_location=map_location, weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError("TrainingCheckpoint payload root must be an object")
    checkpoint = TrainingCheckpoint.from_payload(value)
    if descriptor is not None:
        checkpoint.validate_method(descriptor)
    return checkpoint
