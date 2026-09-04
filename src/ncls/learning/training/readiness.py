from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ncls.learning.method import MethodDescriptor

from .checkpoint import TrainingCheckpoint
from .config import TrainingConfig


CheckpointReadinessMode = Literal[
    "formal", "diagnostic-evaluator", "visual-diagnostic"
]

@dataclass(frozen=True)
class CheckpointReadiness:
    mode: CheckpointReadinessMode
    ready: bool
    exact_method_identity: bool
    complete_training: bool
    training_run_class: str
    required_groups: tuple[str, ...]
    failed_groups: tuple[str, ...]
    reasons: tuple[str, ...]

    def require_ready(self) -> None:
        if not self.ready:
            raise ValueError(
                f"checkpoint is not ready for {self.mode}: " + "; ".join(self.reasons)
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "ncls.checkpoint-readiness@1",
            "mode": self.mode,
            "ready": self.ready,
            "exact_method_identity": self.exact_method_identity,
            "complete_training": self.complete_training,
            "training_run_class": self.training_run_class,
            "required_groups": list(self.required_groups),
            "failed_groups": list(self.failed_groups),
            "reasons": list(self.reasons),
        }


def _required_component_groups(descriptor: MethodDescriptor) -> frozenset[str]:
    return frozenset(
        group
        for component in descriptor.components
        if component.required
        for group in component.parameter_groups
    )


def _failed_coverage(
    checkpoint: TrainingCheckpoint, required_groups: frozenset[str]
) -> tuple[str, ...]:
    fields = (
        "finite_observed",
        "nonzero_gradient_observed",
        "parameter_update_observed",
    )
    return tuple(
        sorted(
            group
            for group in required_groups
            if group not in checkpoint.gradient_coverage
            or not all(bool(checkpoint.gradient_coverage[group].get(field)) for field in fields)
        )
    )


def assess_checkpoint_readiness(
    checkpoint: TrainingCheckpoint,
    descriptor: MethodDescriptor,
    *,
    mode: CheckpointReadinessMode = "formal",
) -> CheckpointReadiness:
    if mode not in {"formal", "diagnostic-evaluator", "visual-diagnostic"}:
        raise ValueError("checkpoint readiness mode is unsupported")
    reasons: list[str] = []
    exact = True
    try:
        checkpoint.validate_method(descriptor)
    except ValueError as error:
        exact = False
        reasons.append(str(error))

    complete = checkpoint.phase_name == "complete"
    try:
        run_class = TrainingConfig.from_dict(checkpoint.training_config).run_class
    except (KeyError, TypeError, ValueError) as error:
        run_class = "invalid"
        reasons.append(f"checkpoint training config is invalid: {error}")
    if mode == "formal":
        required_groups = _required_component_groups(descriptor)
        if run_class != "formal":
            reasons.append(
                f"formal export requires run_class=formal, got {run_class}"
            )
        if not complete:
            reasons.append(
                f"formal export requires complete training, got {checkpoint.phase_name}@{checkpoint.global_step}"
            )
    elif mode == "diagnostic-evaluator":
        policy = descriptor.readiness_policies.get(mode)
        if policy is None:
            required_groups = _required_component_groups(descriptor)
            allowed_phases = set()
            minimum_global_step = 1
            reasons.append("method does not declare a diagnostic evaluator readiness policy")
        else:
            required_groups = frozenset(policy.required_parameter_groups)
            allowed_phases = set(policy.allowed_phases)
            minimum_global_step = policy.minimum_global_step
        if (
            checkpoint.global_step < minimum_global_step
            or checkpoint.phase_name not in allowed_phases
        ):
            reasons.append("diagnostic evaluator preview requires end-to-end training evidence")
    else:
        # Visual eval is a diagnostic, never deployment or selection evidence.
        required_groups = frozenset()
        if checkpoint.global_step < 1:
            reasons.append("visual diagnostic requires at least one completed training step")

    unknown = required_groups - set(descriptor.parameter_groups)
    if unknown:
        reasons.append(f"readiness policy references unknown groups: {sorted(unknown)}")
    failed = _failed_coverage(checkpoint, required_groups - unknown)
    if failed:
        reasons.append(f"required gradient/update coverage is incomplete: {list(failed)}")
    return CheckpointReadiness(
        mode,
        not reasons,
        exact,
        complete,
        run_class,
        tuple(sorted(required_groups)),
        failed,
        tuple(reasons),
    )


__all__ = [
    "CheckpointReadiness",
    "CheckpointReadinessMode",
    "assess_checkpoint_readiness",
]
