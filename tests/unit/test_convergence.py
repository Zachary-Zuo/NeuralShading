"""锁定 validation-relative 收敛门：只看相对改善、后期轨迹与恢复，不看绝对质量。"""

from __future__ import annotations

from pathlib import Path

from ncls.learning.evaluation.convergence import (
    ConvergenceProtocol,
    analyze_convergence_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> ConvergenceProtocol:
    return ConvergenceProtocol.load(
        PROJECT_ROOT / "configs/evaluation/validation-relative-convergence-v1.json"
    )


def _validation(step: int, base: float) -> dict:
    return {
        "step": step,
        "evaluation_role": "validation",
        "valid": True,
        "states": {
            f"state-{index:02d}": {"directional_l1": base + index}
            for index in range(30)
        },
    }


def _trace(step: int) -> dict:
    return {
        "step": step,
        "objective": 1.0 / step,
        "gradient_norm_before_clipping": 1.0,
        "gradient": {"all_finite": True},
        "parameters": {"all_finite": True},
        "optimizer_step_skipped": False,
    }


def test_convergence_accepts_statistical_improvement_at_arbitrarily_high_error() -> None:
    protocol = _protocol()
    initialization = _validation(0, 1000.0)
    history = [_validation(step, 900.0 - 50.0 * step) for step in range(1, 9)]
    result = analyze_convergence_records(
        initialization,
        history,
        [_trace(step) for step in range(1, 9)],
        history[-1],
        protocol,
        best_step=8,
    )
    assert result["passed"]
    assert result["validation_improvement"]["statistically_supported"]
    assert result["late_window"]["passed"]
    assert result["checkpoint_recovery"]["passed"]
    assert result["quality_threshold_used"] is False


def test_convergence_rejects_credible_late_divergence_without_quality_line() -> None:
    protocol = _protocol()
    initialization = _validation(0, 1000.0)
    history = [_validation(step, 400.0 + 10.0 * step) for step in range(1, 9)]
    result = analyze_convergence_records(
        initialization,
        history,
        [_trace(step) for step in range(1, 9)],
        history[-1],
        protocol,
        best_step=8,
    )
    assert result["validation_improvement"]["statistically_supported"]
    assert result["late_window"]["credible_divergence"]
    assert not result["passed"]
    assert result["quality_threshold_used"] is False


def test_convergence_protocol_freezes_main_seed_but_no_quality_threshold() -> None:
    protocol = _protocol()
    assert protocol.required_seeds == (20260824,)
    fields = protocol.to_dict()
    assert not any("quality" in name or "directional_l1" in name for name in fields)
    assert len(protocol.sha256) == 64
