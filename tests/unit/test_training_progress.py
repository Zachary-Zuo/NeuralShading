from __future__ import annotations

import pytest

from ncls.learning.training.engine import _progress_postfix


def test_progress_postfix_labels_total_and_signed_proposal_separately() -> None:
    postfix = _progress_postfix(
        {
            "loss": 0.75,
            "loss/optimization_total": 0.75,
            "loss/appearance": 1.0,
            "loss/proposal": -0.5,
            "loss/proposal_weight": 0.5,
        },
        phase_name="joint-response-fit",
        work_units=128,
    )
    assert postfix == {
        "phase": "joint-response-fit",
        "total": "0.75",
        "appearance": "1",
        "proposal": "-0.5",
        "proposal_w": "0.5",
        "work": 128,
    }


def test_progress_postfix_rejects_nonfinite_standard_metric() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        _progress_postfix(
            {"loss": 0.0, "loss/appearance": float("nan")},
            phase_name="phase",
            work_units=1,
        )
