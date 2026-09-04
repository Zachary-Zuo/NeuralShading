from __future__ import annotations

import os
import platform
from typing import Any, Mapping

import pytest
import torch

from ncls.learning.training import (
    DistributedContext,
    preflight_topology,
    worker_execution_context,
)


class _NCCLObjective:
    implementation_sha256 = "d" * 64

    def compute(
        self,
        model: torch.nn.Module,
        batches: Mapping[str, Any],
        phase: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        del phase
        prediction = model(batches["x"])
        loss = prediction.square().mean()
        return loss, {"quadratic": loss.detach()}


def _is_two_rank_linux_worker() -> bool:
    return (
        platform.system() == "Linux"
        and os.environ.get("WORLD_SIZE") == "2"
        and os.environ.get("NCLS_DDP_WORLD_SIZE") == "2"
    )


@pytest.mark.skipif(
    not _is_two_rank_linux_worker(),
    reason="run through the two-GPU Linux DDP launcher",
)
def test_two_rank_nccl_reducer_and_control_plane() -> None:
    gpu_indices = tuple(
        int(value) for value in os.environ["NCLS_DDP_GPU_LIST"].split(",")
    )
    topology = preflight_topology(gpu_indices)
    execution_context = worker_execution_context(topology)
    distributed = DistributedContext.initialize(execution_context)
    local_error: BaseException | None = None
    try:
        model = torch.nn.Linear(1, 1, bias=False, device="cuda:0")
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        owner, execution = distributed.build_objective(
            _NCCLObjective(),
            model,
            phase_name="synthetic",
        )
        x = torch.tensor(
            [[float(distributed.rank + 1)]],
            device="cuda:0",
        )
        loss = execution({"x": x}, {"name": "synthetic"})
        metrics = owner.take_metrics()
        loss.backward()
        gradient = float(model.weight.grad)
        optimizer.step()
        report_loss, report_metrics = distributed.reduce_report(
            loss,
            metrics,
            scope="integration:metrics",
        )
        stage = distributed.rank_statistics(
            {"synthetic_prepare": float(distributed.rank + 1)},
            scope="integration:stage",
        )
        logging = distributed.ddp_logging_metrics(execution)

        assert gradient == pytest.approx(5.0)
        assert float(model.weight.detach()) == pytest.approx(0.5)
        assert float(report_loss) == pytest.approx(2.5)
        assert float(report_metrics["quadratic"]) == pytest.approx(2.5)
        assert stage["synthetic_prepare_rank_0"] == 1.0
        assert stage["synthetic_prepare_rank_1"] == 2.0
        assert stage["synthetic_prepare_rank_max"] == 2.0
        assert stage["synthetic_prepare_straggler_rank"] == 1.0
        assert logging["profile/ddp_bucket_count"] >= 1.0
        assert logging["profile/ddp_parameter_tensors"] == 1.0
        assert logging["profile/ddp_parameter_bytes"] == 4.0
    except BaseException as error:
        local_error = error
    try:
        distributed.synchronize_rank_errors("NCCL integration", local_error)
    finally:
        distributed.close()
