from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from ncls.learning.training.distributed import (
    DistributedContext,
    DistributedObjective,
    configure_distributed_debug_environment,
)
from ncls.learning.training import TrainingEngine
from tests.unit.test_training_runner_phase_graph import (
    _PhaseMethod,
    _RouteProducer,
    _config,
    _data_session,
    _plugin,
)


class _QuadraticObjective:
    implementation_sha256 = "f" * 64

    def compute(
        self,
        model: torch.nn.Module,
        batches: Mapping[str, Any],
        phase: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        del phase
        prediction = model(batches["x"])
        loss = (prediction - batches["target"]).square().mean()
        return loss, {"quadratic": loss.detach()}


class _TwoPhaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Parameter(torch.tensor(1.0))
        self.second = torch.nn.Parameter(torch.tensor(1.0))


class _TwoPhaseObjective:
    implementation_sha256 = "c" * 64

    def compute(
        self,
        model: torch.nn.Module,
        batches: Mapping[str, Any],
        phase: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        assert isinstance(model, _TwoPhaseModel)
        parameter = model.first if phase["name"] == "first" else model.second
        loss = parameter * batches["x"]
        return loss, {"phase": loss.detach()}


class _ForbiddenCheckpointCodec:
    implementation_sha256 = "e" * 64

    def encode(self, model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
        del model
        raise AssertionError("non-rank0 must not encode a full model checkpoint")

    def restore(
        self,
        model: torch.nn.Module,
        state: Mapping[str, torch.Tensor],
    ) -> None:
        del model, state


class _RankOneCheckpointContext:
    rank = 1
    world_size = 2
    device = torch.device("cpu")
    is_distributed = True
    is_rank_zero = False

    @staticmethod
    def gather_rank_payload(payload: Any) -> None:
        del payload
        return None

    @staticmethod
    def synchronize_rank_errors(
        label: str,
        local_error: BaseException | None,
    ) -> None:
        del label
        if local_error is not None:
            raise local_error


def _gloo_worker(rank: int, init_uri: str, output_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_uri,
        rank=rank,
        world_size=2,
    )
    control_group = dist.new_group(ranks=[0, 1], backend="gloo")
    context = DistributedContext(
        rank=rank,
        world_size=2,
        device=torch.device("cpu"),
        control_group=control_group,
    )
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        owner = DistributedObjective(_QuadraticObjective(), model)
        execution = DistributedDataParallel(
            owner,
            device_ids=None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
        x = torch.tensor([[float(rank + 1)]])
        loss = execution({"x": x, "target": torch.zeros_like(x)}, {"name": "test"})
        metrics = owner.take_metrics()
        loss.backward()
        gradient = float(model.weight.grad)
        optimizer.step()

        report_loss, report_metrics = context.reduce_report(
            loss,
            metrics,
            scope="test:metrics",
        )
        report_row_losses, report_row_metrics = context.reduce_report_rows(
            torch.tensor([rank + 1.0, 2.0 * (rank + 1.0)]),
            {"quadratic": torch.tensor([rank + 3.0, 2.0 * (rank + 3.0)])},
            scope="test:metric-rows",
        )
        rank_stats = context.rank_statistics(
            {"prepare": float(rank + 1)},
            scope="test:stages",
        )
        gathered = context.gather_rank_payload({"cursor": rank + 10})
        rank_zero_result = context.run_rank_zero("test commit", lambda: 17)
        all_rank_result = context.run_all_ranks(
            "test all-rank action",
            lambda: rank + 20,
        )

        phase_model = _TwoPhaseModel()
        phase_gradients: list[float] = []
        for phase_name in ("first", "second"):
            phase_model.first.requires_grad_(phase_name == "first")
            phase_model.second.requires_grad_(phase_name == "second")
            phase_owner = DistributedObjective(_TwoPhaseObjective(), phase_model)
            phase_execution = DistributedDataParallel(
                phase_owner,
                device_ids=None,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
            phase_loss = phase_execution(
                {"x": torch.tensor(float(rank + 1))},
                {"name": phase_name},
            )
            phase_owner.take_metrics()
            phase_loss.backward()
            active = phase_model.first if phase_name == "first" else phase_model.second
            phase_gradients.append(float(active.grad))
            phase_model.zero_grad(set_to_none=True)
            del phase_execution, phase_owner

        mismatch_detected = False
        try:
            context.validate_descriptor("test:mismatch", ("rank", rank))
        except RuntimeError:
            mismatch_detected = True

        failure_propagated = False
        try:
            context.run_rank_zero(
                "test failure",
                lambda: (_ for _ in ()).throw(ValueError("injected")),
            )
        except (RuntimeError, ValueError) as error:
            failure_propagated = "injected" in str(error)

        rank_failure_propagated = False
        try:
            context.synchronize_rank_errors(
                "test rank failure",
                ValueError("rank-one failure") if rank == 1 else None,
            )
        except (RuntimeError, ValueError) as error:
            rank_failure_propagated = "rank-one failure" in str(error)

        all_rank_failure_propagated = False
        try:
            context.run_all_ranks(
                "test all-rank failure",
                lambda: (
                    (_ for _ in ()).throw(ValueError("all-rank rank-one failure"))
                    if rank == 1
                    else rank
                ),
            )
        except (RuntimeError, ValueError) as error:
            all_rank_failure_propagated = "all-rank rank-one failure" in str(error)

        payload = {
            "gradient": gradient,
            "parameter": float(model.weight.detach()),
            "report_loss": float(report_loss),
            "report_metric": float(report_metrics["quadratic"]),
            "report_row_losses": report_row_losses,
            "report_row_metrics": report_row_metrics,
            "rank_stats": rank_stats,
            "gathered": gathered,
            "rank_zero_result": rank_zero_result,
            "all_rank_result": all_rank_result,
            "phase_gradients": phase_gradients,
            "mismatch_detected": mismatch_detected,
            "failure_propagated": failure_propagated,
            "rank_failure_propagated": rank_failure_propagated,
            "all_rank_failure_propagated": all_rank_failure_propagated,
        }
        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    finally:
        context.close()
        dist.destroy_process_group()


def test_gloo_contract_covers_reducer_metrics_rank_state_and_failures(
    tmp_path: Path,
) -> None:
    init_file = tmp_path / "gloo-init"
    mp.spawn(
        _gloo_worker,
        args=(init_file.resolve().as_uri(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rows = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert [row["gradient"] for row in rows] == pytest.approx([5.0, 5.0])
    assert [row["parameter"] for row in rows] == pytest.approx([0.5, 0.5])
    assert [row["report_loss"] for row in rows] == pytest.approx([2.5, 2.5])
    assert [row["report_metric"] for row in rows] == pytest.approx([2.5, 2.5])
    assert all(
        row["report_row_losses"] == pytest.approx([1.5, 3.0])
        for row in rows
    )
    assert all(
        row["report_row_metrics"]["quadratic"] == pytest.approx([3.5, 7.0])
        for row in rows
    )
    assert rows[0]["rank_stats"] == rows[1]["rank_stats"] == {
        "prepare_rank_0": 1.0,
        "prepare_rank_1": 2.0,
        "prepare_rank_min": 1.0,
        "prepare_rank_mean": 1.5,
        "prepare_rank_max": 2.0,
        "prepare_straggler_rank": 1.0,
    }
    assert [item["state"]["cursor"] for item in rows[0]["gathered"]] == [10, 11]
    assert rows[1]["gathered"] is None
    assert rows[0]["rank_zero_result"] == 17
    assert rows[1]["rank_zero_result"] is None
    assert [row["all_rank_result"] for row in rows] == [20, 21]
    assert rows[0]["phase_gradients"] == pytest.approx([1.5, 1.5])
    assert rows[1]["phase_gradients"] == pytest.approx([1.5, 1.5])
    assert all(row["mismatch_detected"] for row in rows)
    assert all(row["failure_propagated"] for row in rows)
    assert all(row["rank_failure_propagated"] for row in rows)
    assert all(row["all_rank_failure_propagated"] for row in rows)


def test_distributed_debug_environment_is_explicit_and_non_destructive() -> None:
    disabled: dict[str, str] = {}
    configure_distributed_debug_environment(disabled)
    assert disabled == {}

    enabled = {
        "NCLS_DDP_DEBUG": "1",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "123",
    }
    configure_distributed_debug_environment(enabled)
    assert enabled == {
        "NCLS_DDP_DEBUG": "1",
        "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "123",
        "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
        "TORCH_NCCL_DESYNC_DEBUG": "1",
        "TORCH_NCCL_ENABLE_TIMING": "1",
    }


def test_non_rank_zero_checkpoint_does_not_encode_full_model_state() -> None:
    plugin = replace(
        _plugin(_PhaseMethod()),
        checkpoint=_ForbiddenCheckpointCodec(),
    )
    producer = _RouteProducer()
    engine = TrainingEngine(
        plugin,
        _data_session(producer),
        _config(),
        distributed_context=_RankOneCheckpointContext(),  # type: ignore[arg-type]
    )
    model = plugin.model_factory.create({})

    def forbidden_optimization_state() -> Mapping[str, Any]:
        raise AssertionError("non-rank0 must not snapshot optimizer state")

    checkpoint = engine._checkpoint(
        model,
        0,
        forbidden_optimization_state,
        {},
        [],
    )

    assert checkpoint is None


def test_distributed_initialization_eagerly_binds_cuda_and_separates_control_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: calls.setdefault("device", index))
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        dist,
        "init_process_group",
        lambda **kwargs: calls.setdefault("init", kwargs),
    )
    monkeypatch.setattr(
        dist,
        "new_group",
        lambda **kwargs: calls.setdefault("control", kwargs) or object(),
    )
    monkeypatch.setenv("NCLS_DDP_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("NCLS_DDP_CONTROL_TIMEOUT_SECONDS", "900")
    execution = SimpleNamespace(
        rank=1,
        world_size=2,
        torch_device="cuda:0",
        topology=SimpleNamespace(distributed_backend="nccl"),
    )

    context = DistributedContext.initialize(execution)

    assert calls["device"] == 0
    assert calls["init"]["backend"] == "nccl"
    assert calls["init"]["device_id"] == torch.device("cuda:0")
    assert calls["init"]["timeout"].total_seconds() == 120
    assert calls["control"]["backend"] == "gloo"
    assert calls["control"]["timeout"].total_seconds() == 900
    assert context.rank == 1 and context.world_size == 2
