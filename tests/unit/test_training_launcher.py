import sys

import pytest

import ncls.cli as cli
import ncls.learning.training.launch as training_launch
from ncls.learning.training import (
    distributed_command,
    prepare_process_environment,
    preflight_topology,
    worker_execution_context,
)


def test_single_gpu_topology_is_direct_on_windows_and_linux() -> None:
    for system in ("Windows", "Linux"):
        topology = preflight_topology((3,), platform_name=system, environment={})
        context = worker_execution_context(topology, environment={})
        assert topology.mode == "single"
        assert topology.distributed_backend is None
        assert context.physical_device == 3
        assert context.torch_device == "cuda:0"
        environment = {}
        prepare_process_environment(topology, environment=environment)
        assert environment == {
            "CUDA_VISIBLE_DEVICES": "3",
            "NCLS_FALCOR_GPU_INDEX": "3",
        }


def test_windows_multi_gpu_fails_before_runtime_initialization() -> None:
    with pytest.raises(RuntimeError, match="only on Linux with NCCL"):
        preflight_topology((0, 1), platform_name="Windows", environment={})


def test_linux_multi_gpu_builds_one_torchrun_job_and_rank_context() -> None:
    topology = preflight_topology((2, 5), platform_name="Linux", environment={})
    assert topology.mode == "distributed-launch"
    assert topology.distributed_backend == "nccl"
    command = distributed_command(
        topology,
        config="configs/training/runs/metal-budgeted-hybrid-pilot.yaml",
        output="artifacts/run/checkpoint.pt",
        extra_arguments=("--stop-at-step", "2"),
    )
    assert command[0] == sys.executable
    assert command[1:3] == ("-m", "torch.distributed.run")
    assert "--nproc-per-node=2" in command
    assert command[-2:] == ("--stop-at-step", "2")

    worker_topology = preflight_topology(
        (2, 5),
        platform_name="Linux",
        environment={"WORLD_SIZE": "2"},
    )
    context = worker_execution_context(
        worker_topology,
        environment={
            "RANK": "1",
            "WORLD_SIZE": "2",
            "LOCAL_RANK": "1",
            "NCLS_DDP_GPU_LIST": "2,5",
            "NCLS_DDP_WORLD_SIZE": "2",
        },
    )
    assert context.rank == 1
    assert context.physical_device == 5
    assert not context.is_rank_zero
    worker_environment = {
        "LOCAL_RANK": "1",
        "CUDA_VISIBLE_DEVICES": "5",
        "NCLS_DDP_GPU_LIST": "2,5",
    }
    prepare_process_environment(worker_topology, environment=worker_environment)
    assert worker_environment["NCLS_FALCOR_GPU_INDEX"] == "5"


def test_launcher_rejects_duplicate_devices_and_world_drift() -> None:
    with pytest.raises(ValueError, match="unique"):
        preflight_topology((0, 0), platform_name="Linux", environment={})
    topology = preflight_topology(
        (0, 1), platform_name="Linux", environment={"WORLD_SIZE": "2"}
    )
    with pytest.raises(RuntimeError, match="world size disagrees"):
        worker_execution_context(
            topology,
            environment={
                "RANK": "0",
                "WORLD_SIZE": "3",
                "LOCAL_RANK": "0",
                "NCLS_DDP_GPU_LIST": "0,1",
            },
        )


def test_new_train_entry_rejects_windows_multi_gpu_before_producer(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("GPU/reference producer must not be constructed")

    monkeypatch.setattr(cli, "OnlineTrainingProducer", forbidden)
    monkeypatch.setattr(training_launch.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="only on Linux with NCCL"):
        cli.main(
            [
                "train",
                "configs/training/runs/nvidia-layer-stack-smoke.yaml",
                "--devices",
                "0,1",
                "--output",
                str(tmp_path / "checkpoint.pt"),
            ]
        )


def test_new_train_entry_does_not_accept_v4_json(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        '{"format_name":"ncls.training-config","format_version":4}',
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="training run fields are invalid"):
        cli.main(
            [
                "train",
                str(legacy),
                "--output",
                str(tmp_path / "checkpoint.pt"),
            ]
        )


def test_legacy_learn_command_is_not_a_public_entry() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["learn", "train"])
