from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Mapping, Sequence

from .distributed import configure_distributed_debug_environment


@dataclass(frozen=True)
class ExecutionTopology:
    devices: tuple[int, ...]
    platform_name: str
    mode: str
    distributed_backend: str | None

    def __post_init__(self) -> None:
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError("execution topology devices must be unique and nonempty")
        if any(item < 0 for item in self.devices):
            raise ValueError("execution topology devices must be nonnegative")
        if self.mode not in {"single", "distributed-launch", "distributed-worker"}:
            raise ValueError("execution topology mode is invalid")
        if self.mode == "single" and self.distributed_backend is not None:
            raise ValueError("single-device topology cannot select a distributed backend")
        if self.mode != "single" and self.distributed_backend != "nccl":
            raise ValueError("formal distributed training requires NCCL")


@dataclass(frozen=True)
class ExecutionContext:
    topology: ExecutionTopology
    rank: int
    world_size: int
    local_rank: int
    physical_device: int
    torch_device: str
    is_rank_zero: bool

    def __post_init__(self) -> None:
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("execution context rank is outside the distributed world")
        if not 0 <= self.local_rank < self.world_size:
            raise ValueError("execution context local rank is outside the distributed world")
        if self.physical_device != self.topology.devices[self.local_rank]:
            raise ValueError("execution context physical device disagrees with topology")
        if self.torch_device != "cuda:0" or self.is_rank_zero != (self.rank == 0):
            raise ValueError("execution context local device or rank-zero identity is invalid")


def preflight_topology(
    devices: Sequence[int],
    *,
    platform_name: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> ExecutionTopology:
    values = tuple(int(item) for item in devices)
    if not values or any(item < 0 for item in values) or len(set(values)) != len(values):
        raise ValueError("devices must contain unique nonnegative GPU indices")
    system = platform.system() if platform_name is None else platform_name
    environment = os.environ if environment is None else environment
    world_raw = environment.get("WORLD_SIZE")
    worker = world_raw is not None
    if len(values) == 1:
        if worker:
            raise RuntimeError("a distributed worker cannot use a single-device plan")
        return ExecutionTopology(values, system, "single", None)
    if system != "Linux":
        raise RuntimeError(
            "multi-GPU online training is supported only on Linux with NCCL; "
            f"requested {len(values)} devices on {system}"
        )
    return ExecutionTopology(
        values,
        system,
        "distributed-worker" if worker else "distributed-launch",
        "nccl",
    )


def worker_execution_context(
    topology: ExecutionTopology,
    *,
    environment: Mapping[str, str] | None = None,
) -> ExecutionContext:
    environment = os.environ if environment is None else environment
    if topology.mode == "single":
        return ExecutionContext(topology, 0, 1, 0, topology.devices[0], "cuda:0", True)
    required = {name: environment.get(name) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}
    if any(value is None for value in required.values()):
        raise RuntimeError("distributed worker requires RANK, WORLD_SIZE and LOCAL_RANK")
    rank_value = required["RANK"]
    world_value = required["WORLD_SIZE"]
    local_value = required["LOCAL_RANK"]
    assert rank_value is not None and world_value is not None and local_value is not None
    try:
        rank = int(rank_value)
        world = int(world_value)
        local = int(local_value)
    except ValueError as error:
        raise RuntimeError("distributed worker rank values must be integers") from error
    if world != len(topology.devices):
        raise RuntimeError("distributed worker world size disagrees with device topology")
    gpu_list = environment.get("NCLS_DDP_GPU_LIST")
    expected_gpu_list = ",".join(str(item) for item in topology.devices)
    if gpu_list != expected_gpu_list:
        raise RuntimeError("distributed worker GPU list disagrees with device topology")
    declared_world = environment.get("NCLS_DDP_WORLD_SIZE")
    if declared_world is not None and int(declared_world) != world:
        raise RuntimeError("distributed worker declared world size disagrees")
    return ExecutionContext(
        topology,
        rank,
        world,
        local,
        topology.devices[local],
        "cuda:0",
        rank == 0,
    )


def prepare_process_environment(
    topology: ExecutionTopology,
    *,
    environment: dict[str, str] | None = None,
) -> None:
    target = os.environ if environment is None else environment
    if topology.mode == "distributed-launch":
        raise ValueError("distributed-launch topology belongs to the outer launcher")
    if topology.mode == "single":
        physical = str(topology.devices[0])
        visible = target.get("CUDA_VISIBLE_DEVICES")
        if visible not in {None, "", physical}:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES disagrees with the selected single physical GPU"
            )
        falcor = target.get("NCLS_FALCOR_GPU_INDEX")
        if falcor not in {None, "", physical}:
            raise RuntimeError(
                "NCLS_FALCOR_GPU_INDEX disagrees with the selected single physical GPU"
            )
        target["CUDA_VISIBLE_DEVICES"] = physical
        target["NCLS_FALCOR_GPU_INDEX"] = physical
        return
    gpu_list = ",".join(str(item) for item in topology.devices)
    if target.get("NCLS_DDP_GPU_LIST") != gpu_list:
        raise RuntimeError("distributed worker GPU list disagrees with execution topology")
    local_raw = target.get("LOCAL_RANK")
    if local_raw is None:
        raise RuntimeError("distributed worker requires LOCAL_RANK")
    local_rank = int(local_raw)
    physical = str(topology.devices[local_rank])
    if target.get("CUDA_VISIBLE_DEVICES") != physical:
        raise RuntimeError(
            "distributed worker must expose only its assigned physical GPU"
        )
    falcor = target.get("NCLS_FALCOR_GPU_INDEX")
    if falcor not in {None, "", physical}:
        raise RuntimeError("distributed worker Falcor GPU disagrees with its assignment")
    target["NCLS_FALCOR_GPU_INDEX"] = physical


def distributed_command(
    topology: ExecutionTopology,
    *,
    config: Path | str,
    output: Path | str,
    extra_arguments: Sequence[str] = (),
) -> tuple[str, ...]:
    if topology.mode != "distributed-launch":
        raise ValueError("torchrun command requires a distributed-launch topology")
    return (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={len(topology.devices)}",
        "-m",
        "ncls.ddp_worker",
        "-m",
        "ncls.cli",
        "train",
        str(config),
        "--output",
        str(output),
        *tuple(extra_arguments),
    )


def launch_distributed(
    topology: ExecutionTopology,
    *,
    config: Path | str,
    output: Path | str,
    extra_arguments: Sequence[str] = (),
) -> int:
    command = distributed_command(
        topology, config=config, output=output, extra_arguments=extra_arguments
    )
    environment = dict(os.environ)
    gpu_list = ",".join(str(item) for item in topology.devices)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_list,
            "NCLS_DDP_GPU_LIST": gpu_list,
            "NCLS_DDP_WORLD_SIZE": str(len(topology.devices)),
        }
    )
    configure_distributed_debug_environment(environment)
    return int(subprocess.run(command, env=environment, check=False).returncode)


__all__ = [
    "ExecutionContext",
    "ExecutionTopology",
    "distributed_command",
    "launch_distributed",
    "prepare_process_environment",
    "preflight_topology",
    "worker_execution_context",
]
