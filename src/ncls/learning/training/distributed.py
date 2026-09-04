from __future__ import annotations

from datetime import timedelta
import os
from typing import Any, Callable, Mapping, Sequence, TypeVar, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from ncls.learning.batches import OnlineTrainingBatch
from ncls.learning.methods.contracts import ObjectiveFacet


_T = TypeVar("_T")


class DistributedObjective(nn.Module):
    """让 method objective 成为 DDP 可追踪的统一 forward owner。"""

    def __init__(self, objective: ObjectiveFacet, model: nn.Module) -> None:
        super().__init__()
        self.objective = objective
        self.model = model
        self._last_metrics: Mapping[str, torch.Tensor | float] | None = None

    def forward(
        self,
        batches: Mapping[str, OnlineTrainingBatch],
        phase: Mapping[str, Any],
    ) -> torch.Tensor:
        loss, metrics = self.objective.compute(self.model, batches, phase)
        self._last_metrics = metrics
        return loss

    def take_metrics(self) -> Mapping[str, torch.Tensor | float]:
        if self._last_metrics is None:
            raise RuntimeError("distributed objective has no completed forward metrics")
        metrics = self._last_metrics
        self._last_metrics = None
        return metrics


class DistributedContext:
    """集中拥有 data/control process group 及其 collective 顺序。"""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        control_group: Any | None = None,
        owns_default_group: bool = False,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("distributed rank is outside the world")
        if world_size > 1 and control_group is None:
            raise ValueError("distributed execution requires a control process group")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = torch.device(device)
        self.control_group = control_group
        self._owns_default_group = bool(owns_default_group)
        self._closed = False

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    @classmethod
    def single(cls, device: torch.device | str) -> "DistributedContext":
        return cls(rank=0, world_size=1, device=torch.device(device))

    @classmethod
    def initialize(
        cls,
        execution_context: Any,
        *,
        timeout_seconds: float | None = None,
    ) -> "DistributedContext":
        rank = int(execution_context.rank)
        world = int(execution_context.world_size)
        device = torch.device(str(execution_context.torch_device))
        if world == 1:
            return cls.single(device)
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("DDP worker must see exactly one remapped CUDA device")
        if device.type != "cuda" or device.index not in {None, 0}:
            raise RuntimeError("DDP worker must use its remapped cuda:0 device")
        torch.cuda.set_device(0)
        timeout = timedelta(
            seconds=(
                float(os.environ.get("NCLS_DDP_TIMEOUT_SECONDS", "300"))
                if timeout_seconds is None
                else float(timeout_seconds)
            )
        )
        if timeout.total_seconds() <= 0:
            raise ValueError("NCLS_DDP_TIMEOUT_SECONDS must be positive")
        control_timeout = timedelta(
            seconds=float(
                os.environ.get("NCLS_DDP_CONTROL_TIMEOUT_SECONDS", "1800")
            )
        )
        if control_timeout.total_seconds() <= 0:
            raise ValueError("NCLS_DDP_CONTROL_TIMEOUT_SECONDS must be positive")
        initialized_here = not dist.is_initialized()
        if initialized_here:
            dist.init_process_group(
                backend=execution_context.topology.distributed_backend,
                rank=rank,
                world_size=world,
                timeout=timeout,
                device_id=torch.device("cuda:0"),
            )
        elif (
            dist.get_rank() != rank
            or dist.get_world_size() != world
            or dist.get_backend() != execution_context.topology.distributed_backend
        ):
            raise RuntimeError("initialized process group disagrees with execution context")
        try:
            control_group = dist.new_group(
                ranks=list(range(world)),
                backend="gloo",
                timeout=control_timeout,
                group_desc="ncls-control",
            )
        except BaseException:
            if initialized_here and dist.is_initialized():
                dist.destroy_process_group()
            raise
        return cls(
            rank=rank,
            world_size=world,
            device=torch.device("cuda:0"),
            control_group=control_group,
            owns_default_group=initialized_here,
        )

    def validate_descriptor(self, scope: str, descriptor: Any) -> None:
        if not scope:
            raise ValueError("distributed descriptor scope is required")
        if not self.is_distributed:
            return
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, descriptor, group=self.control_group)
        if any(value != gathered[0] for value in gathered[1:]):
            raise RuntimeError(
                f"distributed descriptor mismatch for {scope!r}: {gathered!r}"
            )

    def build_objective(
        self,
        objective: ObjectiveFacet,
        model: nn.Module,
        *,
        phase_name: str,
    ) -> tuple[DistributedObjective, nn.Module]:
        owner = DistributedObjective(objective, model)
        if not self.is_distributed:
            return owner, owner
        if self.device.type != "cuda":
            raise RuntimeError("DDP training requires a CUDA objective device")
        parameter_descriptor = tuple(
            (
                name,
                tuple(int(value) for value in parameter.shape),
                str(parameter.dtype),
                bool(parameter.requires_grad),
            )
            for name, parameter in model.named_parameters()
        )
        self.validate_descriptor(
            f"phase:{phase_name}:parameters", parameter_descriptor
        )
        device_index = 0 if self.device.index is None else self.device.index
        wrapped = DistributedDataParallel(
            owner,
            device_ids=[device_index],
            output_device=device_index,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=False,
        )
        return owner, wrapped

    def gather_rank_payload(self, payload: Any) -> tuple[Mapping[str, Any], ...] | None:
        local = {
            "rank": self.rank,
            "world_size": self.world_size,
            "state": payload,
        }
        if not self.is_distributed:
            return (local,)
        gathered: list[Any] | None = [None] * self.world_size if self.is_rank_zero else None
        dist.gather_object(
            local,
            object_gather_list=gathered,
            dst=0,
            group=self.control_group,
        )
        if gathered is None:
            return None
        return tuple(dict(value) for value in gathered)

    def run_rank_zero(self, label: str, action: Callable[[], _T]) -> _T | None:
        if not label:
            raise ValueError("rank-zero operation label is required")
        if not self.is_distributed:
            return action()
        result: _T | None = None
        local_error: BaseException | None = None
        payload: list[Any] = [None]
        if self.is_rank_zero:
            try:
                result = action()
                payload[0] = {"label": label, "success": True}
            except BaseException as error:
                local_error = error
                payload[0] = {
                    "label": label,
                    "success": False,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
        dist.broadcast_object_list(payload, src=0, group=self.control_group)
        status = payload[0]
        if not isinstance(status, Mapping) or status.get("label") != label:
            raise RuntimeError(f"distributed rank-zero protocol mismatch for {label!r}")
        if not bool(status.get("success", False)):
            if local_error is not None:
                raise local_error
            raise RuntimeError(
                f"rank0 {label} failed: {status.get('error_type', 'Error')}: "
                f"{status.get('message', '')}"
            )
        return result

    def synchronize_rank_errors(
        self,
        label: str,
        local_error: BaseException | None,
    ) -> None:
        if not self.is_distributed:
            if local_error is not None:
                raise local_error
            return
        local = {
            "rank": self.rank,
            "label": label,
            "success": local_error is None,
            "error_type": None if local_error is None else type(local_error).__name__,
            "message": None if local_error is None else str(local_error),
        }
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, local, group=self.control_group)
        failures = [dict(value) for value in gathered if not bool(value.get("success"))]
        if failures:
            if local_error is not None:
                raise local_error
            raise RuntimeError(f"distributed {label} failed on ranks: {failures!r}")

    def run_all_ranks(self, label: str, action: Callable[[], _T]) -> _T:
        """在低频控制边界执行rank-local动作，并传播任一rank的异常。"""

        result: _T | None = None
        local_error: BaseException | None = None
        try:
            result = action()
        except BaseException as error:
            local_error = error
        self.synchronize_rank_errors(label, local_error)
        return cast(_T, result)

    def reduce_report(
        self,
        loss: torch.Tensor,
        metrics: Mapping[str, torch.Tensor | float],
        *,
        scope: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        if not self.is_distributed:
            return loss.detach(), dict(metrics)
        scalar_names = tuple(
            sorted(
                name
                for name, value in metrics.items()
                if not isinstance(value, torch.Tensor) or value.ndim == 0
            )
        )
        descriptor = tuple(
            (
                name,
                "tensor" if isinstance(metrics[name], torch.Tensor) else "float",
                str(metrics[name].dtype) if isinstance(metrics[name], torch.Tensor) else "float",
            )
            for name in scalar_names
        )
        self.validate_descriptor(scope, descriptor)
        packed_values = [loss.detach().to(device=loss.device, dtype=torch.float64)]
        for name in scalar_names:
            value = metrics[name]
            packed_values.append(
                value.detach().to(device=loss.device, dtype=torch.float64)
                if isinstance(value, torch.Tensor)
                else torch.tensor(float(value), device=loss.device, dtype=torch.float64)
            )
        packed = torch.stack(packed_values)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(float(self.world_size))
        reduced_scalars = {
            name: packed[index + 1] for index, name in enumerate(scalar_names)
        }
        reduced: dict[str, torch.Tensor | float] = {}
        for name, value in metrics.items():
            if name not in reduced_scalars:
                reduced[name] = value
            elif isinstance(value, torch.Tensor):
                reduced[name] = reduced_scalars[name].to(dtype=value.dtype)
            else:
                reduced[name] = float(reduced_scalars[name].item())
        return packed[0].to(dtype=loss.dtype), reduced

    def rank_statistics(
        self,
        values: Mapping[str, float],
        *,
        scope: str,
    ) -> dict[str, float]:
        if not self.is_distributed or not values:
            return {}
        names = tuple(sorted(values))
        self.validate_descriptor(scope, names)
        local = torch.tensor(
            [float(values[name]) for name in names],
            device=self.device,
            dtype=torch.float64,
        )
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local)
        matrix = torch.stack(gathered).cpu()
        result: dict[str, float] = {}
        for index, name in enumerate(names):
            column = matrix[:, index]
            for rank, value in enumerate(column):
                result[f"{name}_rank_{rank}"] = float(value)
            result[f"{name}_rank_min"] = float(column.min())
            result[f"{name}_rank_mean"] = float(column.mean())
            result[f"{name}_rank_max"] = float(column.max())
            result[f"{name}_straggler_rank"] = float(torch.argmax(column))
        return result

    def ddp_logging_metrics(self, execution_objective: nn.Module) -> dict[str, float]:
        if not isinstance(execution_objective, DistributedDataParallel):
            return {}
        getter = getattr(execution_objective, "_get_ddp_logging_data", None)
        if not callable(getter):
            return {}
        data = dict(getter())
        bucket_sizes_raw = data.get("bucket_sizes", "")
        if isinstance(bucket_sizes_raw, str):
            bucket_sizes = tuple(
                int(value.strip())
                for value in bucket_sizes_raw.split(",")
                if value.strip()
            )
        elif isinstance(bucket_sizes_raw, Sequence):
            bucket_sizes = tuple(int(value) for value in bucket_sizes_raw)
        else:
            bucket_sizes = (int(bucket_sizes_raw),) if bucket_sizes_raw else ()
        result = {
            "profile/ddp_bucket_count": float(len(bucket_sizes)),
            "profile/ddp_bucket_bytes": float(sum(bucket_sizes)),
        }
        for source, target in (
            ("iteration", "profile/ddp_iteration"),
            ("num_parameter_tensors", "profile/ddp_parameter_tensors"),
            ("total_parameter_size_bytes", "profile/ddp_parameter_bytes"),
            ("unused_parameter_size", "profile/ddp_unused_parameter_bytes"),
            ("can_set_static_graph", "profile/ddp_can_set_static_graph"),
        ):
            value = data.get(source)
            if isinstance(value, (int, float)):
                result[target] = float(value)
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.control_group is not None and dist.is_initialized():
            dist.destroy_process_group(self.control_group)
        if self._owns_default_group and dist.is_initialized():
            dist.destroy_process_group()


def configure_distributed_debug_environment(
    environment: dict[str, str],
) -> None:
    """按显式 opt-in 启用 PyTorch 2.11 已存在的 NCCL 诊断开关。"""

    if environment.get("NCLS_DDP_DEBUG") != "1":
        return
    environment.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    environment.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "20000")
    environment.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    environment.setdefault("TORCH_NCCL_DESYNC_DEBUG", "1")
    environment.setdefault("TORCH_NCCL_ENABLE_TIMING", "1")


__all__ = [
    "DistributedContext",
    "DistributedObjective",
    "configure_distributed_debug_environment",
]
