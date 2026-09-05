from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import torch


class ResourceLease(Protocol):
    def release(self) -> None: ...


@dataclass(eq=False)
class ConditioningResource:
    """由 CPU identity 定位、供多个 query row 共享的设备资源。"""

    key: str
    tensors: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    lease: ResourceLease | None = field(default=None, repr=False)
    _owners: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.key or not self.tensors:
            raise ValueError("conditioning resource requires identity and tensors")
        if any(not isinstance(value, torch.Tensor) for value in self.tensors.values()):
            raise TypeError("conditioning resource values must be tensors")
        if len({value.device for value in self.tensors.values()}) != 1:
            raise ValueError("conditioning resource tensors must share one device")
        self.tensors = dict(self.tensors)
        self.metadata = dict(self.metadata)

    @property
    def device(self) -> torch.device:
        return next(iter(self.tensors.values())).device

    def _retain(self) -> None:
        if self._closed:
            raise RuntimeError("cannot retain a released conditioning resource")
        self._owners += 1

    def _release(self) -> None:
        if self._owners < 1:
            raise RuntimeError("conditioning resource owner underflow")
        self._owners -= 1
        if self._owners == 0:
            self._closed = True
            if self.lease is not None:
                self.lease.release()


class ConditioningResources:
    """一个明确的 lease owner；筛选/拼接产生独立 owner，而不复制资源。"""

    def __init__(self, entries: Sequence[ConditioningResource] = ()) -> None:
        values = tuple(entries)
        if len({value.key for value in values}) != len(values):
            raise ValueError("conditioning resource keys must be unique")
        self._entries: tuple[ConditioningResource, ...] = ()
        self._released = False
        retained: list[ConditioningResource] = []
        try:
            for value in values:
                value._retain()
                retained.append(value)
        except BaseException:
            for value in reversed(retained):
                value._release()
            raise
        self._entries = values

    @property
    def entries(self) -> tuple[ConditioningResource, ...]:
        if self._released:
            raise RuntimeError("conditioning resources have been released")
        return self._entries

    def __len__(self) -> int:
        return len(self.entries)

    def retain(self) -> ConditioningResources:
        return ConditioningResources(self.entries)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        entries, self._entries = self._entries, ()
        failure: BaseException | None = None
        for value in reversed(entries):
            try:
                value._release()
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure

    def __del__(self) -> None:
        # 显式 release 是正常路径；构造/调度异常也不能遗留 native lease。
        if hasattr(self, "_released"):
            try:
                self.release()
            except Exception:
                pass

    @staticmethod
    def concatenate(
        collections: Sequence[ConditioningResources],
    ) -> tuple[ConditioningResources, tuple[tuple[int, ...], ...]]:
        entries: list[ConditioningResource] = []
        by_key: dict[str, int] = {}
        remaps: list[tuple[int, ...]] = []
        for collection in collections:
            remap: list[int] = []
            for value in collection.entries:
                index = by_key.get(value.key)
                if index is None:
                    index = len(entries)
                    by_key[value.key] = index
                    entries.append(value)
                else:
                    previous = entries[index]
                    layout = lambda entry: {
                        name: (tuple(tensor.shape), tensor.dtype, tensor.device)
                        for name, tensor in entry.tensors.items()
                    }
                    if previous.metadata != value.metadata or layout(previous) != layout(value):
                        raise ValueError("one conditioning resource key has conflicting metadata")
                remap.append(index)
            remaps.append(tuple(remap))
        return ConditioningResources(entries), tuple(remaps)


@dataclass(frozen=True)
class AdaptedConditioning:
    tensors: Mapping[str, torch.Tensor]
    provenance: Mapping[str, Any]
    resources: ConditioningResources = field(default_factory=ConditioningResources)
    bindings: Mapping[str, torch.Tensor] = field(default_factory=dict)
