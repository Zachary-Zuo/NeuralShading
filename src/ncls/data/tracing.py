from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from time import perf_counter_ns
from typing import Iterator, Mapping


@dataclass(frozen=True)
class PipelineTraceSnapshot:
    counters: Mapping[str, int]
    gauges: Mapping[str, float]
    peaks: Mapping[str, float]
    duration_seconds: Mapping[str, float]

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "peaks": dict(self.peaks),
            "duration_seconds": dict(self.duration_seconds),
        }


class PipelineTrace:
    """Thread-safe stage metrics which never synchronize a GPU on record."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._peaks: dict[str, float] = {}
        self._duration_ns: dict[str, int] = {}

    @staticmethod
    def _name(value: str) -> str:
        name = str(value).strip()
        if not name:
            raise ValueError("pipeline trace metric name is required")
        return name

    def increment(self, name: str, value: int = 1) -> None:
        metric = self._name(name)
        amount = int(value)
        if amount < 0:
            raise ValueError("pipeline trace counter increments must be nonnegative")
        with self._lock:
            self._counters[metric] = self._counters.get(metric, 0) + amount

    def gauge(self, name: str, value: int | float) -> None:
        metric = self._name(name)
        observed = float(value)
        with self._lock:
            self._gauges[metric] = observed
            self._peaks[metric] = max(observed, self._peaks.get(metric, observed))

    def add_duration_ns(self, stage: str, duration_ns: int) -> None:
        metric = self._name(stage)
        value = int(duration_ns)
        if value < 0:
            raise ValueError("pipeline trace duration must be nonnegative")
        with self._lock:
            self._duration_ns[metric] = self._duration_ns.get(metric, 0) + value

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        metric = self._name(stage)
        started = perf_counter_ns()
        try:
            yield
        finally:
            self.add_duration_ns(metric, perf_counter_ns() - started)

    def snapshot(self, *, reset: bool = False) -> PipelineTraceSnapshot:
        with self._lock:
            result = PipelineTraceSnapshot(
                dict(self._counters),
                dict(self._gauges),
                dict(self._peaks),
                {
                    name: duration / 1_000_000_000.0
                    for name, duration in self._duration_ns.items()
                },
            )
            if reset:
                self._counters.clear()
                self._gauges.clear()
                self._peaks.clear()
                self._duration_ns.clear()
            return result


__all__ = ["PipelineTrace", "PipelineTraceSnapshot"]
