from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ncls.core.material import BINARY_SIZE
from ncls.data.reference import FalcorReferenceEvaluator


class _Buffer:
    def __init__(self, output: np.ndarray | None = None) -> None:
        self.output = output
        self.uploads: list[np.ndarray] = []

    def from_numpy(self, value: np.ndarray) -> None:
        self.uploads.append(np.asarray(value).copy())

    def to_numpy(self) -> np.ndarray:
        assert self.output is not None
        return self.output.copy()


class _Device:
    def __init__(self) -> None:
        self.end_frame_calls = 0

    def end_frame(self) -> None:
        self.end_frame_calls += 1


class _Compute:
    def __init__(self) -> None:
        self.globals = SimpleNamespace()
        self.dispatches: list[int] = []

    def execute(self, *, threads_x: int) -> None:
        self.dispatches.append(threads_x)


def test_query_group_readback_advances_frame_fence(monkeypatch) -> None:
    monkeypatch.setattr(
        "ncls.data.reference.pack_layer_stack",
        lambda material: bytes(BINARY_SIZE),
    )
    evaluator = object.__new__(FalcorReferenceEvaluator)
    evaluator.max_query_group_batch = 1
    evaluator.light_count = 2
    evaluator.query_capacity = 2
    evaluator.light_directions = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )
    evaluator.material_buffer = _Buffer()
    evaluator.view_buffer = _Buffer()
    evaluator.seed_buffer = _Buffer()
    evaluator.light_buffer = _Buffer()
    output = np.arange(8, dtype=np.float32).view(np.uint8)
    evaluator.outputs = [_Buffer(output) for _ in range(4)]
    evaluator.compute = _Compute()
    evaluator.device = _Device()

    result = evaluator.evaluate_query_groups(
        [object()],
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        sample_count_per_replica=4,
        query_group_seeds=np.asarray([7], dtype=np.uint32),
    )

    assert evaluator.compute.dispatches == [2]
    assert evaluator.device.end_frame_calls == 1
    assert len(result) == 4
    assert all(value.shape == (1, 2, 3) for value in result)

