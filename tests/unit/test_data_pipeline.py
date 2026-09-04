import os
import time

import pytest

from ncls.data import (
    HostPipeline,
    HostPipelineBackpressure,
    HostRequest,
    HostWorkerError,
)


def _delayed_square(value: tuple[int, float]) -> int:
    number, delay = value
    time.sleep(delay)
    return number * number


def _raise_on_negative(value: int) -> int:
    if value < 0:
        raise ValueError("negative fixture")
    return value


def _exit_on_negative(value: int) -> int:
    if value < 0:
        os._exit(7)
    return value


@pytest.mark.parametrize("num_workers", (0, 1, 2))
def test_host_pipeline_preserves_logical_order_and_provenance(num_workers: int) -> None:
    with HostPipeline(
        _delayed_square,
        num_workers=num_workers,
        capacity=3,
        stage="decode",
        rank=2,
    ) as pipeline:
        pipeline.submit(HostRequest(4, (2, 0.04), {"seed": 11}))
        pipeline.submit(HostRequest(7, (3, 0.0), {"seed": 12}))
        pipeline.submit(HostRequest(10, (4, 0.01), {"seed": 13}))

        results = [pipeline.next_result(timeout=10.0) for _ in range(3)]
        assert [item.logical_id for item in results] == [4, 7, 10]
        assert [item.payload for item in results] == [4, 9, 16]
        assert [item.provenance["seed"] for item in results] == [11, 12, 13]
        assert pipeline.state_dict()["last_consumed_logical_id"] == 10
        assert pipeline.profile_snapshot()["peaks"]["host.queue_depth"] == 3.0


def test_host_pipeline_applies_bounded_backpressure_and_can_discard_boundary() -> None:
    with HostPipeline(
        _delayed_square,
        num_workers=1,
        capacity=1,
        stage="read",
    ) as pipeline:
        pipeline.submit(HostRequest(0, (1, 0.01), {}))
        with pytest.raises(HostPipelineBackpressure, match="capacity"):
            pipeline.submit(HostRequest(1, (2, 0.0), {}), timeout=0.0)
        pipeline.drain(discard=True, timeout=10.0)
        assert pipeline.pending_requests == 0
        pipeline.submit(HostRequest(0, (3, 0.0), {}))
        assert pipeline.next_result(timeout=10.0).payload == 9


def test_host_pipeline_propagates_worker_exception_with_request_and_rank() -> None:
    pipeline = HostPipeline(
        _raise_on_negative,
        num_workers=1,
        capacity=2,
        stage="decode",
        rank=3,
    )
    try:
        pipeline.submit(HostRequest(8, -1, {"asset": "broken"}))
        with pytest.raises(HostWorkerError, match="logical request 8 on rank 3"):
            pipeline.next_result(timeout=10.0)
    finally:
        pipeline.close()


def test_host_pipeline_detects_abrupt_worker_exit() -> None:
    pipeline = HostPipeline(
        _exit_on_negative,
        num_workers=1,
        capacity=1,
        stage="read",
    )
    try:
        pipeline.submit(HostRequest(2, -1, {}))
        with pytest.raises(HostWorkerError, match="exited with code 7"):
            pipeline.next_result(timeout=10.0)
    finally:
        pipeline.close()


def test_host_pipeline_resume_requires_empty_boundary() -> None:
    source = HostPipeline(_raise_on_negative, num_workers=0, capacity=1, stage="decode")
    target = HostPipeline(_raise_on_negative, num_workers=0, capacity=1, stage="decode")
    try:
        source.submit(HostRequest(5, 9, {}))
        with pytest.raises(RuntimeError, match="drained and consumed"):
            source.state_dict()
        assert source.next_result().payload == 9
        target.load_state_dict(source.state_dict())
        target.submit(HostRequest(6, 10, {}))
        assert target.next_result().logical_id == 6
        assert target.consumed_requests == 2
    finally:
        source.close()
        target.close()
