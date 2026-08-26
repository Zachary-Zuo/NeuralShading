from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch

from ncls.learning.pipelines import create_pipeline
from ncls.learning.training.checkpoint import load_checkpoint, sha256_file
from ncls.learning.training.config import TrainingConfig


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.90)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


@torch.no_grad()
def _measure(
    operation: Callable[[], torch.Tensor],
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    prediction = operation()
    for _ in range(warmup):
        prediction = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            prediction = operation()
            end.record()
        torch.cuda.synchronize(device)
        device_ms = [float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)]
        wall_ms = []
        for _ in range(min(iterations, 25)):
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            prediction = operation()
            torch.cuda.synchronize(device)
            wall_ms.append(1000.0 * (time.perf_counter() - begin))
        return {
            "device_execution_ms": _summary(device_ms),
            "synchronized_wall_ms": _summary(wall_ms),
        }, prediction
    wall_ms = []
    for _ in range(iterations):
        begin = time.perf_counter()
        prediction = operation()
        wall_ms.append(1000.0 * (time.perf_counter() - begin))
    return {"synchronized_wall_ms": _summary(wall_ms)}, prediction


def benchmark_checkpoint(
    data_path: Path | str,
    checkpoint_path: Path | str,
    output_path: Path | str,
    *,
    device_name: str | None = None,
    packet_size: int = 256,
    warmup: int = 10,
    iterations: int = 50,
) -> dict[str, Any]:
    if packet_size < 2 or warmup < 0 or iterations < 1:
        raise ValueError("benchmark sizes must use packet>=2, warmup>=0 and iterations>=1")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = TrainingConfig.from_dict(checkpoint["training_config"])
    if checkpoint.get("training_config_sha256") != config.resolved_sha256:
        raise ValueError("benchmark checkpoint training config hash is unsupported")
    pipeline = create_pipeline(str(checkpoint.get("pipeline", "")))
    if checkpoint.get("pipeline_sha256") != pipeline.descriptor.sha256:
        raise ValueError("benchmark checkpoint pipeline identity is unsupported")
    fitted = checkpoint.get("fitted_training_state")
    if not isinstance(fitted, dict) or checkpoint.get("fitted_training_state_sha256") != _sha256_json(fitted):
        raise ValueError("benchmark checkpoint fitted state is missing or invalid")
    pipeline.load_training_state(fitted)
    device = torch.device(device_name or config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA benchmark requested but CUDA is unavailable")
    model = pipeline.create_model(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    store = pipeline.open_store(str(data_path))
    try:
        if checkpoint.get("data_id") != store.data_id:
            raise ValueError("benchmark checkpoint data_id does not match the dataset")
        indices = store.select_indices(
            pipeline.evaluation_indices(store, "validation"),
            config.dataset_selection,
        )
        if not len(indices):
            raise ValueError("benchmark requires a validation query group")
        reference = indices[:1]
        raw = store.batch(reference)
        if raw["wi"].shape[1] < packet_size:
            raise ValueError("benchmark packet exceeds the selected query direction count")
        state_index = torch.as_tensor(raw["state_index"], dtype=torch.long, device=device)
        wo = torch.as_tensor(raw["wo"], dtype=torch.float32, device=device)
        wi = torch.as_tensor(raw["wi"][:, :packet_size], dtype=torch.float32, device=device)

        def operation(count: int) -> torch.Tensor:
            batch = {"state_index": state_index, "wo": wo, "wi": wi[:, :count]}
            return pipeline.predict_f(model, batch, store, device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        single, single_prediction = _measure(
            lambda: operation(1), device, warmup=warmup, iterations=iterations
        )
        packet, packet_prediction = _measure(
            lambda: operation(packet_size), device, warmup=warmup, iterations=iterations
        )
        validation_states = np.asarray(
            store.batch(indices, fields=("state_index",))["state_index"], dtype=np.int64
        )
        state_ids = list(map(str, store.state_strings("state_id").tolist()))
        first_query_by_state = {
            state_index: int(indices[np.flatnonzero(validation_states == state_index)[0]])
            for state_index in range(store.state_count)
            if np.any(validation_states == state_index)
        }
        if len(first_query_by_state) != store.state_count:
            raise ValueError("benchmark requires a validation query for every state")
        single_query_by_state: dict[str, Any] = {}
        single_query_time_microseconds_by_state: dict[str, float] = {}
        for state_index, query_index in first_query_by_state.items():
            state_raw = store.batch(np.asarray([query_index], dtype=np.int64))
            state_tensor = torch.as_tensor(
                state_raw["state_index"], dtype=torch.long, device=device
            )
            state_wo = torch.as_tensor(state_raw["wo"], dtype=torch.float32, device=device)
            state_wi = torch.as_tensor(
                state_raw["wi"][:, :1], dtype=torch.float32, device=device
            )

            def state_operation() -> torch.Tensor:
                batch = {"state_index": state_tensor, "wo": state_wo, "wi": state_wi}
                return pipeline.predict_f(model, batch, store, device)

            timing, state_prediction = _measure(
                state_operation, device, warmup=warmup, iterations=iterations
            )
            if not torch.all(torch.isfinite(state_prediction)).item():
                raise ValueError("benchmark encountered a non-finite per-state prediction")
            selected_timing = timing.get(
                "device_execution_ms", timing["synchronized_wall_ms"]
            )
            state_id = state_ids[state_index]
            single_query_by_state[state_id] = timing
            single_query_time_microseconds_by_state[state_id] = (
                1000.0 * float(selected_timing["median"])
            )
        packet_median = (
            packet.get("device_execution_ms", packet["synchronized_wall_ms"])["median"]
        )
        report: dict[str, Any] = {
            "schema": {"name": "p1-query-benchmark", "version": 1},
            "data_id": store.data_id,
            "checkpoint": str(Path(checkpoint_path)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_step": int(checkpoint["step"]),
            "pipeline": pipeline.descriptor.name,
            "pipeline_sha256": pipeline.descriptor.sha256,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "warmup": warmup,
            "iterations": iterations,
            "single_query": {"directions": 1, **single},
            "single_query_by_state": single_query_by_state,
            "single_query_time_microseconds_by_state": (
                single_query_time_microseconds_by_state
            ),
            "coherent_packet": {
                "directions": packet_size,
                **packet,
                "median_microseconds_per_direction": 1000.0 * packet_median / packet_size,
            },
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "finite_output": bool(
                torch.all(torch.isfinite(single_prediction)).item()
                and torch.all(torch.isfinite(packet_prediction)).item()
            ),
            "cost_model": dict(pipeline.parameter_costs(model)),
        }
    finally:
        store.close()
    report["report_sha256"] = _sha256_json(report)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
