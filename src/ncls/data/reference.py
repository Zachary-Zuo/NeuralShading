from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Sequence

import numpy as np

from ncls.core.material import BINARY_SIZE, LayerStackIR, pack_layer_stack

from .statistics import ReplicaMoments, combine_replica_moments


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_LAYER_STACK_SHADER = PROJECT_ROOT / "shaders" / "ncls" / "data" / "reference_layer_stack.cs.slang"


@dataclass(frozen=True)
class EvaluatedReferenceBatch:
    mean: np.ndarray
    variance: np.ndarray
    replica_mean_a: np.ndarray
    replica_mean_b: np.ndarray
    sample_count: np.ndarray


class FalcorReferenceEvaluator:
    """用锁定 Falcor/Slang 实现执行新的随机游走参考解。"""

    def __init__(
        self,
        light_directions: np.ndarray,
        *,
        max_depth: int = 64,
        max_query_group_batch: int = 64,
        light_index_offset: int = 0,
    ) -> None:
        try:
            falcor = importlib.import_module("falcor")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Falcor reference evaluation must run through scripts/run_falcor_python.ps1"
            ) from exc
        self._falcor = falcor
        directions = np.asarray(light_directions, dtype=np.float32)
        if directions.ndim not in {2, 3} or directions.shape[-1] not in {3, 4}:
            raise ValueError("light_directions must have shape [light, 3 or 4] or [group, light, 3 or 4]")
        self.light_directions = np.ascontiguousarray(directions[..., :3])
        self.light_count = int(self.light_directions.shape[-2])
        if max_query_group_batch < 1 or max_depth < 1:
            raise ValueError("max_query_group_batch and max_depth must be positive")
        self.max_query_group_batch = max_query_group_batch
        self.query_capacity = self.light_count * max_query_group_batch
        self.max_depth = max_depth
        self.device = falcor.Device(type=falcor.DeviceType.D3D12)
        self.material_buffer = self._buffer(BINARY_SIZE, max_query_group_batch)
        self.view_buffer = self._buffer(16, max_query_group_batch)
        self.seed_buffer = self._buffer(4, max_query_group_batch)
        self.light_buffer = self._buffer(16, self.query_capacity)
        self.outputs = [self._buffer(16, self.query_capacity, writable=True) for _ in range(4)]
        self.compute = falcor.ComputePass(
            self.device,
            file=REFERENCE_LAYER_STACK_SHADER,
            cs_entry="evaluateReferenceQueryGroups",
        )
        self.compute.globals.gMaterialStates = self.material_buffer
        self.compute.globals.gViewDirections = self.view_buffer
        self.compute.globals.gLightDirections = self.light_buffer
        self.compute.globals.gQueryGroupSeeds = self.seed_buffer
        for name, output in zip(
            ("gMeanA", "gMeanSquareA", "gMeanB", "gMeanSquareB"),
            self.outputs,
            strict=True,
        ):
            setattr(self.compute.globals, name, output)
        self.compute.globals.gLightCount = self.light_count
        self.compute.globals.gLightIndexOffset = light_index_offset
        self.compute.globals.gMaxDepth = max_depth

    def _buffer(self, stride: int, element_count: int, *, writable: bool = False):
        flags = self._falcor.ResourceBindFlags.ShaderResource
        if writable:
            flags |= self._falcor.ResourceBindFlags.UnorderedAccess
        return self.device.create_structured_buffer(
            struct_size=stride,
            element_count=element_count,
            bind_flags=flags,
        )

    def evaluate_query_groups(
        self,
        materials: Sequence[LayerStackIR],
        view_directions: np.ndarray,
        *,
        sample_count_per_replica: int,
        query_group_seeds: np.ndarray,
        light_directions: np.ndarray | None = None,
        sample_offset: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        query_group_count = len(materials)
        if not 1 <= query_group_count <= self.max_query_group_batch:
            raise ValueError(f"batch must contain 1..{self.max_query_group_batch} query groups")
        if sample_count_per_replica < 1 or sample_offset < 0:
            raise ValueError("sample count must be positive and offset nonnegative")
        views = np.asarray(view_directions, dtype=np.float32)
        seeds = np.asarray(query_group_seeds, dtype=np.uint32)
        if views.shape not in {(query_group_count, 3), (query_group_count, 4)} or seeds.shape != (query_group_count,):
            raise ValueError("materials, view_directions and query_group_seeds must have matching counts")
        lights = self.light_directions if light_directions is None else np.asarray(light_directions, dtype=np.float32)
        if lights.ndim == 2:
            lights = np.broadcast_to(lights[None, ...], (query_group_count, *lights.shape))
        if lights.shape not in {
            (query_group_count, self.light_count, 3),
            (query_group_count, self.light_count, 4),
        }:
            raise ValueError("light directions must match query groups and evaluator light_count")

        packed = np.zeros(BINARY_SIZE * self.max_query_group_batch, dtype=np.uint8)
        packed[: BINARY_SIZE * query_group_count] = np.frombuffer(
            b"".join(pack_layer_stack(material) for material in materials), dtype=np.uint8
        )
        padded_views = np.zeros((self.max_query_group_batch, 4), dtype=np.float32)
        padded_views[:query_group_count, :3] = views[:, :3]
        padded_seeds = np.zeros(self.max_query_group_batch, dtype=np.uint32)
        padded_seeds[:query_group_count] = seeds
        self.material_buffer.from_numpy(packed)
        self.view_buffer.from_numpy(padded_views)
        self.seed_buffer.from_numpy(padded_seeds)
        query_count = query_group_count * self.light_count
        padded_lights = np.zeros((self.query_capacity, 4), dtype=np.float32)
        padded_lights[:query_count, :3] = lights[..., :3].reshape(query_count, 3)
        self.light_buffer.from_numpy(padded_lights)
        self.compute.globals.gQueryCount = query_count
        self.compute.globals.gSampleCountPerReplica = sample_count_per_replica
        self.compute.globals.gSampleOffset = sample_offset
        self.compute.globals.gSeed = 0
        self.compute.execute(threads_x=query_count)
        result = [
            output.to_numpy().view(np.float32).reshape(self.query_capacity, 4)[:query_count, :3]
            .reshape(query_group_count, self.light_count, 3)
            .copy()
            for output in self.outputs
        ]
        # Buffer.to_numpy() routes through Falcor's readback heap. Those pages are
        # released against the frame fence, so an offline compute loop that never
        # advances a frame otherwise retains one readback allocation per Monte
        # Carlo batch. The arrays above own their copied CPU storage; advance the
        # fence now so subsequent batches can reuse the readback pages.
        self.device.end_frame()
        return result[0], result[1], result[2], result[3]


def _merge_batch(
    mean: np.ndarray,
    m2: np.ndarray,
    old_count: np.ndarray,
    active: np.ndarray,
    batch_mean: np.ndarray,
    batch_variance: np.ndarray,
    batch_count: int,
) -> None:
    old = old_count[active].astype(np.float64)[:, None, None]
    new = old + float(batch_count)
    delta = batch_mean - mean[active]
    mean[active] += delta * (float(batch_count) / new)
    m2[active] += batch_variance * float(batch_count)
    m2[active] += delta * delta * old * float(batch_count) / new


def evaluate_reference_adaptive(
    evaluator: FalcorReferenceEvaluator,
    materials: Sequence[LayerStackIR],
    view_directions: np.ndarray,
    *,
    query_group_seeds: np.ndarray,
    light_directions: np.ndarray | None = None,
    batch_samples: int = 256,
    min_samples: int = 512,
    max_samples: int = 16384,
    relative_standard_error: float = 0.03,
    relative_floor_fraction: float = 0.005,
    absolute_floor: float = 1e-5,
) -> EvaluatedReferenceBatch:
    """按 query group 自适应采样，并用并行 Welford 合并每个 GPU batch。"""

    query_group_count = len(materials)
    views = np.asarray(view_directions, dtype=np.float32)
    seeds = np.asarray(query_group_seeds, dtype=np.uint32)
    lights = None if light_directions is None else np.asarray(light_directions, dtype=np.float32)
    if views.shape not in {(query_group_count, 3), (query_group_count, 4)} or seeds.shape != (query_group_count,):
        raise ValueError("materials, view_directions and query_group_seeds must have matching counts")
    if batch_samples < 1 or min_samples < 1 or max_samples < min_samples:
        raise ValueError("adaptive sample limits are invalid")
    if min_samples % batch_samples or max_samples % batch_samples:
        raise ValueError("min_samples and max_samples must be multiples of batch_samples")
    if not 0.0 < relative_standard_error < 1.0:
        raise ValueError("relative_standard_error must lie in (0, 1)")
    shape = (query_group_count, evaluator.light_count, 3)
    means_a = np.zeros(shape, dtype=np.float64)
    means_b = np.zeros(shape, dtype=np.float64)
    m2_a = np.zeros(shape, dtype=np.float64)
    m2_b = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(query_group_count, dtype=np.uint64)
    active = np.arange(query_group_count, dtype=np.int64)
    while len(active):
        offsets = np.unique(counts[active])
        if len(offsets) != 1:
            raise AssertionError("active adaptive query groups must share the same prefix length")
        batch = evaluator.evaluate_query_groups(
            [materials[index] for index in active],
            views[active],
            sample_count_per_replica=batch_samples,
            query_group_seeds=seeds[active],
            light_directions=None if lights is None else lights[active],
            sample_offset=int(offsets[0]),
        )
        for name, values in zip(("mean_a", "second_a", "mean_b", "second_b"), batch, strict=True):
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"reference produced non-finite {name}")
        batch_mean_a, second_a, batch_mean_b, second_b = (np.asarray(item, dtype=np.float64) for item in batch)
        if np.any(batch_mean_a < 0.0) or np.any(batch_mean_b < 0.0):
            raise RuntimeError("reference produced a negative response for an unsupported material state")
        variance_a = np.maximum(second_a - batch_mean_a * batch_mean_a, 0.0)
        variance_b = np.maximum(second_b - batch_mean_b * batch_mean_b, 0.0)
        _merge_batch(means_a, m2_a, counts, active, batch_mean_a, variance_a, batch_samples)
        _merge_batch(means_b, m2_b, counts, active, batch_mean_b, variance_b, batch_samples)
        counts[active] += np.uint64(batch_samples)

        next_active: list[int] = []
        for query_group_index in active:
            count = int(counts[query_group_index])
            if count >= max_samples:
                continue
            if count < min_samples:
                next_active.append(int(query_group_index))
                continue
            replica_means = (means_a[query_group_index], means_b[query_group_index])
            replica_m2 = (m2_a[query_group_index], m2_b[query_group_index])
            peak = max(float(np.max(np.abs(item))) for item in replica_means)
            denominator_floor = max(absolute_floor, relative_floor_fraction * peak)
            scores = []
            for mean, m2 in zip(replica_means, replica_m2, strict=True):
                variance = np.maximum(m2 / count, 0.0)
                standard_error = np.sqrt(variance / count)
                relative_error = standard_error / np.maximum(np.abs(mean), denominator_floor)
                scores.append(float(np.quantile(relative_error, 0.95)))
            if max(scores) > relative_standard_error:
                next_active.append(int(query_group_index))
        active = np.asarray(next_active, dtype=np.int64)

    count_view = counts[:, None, None].astype(np.float64)
    replica_a = ReplicaMoments(means_a, m2_a / count_view, counts)
    replica_b = ReplicaMoments(means_b, m2_b / count_view, counts)
    combined = combine_replica_moments(replica_a, replica_b)
    return EvaluatedReferenceBatch(
        combined.mean,
        combined.variance,
        means_a,
        means_b,
        combined.sample_count,
    )


def evaluate_reference_fixed(
    evaluator: FalcorReferenceEvaluator,
    materials: Sequence[LayerStackIR],
    view_directions: np.ndarray,
    *,
    query_group_seeds: np.ndarray,
    light_directions: np.ndarray | None = None,
    samples_per_replica: int,
) -> EvaluatedReferenceBatch:
    batch = evaluator.evaluate_query_groups(
        materials,
        view_directions,
        sample_count_per_replica=samples_per_replica,
        query_group_seeds=query_group_seeds,
        light_directions=light_directions,
    )
    mean_a, second_a, mean_b, second_b = (np.asarray(item, dtype=np.float64) for item in batch)
    for name, values in zip(("mean_a", "second_a", "mean_b", "second_b"), batch, strict=True):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"reference produced non-finite {name}")
    if np.any(mean_a < 0.0) or np.any(mean_b < 0.0):
        raise RuntimeError("reference produced a negative response for an unsupported material state")
    counts = np.full(len(materials), samples_per_replica, dtype=np.uint64)
    replica_a = ReplicaMoments(mean_a, np.maximum(second_a - mean_a * mean_a, 0.0), counts)
    replica_b = ReplicaMoments(mean_b, np.maximum(second_b - mean_b * mean_b, 0.0), counts)
    combined = combine_replica_moments(replica_a, replica_b)
    return EvaluatedReferenceBatch(
        combined.mean,
        combined.variance,
        mean_a,
        mean_b,
        combined.sample_count,
    )


def evaluate_reference_batched_fixed(
    evaluator: FalcorReferenceEvaluator,
    materials: Sequence[LayerStackIR],
    view_directions: np.ndarray,
    *,
    query_group_seeds: np.ndarray,
    light_directions: np.ndarray | None = None,
    samples_per_replica: int,
    batch_samples_per_replica: int = 256,
) -> EvaluatedReferenceBatch:
    """以连续 sample offset 分批执行，并在 CPU float64 合并固定预算 moments。"""

    query_group_count = len(materials)
    views = np.asarray(view_directions, dtype=np.float32)
    seeds = np.asarray(query_group_seeds, dtype=np.uint32)
    lights = None if light_directions is None else np.asarray(light_directions, dtype=np.float32)
    if views.shape not in {(query_group_count, 3), (query_group_count, 4)} or seeds.shape != (query_group_count,):
        raise ValueError("materials, view_directions and query_group_seeds must have matching counts")
    if samples_per_replica < 1 or batch_samples_per_replica < 1:
        raise ValueError("batched fixed reference sample counts must be positive")
    shape = (query_group_count, evaluator.light_count, 3)
    means_a = np.zeros(shape, dtype=np.float64)
    means_b = np.zeros(shape, dtype=np.float64)
    m2_a = np.zeros(shape, dtype=np.float64)
    m2_b = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(query_group_count, dtype=np.uint64)
    active = np.arange(query_group_count, dtype=np.int64)
    sample_offset = 0
    while sample_offset < samples_per_replica:
        batch_count = min(batch_samples_per_replica, samples_per_replica - sample_offset)
        batch = evaluator.evaluate_query_groups(
            materials,
            views,
            sample_count_per_replica=batch_count,
            query_group_seeds=seeds,
            light_directions=lights,
            sample_offset=sample_offset,
        )
        for name, values in zip(("mean_a", "second_a", "mean_b", "second_b"), batch, strict=True):
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"reference produced non-finite {name}")
        batch_mean_a, second_a, batch_mean_b, second_b = (
            np.asarray(item, dtype=np.float64) for item in batch
        )
        if np.any(batch_mean_a < 0.0) or np.any(batch_mean_b < 0.0):
            raise RuntimeError("reference produced a negative response for an unsupported material state")
        variance_a = np.maximum(second_a - batch_mean_a * batch_mean_a, 0.0)
        variance_b = np.maximum(second_b - batch_mean_b * batch_mean_b, 0.0)
        _merge_batch(means_a, m2_a, counts, active, batch_mean_a, variance_a, batch_count)
        _merge_batch(means_b, m2_b, counts, active, batch_mean_b, variance_b, batch_count)
        counts += np.uint64(batch_count)
        sample_offset += batch_count
    if np.any(counts != samples_per_replica):
        raise AssertionError("batched fixed reference sample offsets are not contiguous")
    count_view = counts[:, None, None].astype(np.float64)
    replica_a = ReplicaMoments(means_a, m2_a / count_view, counts)
    replica_b = ReplicaMoments(means_b, m2_b / count_view, counts)
    combined = combine_replica_moments(replica_a, replica_b)
    return EvaluatedReferenceBatch(
        combined.mean,
        combined.variance,
        means_a,
        means_b,
        combined.sample_count,
    )
