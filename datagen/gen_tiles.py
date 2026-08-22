from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import falcor
import numpy as np

from datagen.directions import equal_area_hemisphere, stratified_view_directions
from datagen.priors import PRIOR_VERSION, sample_stacks
from schema import BINARY_SIZE, SCHEMA_VERSION, LayerStack, pack_stack


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHADER_FILE = PROJECT_ROOT / "datagen" / "kernels" / "tile_kernel.slang"


@dataclass(frozen=True)
class HalfMoments:
    mean_a: np.ndarray
    mean_b: np.ndarray
    variance_a: np.ndarray
    variance_b: np.ndarray
    count: int


@dataclass(frozen=True)
class BatchHalfMoments:
    mean_a: np.ndarray
    mean_b: np.ndarray
    variance_a: np.ndarray
    variance_b: np.ndarray
    counts: np.ndarray


class FalcorTileEvaluator:
    def __init__(
        self,
        light_directions: np.ndarray,
        *,
        max_depth: int = 64,
        max_tile_batch: int = 1,
        light_index_offset: int = 0,
    ) -> None:
        self.light_directions = np.asarray(light_directions, dtype=np.float32)
        self.light_count = len(self.light_directions)
        if max_tile_batch < 1:
            raise ValueError("max_tile_batch must be positive")
        self.max_tile_batch = max_tile_batch
        self.query_capacity = self.light_count * self.max_tile_batch
        self.max_depth = max_depth
        self.device = falcor.Device(type=falcor.DeviceType.D3D12)
        self.stack_buffer = self._buffer(BINARY_SIZE, self.max_tile_batch)
        self.view_buffer = self._buffer(16, self.max_tile_batch)
        self.seed_buffer = self._buffer(4, self.max_tile_batch)
        self.light_buffer = self._buffer(16, self.light_count)
        self.outputs = [self._buffer(16, self.query_capacity, writable=True) for _ in range(4)]
        self.light_buffer.from_numpy(self.light_directions)
        self.compute = falcor.ComputePass(self.device, file=SHADER_FILE, cs_entry="generateTileBatch")
        self.compute.globals.gStacks = self.stack_buffer
        self.compute.globals.gViewDirections = self.view_buffer
        self.compute.globals.gLightDirections = self.light_buffer
        self.compute.globals.gTileSeeds = self.seed_buffer
        self.compute.globals.gMeanA = self.outputs[0]
        self.compute.globals.gMeanSquareA = self.outputs[1]
        self.compute.globals.gMeanB = self.outputs[2]
        self.compute.globals.gMeanSquareB = self.outputs[3]
        self.compute.globals.gLightCount = self.light_count
        self.compute.globals.gLightIndexOffset = light_index_offset
        self.compute.globals.gMaxDepth = self.max_depth
        self.direct_compute = falcor.ComputePass(
            self.device, file=SHADER_FILE, cs_entry="generateDirectTopBatch"
        )
        self.direct_compute.globals.gStacks = self.stack_buffer
        self.direct_compute.globals.gViewDirections = self.view_buffer
        self.direct_compute.globals.gLightDirections = self.light_buffer
        self.direct_compute.globals.gMeanA = self.outputs[0]
        self.direct_compute.globals.gLightCount = self.light_count
        self.direct_compute.globals.gLightIndexOffset = light_index_offset

    def _buffer(self, stride: int, element_count: int, *, writable: bool = False):
        flags = falcor.ResourceBindFlags.ShaderResource
        if writable:
            flags |= falcor.ResourceBindFlags.UnorderedAccess
        return self.device.create_structured_buffer(
            struct_size=stride,
            element_count=element_count,
            bind_flags=flags,
        )

    def evaluate_batch(
        self,
        stack: LayerStack,
        view_direction: np.ndarray,
        *,
        sample_count_per_half: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arrays = self.evaluate_tiles(
            [stack],
            np.asarray(view_direction, dtype=np.float32).reshape(1, 4),
            sample_count_per_half=sample_count_per_half,
            tile_seeds=np.asarray([seed], dtype=np.uint32),
        )
        return tuple(array[0] for array in arrays)  # type: ignore[return-value]

    def evaluate_tiles(
        self,
        stacks: list[LayerStack],
        view_directions: np.ndarray,
        *,
        sample_count_per_half: int,
        tile_seeds: np.ndarray,
        sample_offset: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tile_count = len(stacks)
        if not 1 <= tile_count <= self.max_tile_batch:
            raise ValueError(f"tile batch must contain 1..{self.max_tile_batch} tiles")
        views = np.asarray(view_directions, dtype=np.float32)
        seeds = np.asarray(tile_seeds, dtype=np.uint32)
        if views.shape != (tile_count, 4):
            raise ValueError("view_directions must have shape [tile_count, 4]")
        if seeds.shape != (tile_count,):
            raise ValueError("tile_seeds must have shape [tile_count]")

        packed = np.zeros(BINARY_SIZE * self.max_tile_batch, dtype=np.uint8)
        packed[: BINARY_SIZE * tile_count] = np.frombuffer(
            b"".join(pack_stack(stack) for stack in stacks), dtype=np.uint8
        )
        padded_views = np.zeros((self.max_tile_batch, 4), dtype=np.float32)
        padded_views[:tile_count] = views
        padded_seeds = np.zeros(self.max_tile_batch, dtype=np.uint32)
        padded_seeds[:tile_count] = seeds
        self.stack_buffer.from_numpy(packed)
        self.view_buffer.from_numpy(padded_views)
        self.seed_buffer.from_numpy(padded_seeds)
        query_count = tile_count * self.light_count
        self.compute.globals.gQueryCount = query_count
        self.compute.globals.gSampleCountPerHalf = sample_count_per_half
        self.compute.globals.gSampleOffset = sample_offset
        self.compute.globals.gSeed = 0
        self.compute.execute(threads_x=query_count)
        arrays = [
            output.to_numpy().view(np.float32).reshape(self.query_capacity, 4)[:query_count, :3]
            .reshape(tile_count, self.light_count, 3)
            .copy()
            for output in self.outputs
        ]
        return arrays[0], arrays[1], arrays[2], arrays[3]

    def evaluate_direct_tiles(
        self,
        stacks: list[LayerStack],
        view_directions: np.ndarray,
    ) -> np.ndarray:
        tile_count = len(stacks)
        if not 1 <= tile_count <= self.max_tile_batch:
            raise ValueError(f"tile batch must contain 1..{self.max_tile_batch} tiles")
        views = np.asarray(view_directions, dtype=np.float32)
        if views.shape != (tile_count, 4):
            raise ValueError("view_directions must have shape [tile_count, 4]")
        packed = np.zeros(BINARY_SIZE * self.max_tile_batch, dtype=np.uint8)
        packed[: BINARY_SIZE * tile_count] = np.frombuffer(
            b"".join(pack_stack(stack) for stack in stacks), dtype=np.uint8
        )
        padded_views = np.zeros((self.max_tile_batch, 4), dtype=np.float32)
        padded_views[:tile_count] = views
        self.stack_buffer.from_numpy(packed)
        self.view_buffer.from_numpy(padded_views)
        query_count = tile_count * self.light_count
        self.direct_compute.globals.gQueryCount = query_count
        self.direct_compute.execute(threads_x=query_count)
        return (
            self.outputs[0]
            .to_numpy()
            .view(np.float32)
            .reshape(self.query_capacity, 4)[:query_count, :3]
            .reshape(tile_count, self.light_count, 3)
            .copy()
        )


def evaluate_adaptive_tile(
    evaluator: FalcorTileEvaluator,
    stack: LayerStack,
    view_direction: np.ndarray,
    *,
    batch_samples: int = 64,
    min_samples: int = 64,
    max_samples: int = 2048,
    relative_standard_error: float = 0.05,
    absolute_floor: float = 1e-3,
    seed: int = 1,
) -> HalfMoments:
    if not 1 <= min_samples <= max_samples <= 65535:
        raise ValueError("sample limits must satisfy 1 <= min_samples <= max_samples <= 65535")
    if batch_samples < 1:
        raise ValueError("batch_samples must be positive")
    shape = (evaluator.light_count, 3)
    sum_a = np.zeros(shape, dtype=np.float64)
    sum_b = np.zeros(shape, dtype=np.float64)
    sum_square_a = np.zeros(shape, dtype=np.float64)
    sum_square_b = np.zeros(shape, dtype=np.float64)
    count = 0
    batch_index = 0

    while count < max_samples:
        current_batch = min(batch_samples, max_samples - count)
        mean_a, second_a, mean_b, second_b = evaluator.evaluate_batch(
            stack,
            view_direction,
            sample_count_per_half=current_batch,
            seed=seed + batch_index * 1009,
        )
        sum_a += mean_a * current_batch
        sum_b += mean_b * current_batch
        sum_square_a += second_a * current_batch
        sum_square_b += second_b * current_batch
        count += current_batch
        batch_index += 1

        if count < min_samples:
            continue
        combined_means = [sum_a / count, sum_b / count]
        combined_seconds = [sum_square_a / count, sum_square_b / count]
        relative_errors = []
        for mean, second in zip(combined_means, combined_seconds, strict=True):
            variance = np.maximum(second - mean * mean, 0.0)
            standard_error = np.sqrt(variance / count)
            relative_errors.append(standard_error / np.maximum(np.abs(mean), absolute_floor))
        if max(float(np.quantile(error, 0.95)) for error in relative_errors) <= relative_standard_error:
            break

    mean_a = sum_a / count
    mean_b = sum_b / count
    variance_a = np.maximum(sum_square_a / count - mean_a * mean_a, 0.0)
    variance_b = np.maximum(sum_square_b / count - mean_b * mean_b, 0.0)
    if not np.all(np.isfinite(mean_a)) or not np.all(np.isfinite(mean_b)):
        raise RuntimeError("teacher produced non-finite tile data")
    return HalfMoments(mean_a, mean_b, variance_a, variance_b, count)


def evaluate_adaptive_tiles(
    evaluator: FalcorTileEvaluator,
    stacks: list[LayerStack],
    view_directions: np.ndarray,
    *,
    tile_seeds: np.ndarray,
    batch_samples: int = 256,
    min_samples: int = 512,
    max_samples: int = 16384,
    relative_standard_error: float = 0.03,
    relative_floor_fraction: float = 0.005,
    absolute_floor: float = 1e-5,
) -> BatchHalfMoments:
    """Adapt a tile batch independently while retaining batched GPU dispatches."""
    tile_count = len(stacks)
    views = np.asarray(view_directions, dtype=np.float32)
    seeds = np.asarray(tile_seeds, dtype=np.uint32)
    if views.shape != (tile_count, 4) or seeds.shape != (tile_count,):
        raise ValueError("stacks, view_directions and tile_seeds must have matching tile counts")
    if not 1 <= min_samples <= max_samples <= 65535:
        raise ValueError("sample limits must satisfy 1 <= min_samples <= max_samples <= 65535")
    if batch_samples < 1 or min_samples % batch_samples or max_samples % batch_samples:
        raise ValueError("min_samples and max_samples must be multiples of batch_samples")
    if not 0.0 < relative_standard_error < 1.0:
        raise ValueError("relative_standard_error must lie in (0, 1)")

    shape = (tile_count, evaluator.light_count, 3)
    sums = [np.zeros(shape, dtype=np.float64) for _ in range(4)]
    counts = np.zeros(tile_count, dtype=np.uint32)
    active = np.arange(tile_count, dtype=np.int64)
    while len(active):
        active_seeds = (
            seeds[active].astype(np.uint64)
            + counts[active].astype(np.uint64) * np.uint64(0x9E3779B1)
        ).astype(np.uint32)
        batch_values = evaluator.evaluate_tiles(
            [stacks[index] for index in active],
            views[active],
            sample_count_per_half=batch_samples,
            tile_seeds=active_seeds,
        )
        for statistic_name, values in zip(
            ("mean_a", "second_a", "mean_b", "second_b"), batch_values, strict=True
        ):
            if not np.all(np.isfinite(values)):
                bad_local = np.unique(np.argwhere(~np.isfinite(values))[:, 0])
                bad_tiles = active[bad_local]
                raise RuntimeError(
                    f"teacher produced non-finite {statistic_name} for batch tile indices "
                    f"{bad_tiles.tolist()} at counts {counts[bad_tiles].tolist()}"
                )
        for accumulator, values in zip(sums, batch_values, strict=True):
            accumulator[active] += values * batch_samples
        counts[active] += batch_samples

        next_active: list[int] = []
        for tile_index in active:
            count = int(counts[tile_index])
            if count >= max_samples:
                continue
            if count < min_samples:
                next_active.append(int(tile_index))
                continue
            means = [sums[0][tile_index] / count, sums[2][tile_index] / count]
            seconds = [sums[1][tile_index] / count, sums[3][tile_index] / count]
            combined_peak = max(float(np.max(np.abs(means[0]))), float(np.max(np.abs(means[1]))))
            denominator_floor = max(absolute_floor, relative_floor_fraction * combined_peak)
            scores = []
            for mean, second in zip(means, seconds, strict=True):
                variance = np.maximum(second - mean * mean, 0.0)
                standard_error = np.sqrt(variance / count)
                relative_error = standard_error / np.maximum(np.abs(mean), denominator_floor)
                scores.append(float(np.quantile(relative_error, 0.95)))
            if max(scores) > relative_standard_error:
                next_active.append(int(tile_index))
        active = np.asarray(next_active, dtype=np.int64)

    count_view = counts[:, None, None]
    mean_a = sums[0] / count_view
    mean_b = sums[2] / count_view
    variance_a = np.maximum(sums[1] / count_view - mean_a * mean_a, 0.0)
    variance_b = np.maximum(sums[3] / count_view - mean_b * mean_b, 0.0)
    return BatchHalfMoments(mean_a, mean_b, variance_a, variance_b, counts)


def _stack_hash(stack: LayerStack) -> str:
    return hashlib.sha256(pack_stack(stack)).hexdigest()


def generate_pilot(
    output_dir: Path,
    *,
    stack_count: int,
    view_count: int,
    bin_count: int,
    seed: int,
    max_samples: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stacks = sample_stacks(stack_count, seed)
    views = stratified_view_directions(view_count)
    light_directions, solid_angle_weights = equal_area_hemisphere(bin_count)
    evaluator = FalcorTileEvaluator(light_directions)

    tile_dtype = np.dtype(
        [
            ("mean_a", "<f2", (bin_count, 3)),
            ("mean_b", "<f2", (bin_count, 3)),
            ("count", "<u2", (bin_count,)),
        ],
        align=False,
    )
    tile_count = stack_count * view_count
    tiles = np.lib.format.open_memmap(output_dir / "tiles.npy", mode="w+", dtype=tile_dtype, shape=(tile_count,))
    index = np.empty((tile_count, 2), dtype=np.uint32)

    tile_index = 0
    counts: list[int] = []
    for stack_index, stack in enumerate(stacks):
        for view_index, view in enumerate(views):
            moments = evaluate_adaptive_tile(
                evaluator,
                stack,
                view,
                max_samples=max_samples,
                seed=seed + tile_index * 7919,
            )
            tiles[tile_index]["mean_a"] = moments.mean_a.astype(np.float16)
            tiles[tile_index]["mean_b"] = moments.mean_b.astype(np.float16)
            tiles[tile_index]["count"] = moments.count
            index[tile_index] = (stack_index, view_index)
            counts.append(moments.count)
            tile_index += 1
    tiles.flush()

    (output_dir / "stacks.bin").write_bytes(b"".join(pack_stack(stack) for stack in stacks))
    np.save(output_dir / "views.npy", views)
    np.save(output_dir / "light_directions.npy", light_directions)
    np.save(output_dir / "solid_angle_weights.npy", solid_angle_weights)
    np.save(output_dir / "index.npy", index)
    metadata = {
        "format": "ncls-direction-tiles",
        "format_version": 1,
        "schema_version": SCHEMA_VERSION,
        "prior_version": PRIOR_VERSION,
        "seed": seed,
        "stack_count": stack_count,
        "view_count": view_count,
        "bin_count": bin_count,
        "tile_count": tile_count,
        "stack_stride": BINARY_SIZE,
        "tile_dtype": tile_dtype.descr,
        "direction_parameterization": "equal-area-fibonacci-hemisphere",
        "solid_angle_weight_sum": float(np.sum(solid_angle_weights)),
        "sample_count_min": int(min(counts)),
        "sample_count_max": int(max(counts)),
        "sample_count_mean": float(np.mean(counts)),
        "stack_sha256": [_stack_hash(stack) for stack in stacks],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"generated {tile_count} tiles at {output_dir}; "
        f"samples/half min={min(counts)} mean={np.mean(counts):.1f} max={max(counts)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small v0-compatible direction-tile shard.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "pilot_v0")
    parser.add_argument("--stacks", type=int, default=4)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--max-samples", type=int, default=512)
    args = parser.parse_args()
    generate_pilot(
        args.output,
        stack_count=args.stacks,
        view_count=args.views,
        bin_count=args.bins,
        seed=args.seed,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
