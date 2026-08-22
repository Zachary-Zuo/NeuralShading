from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from datagen.directions import equal_area_hemisphere, stratified_view_directions
from datagen.gen_tiles import FalcorTileEvaluator, evaluate_adaptive_tiles
from datagen.priors import PRIOR_VERSION, assign_family_splits, sample_stack_families
from schema import BINARY_SIZE, SCHEMA_VERSION, LayerStack, pack_stack


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEACHER_SOURCES = (
    PROJECT_ROOT / "teacher" / "sampling.slang",
    PROJECT_ROOT / "teacher" / "interfaces.slang",
    PROJECT_ROOT / "teacher" / "layered_walk.slang",
)
SPLIT_NAMES = ("train", "validation", "test")


def _teacher_hash() -> str:
    digest = hashlib.sha256()
    for path in TEACHER_SOURCES:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _tile_seeds(seed: int, tile_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(tile_indices, dtype=np.uint64)
    mixed = (values * np.uint64(0x9E3779B1) + np.uint64(seed & 0xFFFFFFFF)) & np.uint64(0xFFFFFFFF)
    return mixed.astype(np.uint32)


def _write_stacks(
    output_dir: Path,
    stacks: list[LayerStack],
    family_indices: np.ndarray,
    local_indices: np.ndarray,
    family_splits: np.ndarray,
) -> str:
    payload_digest = hashlib.sha256()
    stack_hashes = np.empty(len(stacks), dtype="S32")
    stack_path = output_dir / "stacks.bin"
    with stack_path.open("wb") as stream:
        for index, stack in enumerate(stacks):
            payload = pack_stack(stack)
            stream.write(payload)
            payload_digest.update(payload)
            stack_hashes[index] = hashlib.sha256(payload).digest()
    np.save(output_dir / "stack_sha256.npy", stack_hashes)

    state_dtype = np.dtype(
        [("family_index", "<u4"), ("local_state", "<u2"), ("split", "u1"), ("reserved", "u1")]
    )
    states = np.zeros(len(stacks), dtype=state_dtype)
    states["family_index"] = family_indices
    states["local_state"] = local_indices
    states["split"] = family_splits[family_indices]
    np.save(output_dir / "states.npy", states)
    np.save(output_dir / "family_splits.npy", family_splits)
    return payload_digest.hexdigest()


def generate_v0(
    output_dir: Path,
    *,
    family_count: int,
    local_state_count: int,
    view_count: int,
    bin_count: int,
    samples_per_half: int,
    tile_batch: int,
    shard_tiles: int,
    seed: int,
    max_depth: int = 64,
    adaptive: bool = False,
    batch_samples: int = 256,
    min_samples: int = 512,
    max_samples: int = 16384,
    relative_standard_error: float = 0.03,
    resume: bool = False,
) -> None:
    if min(family_count, local_state_count, view_count, bin_count, samples_per_half, tile_batch, shard_tiles) < 1:
        raise ValueError("all dataset dimensions and batch sizes must be positive")
    if samples_per_half > 65535:
        raise ValueError("samples_per_half must fit uint16")

    output_dir.mkdir(parents=True, exist_ok=True)
    families = sample_stack_families(family_count, local_state_count, seed)
    stacks = [stack for family in families for stack in family]
    state_count = len(stacks)
    family_indices = np.repeat(np.arange(family_count, dtype=np.uint32), local_state_count)
    local_indices = np.tile(np.arange(local_state_count, dtype=np.uint16), family_count)
    family_splits = assign_family_splits(family_count, seed)
    stack_payload_hash = _write_stacks(
        output_dir, stacks, family_indices, local_indices, family_splits
    )

    views = stratified_view_directions(view_count)
    light_directions, solid_angle_weights = equal_area_hemisphere(bin_count)
    np.save(output_dir / "views.npy", views)
    np.save(output_dir / "light_directions.npy", light_directions)
    np.save(output_dir / "solid_angle_weights.npy", solid_angle_weights)

    tile_dtype = np.dtype(
        [
            ("mean_a", "<f2", (bin_count, 3)),
            ("mean_b", "<f2", (bin_count, 3)),
            ("count", "<u2", (bin_count,)),
        ],
        align=False,
    )
    tile_count = state_count * view_count
    evaluator = FalcorTileEvaluator(
        light_directions,
        max_depth=max_depth,
        max_tile_batch=tile_batch,
    )
    shard_records: list[dict[str, int | str]] = []
    all_sample_counts: list[int] = []
    for shard_index, shard_start in enumerate(range(0, tile_count, shard_tiles)):
        shard_end = min(shard_start + shard_tiles, tile_count)
        shard_count = shard_end - shard_start
        tile_name = f"tiles-{shard_index:05d}.npy"
        index_name = f"index-{shard_index:05d}.npy"
        tile_path = output_dir / tile_name
        index_path = output_dir / index_name
        if resume and tile_path.exists() and index_path.exists():
            completed_tiles = np.load(tile_path, mmap_mode="r")
            completed_index = np.load(index_path, mmap_mode="r")
            if len(completed_tiles) != shard_count or completed_index.shape != (shard_count, 2):
                raise ValueError(f"completed shard {shard_index} has an unexpected shape")
            completed_counts = np.asarray(completed_tiles["count"][:, 0], dtype=np.uint32)
            if np.any(completed_counts == 0):
                raise ValueError(f"completed shard {shard_index} contains zero sample counts")
            if not adaptive and np.any(completed_counts != samples_per_half):
                raise ValueError(f"completed shard {shard_index} has the wrong fixed sample count")
            all_sample_counts.extend(int(value) for value in completed_counts)
            shard_records.append(
                {
                    "tiles": tile_name,
                    "index": index_name,
                    "tile_start": shard_start,
                    "tile_count": shard_count,
                }
            )
            print(f"resumed shard {shard_index + 1}: tiles [{shard_start}, {shard_end})")
            continue
        tiles = np.lib.format.open_memmap(
            tile_path,
            mode="w+",
            dtype=tile_dtype,
            shape=(shard_count,),
        )
        index = np.empty((shard_count, 2), dtype=np.uint32)

        for batch_start in range(shard_start, shard_end, tile_batch):
            batch_end = min(batch_start + tile_batch, shard_end)
            global_indices = np.arange(batch_start, batch_end, dtype=np.uint64)
            state_indices = (global_indices // view_count).astype(np.int64)
            view_indices = (global_indices % view_count).astype(np.int64)
            batch_stacks = [stacks[index] for index in state_indices]
            batch_views = views[view_indices]
            batch_seeds = _tile_seeds(seed, global_indices)
            if adaptive:
                moments = evaluate_adaptive_tiles(
                    evaluator,
                    batch_stacks,
                    batch_views,
                    tile_seeds=batch_seeds,
                    batch_samples=batch_samples,
                    min_samples=min_samples,
                    max_samples=max_samples,
                    relative_standard_error=relative_standard_error,
                )
                mean_a = moments.mean_a
                mean_b = moments.mean_b
                sample_counts = moments.counts
            else:
                mean_a, _, mean_b, _ = evaluator.evaluate_tiles(
                    batch_stacks,
                    batch_views,
                    sample_count_per_half=samples_per_half,
                    tile_seeds=batch_seeds,
                )
                sample_counts = np.full(len(global_indices), samples_per_half, dtype=np.uint32)
            local_start = batch_start - shard_start
            local_end = batch_end - shard_start
            tiles["mean_a"][local_start:local_end] = mean_a.astype(np.float16)
            tiles["mean_b"][local_start:local_end] = mean_b.astype(np.float16)
            tiles["count"][local_start:local_end] = sample_counts[:, None]
            all_sample_counts.extend(int(value) for value in sample_counts)
            index[local_start:local_end, 0] = state_indices
            index[local_start:local_end, 1] = view_indices
        tiles.flush()
        np.save(index_path, index)
        shard_records.append(
            {
                "tiles": tile_name,
                "index": index_name,
                "tile_start": shard_start,
                "tile_count": shard_count,
            }
        )
        print(f"wrote shard {shard_index + 1}: tiles [{shard_start}, {shard_end})")

    split_family_counts = {
        name: int(np.count_nonzero(family_splits == split_index))
        for split_index, name in enumerate(SPLIT_NAMES)
    }
    metadata = {
        "format": "ncls-direction-tiles",
        "format_version": 1,
        "schema_version": SCHEMA_VERSION,
        "prior_version": PRIOR_VERSION,
        "teacher_source_sha256": _teacher_hash(),
        "stack_payload_sha256": stack_payload_hash,
        "seed": seed,
        "family_count": family_count,
        "local_state_count": local_state_count,
        "state_count": state_count,
        "view_count": view_count,
        "bin_count": bin_count,
        "tile_count": tile_count,
        "sampling_mode": "adaptive" if adaptive else "fixed",
        "samples_per_half": samples_per_half if not adaptive else None,
        "sample_count_min": min(all_sample_counts),
        "sample_count_mean": float(np.mean(all_sample_counts)),
        "sample_count_max": max(all_sample_counts),
        "adaptive_settings": {
            "batch_samples": batch_samples,
            "min_samples": min_samples,
            "max_samples": max_samples,
            "relative_standard_error": relative_standard_error,
        }
        if adaptive
        else None,
        "max_depth": max_depth,
        "stack_stride": BINARY_SIZE,
        "tile_dtype": tile_dtype.descr,
        "bytes_per_bin": 14,
        "direction_parameterization": "equal-area-fibonacci-hemisphere",
        "solid_angle_weight_sum": float(np.sum(solid_angle_weights)),
        "split_unit": "family",
        "split_names": SPLIT_NAMES,
        "split_family_counts": split_family_counts,
        "shards": shard_records,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"generated {tile_count} tiles in {len(shard_records)} shards at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a batched v0 direction-response dataset.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "pilot_v0_batched")
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--local-states", type=int, default=4)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--samples-per-half", type=int, default=64)
    parser.add_argument("--tile-batch", type=int, default=64)
    parser.add_argument("--shard-tiles", type=int, default=65536)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--batch-samples", type=int, default=256)
    parser.add_argument("--min-samples", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=16384)
    parser.add_argument("--relative-standard-error", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    generate_v0(
        args.output,
        family_count=args.families,
        local_state_count=args.local_states,
        view_count=args.views,
        bin_count=args.bins,
        samples_per_half=args.samples_per_half,
        tile_batch=args.tile_batch,
        shard_tiles=args.shard_tiles,
        max_depth=args.max_depth,
        adaptive=args.adaptive,
        batch_samples=args.batch_samples,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        relative_standard_error=args.relative_standard_error,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
