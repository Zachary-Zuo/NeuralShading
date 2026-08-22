import json

import numpy as np
import pytest


falcor = pytest.importorskip("falcor")

from datagen.directions import equal_area_hemisphere
from datagen.gen_tiles import (
    FalcorTileEvaluator,
    evaluate_adaptive_tile,
    evaluate_adaptive_tiles,
    generate_pilot,
)
from datagen.gen_v0 import generate_v0
from datagen.priors import sample_stacks
from datagen.two_layer_slice import gray_diffuse_stack


pytestmark = pytest.mark.falcor


def test_adaptive_tile_has_consistent_independent_halves() -> None:
    directions, _ = equal_area_hemisphere(32)
    evaluator = FalcorTileEvaluator(directions, max_depth=32)
    result = evaluate_adaptive_tile(
        evaluator,
        gray_diffuse_stack(),
        np.array([0.3, 0.0, np.sqrt(1.0 - 0.3**2), 0.0], dtype=np.float32),
        batch_samples=64,
        min_samples=64,
        max_samples=256,
        seed=313,
    )
    assert result.count in {64, 128, 192, 256}
    assert np.all(np.isfinite(result.mean_a))
    assert np.all(np.isfinite(result.mean_b))
    standard_error = np.sqrt((result.variance_a + result.variance_b) / result.count)
    assert np.all(np.abs(result.mean_a - result.mean_b) <= 6.0 * standard_error + 5e-3)


def test_pilot_shard_is_memmap_readable_and_uses_14_bytes_per_bin(tmp_path) -> None:
    output = tmp_path / "pilot"
    generate_pilot(output, stack_count=1, view_count=1, bin_count=16, seed=317, max_samples=64)
    tiles = np.load(output / "tiles.npy", mmap_mode="r")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))

    assert tiles.shape == (1,)
    assert tiles.dtype.itemsize == 14 * 16
    assert tiles.dtype.names == ("mean_a", "mean_b", "count")
    assert np.all(tiles["count"] == 64)
    assert np.all(np.isfinite(tiles["mean_a"]))
    assert metadata["tile_count"] == 1
    assert metadata["solid_angle_weight_sum"] == pytest.approx(2.0 * np.pi, rel=1e-6)


def test_multi_tile_batch_matches_individual_queries() -> None:
    directions, _ = equal_area_hemisphere(16)
    evaluator = FalcorTileEvaluator(directions, max_depth=32, max_tile_batch=2)
    stacks = sample_stacks(2, seed=401)
    views = np.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.4, 0.0, np.sqrt(0.84), 0.0]],
        dtype=np.float32,
    )
    seeds = np.asarray([409, 419], dtype=np.uint32)
    batched = evaluator.evaluate_tiles(stacks, views, sample_count_per_half=32, tile_seeds=seeds)
    for tile_index in range(2):
        individual = evaluator.evaluate_tiles(
            [stacks[tile_index]],
            views[tile_index : tile_index + 1],
            sample_count_per_half=32,
            tile_seeds=seeds[tile_index : tile_index + 1],
        )
        for batched_array, individual_array in zip(batched, individual, strict=True):
            np.testing.assert_array_equal(batched_array[tile_index], individual_array[0])


def test_batched_v0_writer_keeps_family_splits_and_shards_memmappable(tmp_path) -> None:
    output = tmp_path / "v0"
    generate_v0(
        output,
        family_count=3,
        local_state_count=2,
        view_count=2,
        bin_count=16,
        samples_per_half=16,
        tile_batch=4,
        shard_tiles=5,
        seed=431,
        max_depth=32,
    )
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    states = np.load(output / "states.npy")
    family_splits = np.load(output / "family_splits.npy")
    assert metadata["tile_count"] == 12
    assert metadata["bytes_per_bin"] == 14
    assert len(metadata["shards"]) == 3
    assert set(family_splits.tolist()) == {0, 1, 2}
    assert np.all(states["split"] == family_splits[states["family_index"]])
    total_tiles = 0
    for shard in metadata["shards"]:
        tiles = np.load(output / shard["tiles"], mmap_mode="r")
        index = np.load(output / shard["index"], mmap_mode="r")
        assert tiles.dtype.itemsize == 14 * 16
        assert np.all(tiles["count"] == 16)
        assert len(tiles) == len(index) == shard["tile_count"]
        total_tiles += len(tiles)
    assert total_tiles == 12

    first_tile_path = output / metadata["shards"][0]["tiles"]
    first_tile_timestamp = first_tile_path.stat().st_mtime_ns
    generate_v0(
        output,
        family_count=3,
        local_state_count=2,
        view_count=2,
        bin_count=16,
        samples_per_half=16,
        tile_batch=4,
        shard_tiles=5,
        seed=431,
        max_depth=32,
        resume=True,
    )
    assert first_tile_path.stat().st_mtime_ns == first_tile_timestamp


def test_batched_adaptive_sampling_stops_deterministic_tiles_at_minimum() -> None:
    directions, _ = equal_area_hemisphere(16)
    evaluator = FalcorTileEvaluator(directions, max_depth=16, max_tile_batch=2)
    diffuse = gray_diffuse_stack().layers[-1]
    single_layer_stacks = [
        type(gray_diffuse_stack())((diffuse,), ()),
        type(gray_diffuse_stack())((diffuse,), ()),
    ]
    views = np.asarray([[0.0, 0.0, 1.0, 0.0], [0.2, 0.0, np.sqrt(0.96), 0.0]], dtype=np.float32)
    result = evaluate_adaptive_tiles(
        evaluator,
        single_layer_stacks,
        views,
        tile_seeds=np.asarray([461, 463], dtype=np.uint32),
        batch_samples=16,
        min_samples=32,
        max_samples=64,
        relative_standard_error=0.01,
    )
    np.testing.assert_array_equal(result.counts, np.asarray([32, 32], dtype=np.uint32))
    np.testing.assert_allclose(result.variance_a, 0.0, atol=1e-8)


def test_direct_top_single_layer_matches_full_teacher() -> None:
    directions, _ = equal_area_hemisphere(16)
    evaluator = FalcorTileEvaluator(directions, max_depth=16, max_tile_batch=1)
    diffuse = gray_diffuse_stack().layers[-1]
    stack = type(gray_diffuse_stack())((diffuse,), ())
    view = np.asarray([[0.2, 0.0, np.sqrt(0.96), 0.0]], dtype=np.float32)
    direct = evaluator.evaluate_direct_tiles([stack], view)
    full = evaluator.evaluate_tiles(
        [stack],
        view,
        sample_count_per_half=8,
        tile_seeds=np.asarray([487], dtype=np.uint32),
    )[0]
    np.testing.assert_allclose(direct, full, rtol=1e-6, atol=1e-7)
