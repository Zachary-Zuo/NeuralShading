from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import falcor

from datagen.gen_tiles import FalcorTileEvaluator
from datagen.gen_v0 import _tile_seeds
from schema import BINARY_SIZE, pack_stack, unpack_stack


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _debug_exact_sample(
    evaluator: FalcorTileEvaluator,
    stack,
    view: np.ndarray,
    light: np.ndarray,
    key_x: int,
    key_y: int,
    max_depth: int,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    device = evaluator.device
    flags = falcor.ResourceBindFlags.ShaderResource
    write_flags = flags | falcor.ResourceBindFlags.UnorderedAccess
    stack_buffer = device.create_structured_buffer(
        struct_size=BINARY_SIZE, element_count=1, bind_flags=flags
    )
    view_buffer = device.create_structured_buffer(struct_size=16, element_count=1, bind_flags=flags)
    light_buffer = device.create_structured_buffer(struct_size=16, element_count=1, bind_flags=flags)
    result_buffer = device.create_structured_buffer(
        struct_size=16, element_count=1, bind_flags=write_flags
    )
    code_buffer = device.create_structured_buffer(
        struct_size=4, element_count=1, bind_flags=write_flags
    )
    auxiliary0_buffer = device.create_structured_buffer(
        struct_size=16, element_count=1, bind_flags=write_flags
    )
    auxiliary1_buffer = device.create_structured_buffer(
        struct_size=16, element_count=1, bind_flags=write_flags
    )
    stack_buffer.from_numpy(np.frombuffer(pack_stack(stack), dtype=np.uint8).copy())
    view_buffer.from_numpy(view)
    light_buffer.from_numpy(light)
    compute = falcor.ComputePass(
        device,
        file=PROJECT_ROOT / "datagen" / "kernels" / "debug_teacher.cs.slang",
        cs_entry="debugTeacherSample",
    )
    compute.globals.gStacks = stack_buffer
    compute.globals.gViewDirections = view_buffer
    compute.globals.gLightDirections = light_buffer
    compute.globals.gResult = result_buffer
    compute.globals.gDebugCode = code_buffer
    compute.globals.gAuxiliary0 = auxiliary0_buffer
    compute.globals.gAuxiliary1 = auxiliary1_buffer
    compute.globals.gKeyX = key_x
    compute.globals.gKeyY = key_y
    compute.globals.gMaxDepth = max_depth
    compute.execute(threads_x=1)
    result = result_buffer.to_numpy().view(np.float32).reshape(1, 4)[0, :3].copy()
    code = int(code_buffer.to_numpy().view(np.uint32)[0])
    auxiliary0 = auxiliary0_buffer.to_numpy().view(np.float32).reshape(1, 4)[0].copy()
    auxiliary1 = auxiliary1_buffer.to_numpy().view(np.float32).reshape(1, 4)[0].copy()
    return result, code, auxiliary0, auxiliary1


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce a non-finite adaptive teacher sample.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--tile", type=int, required=True)
    parser.add_argument("--bin", type=int, required=True)
    parser.add_argument("--half", choices=("a", "b"), required=True)
    parser.add_argument("--batch-samples", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=65024)
    parser.add_argument("--expect-clean", action="store_true")
    args = parser.parse_args()

    metadata = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
    shard = next(
        shard
        for shard in metadata["shards"]
        if shard["tile_start"] <= args.tile < shard["tile_start"] + shard["tile_count"]
    )
    local_tile = args.tile - shard["tile_start"]
    index = np.load(args.dataset / shard["index"], mmap_mode="r")[local_tile]
    state_index, view_index = (int(index[0]), int(index[1]))
    stack_payload = (args.dataset / "stacks.bin").read_bytes()
    offset = state_index * BINARY_SIZE
    stack = unpack_stack(stack_payload[offset : offset + BINARY_SIZE])
    view = np.load(args.dataset / "views.npy")[view_index : view_index + 1]
    light = np.load(args.dataset / "light_directions.npy")[args.bin : args.bin + 1]
    evaluator = FalcorTileEvaluator(light, max_depth=metadata["max_depth"], light_index_offset=args.bin)
    base_seed = _tile_seeds(metadata["seed"], np.asarray([args.tile], dtype=np.uint64))[0]
    half_index = 0 if args.half == "a" else 2

    failing_count = None
    failing_seed = None
    for count in range(0, args.max_samples, args.batch_samples):
        batch_seed = np.asarray(
            [np.uint32(np.uint64(base_seed) + np.uint64(count) * np.uint64(0x9E3779B1))]
        )
        values = evaluator.evaluate_tiles(
            [stack],
            view,
            sample_count_per_half=args.batch_samples,
            tile_seeds=batch_seed,
        )[half_index]
        if not np.all(np.isfinite(values)):
            failing_count = count
            failing_seed = batch_seed
            break
    if failing_count is None or failing_seed is None:
        if args.expect_clean:
            print("no non-finite batch reproduced")
            return
        raise RuntimeError("no non-finite batch reproduced")

    for sample_offset in range(args.batch_samples):
        statistics = evaluator.evaluate_tiles(
            [stack],
            view,
            sample_count_per_half=1,
            tile_seeds=failing_seed,
            sample_offset=sample_offset,
        )
        if not all(np.all(np.isfinite(values)) for values in statistics):
            print(
                f"tile={args.tile} state={state_index} view={view_index} bin={args.bin} half={args.half} "
                f"batch_start={failing_count} sample_offset={sample_offset} seed={int(failing_seed[0])}"
            )
            for name, values in zip(("mean_a", "second_a", "mean_b", "second_b"), statistics, strict=True):
                print(name, values[0, 0].tolist())
            tile_seed = int(failing_seed[0])
            if args.half == "a":
                sample_seed = tile_seed
                multiplier = 1664525
            else:
                sample_seed = tile_seed ^ 0xA511E9B3
                multiplier = 22695477
            global_sample_index = sample_offset
            key_x = args.bin ^ sample_seed
            key_y = (global_sample_index + sample_seed * multiplier) & 0xFFFFFFFF
            exact_result, debug_code, auxiliary0, auxiliary1 = _debug_exact_sample(
                evaluator,
                stack,
                view,
                light,
                key_x,
                key_y,
                metadata["max_depth"],
            )
            print("exact_rng_key", [key_x, key_y], "debug_code", debug_code, "result", exact_result.tolist())
            print("ballistic_view", auxiliary0.tolist(), "ballistic_light", auxiliary1.tolist())
            return
    raise RuntimeError("batch is non-finite but no individual sample reproduced")


if __name__ == "__main__":
    main()
