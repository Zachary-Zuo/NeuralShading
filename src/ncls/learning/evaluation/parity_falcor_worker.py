from __future__ import annotations

import argparse
import json
from pathlib import Path

import falcor  # type: ignore
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
KERNEL = (
    PROJECT_ROOT
    / "shaders/ncls/backends/unified_neural/unified_checkpoint_parity.cs.slang"
)


def _buffer(device, values: np.ndarray, *, integer: bool = False):
    source = np.ascontiguousarray(values, dtype=np.uint32 if integer else np.float32)
    flags = falcor.ResourceBindFlags.ShaderResource
    result = device.create_structured_buffer(
        struct_size=4, element_count=len(source), bind_flags=flags
    )
    result.from_numpy(source)
    return result


def _float4_buffer(device, values: np.ndarray):
    source = np.ascontiguousarray(
        np.column_stack((values, np.zeros(len(values), dtype=np.float32))),
        dtype=np.float32,
    )
    flags = falcor.ResourceBindFlags.ShaderResource
    result = device.create_structured_buffer(
        struct_size=16, element_count=len(source), bind_flags=flags
    )
    result.from_numpy(source)
    return result


def run(input_path: Path, config_path: Path, output_path: Path) -> None:
    values = np.load(input_path, allow_pickle=False)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = falcor.Device(type=falcor.DeviceType.D3D12)
    compute = falcor.ComputePass(
        device, file=KERNEL, cs_entry="evaluateUnifiedCheckpointParity"
    )
    compute.globals.gWeights = _buffer(device, values["weights"])
    compute.globals.gLatent = _buffer(device, values["latent"])
    compute.globals.gResponseScale = _buffer(device, values["response_scale"])
    compute.globals.gTopKind = _buffer(device, values["top_kind"], integer=True)
    compute.globals.gTopFields = _buffer(device, values["top_fields"])
    compute.globals.gStateIndex = _buffer(device, values["state_index"], integer=True)
    compute.globals.gWo = _float4_buffer(device, values["wo"])
    compute.globals.gWi = _float4_buffer(device, values["wi"])
    flags = falcor.ResourceBindFlags.ShaderResource | falcor.ResourceBindFlags.UnorderedAccess
    output = device.create_structured_buffer(
        struct_size=16, element_count=len(values["state_index"]), bind_flags=flags
    )
    compute.globals.gOutput = output
    compute.globals.gCount = len(values["state_index"])
    compute.globals.gPaper = int(config["paper"])
    compute.globals.gCore = int(config["core"])
    compute.globals.gSampler = int(config["sampler"])
    for name, offset in config["offsets"].items():
        setattr(compute.globals, name, int(offset))
    compute.execute(threads_x=len(values["state_index"]))
    actual = output.to_numpy().view(np.float32).reshape(-1, 4).copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, actual)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.config, args.output)


if __name__ == "__main__":
    main()
