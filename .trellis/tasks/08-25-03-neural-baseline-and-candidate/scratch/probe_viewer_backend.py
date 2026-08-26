from __future__ import annotations

import json
from pathlib import Path

import falcor  # type: ignore
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BUNDLE = ROOT / "artifacts/exports/unified-scattering-03-nvidia-original-native-v3"
OUTPUT = ROOT / "artifacts/reports/unified-scattering-03/nvidia-viewer-parity-probe.json"


def buffer(device, values: np.ndarray, struct_size: int, flags):
    source = np.ascontiguousarray(values).copy()
    result = device.create_structured_buffer(
        struct_size=struct_size,
        element_count=source.nbytes // struct_size,
        bind_flags=flags,
    )
    result.from_numpy(source)
    return result


def main() -> None:
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    parity = json.loads((BUNDLE / manifest["files"]["parity"]).read_text(encoding="utf-8"))
    specialization = manifest["runtime"]["shader_specialization"]
    defines = dict(specialization["defines"])
    defines["NCLS_METHOD_BACKEND_HEADER"] = f'"../../../shaders/{specialization["module"]}"'
    device = falcor.Device(type=falcor.DeviceType.D3D12)
    shader = (
        ROOT
        / "external/Falcor/build/windows-vs2022/bin/Release/shaders/NclsViewer/shaders/Parity.cs.slang"
    )
    compute = falcor.ComputePass(device, file=shader, cs_entry="main", defines=defines)
    read_flags = falcor.ResourceBindFlags.ShaderResource
    write_flags = read_flags | falcor.ResourceBindFlags.UnorderedAccess
    weights = np.frombuffer(
        (BUNDLE / manifest["files"]["shared_weights"]).read_bytes(), dtype=np.uint32
    )
    materials = np.frombuffer(
        (BUNDLE / manifest["files"]["compiled_materials"]).read_bytes(), dtype=np.uint32
    )
    view = np.asarray([parity["view_direction_local"] + [0.0]], dtype=np.float32)
    lights = np.column_stack((
        np.asarray(parity["light_directions_local"], dtype=np.float32),
        np.zeros(len(parity["light_directions_local"]), dtype=np.float32),
    ))
    compute.globals.gSharedWeights = buffer(device, weights, 4, read_flags)
    compute.globals.gCompiledMaterials = buffer(
        device, materials, int(specialization["compiled_material_stride"]), read_flags
    )
    compute.globals.gViews = buffer(device, view, 16, read_flags)
    compute.globals.gLights = buffer(device, lights, 16, read_flags)
    output = device.create_structured_buffer(
        struct_size=16, element_count=len(lights), bind_flags=write_flags
    )
    compute.globals.gOutput = output
    compute.globals.gLightCount = len(lights)
    compute.globals.gCompiledMaterialIndex = int(specialization["compiled_material_index"])
    compute.execute(threads_x=len(lights))
    actual = output.to_numpy().view(np.float32).reshape(-1, 4)[:, :3]
    expected = np.asarray(parity["expected_response_cos"], dtype=np.float32)
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), 1e-30)
    report = {
        "maximum_absolute_error": float(absolute.max()),
        "maximum_relative_error": float(relative.max()),
        "actual": actual.tolist(),
        "expected": expected.tolist(),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
