from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ncls.references import deterministic_directional_metrics, load_reference_acceptance
from ncls.source_materials import (
    OpenPBRMaterial,
    OpenPBRReference,
    load_openpbr_luts,
    resolve_openpbr_inputs,
)


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _directions(generator: np.random.Generator, count: int, *, both_hemispheres: bool) -> np.ndarray:
    azimuth = generator.uniform(-np.pi, np.pi, size=count)
    minimum = -1.0 if both_hemispheres else 0.08
    cosine = generator.uniform(minimum, 1.0, size=count)
    if both_hemispheres:
        near_horizon = np.abs(cosine) < 0.04
        cosine[near_horizon] = np.where(cosine[near_horizon] < 0.0, -0.04, 0.04)
    sine = np.sqrt(1.0 - cosine * cosine)
    return np.stack((sine * np.cos(azimuth), sine * np.sin(azimuth), cosine), axis=1).astype(np.float32)


@pytest.mark.falcor
def test_openpbr_slang_runtime_matches_adobe_cpp_reference() -> None:
    executable = PROJECT_ROOT / "build" / "openpbr-probe" / "Release" / "ncls_openpbr_probe.exe"
    if not executable.is_file():
        pytest.skip("OpenPBR reference probe is not built")
    material_ids = (
        "open_pbr_aluminum_brushed",
        "open_pbr_brass",
        "open_pbr_carpaint",
        "open_pbr_glass",
        "open_pbr_pearl",
        "open_pbr_soapbubble",
        "open_pbr_velvet",
    )
    index = json.loads((PROJECT_ROOT / "references" / "openpbr-1.1.1-v1" / "materials.json").read_text(encoding="utf-8"))
    records = {record["material_id"]: record for record in index["materials"]}
    source_root = PROJECT_ROOT / "external" / "OpenPBR"
    generator = np.random.default_rng(20260823)
    query_count = 2048
    views = _directions(generator, query_count, both_hemispheres=False)
    lights = _directions(generator, query_count, both_hemispheres=True)
    native_reference = OpenPBRReference(executable)
    acceptance = load_reference_acceptance(PROJECT_ROOT / "references" / "acceptance.json")

    device = falcor.Device(type=falcor.DeviceType.D3D12)
    compute = falcor.ComputePass(
        device,
        file=PROJECT_ROOT / "tests" / "gpu" / "kernels" / "openpbr_reference.cs.slang",
        cs_entry="evaluateOpenPbrReference",
    )
    lut_data = load_openpbr_luts(PROJECT_ROOT / "external" / "openpbr-bsdf")
    lut_format = falcor.ResourceFormat.RGBA32Float

    def texture_2d(data: np.ndarray):
        texture = device.create_texture(
            width=data.shape[1], height=data.shape[0], format=lut_format, mip_levels=1,
            bind_flags=falcor.ResourceBindFlags.ShaderResource,
        )
        texture.from_numpy(np.ascontiguousarray(data))
        return texture

    def texture_3d(data: np.ndarray):
        texture = device.create_texture(
            width=data.shape[2], height=data.shape[1], depth=data.shape[0], format=lut_format, mip_levels=1,
            bind_flags=falcor.ResourceBindFlags.ShaderResource,
        )
        texture.from_numpy(np.ascontiguousarray(data))
        return texture

    compute.globals.gOpenPbrIdealDielectricEnergy = texture_3d(lut_data.ideal_dielectric_energy)
    compute.globals.gOpenPbrIdealDielectricAverage = texture_2d(lut_data.ideal_dielectric_average)
    compute.globals.gOpenPbrIdealDielectricRatio = texture_2d(lut_data.ideal_dielectric_ratio)
    compute.globals.gOpenPbrOpaqueDielectricEnergy = texture_3d(lut_data.opaque_dielectric_energy)
    compute.globals.gOpenPbrOpaqueDielectricAverage = texture_2d(lut_data.opaque_dielectric_average)
    compute.globals.gOpenPbrIdealMetalEnergy = texture_2d(lut_data.ideal_metal_energy)
    compute.globals.gOpenPbrIdealMetalAverage = texture_2d(lut_data.ideal_metal_average)
    compute.globals.gOpenPbrLtc = texture_2d(lut_data.ltc)
    compute.globals.gOpenPbrLutSampler = device.create_sampler(
        address_mode_u=falcor.TextureAddressingMode.Clamp,
        address_mode_v=falcor.TextureAddressingMode.Clamp,
        address_mode_w=falcor.TextureAddressingMode.Clamp,
    )
    srv = falcor.ResourceBindFlags.ShaderResource
    views4 = np.ascontiguousarray(np.pad(views, ((0, 0), (0, 1))))
    lights4 = np.ascontiguousarray(np.pad(lights, ((0, 0), (0, 1))))
    view_buffer = device.create_structured_buffer(struct_size=16, element_count=query_count, bind_flags=srv)
    light_buffer = device.create_structured_buffer(struct_size=16, element_count=query_count, bind_flags=srv)
    view_buffer.from_numpy(views4)
    light_buffer.from_numpy(lights4)
    output = device.create_structured_buffer(
        struct_size=16,
        element_count=query_count,
        bind_flags=srv | falcor.ResourceBindFlags.UnorderedAccess,
    )
    compute.globals.gViews = view_buffer
    compute.globals.gLights = light_buffer
    compute.globals.gOutput = output
    compute.globals.gQueryCount = query_count

    for material_id in material_ids:
        material = OpenPBRMaterial.from_materialx(source_root / records[material_id]["document"])
        flat = np.ascontiguousarray(resolve_openpbr_inputs(material))
        input_buffer = device.create_structured_buffer(struct_size=4, element_count=flat.size, bind_flags=srv)
        input_buffer.from_numpy(flat)
        compute.globals.gResolvedInputs = input_buffer
        compute.execute(threads_x=query_count)
        candidate = output.to_numpy().view(np.float32).reshape(query_count, 4)[:, :3]
        native = native_reference.evaluate(material, views, lights).response_cos
        metrics = deterministic_directional_metrics(native, candidate, acceptance.deterministic_directional)
        print(
            f"{material_id}: median={metrics.median_relative_l1:.9g}, "
            f"p99={metrics.p99_relative_l1:.9g}, max={metrics.max_relative_l1:.9g}, "
            f"max_abs={metrics.max_absolute_error:.9g}, scaled_abs={metrics.max_scaled_absolute_error:.9g}"
        )
        if not metrics.passed:
            error = np.sum(np.abs(native - candidate), axis=1)
            for query_index in np.argsort(error)[-3:]:
                print(
                    "mismatch",
                    int(query_index),
                    "view", views[query_index].tolist(),
                    "light", lights[query_index].tolist(),
                    "native", native[query_index].tolist(),
                    "candidate", candidate[query_index].tolist(),
                )
        assert metrics.passed, f"{material_id}: {metrics}"
