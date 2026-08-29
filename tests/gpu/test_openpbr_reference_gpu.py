from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ncls.core.scattering import ScatteringEvent
from ncls.references import deterministic_directional_metrics, load_reference_acceptance
from ncls.references.backend import create_reference_backend
from ncls.source_materials import (
    OpenPBRMaterial,
    OpenPBRReference,
    load_openpbr_luts,
    resolve_openpbr_inputs,
)


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _openpbr_lut_resources(device):
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

    return {
        "gOpenPbrIdealDielectricEnergy": texture_3d(lut_data.ideal_dielectric_energy),
        "gOpenPbrIdealDielectricAverage": texture_2d(lut_data.ideal_dielectric_average),
        "gOpenPbrIdealDielectricRatio": texture_2d(lut_data.ideal_dielectric_ratio),
        "gOpenPbrOpaqueDielectricEnergy": texture_3d(lut_data.opaque_dielectric_energy),
        "gOpenPbrOpaqueDielectricAverage": texture_2d(lut_data.opaque_dielectric_average),
        "gOpenPbrIdealMetalEnergy": texture_2d(lut_data.ideal_metal_energy),
        "gOpenPbrIdealMetalAverage": texture_2d(lut_data.ideal_metal_average),
        "gOpenPbrLtc": texture_2d(lut_data.ltc),
        "gOpenPbrLutSampler": device.create_sampler(
            address_mode_u=falcor.TextureAddressingMode.Clamp,
            address_mode_v=falcor.TextureAddressingMode.Clamp,
            address_mode_w=falcor.TextureAddressingMode.Clamp,
        ),
    }


def _bind_openpbr_luts(compute, resources) -> None:
    for name, resource in resources.items():
        setattr(compute.globals, name, resource)


def _directions(generator: np.random.Generator, count: int, *, both_hemispheres: bool) -> np.ndarray:
    azimuth = generator.uniform(-np.pi, np.pi, size=count)
    minimum = -1.0 if both_hemispheres else 0.08
    cosine = generator.uniform(minimum, 1.0, size=count)
    if both_hemispheres:
        near_horizon = np.abs(cosine) < 0.04
        cosine[near_horizon] = np.where(cosine[near_horizon] < 0.0, -0.04, 0.04)
    sine = np.sqrt(1.0 - cosine * cosine)
    return np.stack((sine * np.cos(azimuth), sine * np.sin(azimuth), cosine), axis=1).astype(np.float32)


def _inject_grazing_views(
    generator: np.random.Generator, directions: np.ndarray, count: int
) -> None:
    cosine = np.geomspace(1e-6, 0.08, num=count, dtype=np.float32)
    azimuth = generator.uniform(-np.pi, np.pi, size=count).astype(np.float32)
    sine = np.sqrt(1.0 - cosine * cosine)
    directions[:count] = np.stack(
        (sine * np.cos(azimuth), sine * np.sin(azimuth), cosine), axis=1
    )


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

    device = create_reference_backend()._create_device(falcor)  # noqa: SLF001
    compute = falcor.ComputePass(
        device,
        file=PROJECT_ROOT / "tests" / "gpu" / "kernels" / "openpbr_reference.cs.slang",
        cs_entry="evaluateOpenPbrReference",
    )
    _bind_openpbr_luts(compute, _openpbr_lut_resources(device))
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


@pytest.mark.falcor
@pytest.mark.parametrize(
    ("material_id", "require_transmission"),
    (
        ("open_pbr_aluminum_brushed", False),
        ("open_pbr_carpaint", False),
        ("open_pbr_glass", True),
    ),
)
def test_openpbr_native_sample_and_stable_pdf_contract(
    material_id: str, require_transmission: bool
) -> None:
    query_count = 256 * 1024
    generator = np.random.default_rng(20260828)
    views = _directions(generator, query_count, both_hemispheres=False)
    _inject_grazing_views(generator, views, query_count // 8)
    views4 = np.ascontiguousarray(np.pad(views, ((0, 0), (0, 1))))
    material = OpenPBRMaterial.from_materialx(
        PROJECT_ROOT / "external" / "OpenPBR" / "examples" / f"{material_id}.mtlx"
    )
    flat = np.ascontiguousarray(resolve_openpbr_inputs(material))

    device = create_reference_backend()._create_device(falcor)  # noqa: SLF001
    compute = falcor.ComputePass(
        device,
        file=PROJECT_ROOT / "tests" / "gpu" / "kernels" / "openpbr_scattering_contract.cs.slang",
        cs_entry="sampleOpenPbrContract",
    )
    _bind_openpbr_luts(compute, _openpbr_lut_resources(device))
    srv = falcor.ResourceBindFlags.ShaderResource
    uav = falcor.ResourceBindFlags.UnorderedAccess
    view_buffer = device.create_structured_buffer(
        struct_size=16, element_count=query_count, bind_flags=srv
    )
    view_buffer.from_numpy(views4)
    input_buffer = device.create_structured_buffer(
        struct_size=4, element_count=flat.size, bind_flags=srv
    )
    input_buffer.from_numpy(flat)

    outputs = []
    for name in (
        "gDirectionPdf",
        "gSampleWeight",
        "gEvaluateWeight",
        "gPdfAgreement",
    ):
        output = device.create_structured_buffer(
            struct_size=16, element_count=query_count, bind_flags=srv | uav
        )
        setattr(compute.globals, name, output)
        outputs.append(output)
    event_output = device.create_structured_buffer(
        struct_size=16, element_count=query_count, bind_flags=srv | uav
    )
    compute.globals.gEventAgreement = event_output
    native_outputs = []
    for name in ("gNativeDirectionPdf", "gNativeSampleWeight"):
        output = device.create_structured_buffer(
            struct_size=16, element_count=query_count, bind_flags=srv | uav
        )
        setattr(compute.globals, name, output)
        native_outputs.append(output)
    compute.globals.gViews = view_buffer
    compute.globals.gResolvedInputs = input_buffer
    compute.globals.gQueryCount = query_count
    compute.execute(threads_x=query_count)
    direction_pdf, sampled, evaluated, pdf_agreement = [
        output.to_numpy().view(np.float32).reshape(query_count, 4).copy()
        for output in outputs
    ]
    events = event_output.to_numpy().view(np.uint32).reshape(query_count, 4).copy()
    native_direction_pdf, native_sampled = [
        output.to_numpy().view(np.float32).reshape(query_count, 4).copy()
        for output in native_outputs
    ]
    valid = sampled[:, 3] > 0.5
    stable = valid & (views[:, 2] >= 0.08)
    assert valid.mean() > 0.99
    assert stable.sum() > query_count * 0.85
    assert np.isfinite(direction_pdf[valid]).all()
    assert np.isfinite(sampled[valid]).all()
    assert np.isfinite(pdf_agreement[valid]).all()
    invalid_evaluations = np.flatnonzero(stable & (evaluated[:, 3] < 0.5))
    assert not invalid_evaluations.size, json.dumps({
        "indices": invalid_evaluations[:8].tolist(),
        "views": views[invalid_evaluations[:8]].tolist(),
        "sampled_direction_pdf": direction_pdf[invalid_evaluations[:8]].tolist(),
        "sampled_weight": sampled[invalid_evaluations[:8]].tolist(),
        "native_direction_pdf": native_direction_pdf[invalid_evaluations[:8]].tolist(),
        "native_weight": native_sampled[invalid_evaluations[:8]].tolist(),
    }, indent=2)
    np.testing.assert_allclose(
        pdf_agreement[stable, 0], pdf_agreement[stable, 2], rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        pdf_agreement[stable, 1], pdf_agreement[stable, 3], rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        direction_pdf[valid], native_direction_pdf[valid], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        sampled[valid, :3], native_sampled[valid, :3], rtol=0.0, atol=0.0
    )
    sampled_events = events[valid, 0]
    if require_transmission:
        assert np.any(sampled_events & int(ScatteringEvent.TRANSMISSION) != 0)
    else:
        assert np.all(sampled_events & int(ScatteringEvent.REFLECTION) != 0)
    device.end_frame()
