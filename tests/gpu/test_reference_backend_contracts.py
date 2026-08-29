from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ncls.core.material import (
    BINARY_SIZE,
    DiffuseInterface,
    HomogeneousMedium,
    LayerStackIR,
    RoughDielectricInterface,
    pack_layer_stack,
)
from ncls.core.scattering import ScatteringEvent
from ncls.references.backend import create_reference_backend
from ncls.source_materials import MerlBrdfReference, MerlMaterial


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KERNEL = PROJECT_ROOT / "tests/gpu/kernels/reference_backend_contracts.cs.slang"
QUERY_COUNT = 64 * 1024


def _views() -> np.ndarray:
    generator = np.random.default_rng(20260828)
    azimuth = generator.uniform(-np.pi, np.pi, size=QUERY_COUNT)
    cosine = generator.uniform(0.2, 1.0, size=QUERY_COUNT)
    sine = np.sqrt(1.0 - cosine * cosine)
    directions = np.stack(
        (sine * np.cos(azimuth), sine * np.sin(azimuth), cosine), axis=1
    ).astype(np.float32)
    return np.ascontiguousarray(np.pad(directions, ((0, 0), (0, 1))))


def _make_probe(entry: str):
    device = create_reference_backend()._create_device(falcor)  # noqa: SLF001
    compute = falcor.ComputePass(device, file=KERNEL, cs_entry=entry)
    srv = falcor.ResourceBindFlags.ShaderResource
    uav = falcor.ResourceBindFlags.UnorderedAccess
    view_buffer = device.create_structured_buffer(
        struct_size=16, element_count=QUERY_COUNT, bind_flags=srv
    )
    view_buffer.from_numpy(_views())
    compute.globals.gViews = view_buffer
    outputs = {}
    for name in (
        "gDirectionPdf",
        "gSampleWeightValid",
        "gEvaluateWeightValid",
        "gPdfAgreement",
    ):
        output = device.create_structured_buffer(
            struct_size=16, element_count=QUERY_COUNT, bind_flags=srv | uav
        )
        setattr(compute.globals, name, output)
        outputs[name] = output
    event_output = device.create_structured_buffer(
        struct_size=16, element_count=QUERY_COUNT, bind_flags=srv | uav
    )
    compute.globals.gEventAgreement = event_output
    compute.globals.gQueryCount = QUERY_COUNT
    return device, compute, outputs, event_output


def _execute_and_check(
    device,
    compute,
    outputs,
    event_output,
    *,
    compare_weight: bool,
) -> None:
    compute.execute(threads_x=QUERY_COUNT)
    values = {
        name: output.to_numpy().view(np.float32).reshape(QUERY_COUNT, 4).copy()
        for name, output in outputs.items()
    }
    events = event_output.to_numpy().view(np.uint32).reshape(QUERY_COUNT, 4).copy()
    valid = events[:, 2] != 0
    assert valid.mean() > 0.5
    assert np.isfinite(values["gDirectionPdf"][valid]).all()
    assert np.isfinite(values["gSampleWeightValid"][valid]).all()
    assert np.isfinite(values["gEvaluateWeightValid"][valid]).all()
    assert np.isfinite(values["gPdfAgreement"][valid]).all()
    np.testing.assert_allclose(
        np.linalg.norm(values["gDirectionPdf"][valid, :3], axis=1),
        1.0,
        rtol=2e-6,
        atol=2e-7,
    )
    assert (values["gDirectionPdf"][valid, 3] > 0.0).all()
    assert (values["gPdfAgreement"][valid, 0] > 0.0).all()
    np.testing.assert_allclose(
        values["gDirectionPdf"][valid, 3],
        values["gPdfAgreement"][valid, 0],
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        values["gPdfAgreement"][valid, 0],
        values["gPdfAgreement"][valid, 2],
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        values["gPdfAgreement"][valid, 1],
        values["gPdfAgreement"][valid, 3],
        rtol=2e-6,
        atol=2e-7,
    )
    assert (events[valid, 0] & int(ScatteringEvent.REFLECTION) != 0).all()
    assert (events[valid, 1] & int(ScatteringEvent.REFLECTION) != 0).all()
    assert (events[valid, 3] != 0).all()
    if compare_weight:
        np.testing.assert_allclose(
            values["gSampleWeightValid"][valid, :3],
            values["gEvaluateWeightValid"][valid, :3],
            rtol=3e-5,
            atol=3e-6,
        )
    device.end_frame()


def _bind_layer_material(compute, device, material: LayerStackIR) -> None:
    buffer = device.create_structured_buffer(
        struct_size=BINARY_SIZE,
        element_count=1,
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    buffer.from_numpy(np.frombuffer(pack_layer_stack(material), dtype=np.uint8).copy())
    compute.globals.gLayerMaterials = buffer


@pytest.mark.falcor
@pytest.mark.parametrize(
    ("material", "compare_weight"),
    (
        (LayerStackIR((DiffuseInterface((0.6, 0.3, 0.1)),), ()), True),
        (
            LayerStackIR(
                (
                    RoughDielectricInterface(0.08, 0.16, 1.5, 0.25),
                    DiffuseInterface((0.5, 0.25, 0.1)),
                ),
                (HomogeneousMedium(thickness=0.2),),
            ),
            False,
        ),
    ),
)
def test_layer_stack_canonical_sample_pdf_contract(
    material: LayerStackIR, compare_weight: bool
) -> None:
    device, compute, outputs, events = _make_probe("sampleLayerStackContract")
    _bind_layer_material(compute, device, material)
    _execute_and_check(
        device, compute, outputs, events, compare_weight=compare_weight
    )


@pytest.mark.falcor
def test_merl_canonical_sample_pdf_weight_contract() -> None:
    asset_root = PROJECT_ROOT / "assets/source-materials/merl-brdf/v1"
    marker_path = asset_root / "complete.json"
    if not marker_path.is_file():
        pytest.skip("MERL source material asset is not downloaded")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    material_index = json.loads(
        (PROJECT_ROOT / marker["material_index"]).read_text(encoding="utf-8")
    )
    record = next(
        item for item in material_index["materials"] if item["material_id"] == "chrome"
    )
    table = MerlBrdfReference(
        MerlMaterial("chrome", record["table_uri"]), asset_root
    ).gpu_table()
    device, compute, outputs, events = _make_probe("sampleMerlContract")
    buffer = device.create_structured_buffer(
        struct_size=12,
        element_count=table.shape[0],
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    buffer.from_numpy(table)
    compute.globals.gMerlMaterial = buffer
    _execute_and_check(device, compute, outputs, events, compare_weight=True)


def _constant_texture(device, value: tuple[float, float, float, float]):
    texture = device.create_texture(
        width=1,
        height=1,
        format=falcor.ResourceFormat.RGBA32Float,
        mip_levels=1,
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    texture.from_numpy(np.asarray([[value]], dtype=np.float32))
    return texture


@pytest.mark.falcor
def test_materialx_canonical_sample_pdf_weight_contract() -> None:
    inputs = np.zeros(24, dtype=np.float32)
    inputs[0] = 0.65
    inputs[1:4] = (0.18, 0.42, 0.73)
    inputs[4] = 0.2
    inputs[5] = 0.7
    inputs[8] = 1.0
    inputs[9:12] = 1.0
    inputs[12] = 0.18
    inputs[14] = 1.5
    inputs[15] = 0.72
    inputs[16] = 0.31
    inputs[17] = 1.0
    inputs[23] = 1.0

    device, compute, outputs, events = _make_probe("sampleMaterialXContract")
    buffer = device.create_structured_buffer(
        struct_size=4,
        element_count=inputs.size,
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    buffer.from_numpy(inputs)
    compute.globals.gMaterialXInputs = buffer
    compute.globals.gMaterialXBaseColor = _constant_texture(
        device, (0.18, 0.42, 0.73, 1.0)
    )
    compute.globals.gMaterialXRoughness = _constant_texture(
        device, (0.18, 0.18, 0.18, 1.0)
    )
    compute.globals.gMaterialXMetalness = _constant_texture(
        device, (0.7, 0.7, 0.7, 1.0)
    )
    compute.globals.gMaterialXNormalMap = _constant_texture(
        device, (0.5, 0.5, 1.0, 1.0)
    )
    compute.globals.gMaterialXSampler = device.create_sampler(
        address_mode_u=falcor.TextureAddressingMode.Wrap,
        address_mode_v=falcor.TextureAddressingMode.Wrap,
        address_mode_w=falcor.TextureAddressingMode.Wrap,
    )
    _execute_and_check(device, compute, outputs, events, compare_weight=True)
