from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ncls.learning.metal_runtime import (
    METAL_COMPILED_WORD_COUNT,
    METAL_RAW_WORD_COUNT,
    pack_metal_compiled_material,
    pack_metal_asset,
    pack_metal_program,
    pack_metal_raw_parameters,
    quantize_runtime_model,
    evaluate_metal_cooked_asset,
)
from ncls.learning.metal_asset_cook import (
    MetalAssetLevelRecord,
    MetalCompiledAssetState,
)
from ncls.learning.models.metal_fused import (
    METAL_FUSED_REQUIRED_CONTEXT,
    MetalFusedNeuralMaterialModel,
)
from ncls.references.backend import create_reference_backend


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _entry(
    tmp_path: Path, defines: dict[str, str], *, runtime_only: bool = False
) -> Path:
    path = tmp_path / "metal_runtime_entry.cs.slang"
    lines = [f"#define {name} {value}" for name, value in sorted(defines.items())]
    if runtime_only:
        lines.append("#define NCLS_METAL_RUNTIME_ONLY 1")
    lines.append(
        '#include "'
        + (PROJECT_ROOT / "shaders/ncls/backends/metal_fused/metal_fused.slang").as_posix()
        + '"'
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _evaluation_entry(tmp_path: Path, defines: dict[str, str]) -> Path:
    path = _entry(tmp_path, defines, runtime_only=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            """
struct NclsMetalTestGenerator : ISampleGenerator
{
    uint state;
    [mutating] uint next() { state = state * 1664525u + 1013904223u; return state; }
};
StructuredBuffer<float4> gTestLights;
RWStructuredBuffer<float4> gTestOutput;
[numthreads(1, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    NclsScatteringContext context = {};
    context.surface.shadingFrame.normal = float3(0.0f, 0.0f, 1.0f);
    context.surface.shadingFrame.tangent = float3(1.0f, 0.0f, 0.0f);
    context.surface.shadingFrame.bitangent = float3(0.0f, 1.0f, 0.0f);
    context.surface.geometricNormal = context.surface.shadingFrame.normal;
    context.surface.uv = float2(0.371f, 0.619f);
    context.surface.uvDx = float2(0.13508363f, 0.0f);
    context.surface.uvDy = float2(0.0f, 0.13508363f);
    context.surface.frontFacing = 1u;
    context.woWorld = normalize(float3(0.17364818f, -0.33682409f, 0.92541658f));
    context.transportMode = (uint)NclsTransportMode::Radiance;
    context.componentMask = (uint)NclsScatteringEvent::Reflection;
    const NclsPackageState state = nclsCreatePackageBackend().prepare(
        context, nclsLoadPackageMaterial(0u));
    [unroll]
    for (uint index = 0u; index < 4u; ++index)
    {
        NclsMetalTestGenerator generator = {index + 1u};
        const NclsScatteringEval evaluation = state.evaluate(normalize(gTestLights[index].xyz), generator);
        gTestOutput[index] = float4(evaluation.f, evaluation.pdf.forward);
    }
}
"""
        )
    return path


def _typed_tensors() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(417)
    present = torch.zeros((1, 32), dtype=torch.int64)
    present[:, :7] = 1
    return {
        "metal_graph_index": torch.tensor([17]),
        "metal_schema_index": torch.tensor([11]),
        "metal_recipe_index": torch.tensor([7]),
        "metal_identity_index": torch.tensor([5]),
        "metal_finish_index": torch.tensor([13]),
        "metal_asset_index": torch.tensor([9]),
        "metal_typed_semantic_id": torch.randint(0, 154, (1, 32), generator=generator),
        "metal_typed_type_id": torch.randint(0, 8, (1, 32), generator=generator),
        "metal_typed_responsibility_id": torch.randint(0, 6, (1, 32), generator=generator),
        "metal_typed_discrete": torch.randint(0, 64, (1, 32), generator=generator),
        "metal_typed_continuous": torch.randn((1, 32, 4), generator=generator),
        "metal_typed_presence": present,
        "metal_canonical_optical": torch.randn((1, 16), generator=generator),
        "metal_access_state": torch.tensor(
            [[1.2, 0.8, 0.1, -0.2, 0.9393727, 0.3428978, 1.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0]]
        ),
        "metal_frame_state": torch.tensor([[1.0, 0.25, 0.0, 1.0, 0, 0, 0, 0]]),
    }


def _asset() -> MetalCompiledAssetState:
    generator = torch.Generator().manual_seed(731)
    records = (
        MetalAssetLevelRecord("surface", 0, 3, (8, 8), (4, 4), (1, 1), 0, 0, 0),
        MetalAssetLevelRecord("surface", 1, 3, (4, 4), (2, 2), (1, 1), 128, 8, 16),
    )
    return MetalCompiledAssetState(
        "metal_fused_full_v1",
        "encoder-only",
        "fixture-collection",
        "fixture-asset",
        "fixture-schema",
        records,
        torch.randint(-100, 101, (160,), dtype=torch.int8, generator=generator),
        torch.randint(-100, 101, (16,), dtype=torch.int8, generator=generator),
        (0.004 + 0.002 * torch.rand((32,), generator=generator)).to(torch.float16),
        torch.randn((8,), generator=generator).mul(0.2).to(torch.float16),
        0,
        0.0,
    )


@pytest.mark.falcor
def test_full_metal_runtime_and_material_compiler_compile(tmp_path: Path) -> None:
    torch.manual_seed(91)
    model = quantize_runtime_model(
        MetalFusedNeuralMaterialModel.from_context(METAL_FUSED_REQUIRED_CONTEXT)
    )
    packed = pack_metal_program(model)
    tensors = _typed_tensors()
    with torch.no_grad():
        expected_state = model.typed_compiler(tensors)
    expected = pack_metal_compiled_material(
        expected_state,
        tensors,
        SimpleNamespace(records=()),
        domain_count=1,
        maximum_extent=1024,
        maximum_mip=9,
    )
    device = create_reference_backend()._create_device(falcor)  # noqa: SLF001
    compute = falcor.ComputePass(
        device,
        file=_entry(tmp_path, dict(packed.defines)),
        cs_entry="nclsCompileMaterial",
    )
    srv = falcor.ResourceBindFlags.ShaderResource
    uav = srv | falcor.ResourceBindFlags.UnorderedAccess
    weights = device.create_structured_buffer(
        struct_size=4, element_count=len(packed.payload) // 4, bind_flags=srv
    )
    raw = device.create_structured_buffer(
        struct_size=4, element_count=METAL_RAW_WORD_COUNT, bind_flags=uav
    )
    compiled = device.create_structured_buffer(
        struct_size=4, element_count=METAL_COMPILED_WORD_COUNT, bind_flags=uav
    )
    weights.from_numpy(np.frombuffer(packed.payload, dtype=np.uint32).copy())
    raw.from_numpy(np.frombuffer(pack_metal_raw_parameters(tensors), dtype=np.uint32).copy())
    compiled.from_numpy(np.frombuffer(expected, dtype=np.uint32).copy())
    compute.globals.gNclsRuntimeWeights = weights
    compute.globals.gNclsMetalRawParameters = raw
    compute.globals.gNclsCompiledMaterials = compiled
    compute.execute(threads_x=1)
    actual_words = compiled.to_numpy().view(np.uint32).copy()
    device.end_frame()
    expected_words = np.frombuffer(expected, dtype=np.uint32)
    # Identity/asset fields are preserved by the editor compiler; every neural
    # float field and both derived access/frame blocks are regenerated.
    float_indices = np.r_[0:300, 316:340]
    np.testing.assert_allclose(
        actual_words[float_indices].view(np.float32),
        expected_words[float_indices].view(np.float32),
        rtol=2e-3,
        atol=2e-4,
    )
    np.testing.assert_array_equal(actual_words[300:316], expected_words[300:316])


@pytest.mark.falcor
def test_quantized_decoder_prepare_and_evaluator_match_python(tmp_path: Path) -> None:
    torch.manual_seed(109)
    model = quantize_runtime_model(
        MetalFusedNeuralMaterialModel.from_context(METAL_FUSED_REQUIRED_CONTEXT)
    )
    tensors = _typed_tensors()
    cooked = _asset()
    packed_program = pack_metal_program(model)
    packed_asset = pack_metal_asset(cooked, address_modes={"surface": "wrap"})
    with torch.no_grad():
        state = model.typed_compiler(tensors)
        expected = evaluate_metal_cooked_asset(
            model,
            cooked,
            tensors,
            uv=(0.371, 0.619),
            mip_level=0.375,
            wo=(0.17364818, -0.33682409, 0.92541658),
            wi=(
                (0.0, 0.0, 1.0),
                (0.34202015, 0.16317591, 0.92541658),
                (-0.49240388, 0.41317591, 0.76604444),
                (0.71984631, -0.60402277, 0.34202015),
            ),
            address_modes={"surface": "wrap"},
        ).numpy()
    compiled_payload = pack_metal_compiled_material(
        state,
        tensors,
        cooked,
        domain_count=packed_asset.domain_count,
        maximum_extent=packed_asset.maximum_extent,
        maximum_mip=packed_asset.maximum_mip,
    )
    device = create_reference_backend()._create_device(falcor)  # noqa: SLF001
    compute = falcor.ComputePass(
        device,
        file=_evaluation_entry(tmp_path, dict(packed_program.defines)),
        cs_entry="main",
    )
    srv = falcor.ResourceBindFlags.ShaderResource
    uav = srv | falcor.ResourceBindFlags.UnorderedAccess

    def buffer(payload: bytes, flags=srv):
        result = device.create_structured_buffer(
            struct_size=4, element_count=len(payload) // 4, bind_flags=flags
        )
        result.from_numpy(np.frombuffer(payload, dtype=np.uint32).copy())
        return result

    compute.globals.gNclsRuntimeWeights = buffer(packed_program.payload)
    compute.globals.gNclsMetalDomains = buffer(packed_asset.blobs["metal-domains"])
    compute.globals.gNclsMetalRecords = buffer(packed_asset.blobs["metal-records"])
    compute.globals.gNclsMetalHighGrid = buffer(packed_asset.blobs["metal-high-grid"])
    compute.globals.gNclsMetalLowGrid = buffer(packed_asset.blobs["metal-low-grid"])
    compute.globals.gNclsMetalGridScales = buffer(packed_asset.blobs["metal-grid-scales"])
    compute.globals.gNclsMetalAssetAdapter = buffer(packed_asset.blobs["metal-asset-adapter"])
    compute.globals.gNclsMetalRawParameters = buffer(pack_metal_raw_parameters(tensors), uav)
    compute.globals.gNclsCompiledMaterials = buffer(compiled_payload, uav)
    lights = np.asarray(
        (
            (0.0, 0.0, 1.0, 0.0),
            (0.34202015, 0.16317591, 0.92541658, 0.0),
            (-0.49240388, 0.41317591, 0.76604444, 0.0),
            (0.71984631, -0.60402277, 0.34202015, 0.0),
        ),
        dtype=np.float32,
    )
    light_buffer = device.create_structured_buffer(struct_size=16, element_count=4, bind_flags=srv)
    light_buffer.from_numpy(lights)
    output = device.create_structured_buffer(struct_size=16, element_count=4, bind_flags=uav)
    compute.globals.gTestLights = light_buffer
    compute.globals.gTestOutput = output
    compute.execute(threads_x=1)
    actual = output.to_numpy().view(np.float32).reshape(4, 4).copy()
    device.end_frame()
    np.testing.assert_allclose(actual[:, :3], expected, rtol=6e-3, atol=8e-4)
    assert np.isfinite(actual).all()
    assert np.all(actual[:, 3] > 0.0)
