from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ncls.learning.metal_budgeted_asset_cook import MetalBudgetedCompiledAsset
from ncls.learning.metal_budgeted_runtime import (
    evaluate_metal_budgeted_cooked_asset,
    pack_metal_budgeted_compiled_material,
    pack_metal_budgeted_program,
    quantize_metal_budgeted_program_state,
    quantize_metal_budgeted_runtime_model,
)
from ncls.learning.models.metal_budgeted import MetalBudgetedModel
from ncls.references.backend import create_reference_backend


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _entry(tmp_path: Path, defines: dict[str, str]) -> Path:
    path = tmp_path / "metal_budgeted_runtime.cs.slang"
    lines = [f"#define {name} {value}" for name, value in sorted(defines.items())]
    lines.append(
        '#include "'
        + (
            PROJECT_ROOT
            / "shaders/ncls/backends/metal_budgeted/metal_budgeted.slang"
        ).as_posix()
        + '"'
    )
    lines.append(
        """
struct NclsMetalBudgetedTestGenerator : ISampleGenerator
{
    uint state;
    [mutating] uint next()
    {
        state = state * 1664525u + 1013904223u;
        return state;
    }
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
    context.surface.uv = float2(0.0f, 0.0f);
    context.surface.uvDx = float2(0.0f);
    context.surface.uvDy = float2(0.0f);
    context.surface.frontFacing = 1u;
    context.woWorld = normalize(
        float3(0.17364818f, -0.33682409f, 0.92541658f));
    context.transportMode = (uint)NclsTransportMode::Radiance;
    context.componentMask = (uint)NclsScatteringEvent::Reflection;
    context.filterRandom = 0.0f;
    const NclsPackageState prepared = nclsCreatePackageBackend().prepare(
        context, nclsLoadPackageMaterial(0u));
    const NclsMetalBudgetedPackedState packed = nclsPackMethodState(prepared);
    const NclsPackageState state = nclsUnpackMethodState(
        context, nclsLoadPackageMaterial(0u), packed, gNclsRuntimeWeights);
    [unroll]
    for (uint index = 0u; index < 4u; ++index)
    {
        NclsMetalBudgetedTestGenerator generator = {index + 1u};
        const NclsScatteringEval evaluation = state.evaluate(
            normalize(gTestLights[index].xyz), generator);
        gTestOutput[index] = float4(evaluation.f, float(evaluation.valid));
    }
}
"""
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _tensors() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    presence = torch.zeros((1, 32), dtype=torch.int64)
    presence[:, :9] = 1
    return {
        "metal_graph_index": torch.tensor([3]),
        "metal_schema_index": torch.tensor([2]),
        "metal_recipe_index": torch.tensor([1]),
        "metal_identity_index": torch.tensor([4]),
        "metal_finish_index": torch.tensor([5]),
        "metal_asset_index": torch.tensor([0]),
        "metal_typed_semantic_id": torch.randint(
            0, 154, (1, 32), generator=generator
        ),
        "metal_typed_type_id": torch.randint(0, 8, (1, 32), generator=generator),
        "metal_typed_responsibility_id": torch.randint(
            0, 6, (1, 32), generator=generator
        ),
        "metal_typed_discrete": torch.randint(
            0, 64, (1, 32), generator=generator
        ),
        "metal_typed_continuous": torch.randn((1, 32, 4), generator=generator),
        "metal_typed_presence": presence,
        "metal_canonical_optical": torch.randn((1, 16), generator=generator),
        "metal_access_state": torch.tensor(
            [[1.0, 1.0, 0.125, -0.25, 0.9393727, 0.3428978, 1.0, 0.0]
             + [0.0] * 8]
        ),
        "metal_frame_state": torch.zeros((1, 8)),
        "metal_distribution_id": torch.tensor([1]),
    }


def _asset() -> MetalBudgetedCompiledAsset:
    generator = np.random.default_rng(43)

    def make(base: int) -> tuple[np.ndarray, ...]:
        result = []
        extent = base
        while True:
            result.append(
                generator.integers(
                    -100, 101, size=(extent, extent, 4), dtype=np.int8
                )
            )
            if extent == 1:
                return tuple(result)
            extent = max(1, extent // 2)

    return MetalBudgetedCompiledAsset(
        "metal_budgeted_hybrid_v3",
        "encoder-only@1",
        "fixture-collection",
        "fixture-asset",
        "fixture-schema",
        "wrap",
        make(8),
        make(2),
    )


def _texture(device, levels: tuple[np.ndarray, ...]):
    texture = device.create_texture(
        width=levels[0].shape[1],
        height=levels[0].shape[0],
        format=falcor.ResourceFormat.RGBA8Snorm,
        mip_levels=len(levels),
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    for index, level in enumerate(levels):
        texture.from_numpy(np.ascontiguousarray(level), mip_level=index)
    return texture


@pytest.mark.falcor
def test_budgeted_fp16_snorm8_prepare_and_evaluator_match_python(
    tmp_path: Path,
) -> None:
    torch.manual_seed(47)
    model = quantize_metal_budgeted_runtime_model(MetalBudgetedModel().eval())
    tensors = _tensors()
    asset = _asset()
    packed_program = pack_metal_budgeted_program(model)
    program = quantize_metal_budgeted_program_state(
        model.compile_program_state(tensors)
    )
    compiled = pack_metal_budgeted_compiled_material(program, asset)
    lights = (
        (0.0, 0.0, 1.0),
        (0.34202015, 0.16317591, 0.92541658),
        (-0.49240388, 0.41317591, 0.76604444),
        (0.71984631, -0.60402277, 0.34202015),
    )
    expected = evaluate_metal_budgeted_cooked_asset(
        model,
        asset,
        tensors,
        uv=(0.0, 0.0),
        mip_level=0.0,
        filter_random=0.0,
        wo=(0.17364818, -0.33682409, 0.92541658),
        wi=lights,
    ).numpy()
    device = create_reference_backend()._create_device(falcor)  # noqa: SLF001
    compute = falcor.ComputePass(
        device,
        file=_entry(tmp_path, dict(packed_program.defines)),
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
    compute.globals.gNclsCompiledMaterials = buffer(compiled)
    compute.globals.gNclsMetalBudgetedDetail = _texture(
        device, asset.detail_levels
    )
    compute.globals.gNclsMetalBudgetedContext = _texture(
        device, asset.context_levels
    )
    compute.globals.gNclsMetalBudgetedSampler = device.create_sampler(
        mag_filter=falcor.TextureFilteringMode.Linear,
        min_filter=falcor.TextureFilteringMode.Linear,
        mip_filter=falcor.TextureFilteringMode.Point,
        address_mode_u=falcor.TextureAddressingMode.Wrap,
        address_mode_v=falcor.TextureAddressingMode.Wrap,
        address_mode_w=falcor.TextureAddressingMode.Wrap,
    )
    light_values = np.asarray([(*value, 0.0) for value in lights], dtype=np.float32)
    light_buffer = device.create_structured_buffer(
        struct_size=16, element_count=4, bind_flags=srv
    )
    light_buffer.from_numpy(light_values)
    output = device.create_structured_buffer(
        struct_size=16, element_count=4, bind_flags=uav
    )
    compute.globals.gTestLights = light_buffer
    compute.globals.gTestOutput = output
    compute.execute(threads_x=1)
    actual = output.to_numpy().view(np.float32).reshape(4, 4).copy()
    device.end_frame()
    np.testing.assert_allclose(actual[:, :3], expected, rtol=3e-2, atol=5e-4)
    np.testing.assert_array_equal(actual[:, 3], np.ones(4, dtype=np.float32))
