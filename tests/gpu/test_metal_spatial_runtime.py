from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from ncls.learning.methods.metal.model import MetalBudgetedModel
from ncls.learning.methods.metal.native_uv import UVGroup, UVMapping
from ncls.learning.methods.metal.profile import METAL_SPATIAL_PROFILE
from ncls.learning.methods.metal.runtime import pack_metal_budgeted_compiled_material, pack_metal_budgeted_program, quantize_metal_budgeted_runtime_model
from ncls.learning.methods.metal.spatial_cook import MetalSpatialCompiledAsset, SpatialCompiledGroup
from ncls.learning.methods.metal.spatial_runtime import prepare_spatial_cooked_asset
from ncls.references.backend import create_reference_backend
from tests.gpu.test_metal_budgeted_runtime_package import _asset, _entry, _tensors, _texture


falcor = pytest.importorskip("falcor")
pytestmark = pytest.mark.falcor


def test_spatial_groups_native_nonrepeat_filter_and_reverse_prepare_match_gpu(tmp_path: Path):
    torch.manual_seed(316)
    model = quantize_metal_budgeted_runtime_model(MetalBudgetedModel(METAL_SPATIAL_PROFILE).eval())
    tensors = _tensors()
    grid = replace(_asset(), profile_id=METAL_SPATIAL_PROFILE.profile_id)
    groups = (
        SpatialCompiledGroup(UVGroup(UVMapping("rotated", (1.3, 0.2, 0.15, -0.3, 0.7, -0.2), "nonrepeat", 8., 1., 935), (0,)), grid),
        SpatialCompiledGroup(UVGroup(UVMapping("direct", (0.4, -0.1, -0.2, 0.3, 1.2, 0.1)), (2,)), replace(grid, address_mode="clamp")),
    )
    asset = MetalSpatialCompiledAsset(METAL_SPATIAL_PROFILE.profile_id, groups, np.linspace(-0.3, 0.2, 8).astype(np.float16).astype(np.float32))
    packed = pack_metal_budgeted_program(model)
    program = model.compile_program_state(tensors)
    compiled = pack_metal_budgeted_compiled_material(program, asset)
    view = (0.17364818, -0.33682409, 0.92541658)
    lights = ((0.,0.,1.), (0.34202015,0.16317591,0.92541658), (-0.49240388,0.41317591,0.76604444), (0.71984631,-0.60402277,0.34202015))
    uv, dx, dy, random = (0.217, -0.123), (0.17, 0.03), (0.01, 0.09), 0.3
    with torch.no_grad():
        expected_state = prepare_spatial_cooked_asset(model, asset, tensors, uv=uv, uv_dx=dx, uv_dy=dy, filter_random=random, wo=view)
        expected = model.evaluate_prepared(expected_state, torch.tensor([view]), torch.tensor([lights])).f[0].numpy()
    entry = _entry(tmp_path, dict(packed.defines))
    source = entry.read_text(encoding="utf-8")
    # 独立 reverse prepare 放在动态测试循环中，避免编译器复制四份完整网络。
    source = source.replace("[unroll]", "[loop]")
    source = source.replace('context.surface.uv = float2(0.0f, 0.0f);', 'context.surface.uv = float2(0.217f, -0.123f);')
    source = source.replace('context.surface.uvDx = float2(0.0f);', 'context.surface.uvDx = float2(0.17f, 0.03f);')
    source = source.replace('context.surface.uvDy = float2(0.0f);', 'context.surface.uvDy = float2(0.01f, 0.09f);')
    source = source.replace('context.filterRandom = 0.0f;', 'context.filterRandom = 0.3f;')
    source = source.replace('gTestOutput[10u + index * 4u] = float4(sampledEval.f, 0.0f);', '''gTestOutput[10u + index * 4u] = float4(sampledEval.f, 0.0f);
        NclsScatteringContext reverseContext = context;
        reverseContext.woWorld = sampleValue.wiWorld;
        const NclsPackageState independent = nclsCreatePackageBackend().prepare(reverseContext, nclsLoadPackageMaterial(0u));
        const NclsScatteringPdf reverseDensity = independent.pdf(context.woWorld);
        gTestOutput[23u + index] = float4(density.reverse, reverseDensity.forward, float(independent.prepared.valid), 0.0f);''')
    closing = source.rfind("}")
    source = source[:closing] + '''
    NclsPackageState corrupted = state;
    corrupted.prepared.semantic[0] = asfloat(0x7fc00000u);
    NclsMetalBudgetedTestGenerator invalidRng = {17u};
    const NclsScatteringEval invalidEval = corrupted.evaluate(float3(0.0f, 0.0f, 1.0f), invalidRng);
    NclsScatteringSample invalidSample;
    corrupted.sample(invalidSample, invalidRng);
    gTestOutput[27u] = float4(float(invalidEval.valid), float(invalidSample.valid),
        length(invalidEval.f), length(invalidSample.weight));
    const NclsScatteringEval outside = state.evaluate(float3(0.0f, 0.0f, -1.0f), invalidRng);
    gTestOutput[28u] = float4(outside.f, float(outside.valid));
''' + source[closing:]
    entry.write_text(source, encoding="utf-8")
    device = create_reference_backend()._create_device(falcor)
    compute = falcor.ComputePass(device, file=entry, cs_entry="main")
    srv = falcor.ResourceBindFlags.ShaderResource
    def buffer(payload):
        value = device.create_structured_buffer(struct_size=4, element_count=len(payload)//4, bind_flags=srv)
        value.from_numpy(np.frombuffer(payload, dtype=np.uint32).copy())
        return value
    compute.globals.gNclsRuntimeWeights = buffer(packed.payload)
    compute.globals.gNclsCompiledMaterials = buffer(compiled)
    for index in range(9):
        group = asset.groups[index].asset if index < len(asset.groups) else None
        for label in ("Detail", "Context"):
            levels = getattr(group, label.lower()+"_levels") if group else (np.zeros((1,1,4), dtype=np.int8),)
            compute.globals[f"gNclsMetalSpatial{label}{index}"] = _texture(device, levels)
        mode = falcor.TextureAddressingMode.Wrap if group and group.address_mode == "wrap" else falcor.TextureAddressingMode.Clamp
        compute.globals[f"gNclsMetalSpatialSampler{index}"] = device.create_sampler(
            mag_filter=falcor.TextureFilteringMode.Linear, min_filter=falcor.TextureFilteringMode.Linear,
            mip_filter=falcor.TextureFilteringMode.Point, address_mode_u=mode, address_mode_v=mode, address_mode_w=mode)
    light_buffer = device.create_structured_buffer(struct_size=16, element_count=4, bind_flags=srv)
    light_buffer.from_numpy(np.asarray([(*value,0.) for value in lights], dtype=np.float32))
    output = device.create_structured_buffer(struct_size=16, element_count=29, bind_flags=srv | falcor.ResourceBindFlags.UnorderedAccess)
    compute.globals.gTestLights, compute.globals.gTestOutput = light_buffer, output
    compute.execute(threads_x=1)
    actual = output.to_numpy().view(np.float32).reshape(29,4).copy()
    device.end_frame()
    np.testing.assert_array_equal(actual[:4,3], 1.)
    np.testing.assert_allclose(actual[:4,:3], expected, rtol=3e-2, atol=5e-4)
    np.testing.assert_allclose(actual[4:7], expected_state.proposal_state[0].numpy(), rtol=2e-3, atol=2e-5)
    samples = actual[7:23].reshape(4,4,4)
    np.testing.assert_array_equal(samples[:,0,3], 1.)
    np.testing.assert_allclose(samples[:,1,:2], samples[:,1,2:], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(samples[:,2,:3], samples[:,3,:3]*samples[:,0,2:3]/samples[:,1,0:1], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual[23:27,0], actual[23:27,1], rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(actual[23:27,2], 1.)
    np.testing.assert_array_equal(actual[27:], 0.)
