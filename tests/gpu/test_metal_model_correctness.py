from pathlib import Path

import numpy as np
import pytest

from ncls.paths import PROJECT_ROOT
from ncls.references.backend import create_reference_backend

falcor = pytest.importorskip("falcor")


@pytest.mark.falcor
def test_softplus_normal_tail_and_continuous_frame_against_float64(tmp_path: Path):
    shader = tmp_path / "metal_math.cs.slang"
    common = (PROJECT_ROOT / "shaders/ncls/backends/metal_budgeted/metal_budgeted_common.slang").as_posix()
    shader.write_text('StructuredBuffer<uint> gNclsRuntimeWeights;\nStructuredBuffer<uint> gNclsCompiledMaterials;\n'
                      f'#include "{common}"\n' + '''
StructuredBuffer<float4> gInput;
RWStructuredBuffer<float4> gOutput;
[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    float4 input = gInput[tid.x];
    NclsMetalBudgetedFrame frame = nclsMetalBudgetedOrthonormalFrame(normalize(input.yzw));
    gOutput[3u*tid.x] = float4(frame.tangent, nclsMetalBudgetedSoftplus(input.x));
    gOutput[3u*tid.x+1u] = float4(frame.bitangent, 0.0f);
    gOutput[3u*tid.x+2u] = float4(frame.normal, 0.0f);
}
''', encoding="utf-8")
    x = np.array([-80, -64, -40, -24, -20, -16, -10, -2, 0, 2, 16, 20, 40, 80], dtype=np.float32)
    z = np.linspace(0.99, 1.0, len(x), dtype=np.float32)
    inputs = np.stack((x, np.sqrt(1 - z * z), np.zeros_like(z), z), axis=1)
    device = create_reference_backend()._create_device(falcor)
    compute = falcor.ComputePass(device, file=shader, cs_entry="main")
    srv = falcor.ResourceBindFlags.ShaderResource
    source = device.create_structured_buffer(struct_size=16, element_count=len(x), bind_flags=srv)
    source.from_numpy(inputs)
    output = device.create_structured_buffer(struct_size=16, element_count=3 * len(x), bind_flags=srv | falcor.ResourceBindFlags.UnorderedAccess)
    compute.globals.gInput = source
    compute.globals.gOutput = output
    compute.execute(threads_x=len(x))
    actual = output.to_numpy().view(np.float32).reshape(len(x), 3, 4).copy()
    device.end_frame()
    oracle = np.maximum(x.astype(np.float64), 0) + np.log1p(np.exp(-np.abs(x.astype(np.float64))))
    # 截断 <1.1e-9；exp 与不超过 32 次 float32 基本运算取 64 epsilon 相对预算。
    np.testing.assert_allclose(actual[:, 0, 3], oracle, rtol=64 * np.finfo(np.float32).eps, atol=0)
    assert np.all(np.diff(actual[:, 0, 3]) > 0)
    frame = actual[:, :, :3].astype(np.float64)
    gram = np.sum(frame[:, :, None, :] * frame[:, None, :, :], axis=-1)
    np.testing.assert_allclose(gram, np.broadcast_to(np.eye(3), frame.shape), atol=16 * np.finfo(np.float32).eps, rtol=0)
