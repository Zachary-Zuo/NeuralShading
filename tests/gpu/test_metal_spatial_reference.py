import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ncls.core.source import create_source_family
from ncls.references.backend import create_reference_backend
from ncls.references.plan import compile_single_program_plan
from ncls.references.programs import get_reference_program_for_source
from ncls.references.query import ScatteringQuery


pytest.importorskip("falcor")
pytestmark = pytest.mark.falcor


def test_native_texture_axes_absolute_color_and_full_response_footprint(tmp_path: Path):
    image = np.empty((4, 8, 3), dtype=np.uint8)
    for y in range(4):
        for x in range(8):
            image[y, x] = (20+17*x, 30+41*y, 40+9*x+11*y)
    (tmp_path / "asymmetric.ppm").write_bytes(b"P6\n8 4\n255\n" + image.tobytes())
    (tmp_path / "spatial.mdl").write_text('''mdl 1.7;
import ::df::*;
import ::state::*;
import ::tex::*;
export material spatial() = material(surface: material_surface(
    scattering: df::diffuse_reflection_bsdf(tint: tex::lookup_color(
        texture_2d("./asymmetric.ppm", ::tex::gamma_linear),
        float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y)))));
''', encoding="utf-8")
    family = create_source_family("mdl.program@1")
    snapshot = family.load_snapshot({"kind": "mdl-export", "module_root": str(tmp_path), "module": "::spatial", "export": "spatial"})
    definition = get_reference_program_for_source(snapshot.family_id, snapshot.source_contract_version)
    plan = compile_single_program_plan(definition, (snapshot,), query_recipe={"recipe_id": "spatial-d0-independent@1", "evaluation_samples": 1, "footprint_samples": 64})
    session = create_reference_backend().open(plan, query_capacity=192, device="cuda:0", requested_operations=("evaluate",))
    coordinates = [(1,0), (5,2), (2,3)]
    uv = torch.tensor([((x+0.5)/8, 1.-(y+0.5)/4) for x,y in coordinates], device="cuda:0")
    def evaluate(query_uv, dx, dy, samples):
        count = len(query_uv)
        direction = query_uv.new_tensor((0.,0.,1.))[None].expand(count,-1)
        result = session.evaluate(ScatteringQuery(torch.zeros(count,dtype=torch.int64,device="cuda:0"), direction,
            plan.groups[0].group_id, uv=query_uv, uv_dx=dx, uv_dy=dy,
            filter_random=torch.linspace(0.1,0.9,count,device="cuda:0")), direction[:,None],
            torch.arange(count,dtype=torch.int64,device="cuda:0")[:,None], evaluation_samples=1, footprint_samples=samples)
        try:
            assert result.valid.all()
            return result.f[:,0].clone()
        finally:
            result.lease.release()
            session.end_iteration()
    try:
        zero = torch.zeros_like(uv)
        center = evaluate(uv,zero,zero,1)
        expected = torch.tensor(np.asarray([image[y,x] for x,y in coordinates]).astype(np.float32)/255/math.pi, device="cuda:0")
        torch.testing.assert_close(center,expected,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(evaluate(uv,zero,zero,64),center,rtol=2e-5,atol=2e-6)
        dx = uv.new_tensor((0.25,0.03125))[None].expand(3,-1)
        dy = uv.new_tensor((0.0625,0.25))[None].expand(3,-1)
        for count in (16,64):
            # 独立枚举积分节点后逐点求完整 native response；不把平均纹理当 GT。
            radical = [sum(((i >> bit)&1)*2.**(-bit-1) for bit in range(32)) for i in range(count)]
            nodes = uv.new_tensor([((i+0.5)/count-0.5,radical[i]-0.5) for i in range(count)])
            expanded_uv = (uv[:,None] + dx[:,None]*nodes[None,:,0:1] + dy[:,None]*nodes[None,:,1:2]).reshape(-1,2)
            point = evaluate(expanded_uv,dx[:,None].expand(-1,count,-1).reshape(-1,2),dy[:,None].expand(-1,count,-1).reshape(-1,2),1)
            filtered = evaluate(uv,dx,dy,count)
            torch.testing.assert_close(filtered,point.reshape(3,count,3).mean(1),rtol=3e-5,atol=2e-6)
    finally:
        session.close()
