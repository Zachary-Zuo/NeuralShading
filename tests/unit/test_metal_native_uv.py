import json
import math

import numpy as np
import torch

from ncls.paths import PROJECT_ROOT
from ncls.learning.methods.metal.native_uv import (
    UVMapping, group_compatible_uv, native_hash22, native_slot_mappings, native_uv_lookups,
)


def test_only_same_uv_expression_is_fused_and_parameters_do_not_hide_differences():
    a = UVMapping("authored_uv", (1., 0., 0., 0., 1., 0.))
    b = UVMapping("independent_uv", a.affine)
    groups = group_compatible_uv({0: (a,), 1: (a,), 2: (b,)})
    assert [g.slots for g in groups] == [(0, 1), (2,)]


def test_native_hash_matches_independent_unsigned_scalar_oracle():
    def lowbias(x):
        x &= 0xFFFFFFFF
        x = ((x ^ (x >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
        x = ((x ^ (x >> 15)) * 0x846CA68B) & 0xFFFFFFFF
        return x ^ (x >> 16)
    cells = [[0, 0], [-1, 4], [935, 938], [1 << 28, -(1 << 27)]]
    def converted(value):
        return np.float32(np.float32(value & 0x7FFFFFFF) + (np.float32(2147483648) if value & 0x80000000 else np.float32(0))) / np.float32(4294967296)
    expected = [[converted(lowbias(x + lowbias(y))), converted(lowbias(x + 32000 + lowbias(y)))] for x, y in cells]
    torch.testing.assert_close(native_hash22(torch.tensor(cells)), torch.tensor(np.array(expected)), rtol=0, atol=0)


def test_tiling_uses_three_remote_coordinates_and_native_length_weight():
    mapping = UVMapping("native", (1., 0., 0., 0., 1., 0.), "nonrepeat", 2., 0.5, 935)
    uv = torch.tensor([[0.1, 0.05], [0.7, -0.2]])
    dx = torch.tensor([[0.001, 0.], [0.001, 0.]])
    coords, ddx, ddy, weights = native_uv_lookups(mapping, uv, dx, dx.flip(1))
    assert coords.shape == (2, 3, 2)
    assert torch.all(weights >= 0)
    torch.testing.assert_close(weights.square().sum(dim=1), torch.ones(2))
    assert torch.all(weights.sum(dim=1) > 1)
    torch.testing.assert_close(ddx, (dx * 0.5)[:, None].expand(-1, 3, -1))
    torch.testing.assert_close(ddy, ddx.flip(-1))
    assert not torch.equal(coords[:, 0], torch.remainder(uv, 1.0))


def test_diagnostic_mdl_mappings_are_source_backed():
    registry = json.loads((PROJECT_ROOT / "references/mdl-vmaterials2-v1/metal-opaque-v1.json").read_text(encoding="utf-8"))
    texture_sets = {value["id"]: value for value in registry["tables"]["texture_sets"]}
    module_root = PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0/Materials/vMaterials_2/Metal"
    if not module_root.is_dir():
        import pytest
        pytest.skip("native MDL asset installation is not present")
    for name in ("Tungsten", "Bronze_Scratched", "Steel_Painted_Cracked"):
        export = next(record for record in registry["opaque_exports"] if record["export_name"] == name)
        values = {parameter["name"]: parameter["value"] for parameter in export["parameters"]}
        spatial_slots = {slot["slot_index"] - 1: slot["source_path"]
                         for slot in texture_sets[export["texture_set_id"]]["slots"]
                         if slot["shape"] == "2d" and "color-lookup" not in slot["roles"]}
        mappings = native_slot_mappings(module_root / f"{name}.mdl", values, spatial_slots)
        groups = group_compatible_uv(mappings)
        assert groups and any(group.mapping.mode == "nonrepeat" for group in groups)
        if name == "Tungsten":
            assert len(groups) == 1
            assert groups[0].mapping.cell_scale == 8.
            assert groups[0].mapping.lookup_scale == 1.
        if name == "Bronze_Scratched":
            assert len(groups) == 2
            # dents 图使用 authored scale=0.5 的倒数；scratch/normal 保持原始 UV。
            assert mappings[1][0].affine == (2., 0., 0., -0., 2., 0.)
            assert mappings[2][0].affine == (1., 0., 0., 0., 1., 0.)
        if name == "Steel_Painted_Cracked":
            # tex_rescale=texture_scale*0.5，不能只读用户参数 texture_scale=1。
            assert mappings[2][0].mode == "direct" and mappings[2][0].affine[0] == 2.
            # met_norm 不在 infinite_tiling 条件分支内，还要除 damages_scale*0.5。
            assert mappings[3][0].mode == "direct" and mappings[3][0].affine[0] == 4.
