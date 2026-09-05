import pytest
import torch
from torch.nn import functional as F

from ncls.learning.methods.metal.asset_read import mip_shapes
from ncls.learning.methods.metal.spatial_encoder import (
    MetalSpatialEncoder, RawSlot, SEMANTICS, build_encoding_plan, semantic_group,
)


def _raw(plan, sources):
    return {f"raw-{index}": sources[plan.slots[read.slot].slot][..., read.rect[0]:read.rect[0] + read.rect[2],
                                                               read.rect[1]:read.rect[1] + read.rect[3]]
            for index, read in enumerate(plan.raw_reads)}


def _conv_full(value, layer, mode):
    size, stride = layer.kernel_size[0], layer.stride[0]
    if stride == 1:
        # index_select 支持 1×N 与长度大于轴的 repeat padding。
        y = torch.arange(-1, value.shape[-2] + 1)
        x = torch.arange(-1, value.shape[-1] + 1)
    else:
        y = torch.arange(2 * max(1, value.shape[-2] // 2))
        x = torch.arange(2 * max(1, value.shape[-1] // 2))
    if mode == "wrap":
        y, x = y % value.shape[-2], x % value.shape[-1]
    else:
        y, x = y.clamp(0, value.shape[-2] - 1), x.clamp(0, value.shape[-1] - 1)
    return F.silu(layer(value.index_select(-2, y).index_select(-1, x)))


@pytest.mark.parametrize("shape", [(17, 23), (1, 19), (21, 1), (1, 1)])
@pytest.mark.parametrize("mode", ["wrap", "clamp"])
def test_tiled_hierarchy_matches_independent_full_field(shape, mode):
    torch.manual_seed(92)
    encoder = MetalSpatialEncoder()
    slot = RawSlot(2, shape, ("roughness",), mode)
    source = torch.rand(1, 1, *shape)
    shapes = mip_shapes(*shape)
    context_shapes = mip_shapes(*shapes[min(2, len(shapes) - 1)])
    plan = build_encoding_plan((slot,), shape, mode,
                               [(0, 0, *size) for size in shapes],
                               [(0, 0, *size) for size in context_shapes])
    detail, context = encoder.encode_tiles(plan, _raw(plan, {2: source}))
    value = torch.cat((source, torch.zeros(1, 3, *shape), torch.ones_like(source), torch.zeros(1, 3, *shape)), dim=1)
    for layer in encoder.stems["scalar"]:
        value = _conv_full(value, layer, mode)
    semantic = encoder.semantic_embedding.weight[SEMANTICS.index("roughness")]
    slots = [torch.zeros(1, 25, *shape) for _ in range(9)]
    slots[2] = torch.cat((value, semantic[None, :, None, None].expand(1, -1, *shape), torch.ones(1, 1, *shape)), dim=1)
    value = F.silu(encoder.fusion(torch.cat(slots, dim=1)))
    for layer in encoder.trunk:
        value = _conv_full(value, layer, mode)
    hierarchy = [value]
    for _ in shapes[1:]:
        for layer in encoder.mip:
            value = _conv_full(value, layer, mode)
        hierarchy.append(value)
    for index, tile in enumerate(detail):
        torch.testing.assert_close(tile.values, torch.tanh(encoder.detail_head(hierarchy[index])), rtol=3e-5, atol=1e-7)
    for index, tile in enumerate(context):
        torch.testing.assert_close(tile.values, torch.tanh(encoder.context_head(hierarchy[min(index + 2, len(shapes) - 1)])), rtol=3e-5, atol=1e-7)


def test_crop_seam_and_changed_slot_resolution_preserve_native_stem():
    torch.manual_seed(73)
    encoder = MetalSpatialEncoder()
    shape = (32, 48)
    slots = (RawSlot(0, shape, ("base-color",) * 3),
             RawSlot(1, (16, 24), ("normal-tangent",) * 3, affine=(0.5, -0.2, 0.4, 0.0, 1.0, 0.1)))
    sources = {0: torch.rand(1, 3, *shape), 1: torch.rand(1, 3, 16, 24)}
    full = build_encoding_plan(slots, shape, "wrap", [(0, 0, *shape)], [(0, 0, 8, 12)])
    a, b = encoder.encode_tiles(full, _raw(full, sources))
    crop = build_encoding_plan(slots, shape, "wrap", [(-2, 43, 8, 9)], [(-1, 10, 4, 5)])
    c, d = encoder.encode_tiles(crop, _raw(crop, sources))
    for complete, tile in ((a[0], c[0]), (b[0], d[0])):
        y, x = tile.origin_yx
        yy = torch.arange(y, y + tile.values.shape[-2]) % complete.level_shape[0]
        xx = torch.arange(x, x + tile.values.shape[-1]) % complete.level_shape[1]
        expected = complete.values.index_select(-2, yy).index_select(-1, xx)
        torch.testing.assert_close(tile.values, expected, rtol=3e-5, atol=1e-7)
    loss = c[0].values.square().sum() + d[0].values.square().sum()
    loss.backward()
    for name in ("stems.color.0.weight", "stems.normal.0.weight", "fusion.weight", "trunk.0.weight", "mip.0.weight", "detail_head.weight", "context_head.weight"):
        gradient = dict(encoder.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all("variant" not in name for name, _ in encoder.named_parameters())


def test_roles_are_explicit_and_lookup_slices_are_not_surface_uv():
    assert semantic_group(("dirt", "nrm", "wash")) == "packed"
    assert semantic_group(("height",) * 3) == "height"
    with pytest.raises(ValueError, match="unsupported declared"):
        semantic_group(("some-new-roughness-name",))
    encoder = MetalSpatialEncoder()
    slots = (RawSlot(0, (3, 5), ("color-lookup",) * 3, "clamp", spatial=False, depth=2),)
    plan = build_encoding_plan(slots, (7, 9), "wrap", [(0, 0, 7, 9)], [])
    detail, _ = encoder.encode_tiles(plan, _raw(plan, {0: torch.rand(2, 3, 3, 5)}))
    torch.testing.assert_close(detail[0].values, detail[0].values[..., :1, :1].expand_as(detail[0].values))
    assert encoder.receptive_field(0) == 9 and encoder.receptive_field(4) == 84
