import numpy as np
import pytest
import torch

from ncls.learning.methods.metal.asset_read import mip_shapes
from ncls.learning.methods.metal.spatial_cook import SpatialHierarchyCooker
from ncls.learning.methods.metal.spatial_encoder import MetalSpatialEncoder, RawSlot, build_encoding_plan
from tests.unit.test_metal_spatial_encoder import _raw


@pytest.mark.parametrize("shape", [(17, 23), (1, 19), (21, 1), (1, 1)])
@pytest.mark.parametrize("address", ["wrap", "clamp"])
def test_streamed_cook_matches_training_hierarchy_including_odd_mip_tails(shape, address):
    torch.manual_seed(812)
    encoder = MetalSpatialEncoder()
    slots = (RawSlot(0, shape, ("roughness",), address),
             RawSlot(3, (3, 7), ("normal-tangent",)*3, address))
    sources = {slot.slot: torch.rand(slot.depth, len(slot.channel_roles), *slot.shape) for slot in slots}
    def load(slot, rect):
        y, x, h, w = rect
        yy, xx = torch.arange(y, y+h), torch.arange(x, x+w)
        if address == "wrap":
            yy, xx = yy % slot.shape[0], xx % slot.shape[1]
        else:
            yy, xx = yy.clamp(0, slot.shape[0]-1), xx.clamp(0, slot.shape[1]-1)
        return sources[slot.slot].index_select(-2, yy).index_select(-1, xx)
    shape = (max(s.shape[0] for s in slots), max(s.shape[1] for s in slots))
    shapes = mip_shapes(*shape)
    contexts = mip_shapes(*shapes[min(2, len(shapes)-1)])
    plan = build_encoding_plan(slots, shape, address, [(0, 0, *s) for s in shapes], [(0, 0, *s) for s in contexts])
    with torch.no_grad():
        expected = encoder.encode_tiles(plan, _raw(plan, sources))
    cooker = SpatialHierarchyCooker(encoder, tile_size=5)
    actual = cooker.encode(slots, shape, address, load, quantized=False)
    for result_levels, expected_levels in zip(actual, expected, strict=True):
        for result, tile in zip(result_levels, expected_levels, strict=True):
            torch.testing.assert_close(torch.from_numpy(result).permute(2, 0, 1)[None], tile.values, rtol=3e-5, atol=1e-7)
    assert all(value.grad is None for value in encoder.parameters())


def test_streamed_global_lookup_keeps_native_slices_and_budget_fails_before_loading():
    encoder = MetalSpatialEncoder()
    slots = (RawSlot(1, (3, 5), ("color-lookup",)*3, "clamp", spatial=False, depth=3),)
    source = torch.arange(135).float().reshape(3, 3, 3, 5)/100
    def load(slot, rect):
        y, x, h, w = rect
        return source.index_select(-2, torch.arange(y, y+h).clamp(0, 2)).index_select(-1, torch.arange(x, x+w).clamp(0, 4))
    plan = build_encoding_plan(slots, (1, 1), "clamp", [(0, 0, 1, 1)], [(0, 0, 1, 1)])
    expected = encoder.encode_tiles(plan, _raw(plan, {1: source}))
    result = SpatialHierarchyCooker(encoder, tile_size=2).encode(slots, (1, 1), "clamp", load, quantized=False)
    for levels, reference in zip(result, expected, strict=True):
        np.testing.assert_allclose(levels[0].ravel(), reference[0].values.detach().numpy().ravel(), rtol=3e-5, atol=1e-7)
    with pytest.raises(RuntimeError, match="host staging"):
        SpatialHierarchyCooker(encoder, host_budget_bytes=1).encode(slots, (1, 1), "clamp", lambda *_: pytest.fail("must reject before raw read"))
