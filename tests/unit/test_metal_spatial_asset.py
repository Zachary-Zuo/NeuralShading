import torch

from ncls.learning.conditioning_resources import ConditioningResource, ConditioningResources
from ncls.learning.methods.metal.compiler import MetalBudgetedProgramState
from ncls.learning.methods.metal.native_uv import UVGroup, UVMapping
from ncls.learning.methods.metal.spatial_asset import MetalSpatialAsset
from ncls.learning.methods.metal.spatial_bundle import build_spatial_bundle
from ncls.learning.methods.metal.spatial_encoder import RawSlot


def _resource(bundle, sources):
    raw = {}
    for index, part in enumerate(bundle.parts):
        for read_id, read in enumerate(part.plan.raw_reads):
            slot = part.plan.slots[read.slot]
            y, x, h, w = read.rect
            raw[f"part-{index}/raw-{read_id}"] = sources[slot.slot][..., y:y+h, x:x+w]
    return ConditioningResources((ConditioningResource("fixture", raw, {"bundle": bundle}),))


def _program(batch):
    z = torch.zeros(batch, 8)
    return MetalBudgetedProgramState(z, z.clone(), z.clone(), z.clone(), torch.ones(batch, 3) / 3,
                                     torch.zeros(batch, dtype=torch.int64), torch.zeros(batch, 8, dtype=torch.int64),
                                     torch.zeros(batch, 16), z.clone(), {})


def test_uv_groups_remain_separate_and_main_pair_share_one_current_graph():
    torch.manual_seed(112)
    slots = (RawSlot(0, (32, 32), ("roughness",)), RawSlot(1, (16, 16), ("normal-tangent",) * 3))
    groups = (UVGroup(UVMapping("uv_a", (1., 0., 0., 0., 1., 0.)), (0,)),
              UVGroup(UVMapping("uv_b", (0., 0.5, 0.2, 2., 0., -0.1)), (1,)))
    bundle = build_spatial_bundle(slots, groups, (0.1, 0.2, 0.4, 0.5), (0.01, 0.), (0., 0.01))
    assert all(len(part.plan.slots) == 1 for part in bundle.parts)
    sources = {0: torch.rand(1, 1, 32, 32), 1: torch.rand(1, 3, 16, 16)}
    resource = _resource(bundle, sources)
    model = MetalSpatialAsset()
    uv = torch.tensor([[0.2, 0.3], [0.3, 0.4]])
    values = {"uv": uv, "uv_dx": torch.zeros_like(uv), "uv_dy": torch.zeros_like(uv), "filter_random": torch.tensor([0.3, 0.9])}
    binding = torch.zeros(2, dtype=torch.int64)
    encoding = model.encode_resources(resource)
    primary = model(values, _program(2), resources=resource, binding=binding, encoded=encoding)
    pair = model({**values, "uv": uv + 0.002}, _program(2), resources=resource, binding=binding, encoded=encoding)
    assert primary.group_features.shape == (2, 9, 14)
    assert primary.group_latent.shape == (2, 9, 8)
    assert torch.all(primary.group_features[:, :2, -1] == 1)
    assert torch.all(primary.group_features[:, 2:] == 0)
    assert not torch.equal(primary.group_latent, pair.group_latent)
    loss = (pair.group_latent - primary.group_latent).square().sum()
    loss.backward()
    assert model.encoder.stems["scalar"][0].weight.grad.abs().sum() > 0
    assert model.encoder.stems["normal"][0].weight.grad.abs().sum() > 0
    before = primary.group_latent.detach().clone()
    with torch.no_grad():
        model.encoder.detail_head.bias.add_(0.05)
    fresh = model(values, _program(2), resources=resource, binding=binding)
    assert not torch.equal(before, fresh.group_latent)
    resource.release()


def test_nonrepeat_bundle_contains_remote_raw_inputs_and_keeps_lookup_global():
    slots = (RawSlot(0, (16, 16), ("roughness",)),
             RawSlot(2, (3, 8), ("color-lookup",) * 3, "clamp", spatial=False))
    groups = (UVGroup(UVMapping("native-hex", (1., 0., 0., 0., 1., 0.), "nonrepeat", 8., 1.), (0,)),)
    bundle = build_spatial_bundle(slots, groups, (0.2, 0.2, 0.22, 0.22), (0.01, 0.), (0., 0.01))
    assert sum(part.group == -1 for part in bundle.parts) == 1
    assert len({part.cell for part in bundle.parts if part.group == 0}) >= 4
    resources = _resource(bundle, {0: torch.rand(1, 1, 16, 16), 2: torch.rand(1, 3, 3, 8)})
    values = {"uv": torch.tensor([[0.21, 0.21]]), "uv_dx": torch.zeros(1, 2), "uv_dy": torch.zeros(1, 2), "filter_random": torch.tensor([0.3])}
    result = MetalSpatialAsset()(values, _program(1), resources=resources, binding=torch.zeros(1, dtype=torch.int64))
    assert result.valid.all() and torch.isfinite(result.global_condition).all()
    assert result.global_condition.shape == (1, 8)
    resources.release()
