from types import SimpleNamespace

import pytest
import torch

from ncls.learning.conditioning_resources import ConditioningResource
from ncls.learning.methods.metal.data import MetalBudgetedMdlSourceAdapter
from ncls.learning.methods.metal.native_uv import UVGroup, UVMapping
from ncls.learning.methods.metal.spatial_asset import MetalSpatialAsset
from ncls.learning.methods.metal.spatial_encoder import RawSlot
from tests.unit.test_metal_spatial_asset import _program


class _Assets:
    collection_id = "raw-fixture"

    def __init__(self):
        self.releases = 0

    def release(self):
        self.releases += 1

    def acquire_spatial_bundle(self, asset_index, plan):
        assert asset_index == 7
        raw = {}
        for index, part in enumerate(plan.parts):
            for read_id, read in enumerate(part.plan.raw_reads):
                y, x, h, w = read.rect
                yy, xx = torch.meshgrid(torch.arange(y, y+h), torch.arange(x, x+w), indexing="ij")
                raw[f"part-{index}/raw-{read_id}"] = (0.1 + (xx + 2*yy).float() / 100)[None, None]
        return ConditioningResource("fixture", raw, {"bundle": plan}, self)


def _adapter():
    adapter = object.__new__(MetalBudgetedMdlSourceAdapter)
    adapter.device = torch.device("cpu")
    adapter.snapshots = (None, None)
    adapter.registry = SimpleNamespace(identity="registry")
    adapter._assets = _Assets()
    adapter._tables = {name: torch.tensor([11, 29]) for name in (
        "graph", "schema", "recipe", "metal", "finish", "asset", "semantic", "type",
        "responsibility", "discrete", "continuous", "presence", "optical", "access", "frame", "distribution")}
    slots = (RawSlot(0, (32, 16), ("roughness",)),)
    groups = (UVGroup(UVMapping("native", (2., 0., 0., 0., 1., 0.), "nonrepeat", 8., 1.), (0,)),)
    adapter._spatial_contracts = ((7, slots, groups),) * 2
    adapter._spatial_cohort_keys = ("same", "same")
    adapter._spatial_tile_schedules = {}
    return adapter


def test_raw_adapter_uses_shared_resources_and_preserves_reordered_source_rows():
    adapter = _adapter()
    options = {"paired_uv": True, "paired_uv_recipe": "one-native-texel-axis-balanced@1",
               "logical_request_index": 7, "spatial_core_texels": 2}
    rows = torch.tensor([1, 0, 0, 1])
    def sample():
        return adapter.sample_tensors(rows, torch.Generator().manual_seed(13), options,
                                      execution_source_indices=(0, 1))
    first, restored = sample(), sample()
    assert len(first.resources) == 1
    assert not any("patch" in key for key in first.tensors)
    torch.testing.assert_close(first.tensors["metal_identity_index"], torch.tensor([29, 11, 11, 29]))
    for key in ("uv", "paired_uv", "filter_random", "uv_dx", "uv_dy"):
        torch.testing.assert_close(first.tensors[key], restored.tensors[key], rtol=0, atol=0)
    assert first.provenance == restored.provenance
    model = MetalSpatialAsset()
    encoded = model.encode_resources(first.resources)
    main = model(first.tensors, _program(4), resources=first.resources, binding=first.bindings["metal_spatial"], encoded=encoded)
    pair = model({**first.tensors, "uv": first.tensors["paired_uv"]}, _program(4),
                 resources=first.resources, binding=first.bindings["metal_spatial"], encoded=encoded)
    assert main.valid.all() and pair.valid.all()
    (main.group_latent - pair.group_latent).square().sum().backward()
    assert model.encoder.stems["scalar"][0].weight.grad.abs().sum() > 0
    first.resources.release()
    restored.resources.release()
    assert adapter._assets.releases == 2


def test_raw_adapter_rejects_mixed_cpu_cohort_before_acquiring_resources():
    adapter = _adapter()
    adapter._spatial_cohort_keys = ("first", "second")
    with pytest.raises(ValueError, match="CPU-declared"):
        adapter.sample_tensors(torch.tensor([0, 1]), torch.Generator(), {}, execution_source_indices=(0, 1))
    assert adapter._assets.releases == 0
