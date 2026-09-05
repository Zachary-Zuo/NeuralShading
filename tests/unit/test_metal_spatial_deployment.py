from types import SimpleNamespace

import torch

from ncls.learning.methods.metal.method import METHOD
from ncls.learning.methods.metal.model import METAL_BUDGETED_REQUIRED_CONTEXT
from ncls.learning.methods.metal.native_uv import UVGroup, UVMapping
from ncls.learning.methods.metal.spatial_cook import _read
from ncls.learning.methods.metal.spatial_encoder import RawSlot
from ncls.learning.methods.metal.spatial_runtime import SPATIAL_COMPILED_WORD_COUNT, pack_spatial_compiled_material, spatial_material_payload
from tests.unit.test_mdl_fixed_source_adapter import _snapshot, _arguments
from tests.unit.test_metal_budgeted_method import _values


def test_unseen_snapshot_cooks_with_frozen_model_without_asset_id_learning(monkeypatch):
    import ncls.learning.methods.metal.method as module
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    raw = torch.linspace(0.1,0.9,16*16).reshape(1,1,16,16)
    slots = (RawSlot(0,(16,16),("roughness",)),)
    groups = (UVGroup(UVMapping("source",(1.,0.,0.,0.,1.,0.)),(0,)),)
    class Adapter:
        def __init__(self, snapshots, device):
            self.device = device
        def compiler_tensors_for_source(self, index, device):
            return _values(1)
        def spatial_contract_for_source(self, index):
            return 0,slots,groups
        def native_assets(self):
            return SimpleNamespace(collection_id="fixture", descriptors=(SimpleNamespace(asset_id="raw",schema_id="roughness"),),
                read_raw_tile=lambda asset,slot,rect: _read(raw,rect,"wrap"))
        def close(self):
            pass
    monkeypatch.setattr(module,"MetalBudgetedMdlSourceAdapter",Adapter)
    torch.manual_seed(124)
    model = METHOD.create_trainable(METAL_BUDGETED_REQUIRED_CONTEXT)
    checkpoint = {"model_state": METHOD.export_training_state(model), "source_snapshot_ids": ["a"*64],
                  "training_config": {"model_context": METAL_BUDGETED_REQUIRED_CONTEXT}}
    before = {name: value.clone() for name,value in checkpoint["model_state"].items()}
    source = _snapshot(_arguments())
    assert source.snapshot_id not in checkpoint["source_snapshot_ids"]
    deployment = METHOD._deployment(source,checkpoint)
    compiled = pack_spatial_compiled_material(deployment["program_state"],deployment["asset"])
    assert len(compiled) == SPATIAL_COMPILED_WORD_COUNT*4
    assert deployment["asset"].texture_reads == 2
    payload = spatial_material_payload(source.snapshot_id,deployment["asset"])
    assert len(payload.resources) == 18
    assert all("variant_scale_bias" not in name for name in before)
    for name,value in checkpoint["model_state"].items():
        torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    assert all(parameter.grad is None for parameter in deployment["model"].parameters())
