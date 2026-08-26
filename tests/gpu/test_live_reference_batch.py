from __future__ import annotations

import ast
import inspect

import pytest
import torch

pytest.importorskip("falcor")

from ncls.core.material import DiffuseInterface, LayerStackIR
from ncls.data.batch_sources import LiveReferenceBatchSource
from ncls.source_materials.families.layer_stack import snapshot_from_layer_stack


@pytest.mark.falcor
def test_live_reference_batch_stays_on_cuda_and_enforces_lease():
    source_text = inspect.getsource(LiveReferenceBatchSource)
    assert not any(isinstance(node, ast.Attribute) and node.attr == "to_numpy" for node in ast.walk(ast.parse(source_text)))
    stack = LayerStackIR((DiffuseInterface((0.6, 0.3, 0.1)),), ())
    snapshot = snapshot_from_layer_stack(stack)
    source = LiveReferenceBatchSource((stack,), (snapshot.snapshot_id,), light_count=8,
                                      samples_per_replica=2, max_batch_size=2, seed=17)
    try:
        batch = source.next_batch(2)
        assert batch.device.type == "cuda"
        assert all(value.is_cuda for value in batch.tensors.values())
        assert batch.provenance["host_readback"] is False
        parameter = torch.nn.Parameter(torch.zeros_like(batch.tensors["target"]))
        torch.nn.functional.mse_loss(parameter, batch.tensors["target"]).backward()
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        with pytest.raises(RuntimeError):
            source.next_batch(1)
        batch.release()
        source.next_batch(1).release()
    finally:
        source.close()
