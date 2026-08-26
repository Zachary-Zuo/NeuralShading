from __future__ import annotations

import json

import torch

from ncls.core.material import DiffuseInterface, LayerStackIR
from ncls.data.batch_sources import LiveReferenceBatchSource
from ncls.source_materials.families.layer_stack import snapshot_from_layer_stack


def main() -> None:
    stack = LayerStackIR((DiffuseInterface((0.6, 0.3, 0.1)),), ())
    snapshot = snapshot_from_layer_stack(stack)
    source = LiveReferenceBatchSource(
        (stack,), (snapshot.snapshot_id,), light_count=8,
        samples_per_replica=2, max_batch_size=2, seed=17,
    )
    batch = source.next_batch(2)
    target = batch.tensors["target"]
    assert target.is_cuda and target.shape == (2, 8, 3)
    assert batch.provenance["host_readback"] is False
    parameter = torch.nn.Parameter(torch.zeros_like(target))
    loss = torch.nn.functional.mse_loss(parameter, target)
    loss.backward()
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    blocked = False
    try:
        source.next_batch(1)
    except RuntimeError:
        blocked = True
    assert blocked
    batch.release()
    second = source.next_batch(1)
    second.release()
    source.close()
    print(json.dumps({"device": str(target.device), "finite": bool(torch.isfinite(target).all()), "lease_blocked": blocked}))


if __name__ == "__main__":
    main()
