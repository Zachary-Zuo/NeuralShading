from __future__ import annotations

import json
from pathlib import Path

import torch

from ncls.learning.models import UnifiedNeuralModel


top = {
    "interface_kind": 3,
    "alpha": [0.2, 0.2],
    "relative_ior": 1.0,
    "eta": [0.0, 0.0, 0.0],
    "k": [0.0, 0.0, 0.0],
    "color": [0.5, 0.5, 0.5],
    "tangent_rotation": 0.0,
}
model = UnifiedNeuralModel(
    state_count=1,
    response_scale=[[1.0, 1.0, 1.0]],
    top_rows=[top],
    evaluator="nvidia-frame-two-lobe-v1",
    runtime_class="diagnostic",
).cuda()
state = torch.zeros(1, dtype=torch.int64, device="cuda")
wo = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
wi = torch.tensor([[[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]]], device="cuda")
prediction = model(state, wo, wi)
loss = prediction.square().mean()
loss.backward()
payload = {
    "contract": "ncls.unified-paper-backward-spike@1",
    "prediction_finite": bool(torch.isfinite(prediction).all()),
    "loss": float(loss),
    "gradient_finite": all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ),
}
Path("artifacts/spikes/unified-scattering-03-paper-backward-v1.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload))
