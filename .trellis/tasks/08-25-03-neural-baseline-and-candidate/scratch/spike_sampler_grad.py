from __future__ import annotations

import torch

from ncls.learning.models import UnifiedNeuralModel


top = {
    "interface_kind": 3, "alpha": [0.2, 0.2], "relative_ior": 1.0,
    "eta": [0.0, 0.0, 0.0], "k": [0.0, 0.0, 0.0],
    "color": [0.5, 0.5, 0.5], "tangent_rotation": 0.0,
}
model = UnifiedNeuralModel(
    state_count=1, response_scale=[[1.0, 1.0, 1.0]], top_rows=[top],
    evaluator="nvidia-frame-two-lobe-v1", runtime_class="realtime",
).cuda()
model.set_sampler_training("nvidia-diffuse-ggx9")
state = torch.zeros(1, dtype=torch.int64, device="cuda")
wo = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
wi = torch.tensor([[[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]]], device="cuda")
hidden = model.prepare_hidden(state, wo).detach()
print({
    "weight": model.nvidia_sampler_w.requires_grad,
    "weight_view": model.nvidia_sampler_w[:, None, :].requires_grad,
    "bias": model.nvidia_sampler_b.requires_grad,
    "hidden": hidden.requires_grad,
})
shared_head = model._linear("nclsUnifiedDot64", model.nvidia_sampler_w, model.nvidia_sampler_b, hidden)
head = model._linear("nclsUnifiedPaperDot64Out", model.nvidia_sampler_w, model.nvidia_sampler_b, hidden)
_ = model._linear(
    "nclsUnifiedPaperDot64A",
    model.nvidia_sampler_w.detach(),
    model.nvidia_sampler_b.detach(),
    hidden,
)
train_after_frozen = model._linear(
    "nclsUnifiedPaperDot64A",
    model.nvidia_sampler_w,
    model.nvidia_sampler_b,
    hidden,
)
train_first = model._linear(
    "nclsUnifiedPaperDot64B",
    model.nvidia_sampler_w,
    model.nvidia_sampler_b,
    hidden,
)
frozen_after_train = model._linear(
    "nclsUnifiedPaperDot64B",
    model.nvidia_sampler_w.detach(),
    model.nvidia_sampler_b.detach(),
    hidden,
)
prepared = model.session.module.nclsUnifiedJoinNvidiaState(torch.zeros((1, 14), device="cuda"), head)
payload = prepared[:, None, :].expand(-1, 2, -1).contiguous()
views = wo[:, None, :].expand(-1, 2, -1).contiguous()
pdf = model.session.module.nclsUnifiedNvidiaPreparedPdfTraining(payload, views, wi)
print({
    "shared_head": shared_head.requires_grad,
    "unique_head": head.requires_grad,
    "train_after_frozen": train_after_frozen.requires_grad,
    "train_first": train_first.requires_grad,
    "frozen_after_train": frozen_after_train.requires_grad,
    "prepared": prepared.requires_grad,
    "pdf": pdf.requires_grad,
})
pdf.mean().backward()
print({"grad": model.nvidia_sampler_w.grad is not None, "finite": bool(torch.isfinite(model.nvidia_sampler_w.grad).all())})
