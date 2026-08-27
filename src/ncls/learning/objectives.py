from __future__ import annotations

import torch


def sampler_forward_kl_score(
    evaluator_response_cos: torch.Tensor,
    proposal_pdf: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """论文 sampler 自采样的 forward-KL score-function Monte Carlo estimator。"""

    luminance = (
        0.2126 * evaluator_response_cos[..., 0]
        + 0.7152 * evaluator_response_cos[..., 1]
        + 0.0722 * evaluator_response_cos[..., 2]
    )
    safe_pdf = torch.clamp(proposal_pdf, min=1e-12)
    target = torch.clamp(luminance, min=0.0).detach()
    integrand = -target * torch.log(safe_pdf) / safe_pdf.detach()
    finite = valid & torch.isfinite(integrand) & (proposal_pdf > 0.0)
    loss = torch.where(finite, integrand, torch.zeros_like(integrand)).mean()
    return loss, finite.to(torch.float32).mean()


__all__ = ["sampler_forward_kl_score"]
