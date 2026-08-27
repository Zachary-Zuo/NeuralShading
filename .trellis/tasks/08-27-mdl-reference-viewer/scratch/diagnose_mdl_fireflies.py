from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from ncls.data.providers.mdl import MdlGpuQueryRuntime
from ncls.paths import PROJECT_ROOT
from ncls.references.mdl import MDL_SDK_DIRECTORY, MdlCompiledArtifact


def _directions(count: int, device: torch.device) -> torch.Tensor:
    z = torch.rand(count, device=device)
    phi = 2.0 * math.pi * torch.rand(count, device=device)
    radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    return torch.stack((radius * torch.cos(phi), radius * torch.sin(phi), z), dim=1)


def _generic_pdf(views: torch.Tensor, lights: torch.Tensor) -> torch.Tensor:
    alpha = 0.2
    half_vector = torch.nn.functional.normalize(views + lights, dim=1)
    projected = alpha * alpha * (views[:, 0] ** 2 + views[:, 1] ** 2)
    lambda_view = 0.5 * (
        torch.sqrt(1.0 + projected / torch.clamp(views[:, 2] ** 2, min=1e-20)) - 1.0
    )
    g1 = 1.0 / (1.0 + lambda_view)
    wh = torch.sum(views * half_vector, dim=1).abs()
    hx = half_vector[:, 0] / alpha
    hy = half_vector[:, 1] / alpha
    denominator = hx * hx + hy * hy + half_vector[:, 2] ** 2
    distribution = 1.0 / (math.pi * alpha * alpha * denominator * denominator)
    visible_normal_pdf = g1 * wh * distribution / torch.clamp(views[:, 2].abs(), min=1e-8)
    specular_pdf = visible_normal_pdf / torch.clamp(4.0 * wh, min=1e-8)
    cosine_pdf = lights[:, 2] / math.pi
    return 0.5 * (cosine_pdf + specular_pdf)


def main() -> None:
    catalog = json.loads(
        (PROJECT_ROOT / "build/mdl-reference/viewer/catalog.json").read_text(encoding="utf-8")
    )
    device = torch.device("cuda:0")
    torch.manual_seed(0x4D444C46)
    batch_size = 65_536
    batch_count = 8
    for asset_id in ("carpaint-shifting-flakes", "ceramic-tiles-glazed-versailles"):
        entry = next(item for item in catalog["assets"] if item["asset_id"] == asset_id)
        artifact = MdlCompiledArtifact.load(Path(entry["artifact_root"]))
        runtime = MdlGpuQueryRuntime(
            artifact,
            sdk_root=PROJECT_ROOT / "external" / MDL_SDK_DIRECTORY,
            query_capacity=batch_size,
            slot_count=2,
        )
        generic_weights: list[torch.Tensor] = []
        mdl_weights: list[torch.Tensor] = []
        response_values: list[torch.Tensor] = []
        try:
            for _ in range(batch_count):
                views = _directions(batch_size, device)
                lights = _directions(batch_size, device)
                uv = torch.rand((batch_size, 2), device=device)
                gradients = torch.zeros((batch_size, 4), device=device)
                response, mdl_pdf = runtime.evaluate_torch(0, views, lights, uv, gradients)
                response_max = response.max(dim=1).values
                generic_pdf = _generic_pdf(views, lights)
                generic_weights.append((response_max / torch.clamp(generic_pdf, min=1e-8)).cpu())
                mdl_weights.append((response_max / torch.clamp(mdl_pdf, min=1e-8)).cpu())
                response_values.append(response_max.cpu())
            runtime._device.end_frame()
        finally:
            runtime.close()
        generic = torch.cat(generic_weights)
        matched = torch.cat(mdl_weights)
        response = torch.cat(response_values)
        print(
            json.dumps(
                {
                    "asset_id": asset_id,
                    "queries": int(generic.numel()),
                    "response_max": float(response.max()),
                    "generic_weight_p9999": float(torch.quantile(generic, 0.9999)),
                    "generic_weight_max": float(generic.max()),
                    "mdl_pdf_weight_p9999": float(torch.quantile(matched, 0.9999)),
                    "mdl_pdf_weight_max": float(matched.max()),
                    "generic_weight_over_100": int((generic > 100.0).sum()),
                    "mdl_pdf_weight_over_100": int((matched > 100.0).sum()),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
