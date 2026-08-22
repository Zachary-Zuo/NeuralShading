from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baselines.closure_families import _hemisphere_axis, _inverse_softplus, directional_smape
from baselines.oracle_fit import _label_groups, _summary, load_oracle_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SharedSgDictionary(nn.Module):
    def __init__(
        self,
        target: torch.Tensor,
        atom_count: int,
        *,
        seed: int,
        reflection_frame: bool = False,
    ) -> None:
        super().__init__()
        tile_count = len(target)
        generator = torch.Generator(device=target.device)
        generator.manual_seed(seed)
        index = torch.arange(atom_count, device=target.device, dtype=target.dtype)
        z = (index + 0.5) / atom_count
        phi = index * (math.pi * (3.0 - math.sqrt(5.0)))
        radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        axis = torch.stack((radius * torch.cos(phi), radius * torch.sin(phi), z), dim=-1)
        raw_axis = torch.cat((axis[:, :2], _inverse_softplus(axis[:, 2:3])), dim=-1)
        raw_axis += 0.01 * torch.randn(raw_axis.shape, generator=generator, device=target.device)
        self.raw_axis = nn.Parameter(raw_axis)
        sharpness = torch.logspace(
            math.log10(0.5), math.log10(128.0), atom_count, device=target.device
        )
        self.log_sharpness = nn.Parameter(torch.log(sharpness))
        self.reflection_frame = reflection_frame
        if reflection_frame:
            initial_blend = torch.cat(
                (
                    torch.full((atom_count // 3,), 0.05, device=target.device),
                    torch.full((atom_count - atom_count // 3,), 0.95, device=target.device),
                )
            )
            self.raw_frame_blend = nn.Parameter(torch.logit(initial_blend))
        mean = torch.mean(target, dim=1).clamp_min(1e-4)
        initial_amplitude = (0.05 * mean[:, None, :]).repeat(1, atom_count, 1)
        self.raw_amplitude = nn.Parameter(_inverse_softplus(initial_amplitude))

    def basis(self, lights: torch.Tensor, views: torch.Tensor | None = None) -> torch.Tensor:
        local_axis = _hemisphere_axis(self.raw_axis)
        sharpness = torch.exp(torch.clamp(self.log_sharpness, math.log(0.05), math.log(1024.0)))
        if not self.reflection_frame:
            cosine = torch.einsum("mc,bc->mb", local_axis, lights)
            return torch.exp(sharpness[:, None] * (cosine - 1.0))
        if views is None:
            raise ValueError("views are required for a reflection-frame dictionary")
        reflection = torch.stack((-views[:, 0], -views[:, 1], views[:, 2]), dim=-1)
        tangent_xy = torch.stack((-reflection[:, 1], reflection[:, 0]), dim=-1)
        tangent_length = torch.linalg.vector_norm(tangent_xy, dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0], device=views.device, dtype=views.dtype)
        tangent_xy = torch.where(tangent_length > 1e-6, tangent_xy / tangent_length.clamp_min(1e-6), fallback)
        tangent = torch.cat((tangent_xy, torch.zeros_like(tangent_xy[:, :1])), dim=-1)
        bitangent = torch.linalg.cross(reflection, tangent)
        reflected_axis = (
            local_axis[None, :, 0:1] * tangent[:, None, :]
            + local_axis[None, :, 1:2] * bitangent[:, None, :]
            + local_axis[None, :, 2:3] * reflection[:, None, :]
        )
        frame_blend = torch.sigmoid(self.raw_frame_blend)[None, :, None]
        axis = (1.0 - frame_blend) * local_axis[None, :, :] + frame_blend * reflected_axis
        axis = torch.cat((axis[..., :2], torch.clamp(axis[..., 2:3], min=1e-4)), dim=-1)
        axis = F.normalize(axis, dim=-1)
        cosine = torch.einsum("tmc,bc->tmb", axis, lights)
        return torch.exp(sharpness[None, :, None] * (cosine - 1.0))

    def prediction(self, basis: torch.Tensor, tile_indices: torch.Tensor) -> torch.Tensor:
        amplitude = F.softplus(self.raw_amplitude[tile_indices])
        if basis.ndim == 2:
            return torch.einsum("mb,tmc->tbc", basis, amplitude)
        return torch.einsum("tmb,tmc->tbc", basis[tile_indices], amplitude)


def _log_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    floor = 1e-3
    delta = torch.log(prediction + floor) - torch.log(target + floor)
    return torch.mean(delta * delta) + 0.05 * torch.mean(torch.abs(prediction - target))


def fit_dictionary(
    dataset_dir: Path,
    output_dir: Path,
    *,
    atom_count: int,
    steps: int,
    restarts: int,
    learning_rate: float,
    device: str | None,
    seed: int,
    reflection_frame: bool = False,
) -> dict[str, object]:
    dataset = load_oracle_dataset(dataset_dir)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    target = torch.as_tensor(dataset.target, dtype=torch.float32, device=torch_device)
    lights = torch.as_tensor(dataset.lights, dtype=torch.float32, device=torch_device)
    views = torch.as_tensor(dataset.views[:, :3], dtype=torch.float32, device=torch_device)
    scale = torch.amax(target, dim=(1, 2), keepdim=True).clamp_min(1e-4)
    normalized_target = target / scale
    train_indices = torch.as_tensor(np.flatnonzero(dataset.splits == 0), device=torch_device)
    heldout_indices = torch.as_tensor(np.flatnonzero(dataset.splits != 0), device=torch_device)

    best_train_loss = math.inf
    best_state: dict[str, np.ndarray] = {}
    start = time.perf_counter()
    for restart in range(restarts):
        module = SharedSgDictionary(
            normalized_target,
            atom_count,
            seed=seed + restart * 1009,
            reflection_frame=reflection_frame,
        ).to(torch_device)
        optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            basis = module.basis(lights, views)
            train_prediction = module.prediction(basis, train_indices)
            if basis.ndim == 3:
                heldout_basis = basis.detach()
            else:
                heldout_basis = basis.detach()
            heldout_prediction = module.prediction(heldout_basis, heldout_indices)
            loss = _log_loss(train_prediction, normalized_target[train_indices])
            if len(heldout_indices):
                loss = loss + _log_loss(heldout_prediction, normalized_target[heldout_indices])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            basis = module.basis(lights, views)
            train_loss = float(_log_loss(module.prediction(basis, train_indices), normalized_target[train_indices]))
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_state = {
                    "basis_local_axis" if reflection_frame else "basis_axis": _hemisphere_axis(
                        module.raw_axis
                    ).cpu().numpy(),
                    "basis_sharpness": torch.exp(module.log_sharpness).cpu().numpy(),
                    "amplitude": (F.softplus(module.raw_amplitude) * scale).cpu().numpy(),
                }
                if reflection_frame:
                    best_state["basis_frame_blend"] = torch.sigmoid(
                        module.raw_frame_blend
                    ).cpu().numpy()

    final_module = SharedSgDictionary(
        normalized_target, atom_count, seed=seed, reflection_frame=reflection_frame
    ).to(torch_device)
    with torch.no_grad():
        axis_key = "basis_local_axis" if reflection_frame else "basis_axis"
        axis = torch.as_tensor(best_state[axis_key], dtype=torch.float32, device=torch_device)
        final_module.raw_axis.copy_(
            torch.cat((axis[..., :2], _inverse_softplus(axis[..., 2:3].clamp_min(1e-5))), dim=-1)
        )
        final_module.log_sharpness.copy_(
            torch.log(torch.as_tensor(best_state["basis_sharpness"], device=torch_device))
        )
        final_module.raw_amplitude.copy_(
            _inverse_softplus(
                torch.as_tensor(best_state["amplitude"], device=torch_device) / scale
            )
        )
        if reflection_frame:
            final_module.raw_frame_blend.copy_(
                torch.logit(
                    torch.as_tensor(best_state["basis_frame_blend"], device=torch_device).clamp(
                        1e-5, 1.0 - 1e-5
                    )
                )
            )
        basis = final_module.basis(lights, views)
        all_indices = torch.arange(len(target), device=torch_device)
        prediction = final_module.prediction(basis, all_indices) * scale
    smape = directional_smape(prediction, target).cpu().numpy()
    relative_l1 = (
        torch.sum(torch.abs(prediction - target), dim=(1, 2))
        / torch.sum(torch.abs(target), dim=(1, 2)).clamp_min(1e-8)
    ).cpu().numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"dictionary-rf-m{atom_count}" if reflection_frame else f"dictionary-m{atom_count}"
    np.savez_compressed(
        output_dir / f"{name}.npz",
        smape=smape,
        relative_l1=relative_l1,
        state_indices=dataset.state_indices,
        view_indices=dataset.view_indices,
        **best_state,
    )
    groups = _label_groups(dataset)
    summary = {
        "dataset": str(dataset_dir),
        "atom_count": atom_count,
        "reflection_frame": reflection_frame,
        "steps": steps,
        "restarts": restarts,
        "seconds": time.perf_counter() - start,
        "train_loss": best_train_loss,
        "smape": _summary(smape),
        "relative_l1": _summary(relative_l1),
        "splits": {
            split_name: {
                "smape": _summary(smape[dataset.splits == split_index]),
                "relative_l1": _summary(relative_l1[dataset.splits == split_index]),
            }
            for split_index, split_name in enumerate(("train", "validation", "test"))
        },
        "groups": {
            group_name: _summary(smape[mask])
            for group_name, mask in groups.items()
            if np.any(mask)
        },
    }
    (output_dir / f"{name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# 共享{'反射坐标 ' if reflection_frame else ''}SG 字典 M={atom_count}",
        "",
        f"字典原子只用训练材质族学习；每个 tile 的 RGB 系数仍单独做 oracle 拟合。"
        f"优化步数：{steps}；随机重启：{restarts}；用时：{summary['seconds']:.1f} 秒。",
        "",
        "| 数据划分 | median SMAPE | p90 SMAPE | median relative-L1 |",
        "|---|---:|---:|---:|",
    ]
    split_labels = {"train": "训练", "validation": "验证", "test": "测试"}
    for split_name, metrics in summary["splits"].items():
        lines.append(
            f"| {split_labels[split_name]} | {100 * metrics['smape']['median']:.2f}% | "
            f"{100 * metrics['smape']['p90']:.2f}% | {100 * metrics['relative_l1']['median']:.2f}% |"
        )
    (output_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a train-family shared SG dictionary oracle.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "v0_oracle_512")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "oracle_v0_512")
    parser.add_argument("--atoms", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--device", type=str)
    parser.add_argument("--reflection-frame", action="store_true")
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    fit_dictionary(
        args.dataset,
        args.output,
        atom_count=args.atoms,
        steps=args.steps,
        restarts=args.restarts,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        reflection_frame=args.reflection_frame,
    )


if __name__ == "__main__":
    main()
