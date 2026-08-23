from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ncls.core.representations.legacy_ltc_k2.torch_eval import LegacyLtcK2Tensors, eval_direct_top
from ncls.data import SPLIT_NAMES
from ncls.learning.data import LayerStackReferenceStore
from ncls.learning.evaluation.metrics import summarize

from .families import fit_direct_batch


@dataclass(frozen=True)
class DirectFitConfig:
    family: str = "ltc"
    lobe_count: int = 2
    fit_batch: int = 256
    steps: int = 800
    restarts: int = 3
    learning_rate: float = 0.03
    seed: int = 20260822
    device: str | None = None

    def __post_init__(self) -> None:
        if self.family not in {"ggx", "ltc", "sg"}:
            raise ValueError("direct-fit family must be ggx, ltc or sg")
        if min(self.lobe_count, self.fit_batch, self.steps, self.restarts, self.learning_rate) <= 0:
            raise ValueError("direct-fit numeric settings must be positive")


def _direct_top(batch: dict[str, np.ndarray], lights: np.ndarray, device: torch.device) -> np.ndarray:
    count = len(batch["view"])
    zeros_amplitude = torch.zeros((count, 2, 3), dtype=torch.float32, device=device)
    state = LegacyLtcK2Tensors(
        interface_kind=torch.as_tensor(batch["top_kind"], device=device),
        alpha=torch.as_tensor(batch["top_alpha"], device=device),
        relative_ior=torch.as_tensor(batch["top_relative_ior"], device=device),
        eta=torch.as_tensor(batch["top_eta"], device=device),
        k=torch.as_tensor(batch["top_k"], device=device),
        color=torch.as_tensor(batch["top_color"], device=device),
        tangent_rotation=torch.as_tensor(batch["top_rotation"], device=device),
        amplitude=zeros_amplitude,
        inverse_scale=torch.ones((count, 2, 2), dtype=torch.float32, device=device),
        shear=torch.zeros((count, 2, 3), dtype=torch.float32, device=device),
        angle=torch.zeros((count, 2), dtype=torch.float32, device=device),
    )
    return eval_direct_top(
        state,
        torch.as_tensor(batch["view"], dtype=torch.float32, device=device),
        torch.as_tensor(lights, dtype=torch.float32, device=device),
    ).cpu().numpy()


def run_direct_fit(
    dataset_path: Path | str,
    output_dir: Path | str,
    *,
    split: str,
    config: DirectFitConfig,
    max_query_groups: int | None = None,
) -> dict[str, object]:
    """逐 query group 直接优化表示参数，用来测量方向切片上界。"""

    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("direct-fit output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output / "tensorboard"))
    store = LayerStackReferenceStore(dataset_path)
    indices = store.source_split_indices[split]
    if max_query_groups is not None and len(indices) > max_query_groups:
        indices = indices[np.linspace(0, len(indices) - 1, max_query_groups, dtype=np.int64)]
    if len(indices) == 0:
        raise ValueError("selected direct-fit split is empty")
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    prediction_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    query_group_parts: list[np.ndarray] = []
    parameter_parts: dict[str, list[np.ndarray]] = {}
    clamped_count = 0
    response_count = 0
    start_time = time.perf_counter()
    try:
        for batch_index, start in enumerate(range(0, len(indices), config.fit_batch)):
            selected = indices[start : start + config.fit_batch]
            batch = store.batch(selected)
            target = batch["mean"]
            direct = _direct_top(batch, store.lights, device)
            residual = np.maximum(target - direct, 0.0)
            clamped_count += int(np.count_nonzero(target < direct))
            response_count += int(target.size)
            fitted = fit_direct_batch(
                residual,
                batch["view"],
                store.lights,
                family=config.family,
                lobe_count=config.lobe_count,
                steps=config.steps,
                restarts=config.restarts,
                learning_rate=config.learning_rate,
                device=device,
                seed=config.seed + start * 17,
            )
            prediction = direct + fitted.prediction
            prediction_parts.append(prediction)
            target_parts.append(target)
            query_group_parts.append(selected)
            for name, values in fitted.parameters.items():
                parameter_parts.setdefault(name, []).append(values)
            relative = np.sum(np.abs(prediction - target), axis=(1, 2)) / np.maximum(
                np.sum(np.abs(target), axis=(1, 2)), 1e-8
            )
            writer.add_scalar("direct_fit/batch_relative_l1_median", float(np.median(relative)), batch_index)
        writer.flush()
    finally:
        writer.close()
    prediction = np.concatenate(prediction_parts)
    target = np.concatenate(target_parts)
    query_group_ids = np.concatenate(query_group_parts)
    smape_floor = 1e-3 * np.max(target, axis=(1, 2), keepdims=True) + 1e-5
    smape = np.mean(
        2.0 * np.abs(prediction - target) / (np.abs(prediction) + np.abs(target) + smape_floor),
        axis=(1, 2),
    )
    relative_l1 = np.sum(np.abs(prediction - target), axis=(1, 2)) / np.maximum(
        np.sum(np.abs(target), axis=(1, 2)), 1e-8
    )
    representation_id = (
        "legacy-ltc-k2@1"
        if config.family == "ltc" and config.lobe_count == 2
        else f"exact-top-{config.family}-k{config.lobe_count}@research"
    )
    np.savez_compressed(
        output / "parameters.npz",
        query_group_id=query_group_ids,
        smape=smape,
        relative_l1=relative_l1,
        **{name: np.concatenate(parts) for name, parts in parameter_parts.items()},
    )
    result: dict[str, object] = {
        "format_name": "ncls.representation-ceiling",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": store.dataset.manifest.dataset_id,
        "split": split,
        "representation_id": representation_id,
        "config": asdict(config),
        "query_group_count": len(query_group_ids),
        "clamped_negative_residual_fraction": clamped_count / max(response_count, 1),
        "smape": summarize(smape),
        "relative_l1": summarize(relative_l1),
        "seconds": time.perf_counter() - start_time,
        "parameters_uri": "parameters.npz",
    }
    (output / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
