from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from baselines.closure_families import fit_oracle_batch
from schema import BINARY_SIZE, LayerType, unpack_stack


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OracleDataset:
    target: np.ndarray
    mean_a: np.ndarray
    mean_b: np.ndarray
    views: np.ndarray
    lights: np.ndarray
    state_indices: np.ndarray
    view_indices: np.ndarray
    splits: np.ndarray
    base_types: np.ndarray
    layer_counts: np.ndarray
    roughness: np.ndarray


def _smape_numpy(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    peak = np.max(second, axis=(1, 2), keepdims=True)
    floor = 1e-3 * peak + 1e-5
    return np.mean(2.0 * np.abs(first - second) / (np.abs(first) + np.abs(second) + floor), axis=(1, 2))


def load_oracle_dataset(dataset_dir: Path, max_tiles: int | None = None) -> OracleDataset:
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    mean_a_parts: list[np.ndarray] = []
    mean_b_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    remaining = max_tiles
    for shard in metadata["shards"]:
        if remaining is not None and remaining <= 0:
            break
        tiles = np.load(dataset_dir / shard["tiles"], mmap_mode="r")
        index = np.load(dataset_dir / shard["index"], mmap_mode="r")
        take = len(tiles) if remaining is None else min(len(tiles), remaining)
        mean_a_parts.append(np.asarray(tiles["mean_a"][:take], dtype=np.float32))
        mean_b_parts.append(np.asarray(tiles["mean_b"][:take], dtype=np.float32))
        index_parts.append(np.asarray(index[:take], dtype=np.uint32))
        if remaining is not None:
            remaining -= take
    if not mean_a_parts:
        raise ValueError("dataset contains no tiles")

    mean_a = np.concatenate(mean_a_parts)
    mean_b = np.concatenate(mean_b_parts)
    index = np.concatenate(index_parts)
    target = 0.5 * (mean_a + mean_b)
    all_views = np.load(dataset_dir / "views.npy")
    lights = np.load(dataset_dir / "light_directions.npy")[:, :3]
    states = np.load(dataset_dir / "states.npy")
    state_indices = index[:, 0]
    view_indices = index[:, 1]

    payload = (dataset_dir / "stacks.bin").read_bytes()
    unique_states = np.unique(state_indices)
    state_descriptors: dict[int, tuple[int, int, float]] = {}
    for state_index in unique_states:
        offset = int(state_index) * BINARY_SIZE
        stack = unpack_stack(payload[offset : offset + BINARY_SIZE])
        state_descriptors[int(state_index)] = (
            int(stack.layers[-1].layer_type),
            len(stack.layers),
            0.5 * (stack.layers[0].roughness_x + stack.layers[0].roughness_y),
        )
    descriptors = np.asarray([state_descriptors[int(index)] for index in state_indices], dtype=np.float32)
    return OracleDataset(
        target=target,
        mean_a=mean_a,
        mean_b=mean_b,
        views=all_views[view_indices],
        lights=lights,
        state_indices=state_indices,
        view_indices=view_indices,
        splits=states["split"][state_indices],
        base_types=descriptors[:, 0].astype(np.uint8),
        layer_counts=descriptors[:, 1].astype(np.uint8),
        roughness=descriptors[:, 2],
    )


def _parse_config(value: str) -> tuple[str, int]:
    try:
        family, count_text = value.lower().split(":", maxsplit=1)
        count = int(count_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("config must use FAMILY:K, for example ltc:3") from error
    if family not in {"ggx", "ltc", "sg"} or count < 1:
        raise argparse.ArgumentTypeError("supported families are ggx, ltc and sg with K >= 1")
    return family, count


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _label_groups(dataset: OracleDataset) -> dict[str, np.ndarray]:
    base_names = {
        int(LayerType.DIFFUSE): "diffuse-base",
        int(LayerType.ROUGH_CONDUCTOR): "conductor-base",
        int(LayerType.SHEEN): "sheen-base",
    }
    groups: dict[str, np.ndarray] = {}
    for base_type, name in base_names.items():
        groups[name] = dataset.base_types == base_type
    for layer_count in np.unique(dataset.layer_counts):
        groups[f"layers-{int(layer_count)}"] = dataset.layer_counts == layer_count
    theta = np.degrees(np.arccos(np.clip(dataset.views[:, 2], 0.0, 1.0)))
    groups["view-0-30"] = theta < 30.0
    groups["view-30-60"] = (theta >= 30.0) & (theta < 60.0)
    groups["view-60-plus"] = theta >= 60.0
    groups["roughness-under-0.1"] = dataset.roughness < 0.1
    groups["roughness-0.1-0.3"] = (dataset.roughness >= 0.1) & (dataset.roughness < 0.3)
    groups["roughness-0.3-plus"] = dataset.roughness >= 0.3
    return groups


def run_oracle(
    dataset_dir: Path,
    output_dir: Path,
    *,
    configs: list[tuple[str, int]],
    max_tiles: int | None,
    fit_batch: int,
    steps: int,
    restarts: int,
    learning_rate: float,
    device: str | None,
    seed: int,
) -> dict[str, object]:
    dataset = load_oracle_dataset(dataset_dir, max_tiles)
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_smape = _smape_numpy(dataset.mean_a, dataset.mean_b)
    result_summary: dict[str, object] = {
        "dataset": str(dataset_dir),
        "tile_count": len(dataset.target),
        "steps": steps,
        "restarts": restarts,
        "noise_smape": _summary(noise_smape),
        "families": {},
    }
    groups = _label_groups(dataset)

    for family, lobe_count in configs:
        name = f"{family}-k{lobe_count}"
        start = time.perf_counter()
        smape_parts: list[np.ndarray] = []
        relative_l1_parts: list[np.ndarray] = []
        parameter_parts: dict[str, list[np.ndarray]] = {}
        for batch_start in range(0, len(dataset.target), fit_batch):
            batch_end = min(batch_start + fit_batch, len(dataset.target))
            fitted = fit_oracle_batch(
                dataset.target[batch_start:batch_end],
                dataset.views[batch_start:batch_end, :3],
                dataset.lights,
                family=family,
                lobe_count=lobe_count,
                steps=steps,
                restarts=restarts,
                learning_rate=learning_rate,
                device=device,
                seed=seed + batch_start * 17,
            )
            smape_parts.append(fitted.smape)
            relative_l1_parts.append(fitted.relative_l1)
            for parameter_name, values in fitted.parameters.items():
                parameter_parts.setdefault(parameter_name, []).append(values)
            print(f"{name}: fitted tiles [{batch_start}, {batch_end})")
        smape = np.concatenate(smape_parts)
        relative_l1 = np.concatenate(relative_l1_parts)
        parameters = {key: np.concatenate(parts) for key, parts in parameter_parts.items()}
        np.savez_compressed(
            output_dir / f"{name}.npz",
            smape=smape,
            relative_l1=relative_l1,
            state_indices=dataset.state_indices,
            view_indices=dataset.view_indices,
            **parameters,
        )
        group_summary = {
            group_name: _summary(smape[mask])
            for group_name, mask in groups.items()
            if np.any(mask)
        }
        result_summary["families"][name] = {
            "smape": _summary(smape),
            "relative_l1": _summary(relative_l1),
            "seconds": time.perf_counter() - start,
            "groups": group_summary,
        }

    (output_dir / "summary.json").write_text(
        json.dumps(result_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_dir / "report.md", result_summary)
    return result_summary


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _write_markdown(path: Path, summary: dict[str, object]) -> None:
    noise = summary["noise_smape"]
    assert isinstance(noise, dict)
    families = summary["families"]
    assert isinstance(families, dict)
    lines = [
        "# 解析着色表示上界实验",
        "",
        f"Tiles：{summary['tile_count']}；优化步数：{summary['steps']}；随机重启：{summary['restarts']}。",
        "",
        f"A/B 随机游走参考噪声 SMAPE：median {_percent(noise['median'])}，p90 {_percent(noise['p90'])}。",
        "",
        "| closure | median SMAPE | p90 SMAPE | median relative-L1 | 用时（秒） |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, raw_metrics in families.items():
        metrics = raw_metrics
        smape = metrics["smape"]
        relative_l1 = metrics["relative_l1"]
        lines.append(
            f"| {name} | {_percent(smape['median'])} | {_percent(smape['p90'])} | "
            f"{_percent(relative_l1['median'])} | {metrics['seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "这是表示上界实验：每个 tile 的参数都直接优化，没有预测网络。"
            "选择最终函数族时还要结合至少 500 个材质族的实验和环境光加权指标。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit analytic closure families directly to teacher tiles.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "pilot_v0_batched")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "oracle_pilot")
    parser.add_argument(
        "--config",
        type=_parse_config,
        action="append",
        default=None,
        help="Repeatable FAMILY:K entry; defaults to ggx:2, ggx:3, ltc:3, sg:8.",
    )
    parser.add_argument("--max-tiles", type=int)
    parser.add_argument("--fit-batch", type=int, default=256)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    run_oracle(
        args.dataset,
        args.output,
        configs=args.config or [("ggx", 2), ("ggx", 3), ("ltc", 3), ("sg", 8)],
        max_tiles=args.max_tiles,
        fit_batch=args.fit_batch,
        steps=args.steps,
        restarts=args.restarts,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
