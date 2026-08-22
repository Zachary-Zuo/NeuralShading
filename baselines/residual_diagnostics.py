from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baselines.oracle_fit import _label_groups, load_oracle_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
    }


def analyze_residual_packets(
    dataset_dir: Path,
    oracle_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    dataset = load_oracle_dataset(dataset_dir)
    archives = {
        "K2": np.load(oracle_dir / "direct-ltc-k2.npz"),
        "K3": np.load(oracle_dir / "direct-ltc-k3.npz"),
    }
    errors = {
        name: np.asarray(archive["relative_l1"], dtype=np.float32)
        for name, archive in archives.items()
    }
    noise = np.sum(np.abs(dataset.mean_a - dataset.mean_b), axis=(1, 2)) / np.maximum(
        0.5 * np.sum(np.abs(dataset.mean_a) + np.abs(dataset.mean_b), axis=(1, 2)), 1e-8
    )
    groups = {"all": np.ones(len(noise), dtype=bool), **_label_groups(dataset)}
    groups.update(
        {
            "split-train": dataset.splits == 0,
            "split-validation": dataset.splits == 1,
            "split-test": dataset.splits == 2,
        }
    )
    grouped = {
        group_name: {
            "tile_count": int(np.count_nonzero(mask)),
            "K2": _summary(errors["K2"][mask]),
            "K3": _summary(errors["K3"][mask]),
            "noise": _summary(noise[mask]),
        }
        for group_name, mask in groups.items()
        if np.any(mask)
    }
    thresholds = {
        name: {
            "under_5_percent": float(np.mean(values < 0.05)),
            "under_10_percent": float(np.mean(values < 0.10)),
            "over_20_percent": float(np.mean(values > 0.20)),
            "over_30_percent": float(np.mean(values > 0.30)),
        }
        for name, values in errors.items()
    }
    result: dict[str, object] = {
        "dataset": str(dataset_dir),
        "tile_count": len(noise),
        "groups": grouped,
        "thresholds": thresholds,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    labels = {
        "all": "全部",
        "split-train": "训练族",
        "split-validation": "验证族",
        "split-test": "测试族",
        "diffuse-base": "漫反射基底",
        "conductor-base": "导体基底",
        "sheen-base": "绒面基底",
        "view-0-30": "视角 0–30°",
        "view-30-60": "视角 30–60°",
        "view-60-plus": "视角 60° 以上",
        "roughness-under-0.1": "顶层粗糙度 < 0.1",
        "roughness-0.1-0.3": "顶层粗糙度 0.1–0.3",
        "roughness-0.3-plus": "顶层粗糙度 ≥ 0.3",
    }
    for layer_count in range(1, 9):
        labels[f"layers-{layer_count}"] = f"{layer_count} 层"
    order = [
        "all",
        "split-train",
        "split-validation",
        "split-test",
        "diffuse-base",
        "conductor-base",
        "sheen-base",
        *(f"layers-{count}" for count in range(1, 9)),
        "view-0-30",
        "view-30-60",
        "view-60-plus",
        "roughness-under-0.1",
        "roughness-0.1-0.3",
        "roughness-0.3-plus",
    ]
    lines = [
        "# 顶层解析项加 LTC 残差瓣的误差诊断",
        "",
        "方向误差采用每个 tile 的 RGB relative-L1。A/B 列表示两组独立 teacher 均值之间的差异，用来判断拟合误差中有多少可能来自参考噪声。",
        "",
        "| 分组 | tiles | K2 median / p90 | K3 median / p90 | A/B median / p90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in order:
        if name not in grouped:
            continue
        values = grouped[name]
        lines.append(
            f"| {labels[name]} | {values['tile_count']} | "
            f"{100 * values['K2']['median']:.2f}% / {100 * values['K2']['p90']:.2f}% | "
            f"{100 * values['K3']['median']:.2f}% / {100 * values['K3']['p90']:.2f}% | "
            f"{100 * values['noise']['median']:.2f}% / {100 * values['noise']['p90']:.2f}% |"
        )
    lines.extend(["", "## 覆盖率", ""])
    for name in ("K2", "K3"):
        values = thresholds[name]
        lines.append(
            f"- {name}：{100 * values['under_5_percent']:.1f}% 的 tiles 低于 5%，"
            f"{100 * values['under_10_percent']:.1f}% 低于 10%；"
            f"{100 * values['over_20_percent']:.1f}% 高于 20%，"
            f"{100 * values['over_30_percent']:.1f}% 高于 30%。"
        )
    lines.extend(
        [
            "",
            "## 如何解读",
            "",
            "K3 比 K2 更准，但它多占一个 48-byte LTC 槽位。若方向长尾仍主要集中在深层、绒面或掠射视角，下一步应先改进残差函数族或按困难区域使用可变容量，而不是把所有像素统一扩成四槽 packet。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 direct-top + LTC 残差 packet 的分组误差。")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "v0_oracle_512")
    parser.add_argument("--oracle", type=Path, default=PROJECT_ROOT / "reports" / "oracle_v0_512")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "oracle_v0_512" / "residual_diagnostics.md",
    )
    args = parser.parse_args()
    analyze_residual_packets(args.dataset, args.oracle, args.output)


if __name__ == "__main__":
    main()
