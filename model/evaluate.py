from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.compiler import RecurrentCompilerBaseline
from model.data import DirectionTileStore
from model.train import _evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_checkpoint(
    dataset_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    batch_size: int,
    max_tiles: int,
    device_name: str | None,
) -> dict[str, object]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    training_result = checkpoint["result"]
    model = RecurrentCompilerBaseline(width=int(training_result["width"])).to(device)
    model.load_state_dict(checkpoint["model"])
    store = DirectionTileStore(dataset_dir)
    lights = torch.as_tensor(store.lights, dtype=torch.float32, device=device)
    metrics = {
        split: _evaluate(
            model,
            store,
            indices,
            lights,
            device,
            batch_size=batch_size,
            max_tiles=max_tiles,
        )
        for split, indices in store.split_indices.items()
        if len(indices)
    }
    result: dict[str, object] = {
        "dataset": str(dataset_dir),
        "checkpoint": str(checkpoint_path),
        "model": training_result["model"],
        "evaluation_tiles_per_split": max_tiles,
        "sampling": "evenly-spaced-across-full-split",
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a stack-to-packet compiler checkpoint.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "v0_train")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "reports" / "compiler_v0" / "compiler.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "compiler_v0" / "evaluation.json",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-tiles", type=int, default=16384)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    evaluate_checkpoint(
        args.dataset,
        args.checkpoint,
        args.output,
        batch_size=args.batch_size,
        max_tiles=args.max_tiles,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
