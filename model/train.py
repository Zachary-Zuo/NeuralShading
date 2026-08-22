from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from closures.torch_eval import (
    ClosurePacketTensors,
    decode_ltc_residual,
    evaluate_closure_packet,
)
from model.compiler import RecurrentCompilerBaseline
from model.data import DirectionTileStore
from model.features import FEATURE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tensor_batch(raw: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(values, device=device)
        for name, values in raw.items()
    }


def _predict_response(
    model: RecurrentCompilerBaseline,
    batch: dict[str, torch.Tensor],
    lights: torch.Tensor,
) -> torch.Tensor:
    raw = model(
        batch["layer_types"].long(),
        batch["continuous"].float(),
        batch["layer_counts"].long(),
        batch["view"].float(),
    )
    amplitude, inverse_scale, shear, angle = decode_ltc_residual(raw)
    packet = ClosurePacketTensors(
        layer_type=batch["top_type"].long(),
        roughness=batch["top_roughness"].float(),
        eta=batch["top_eta"].float(),
        k=batch["top_k"].float(),
        albedo=batch["top_albedo"].float(),
        tangent_rotation=batch["top_rotation"].float(),
        amplitude=amplitude,
        inverse_scale=inverse_scale,
        shear=shear,
        angle=angle,
    )
    return evaluate_closure_packet(packet, batch["view"].float(), lights)


def response_loss(
    prediction: torch.Tensor,
    mean_a: torch.Tensor,
    mean_b: torch.Tensor,
) -> torch.Tensor:
    target = 0.5 * (mean_a + mean_b)
    peak = torch.amax(target, dim=(1, 2), keepdim=True)
    floor = 1e-3 * peak + 1e-5
    noise = 0.5 * torch.abs(mean_a - mean_b)
    confidence = torch.clamp((target + floor) / (target + floor + noise), 0.1, 1.0).detach()
    log_prediction = torch.log(prediction + floor)
    log_delta = log_prediction - torch.log(target + floor)
    log_loss = torch.sum(confidence * log_delta * log_delta) / torch.sum(confidence)
    relative = torch.sum(
        confidence * 2.0 * torch.abs(prediction - target) / (prediction + target + floor)
    ) / torch.sum(confidence)
    return log_loss + 0.05 * relative


@torch.no_grad()
def _evaluate(
    model: RecurrentCompilerBaseline,
    store: DirectionTileStore,
    indices: np.ndarray,
    lights: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    max_tiles: int,
) -> dict[str, float]:
    if len(indices) > max_tiles:
        selected = indices[
            np.linspace(0, len(indices) - 1, max_tiles, dtype=np.int64)
        ]
    else:
        selected = indices
    losses: list[float] = []
    relative_l1: list[np.ndarray] = []
    ab_relative_l1: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(selected), batch_size):
        raw = store.batch(selected[start : start + batch_size])
        batch = _tensor_batch(raw, device)
        prediction = _predict_response(model, batch, lights)
        losses.append(float(response_loss(prediction, batch["mean_a"], batch["mean_b"])))
        target = 0.5 * (batch["mean_a"] + batch["mean_b"])
        relative = torch.sum(torch.abs(prediction - target), dim=(1, 2)) / torch.clamp(
            torch.sum(torch.abs(target), dim=(1, 2)), min=1e-8
        )
        relative_l1.append(relative.cpu().numpy())
        noise = torch.sum(torch.abs(batch["mean_a"] - batch["mean_b"]), dim=(1, 2)) / torch.clamp(
            torch.sum(torch.abs(target), dim=(1, 2)), min=1e-8
        )
        ab_relative_l1.append(noise.cpu().numpy())
    values = np.concatenate(relative_l1)
    noise_values = np.concatenate(ab_relative_l1)
    return {
        "loss": float(np.mean(losses)),
        "relative_l1_median": float(np.median(values)),
        "relative_l1_p90": float(np.quantile(values, 0.9)),
        "ab_relative_l1_median": float(np.median(noise_values)),
        "ab_relative_l1_p90": float(np.quantile(noise_values, 0.9)),
    }


def train_compiler(
    dataset_dir: Path,
    output_dir: Path,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    width: int,
    seed: int,
    device_name: str | None,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    store = DirectionTileStore(dataset_dir)
    model = RecurrentCompilerBaseline(width=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    lights = torch.as_tensor(store.lights, dtype=torch.float32, device=device)
    start_time = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        indices = store.sample_batch_indices("train", batch_size, np_rng)
        batch = _tensor_batch(store.batch(indices), device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict_response(model, batch, lights)
        loss = response_loss(prediction, batch["mean_a"], batch["mean_b"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            print(f"step {step:6d}/{steps}: loss={float(loss.detach()):.6f}")

    metrics = {
        split: _evaluate(
            model,
            store,
            indices,
            lights,
            device,
            batch_size=batch_size,
            max_tiles=4096,
        )
        for split, indices in store.split_indices.items()
        if len(indices)
    }
    result: dict[str, object] = {
        "dataset": str(dataset_dir),
        "feature_version": FEATURE_VERSION,
        "model": "recurrent-compiler-baseline",
        "width": width,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(device),
        "seconds": time.perf_counter() - start_time,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "result": result},
        output_dir / "compiler.pt",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the P1 stack-to-LTC-K2 compiler baseline.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "v0_train")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "compiler_v0")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    train_compiler(
        args.dataset,
        args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        width=args.width,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
