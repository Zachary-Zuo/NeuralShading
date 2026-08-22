from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-NumPy IBL evaluation for a shared SG dictionary.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "v0_oracle_512")
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=PROJECT_ROOT / "reports" / "oracle_v0_512" / "dictionary-m16.npz",
    )
    parser.add_argument(
        "--probes",
        type=Path,
        default=PROJECT_ROOT / "data" / "hdris" / "polyhaven_probes_v0.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "oracle_v0_512" / "ibl_dictionary_real.json",
    )
    args = parser.parse_args()

    metadata = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
    mean_a = []
    mean_b = []
    tile_indices = []
    for shard in metadata["shards"]:
        tiles = np.load(args.dataset / shard["tiles"], mmap_mode="r")
        mean_a.append(np.asarray(tiles["mean_a"], dtype=np.float32))
        mean_b.append(np.asarray(tiles["mean_b"], dtype=np.float32))
        tile_indices.append(np.asarray(np.load(args.dataset / shard["index"]), dtype=np.uint32))
    mean_a_array = np.concatenate(mean_a)
    mean_b_array = np.concatenate(mean_b)
    tile_index = np.concatenate(tile_indices)
    target = 0.5 * (mean_a_array + mean_b_array)
    lights = np.load(args.dataset / "light_directions.npy")[:, :3]
    weights = np.load(args.dataset / "solid_angle_weights.npy")
    probe_archive = np.load(args.probes)
    probes = probe_archive["probes"]
    dictionary = np.load(args.dictionary)
    if "prediction" in dictionary.files:
        prediction = np.asarray(dictionary["prediction"], dtype=np.float32)
    elif "basis_local_axis" in dictionary.files:
        views = np.load(args.dataset / "views.npy")[tile_index[:, 1], :3]
        reflection = np.stack((-views[:, 0], -views[:, 1], views[:, 2]), axis=-1)
        tangent_xy = np.stack((-reflection[:, 1], reflection[:, 0]), axis=-1)
        tangent_length = np.linalg.norm(tangent_xy, axis=-1, keepdims=True)
        tangent_xy = np.where(
            tangent_length > 1e-6,
            tangent_xy / np.maximum(tangent_length, 1e-6),
            np.asarray([1.0, 0.0]),
        )
        tangent = np.concatenate((tangent_xy, np.zeros((len(views), 1))), axis=-1)
        bitangent = np.cross(reflection, tangent)
        local_axis = dictionary["basis_local_axis"]
        reflected_axis = (
            local_axis[None, :, 0:1] * tangent[:, None, :]
            + local_axis[None, :, 1:2] * bitangent[:, None, :]
            + local_axis[None, :, 2:3] * reflection[:, None, :]
        )
        blend = dictionary["basis_frame_blend"][None, :, None]
        axis = (1.0 - blend) * local_axis[None, :, :] + blend * reflected_axis
        axis[..., 2] = np.maximum(axis[..., 2], 1e-4)
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
        basis = np.exp(
            dictionary["basis_sharpness"][None, :, None]
            * (np.einsum("tmc,bc->tmb", axis, lights) - 1.0)
        )
        prediction = np.einsum("tmb,tmc->tbc", basis, dictionary["amplitude"], optimize=True)
    else:
        basis = np.exp(
            dictionary["basis_sharpness"][:, None]
            * (dictionary["basis_axis"] @ lights.T - 1.0)
        )
        prediction = np.einsum("mb,tmc->tbc", basis, dictionary["amplitude"], optimize=True)
    target_ibl = np.einsum("tbc,pbc,b->tpc", target, probes, weights, optimize=True)
    prediction_ibl = np.einsum("tbc,pbc,b->tpc", prediction, probes, weights, optimize=True)
    mean_a_ibl = np.einsum("tbc,pbc,b->tpc", mean_a_array, probes, weights, optimize=True)
    mean_b_ibl = np.einsum("tbc,pbc,b->tpc", mean_b_array, probes, weights, optimize=True)
    relative_l1 = np.sum(np.abs(prediction_ibl - target_ibl), axis=2) / np.maximum(
        np.sum(np.abs(target_ibl), axis=2), 1e-6
    )
    noise = np.sum(np.abs(mean_a_ibl - mean_b_ibl), axis=2) / np.maximum(
        0.5 * np.sum(np.abs(mean_a_ibl) + np.abs(mean_b_ibl), axis=2), 1e-6
    )
    result = {
        "dictionary": args.dictionary.stem,
        "tile_count": len(target),
        "probe_count": len(probes),
        "relative_l1": _summary(relative_l1),
        "noise_relative_l1": _summary(noise),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
