from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ncls.learning.data import ReferenceCorpusStore


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # NumPy BLAS and the Falcor collector can coexist poorly on the Windows
    # workstation. Small center blocks keep memory bounded and avoid a threaded
    # GEMM in this direct-fit diagnostic.
    distances = np.empty((len(values), len(centers)), dtype=np.float32)
    for start in range(0, len(centers), 4):
        stop = min(start + 4, len(centers))
        difference = values[:, None, :] - centers[None, start:stop, :]
        distances[:, start:stop] = np.einsum(
            "nkd,nkd->nk", difference, difference, optimize=False
        )
    return distances


def _kmeans_plus_plus(
    values: np.ndarray,
    cluster_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = len(values)
    first = int(rng.integers(count))
    chosen = [first]
    closest = _squared_distances(values, values[first : first + 1])[:, 0]
    for _ in range(1, cluster_count):
        weights = closest.copy()
        weights[np.asarray(chosen, dtype=np.int64)] = 0.0
        total = float(np.sum(weights))
        if total <= 0.0:
            remaining = np.setdiff1d(np.arange(count), np.asarray(chosen), assume_unique=False)
            next_index = int(remaining[0])
        else:
            next_index = int(rng.choice(count, p=weights / total))
        chosen.append(next_index)
        candidate = _squared_distances(values, values[next_index : next_index + 1])[:, 0]
        closest = np.minimum(closest, candidate)
    return values[np.asarray(chosen, dtype=np.int64)].copy()


def _fit_kmeans(
    values: np.ndarray,
    cluster_count: int,
    *,
    seed: int,
    maximum_iterations: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if cluster_count < 2 or cluster_count >= len(values):
        raise ValueError("dictionary cluster count must be in [2, unit_count)")
    rng = np.random.default_rng(seed)
    centers = _kmeans_plus_plus(values, cluster_count, rng)
    labels = np.full(len(values), -1, dtype=np.int64)
    for iteration in range(1, maximum_iterations + 1):
        distances = _squared_distances(values, centers)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, next_labels):
            return centers, labels, iteration - 1
        labels = next_labels
        minimum = distances[np.arange(len(values)), labels]
        for cluster in range(cluster_count):
            members = labels == cluster
            if np.any(members):
                centers[cluster] = np.mean(values[members], axis=0)
                continue
            replacement = int(np.argmax(minimum))
            centers[cluster] = values[replacement]
            labels[replacement] = cluster
            minimum[replacement] = -1.0
    return centers, labels, maximum_iterations


def _top2_reconstruct(
    values: np.ndarray,
    centers: np.ndarray,
    *,
    candidate_count: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = _squared_distances(values, centers)
    count = min(candidate_count, len(centers))
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :count]
    first = nearest[:, 0]
    best_second = first.copy()
    best_weight = np.zeros(len(values), dtype=np.float32)
    best_error = distances[np.arange(len(values)), first].copy()
    for candidate_slot in range(1, count):
        second = nearest[:, candidate_slot]
        start = centers[first]
        delta = centers[second] - start
        denominator = np.sum(delta * delta, axis=1)
        numerator = np.sum((values - start) * delta, axis=1)
        weight = np.clip(
            numerator / np.maximum(denominator, np.finfo(np.float32).tiny),
            0.0,
            1.0,
        )
        reconstruction = start + weight[:, None] * delta
        error = np.sum((values - reconstruction) ** 2, axis=1)
        improved = error < best_error
        best_error[improved] = error[improved]
        best_second[improved] = second[improved]
        best_weight[improved] = weight[improved]
    quantized_weight = best_weight.astype(np.float16).astype(np.float32)
    reconstruction = (
        centers[first]
        + quantized_weight[:, None] * (centers[best_second] - centers[first])
    )
    indices = np.column_stack((first, best_second)).astype(np.uint16)
    return reconstruction.astype(np.float32), indices, quantized_weight


def _matched_pca(
    values: np.ndarray,
    byte_budget: int,
) -> tuple[np.ndarray, int, int]:
    import torch

    unit_count, dimension = values.shape
    maximum_rank = min(unit_count - 1, dimension)
    base_bytes = 4 * dimension
    per_rank_bytes = 4 * dimension + 2 * unit_count
    rank = min(maximum_rank, max(0, (byte_budget - base_bytes) // per_rank_bytes))
    tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    mean = torch.mean(tensor, dim=0, keepdim=True)
    if rank == 0:
        return mean.expand_as(tensor).numpy().copy(), 0, base_bytes
    centered = tensor - mean
    left, singular, components = torch.linalg.svd(centered, full_matrices=False)
    rank = min(rank, int(torch.count_nonzero(singular > torch.finfo(torch.float32).eps)))
    if rank == 0:
        return mean.expand_as(tensor).numpy().copy(), 0, base_bytes
    components = components[:rank]
    coefficients = (left[:, :rank] * singular[None, :rank]).to(torch.float16).to(torch.float32)
    reconstruction = mean + coefficients @ components
    stored_bytes = base_bytes + rank * per_rank_bytes
    return reconstruction.numpy().astype(np.float32, copy=False), rank, stored_bytes


def _summary(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)),
        "maximum": float(np.max(data)),
    }


def _floored_relative_l1_rows(
    absolute_error: np.ndarray,
    reference: np.ndarray,
    *,
    floor_fraction: float = 0.01,
) -> tuple[np.ndarray, float]:
    reference_l1 = np.sum(np.abs(reference), axis=1)
    nonzero = reference_l1[reference_l1 > 1e-12]
    median_nonzero = float(np.median(nonzero)) if len(nonzero) else 1e-12
    floor = max(floor_fraction * median_nonzero, 1e-12)
    numerator = np.sum(np.abs(absolute_error), axis=1)
    return numerator / np.maximum(reference_l1, floor), floor


def _distortion_rows(
    reference_transform: np.ndarray,
    reconstruction_transform: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    transformed_l1, transformed_floor = _floored_relative_l1_rows(
        reconstruction_transform - reference_transform,
        reference_transform,
    )
    safe = np.clip(reconstruction_transform, 0.0, 20.0)
    reference_linear = scale[None, :] * np.expm1(reference_transform)
    reconstruction_linear = scale[None, :] * np.expm1(safe)
    linear_l1, linear_floor = _floored_relative_l1_rows(
        reconstruction_linear - reference_linear,
        reference_linear,
    )
    return transformed_l1, linear_l1, {
        "fraction_of_median_nonzero_reference_l1": 0.01,
        "transformed_l1_floor": transformed_floor,
        "linear_l1_floor": linear_floor,
    }


def _distortion(
    reference_transform: np.ndarray,
    reconstruction_transform: np.ndarray,
    scale: np.ndarray,
) -> dict[str, Any]:
    transformed_l1, linear_l1, normalization = _distortion_rows(
        reference_transform, reconstruction_transform, scale
    )
    return {
        "transformed_relative_l1": _summary(transformed_l1),
        "linear_relative_l1": _summary(linear_l1),
        "relative_l1_normalization": normalization,
        "transformed_rmse": float(
            np.sqrt(np.mean((reconstruction_transform - reference_transform) ** 2))
        ),
    }


def _paired_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    iterations: int = 1000,
) -> dict[str, Any]:
    if baseline.shape != candidate.shape or baseline.ndim != 1 or not len(baseline):
        raise ValueError("M3 paired bootstrap requires matched nonempty unit rows")
    difference = candidate - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(iterations, len(difference)))
    samples = np.mean(difference[indices], axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "metric": "floored_linear_relative_l1_per_unit",
        "difference": "dictionary-minus-matched-pca",
        "mean": float(np.mean(difference)),
        "confidence": 0.95,
        "interval": [float(low), float(high)],
        "iterations": iterations,
        "dictionary_significantly_better": bool(high < 0.0),
        "no_significant_difference": bool(low <= 0.0 <= high),
    }


def _canonical_dense_units(
    store: ReferenceCorpusStore,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    str,
]:
    indices = store.indices_for_query_role("dense_slice")
    if not len(indices):
        raise ValueError("M3 response-space oracle requires dense_slice queries")
    by_state: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    probe_digest = hashlib.sha256()
    canonical: list[tuple[np.ndarray, np.ndarray]] | None = None
    for raw in store.iter_batches(indices, batch_size=8):
        for row, state_index in enumerate(np.asarray(raw["state_index"], dtype=np.int64)):
            item = (
                np.asarray(raw["wo"][row], dtype=np.float32),
                np.asarray(raw["wi"][row], dtype=np.float32),
                np.asarray(raw["mean"][row], dtype=np.float32),
                np.asarray(raw["standard_error"][row], dtype=np.float32),
            )
            by_state.setdefault(int(state_index), []).append(item)
    state_ids = list(map(str, store.state_strings("state_id").tolist()))
    state_vectors: list[np.ndarray] = []
    view_vectors: list[np.ndarray] = []
    state_standard_errors: list[np.ndarray] = []
    view_standard_errors: list[np.ndarray] = []
    view_ids: list[str] = []
    for state_index in range(store.state_count):
        rows = by_state.get(state_index, [])
        rows.sort(key=lambda item: tuple(map(float, item[0].tolist())))
        if not rows:
            raise ValueError(f"dense_slice has no canonical probe for {state_ids[state_index]}")
        probes = [(row[0], row[1]) for row in rows]
        if canonical is None:
            canonical = probes
            for wo, wi in probes:
                probe_digest.update(np.ascontiguousarray(wo).tobytes())
                probe_digest.update(np.ascontiguousarray(wi).tobytes())
        elif len(probes) != len(canonical) or any(
            not np.array_equal(wo, expected_wo) or not np.array_equal(wi, expected_wi)
            for (wo, wi), (expected_wo, expected_wi) in zip(probes, canonical, strict=True)
        ):
            raise ValueError("dense_slice probes are not canonical across states")
        response_rows = [row[2].reshape(-1) for row in rows]
        standard_error_rows = [row[3].reshape(-1) for row in rows]
        state_vectors.append(np.concatenate(response_rows))
        state_standard_errors.append(np.concatenate(standard_error_rows))
        for view_index, (response, standard_error) in enumerate(
            zip(response_rows, standard_error_rows, strict=True)
        ):
            view_vectors.append(response)
            view_standard_errors.append(standard_error)
            view_ids.append(f"{state_ids[state_index]}#view-{view_index:02d}")
    return (
        np.asarray(state_vectors, dtype=np.float32),
        np.asarray(view_vectors, dtype=np.float32),
        np.asarray(state_standard_errors, dtype=np.float32),
        np.asarray(view_standard_errors, dtype=np.float32),
        state_ids,
        view_ids,
        probe_digest.hexdigest(),
    )


def _run_unit_oracle(
    linear_values: np.ndarray,
    linear_standard_error: np.ndarray,
    unit_ids: list[str],
    codebook_sizes: tuple[int, ...],
    *,
    seed: int,
    maximum_iterations: int,
) -> dict[str, Any]:
    channel_scale = np.maximum(
        np.quantile(linear_values.reshape(-1, 3), 0.99, axis=0), 1e-8
    ).astype(np.float32)
    tiled_scale = np.tile(channel_scale, linear_values.shape[1] // 3)
    transformed = np.log1p(
        np.maximum(linear_values, 0.0) / tiled_scale[None, :]
    ).astype(np.float32)
    transformed_standard_error = np.abs(linear_standard_error) / (
        tiled_scale[None, :] + np.maximum(linear_values, 0.0)
    )
    transformed_noise, transformed_noise_floor = _floored_relative_l1_rows(
        transformed_standard_error,
        transformed,
    )
    linear_noise, linear_noise_floor = _floored_relative_l1_rows(
        linear_standard_error,
        linear_values,
    )
    runs = []
    for cluster_count in codebook_sizes:
        if cluster_count >= len(transformed):
            continue
        centers, _, iterations = _fit_kmeans(
            transformed,
            cluster_count,
            seed=seed + cluster_count,
            maximum_iterations=maximum_iterations,
        )
        dictionary_reconstruction, indices, weights = _top2_reconstruct(
            transformed, centers
        )
        dictionary_bytes = int(4 * centers.size + indices.nbytes + weights.astype(np.float16).nbytes)
        pca_reconstruction, pca_rank, pca_bytes = _matched_pca(
            transformed, dictionary_bytes
        )
        _, dictionary_linear_rows, _ = _distortion_rows(
            transformed, dictionary_reconstruction, tiled_scale
        )
        _, pca_linear_rows, _ = _distortion_rows(
            transformed, pca_reconstruction, tiled_scale
        )
        runs.append({
            "codebook_size": cluster_count,
            "lloyd_iterations": iterations,
            "dictionary": {
                "shared_bytes": int(4 * centers.size),
                "asset_bytes": int(indices.nbytes + weights.astype(np.float16).nbytes),
                "total_bytes": dictionary_bytes,
                "distortion": _distortion(
                    transformed, dictionary_reconstruction, tiled_scale
                ),
            },
            "matched_pca": {
                "rank": pca_rank,
                "total_bytes": pca_bytes,
                "distortion": _distortion(transformed, pca_reconstruction, tiled_scale),
            },
            "paired_bootstrap": _paired_bootstrap(
                pca_linear_rows,
                dictionary_linear_rows,
                seed=seed + 10000 + cluster_count,
            ),
        })
    if not runs:
        raise ValueError("no requested M3 codebook size is smaller than the unit count")
    return {
        "unit_count": len(linear_values),
        "unit_dimension": int(linear_values.shape[1]),
        "unit_ids_sha256": _sha256_json(unit_ids),
        "transform": {
            "name": "global-channel-log1p-q99-v1",
            "channel_scale": channel_scale.tolist(),
            "fit_scope": "same canonical response units; direct-fit oracle only",
        },
        "reference_noise_floor": {
            "meaning": "one-standard-error relative L1; diagnostic, not a hard correction",
            "transformed_relative_l1": _summary(transformed_noise),
            "linear_relative_l1": _summary(linear_noise),
            "relative_l1_normalization": {
                "fraction_of_median_nonzero_reference_l1": 0.01,
                "transformed_l1_floor": transformed_noise_floor,
                "linear_l1_floor": linear_noise_floor,
            },
        },
        "runs": runs,
    }


def run_response_dictionary_oracle(
    corpus_path: Path | str,
    output_path: Path | str,
    *,
    codebook_sizes: tuple[int, ...] = (8, 16, 32, 64),
    seed: int = 20260824,
    maximum_iterations: int = 30,
) -> dict[str, Any]:
    requested = tuple(sorted(set(map(int, codebook_sizes))))
    if any(value < 2 or value > np.iinfo(np.uint16).max for value in requested):
        raise ValueError("M3 codebook sizes must fit nonzero uint16 IDs")
    if maximum_iterations < 1:
        raise ValueError("M3 maximum_iterations must be positive")
    with ReferenceCorpusStore(corpus_path) as store:
        (
            state,
            state_view,
            state_standard_error,
            state_view_standard_error,
            state_ids,
            view_ids,
            probe_sha256,
        ) = _canonical_dense_units(store)
        report: dict[str, Any] = {
            "schema": {"name": "m3-response-space-oracle", "version": 2},
            "role": "P1 diagnostic; not a quality-v1 candidate or runtime representation",
            "data_id": store.data_id,
            "query_role": "dense_slice",
            "probe_sha256": probe_sha256,
            "seed": seed,
            "maximum_iterations": maximum_iterations,
            "requested_codebook_sizes": list(requested),
            "units": {
                "per_state": _run_unit_oracle(
                    state,
                    state_standard_error,
                    state_ids,
                    requested,
                    seed=seed,
                    maximum_iterations=maximum_iterations,
                ),
                "per_state_view": _run_unit_oracle(
                    state_view,
                    state_view_standard_error,
                    view_ids,
                    requested,
                    seed=seed + 1000,
                    maximum_iterations=maximum_iterations,
                ),
            },
        }
    report["report_sha256"] = _sha256_json(report)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
