from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from ncls.core.material import material_program_from_layer_stack, pack_layer_stack

from .dataset import (
    INDEX_DTYPE,
    MATERIAL_STATE_DTYPE,
    ReferenceDataset,
    dataset_identity,
    make_response_dtype,
    resume_response_shard,
    write_common_files,
    write_manifest_atomic,
    write_response_shard,
)
from .directions import (
    DIRECTION_PARAMETERIZATION_ID,
    VIEW_PARAMETERIZATION_ID,
    equal_area_hemisphere,
    stratified_view_directions,
)
from .manifest import ReferenceDatasetManifest
from .priors import PRIOR_ID, PRIOR_VERSION, assign_family_splits, sample_stack_families
from .reference import (
    FalcorReferenceEvaluator,
    evaluate_reference_adaptive,
    evaluate_reference_fixed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_SOURCES = (
    PROJECT_ROOT / "shaders" / "ncls" / "contracts" / "layer_stack_ir.slang",
    PROJECT_ROOT / "shaders" / "ncls" / "reference" / "sampling.slang",
    PROJECT_ROOT / "shaders" / "ncls" / "reference" / "interfaces.slang",
    PROJECT_ROOT / "shaders" / "ncls" / "reference" / "random_walk_reference.slang",
    PROJECT_ROOT / "shaders" / "ncls" / "data" / "reference_tile.cs.slang",
    PROJECT_ROOT / "src" / "ncls" / "core" / "material" / "abi" / "layer_stack_ir_v1.json",
)


@dataclass(frozen=True)
class ReferenceGenerationConfig:
    family_count: int = 8
    local_state_count: int = 4
    view_count: int = 4
    light_count: int = 128
    samples_per_replica: int = 64
    tile_batch: int = 64
    shard_tiles: int = 65536
    seed: int = 20260822
    max_depth: int = 64
    adaptive: bool = False
    batch_samples: int = 256
    min_samples: int = 512
    max_samples: int = 16384
    relative_standard_error: float = 0.03

    def __post_init__(self) -> None:
        positive = (
            self.family_count,
            self.local_state_count,
            self.view_count,
            self.light_count,
            self.samples_per_replica,
            self.tile_batch,
            self.shard_tiles,
            self.max_depth,
            self.batch_samples,
            self.min_samples,
            self.max_samples,
        )
        if min(positive) < 1 or self.seed < 0:
            raise ValueError("reference generation sizes must be positive and seed nonnegative")
        if self.max_samples < self.min_samples:
            raise ValueError("max_samples must not be smaller than min_samples")
        if self.adaptive and (self.min_samples % self.batch_samples or self.max_samples % self.batch_samples):
            raise ValueError("adaptive sample limits must be multiples of batch_samples")
        if not 0.0 < self.relative_standard_error < 1.0:
            raise ValueError("relative_standard_error must lie in (0, 1)")
        maximum_per_replica = self.max_samples if self.adaptive else self.samples_per_replica
        if 2 * maximum_per_replica > np.iinfo(np.uint32).max:
            raise ValueError("total sample count must fit uint32")


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reference_source_hash() -> str:
    digest = hashlib.sha256()
    for path in REFERENCE_SOURCES:
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unknown"


def _tile_seeds(seed: int, tile_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(tile_indices, dtype=np.uint64)
    mixed = (values * np.uint64(0x9E3779B1) + np.uint64(seed & 0xFFFFFFFF)) & np.uint64(0xFFFFFFFF)
    return mixed.astype(np.uint32)


def generate_reference_dataset(
    output: Path | str,
    config: ReferenceGenerationConfig,
    *,
    resume: bool = False,
    evaluator: Any | None = None,
    created_at: str | None = None,
    generator_git_commit: str | None = None,
) -> ReferenceDatasetManifest:
    """生成唯一的 v2 参考数据格式；测试可注入与 Falcor evaluator 同合同的实现。"""

    output_path = Path(output)
    families = sample_stack_families(config.family_count, config.local_state_count, config.seed)
    materials = [material for family in families for material in family]
    programs = [
        material_program_from_layer_stack(
            material,
            metadata={
                "family_index": index // config.local_state_count,
                "local_state_index": index % config.local_state_count,
            },
        )
        for index, material in enumerate(materials)
    ]
    canonical_irs = [pack_layer_stack(material) for material in materials]
    family_splits = assign_family_splits(config.family_count, config.seed)
    states = np.zeros(len(materials), dtype=MATERIAL_STATE_DTYPE)
    states["family_index"] = np.repeat(
        np.arange(config.family_count, dtype=np.uint32), config.local_state_count
    )
    states["local_state_index"] = np.tile(
        np.arange(config.local_state_count, dtype=np.uint32), config.family_count
    )
    states["program_index"] = np.arange(len(materials), dtype=np.uint32)
    states["canonical_ir_index"] = np.arange(len(materials), dtype=np.uint32)
    states["split"] = family_splits[states["family_index"]]
    views = stratified_view_directions(config.view_count)
    lights, weights = equal_area_hemisphere(config.light_count)
    files, content_hashes = write_common_files(
        output_path,
        material_programs=programs,
        canonical_material_irs=canonical_irs,
        material_states=states,
        family_splits=family_splits,
        view_directions=views,
        light_directions=lights,
        solid_angle_weights=weights,
        reuse_identical=resume,
    )

    tile_count = len(materials) * len(views)
    response_dtype = make_response_dtype(len(lights))
    shard_records = []
    active_evaluator = evaluator
    for shard_id, shard_start in enumerate(range(0, tile_count, config.shard_tiles)):
        shard_end = min(shard_start + config.shard_tiles, tile_count)
        shard_count = shard_end - shard_start
        if resume:
            completed = resume_response_shard(
                output_path,
                shard_id=shard_id,
                tile_start=shard_start,
                tile_count=shard_count,
            )
            if completed is not None:
                shard_records.append(completed)
                continue
        if active_evaluator is None:
            active_evaluator = FalcorReferenceEvaluator(
                lights,
                max_depth=config.max_depth,
                max_tile_batch=config.tile_batch,
            )
        if int(active_evaluator.light_count) != len(lights):
            raise ValueError("reference evaluator light count disagrees with generation config")
        evaluator_lights = getattr(active_evaluator, "light_directions", None)
        if evaluator_lights is not None and not np.array_equal(
            np.asarray(evaluator_lights, dtype=np.float32), lights
        ):
            raise ValueError("reference evaluator directions disagree with persisted light directions")
        index = np.zeros(shard_count, dtype=INDEX_DTYPE)
        response = np.zeros(shard_count, dtype=response_dtype)
        for batch_start in range(shard_start, shard_end, config.tile_batch):
            batch_end = min(batch_start + config.tile_batch, shard_end)
            global_indices = np.arange(batch_start, batch_end, dtype=np.uint64)
            state_indices = (global_indices // config.view_count).astype(np.int64)
            view_indices = (global_indices % config.view_count).astype(np.int64)
            batch_materials = [materials[index] for index in state_indices]
            batch_views = views[view_indices]
            seeds = _tile_seeds(config.seed, global_indices)
            if config.adaptive:
                evaluated = evaluate_reference_adaptive(
                    active_evaluator,
                    batch_materials,
                    batch_views,
                    tile_seeds=seeds,
                    batch_samples=config.batch_samples,
                    min_samples=config.min_samples,
                    max_samples=config.max_samples,
                    relative_standard_error=config.relative_standard_error,
                )
            else:
                evaluated = evaluate_reference_fixed(
                    active_evaluator,
                    batch_materials,
                    batch_views,
                    tile_seeds=seeds,
                    samples_per_replica=config.samples_per_replica,
                )
            local_start = batch_start - shard_start
            local_end = batch_end - shard_start
            response["mean"][local_start:local_end] = evaluated.mean.astype(np.float32)
            response["variance"][local_start:local_end] = evaluated.variance.astype(np.float32)
            response["replica_mean_a"][local_start:local_end] = evaluated.replica_mean_a.astype(np.float32)
            response["replica_mean_b"][local_start:local_end] = evaluated.replica_mean_b.astype(np.float32)
            response["sample_count"][local_start:local_end] = evaluated.sample_count[:, None]
            index["tile_id"][local_start:local_end] = global_indices
            index["material_state_index"][local_start:local_end] = state_indices
            index["view_index"][local_start:local_end] = view_indices
            index["family_index"][local_start:local_end] = states["family_index"][state_indices]
            index["split"][local_start:local_end] = states["split"][state_indices]
        shard_records.append(
            write_response_shard(
                output_path,
                shard_id=shard_id,
                tile_start=shard_start,
                index=index,
                response=response,
            )
        )

    source_hash = reference_source_hash()
    config_dict = asdict(config)
    config_hash = _canonical_json_hash(config_dict)
    identity_parts = [
        source_hash.encode("ascii"),
        config_hash.encode("ascii"),
        *[digest.encode("ascii") for _, digest in sorted(content_hashes.items())],
        *[record.index_sha256.encode("ascii") for record in shard_records],
        *[record.response_sha256.encode("ascii") for record in shard_records],
    ]
    manifest = ReferenceDatasetManifest(
        dataset_id=dataset_identity(identity_parts),
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        reference_implementation_id=f"random-walk-reference@1:{source_hash[:16]}",
        reference_source_sha256=source_hash,
        generator_git_commit=generator_git_commit or _git_commit(),
        prior_id=PRIOR_ID,
        prior_version=PRIOR_VERSION,
        resolved_config_sha256=config_hash,
        seed=config.seed,
        direction_parameterization={
            "light": DIRECTION_PARAMETERIZATION_ID,
            "view": VIEW_PARAMETERIZATION_ID,
            "solid_angle_measure": "steradian",
        },
        counts={
            "family_count": config.family_count,
            "material_state_count": len(materials),
            "view_count": len(views),
            "light_count": len(lights),
            "tile_count": tile_count,
        },
        shapes={
            "view_directions": list(views.shape),
            "light_directions": list(lights.shape),
            "response_per_tile": [len(lights), 3],
        },
        statistics_encoding={
            "mean": "float32",
            "variance": "float32",
            "replicas": "float32",
            "sample_count": "uint32-total-across-replicas",
            "uncertainty_kind": "sample-population-variance",
            "variance_definition": "population variance across both independent random streams",
        },
        quantization={
            "encoding": "float32",
            "lossy": False,
            "declared_additional_absolute_error": 0.0,
        },
        split_policy={
            "unit": "family",
            "names": ["train", "validation", "test"],
            "codes": {"train": 0, "validation": 1, "test": 2},
        },
        files=files,
        content_hashes=content_hashes,
        shards=tuple(shard_records),
        extras={
            "generation_config": config_dict,
            "reference_settings": {
                "max_depth": config.max_depth,
                "independent_stream_count": 2,
                "next_event_estimation": True,
                "multiple_importance_sampling": "power-heuristic",
                "transport_scope": "local-reflection-opaque-base-v1",
            },
        },
    )
    write_manifest_atomic(output_path, manifest)
    ReferenceDataset.open(output_path, verify_hashes=True)
    return manifest
