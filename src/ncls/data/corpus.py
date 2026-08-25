from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from ncls.paths import PROJECT_ROOT

from .collector import collect_reference_dataset
from .contract import SourceState
from .dataset import ReferenceDataset, ReferenceDatasetManifest
from .profiles import CorpusPlan
from .providers.layer_stack import LayerStackProvider, LayerStackProviderConfig
from .selection import CorpusSelection


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CorpusShard:
    shard_id: str
    uri: str
    role: str
    structure_family_id: str
    difficulty_class: str
    difficulty_tags: tuple[str, ...]
    state_ids: tuple[str, ...]
    view_count: int
    direction_count: int
    status: str = "planned"
    dataset_id: str | None = None
    sha256: str | None = None
    seconds: float | None = None
    combined_reference_samples: int | None = None
    reciprocal_combined_reference_samples: int | None = None


@dataclass(frozen=True)
class ReferenceCorpusManifest:
    name: str
    plan: Mapping[str, Any]
    plan_sha256: str
    created_at: str
    shards: tuple[CorpusShard, ...]
    corpus_id: str | None = None
    selection: Mapping[str, Any] | None = None
    selection_sha256: str | None = None
    format_name: str = "reference-corpus"
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_name != "reference-corpus" or self.format_version not in {1, 2}:
            raise ValueError("unsupported reference corpus manifest")
        if self.format_version == 1 and (
            self.selection is not None or self.selection_sha256 is not None
        ):
            raise ValueError("reference-corpus v1 cannot contain a selection")
        if self.format_version == 2 and (
            self.selection is None or self.selection_sha256 is None
        ):
            raise ValueError("reference-corpus v2 requires an embedded selection")

    def payload(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if self.format_version == 1:
            value.pop("selection")
            value.pop("selection_sha256")
        value["totals"] = {
            "seconds": float(sum(shard.seconds or 0.0 for shard in self.shards)),
            "combined_reference_samples": int(sum(
                shard.combined_reference_samples or 0 for shard in self.shards
            )),
            "reciprocal_combined_reference_samples": int(sum(
                shard.reciprocal_combined_reference_samples or 0 for shard in self.shards
            )),
        }
        if not include_identity:
            value.pop("corpus_id")
            value.pop("created_at")
            for shard in value["shards"]:
                shard.pop("seconds")
                shard.pop("shard_id")
                shard.pop("uri")
                # 文件 hash 用于定位字节损坏；语义身份由 dataset_id 与计划决定。
                shard.pop("sha256")
            value["totals"].pop("seconds")
        return value

    def resolved(self) -> "ReferenceCorpusManifest":
        corpus_id = hashlib.sha256(
            _canonical_json(self.payload(include_identity=False)).encode("utf-8")
        ).hexdigest()
        return replace(self, corpus_id=corpus_id)

    def write(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        value = self.resolved()
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(
            json.dumps(value.payload(), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)

    @classmethod
    def load(cls, path: Path | str) -> "ReferenceCorpusManifest":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        format_version = value.get("format_version")
        expected_fields = {
            "name", "plan", "plan_sha256", "created_at", "shards", "corpus_id",
            "format_name", "format_version", "totals",
        }
        if format_version == 2:
            expected_fields |= {"selection", "selection_sha256"}
        if set(value) != expected_fields:
            raise ValueError("reference corpus manifest fields do not match its version")
        if value.get("format_name") != "reference-corpus" or format_version not in {1, 2}:
            raise ValueError("unsupported reference corpus manifest")
        shards = tuple(CorpusShard(
            **{
                **item,
                "difficulty_tags": tuple(item["difficulty_tags"]),
                "state_ids": tuple(item["state_ids"]),
            }
        ) for item in value["shards"])
        manifest = cls(
            name=value["name"],
            plan=dict(value["plan"]),
            plan_sha256=value["plan_sha256"],
            created_at=value["created_at"],
            shards=shards,
            corpus_id=value.get("corpus_id"),
            selection=(
                dict(value["selection"])
                if format_version == 2 else None
            ),
            selection_sha256=(
                str(value["selection_sha256"])
                if format_version == 2 else None
            ),
            format_version=int(format_version),
        )
        if manifest.resolved().corpus_id != manifest.corpus_id:
            raise ValueError("reference corpus identity mismatch")
        if hashlib.sha256(
            _canonical_json(manifest.plan).encode("utf-8")
        ).hexdigest() != manifest.plan_sha256:
            raise ValueError("reference corpus plan hash mismatch")
        if manifest.format_version == 2 and hashlib.sha256(
            _canonical_json(manifest.selection).encode("utf-8")
        ).hexdigest() != manifest.selection_sha256:
            raise ValueError("reference corpus selection hash mismatch")
        if value["totals"] != manifest.payload()["totals"]:
            raise ValueError("reference corpus totals disagree with shard records")
        return manifest


def _project_uri(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_uri(uri: str) -> Path:
    path = Path(uri)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sample_spending(dataset: ReferenceDataset) -> tuple[int, int]:
    return (
        int(np.sum(dataset.stream["responses/sample_count"], dtype=np.uint64)),
        int(np.sum(dataset.stream["responses/reciprocal_sample_count"], dtype=np.uint64)),
    )


def _layer_stack_provider_config(
    plan: CorpusPlan,
    *,
    adaptive: bool,
    selected_state_ids: tuple[str, ...] = (),
    query_role: str | None = None,
) -> LayerStackProviderConfig:
    relative_standard_error = float(plan.reference_budget["target_relative_se_p95"])
    maximum_group_relative_standard_error = float(
        plan.reference_budget["maximum_query_group_relative_se_p95"]
    )
    maximum_combined_samples = int(plan.reference_budget["maximum_combined_samples"])
    enforce_maximum_group_relative_standard_error = True
    if query_role == "train":
        relative_standard_error = float(
            plan.reference_budget["training_target_relative_se_p95"]
        )
        maximum_group_relative_standard_error = float(
            plan.reference_budget["training_maximum_query_group_relative_se_p95"]
        )
        maximum_combined_samples = int(
            plan.reference_budget["training_maximum_combined_samples"]
        )
    elif query_role in set(plan.reference_budget["diagnostic_query_roles"]):
        relative_standard_error = float(
            plan.reference_budget["diagnostic_target_relative_se_p95"]
        )
        maximum_group_relative_standard_error = float(
            plan.reference_budget["diagnostic_maximum_query_group_relative_se_p95"]
        )
        maximum_combined_samples = int(
            plan.reference_budget["diagnostic_maximum_combined_samples"]
        )
        enforce_maximum_group_relative_standard_error = False
    if selected_state_ids and query_role is not None:
        promotions = {
            str(item["state_id"]): item
            for item in plan.reference_budget["state_sample_promotions"]
            if query_role in item["query_roles"]
        }
        for state_id in selected_state_ids:
            promotion = promotions.get(state_id)
            if promotion is None:
                continue
            maximum_combined_samples = max(
                maximum_combined_samples,
                int(promotion["maximum_combined_samples"]),
            )
            maximum_group_relative_standard_error = max(
                maximum_group_relative_standard_error,
                float(promotion["maximum_query_group_relative_se_p95"]),
            )
    return LayerStackProviderConfig(
        family_count=int(plan.provider["family_count"]),
        states_per_family=int(plan.provider["states_per_family"]),
        heldout_family_count=int(plan.split["heldout_family_count"]),
        state_profile=str(plan.provider["state_profile"]),
        max_depth=int(plan.provider["maximum_path_depth"]),
        adaptive=adaptive,
        batch_samples_per_replica=int(plan.reference_budget["batch_samples_per_replica"]),
        min_combined_samples=int(plan.reference_budget["minimum_combined_samples"]),
        max_combined_samples=maximum_combined_samples,
        relative_standard_error=relative_standard_error,
        maximum_group_relative_standard_error=maximum_group_relative_standard_error,
        enforce_maximum_group_relative_standard_error=(
            enforce_maximum_group_relative_standard_error
        ),
        max_dispatch_queries=int(plan.reference_budget["maximum_dispatch_queries"]),
        peak_calibration_directions=int(
            plan.document["sampling"]["moving_peak"]["probe_directions"]
        ),
        peak_calibration_samples_per_replica=int(
            plan.document["sampling"]["moving_peak"]["samples_per_replica"]
        ),
        selected_state_ids=selected_state_ids,
    )


def select_layer_stack_state(
    plan: CorpusPlan,
    structure_family_id: str,
    local_state_index: int,
) -> SourceState:
    """用可读 family 与族内 index 解析一个确定的 LayerStack v1 state。"""

    if local_state_index < 0:
        raise ValueError("local state index must be nonnegative")
    discovery = LayerStackProvider(
        plan.collection_config("W", (), "test"),
        _layer_stack_provider_config(plan, adaptive=False),
    )
    asset_id = f"{structure_family_id}/state-{local_state_index:04d}"
    matches = [state for state in discovery.source_states() if state.asset_id == asset_id]
    if len(matches) != 1:
        families = sorted({state.structure_family_id for state in discovery.source_states()})
        if structure_family_id not in families:
            raise ValueError(
                f"unknown LayerStack structure family {structure_family_id!r}; "
                f"available examples: {families[:4]}"
            )
        raise ValueError(
            f"LayerStack state index {local_state_index} is absent from {structure_family_id!r}"
        )
    return matches[0]


def collect_layer_stack_state(
    plan: CorpusPlan,
    structure_family_id: str,
    local_state_index: int,
    role: str,
    output: Path | str,
) -> tuple[SourceState, ReferenceDatasetManifest]:
    """采集一个正式密度的单-state shard；它用于 smoke，不冒充完整 corpus。"""

    state = select_layer_stack_state(plan, structure_family_id, local_state_index)
    collection = plan.collection_config(
        state.difficulty_class,
        state.difficulty_tags,
        role,
        state.state_id,
    )
    provider = LayerStackProvider(
        collection,
        _layer_stack_provider_config(
            plan,
            adaptive=True,
            selected_state_ids=(state.state_id,),
            query_role=role,
        ),
    )
    manifest = collect_reference_dataset(output, (provider,), collection)
    return state, manifest


def plan_layer_stack_corpus(
    plan: CorpusPlan,
    shard_root: Path | str,
    selection: CorpusSelection | None = None,
) -> ReferenceCorpusManifest:
    if plan.provider.get("name") != "layer-stack":
        raise ValueError("the first v1 corpus planner supports LayerStack only")
    provider_config = _layer_stack_provider_config(plan, adaptive=False)
    discovery = LayerStackProvider(
        plan.collection_config("W", (), "test"),
        provider_config,
    )
    all_states = tuple(discovery.source_states())
    known_state_ids = {state.state_id for state in all_states}
    unknown_promotions = sorted(
        set(plan.document["sampling"]["dense_promotions"]) - known_state_ids
    )
    if unknown_promotions:
        raise ValueError(
            "dense_promotions contains state IDs outside this CorpusPlan: "
            f"{unknown_promotions[:4]}"
        )
    unknown_sample_promotions = sorted(
        {
            str(item["state_id"])
            for item in plan.reference_budget["state_sample_promotions"]
        }
        - known_state_ids
    )
    if unknown_sample_promotions:
        raise ValueError(
            "state_sample_promotions contains state IDs outside this CorpusPlan: "
            f"{unknown_sample_promotions[:4]}"
        )
    if selection is not None and selection.base_corpus != plan.name:
        raise ValueError("corpus selection targets a different base CorpusPlan")
    states = (
        selection.select_states(all_states)
        if selection is not None else all_states
    )
    if selection is None and sum(state.split == 2 for state in states) < int(plan.split["minimum_test_state_count"]):
        raise ValueError("CorpusPlan does not provide the required number of test states")
    grouped: dict[tuple[str, str, str, tuple[str, ...]], list[Any]] = {}
    for state in states:
        for role in ("train", "validation", "test", "adversarial_probe", "dense_slice"):
            density_key = (
                "S"
                if role == "train" and "M" in state.difficulty_tags
                else state.difficulty_class if role == "train"
                else (
                    "promoted"
                    if role == "dense_slice"
                    and state.state_id in plan.document["sampling"]["dense_promotions"]
                    else "base" if role == "dense_slice" else "all"
                )
            )
            key = (
                state.structure_family_id,
                role,
                density_key,
                tuple(state.difficulty_tags) if role == "train" else (),
            )
            grouped.setdefault(key, []).append(state)
    corpus_name = selection.name if selection is not None else plan.name
    root = Path(shard_root)
    shards = []
    for (family, role, density_key, tags), selected in sorted(grouped.items()):
        difficulty = density_key if role == "train" else "W"
        resolved = plan.resolve_query(difficulty, tags, role, selected[0].state_id)
        tag_suffix = "-" + "".join(tags).lower() if tags else ""
        shard_id = f"{family}-{role}-{density_key.lower()}{tag_suffix}"
        shards.append(CorpusShard(
            shard_id=shard_id,
            uri=_project_uri(root / corpus_name / f"{shard_id}.h5"),
            role=role,
            structure_family_id=family,
            difficulty_class=density_key,
            difficulty_tags=tags,
            state_ids=tuple(state.state_id for state in selected),
            view_count=resolved.views,
            direction_count=resolved.directions,
        ))
    return ReferenceCorpusManifest(
        name=corpus_name,
        plan=plan.document,
        plan_sha256=plan.sha256,
        created_at=datetime.now(timezone.utc).isoformat(),
        shards=tuple(shards),
        selection=selection.document if selection is not None else None,
        selection_sha256=selection.sha256 if selection is not None else None,
        format_version=2 if selection is not None else 1,
    ).resolved()


def collect_layer_stack_corpus(
    plan: CorpusPlan,
    shard_root: Path | str,
    manifest_path: Path | str,
    selection: CorpusSelection | None = None,
) -> ReferenceCorpusManifest:
    manifest = plan_layer_stack_corpus(plan, shard_root, selection)
    previous_shards: dict[str, CorpusShard] = {}
    previous_path = Path(manifest_path)
    if previous_path.is_file():
        previous = ReferenceCorpusManifest.load(previous_path)
        if previous.plan_sha256 != plan.sha256:
            raise ValueError("existing corpus manifest belongs to a different CorpusPlan")
        if previous.selection_sha256 != manifest.selection_sha256:
            raise ValueError("existing corpus manifest belongs to a different selection")
        previous_shards = {shard.shard_id: shard for shard in previous.shards}
        manifest = replace(manifest, created_at=previous.created_at)
    completed: list[CorpusShard] = []
    base_provider = _layer_stack_provider_config(plan, adaptive=True)
    state_lookup: dict[str, Any] | None = None
    for shard in manifest.shards:
        output = _resolve_uri(shard.uri)
        if output.is_file():
            try:
                with ReferenceDataset.open(output, verify_hashes=True) as dataset:
                    expected_role = ("train", "validation", "test", "adversarial_probe", "dense_slice").index(shard.role)
                    if (
                        set(dataset.state_strings("state_id")) == set(shard.state_ids)
                        and dataset.manifest.sampling_name == plan.sampling_name
                        and dataset.direction_count == shard.direction_count
                        and dataset.query_group_count == len(shard.state_ids) * shard.view_count
                        and np.all(dataset.query_roles == expected_role)
                    ):
                        original_samples, reciprocal_samples = _sample_spending(dataset)
                        previous_shard = previous_shards.get(shard.shard_id)
                        completed.append(replace(
                            shard,
                            status="complete",
                            dataset_id=dataset.manifest.dataset_id,
                            sha256=_sha256_file(output),
                            seconds=(
                                previous_shard.seconds
                                if previous_shard is not None
                                and previous_shard.dataset_id == dataset.manifest.dataset_id
                                else 0.0
                            ),
                            combined_reference_samples=original_samples,
                            reciprocal_combined_reference_samples=reciprocal_samples,
                        ))
                        continue
            except (OSError, ValueError):
                pass
            raise ValueError(f"existing shard is not the verified planned file: {output}")
        if state_lookup is None:
            discovery = LayerStackProvider(
                plan.collection_config("W", (), "test"), base_provider
            )
            state_lookup = {state.state_id: state for state in discovery.source_states()}
        first = state_lookup[shard.state_ids[0]]
        collection = plan.collection_config(
            first.difficulty_class,
            shard.difficulty_tags,
            shard.role,
            first.state_id,
        )
        provider = LayerStackProvider(
            collection,
            _layer_stack_provider_config(
                plan,
                adaptive=True,
                selected_state_ids=shard.state_ids,
                query_role=shard.role,
            ),
        )
        started = time.perf_counter()
        dataset_manifest = collect_reference_dataset(output, (provider,), collection)
        with ReferenceDataset.open(output, verify_hashes=True) as dataset:
            original_samples, reciprocal_samples = _sample_spending(dataset)
        completed.append(replace(
            shard,
            status="complete",
            dataset_id=dataset_manifest.dataset_id,
            sha256=_sha256_file(output),
            seconds=time.perf_counter() - started,
            combined_reference_samples=original_samples,
            reciprocal_combined_reference_samples=reciprocal_samples,
        ))
        ReferenceCorpusManifest(
            name=manifest.name,
            plan=manifest.plan,
            plan_sha256=manifest.plan_sha256,
            created_at=manifest.created_at,
            shards=tuple(completed) + manifest.shards[len(completed):],
            selection=manifest.selection,
            selection_sha256=manifest.selection_sha256,
            format_version=manifest.format_version,
        ).write(manifest_path)
    result = ReferenceCorpusManifest(
        name=manifest.name,
        plan=manifest.plan,
        plan_sha256=manifest.plan_sha256,
        created_at=manifest.created_at,
        shards=tuple(completed),
        selection=manifest.selection,
        selection_sha256=manifest.selection_sha256,
        format_version=manifest.format_version,
    ).resolved()
    result.write(manifest_path)
    return validate_reference_corpus(manifest_path)


def validate_reference_corpus(path: Path | str) -> ReferenceCorpusManifest:
    manifest = ReferenceCorpusManifest.load(path)
    plan = CorpusPlan.from_dict(manifest.plan)
    selection = (
        CorpusSelection.from_dict(manifest.selection)
        if manifest.selection is not None else None
    )
    if not manifest.shards or any(shard.status != "complete" for shard in manifest.shards):
        raise ValueError("reference corpus is incomplete")
    if len({shard.shard_id for shard in manifest.shards}) != len(manifest.shards):
        raise ValueError("reference corpus contains duplicate shard IDs")
    if len({shard.uri for shard in manifest.shards}) != len(manifest.shards):
        raise ValueError("reference corpus contains duplicate shard URIs")
    expected_manifest = plan_layer_stack_corpus(
        plan,
        PROJECT_ROOT / ".corpus-layout-check",
        selection,
    )
    expected_shards = {shard.shard_id: shard for shard in expected_manifest.shards}
    if set(expected_shards) != {shard.shard_id for shard in manifest.shards}:
        raise ValueError("reference corpus shard set disagrees with CorpusPlan")
    layout_fields = (
        "role", "structure_family_id", "difficulty_class", "difficulty_tags",
        "state_ids", "view_count", "direction_count",
    )
    for shard in manifest.shards:
        expected = expected_shards[shard.shard_id]
        if any(getattr(shard, name) != getattr(expected, name) for name in layout_fields):
            raise ValueError(
                f"reference corpus shard layout disagrees with CorpusPlan: {shard.shard_id}"
            )
    state_records: dict[str, tuple[Any, ...]] = {}
    direction_hashes: dict[str, set[str]] = {}
    state_roles: dict[str, set[str]] = {}
    split_group_splits: dict[str, set[int]] = {}
    source_hash_splits: dict[str, set[int]] = {}
    for shard in manifest.shards:
        expected_density = plan.resolve_query(
            shard.difficulty_class if shard.role == "train" else "W",
            shard.difficulty_tags,
            shard.role,
            shard.state_ids[0],
        )
        if (shard.view_count, shard.direction_count) != (
            expected_density.views,
            expected_density.directions,
        ):
            raise ValueError(f"reference corpus shard disagrees with CorpusPlan density: {shard.shard_id}")
        target = _resolve_uri(shard.uri)
        if not target.is_file() or _sha256_file(target) != shard.sha256:
            raise ValueError(f"reference corpus shard hash mismatch: {shard.uri}")
        with ReferenceDataset.open(target, verify_hashes=True) as dataset:
            if dataset.manifest.dataset_id != shard.dataset_id:
                raise ValueError(f"reference corpus shard dataset identity mismatch: {shard.uri}")
            if dataset.manifest.sampling_name != plan.sampling_name:
                raise ValueError(f"reference corpus shard sampling mismatch: {shard.uri}")
            if dataset.direction_count != shard.direction_count:
                raise ValueError(f"reference corpus shard direction count mismatch: {shard.uri}")
            state_ids = tuple(map(str, dataset.state_strings("state_id").tolist()))
            if set(state_ids) != set(shard.state_ids):
                raise ValueError(f"reference corpus shard state set mismatch: {shard.uri}")
            if dataset.query_group_count != len(state_ids) * shard.view_count:
                raise ValueError(f"reference corpus shard view count mismatch: {shard.uri}")
            expected_role = ("train", "validation", "test", "adversarial_probe", "dense_slice").index(shard.role)
            if np.any(dataset.query_roles != expected_role):
                raise ValueError(f"reference corpus shard query role mismatch: {shard.uri}")
            structure = dataset.state_strings("structure_family_id")
            classes = dataset.state_strings("difficulty_class")
            tags = dataset.state_strings("difficulty_tags_json")
            if np.any(structure != shard.structure_family_id):
                raise ValueError(f"reference corpus shard structure family mismatch: {shard.uri}")
            if shard.role == "train":
                effective = np.asarray([
                    "S" if "M" in json.loads(str(value)) else str(classes[index])
                    for index, value in enumerate(tags)
                ])
                if np.any(effective != shard.difficulty_class):
                    raise ValueError(f"reference corpus shard difficulty density mismatch: {shard.uri}")
            splits = np.asarray(dataset.stream["states/split"], dtype=np.uint8)
            metadata_fields = {
                name: dataset.state_strings(name)
                for name in (
                    "family_id", "reference_id", "asset_id", "split_group_id",
                    "native_schema_id", "source_uri", "source_sha256", "parent_state_id",
                    "evaluation_cohort",
                )
            }
            for local_index, state_id in enumerate(state_ids):
                state_roles.setdefault(state_id, set()).add(shard.role)
                split_value = int(splits[local_index])
                split_group_splits.setdefault(
                    str(metadata_fields["split_group_id"][local_index]), set()
                ).add(split_value)
                source_hash_splits.setdefault(
                    str(metadata_fields["source_sha256"][local_index]), set()
                ).add(split_value)
                record = (
                    split_value,
                    str(structure[local_index]),
                    str(classes[local_index]),
                    str(tags[local_index]),
                    *(
                        str(metadata_fields[name][local_index])
                        for name in metadata_fields
                    ),
                )
                previous = state_records.setdefault(state_id, record)
                if previous != record:
                    raise ValueError(f"state metadata changes across corpus shards: {state_id}")
                group_indices = dataset.group_indices()[
                    np.asarray(dataset.stream["queries/state_index"], dtype=np.int64) == local_index
                ]
                digest = hashlib.sha256()
                digest.update(np.ascontiguousarray(dataset.stream["queries/wo"][group_indices]).tobytes())
                digest.update(np.ascontiguousarray(dataset.stream["queries/wi"][group_indices]).tobytes())
                role_hashes = direction_hashes.setdefault(state_id, set())
                value = digest.hexdigest()
                if value in role_hashes:
                    raise ValueError(f"query directions collide across roles for state {state_id}")
                role_hashes.add(value)
    test_state_count = sum(record[0] == 2 for record in state_records.values())
    if selection is None and test_state_count < int(plan.split["minimum_test_state_count"]):
        raise ValueError("reference corpus has too few test states")
    if any(len(splits) != 1 for splits in split_group_splits.values()):
        raise ValueError("reference corpus leaks a split_group_id across source splits")
    if any(len(splits) != 1 for splits in source_hash_splits.values()):
        raise ValueError("reference corpus leaks source content across source splits")
    required_roles = {"train", "validation", "test", "adversarial_probe", "dense_slice"}
    incomplete = sorted(
        state_id for state_id, roles in state_roles.items() if roles != required_roles
    )
    if incomplete:
        raise ValueError(f"reference corpus states are missing query roles: {incomplete[:4]}")
    return manifest


def audit_dense_slice_resolution(
    path: Path | str,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """从实际 dense response 计算峰邻域方向间距，并给出 16,384 晋级清单。"""

    manifest = validate_reference_corpus(path)
    state_rows: dict[str, list[tuple[float, float, int]]] = {}
    for shard in manifest.shards:
        if shard.role != "dense_slice":
            continue
        with ReferenceDataset.open(_resolve_uri(shard.uri), verify_hashes=True) as dataset:
            state_ids = dataset.state_strings("state_id")
            for group_index in range(dataset.query_group_count):
                batch = dataset.group_batch((group_index,))
                state_id = str(state_ids[int(batch["state_index"][0])])
                directions = np.asarray(batch["wi"][0], dtype=np.float64)
                contribution = np.sum(np.abs(batch["mean"][0]), axis=-1) * np.asarray(
                    batch["solid_angle_weight"][0], dtype=np.float64
                )
                top_count = max(1, int(np.ceil(0.01 * len(contribution))))
                top = np.argpartition(contribution, -top_count)[-top_count:]
                concentration = float(
                    np.sum(contribution[top]) / max(np.sum(contribution), 1e-12)
                )
                peak = int(np.argmax(contribution))
                cosine = directions @ directions[peak]
                cosine[peak] = -1.0
                spacing = float(np.degrees(np.arccos(np.clip(np.max(cosine), -1.0, 1.0))))
                state_rows.setdefault(state_id, []).append(
                    (concentration, spacing, dataset.direction_count)
                )
    states = {}
    promote = []
    for state_id, rows in sorted(state_rows.items()):
        concentrated_spacing = [
            spacing for concentration, spacing, _ in rows if concentration >= 0.10
        ]
        spacing_p95 = (
            float(np.quantile(concentrated_spacing, 0.95))
            if concentrated_spacing else 0.0
        )
        direction_count = max(row[2] for row in rows)
        needs_promotion = direction_count == 8192 and spacing_p95 > 2.0
        if needs_promotion:
            promote.append(state_id)
        states[state_id] = {
            "direction_count": direction_count,
            "maximum_top_1_percent_energy_fraction": max(row[0] for row in rows),
            "concentrated_peak_neighbor_spacing_p95_degrees": spacing_p95,
            "promote_to_16384": needs_promotion,
        }
    report: dict[str, Any] = {
        "format_name": "dense-slice-audit",
        "format_version": 1,
        "data_id": manifest.corpus_id,
        "concentrated_query_threshold": 0.10,
        "peak_spacing_p95_limit_degrees": 2.0,
        "default_direction_count": 8192,
        "promoted_direction_count": 16384,
        "promote_state_ids": promote,
        "corpus_plan_update": {
            "field": "sampling.dense_promotions",
            "value": promote,
        },
        "states": states,
    }
    payload = _canonical_json(report)
    report["report_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    return report
