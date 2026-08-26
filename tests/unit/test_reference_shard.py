from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch

from ncls.data import (
    CollectionConfig,
    EvaluatedBlock,
    PositionKind,
    QueryPlan,
    QueryRole,
    QUERY_ROLE_NAMES,
    CorpusShard,
    ReferenceCorpusManifest,
    ReferenceDataset,
    ReferenceDescriptor,
    SourceState,
    SurfaceSample,
    collect_reference_dataset,
    peak_grazing_mixture_pdf,
    peak_grazing_mixture_query,
)
from ncls.core.source import SourceSnapshot
from ncls.data.collector import _reciprocal_block


@dataclass
class _Provider:
    config: CollectionConfig

    def __post_init__(self) -> None:
        self.descriptor = ReferenceDescriptor(
            family_id="test-family",
            reference_id="test-reference",
            native_schema_id="test-state",
            implementation_sha256="a" * 64,
        )
        payload = b'{"value":1}'
        source_hash = hashlib.sha256(payload).hexdigest()
        snapshot = SourceSnapshot(
            "test-family", 1, "test-state", source_hash, payload,
        )
        self.state = SourceState(
            snapshot=snapshot,
            reference_id="test-reference",
            asset_id="asset-01",
            split_group_id="asset-01",
            source_uri="",
            split=2,
            structure_family_id="structure-01",
            difficulty_class="S",
            difficulty_tags=("M",),
            evaluation_cohort="g2",
            runtime_state=None,
        )

    def source_states(self):
        return (self.state,)

    def surface_samples(self, state):
        return (SurfaceSample(),)

    def query_plan(self, state, surfaces=()):
        views = np.asarray(((0.0, 0.0, 1.0), (0.6, 0.0, 0.8)), dtype=np.float32)
        lights = np.asarray(
            ((0.0, 0.0, 1.0), (0.8, 0.0, 0.6), (0.0, 0.8, 0.6), (-0.8, 0.0, 0.6)),
            dtype=np.float32,
        )
        return QueryPlan(
            views,
            lights,
            np.full(4, 2.0 * np.pi / 4, dtype=np.float32),
            np.full(4, 1.0 / (2.0 * np.pi), dtype=np.float32),
            "test-uniform",
            7,
            np.full(2, QUERY_ROLE_NAMES.index(self.config.query_role), dtype=np.uint8),
        )

    def evaluate(self, state, surfaces, plan):
        wi = plan.light_directions
        response = np.broadcast_to(np.abs(wi[..., 2:3]), (*wi.shape[:-1], 3)).copy()
        return EvaluatedBlock.deterministic(response[None])

    def metadata(self):
        return {"name": "test-reference"}

    def close(self):
        pass


@dataclass(frozen=True)
class _DiagnosticBudget:
    relative_standard_error: float
    maximum_group_relative_standard_error: float
    max_combined_samples: int
    enforce_maximum_group_relative_standard_error: bool = True


def test_reciprocal_block_uses_and_restores_diagnostic_budget() -> None:
    config = CollectionConfig(
        query_role="test",
        view_count=2,
        light_count=4,
        reciprocal_target_relative_se_p95=0.10,
        reciprocal_maximum_query_group_relative_se_p95=0.50,
        reciprocal_maximum_combined_samples=262144,
    )
    provider = _Provider(config)
    original_budget = _DiagnosticBudget(0.04, 0.10, 524288)
    provider.provider_config = original_budget
    original_evaluate = provider.evaluate
    observed = []

    def evaluate(state, surfaces, plan):
        observed.append(provider.provider_config)
        return original_evaluate(state, surfaces, plan)

    provider.evaluate = evaluate
    plan = provider.query_plan(provider.state, (SurfaceSample(),))
    block = _reciprocal_block(
        provider,
        provider.state,
        (SurfaceSample(),),
        plan,
        config,
    )
    assert block.mean.shape == (1, 2, 4, 3)
    assert observed == [_DiagnosticBudget(0.10, 0.50, 262144, False)]
    assert provider.provider_config == original_budget


def test_reference_shard_v5_round_trip(tmp_path) -> None:
    config = CollectionConfig(
        name="uniform-v1",
        query_role="test",
        view_count=2,
        light_count=4,
        seed=7,
    )
    path = tmp_path / "reference.h5"
    manifest = collect_reference_dataset(
        path,
        (_Provider(config),),
        config,
        created_at="2026-08-24T00:00:00+00:00",
        generator_git_commit="test",
    )
    assert manifest.format_name == "reference-shard"
    assert manifest.format_version == 5
    assert manifest.sampling_name == "uniform-v1"
    with ReferenceDataset.open(path) as dataset:
        assert dataset.state_strings("structure_family_id").tolist() == ["structure-01"]
        assert dataset.state_strings("difficulty_class").tolist() == ["S"]
        assert json.loads(dataset.state_strings("difficulty_tags_json")[0]) == ["M"]
        assert dataset.state_strings("evaluation_cohort").tolist() == ["g2"]
        assert dataset.query_roles.tolist() == [int(QueryRole.TEST), int(QueryRole.TEST)]
        assert dataset.group_batch((0, 1))["mean"].shape == (2, 4, 3)
        minimal = dataset.group_batch(
            (0,),
            fields=("state_index", "wo", "wi", "mean", "solid_angle_weight"),
        )
        assert set(minimal) == {
            "state_index", "wo", "wi", "mean", "solid_angle_weight"
        }


def test_reference_identity_excludes_capture_time_and_container_bytes(tmp_path) -> None:
    config = CollectionConfig(
        name="uniform-v1",
        query_role="test",
        view_count=2,
        light_count=4,
        seed=7,
    )
    manifests = [
        collect_reference_dataset(
            tmp_path / f"reference-{index}.h5",
            (_Provider(config),),
            config,
            created_at=f"2026-08-2{index + 4}T00:00:00+00:00",
            generator_git_commit="test",
        )
        for index in range(2)
    ]
    assert manifests[0].dataset_id == manifests[1].dataset_id

    state_id = _Provider(config).state.state_id
    shards = []
    for index, dataset_manifest in enumerate(manifests):
        shard = CorpusShard(
            shard_id="test",
            uri=f"reference-{index}.h5",
            role="test",
            structure_family_id="structure-01",
            difficulty_class="all",
            difficulty_tags=(),
            state_ids=(state_id,),
            view_count=2,
            direction_count=4,
            status="complete",
            dataset_id=dataset_manifest.dataset_id,
            sha256=str(index) * 64,
            seconds=float(index + 1),
            combined_reference_samples=8,
            reciprocal_combined_reference_samples=8,
        )
        shards.append(shard)
    left = ReferenceCorpusManifest(
        name="identity-test-v1",
        plan={"name": "identity-test-v1"},
        plan_sha256="a" * 64,
        created_at="2026-08-24T00:00:00+00:00",
        shards=(shards[0],),
    ).resolved()
    right = ReferenceCorpusManifest(
        name="identity-test-v1",
        plan={"name": "identity-test-v1"},
        plan_sha256="a" * 64,
        created_at="2026-08-25T00:00:00+00:00",
        shards=(shards[1],),
    ).resolved()
    assert left.corpus_id == right.corpus_id


def test_full_sphere_t_mixture_has_five_normalized_components() -> None:
    views = np.asarray(((0.0, 0.0, 1.0),), dtype=np.float32)
    weights = (0.26, 0.26, 0.25, 0.10, 0.13)
    directions, quadrature, pdf = peak_grazing_mixture_query(
        views,
        1536,
        full_sphere=True,
        seed=19,
        component_weights=weights,
    )
    np.testing.assert_allclose(pdf * quadrature, 1.0 / 1536.0, rtol=2e-6, atol=1e-8)
    assert np.mean(directions[..., 2] < 0.0) >= 0.35

    rng = np.random.default_rng(31)
    count = 200000
    z = rng.uniform(-1.0, 1.0, count)
    phi = rng.uniform(-np.pi, np.pi, count)
    radius = np.sqrt(1.0 - z * z)
    uniform = np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=1)
    values = peak_grazing_mixture_pdf(
        uniform,
        views[0],
        full_sphere=True,
        component_weights=weights,
    )
    # 三尺度最窄 vMF 的面积很小，uniform MC 的方差本来就高；这里只抓明显失归一。
    assert abs(float(np.mean(values) * 4.0 * np.pi) - 1.0) < 0.10


def test_corpus_store_keeps_batches_rectangular_and_state_indices_global(tmp_path, monkeypatch) -> None:
    paths = []
    shards = []
    for role in ("train", "test"):
        config = CollectionConfig(
            name="uniform-v1",
            query_role=role,
            view_count=2,
            light_count=4,
            seed=7,
        )
        path = tmp_path / f"{role}.h5"
        dataset_manifest = collect_reference_dataset(
            path,
            (_Provider(config),),
            config,
            created_at="2026-08-24T00:00:00+00:00",
            generator_git_commit="test",
        )
        paths.append(path)
        shards.append(CorpusShard(
            shard_id=role,
            uri=str(path),
            role=role,
            structure_family_id="structure-01",
            difficulty_class="S" if role == "train" else "all",
            difficulty_tags=("M",) if role == "train" else (),
            state_ids=(_Provider(config).state.state_id,),
            view_count=2,
            direction_count=4,
            status="complete",
            dataset_id=dataset_manifest.dataset_id,
            sha256="b" * 64,
        ))
    corpus = ReferenceCorpusManifest(
        name="test-corpus-v1",
        plan={},
        plan_sha256="a" * 64,
        created_at="2026-08-24T00:00:00+00:00",
        shards=tuple(shards),
        corpus_id="c" * 64,
    )
    monkeypatch.setattr("ncls.data.stores.validate_reference_corpus", lambda path: corpus)
    from ncls.data.stores import ReferenceCorpusStore

    with ReferenceCorpusStore(tmp_path / "corpus.json") as store:
        train = store.partition_indices("target-visible-v1", "train")
        test = store.partition_indices("parametric-v1", "test")
        assert train.shape == test.shape == (2, 2)
        assert set(train[:, 0]) != set(test[:, 0])
        batch = store.batch(test)
        assert batch["mean"].shape == (2, 4, 3)
        assert batch["state_index"].tolist() == [0, 0]
        assert store.state_count == 1
