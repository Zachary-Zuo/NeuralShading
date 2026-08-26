from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np

from ncls.data import (
    CorpusPlan,
    CorpusSelection,
    ReferenceCorpusManifest,
    plan_layer_stack_corpus,
    select_layer_stack_state,
)
from ncls.data.priors import (
    LAYER_STACK_PROFILE,
    layer_stack_difficulty,
    layer_stack_v1_families,
    layer_stack_v1_splits,
)
from ncls.data.corpus import _layer_stack_provider_config


def test_layer_stack_v1_states_and_split_are_deterministic() -> None:
    left = layer_stack_v1_families(28, 10, 20260824)
    right = layer_stack_v1_families(28, 10, 20260824)
    assert LAYER_STACK_PROFILE == "layer-stack-v1"
    assert len(left) == 28
    assert all(len(states) == 10 for _, states in left)
    assert left == right
    assert all(layer_stack_difficulty(stack)[0] in {"W", "G", "S"} for _, family in left for stack in family)

    split = layer_stack_v1_splits(28, 10, 4, 20260824)
    counts = Counter(value[0] for value in split.values())
    cohorts = Counter(value[1] for value in split.values())
    assert counts == {0: 192, 1: 24, 2: 64}
    assert cohorts == {"train": 192, "validation": 24, "g2": 24, "g2s": 40}
    for family_index in range(28):
        values = [split[(family_index, state_index)] for state_index in range(10)]
        if all(cohort == "g2s" for _, cohort in values):
            assert {source_split for source_split, _ in values} == {2}
        else:
            assert Counter(source_split for source_split, _ in values) == {0: 8, 1: 1, 2: 1}


def test_density_table_resolves_m_and_t_without_hidden_ids() -> None:
    plan = CorpusPlan.load("configs/corpus/layer-stack-v1.json")
    assert plan.name == "layer-stack-v1"
    assert plan.sampling_name == "peak-aware-v1"
    assert plan.resolve_query("W", (), "train").views == 48
    assert plan.resolve_query("G", (), "train").directions == 1024
    moving = plan.resolve_query("W", ("M",), "train")
    assert (moving.views, moving.directions) == (96, 2048)
    transmission = plan.resolve_query("G", ("T",), "train")
    assert (transmission.views, transmission.directions) == (96, 1536)
    assert np.isclose(transmission.components["transmission_peak"], 0.25)
    assert np.isclose(transmission.components["critical_band"], 0.10)
    assert np.isclose(sum(transmission.components.values()), 1.0)
    config = plan.collection_config("G", ("T",), "train")
    assert config.transmission_view_count == 32
    assert config.mixture_weights == (0.26, 0.26, 0.25, 0.10, 0.13)
    assert config.reciprocal_target_relative_se_p95 == 0.50
    assert config.reciprocal_maximum_query_group_relative_se_p95 == 0.999
    assert config.reciprocal_maximum_combined_samples == 4096
    test_config = plan.collection_config("G", (), "test")
    assert test_config.reciprocal_target_relative_se_p95 == 0.20
    assert test_config.reciprocal_maximum_query_group_relative_se_p95 == 0.999
    assert test_config.reciprocal_maximum_combined_samples == 65536
    assert plan.resolve_query("S", (), "dense_slice").directions == 8192
    promoted_state_id = select_layer_stack_state(
        plan, "layers-01-diffuse-variant-00", 3
    ).state_id
    promoted_document = deepcopy(plan.document)
    promoted_document["reference_budget"]["state_sample_promotions"] = [{
        "state_id": promoted_state_id,
        "maximum_combined_samples": 524288,
        "maximum_query_group_relative_se_p95": 0.20,
        "query_roles": ["test"],
    }]
    promoted_plan = CorpusPlan.from_dict(promoted_document)
    base_provider = _layer_stack_provider_config(plan, adaptive=True)
    promoted_provider = _layer_stack_provider_config(
        promoted_plan,
        adaptive=True,
        selected_state_ids=(promoted_state_id,),
        query_role="test",
    )
    dense_provider = _layer_stack_provider_config(
        promoted_plan,
        adaptive=True,
        selected_state_ids=(promoted_state_id,),
        query_role="dense_slice",
    )
    assert base_provider.max_combined_samples == 262144
    assert promoted_provider.max_combined_samples == 524288
    assert promoted_provider.maximum_group_relative_standard_error == 0.20
    assert dense_provider.max_combined_samples == 262144
    assert dense_provider.relative_standard_error == 0.08
    assert dense_provider.maximum_group_relative_standard_error == 0.50
    assert dense_provider.enforce_maximum_group_relative_standard_error is False
    train_provider = _layer_stack_provider_config(
        promoted_plan,
        adaptive=True,
        selected_state_ids=(promoted_state_id,),
        query_role="train",
    )
    assert train_provider.max_combined_samples == 262144
    assert train_provider.relative_standard_error == 0.06
    assert train_provider.maximum_group_relative_standard_error == 0.25
    assert train_provider.enforce_maximum_group_relative_standard_error is True
    extreme_state_id = "2dd43e2c1ec462cb8265c467a2b28301a97622cba78a65f014f83f72b676d324"
    extreme_train_provider = _layer_stack_provider_config(
        plan,
        adaptive=True,
        selected_state_ids=(extreme_state_id,),
        query_role="train",
    )
    assert extreme_train_provider.max_combined_samples == 262144
    assert extreme_train_provider.maximum_group_relative_standard_error == 0.75
    extreme_test_provider = _layer_stack_provider_config(
        plan,
        adaptive=True,
        selected_state_ids=(extreme_state_id,),
        query_role="test",
    )
    assert extreme_test_provider.max_combined_samples == 1048576
    assert extreme_test_provider.maximum_group_relative_standard_error == 0.60
    assert extreme_test_provider.enforce_maximum_group_relative_standard_error is True
    dense_collection = plan.collection_config("S", (), "dense_slice")
    assert dense_collection.reciprocal_target_relative_se_p95 == 0.20
    assert dense_collection.reciprocal_maximum_query_group_relative_se_p95 == 0.999
    assert dense_collection.reciprocal_maximum_combined_samples == 65536
    assert plan.reference_budget["state_sample_promotions"] == [{
        "state_id": "235974d303955b8801f627df5fd85656c8272e44983f164cb1eaee85d94ad009",
        "maximum_combined_samples": 1048576,
        "maximum_query_group_relative_se_p95": 0.10,
        "query_roles": ["validation", "test"],
    }, {
        "state_id": "2dd43e2c1ec462cb8265c467a2b28301a97622cba78a65f014f83f72b676d324",
        "maximum_combined_samples": 1048576,
        "maximum_query_group_relative_se_p95": 0.60,
        "query_roles": ["validation", "test"],
    }, {
        "state_id": "2dd43e2c1ec462cb8265c467a2b28301a97622cba78a65f014f83f72b676d324",
        "maximum_combined_samples": 262144,
        "maximum_query_group_relative_se_p95": 0.75,
        "query_roles": ["train"],
    }, {
        "state_id": "f103a1304e64c8e9dcbd7dbc424ab56091c360c9e43ebe9fb1d3318b7e1db47f",
        "maximum_combined_samples": 262144,
        "maximum_query_group_relative_se_p95": 0.65,
        "query_roles": ["validation", "test"],
    }, {
        "state_id": "f103a1304e64c8e9dcbd7dbc424ab56091c360c9e43ebe9fb1d3318b7e1db47f",
        "maximum_combined_samples": 262144,
        "maximum_query_group_relative_se_p95": 0.75,
        "query_roles": ["train"],
    }]


def test_single_state_selector_uses_readable_family_and_local_index() -> None:
    plan = CorpusPlan.load("configs/corpus/layer-stack-v1.json")
    state = select_layer_stack_state(
        plan,
        "layers-01-diffuse-variant-00",
        3,
    )
    assert state.asset_id == "layers-01-diffuse-variant-00/state-0003"
    assert state.structure_family_id == "layers-01-diffuse-variant-00"
    assert len(state.state_id) == 64
    with np.testing.assert_raises_regex(ValueError, "unknown LayerStack structure family"):
        select_layer_stack_state(plan, "missing-family", 0)


def test_layer_stack_corpus_plan_is_rectangular_and_has_enough_test_states(tmp_path) -> None:
    plan = CorpusPlan.load("configs/corpus/layer-stack-v1.json")
    manifest = plan_layer_stack_corpus(plan, tmp_path / "shards")
    assert manifest.plan_sha256 == plan.sha256
    assert manifest.plan == plan.document
    assert manifest.corpus_id is not None
    assert len(manifest.shards) == 148
    assert all(shard.view_count > 0 and shard.direction_count > 0 for shard in manifest.shards)
    roles = Counter(shard.role for shard in manifest.shards)
    assert roles["validation"] == roles["test"] == roles["adversarial_probe"] == roles["dense_slice"] == 28
    test_states = {
        state_id
        for shard in manifest.shards
        if shard.role == "test"
        for state_id in shard.state_ids
    }
    assert len(test_states) == 280
    assert all(len(set(shard.state_ids)) == len(shard.state_ids) for shard in manifest.shards)


def test_p1_selection_freezes_six_strata_without_source_test_leakage(tmp_path) -> None:
    plan = CorpusPlan.load("configs/corpus/layer-stack-v1.json")
    selection = CorpusSelection.load(
        "configs/corpus/layer-stack-p1-v1.selection.json"
    )
    manifest = plan_layer_stack_corpus(plan, tmp_path / "shards", selection)

    assert manifest.name == "layer-stack-p1-v1"
    assert manifest.format_version == 2
    assert manifest.selection_sha256 == selection.sha256
    state_ids = {
        state_id for shard in manifest.shards for state_id in shard.state_ids
    }
    assert state_ids == set(selection.state_ids)
    assert len(state_ids) == 30
    assert Counter(shard.role for shard in manifest.shards) == {
        "train": 17,
        "validation": 13,
        "test": 13,
        "adversarial_probe": 13,
        "dense_slice": 13,
    }
    assert all(
        {shard.role for shard in manifest.shards if state_id in shard.state_ids}
        == {"train", "validation", "test", "adversarial_probe", "dense_slice"}
        for state_id in state_ids
    )

    path = tmp_path / "p1-plan.json"
    manifest.write(path)
    restored = ReferenceCorpusManifest.load(path)
    assert restored.corpus_id == manifest.corpus_id
    assert restored.selection == selection.document


def test_dense_slice_audit_promotion_is_per_state_and_rectangular(tmp_path) -> None:
    base = CorpusPlan.load("configs/corpus/layer-stack-v1.json")
    base_manifest = plan_layer_stack_corpus(base, tmp_path / "base")
    dense = next(shard for shard in base_manifest.shards if shard.role == "dense_slice")
    promoted_state_id = dense.state_ids[0]

    document = deepcopy(base.document)
    document["sampling"]["dense_promotions"] = [promoted_state_id]
    promoted = CorpusPlan.from_dict(document)
    manifest = plan_layer_stack_corpus(promoted, tmp_path / "promoted")

    promoted_shards = [
        shard
        for shard in manifest.shards
        if shard.role == "dense_slice" and shard.difficulty_class == "promoted"
    ]
    assert len(manifest.shards) == 149
    assert len(promoted_shards) == 1
    assert promoted_shards[0].state_ids == (promoted_state_id,)
    assert promoted_shards[0].direction_count == 16384
    assert all(
        shard.direction_count == 8192
        for shard in manifest.shards
        if shard.role == "dense_slice" and shard.difficulty_class == "base"
    )


def test_dense_slice_promotion_rejects_unknown_state(tmp_path) -> None:
    base = CorpusPlan.load("configs/corpus/layer-stack-v1.json")
    document = deepcopy(base.document)
    document["sampling"]["dense_promotions"] = ["0" * 64]
    promoted = CorpusPlan.from_dict(document)
    with np.testing.assert_raises_regex(ValueError, "outside this CorpusPlan"):
        plan_layer_stack_corpus(promoted, tmp_path / "unknown")


def test_reference_sample_promotion_rejects_unknown_state(tmp_path) -> None:
    base = CorpusPlan.load("configs/corpus/layer-stack-v1.json")
    document = deepcopy(base.document)
    document["reference_budget"]["state_sample_promotions"] = [{
        "state_id": "0" * 64,
        "maximum_combined_samples": 524288,
        "maximum_query_group_relative_se_p95": 0.20,
        "query_roles": ["test"],
    }]
    promoted = CorpusPlan.from_dict(document)
    with np.testing.assert_raises_regex(ValueError, "outside this CorpusPlan"):
        plan_layer_stack_corpus(promoted, tmp_path / "unknown")
