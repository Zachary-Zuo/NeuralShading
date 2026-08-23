from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ncls.data import CollectionConfig, collect_reference_dataset
from ncls.data.providers import LayerStackProvider, LayerStackProviderConfig
from ncls.learning.audit import audit_supervision
from ncls.learning.gates import evaluate_supervision_gate, load_supervision_gate


class _ConstantEvaluator:
    def __init__(self, light_count: int) -> None:
        self.light_count = light_count

    def evaluate_query_groups(
        self,
        materials,
        view_directions,
        *,
        sample_count_per_replica: int,
        query_group_seeds: np.ndarray,
        light_directions: np.ndarray | None = None,
        sample_offset: int = 0,
    ):
        shape = (len(materials), self.light_count, 3)
        mean_a = np.full(shape, 0.2, dtype=np.float32)
        mean_b = np.full(shape, 0.22, dtype=np.float32)
        variance = np.full(shape, 0.01, dtype=np.float32)
        return mean_a, variance + mean_a * mean_a, mean_b, variance + mean_b * mean_b


def _dataset(path: Path) -> None:
    collection = CollectionConfig(
        view_count=2,
        validation_view_count=1,
        test_view_count=1,
        adversarial_view_count=1,
        light_count=8,
        seed=53,
    )
    provider = LayerStackProvider(
        collection,
        LayerStackProviderConfig(
            family_count=3,
            local_state_count=1,
            samples_per_replica=4,
            query_group_batch=3,
            max_depth=4,
        ),
        evaluator=_ConstantEvaluator(8),
    )
    collect_reference_dataset(
        path,
        [provider],
        collection,
        created_at="2026-08-24T00:00:00+00:00",
        generator_git_commit="test",
    )


def test_supervision_audit_is_family_neutral_and_transform_stats_are_train_only(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.h5"
    output = tmp_path / "audit"
    _dataset(dataset_path)

    result = audit_supervision(dataset_path, output, max_distribution_query_groups=4)

    assert result["format_name"] == "ncls.supervision-audit"
    assert result["format_version"] == 5
    assert result["dataset"]["content_hash_verified"] is True
    assert result["split_audit"]["split_group_id"]["leak_count"] == 0
    assert result["split_audit"]["source_sha256"]["leak_count"] == 0
    assert result["sampling"]["query_group_count_used"] == 4
    assert "uniform-solid-angle" in result["coverage"]["proposals"][0]["proposal_id"]
    assert all(
        count == 0
        for count in result["coverage"]["direction_grid_independence"]["query_role"]["wi_grid"]["overlap_counts"].values()
    )
    assert "ncls.layer-stack@1" in result["response"]["by_family"]
    assert len(result["reference_uncertainty"]["by_state"]) == 3
    assert sum(
        item["query_group_count"]
        for item in result["reference_uncertainty"]["by_source_split"].values()
    ) == 4
    assert result["reference_uncertainty"]["by_integrated_energy"]["high_ge_p90"]["query_group_count"] >= 1
    assert result["reference_uncertainty"]["highest_relative_standard_error_groups"][0]["asset_id"]
    assert result["reference_uncertainty"]["worst_query_group"]["relative_standard_error_p95"] >= result["reference_uncertainty"]["relative_standard_error"]["p95"]
    assert result["dataset"]["provider_metadata"][0]["provider_config"]["state_profile_id"]
    assert result["response"]["by_query_role"]["adversarial_probe"]["query_group_count"] >= 1
    assert result["coverage"]["by_query_role"]["adversarial_probe"]["query_group_count"] >= 1

    statistics = json.loads((output / "target_transform_statistics.json").read_text(encoding="utf-8"))
    assert statistics["fit_source_split"] == "train"
    assert statistics["fit_query_role"] == "train"
    assert statistics["query_group_count_available"] == 2
    assert statistics["families"]["ncls.layer-stack@1"]["channels"][0]["nonnegative_log1p_scale"] > 0.0
    assert (output / "audit.json").is_file()
    assert (output / "report.md").read_text(encoding="utf-8").startswith("# Supervision audit")


def test_frozen_supervision_gate_reports_all_failures_without_mutating_audit(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.h5"
    output = tmp_path / "audit"
    _dataset(dataset_path)
    result = audit_supervision(dataset_path, output)
    gate = load_supervision_gate(Path("configs/research/e0-supervision-gates-v5.json"))

    gate_result = evaluate_supervision_gate(result, gate)

    assert gate_result["passed"] is False
    failures = {item["name"] for item in gate_result["checks"] if not item["passed"]}
    assert "dataset.query_profile_id" in failures
    assert "coverage.proposal.peak" in failures
    check_names = {item["name"] for item in gate_result["checks"]}
    assert "noise.worst_query_group_relative_standard_error_p95" in check_names
    assert "noise.worst_query_group_replica_normalized_l1" in check_names


def test_supervision_audit_refuses_to_overwrite_an_existing_run(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.h5"
    output = tmp_path / "audit"
    _dataset(dataset_path)
    output.mkdir()
    (output / "user-file.txt").write_text("keep", encoding="utf-8")

    with np.testing.assert_raises_regex(ValueError, "new or empty"):
        audit_supervision(dataset_path, output)
    assert (output / "user-file.txt").read_text(encoding="utf-8") == "keep"
