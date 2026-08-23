from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


GATE_FORMAT = "ncls.supervision-gate"
GATE_VERSION = 3


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_supervision_gate(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("supervision gate root must be an object")
    if value.get("format_name") != GATE_FORMAT or value.get("format_version") != GATE_VERSION:
        raise ValueError("unsupported supervision gate format")
    if "@" not in str(value.get("gate_id", "")) or value.get("status") != "frozen":
        raise ValueError("supervision gate requires a versioned gate_id and frozen status")
    if not isinstance(value.get("base_requirements"), dict) or not isinstance(value.get("family_requirements"), dict):
        raise ValueError("supervision gate requirements must be objects")
    return value


def evaluate_supervision_gate(audit: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, actual: Any, expected: Any, passed: bool) -> None:
        checks.append({"name": name, "passed": bool(passed), "actual": actual, "expected": expected})

    base = gate["base_requirements"]
    split = audit["split_audit"]
    for field in ("split_group_id", "asset_id", "source_sha256"):
        maximum = int(base["maximum_split_leaks"])
        actual = int(split[field]["leak_count"])
        check(f"split.{field}", actual, {"maximum": maximum}, actual <= maximum)
    parent_actual = int(split["parent_child_cross_split_count"])
    check("split.parent_child", parent_actual, {"maximum": 0}, parent_actual == 0)
    for split_name, count in split["query_group_counts"].items():
        check(f"source_split.nonempty.{split_name}", int(count), {"minimum": 1}, int(count) >= 1)
    for role_name in base["required_query_roles"]:
        count = int(split["query_role_counts"].get(role_name, 0))
        check(f"query_role.nonempty.{role_name}", count, {"minimum": 1}, count >= 1)

    grids = audit["coverage"]["direction_grid_independence"]
    maximum_grid_overlap = int(base["maximum_cross_partition_direction_grid_overlap"])
    for partition_axis in ("source_split", "query_role"):
        for direction_kind in ("wi_grid", "wo"):
            for pair, value in grids[partition_axis][direction_kind]["overlap_counts"].items():
                actual = int(value)
                check(
                    f"coverage.{partition_axis}.{direction_kind}.{pair}",
                    actual,
                    {"maximum": maximum_grid_overlap},
                    actual <= maximum_grid_overlap,
                )

    proposal_ids = [str(item["proposal_id"]).lower() for item in audit["coverage"]["proposals"]]
    for keyword in base["required_proposal_keywords"]:
        present = any(str(keyword).lower() in proposal_id for proposal_id in proposal_ids)
        check(f"coverage.proposal.{keyword}", proposal_ids, {"contains": keyword}, present)
    adversarial = audit["coverage"]["adversarial_profile_presence"]
    for profile in base["required_adversarial_profiles"]:
        actual = bool(adversarial[profile])
        check(f"coverage.adversarial.{profile}", actual, True, actual)
    adversarial_response = audit["response"]["by_query_role"]["adversarial_probe"]
    peak_spacing_value = adversarial_response["peak_nearest_neighbor_angle_degrees"].get("p95")
    peak_spacing_p95 = float(peak_spacing_value) if peak_spacing_value is not None else None
    maximum_peak_spacing = float(base["maximum_adversarial_peak_spacing_p95_degrees"])
    check(
        "coverage.adversarial.peak_spacing_p95_degrees",
        peak_spacing_p95,
        {"maximum": maximum_peak_spacing},
        peak_spacing_p95 is not None and peak_spacing_p95 <= maximum_peak_spacing,
    )
    adversarial_coverage = audit["coverage"]["by_query_role"]["adversarial_probe"]
    grazing_fraction = float(adversarial_coverage["wi_grazing_fraction_abs_cos_below_sin_5deg"])
    minimum_grazing = float(base["minimum_adversarial_wi_grazing_fraction"])
    check(
        "coverage.adversarial.wi_grazing_fraction",
        grazing_fraction,
        {"minimum": minimum_grazing},
        grazing_fraction >= minimum_grazing,
    )

    transform = audit["target_transform_statistics"]
    expected_source_split = str(base["target_transform_fit_source_split"])
    expected_query_role = str(base["target_transform_fit_query_role"])
    check(
        "transform.fit_source_split",
        transform["fit_source_split"],
        expected_source_split,
        transform["fit_source_split"] == expected_source_split,
    )
    check(
        "transform.fit_query_role",
        transform["fit_query_role"],
        expected_query_role,
        transform["fit_query_role"] == expected_query_role,
    )
    check(
        "transform.statistics_sha256",
        transform["statistics_sha256"],
        "64 位小写 SHA-256",
        len(str(transform["statistics_sha256"])) == 64,
    )

    uncertainty = audit["reference_uncertainty"]
    if float(uncertainty["deterministic_query_fraction"]) < 1.0:
        relative_p95 = float(uncertainty["relative_standard_error"].get("p95", 0.0))
        replica_p95 = float(uncertainty["replica_normalized_l1"].get("p95", 0.0))
        max_relative = float(base["maximum_stochastic_relative_standard_error_p95"])
        max_replica = float(base["maximum_stochastic_replica_normalized_l1_p95"])
        check("noise.relative_standard_error_p95", relative_p95, {"maximum": max_relative}, relative_p95 <= max_relative)
        check("noise.replica_normalized_l1_p95", replica_p95, {"maximum": max_replica}, replica_p95 <= max_replica)

    dataset_families = set(audit["state_distribution"])
    for family_id, requirements in gate["family_requirements"].items():
        if family_id not in dataset_families:
            continue
        for profile in requirements.get("required_adversarial_profiles", []):
            actual = bool(adversarial[profile])
            check(f"family.{family_id}.adversarial.{profile}", actual, True, actual)
        minimum_scales = requirements.get("minimum_unique_footprint_scales")
        if minimum_scales is not None:
            actual = int(audit["coverage"]["unique_footprint_scale_count"])
            check(
                f"family.{family_id}.footprint_scales",
                actual,
                {"minimum": int(minimum_scales)},
                actual >= int(minimum_scales),
            )
        minimum_rotations = requirements.get("minimum_unique_footprint_rotations")
        if minimum_rotations is not None:
            actual = int(audit["coverage"]["unique_footprint_rotation_count"])
            check(
                f"family.{family_id}.footprint_rotations",
                actual,
                {"minimum": int(minimum_rotations)},
                actual >= int(minimum_rotations),
            )
        minimum_transmission = requirements.get("minimum_adversarial_wi_transmission_fraction")
        if minimum_transmission is not None:
            actual = float(adversarial_coverage["wi_transmission_fraction"])
            check(
                f"family.{family_id}.adversarial.wi_transmission_fraction",
                actual,
                {"minimum": float(minimum_transmission)},
                actual >= float(minimum_transmission),
            )

    return {
        "format_name": "ncls.supervision-gate-result",
        "format_version": 3,
        "gate_id": gate["gate_id"],
        "gate_sha256": _sha256(gate),
        "audit_sha256": audit["audit_sha256"],
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
