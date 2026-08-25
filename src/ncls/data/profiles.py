from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .collector import CollectionConfig


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class QueryDensity:
    views: int
    directions: int
    proposal: str
    components: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.views < 1 or self.directions < 1:
            raise ValueError("query density counts must be positive")
        if self.proposal not in {"uniform", "peak-aware", "adversarial"}:
            raise ValueError("unsupported query proposal")
        if self.proposal == "uniform":
            if self.components:
                raise ValueError("uniform query density must not declare mixture components")
        elif not self.components or any(value < 0.0 for value in self.components.values()):
            raise ValueError("mixture components must be nonnegative")
        elif abs(sum(self.components.values()) - 1.0) > 1e-8:
            raise ValueError("mixture components must sum to one")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QueryDensity":
        return cls(
            views=int(value["views"]),
            directions=int(value["directions"]),
            proposal=str(value["proposal"]),
            components={str(name): float(weight) for name, weight in value.get("components", {}).items()},
        )


@dataclass(frozen=True)
class CorpusPlan:
    name: str
    seed: int
    provider: Mapping[str, Any]
    sampling_name: str
    train_density: Mapping[str, QueryDensity]
    role_density: Mapping[str, QueryDensity]
    split: Mapping[str, Any]
    reference_budget: Mapping[str, Any]
    shard_policy: Mapping[str, Any]
    document: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.name != "layer-stack-v1" or self.seed != 20260824:
            raise ValueError("the frozen v1 corpus must be layer-stack-v1 with seed 20260824")
        if self.provider != {
            "name": "layer-stack",
            "state_profile": "layer-stack-v1",
            "family_count": 28,
            "states_per_family": 10,
            "maximum_path_depth": 64,
        }:
            raise ValueError("layer-stack-v1 provider configuration is frozen")
        if set(self.train_density) != {"W", "G", "S"}:
            raise ValueError("CorpusPlan train density must define W, G and S")
        if set(self.role_density) != {"validation", "test", "adversarial_probe", "dense_slice"}:
            raise ValueError("CorpusPlan must define all non-train query roles")
        schema = self.document.get("schema")
        if schema != {"name": "corpus-plan", "version": 1}:
            raise ValueError("unsupported CorpusPlan schema")
        if self.sampling_name != "peak-aware-v1":
            raise ValueError("CorpusPlan v1 requires peak-aware-v1 sampling")
        expected_train = {
            "W": (48, 512, {"uniform": 0.60, "reflection_peak": 0.25, "grazing": 0.15}),
            "G": (64, 1024, {"uniform": 0.40, "reflection_peak": 0.40, "grazing": 0.20}),
            "S": (96, 2048, {"uniform": 0.30, "reflection_peak": 0.50, "grazing": 0.20}),
        }
        for name, (views, directions, components) in expected_train.items():
            density = self.train_density[name]
            if (density.views, density.directions, density.proposal) != (
                views, directions, "peak-aware"
            ) or dict(density.components) != components:
                raise ValueError(f"CorpusPlan v1 {name} density does not match the frozen table")
        validation = self.role_density["validation"]
        test = self.role_density["test"]
        adversarial = self.role_density["adversarial_probe"]
        dense = self.role_density["dense_slice"]
        if (validation.views, validation.directions, validation.proposal) != (16, 256, "uniform"):
            raise ValueError("CorpusPlan v1 validation density is frozen at 16x256 uniform")
        if (test.views, test.directions, test.proposal) != (24, 512, "uniform"):
            raise ValueError("CorpusPlan v1 test density is frozen at 24x512 uniform")
        if (
            adversarial.views,
            adversarial.directions,
            adversarial.proposal,
            dict(adversarial.components),
        ) != (16, 128, "adversarial", {
            "uniform": 0.20,
            "reflection_peak": 0.55,
            "grazing": 0.25,
        }):
            raise ValueError("CorpusPlan v1 adversarial density is frozen")
        if (dense.views, dense.directions, dense.proposal) != (4, 8192, "uniform"):
            raise ValueError("dense_slice base density is frozen at 4x8192")
        if self.split != {
            "name": "parametric-v1",
            "heldout_family_count": 4,
            "validation_states_per_fitted_family": 1,
            "test_states_per_fitted_family": 1,
            "minimum_test_state_count": 50,
        }:
            raise ValueError("LayerStack v1 split configuration is frozen")
        base_reference_budget = dict(self.reference_budget)
        sample_promotions = base_reference_budget.pop("state_sample_promotions", None)
        if base_reference_budget != {
            "name": "adaptive-v1",
            "double_replica": True,
            "target_relative_se_p95": 0.04,
            "maximum_query_group_relative_se_p95": 0.10,
            "minimum_combined_samples": 1024,
            "maximum_combined_samples": 262144,
            "reciprocal_target_relative_se_p95": 0.20,
            "reciprocal_maximum_query_group_relative_se_p95": 0.999,
            "reciprocal_maximum_combined_samples": 65536,
            "diagnostic_query_roles": ["adversarial_probe", "dense_slice"],
            "diagnostic_target_relative_se_p95": 0.08,
            "diagnostic_maximum_query_group_relative_se_p95": 0.50,
            "diagnostic_maximum_combined_samples": 262144,
            "training_target_relative_se_p95": 0.06,
            "training_maximum_query_group_relative_se_p95": 0.25,
            "training_maximum_combined_samples": 262144,
            "training_reciprocal_target_relative_se_p95": 0.50,
            "training_reciprocal_maximum_query_group_relative_se_p95": 0.999,
            "training_reciprocal_maximum_combined_samples": 4096,
            "diagnostic_reciprocal_target_relative_se_p95": 0.20,
            "diagnostic_reciprocal_maximum_query_group_relative_se_p95": 0.999,
            "diagnostic_reciprocal_maximum_combined_samples": 65536,
            "batch_samples_per_replica": 256,
            "maximum_dispatch_queries": 4096,
        }:
            raise ValueError("CorpusPlan v1 reference budget is frozen")
        if not isinstance(sample_promotions, list):
            raise ValueError("state_sample_promotions must be an array")
        promotion_role_keys = []
        for promotion in sample_promotions:
            if not isinstance(promotion, Mapping) or set(promotion) != {
                "state_id", "maximum_combined_samples",
                "maximum_query_group_relative_se_p95", "query_roles",
            }:
                raise ValueError("state_sample_promotions entries are invalid")
            state_id = promotion["state_id"]
            maximum = promotion["maximum_combined_samples"]
            maximum_group = promotion["maximum_query_group_relative_se_p95"]
            query_roles = promotion["query_roles"]
            if (
                not isinstance(state_id, str)
                or re.fullmatch(r"[0-9a-f]{64}", state_id) is None
                or not isinstance(maximum, int)
                or not 262144 <= maximum <= 4194304
                or maximum % 512
                or isinstance(maximum_group, bool)
                or not isinstance(maximum_group, (int, float))
                or not 0.10 <= float(maximum_group) < 1.0
                or not isinstance(query_roles, list)
                or not query_roles
                or len(query_roles) != len(set(query_roles))
                or any(role not in {"train", "validation", "test"} for role in query_roles)
            ):
                raise ValueError("state_sample_promotions values are invalid")
            promotion_role_keys.extend((state_id, role) for role in query_roles)
        if len(promotion_role_keys) != len(set(promotion_role_keys)):
            raise ValueError(
                "state_sample_promotions may not repeat a state ID for the same query role"
            )
        if self.shard_policy != {
            "name": "family-role-v1",
            "resume": "verified-file",
            "manifest": "reference-corpus",
        }:
            raise ValueError("CorpusPlan v1 shard policy is frozen")
        if self.document["sampling"].get("wo") != {
            "distribution": "stratified-cosine",
            "grazing_fraction": 0.20,
            "grazing_band_degrees": [75.0, 89.0],
        }:
            raise ValueError("CorpusPlan v1 view sampling is frozen")
        transmission = self.document["sampling"].get("transmission")
        if transmission != {
            "additional_views": 32,
            "direction_multiplier": 1.5,
            "transmission_peak_weight": 0.25,
            "critical_band_weight": 0.10,
            "critical_view_band_degrees": [35.0, 55.0],
            "critical_wi_abs_cosine_band": [0.65, 0.85],
        }:
            raise ValueError("CorpusPlan v1 transmission sampling does not match the frozen table")
        moving_peak = self.document["sampling"].get("moving_peak")
        if moving_peak != {
            "name": "reference-peak-v1",
            "probe_directions": 4096,
            "samples_per_replica": 64,
        }:
            raise ValueError("CorpusPlan v1 moving-peak calibration is unsupported")
        promotions = self.document["sampling"].get("dense_promotions")
        if (
            not isinstance(promotions, list)
            or len(promotions) != len(set(promotions))
            or any(
                not isinstance(state_id, str)
                or re.fullmatch(r"[0-9a-f]{64}", state_id) is None
                for state_id in promotions
            )
        ):
            raise ValueError("dense_promotions must contain unique lowercase SHA-256 state IDs")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.document).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorpusPlan":
        expected_fields = {
            "schema", "name", "seed", "provider", "sampling", "split",
            "reference_budget", "shards",
        }
        if set(value) != expected_fields:
            raise ValueError("CorpusPlan root fields do not match v1")
        sampling = value["sampling"]
        if not isinstance(sampling, Mapping):
            raise ValueError("CorpusPlan sampling must be an object")
        if set(sampling) != {
            "name", "wo", "transmission", "moving_peak", "dense_promotions",
            "train", "roles",
        }:
            raise ValueError("CorpusPlan sampling fields do not match v1")
        return cls(
            name=str(value["name"]),
            seed=int(value["seed"]),
            provider=dict(value["provider"]),
            sampling_name=str(sampling["name"]),
            train_density={
                str(name): QueryDensity.from_dict(density)
                for name, density in sampling["train"].items()
            },
            role_density={
                str(name): QueryDensity.from_dict(density)
                for name, density in sampling["roles"].items()
            },
            split=dict(value["split"]),
            reference_budget=dict(value["reference_budget"]),
            shard_policy=dict(value["shards"]),
            document=dict(value),
        )

    @classmethod
    def load(cls, path: Path | str) -> "CorpusPlan":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("CorpusPlan root must be an object")
        return cls.from_dict(document)

    def resolve_query(
        self,
        difficulty_class: str,
        difficulty_tags: tuple[str, ...],
        role: str,
        state_id: str | None = None,
    ) -> QueryDensity:
        effective_class = "S" if role == "train" and "M" in difficulty_tags else difficulty_class
        density = self.train_density[effective_class] if role == "train" else self.role_density[role]
        if (
            role == "dense_slice"
            and state_id is not None
            and state_id in self.document["sampling"]["dense_promotions"]
        ):
            density = QueryDensity(
                views=density.views,
                directions=16384,
                proposal=density.proposal,
                components=density.components,
            )
        if role != "train" or "T" not in difficulty_tags:
            return density
        transmission = self.document["sampling"]["transmission"]
        reserved = float(transmission["transmission_peak_weight"]) + float(
            transmission["critical_band_weight"]
        )
        components = {
            name: (1.0 - reserved) * weight for name, weight in density.components.items()
        }
        components["transmission_peak"] = float(transmission["transmission_peak_weight"])
        components["critical_band"] = float(transmission["critical_band_weight"])
        return QueryDensity(
            views=density.views + int(transmission["additional_views"]),
            directions=int(round(float(transmission["direction_multiplier"]) * density.directions)),
            proposal=density.proposal,
            components=components,
        )

    def collection_config(
        self,
        difficulty_class: str,
        difficulty_tags: tuple[str, ...],
        role: str,
        state_id: str | None = None,
    ) -> CollectionConfig:
        resolved = self.resolve_query(difficulty_class, difficulty_tags, role, state_id)
        transmission = self.document["sampling"]["transmission"]
        ordered = (
            ("uniform", "reflection_peak", "grazing")
            if "T" not in difficulty_tags
            else (
                "uniform",
                "reflection_peak",
                "transmission_peak",
                "critical_band",
                "grazing",
            )
        )
        missing = [name for name in resolved.components if name not in ordered]
        if missing:
            raise ValueError(f"unsupported mixture components: {missing}")
        if role == "train":
            reciprocal_budget_prefix = "training_reciprocal"
        elif role in set(self.reference_budget["diagnostic_query_roles"]):
            reciprocal_budget_prefix = "diagnostic_reciprocal"
        else:
            reciprocal_budget_prefix = "reciprocal"
        return CollectionConfig(
            name=self.sampling_name,
            query_role=role,
            view_count=resolved.views,
            light_count=resolved.directions,
            proposal=resolved.proposal,
            mixture_weights=tuple(resolved.components[name] for name in ordered)
            if resolved.components
            else (0.5, 0.35, 0.15),
            grazing_view_fraction=float(self.document["sampling"]["wo"]["grazing_fraction"]),
            grazing_min_degrees=float(self.document["sampling"]["wo"]["grazing_band_degrees"][0]),
            grazing_max_degrees=float(self.document["sampling"]["wo"]["grazing_band_degrees"][1]),
            transmission_view_count=(
                int(transmission["additional_views"])
                if role == "train" and "T" in difficulty_tags else 0
            ),
            critical_view_min_degrees=float(transmission["critical_view_band_degrees"][0]),
            critical_view_max_degrees=float(transmission["critical_view_band_degrees"][1]),
            critical_wi_abs_cosine_min=float(transmission["critical_wi_abs_cosine_band"][0]),
            critical_wi_abs_cosine_max=float(transmission["critical_wi_abs_cosine_band"][1]),
            reciprocal_target_relative_se_p95=float(
                self.reference_budget[f"{reciprocal_budget_prefix}_target_relative_se_p95"]
            ),
            reciprocal_maximum_query_group_relative_se_p95=float(
                self.reference_budget[
                    f"{reciprocal_budget_prefix}_maximum_query_group_relative_se_p95"
                ]
            ),
            reciprocal_maximum_combined_samples=int(
                self.reference_budget[f"{reciprocal_budget_prefix}_maximum_combined_samples"]
            ),
            seed=self.seed,
        )
