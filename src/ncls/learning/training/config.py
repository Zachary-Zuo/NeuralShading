from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping

from ncls.core.identity import sha256_json


@dataclass(frozen=True)
class TrainingRoute:
    name: str
    kind: Literal["reference-evaluator", "method-sampler"]
    batch_size: int
    direction_count: int
    seed_offset: int
    options: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.kind not in {"reference-evaluator", "method-sampler"}
            or self.batch_size < 1
            or self.direction_count < 1
        ):
            raise ValueError("training route identity and sizes must be positive")
        if self.seed_offset < 0:
            raise ValueError("training route seed_offset must be nonnegative")
        if "target_estimator" in self.options or "query_role" in self.options:
            raise ValueError("training route contains a removed legacy field")
        object.__setattr__(self, "options", dict(self.options))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "batch_size": self.batch_size,
            "direction_count": self.direction_count,
            "seed_offset": self.seed_offset,
            "options": dict(self.options),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingRoute":
        required = {
            "name", "kind", "batch_size", "direction_count", "seed_offset", "options"
        }
        if set(value) != required or not isinstance(value["options"], Mapping):
            raise ValueError(f"training route fields must be exactly {sorted(required)}")
        return cls(
            str(value["name"]), str(value["kind"]),  # type: ignore[arg-type]
            int(value["batch_size"]), int(value["direction_count"]),
            int(value["seed_offset"]), value["options"],
        )


@dataclass(frozen=True)
class TrainingConfig:
    method_key: str
    run_class: str
    correspondence_id: str
    recipe_id: str
    source_adaptation_id: str
    source: Mapping[str, Any]
    online_query: Mapping[str, Any]
    model_context: Mapping[str, Any]
    lifecycle: Mapping[str, Any]
    routes: tuple[TrainingRoute, ...]
    seed: int
    device: str
    optimizer: Mapping[str, Any]
    schedule: Mapping[str, Any]
    mollification: Mapping[str, Any]
    filtering: Mapping[str, Any]
    loss: Mapping[str, Any]
    validation: Mapping[str, Any]
    checkpoint_selection: str
    format_name: str = "ncls.training-config"
    format_version: int = 3

    def __post_init__(self) -> None:
        if self.format_name != "ncls.training-config" or self.format_version != 3:
            raise ValueError("unsupported training config format")
        if not self.method_key or self.run_class not in {"smoke", "profile", "adapted", "formal"}:
            raise ValueError("training method identity or run_class is invalid")
        if not self.correspondence_id or not self.recipe_id or not self.source_adaptation_id or self.seed < 0 or not self.device:
            raise ValueError("training correspondence/source identity, seed and device are required")

        source = dict(self.source)
        if set(source) != {"family_id", "materials"}:
            raise ValueError("source fields must be family_id and materials")
        materials = source["materials"]
        if (
            not isinstance(source["family_id"], str)
            or not source["family_id"]
            or not isinstance(materials, (list, tuple))
            or not materials
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"locator"}
                or not isinstance(item["locator"], Mapping)
                or not item["locator"]
                for item in materials
            )
        ):
            raise ValueError("source family_id and material locators are invalid")
        online_query = dict(self.online_query)
        if not online_query:
            raise ValueError("online_query recipe is required")
        if any(name in online_query for name in ("offline", "recorded", "hdf5")):
            raise ValueError("online_query cannot declare an offline or recorded data path")

        lifecycle = dict(self.lifecycle)
        if set(lifecycle) != {"total_steps", "materialization_step"}:
            raise ValueError("lifecycle fields must be total_steps and materialization_step")
        total_steps = int(lifecycle["total_steps"])
        materialization_step = int(lifecycle["materialization_step"])
        if total_steps < 2 or not 1 <= materialization_step < total_steps:
            raise ValueError("training lifecycle bounds are invalid")

        routes = tuple(self.routes)
        if not routes or len({route.name for route in routes}) != len(routes):
            raise ValueError("training routes must be nonempty with unique names")
        if len({route.seed_offset for route in routes}) != len(routes):
            raise ValueError("training routes require independent seed offsets")
        if {route.kind for route in routes} != {"reference-evaluator", "method-sampler"}:
            raise ValueError("training requires evaluator and method-sampler route kinds")

        optimizer = dict(self.optimizer)
        if set(optimizer) != {"kind", "betas", "epsilon", "weight_decay"} or optimizer["kind"] != "adam":
            raise ValueError("optimizer must be Adam with betas/epsilon/weight_decay")
        betas = tuple(float(value) for value in optimizer["betas"])
        if len(betas) != 2 or not all(0.0 <= value < 1.0 for value in betas):
            raise ValueError("Adam betas must contain two values in [0, 1)")
        if float(optimizer["epsilon"]) <= 0.0 or float(optimizer["weight_decay"]) < 0.0:
            raise ValueError("Adam epsilon/weight_decay are invalid")
        optimizer["betas"] = list(betas)

        schedule = dict(self.schedule)
        if set(schedule) != {"kind", "start", "end", "total_steps"} or schedule["kind"] != "cosine":
            raise ValueError("schedule must be one global cosine decay")
        if int(schedule["total_steps"]) != total_steps:
            raise ValueError("schedule total_steps must match lifecycle")
        if not float(schedule["start"]) >= float(schedule["end"]) > 0.0:
            raise ValueError("cosine schedule learning rates are invalid")

        mollification = dict(self.mollification)
        if set(mollification) != {"steps", "start_degrees", "samples"}:
            raise ValueError("mollification fields are invalid")
        if int(mollification["steps"]) < 0 or float(mollification["start_degrees"]) < 0.0 or int(mollification["samples"]) < 1:
            raise ValueError("mollification values are invalid")

        validation = dict(self.validation)
        if set(validation) != {"interval", "batches"} or min(int(value) for value in validation.values()) < 1:
            raise ValueError("validation interval and batches must be positive")
        if self.checkpoint_selection != "tail_guard":
            raise ValueError("training configs require tail_guard checkpoint selection")
        if not self.model_context or not self.filtering or not self.loss:
            raise ValueError("model_context, filtering and loss recipes are required")

        object.__setattr__(
            self,
            "source",
            {
                "family_id": source["family_id"],
                "materials": [{"locator": dict(item["locator"])} for item in materials],
            },
        )
        object.__setattr__(self, "online_query", online_query)
        object.__setattr__(self, "model_context", dict(self.model_context))
        object.__setattr__(self, "lifecycle", {"total_steps": total_steps, "materialization_step": materialization_step})
        object.__setattr__(self, "routes", routes)
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "mollification", mollification)
        object.__setattr__(self, "filtering", dict(self.filtering))
        object.__setattr__(self, "loss", dict(self.loss))
        object.__setattr__(self, "validation", validation)

    @property
    def total_steps(self) -> int:
        return int(self.lifecycle["total_steps"])

    @property
    def materialization_step(self) -> int:
        return int(self.lifecycle["materialization_step"])

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "method_key": self.method_key,
            "run_class": self.run_class,
            "correspondence_id": self.correspondence_id,
            "recipe_id": self.recipe_id,
            "source_adaptation_id": self.source_adaptation_id,
            "source": {
                "family_id": self.source["family_id"],
                "materials": [
                    {"locator": dict(item["locator"])}
                    for item in self.source["materials"]
                ],
            },
            "online_query": dict(self.online_query),
            "model_context": dict(self.model_context),
            "lifecycle": dict(self.lifecycle),
            "routes": [route.to_dict() for route in self.routes],
            "seed": self.seed,
            "device": self.device,
            "optimizer": dict(self.optimizer),
            "schedule": dict(self.schedule),
            "mollification": dict(self.mollification),
            "filtering": dict(self.filtering),
            "loss": dict(self.loss),
            "validation": dict(self.validation),
            "checkpoint_selection": self.checkpoint_selection,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingConfig":
        required = {
            "format_name", "format_version", "method_key", "run_class", "correspondence_id", "recipe_id",
            "source_adaptation_id", "source", "online_query", "model_context", "lifecycle", "routes",
            "seed", "device", "optimizer", "schedule", "mollification", "filtering", "loss",
            "validation", "checkpoint_selection",
        }
        if set(value) != required:
            raise ValueError(f"training config fields must be exactly {sorted(required)}")
        return cls(
            str(value["method_key"]), str(value["run_class"]), str(value["correspondence_id"]), str(value["recipe_id"]),
            str(value["source_adaptation_id"]), value["source"], value["online_query"], value["model_context"],
            value["lifecycle"], tuple(TrainingRoute.from_dict(item) for item in value["routes"]),
            int(value["seed"]), str(value["device"]), value["optimizer"], value["schedule"],
            value["mollification"], value["filtering"], value["loss"], value["validation"],
            str(value["checkpoint_selection"]), str(value["format_name"]), int(value["format_version"]),
        )

    @classmethod
    def load(cls, path: Path | str) -> "TrainingConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("training config root must be an object")
        return cls.from_dict(value)
