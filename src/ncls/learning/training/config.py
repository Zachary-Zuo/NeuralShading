from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast

from ncls.core.identity import sha256_json


TrainingRouteKind = Literal["asset-tile", "reference-evaluator", "method-sampler"]


@dataclass(frozen=True)
class TrainingRoute:
    name: str
    kind: TrainingRouteKind
    batch_size: int
    direction_count: int
    seed_offset: int
    options: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name or self.kind not in {
            "asset-tile", "reference-evaluator", "method-sampler"
        }:
            raise ValueError("training route identity or kind is invalid")
        if self.batch_size < 1 or self.direction_count < 1 or self.seed_offset < 0:
            raise ValueError("training route sizes must be positive and seed_offset nonnegative")
        if self.kind == "asset-tile" and self.direction_count != 1:
            raise ValueError("asset-tile routes require direction_count=1")
        options = dict(self.options)
        if any(name in options for name in ("target_estimator", "query_role")):
            raise ValueError("training route contains a removed field")
        object.__setattr__(self, "options", options)

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
            str(value["name"]),
            cast(TrainingRouteKind, str(value["kind"])),
            int(value["batch_size"]),
            int(value["direction_count"]),
            int(value["seed_offset"]),
            value["options"],
        )


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    steps: int
    routes: tuple[TrainingRoute, ...]
    parameter_groups: tuple[str, ...]
    loss_terms: tuple[str, ...]
    recipes: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    optimizer_state_policy: Literal["reset", "carry-overlap"]
    schedule: Mapping[str, Any]
    precision: Mapping[str, Any]
    checkpoint_boundary: bool
    transition: str | None
    log_interval: int
    gradient_audit_interval: int
    prefetch_depth: int

    def __post_init__(self) -> None:
        if not self.name or self.steps < 1:
            raise ValueError("training phase identity and step count are required")
        routes = tuple(self.routes)
        if not routes or len({route.name for route in routes}) != len(routes):
            raise ValueError("training phase routes must be nonempty and uniquely named")
        if len({route.seed_offset for route in routes}) != len(routes):
            raise ValueError("training phase route seed offsets must be unique")
        groups = tuple(str(value) for value in self.parameter_groups)
        losses = tuple(str(value) for value in self.loss_terms)
        if not groups or len(set(groups)) != len(groups) or any(not value for value in groups):
            raise ValueError("training phase parameter_groups must be unique and nonempty")
        if not losses or len(set(losses)) != len(losses) or any(not value for value in losses):
            raise ValueError("training phase loss_terms must be unique and nonempty")
        recipes = dict(self.recipes)
        if not recipes:
            raise ValueError("training phase recipes are required")

        optimizer = dict(self.optimizer)
        if set(optimizer) != {"kind", "betas", "epsilon", "weight_decay"}:
            raise ValueError("phase optimizer fields are invalid")
        if optimizer["kind"] != "adam":
            raise ValueError("only the canonical Adam optimizer is supported")
        betas = tuple(float(value) for value in optimizer["betas"])
        if len(betas) != 2 or not all(0.0 <= value < 1.0 for value in betas):
            raise ValueError("Adam betas must contain two values in [0,1)")
        if float(optimizer["epsilon"]) <= 0.0 or float(optimizer["weight_decay"]) < 0.0:
            raise ValueError("Adam epsilon/weight_decay are invalid")
        optimizer["betas"] = list(betas)
        if self.optimizer_state_policy not in {"reset", "carry-overlap"}:
            raise ValueError("phase optimizer_state_policy is invalid")

        schedule = dict(self.schedule)
        if set(schedule) != {"kind", "start", "end", "total_steps", "offset"}:
            raise ValueError("phase schedule fields are invalid")
        if schedule["kind"] != "cosine":
            raise ValueError("only the canonical cosine schedule is supported")
        schedule_total = int(schedule["total_steps"])
        schedule_offset = int(schedule["offset"])
        if schedule_total < 1 or schedule_offset < 0 or schedule_offset + self.steps > schedule_total:
            raise ValueError("phase schedule range is outside its declared total_steps")
        if not float(schedule["start"]) >= float(schedule["end"]) > 0.0:
            raise ValueError("phase cosine schedule learning rates are invalid")
        schedule["total_steps"] = schedule_total
        schedule["offset"] = schedule_offset

        precision = dict(self.precision)
        if set(precision) != {"autocast", "gradient_scaler"}:
            raise ValueError("phase precision fields are invalid")
        autocast = str(precision["autocast"])
        if autocast not in {"fp32", "float16", "bfloat16"}:
            raise ValueError("phase autocast must be fp32, float16 or bfloat16")
        scaler = bool(precision["gradient_scaler"])
        if scaler and autocast != "float16":
            raise ValueError("gradient scaling is only valid with float16 autocast")
        precision = {"autocast": autocast, "gradient_scaler": scaler}

        if self.transition is not None and not str(self.transition):
            raise ValueError("phase transition must be null or a nonempty identifier")
        if min(self.log_interval, self.gradient_audit_interval, self.prefetch_depth) < 1:
            raise ValueError("phase cadence and prefetch depth must be positive")

        object.__setattr__(self, "routes", routes)
        object.__setattr__(self, "parameter_groups", groups)
        object.__setattr__(self, "loss_terms", losses)
        object.__setattr__(self, "recipes", recipes)
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "transition", None if self.transition is None else str(self.transition))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": self.steps,
            "routes": [route.to_dict() for route in self.routes],
            "parameter_groups": list(self.parameter_groups),
            "loss_terms": list(self.loss_terms),
            "recipes": dict(self.recipes),
            "optimizer": dict(self.optimizer),
            "optimizer_state_policy": self.optimizer_state_policy,
            "schedule": dict(self.schedule),
            "precision": dict(self.precision),
            "checkpoint_boundary": self.checkpoint_boundary,
            "transition": self.transition,
            "log_interval": self.log_interval,
            "gradient_audit_interval": self.gradient_audit_interval,
            "prefetch_depth": self.prefetch_depth,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingPhase":
        required = {
            "name", "steps", "routes", "parameter_groups", "loss_terms", "recipes",
            "optimizer", "optimizer_state_policy", "schedule", "precision",
            "checkpoint_boundary", "transition", "log_interval",
            "gradient_audit_interval", "prefetch_depth",
        }
        if set(value) != required:
            raise ValueError(f"training phase fields must be exactly {sorted(required)}")
        if not isinstance(value["recipes"], Mapping):
            raise ValueError("training phase recipes must be an object")
        return cls(
            str(value["name"]),
            int(value["steps"]),
            tuple(TrainingRoute.from_dict(item) for item in value["routes"]),
            tuple(str(item) for item in value["parameter_groups"]),
            tuple(str(item) for item in value["loss_terms"]),
            value["recipes"],
            value["optimizer"],
            cast(
                Literal["reset", "carry-overlap"],
                str(value["optimizer_state_policy"]),
            ),
            value["schedule"],
            value["precision"],
            bool(value["checkpoint_boundary"]),
            None if value["transition"] is None else str(value["transition"]),
            int(value["log_interval"]),
            int(value["gradient_audit_interval"]),
            int(value["prefetch_depth"]),
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
    phases: tuple[TrainingPhase, ...]
    seed: int
    device: str
    validation: Mapping[str, Any]
    checkpoint_selection: str
    format_name: str = "ncls.training-config"
    format_version: int = 4

    def __post_init__(self) -> None:
        if self.format_name != "ncls.training-config" or self.format_version != 4:
            raise ValueError("unsupported training config format")
        if not self.method_key or self.run_class not in {"smoke", "profile", "adapted", "formal"}:
            raise ValueError("training method identity or run_class is invalid")
        if (
            not self.correspondence_id
            or not self.recipe_id
            or not self.source_adaptation_id
            or self.seed < 0
            or not self.device
        ):
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
        if not online_query or any(name in online_query for name in ("offline", "recorded", "hdf5")):
            raise ValueError("online_query must be a nonempty online recipe")
        phases = tuple(self.phases)
        if not phases or len({phase.name for phase in phases}) != len(phases):
            raise ValueError("training phase graph must be nonempty and uniquely named")
        route_kinds = {route.kind for phase in phases for route in phase.routes}
        required_kinds = {
            str(kind) for kind in self.model_context.get("required_route_kinds", ())
        }
        if required_kinds and not required_kinds.issubset(route_kinds):
            raise ValueError("training phase graph omits a required typed route kind")
        validation = dict(self.validation)
        if set(validation) != {"interval", "batches"} or min(int(value) for value in validation.values()) < 1:
            raise ValueError("validation interval and batches must be positive")
        if self.checkpoint_selection != "tail_guard":
            raise ValueError("training configs require tail_guard checkpoint selection")
        if not self.model_context:
            raise ValueError("model_context is required")
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
        object.__setattr__(self, "phases", phases)
        object.__setattr__(
            self,
            "validation",
            {"interval": int(validation["interval"]), "batches": int(validation["batches"])},
        )

    @property
    def total_steps(self) -> int:
        return sum(phase.steps for phase in self.phases)

    @property
    def all_routes(self) -> tuple[TrainingRoute, ...]:
        ordered: dict[str, TrainingRoute] = {}
        for phase in self.phases:
            for route in phase.routes:
                previous = ordered.setdefault(route.name, route)
                if previous != route:
                    raise ValueError(f"route {route.name!r} has conflicting phase definitions")
        return tuple(ordered.values())

    def phase_start_step(self, phase_index: int) -> int:
        if not 0 <= phase_index < len(self.phases):
            raise IndexError("training phase index is out of range")
        return sum(phase.steps for phase in self.phases[:phase_index])

    def locate_step(self, global_step: int) -> tuple[int, int]:
        if not 0 <= global_step <= self.total_steps:
            raise ValueError("global training step is out of range")
        if global_step == self.total_steps:
            return len(self.phases), 0
        cursor = global_step
        for phase_index, phase in enumerate(self.phases):
            if cursor < phase.steps:
                return phase_index, cursor
            cursor -= phase.steps
        raise AssertionError("unreachable phase cursor")

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
                    {"locator": dict(item["locator"])} for item in self.source["materials"]
                ],
            },
            "online_query": dict(self.online_query),
            "model_context": dict(self.model_context),
            "phases": [phase.to_dict() for phase in self.phases],
            "seed": self.seed,
            "device": self.device,
            "validation": dict(self.validation),
            "checkpoint_selection": self.checkpoint_selection,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingConfig":
        required = {
            "format_name", "format_version", "method_key", "run_class",
            "correspondence_id", "recipe_id", "source_adaptation_id", "source",
            "online_query", "model_context", "phases", "seed", "device",
            "validation", "checkpoint_selection",
        }
        if set(value) != required:
            raise ValueError(f"training config fields must be exactly {sorted(required)}")
        return cls(
            str(value["method_key"]),
            str(value["run_class"]),
            str(value["correspondence_id"]),
            str(value["recipe_id"]),
            str(value["source_adaptation_id"]),
            value["source"],
            value["online_query"],
            value["model_context"],
            tuple(TrainingPhase.from_dict(item) for item in value["phases"]),
            int(value["seed"]),
            str(value["device"]),
            value["validation"],
            str(value["checkpoint_selection"]),
            str(value["format_name"]),
            int(value["format_version"]),
        )
