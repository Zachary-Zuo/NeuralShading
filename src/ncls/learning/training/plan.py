from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml

from ncls.core.identity import sha256_file, sha256_json
from ncls.learning.methods import get_method
from .config import TrainingConfig

_PUBLIC_KEY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_PAYLOAD_FIELDS = {"training", "execution", "hooks"}

def _require_public_key(name: str, value: object) -> str:
    result = str(value)
    if not _PUBLIC_KEY.fullmatch(result) or "@" in result:
        raise ValueError(f"{name} must be a lower-kebab public key without a version suffix")
    return result


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed object")
    return value


def _require_exact_fields(
    name: str,
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    actual = set(value)
    missing = required - actual
    unknown = actual - required - optional
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ValueError(f"{name} fields are invalid: {', '.join(details)}")


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any], path: str) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in overlay.items():
        child_path = f"{path}.{key}" if path else key
        if key not in result:
            result[key] = deepcopy(value)
            continue
        previous = result[key]
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(previous, value, child_path)
            continue
        result[key] = deepcopy(value)
    return result


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read training YAML {path}") from error

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(
        loader: UniqueKeySafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as error:
                raise ValueError(f"YAML mapping key in {path} is not hashable") from error
            if duplicate:
                raise ValueError(f"duplicate YAML key {key!r} in {path}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    try:
        value = yaml.load(text, Loader=UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid training YAML {path}: {error}") from error
    return _require_mapping(f"training YAML {path}", value)


@dataclass(frozen=True)
class ComponentSelection:
    method: str
    data: str
    recipe: str
    base: str = "default"

    def __post_init__(self) -> None:
        for name in ("method", "data", "recipe", "base"):
            object.__setattr__(self, name, _require_public_key(name, getattr(self, name)))

    def to_dict(self) -> dict[str, str]:
        return {
            "base": self.base,
            "method": self.method,
            "data": self.data,
            "recipe": self.recipe,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ComponentSelection":
        _require_exact_fields(
            "training plan selection",
            value,
            required={"base", "method", "data", "recipe"},
        )
        return cls(
            str(value["method"]),
            str(value["data"]),
            str(value["recipe"]),
            str(value["base"]),
        )


@dataclass(frozen=True)
class ExecutionSettings:
    devices: tuple[int, ...]
    num_workers: int
    host_prefetch: int
    ready_batches: int
    reference_batch_steps: int
    reference_inflight: int
    transfer_streams: int
    residency_budget_mib: int
    batch_size_multiplier: int = 1

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionSettings":
        required = {
            "num_workers",
            "host_prefetch",
            "ready_batches",
            "reference_batch_steps",
            "reference_inflight",
            "transfer_streams",
            "residency",
        }
        _require_exact_fields(
            "execution",
            value,
            required=required,
            optional={"batch_size_multiplier", "devices"},
        )
        devices_value = value.get("devices", [0])
        if not isinstance(devices_value, (list, tuple)):
            raise ValueError("execution.devices must be a list")
        devices = tuple(int(item) for item in devices_value)
        if not devices or any(item < 0 for item in devices) or len(set(devices)) != len(devices):
            raise ValueError("execution.devices must contain unique nonnegative GPU indices")
        residency = _require_mapping("execution.residency", value["residency"])
        _require_exact_fields("execution.residency", residency, required={"budget_mib"})
        numbers = {
            "num_workers": int(value["num_workers"]),
            "host_prefetch": int(value["host_prefetch"]),
            "ready_batches": int(value["ready_batches"]),
            "reference_batch_steps": int(value["reference_batch_steps"]),
            "reference_inflight": int(value["reference_inflight"]),
            "transfer_streams": int(value["transfer_streams"]),
            "residency_budget_mib": int(residency["budget_mib"]),
            "batch_size_multiplier": int(value.get("batch_size_multiplier", 1)),
        }
        if numbers["num_workers"] < 0 or numbers["transfer_streams"] < 0:
            raise ValueError("execution worker/stream counts must be nonnegative")
        if any(
            numbers[name] < 1
            for name in (
                "host_prefetch",
                "ready_batches",
                "reference_batch_steps",
                "reference_inflight",
                "residency_budget_mib",
                "batch_size_multiplier",
            )
        ):
            raise ValueError("execution queue, reference and residency sizes must be positive")
        return cls(devices=devices, **numbers)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "devices": list(self.devices),
            "num_workers": self.num_workers,
            "host_prefetch": self.host_prefetch,
            "ready_batches": self.ready_batches,
            "reference_batch_steps": self.reference_batch_steps,
            "reference_inflight": self.reference_inflight,
            "transfer_streams": self.transfer_streams,
            "residency": {"budget_mib": self.residency_budget_mib},
        }
        if self.batch_size_multiplier != 1:
            result["batch_size_multiplier"] = self.batch_size_multiplier
        return result


@dataclass(frozen=True)
class TensorBoardSettings:
    enabled: bool
    flush_seconds: int
    queue_capacity: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorBoardSettings":
        _require_exact_fields(
            "hooks.tensorboard", value, required={"enabled", "flush_seconds", "queue_capacity"}
        )
        result = cls(
            bool(value["enabled"]), int(value["flush_seconds"]), int(value["queue_capacity"])
        )
        if result.flush_seconds < 1 or result.queue_capacity < 1:
            raise ValueError("TensorBoard flush interval and queue capacity must be positive")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "flush_seconds": self.flush_seconds,
            "queue_capacity": self.queue_capacity,
        }


@dataclass(frozen=True)
class VisualEvalSettings:
    enabled: bool = True
    interval_steps: int = 5000
    reference_spp: int = 128
    neural_mode: str = "deferred"
    neural_spp: int = 0
    seed: int = 20260904
    width: int = 640
    height: int = 360

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualEvalSettings":
        result = cls(**value)
        for name in ("interval_steps", "reference_spp", "width", "height"):
            number = getattr(result, name)
            if type(number) is not int or number < 1:
                raise ValueError(f"hooks.visual_eval.{name} 必须为正整数")
        if type(result.neural_spp) is not int or result.neural_spp < 0:
            raise ValueError("hooks.visual_eval.neural_spp 必须为非负整数")
        if result.neural_mode not in {"deferred", "path-tracing"}:
            raise ValueError("图像模式应为 deferred 或 path-tracing")
        if result.neural_mode == "path-tracing" and result.neural_spp < 1:
            raise ValueError("path-tracing 的 neural_spp 必须为正整数")
        return result

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass(frozen=True)
class HookSettings:
    tensorboard: TensorBoardSettings
    visual_eval: VisualEvalSettings

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HookSettings":
        _require_exact_fields("hooks", value, required={"tensorboard", "visual_eval"})
        return cls(
            TensorBoardSettings.from_dict(
                _require_mapping("hooks.tensorboard", value["tensorboard"])
            ),
            VisualEvalSettings.from_dict(
                _require_mapping("hooks.visual_eval", value["visual_eval"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensorboard": self.tensorboard.to_dict(),
            "visual_eval": self.visual_eval.to_dict(),
        }


@dataclass
class ResolvedTrainingPlan:
    selection: ComponentSelection
    training: TrainingConfig
    execution: ExecutionSettings
    hooks: HookSettings
    inputs: tuple[Mapping[str, str], ...] = ()


    def to_dict(self) -> dict[str, Any]:
        return {
            "selection": self.selection.to_dict(), "training": self.training.to_dict(),
            "execution": self.execution.to_dict(), "hooks": self.hooks.to_dict(),
            "inputs": list(self.inputs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResolvedTrainingPlan":
        return cls(
            ComponentSelection.from_dict(value["selection"]), TrainingConfig.from_dict(value["training"]),
            ExecutionSettings.from_dict(value["execution"]), HookSettings.from_dict(value["hooks"]),
            tuple(value["inputs"]),
        )

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


class TrainingPlanResolver:
    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root).resolve()
        self.config_root = self.project_root / "configs/training"

    def _record(self, path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": sha256_file(path)}

    def _fragment(self, kind: str, key: str, stack: tuple[str, ...] = ()):
        key = _require_public_key(kind, key)
        if key in stack:
            raise ValueError(f"配置继承成环：{' -> '.join((*stack, key))}")
        directory = {"method": "methods", "recipe": "recipes"}.get(kind, kind)
        path = self.config_root / directory / f"{key}.yaml"
        value = dict(_load_yaml_mapping(path))
        parent = value.pop("extends", None)
        _require_exact_fields(f"fragment {path}", value, required=set(), optional=_PAYLOAD_FIELDS)
        merged, inputs = ({}, []) if parent is None else self._fragment(kind, parent, (*stack, key))
        return _deep_merge(merged, value, ""), [*inputs, self._record(path)]

    def _expand_material_set(self, training: dict[str, Any]) -> None:
        source = training["source"]
        if "material_set" not in source:
            return
        spec = source.pop("material_set")
        if "materials" in source:
            raise ValueError("source 只能指定 materials 或 material_set 其中一个")
        if spec["resolver"] != "mdl-metal-registry":
            raise ValueError(f"未知 material_set resolver：{spec['resolver']}")
        from ncls.source_materials.mdl_metal import MdlMetalRegistry

        registry = MdlMetalRegistry.load(self.project_root / spec["path"])
        source["materials"] = [
            {"locator": {**item.exact_locator, "module_root": spec["module_root"]}}
            for item in registry.exports
        ]

    @staticmethod
    def _materialize_phase_graph(training: dict[str, Any]) -> dict[str, Any]:
        if "phase_graph" not in training:
            return training
        if "phases" in training:
            raise ValueError("resolved training cannot contain both phases and phase_graph")
        graph = _require_mapping("training.phase_graph", training["phase_graph"])
        _require_exact_fields(
            "training.phase_graph", graph, required={"order", "definitions"}
        )
        order_value = graph["order"]
        if not isinstance(order_value, (list, tuple)):
            raise ValueError("training.phase_graph.order must be a list")
        order = tuple(str(item) for item in order_value)
        if not order or len(set(order)) != len(order) or any(not item for item in order):
            raise ValueError("training.phase_graph.order must be unique and nonempty")
        definitions = _require_mapping(
            "training.phase_graph.definitions", graph["definitions"]
        )
        if set(definitions) != set(order):
            raise ValueError(
                "training.phase_graph definitions must exactly match the declared order"
            )
        phases = []
        for name in order:
            definition = deepcopy(
                dict(
                    _require_mapping(
                        f"training.phase_graph.definitions.{name}", definitions[name]
                    )
                )
            )
            if "name" in definition and definition["name"] != name:
                raise ValueError(f"training phase definition {name!r} has a conflicting name")
            definition["name"] = name
            phases.append(definition)
        result = deepcopy(training)
        del result["phase_graph"]
        result["phases"] = phases
        return result

    def resolve(self, run_path: Path | str, *, devices: Sequence[int] | None = None) -> ResolvedTrainingPlan:
        path = Path(run_path)
        if not path.is_absolute():
            path = self.project_root / path
        run = dict(_load_yaml_mapping(path))
        compose = run.pop("compose")
        selection = ComponentSelection(**compose)
        _require_exact_fields("training run", run, required=set(), optional=_PAYLOAD_FIELDS)
        merged: dict[str, Any] = {}
        inputs = []
        for kind in ("base", "method", "data", "recipe"):
            fragment, records = self._fragment(kind, getattr(selection, kind))
            merged = _deep_merge(merged, fragment, "")
            inputs.extend(records)
        merged = _deep_merge(merged, run, "")
        inputs.append(self._record(path))
        if devices is not None:
            merged["execution"]["devices"] = list(devices)
        execution = ExecutionSettings.from_dict(merged["execution"])
        hooks = HookSettings.from_dict(merged["hooks"])
        training = self._materialize_phase_graph(merged["training"])
        self._expand_material_set(training)
        if execution.batch_size_multiplier != 1:
            for phase in training["phases"]:
                for route in phase["routes"]:
                    route["batch_size"] *= execution.batch_size_multiplier
        config = TrainingConfig.from_dict(training)
        method = get_method(selection.method)
        if config.method_key != method.descriptor.method_key:
            raise ValueError("config 的方法与 compose.method 不一致")
        method.validate_training_config(config.to_dict())
        return ResolvedTrainingPlan(selection, config, execution, hooks, tuple(inputs))
