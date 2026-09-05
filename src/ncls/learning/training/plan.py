from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from ncls.core.identity import sha256_file, sha256_json
from ncls.learning.methods import get_method_plugin

from .config import TrainingConfig


_PUBLIC_KEY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRAGMENT_KINDS = ("base", "method", "data", "recipe")
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


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _same_value_category(left: object, right: object) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
    if left is None or right is None:
        return True
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return True
    return type(left) is type(right)


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
        if not _same_value_category(previous, value):
            raise ValueError(
                f"configuration merge changes the value category at {child_path!r}"
            )
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


def _safe_project_path(project_root: Path, value: object, name: str) -> Path:
    relative = str(value)
    posix = PurePosixPath(relative)
    if not relative or posix.is_absolute() or ".." in posix.parts or "\\" in relative:
        raise ValueError(f"{name} must be a safe project-relative POSIX path")
    path = (project_root / Path(*posix.parts)).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"{name} resolves outside the project root") from error
    return path


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
            "devices",
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
            optional={"batch_size_multiplier"},
        )
        devices_value = value["devices"]
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
    enabled: bool
    interval_steps: int
    reference_spp: int
    neural_mode: str
    neural_spp: int
    seed: int
    queue_capacity: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualEvalSettings":
        _require_exact_fields(
            "hooks.visual_eval",
            value,
            required={
                "enabled",
                "interval_steps",
                "reference_spp",
                "neural_mode",
                "neural_spp",
                "seed",
                "queue_capacity",
            },
        )
        result = cls(
            bool(value["enabled"]),
            int(value["interval_steps"]),
            int(value["reference_spp"]),
            str(value["neural_mode"]),
            int(value["neural_spp"]),
            int(value["seed"]),
            int(value["queue_capacity"]),
        )
        if min(result.interval_steps, result.reference_spp, result.queue_capacity) < 1:
            raise ValueError("visual eval cadence, reference spp and queue capacity must be positive")
        if result.neural_mode not in {"deferred", "path-tracing"}:
            raise ValueError("visual eval neural mode must be deferred or path-tracing")
        if result.neural_mode == "deferred" and result.neural_spp != 0:
            raise ValueError("visual eval deferred neural mode requires neural_spp=0")
        if (
            result.neural_mode == "path-tracing"
            and not 1 <= result.neural_spp <= result.reference_spp
        ):
            raise ValueError(
                "visual eval path-tracing neural spp must be within [1, reference_spp]"
            )
        if result.seed < 0:
            raise ValueError("visual eval seed must be nonnegative")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "interval_steps": self.interval_steps,
            "reference_spp": self.reference_spp,
            "neural_mode": self.neural_mode,
            "neural_spp": self.neural_spp,
            "seed": self.seed,
            "queue_capacity": self.queue_capacity,
        }


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


@dataclass(frozen=True)
class PlanInputRecord:
    kind: str
    key: str
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "key": self.key,
            "path": self.path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanInputRecord":
        _require_exact_fields(
            "training plan input",
            value,
            required={"kind", "key", "path", "sha256"},
        )
        result = cls(
            str(value["kind"]),
            str(value["key"]),
            str(value["path"]),
            str(value["sha256"]),
        )
        if not result.kind or not result.key or not result.path:
            raise ValueError("training plan input identity is incomplete")
        if not re.fullmatch(r"[0-9a-f]{64}", result.sha256):
            raise ValueError("training plan input SHA-256 is invalid")
        return result


@dataclass(frozen=True)
class ResolvedTrainingPlan:
    selection: ComponentSelection
    training: Mapping[str, Any]
    execution: ExecutionSettings
    hooks: HookSettings
    method_descriptor: Mapping[str, Any]
    inputs: tuple[PlanInputRecord, ...]
    overrides: Mapping[str, Any]
    format_name: str = "ncls.training-plan"
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_name != "ncls.training-plan" or self.format_version != 1:
            raise ValueError("unsupported resolved training plan format")
        object.__setattr__(self, "training", _freeze(_thaw(self.training)))
        object.__setattr__(
            self, "method_descriptor", _freeze(_thaw(self.method_descriptor))
        )
        object.__setattr__(self, "overrides", _freeze(_thaw(self.overrides)))

    def to_runtime_config(self) -> TrainingConfig:
        """Materialize the validated phase graph consumed by TrainingEngine."""

        value = _thaw(self.training)
        if self.execution.batch_size_multiplier != 1:
            for phase in value["phases"]:
                for route in phase["routes"]:
                    route["batch_size"] = (
                        int(route["batch_size"])
                        * self.execution.batch_size_multiplier
                    )
        value["format_name"] = "ncls.training-config"
        value["format_version"] = 4
        return TrainingConfig.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "selection": self.selection.to_dict(),
            "training": _thaw(self.training),
            "execution": self.execution.to_dict(),
            "hooks": self.hooks.to_dict(),
            "method_descriptor": _thaw(self.method_descriptor),
            "inputs": [item.to_dict() for item in self.inputs],
            "overrides": _thaw(self.overrides),
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "ResolvedTrainingPlan":
        """读取自包含的冻结计划；不把其历史实现身份改写成当前实现。"""
        _require_exact_fields(
            "resolved training plan",
            value,
            required={
                "format_name",
                "format_version",
                "selection",
                "training",
                "execution",
                "hooks",
                "method_descriptor",
                "inputs",
                "overrides",
            },
        )
        input_values = value["inputs"]
        if not isinstance(input_values, (list, tuple)) or not input_values:
            raise ValueError("resolved training plan inputs must be a nonempty list")
        selection = ComponentSelection.from_dict(
            _require_mapping("resolved training plan selection", value["selection"])
        )
        result = cls(
            selection,
            _require_mapping("resolved training plan training", value["training"]),
            ExecutionSettings.from_dict(
                _require_mapping("resolved training plan execution", value["execution"])
            ),
            HookSettings.from_dict(
                _require_mapping("resolved training plan hooks", value["hooks"])
            ),
            _require_mapping(
                "resolved training plan method descriptor", value["method_descriptor"]
            ),
            tuple(
                PlanInputRecord.from_dict(
                    _require_mapping("resolved training plan input", item)
                )
                for item in input_values
            ),
            _require_mapping("resolved training plan overrides", value["overrides"]),
            str(value["format_name"]),
            int(value["format_version"]),
        )
        result.to_runtime_config()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResolvedTrainingPlan":
        result = cls.from_manifest(value)
        plugin = get_method_plugin(result.selection.method)
        expected_descriptor = {
            "public_key": plugin.key,
            "implementation_key": plugin.descriptor.method_key,
            "descriptor_sha256": plugin.descriptor.descriptor_sha256,
            "implementation_sha256": plugin.descriptor.implementation_sha256,
            "facets": dict(plugin.facet_identities),
        }
        if _thaw(result.method_descriptor) != expected_descriptor:
            raise ValueError("resolved training plan method implementation drifted")
        result.to_runtime_config()
        return result

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class _Fragment:
    kind: str
    key: str
    extends: str | None
    compatible_methods: tuple[str, ...]
    payload: Mapping[str, Any]
    record: PlanInputRecord


class TrainingPlanResolver:
    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root).resolve()
        self.config_root = self.project_root / "configs" / "training"

    def _relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.project_root).as_posix()

    def _fragment_path(self, kind: str, key: str) -> Path:
        directory = {"method": "methods", "recipe": "recipes"}.get(kind, kind)
        return self.config_root / directory / f"{key}.yaml"

    def _load_fragment(self, kind: str, key: str) -> _Fragment:
        if kind not in _FRAGMENT_KINDS:
            raise ValueError(f"unsupported training fragment kind {kind!r}")
        key = _require_public_key(f"{kind} key", key)
        path = self._fragment_path(kind, key)
        value = _load_yaml_mapping(path)
        _require_exact_fields(
            f"{kind} fragment {key!r}",
            value,
            required={
                "format_name", "format_version", "kind", "key", "extends",
                "compatible_methods", "payload",
            },
        )
        if value["format_name"] != "ncls.training-fragment" or int(value["format_version"]) != 1:
            raise ValueError(f"unsupported format for {kind} fragment {key!r}")
        if value["kind"] != kind or value["key"] != key:
            raise ValueError(f"{kind} fragment path and declared identity disagree")
        extends = value["extends"]
        if extends is not None:
            extends = _require_public_key(f"{kind} fragment extends", extends)
        compatible_value = value["compatible_methods"]
        if not isinstance(compatible_value, (list, tuple)):
            raise ValueError(f"{kind} fragment compatible_methods must be a list")
        compatible = tuple(
            _require_public_key("compatible method", item) for item in compatible_value
        )
        if len(set(compatible)) != len(compatible):
            raise ValueError(f"{kind} fragment repeats a compatible method")
        payload = _require_mapping(f"{kind} fragment payload", value["payload"])
        unknown_payload = set(payload) - _PAYLOAD_FIELDS
        if unknown_payload:
            raise ValueError(
                f"{kind} fragment payload has unknown fields {sorted(unknown_payload)}"
            )
        for name, item in payload.items():
            _require_mapping(f"{kind} fragment payload.{name}", item)
        return _Fragment(
            kind,
            key,
            extends,
            compatible,
            deepcopy(dict(payload)),
            PlanInputRecord(kind, key, self._relative_path(path), sha256_file(path)),
        )

    def _fragment_chain(self, kind: str, key: str) -> tuple[_Fragment, ...]:
        result: list[_Fragment] = []
        visiting: list[str] = []
        current: str | None = key
        while current is not None:
            if current in visiting:
                cycle = " -> ".join([*visiting, current])
                raise ValueError(f"training fragment inheritance cycle: {cycle}")
            visiting.append(current)
            fragment = self._load_fragment(kind, current)
            result.append(fragment)
            current = fragment.extends
        result.reverse()
        return tuple(result)

    def _expand_material_set(
        self, training: dict[str, Any]
    ) -> tuple[dict[str, Any], PlanInputRecord | None]:
        source = _require_mapping("training.source", training.get("source"))
        if "material_set" not in source:
            return training, None
        if "materials" in source:
            raise ValueError("training.source cannot contain both materials and material_set")
        material_set = _require_mapping("training.source.material_set", source["material_set"])
        _require_exact_fields(
            "training.source.material_set",
            material_set,
            required={"resolver", "path", "module_root", "expected_count"},
        )
        if material_set["resolver"] != "mdl-metal-registry":
            raise ValueError(f"unsupported material set resolver {material_set['resolver']!r}")
        registry_path = _safe_project_path(
            self.project_root, material_set["path"], "training.source.material_set.path"
        )
        from ncls.source_materials.mdl_metal import MdlMetalRegistry

        registry = MdlMetalRegistry.load(registry_path)
        module_root = str(material_set["module_root"])
        materials = []
        for item in registry.exports:
            locator = dict(item.exact_locator)
            locator["module_root"] = module_root
            materials.append({"locator": locator})
        expected_count = int(material_set["expected_count"])
        if len(materials) != expected_count:
            raise ValueError(
                f"material set expected {expected_count} records, found {len(materials)}"
            )
        result = deepcopy(training)
        result["source"] = dict(source)
        del result["source"]["material_set"]
        result["source"]["materials"] = materials
        return result, PlanInputRecord(
            "source-set",
            str(material_set["resolver"]),
            self._relative_path(registry_path),
            sha256_file(registry_path),
        )

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

    def resolve(
        self,
        run_path: Path | str,
        *,
        devices: Sequence[int] | None = None,
    ) -> ResolvedTrainingPlan:
        path = Path(run_path)
        if not path.is_absolute():
            path = self.project_root / path
        path = path.resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as error:
            raise ValueError("training run YAML must be inside the project root") from error
        run = _load_yaml_mapping(path)
        _require_exact_fields(
            "training run",
            run,
            required={"format_name", "format_version", "compose"},
            optional={"training", "execution", "hooks"},
        )
        if run["format_name"] != "ncls.training-run" or int(run["format_version"]) != 1:
            raise ValueError("unsupported training run format")
        compose = _require_mapping("training run compose", run["compose"])
        _require_exact_fields(
            "training run compose", compose, required={"method", "data", "recipe"}
        )
        selection = ComponentSelection(
            method=str(compose["method"]),
            data=str(compose["data"]),
            recipe=str(compose["recipe"]),
        )

        merged: dict[str, Any] = {}
        records: list[PlanInputRecord] = []
        for kind, key in (
            ("base", selection.base),
            ("method", selection.method),
            ("data", selection.data),
            ("recipe", selection.recipe),
        ):
            for fragment in self._fragment_chain(kind, key):
                if fragment.compatible_methods and selection.method not in fragment.compatible_methods:
                    raise ValueError(
                        f"{kind} fragment {fragment.key!r} is incompatible with method "
                        f"{selection.method!r}"
                    )
                merged = _deep_merge(merged, fragment.payload, "")
                records.append(fragment.record)

        explicit = {
            name: deepcopy(dict(_require_mapping(f"training run {name}", run[name])))
            for name in _PAYLOAD_FIELDS
            if name in run
        }
        merged = _deep_merge(merged, explicit, "")
        records.append(
            PlanInputRecord("run", path.stem, self._relative_path(path), sha256_file(path))
        )
        override_manifest: dict[str, Any] = {}
        if devices is not None:
            device_values = [int(item) for item in devices]
            merged = _deep_merge(merged, {"execution": {"devices": device_values}}, "")
            override_manifest["execution.devices"] = device_values

        _require_exact_fields(
            "resolved training payload", merged, required={"training", "execution", "hooks"}
        )
        training_value = deepcopy(
            dict(_require_mapping("resolved training", merged["training"]))
        )
        if "format_name" in training_value or "format_version" in training_value:
            raise ValueError("resolved training payload must not expose the legacy config format")
        training_value, source_record = self._expand_material_set(training_value)
        training_value = self._materialize_phase_graph(training_value)
        if source_record is not None:
            records.append(source_record)
        legacy_value = {
            **training_value,
            "format_name": "ncls.training-config",
            "format_version": 4,
        }
        legacy_config = TrainingConfig.from_dict(legacy_value)

        plugin = get_method_plugin(selection.method)
        descriptor = plugin.descriptor
        if legacy_config.method_key != descriptor.method_key:
            raise ValueError("resolved method component disagrees with its product descriptor")
        plugin.lifecycle.validate_training_plan(legacy_config.to_dict())
        method_descriptor = {
            "public_key": selection.method,
            "implementation_key": descriptor.method_key,
            "descriptor_sha256": descriptor.descriptor_sha256,
            "implementation_sha256": descriptor.implementation_sha256,
            "facets": dict(plugin.facet_identities),
        }
        execution = ExecutionSettings.from_dict(
            _require_mapping("resolved execution", merged["execution"])
        )
        hooks = HookSettings.from_dict(_require_mapping("resolved hooks", merged["hooks"]))
        return ResolvedTrainingPlan(
            selection,
            training_value,
            execution,
            hooks,
            method_descriptor,
            tuple(records),
            override_manifest,
        )


__all__ = [
    "ComponentSelection",
    "ExecutionSettings",
    "HookSettings",
    "PlanInputRecord",
    "ResolvedTrainingPlan",
    "TensorBoardSettings",
    "TrainingPlanResolver",
    "VisualEvalSettings",
]
