from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[4]
path = root / 'src/ncls/learning/training/plan.py'
old = path.read_text(encoding='utf-8')
header = '''from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml

from ncls.core.identity import sha256_file, sha256_json
from ncls.learning.methods import get_method_plugin
from .config import TrainingConfig

_PUBLIC_KEY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_PAYLOAD_FIELDS = {"training", "execution", "hooks"}

'''
helpers = old[old.index('def _require_public_key'):old.index('def _freeze')]
merge = old[old.index('def _deep_merge'):old.index('def _safe_project_path')]
start = merge.index('        if not _same_value_category')
end = merge.index('        result[key] = deepcopy(value)', start)
merge = merge[:start] + merge[end:]
settings = old[old.index('@dataclass(frozen=True)\nclass ComponentSelection'):old.index('@dataclass(frozen=True)\nclass VisualEvalSettings')]
visual = '''@dataclass(frozen=True)
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


'''
hooks = old[old.index('@dataclass(frozen=True)\nclass HookSettings'):old.index('@dataclass(frozen=True)\nclass PlanInputRecord')]
plan = '''@dataclass
class ResolvedTrainingPlan:
    selection: ComponentSelection
    training: TrainingConfig
    execution: ExecutionSettings
    hooks: HookSettings
    inputs: tuple[Mapping[str, str], ...] = ()

    def to_runtime_config(self) -> TrainingConfig:
        return self.training

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

'''
phase = old[old.index('    @staticmethod\n    def _materialize_phase_graph'):old.index('    def resolve(', old.index('    @staticmethod\n    def _materialize_phase_graph'))]
resolve = '''    def resolve(self, run_path: Path | str, *, devices: Sequence[int] | None = None) -> ResolvedTrainingPlan:
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
        method = get_method_plugin(selection.method)
        if config.method_key != method.descriptor.method_key:
            raise ValueError("config 的方法与 compose.method 不一致")
        method.validate_training_config(config.to_dict())
        return ResolvedTrainingPlan(selection, config, execution, hooks, tuple(inputs))
'''
path.write_text(header + helpers + merge + settings + visual + hooks + plan + phase + resolve, encoding='utf-8', newline='\n')

config_path = root / 'src/ncls/learning/training/config.py'
value = config_path.read_text(encoding='utf-8')
value = value.replace('    format_name: str = "ncls.training-config"\n    format_version: int = 4\n', '    checkpoint_interval: int = 5000\n')
value = value.replace('        if self.format_name != "ncls.training-config" or self.format_version != 4:\n            raise ValueError("unsupported training config format")\n', '        if self.checkpoint_interval < 1:\n            raise ValueError("checkpoint_interval 必须为正整数")\n')
value = value.replace('            "format_name": self.format_name,\n            "format_version": self.format_version,\n', '            "checkpoint_interval": self.checkpoint_interval,\n')
value = value.replace('            "format_name", "format_version", "method_key", "run_class",', '            "method_key", "run_class",')
value = value.replace('        if set(value) != required:\n            raise ValueError(f"training config fields must be exactly {sorted(required)}")', '        if set(value) - required - {"checkpoint_interval"} or required - set(value):\n            raise ValueError(f"training config fields are invalid: {sorted(set(value) ^ required)}")')
value = value.replace('            str(value["format_name"]),\n            int(value["format_version"]),', '            int(value.get("checkpoint_interval", 5000)),')
config_path.write_text(value, encoding='utf-8', newline='\n')
init_path = root / 'src/ncls/learning/training/__init__.py'
value = init_path.read_text(encoding='utf-8').replace('    PlanInputRecord,\n','').replace('    "PlanInputRecord",\n','')
init_path.write_text(value, encoding='utf-8', newline='\n')

for path in (root / 'configs/training').rglob('*.yaml'):
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if value.get('format_name') == 'ncls.training-fragment':
        value = {**({'extends': value['extends']} if value.get('extends') else {}), **value['payload']}
    else:
        value.pop('format_name', None)
        value.pop('format_version', None)
    visual = value.get('hooks', {}).get('visual_eval', {})
    visual.pop('queue_capacity', None)
    if 'reference_spp' in visual:
        visual['reference_spp'] = 128
    # 新的默认图像观测生效；Linux 在装配处绑定空实现。
    if visual:
        visual['enabled'] = True
    if path.name == 'default.yaml':
        value['training']['checkpoint_interval'] = 5000
        visual.update(width=640, height=360)
    material_set = value.get('training', {}).get('source', {}).get('material_set', {})
    material_set.pop('expected_count', None)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding='utf-8', newline='\n')
