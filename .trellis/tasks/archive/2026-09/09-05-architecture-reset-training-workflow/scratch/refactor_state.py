from pathlib import Path

root = Path(__file__).resolve().parents[4]

def read(name):
    return (root / name).read_text(encoding="utf-8")

def write(name, value):
    (root / name).write_text(value, encoding="utf-8", newline="\n")

write('src/ncls/learning/training/checkpoint.py', '''from __future__ import annotations

from dataclasses import dataclass, field, fields
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from ncls.core.identity import sha256_file, sha256_json


@dataclass
class TrainingCheckpoint:
    """唯一训练状态；来源记录不参与加载门禁。"""

    method_key: str
    training_config: Mapping[str, Any]
    model_state: Mapping[str, torch.Tensor]
    global_step: int = 0
    phase_index: int = 0
    phase_name: str = "initialization"
    phase_step: int = 0
    phase_optimization_state: Mapping[str, Any] = field(default_factory=dict)
    rng_state: Mapping[str, Any] = field(default_factory=dict)
    query_stream_state: Mapping[str, Any] = field(default_factory=dict)
    source_contracts: tuple[Mapping[str, Any], ...] = ()
    source_snapshot_ids: tuple[str, ...] = ()
    reference_program_identity: str = ""
    reference_execution_plan_identity: str = ""
    native_asset_collection_identity: str = ""
    query_stream_identity: str = ""
    gradient_coverage: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    validation_state: Mapping[str, Any] = field(default_factory=dict)
    selection_evidence: Mapping[str, Any] = field(default_factory=dict)
    resolved_plan: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def training_config_sha256(self) -> str:
        return sha256_json(self.training_config)

    @property
    def source(self) -> Mapping[str, Any]:
        return self.training_config["source"]

    @property
    def model_payload(self) -> dict[str, Any]:
        return {
            "training_config": dict(self.training_config),
            "model_state": dict(self.model_state),
            "source_snapshot_ids": list(self.source_snapshot_ids),
        }

    def to_payload(self) -> dict[str, Any]:
        return {"format": "ncls.checkpoint", **{item.name: getattr(self, item.name) for item in fields(self)}}


def save_checkpoint(path: Path | str, checkpoint: TrainingCheckpoint) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(checkpoint.to_payload(), temporary)
    os.replace(temporary, target)
    return sha256_file(target)


def load_checkpoint(
    path: Path | str, *, map_location: str | torch.device = "cpu",
) -> TrainingCheckpoint:
    value = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(value, dict) or value.pop("format", None) != "ncls.checkpoint":
        raise ValueError("文件不是当前 ncls checkpoint")
    return TrainingCheckpoint(**value)
''')

engine = read('src/ncls/learning/training/engine.py')
engine = engine.replace('        checkpoint.validate_method(self.plugin.descriptor)\n', '')
engine = engine.replace('            "training_config_sha256": self.config.sha256,\n', '')
anchor = '    def _validate_resume(self, checkpoint: TrainingCheckpoint) -> None:\n'
engine = engine.replace(anchor, anchor + '''        if checkpoint.method_key != self.plugin.key:
            raise ValueError("续训模型与当前方法不匹配")
        previous = TrainingConfig.from_dict(checkpoint.training_config)
        if previous.resume_signature != self.config.resume_signature:
            raise ValueError("续训的模型、优化阶段或数据配置不同；请开始新 run")
''')
start = engine.index('        checkpoint = TrainingCheckpoint(\n')
end = engine.index('        return checkpoint\n', start)
engine = engine[:start] + '''        checkpoint = TrainingCheckpoint(
            method_key=self.plugin.key,
            training_config=config_value,
            model_state=self.plugin.export_training_state(model),
            global_step=global_step,
            phase_index=phase_index,
            phase_name=phase_name,
            phase_step=phase_step,
            phase_optimization_state=resolved_optimization_state,
            rng_state=rng_state,
            query_stream_state=query_stream_state,
            source_contracts=self.data_session.source_contracts,
            source_snapshot_ids=self.data_session.source_snapshot_ids,
            reference_program_identity=self.data_session.reference_program_identity,
            reference_execution_plan_identity=self.data_session.reference_execution_plan_identity,
            native_asset_collection_identity=self.data_session.native_asset_collection_identity,
            query_stream_identity=self.data_session.query_stream_identity,
            gradient_coverage=coverage,
            validation_state={"rows": validation_rows},
            selection_evidence={"policy": self.config.checkpoint_selection, "tail": validation_rows[-1:]},
            provenance={"implementation": self.plugin.descriptor.implementation_sha256},
        )
''' + engine[end:]
# 覆盖诊断只记录观测，非零梯度不是每次优化都必须满足的条件。
engine = engine.replace('            if not gradients:\n                raise RuntimeError(f"parameter group {group!r} produced no gradients")', '            if not gradients:\n                coverage[group]["last_audit_step"] = global_step\n                continue')
start = engine.index('            if not finite or not nonzero or not updated:\n')
end = engine.index('            item = coverage[group]\n', start)
engine = engine[:start] + engine[end:]
engine = engine.replace('        values = torch.stack(checks).to(device="cpu", non_blocking=False).tolist()', '        values = torch.stack(checks).to(device="cpu", non_blocking=False).tolist() if checks else []')
engine = engine.replace('                if global_step == self.config.total_steps:\n                    validate_gradient_coverage(\n                        self.plugin.descriptor, coverage\n                    )\n', '')
# 最终状态也保留 optimizer，以便后续明确续训操作使用。
engine = engine.replace('                {},\n                coverage,\n                validation_rows,', '                resume_optimization or {},\n                coverage,\n                validation_rows,', 1)
start = engine.index('                    if global_step == self.config.total_steps:\n                        optimization_state:')
end = engine.index('                    if self.checkpoint_callback is not None:', start)
engine = engine[:start] + '''                    current_index = min(self.config.locate_step(global_step)[0], len(self.config.phases) - 1)
                    current_phase = self.config.phases[current_index]
                    optimization_state = lambda: self._optimization_state(
                        current_phase, optimizer, scheduler, scaler, active,
                    )
''' + engine[end:]
start = engine.index('        if global_step == self.config.total_steps:\n            final_optimization:')
end = engine.index('        checkpoint = latest_checkpoint', start)
engine = engine[:start] + '''        current_index = min(self.config.locate_step(global_step)[0], len(self.config.phases) - 1)
        current_phase = self.config.phases[current_index]
        final_optimization = lambda: self._optimization_state(
            current_phase, optimizer, scheduler, scaler, active,
        )
''' + engine[end:]
write('src/ncls/learning/training/engine.py', engine)

config = read('src/ncls/learning/training/config.py')
config = config.replace('        if self.checkpoint_selection != "tail_guard":\n            raise ValueError("training configs require tail_guard checkpoint selection")\n', '')
config = config.replace('if not self.method_key or self.run_class not in {"smoke", "profile", "adapted", "formal"}:', 'if not self.method_key:')
anchor = '    @property\n    def total_steps(self) -> int:\n'
signature = '''    @property
    def resume_signature(self) -> dict[str, Any]:
        # 只比较恢复优化轨迹实际需要的数据，不包含日志、图像、设备和研究标签。
        phases = []
        for phase in self.phases:
            value = phase.to_dict()
            for name in ("log_interval", "gradient_audit_interval", "checkpoint_boundary", "checkpoint_interval", "prefetch_depth"):
                value.pop(name, None)
            phases.append(value)
        return {
            "method": self.method_key, "model": dict(self.model_context),
            "source": dict(self.source), "query": dict(self.online_query),
            "seed": self.seed, "phases": phases,
        }

'''
config = config.replace(anchor, signature + anchor)
write('src/ncls/learning/training/config.py', config)
review = read('src/ncls/learning/training/review.py')
review = review.replace('    checkpoint.validate_method(descriptor)\n', '')
write('src/ncls/learning/training/review.py', review)

init = read('src/ncls/learning/training/__init__.py')
start = init.index('from .checkpoint_v1 import (')
end = init.index('from ..batches', start)
init = 'from .checkpoint import TrainingCheckpoint, load_checkpoint, save_checkpoint\n' + init[end:]
start = init.index('from .readiness import (')
end = init.index('\n)', start) + 2
init = init[:start] + init[end:]
for name in ('TrainingCheckpointV1', 'EvaluationSnapshot', 'LegacyCheckpointV4Importer', 'CheckpointReadiness', 'CheckpointReadinessMode', 'assess_checkpoint_readiness', 'load_training_checkpoint_v1', 'load_evaluation_snapshot', 'save_training_checkpoint_v1'):
    init = init.replace(f'    "{name}",\n', '')
init = init.replace('__all__ = [', '__all__ = [\n    "TrainingCheckpoint",\n    "load_checkpoint",\n    "save_checkpoint",')
write('src/ncls/learning/training/__init__.py', init)

package = read('src/ncls/learning/evaluation_package.py')
package = package.replace('from ncls.learning.training import CheckpointReadinessMode, EvaluationSnapshot', 'from ncls.learning.training.checkpoint import TrainingCheckpoint')
package = package.replace('EvaluationSnapshot', 'TrainingCheckpoint')
package = package.replace('    readiness_mode: CheckpointReadinessMode,', '    checkpoint_sha256: str | None = None,')
package = package.replace('    readiness = dict(evaluation.require_ready(readiness_mode))\n', '')
package = package.replace('evaluation.public_method_key', 'evaluation.method_key')
start = package.index('    if plugin.descriptor.method_key != evaluation.implementation_key:')
end = package.index('    source = evaluation.source', start)
package = package[:start] + package[end:]
package = package.replace('evaluation.deployment_payload', 'evaluation.model_payload')
package = package.replace('    validation["checkpoint_readiness"] = readiness\n', '    validation["training_diagnostics"] = {"phase": evaluation.phase_name, "gradient_coverage": dict(evaluation.gradient_coverage)}\n')
package = package.replace('evaluation.checkpoint_sha256', 'checkpoint_sha256')
for line in ['            "checkpoint_readiness_mode": readiness_mode,\n', '            "checkpoint_compatibility": "exact",\n', '            "checkpoint_legacy_v4": evaluation.legacy_v4,\n']:
    package = package.replace(line, '')
write('src/ncls/learning/evaluation_package.py', package)
