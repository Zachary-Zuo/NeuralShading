from pathlib import Path

root = Path(__file__).resolve().parents[4]

def read(name):
    return (root / name).read_text(encoding="utf-8")

def write(name, value):
    (root / name).write_text(value, encoding="utf-8", newline="\n")

# 命令解析不导入训练模块，使 --help 和外层 launcher 保持轻量。
cli = read("src/ncls/cli.py")
parser = cli[cli.index("def build_parser()"):cli.index("def main(")]
start = parser.index('    train_yaml.add_argument("config"')
end = parser.index('    train_yaml.add_argument("--resume"', start)
parser = parser[:start] + '''    train_yaml.add_argument("devices", type=parse_devices, help="物理 GPU 编号，例如 0 或 0,1")
    train_yaml.add_argument("--config", type=Path, required=True)
''' + parser[end:]
start = parser.index('    visual_eval = commands.add_parser(')
end = parser.index('    package = commands.add_parser(', start)
parser = parser[:start] + '''    visual_eval = commands.add_parser("eval", help="使用 checkpoint 的图像设置评估当前模型")
    visual_eval.add_argument("checkpoint", type=Path)
    visual_eval.add_argument("--config", type=Path, help="可覆盖图像设置的 YAML")
    visual_eval.add_argument("--device", type=int, default=0)

''' + parser[end:]
parser = parser.replace('从新 checkpoint 或只读 legacy v4 导出正式 ScatteringPackage', '从当前 checkpoint 导出 ScatteringPackage')
parser = parser.replace('export_checkpoint.add_argument("output", type=Path)', 'export_checkpoint.add_argument("--output", type=Path, help="默认写入 run/exports/step-N/")')
write("src/ncls/commands.py", 'from __future__ import annotations\n\nimport argparse\nfrom pathlib import Path\nfrom .runtime import parse_devices\n\n\n' + parser)
cli = cli[:cli.index("def build_parser()")] + cli[cli.index("def main("):]
cli = cli.replace('import argparse\n', '').replace('import torch\n', 'import torch\n\nfrom ncls.commands import build_parser\n')
write("src/ncls/cli.py", cli)

# 方法直接实现生命周期和 objective，不再包装同一 definition 六次。
method = read("src/ncls/learning/method.py").replace('class MethodDefinition(ABC):', 'class Method(ABC):\n    key: str')
insert = '''    def requirements(self):
        from ncls.data import DataRequirement

        return tuple(DataRequirement(kind, tuple(fields)) for kind, fields in self.descriptor.training_batch_requirements.items())

    @abstractmethod
    def create_source_adapter(self, snapshots, device):
        raise NotImplementedError

'''
method = method.replace('class Method(ABC):\n    key: str\n    descriptor: MethodDescriptor\n', 'class Method(ABC):\n    key: str\n    descriptor: MethodDescriptor\n\n' + insert)
write("src/ncls/learning/method.py", method)
for filename, old_class, new_class, key in [
    ("nvidia", "NvidiaMethodDefinition", "NvidiaMethod", "nvidia"),
    ("metal_budgeted", "MetalBudgetedMethodDefinition", "MetalBudgetedMethod", "metal"),
]:
    name = f"src/ncls/learning/methods/{filename}.py"
    value = read(name).replace('MethodDefinition', 'Method').replace(old_class, new_class)
    value = value.replace(f'class {new_class}(Method):', f'class {new_class}(Method):\n    key = "{key}"\n\n    def create_source_adapter(self, snapshots, device):\n        return _create_source_adapter(snapshots, device)\n')
    value = value.replace(f'METHOD_DEFINITION = {new_class}()', '')
    start = value.index('METHOD_PLUGIN = MethodPlugin.adapt_definition(')
    end = value.index('\n)', start) + 2
    value = value[:start] + f'METHOD = {new_class}()' + value[end:]
    value = value.replace('from .contracts import MethodPlugin\n', '').replace('"METHOD_DEFINITION",', '"METHOD",').replace('"METHOD_PLUGIN",', '')
    write(name, value)

replacements = {
    'plugin.model_factory.create(': 'plugin.create_trainable(',
    'plugin.data.requirements(': 'plugin.requirements(',
    'plugin.data.create_source_adapter(': 'plugin.create_source_adapter(',
    'plugin.objective.compute(': 'plugin.training_objective(',
    'plugin.lifecycle.validate_training_plan(': 'plugin.validate_training_config(',
    'plugin.lifecycle.initialization_requests(': 'plugin.initialization_requests(',
    'plugin.lifecycle.initialize_training_state(': 'plugin.initialize_training_state(',
    'plugin.lifecycle.configure_phase(': 'plugin.configure_phase(',
    'plugin.lifecycle.parameter_registry(': 'plugin.parameter_registry(',
    'plugin.lifecycle.apply_transition(': 'plugin.apply_phase_transition(',
    'plugin.checkpoint.encode(': 'plugin.export_training_state(',
    'plugin.checkpoint.restore(': 'plugin.restore_training_state(',
    'plugin.deployment.': 'plugin.',
    'self.plugin.objective,': 'self.plugin,',
    'from ncls.learning.methods.contracts import MethodPlugin': 'from ncls.learning.method import Method',
    'from .contracts import MethodPlugin': 'from ncls.learning.method import Method',
    'MethodPlugin': 'Method',
    'MethodDefinition': 'Method',
}
for path in (root / "src/ncls").rglob("*.py"):
    if path.name == "contracts.py" and path.parent.name == "methods":
        continue
    value = path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        value = value.replace(old, new)
    if path.name == "registry.py" and path.parent.name == "methods":
        value = value.replace('"METHOD_PLUGIN"', '"METHOD"').replace('.METHOD_PLUGIN', '.METHOD')
    if path.name == "__init__.py" and path.parent.name == "methods":
        value = value.replace('from .contracts import (', 'from ncls.learning.method import (')
    if value != path.read_text(encoding="utf-8"):
        path.write_text(value, encoding="utf-8", newline="\n")

for name in ['prd.md', 'design.md', 'implement.md']:
    path = root / '.trellis/tasks/09-05-architecture-reset-training-workflow' / name
    value = path.read_text(encoding='utf-8')
    value = value.replace('状态：planning，需求已收敛，等待最终规划审阅。', '状态：in_progress，用户已于 2026-09-05 审阅最终规划后明确要求“开始执行”。')
    value = value.replace('状态：planning，需求已收敛，等待最终规划审阅；本文件不是实施授权。', '状态：in_progress，用户已于 2026-09-05 审阅最终规划后明确要求“开始执行”。')
    value = value.replace('尚未批准实施。', '已批准实施。').replace('等待最终规划审阅；尚未批准实施。', '已获实施批准。')
    value = value.replace('- [ ] 向用户呈现最新完整规划，取得后续明确实施批准后才运行 task.py start。', '- [x] 已呈现最终规划，用户随后回复“开始执行”；已运行 task.py start。')
    path.write_text(value, encoding='utf-8', newline='\n')
