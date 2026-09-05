from pathlib import Path

root = Path(__file__).resolve().parents[4]
replacements = {
    'MethodDefinition': 'Method', 'MethodPlugin': 'Method',
    'METHOD_DEFINITION': 'METHOD', 'METHOD_PLUGIN': 'METHOD',
    '.model_factory.create(': '.create_trainable(',
    '.data.requirements(': '.requirements(', '.data.create_source_adapter(': '.create_source_adapter(',
    '.objective.compute(': '.training_objective(', '.lifecycle.validate_training_plan(': '.validate_training_config(',
    '.lifecycle.initialization_requests(': '.initialization_requests(',
    '.lifecycle.initialize_training_state(': '.initialize_training_state(',
    '.lifecycle.configure_phase(': '.configure_phase(', '.lifecycle.parameter_registry(': '.parameter_registry(',
    '.lifecycle.apply_transition(': '.apply_phase_transition(', '.checkpoint.encode(': '.export_training_state(',
    '.checkpoint.restore(': '.restore_training_state(', '.deployment.': '.',
    'tests.fixtures.method_definition': 'tests.fixtures.method',
}
for path in (root / 'tests').rglob('*.py'):
    value = path.read_text(encoding='utf-8')
    for old, new in replacements.items():
        value = value.replace(old, new)
    path.write_text(value, encoding='utf-8', newline='\n')

path = root / 'tests/unit/test_training_runner_phase_graph.py'
value = path.read_text(encoding='utf-8')
value = value.replace('class _PhaseMethod(Method):', '''class _PhaseMethod(Method):
    key = "phase-fixture"

    def create_source_adapter(self, snapshots, device):
        raise AssertionError("fixture uses an explicit producer")
''')
start = value.index('def _plugin(')
end = value.index('\ndef ', start + 1)
value = value[:start] + 'def _plugin(definition: Method) -> Method:\n    return definition\n\n' + value[end:]
path.write_text(value, encoding='utf-8', newline='\n')
path = root / 'tests/fixtures/method_definition.py'
value = path.read_text(encoding='utf-8').replace('class ContractFixtureMethod(Method):', '''class ContractFixtureMethod(Method):
    key = "contract-fixture"

    def create_source_adapter(self, snapshots, device):
        raise AssertionError("fixture uses an explicit producer")
''')
(root / 'tests/fixtures/method.py').write_text(value, encoding='utf-8', newline='\n')
(root / 'tests/unit/test_methods.py').write_text((root / 'tests/unit/test_method_definition.py').read_text(encoding='utf-8'), encoding='utf-8', newline='\n')

# viewer 工具保留当前消费能力，统一使用 checkpoint，不复制权重到部署包。
path = root / 'tools/viewer/prepare_metal_catalog.py'
value = path.read_text(encoding='utf-8')
value = value.replace('from ncls.learning.training import EvaluationSnapshot\nfrom ncls.learning.deployment_snapshot import load_deployment_snapshot', 'from ncls.learning.training.checkpoint import TrainingCheckpoint, load_checkpoint')
value = value.replace('EvaluationSnapshot', 'TrainingCheckpoint').replace('load_deployment_snapshot(', 'load_checkpoint(').replace('snapshot.public_method_key', 'snapshot.method_key').replace('snapshot.deployment_payload', 'snapshot.model_payload')
start = value.index('def validate_deployment_checkpoint(')
end = value.index('def _portable(', start)
value = value[:start] + value[end:]
value = value.replace('        validate_deployment_checkpoint(snapshot)\n','')
value = value.replace('                readiness_mode="diagnostic-evaluator",\n','')
start = value.index('        checkpoints_root = staging / "checkpoints"')
end = value.index('        reference_catalog = ', start)
value = value[:start] + value[end:]
value = value.replace('        "checkpoint_path": f"checkpoints/{role}.pt",\n', '').replace('        "checkpoint_sha256": evaluation.checkpoint_sha256,\n', '')
value = value.replace('PROJECT_ROOT / "artifacts/viewer/metal-budgeted-pair"', 'None')
value = value.replace('    output_root: Path,\n    hybrid_checkpoint:', '    output_root: Path | None,\n    hybrid_checkpoint:')
value = value.replace('    output_root = output_root.resolve()\n', '    from ncls.runs import RunPaths\n    if output_root is None:\n        output_root = RunPaths.from_checkpoint(hybrid_checkpoint).exports / "viewer"\n    output_root = output_root.resolve()\n')
path.write_text(value, encoding='utf-8', newline='\n')
