from pathlib import Path
import ast
import yaml

root = Path.cwd()
launch = root / 'src/ncls/learning/training/launch.py'
value = launch.read_text(encoding='utf-8')
value = value[:value.index('\ndef prepare_process_environment(')].rstrip() + '\n'
for item in ('from pathlib import Path\n', 'import subprocess\n', 'import sys\n', 'from .distributed import configure_distributed_debug_environment\n'):
    value = value.replace(item, '')
launch.write_text(value, encoding='utf-8')
for path in (root / 'src/ncls/cli.py', root / 'src/ncls/learning/training/__init__.py'):
    value = path.read_text(encoding='utf-8').replace('    prepare_process_environment,\n', '').replace('    "prepare_process_environment",\n', '').replace('    prepare_process_environment(topology)\n', '')
    path.write_text(value, encoding='utf-8')

for path in (root / 'configs/training').rglob('*.yaml'):
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    execution = value.get('execution', {})
    if 'devices' in execution:
        del execution['devices']
        path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding='utf-8')

path = root / 'src/ncls/cli.py'
value = path.read_text(encoding='utf-8')
value = value.replace('from ncls.visual_eval.evaluator import VisualContext', 'from ncls.visual_eval.evaluator import NoVisualEvaluation, VisualContext')
value = value.replace('    checkpoint = load_checkpoint(checkpoint_path)\n    plan = ResolvedTrainingPlan.from_dict(checkpoint.resolved_plan)', '''    from ncls.learning.training.plan import VisualEvalSettings

    settings = TrainingPlanResolver(PROJECT_ROOT).resolve(config_path).hooks.visual_eval if config_path else VisualEvalSettings()
    evaluator = create_visual_evaluator(settings)
    if isinstance(evaluator, NoVisualEvaluation):
        return 0
    checkpoint = load_checkpoint(checkpoint_path)
    plan = ResolvedTrainingPlan.from_dict(checkpoint.resolved_plan)''')
value = value.replace('result = create_visual_evaluator(context.settings).evaluate(model, context)', 'result = evaluator.evaluate(model, context)')
path.write_text(value, encoding='utf-8')
