"""顺序执行当前入口的最终 smoke，不读取旧权重或改写历史证据。"""
from pathlib import Path
import ast
import json
import subprocess
import sys

scratch = Path(__file__).resolve().parent
root = scratch.parents[3]
run = root / 'outputs/visual-33/260905-181003-ed3da4'

def command(name, *arguments):
    print(name, flush=True)
    with (scratch / (name + '.txt')).open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print((scratch / (name + '.txt')).read_text(encoding='utf-8')[-12000:], flush=True)
        raise SystemExit(result.returncode)

command('final-validate', '-m', 'ncls', 'validate', run / 'checkpoints/latest.pt', '--batches', '1', '--device', '0')
command('final-eval-override', '-m', 'ncls', 'eval', run / 'checkpoints/latest.pt', '--config', scratch / 'visual-33.yaml')
command('final-check-33', scratch / 'check_visual.py', run)
command('final-check-128', scratch / 'check_visual.py', root / 'outputs/nvidia-layer-stack-smoke/260905-172009-9d5b6c')

config = root / 'configs/training/runs/nvidia-mdl-effect-pigment-smoke.yaml'
directory = root / 'outputs' / config.stem
before = set(directory.glob('*'))
command('final-mdl-init', '-m', 'ncls', 'train', '0', '--config', config, '--stop-at-step', '0')
mdl_run, = set(directory.glob('*')) - before
(scratch / 'final-mdl-run.txt').write_text(str(mdl_run), encoding='utf-8')
command('final-mdl-export', '-m', 'ncls', 'export', mdl_run / 'checkpoints/latest.pt')
command('final-mdl-eval', '-m', 'ncls', 'eval', mdl_run / 'checkpoints/latest.pt', '--config', scratch / 'visual-33.yaml')
command('final-check-mdl', scratch / 'check_visual.py', mdl_run)

inventory = json.loads((scratch / 'visual-evidence-inventory.json').read_text(encoding='utf-8'))
images = {item['path'] for capture in inventory['captures'] for item in capture['images']}
assert len(images) == inventory['image_count']
assert all((root / capture['capture']).is_file() for capture in inventory['captures'])
assert all((root / path).is_file() for path in images | set(inventory['unassociated_images']))
size = sum((root / path).stat().st_size for path in images)
assert size == inventory['image_bytes']
print('旧视觉证据原路径存在：', len(images), 'PNG/EXR；', size, 'bytes，与规划盘点相同。', flush=True)

count = 0
for folder in ('src', 'tests', 'tools'):
    for path in (root / folder).rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        count += 1
print('Python AST:', count, 'files', flush=True)
print('最终短流程检查完成', flush=True)
