"""列出本任务提交范围，排除其他任务、运行产物和用户文件。"""
from pathlib import Path
import json
import subprocess

root = Path.cwd()
task = root / '.trellis/tasks/09-05-architecture-reset-training-workflow'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=root).decode('utf-8')

if git('diff', '--cached', '--name-only').strip():
    raise SystemExit('暂存区已有内容，需先确认其归属。')
tracked = git('diff', '--name-only', '-z').split('\0')[:-1]
if any(path.startswith(('.trellis/tasks/', '.trellis/workspace/')) for path in tracked):
    raise SystemExit('受管任务记录发生额外变更，不能混入代码提交。')
new_roots = [
    'scripts/launch_viewer.ps1',
    'src/ncls/commands.py', 'src/ncls/launcher.py', 'src/ncls/runs.py', 'src/ncls/runtime.py',
    'src/ncls/learning/methods/metal/', 'src/ncls/learning/methods/nvidia/',
    'src/ncls/viewer/export.py', 'src/ncls/visual_eval/evaluator.py', 'src/ncls/visual_eval/windows.py',
    'tests/fixtures/method.py', 'tests/unit/test_methods.py',
]
new = git('ls-files', '--others', '--exclude-standard', '-z', '--', *new_roots).split('\0')[:-1]
selected = sorted(set(tracked + new))
all_untracked = git('ls-files', '--others', '--exclude-standard', '-z').split('\0')[:-1]
excluded = [path for path in all_untracked
            if path not in selected and not path.startswith('.trellis/tasks/09-05-architecture-reset-training-workflow/')]
plan = {
    'authorization': '用户明确要求提交并归档；仅本任务本地提交，不 push。',
    'base_commit': git('rev-parse', 'HEAD').strip(),
    'work_commit': {'message': 'refactor(training): 重置项目架构并统一跨平台训练入口', 'files': selected},
    'bookkeeping': [
        'chore(task): archive 09-05-architecture-reset-training-workflow',
        'chore: record journal',
    ],
    'excluded_untracked': excluded,
}
(task / 'research/commit-plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
pathspec = task / 'scratch/work-commit-paths.nul'
pathspec.write_bytes(b'\0'.join(path.encode('utf-8') for path in selected) + b'\0')
summary = {'tracked': len(tracked), 'new': len(new), 'selected': len(selected), 'excluded_untracked': len(excluded)}
print(json.dumps(summary, ensure_ascii=False))
print('提交文件清单：', task / 'research/commit-plan.json')
