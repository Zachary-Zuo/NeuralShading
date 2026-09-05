from pathlib import Path
import json
import subprocess

task = Path(__file__).resolve().parents[1]
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
metadata = json.loads((task / 'task.json').read_text(encoding='utf-8'))
metadata['commit'] = commit
metadata['notes'] = (
    f'用户明确要求提交并归档，代码提交 {commit[:7]}。'
    'Windows 已通过 314 unit、19 GPU、Release viewer、训练/续训、128/33 spp 与初始化 MDL 导出/出图。'
    'Linux 单卡/NCCL 实机检查继续交接至 TESTING.md；AC2/AC5 未冒充实机通过。'
    '旧视觉证据原地保留，未迁移旧成果。完整证据见 research/validation.md。'
)
(task / 'task.json').write_bytes((json.dumps(metadata, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
for name in ('design.md', 'implement.md'):
    path = task / name
    text = path.read_text(encoding='utf-8').replace('状态：review', '状态：已交付，用户授权提交并归档')
    path.write_bytes(text.encode('utf-8'))
path = task / 'research/validation.md'
text = path.read_text(encoding='utf-8').replace('进入 review。', '经用户确认提交并归档。')
text = text.replace('本次修改保留在工作区供审阅，未执行 Git 提交或任务归档。',
    f'用户随后明确要求提交并归档，代码提交为 `{commit[:7]}`；本任务按 Trellis 工作流归档。Linux 实机检查仍以待执行项保留。')
path.write_bytes(text.encode('utf-8'))
print('已记录代码提交：', commit)
