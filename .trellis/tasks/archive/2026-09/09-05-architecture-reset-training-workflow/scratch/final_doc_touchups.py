from pathlib import Path
import re
import yaml

root = Path.cwd()
def edit(name, function):
    path = root / name
    value = path.read_text(encoding='utf-8')
    result = function(value)
    if result != value:
        path.write_text(result, encoding='utf-8')

edit('apps/viewer/README.md', lambda s: s.replace('headless capture 从 replay 的 `reference_spp` 读取 reference 目标，正式基线使用 1024 spp；', 'headless capture 从 replay 的各 slot `target_spp` 读取采样目标，reference 默认 128 spp，可只改 YAML；')
    .replace('`comparison_purpose=formal` 要求所有 path-tracing slot 使用 matched spp；`training-diagnostic` 允许 1024 spp source reference 配合同条件的 neural deferred，或显式有界的低 spp neural path tracing，并在每个 slot 分别记录 `mode/target_spp/spp`。', '双方 PT 的预算互相独立；training diagnostic 默认 reference PT 与 neural deferred，在每个 slot 分别记录 `mode/target_spp/spp`。研究对照是否 matched 由实验设计解释，不是 renderer 的固定数字门禁。')
    .replace('"reference_spp": 1024', '"reference_spp": 128')
    .replace('{"package_id": "source-reference", "mode": "path-tracing"}', '{"package_id": "source-reference", "mode": "path-tracing", "target_spp": 128}')
    .replace('{"package_id": "<package-id>", "mode": "path-tracing"}', '{"package_id": "<package-id>", "mode": "deferred", "target_spp": 0}')
    .replace('-BundleRoot artifacts\\exports', '-BundleRoot outputs\\<config>\\<run>\\exports'))
edit('docs/viewer_spec.md', lambda s: s.replace('对所有 ready 的 path-tracing slot 固定累计到 1024 spp 后才导出', '对每个 ready 的 path-tracing slot 累计到 YAML/replay 指定的 target_spp 后才导出')
    .replace('正式基线使用 1024 spp，显式 headless target 可用于 smoke', 'reference 默认 128 spp，合法采样数可直接调整'))
edit('.trellis/spec/viewer/conventions.md', lambda s: s.replace('不把 1024 伪装成当前值', '不把配置 target 伪装成当前值').replace('交互到 1024 spp 后停止', '交互到 headless target 后停止').replace('做 1024 spp capture', '按当前 YAML/replay 预算做 capture'))
edit('.trellis/spec/viewer/index.md', lambda s: s.replace('replay 的固定 spp', 'replay 的独立 per-slot spp'))
for folder in ('docs', '.trellis/spec'):
    for path in (root / folder).rglob('*.md'):
        if folder == 'docs' and 'research' in path.parts:
            continue
        edit(path.relative_to(root), lambda s: s.replace('TrainingCheckpoint@1', 'TrainingCheckpoint').replace('MethodDefinition', 'Method').replace('data facet', 'Method 数据接口'))

def guide(s):
    start, end = s.index('## Learned Checkpoint Capability Boundary'), s.index('## Distributed Initialization / Calibration Boundary')
    replacement = '''## 模型状态、执行设置与部署

- [ ] 当前 tensor/资源是否可执行，是否在加载边界验证实际 shape/dtype？
- [ ] 模型/optimizer/query 身份是否与 GPU 编号、日志、图像和预取设置分开？
- [ ] 当前状态是否可以直接预览/导出，训练阶段和 coverage 只作诊断？
- [ ] profile 改变时，模型与 compiler/资源 ABI 是否对应实际配置，而非默认模型的固定 shape？
- [ ] 新训练是否写入 config/run 下的 outputs，旧图像是否保持原位置？
- [ ] 是否有逐层重复解析、源码 hash gate、旧 reader 或只剩测试消费者的历史实现需要删除？

具体合同见 `../learning/online-training.md` 和 `../learning/deployment.md`。

'''
    s = s[:start] + replacement + s[end:]
    return s.replace('- [ ] 旧schedule语义是否通过版本保留，而不是在同一recipe identity下静默改变已有checkpoint的query stream？', '- [ ] query recipe 改变后是否明确新建实验，而不添加旧 schedule 的兼容分支？')
edit('.trellis/spec/guides/cross-layer-thinking-guide.md', guide)

scratch = root / '.trellis/tasks/09-05-architecture-reset-training-workflow/scratch'
config = yaml.safe_load((scratch / 'visual-33.yaml').read_text(encoding='utf-8'))
config['hooks']['visual_eval']['reference_spp'] = 77
config.setdefault('execution', {}).update(host_prefetch=3, ready_batches=3)
(scratch / 'resume-settings.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

# 本任务改写的受管文本统一 LF；不处理用户未跟踪论文/字体或旧成果。
import subprocess
changed = subprocess.check_output(['git', '-c', 'core.quotePath=false', 'diff', '--name-only'], text=True, encoding='utf-8')
for name in changed.splitlines():
    path = root / name
    if path.is_file() and path.suffix in {'.py', '.md', '.yaml', '.toml', '.json', '.cpp', '.h', '.slang', '.ps1', '.sh'}:
        value = path.read_text(encoding='utf-8')
        path.write_bytes(value.rstrip().encode('utf-8') + b'\n')
