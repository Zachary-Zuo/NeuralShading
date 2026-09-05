from pathlib import Path
import re

root = Path.cwd()
def edit(path, function):
    file = root / path
    old = file.read_text(encoding='utf-8')
    new = function(old)
    if old != new:
        file.write_text(new, encoding='utf-8')

edit('AGENTS.md', lambda s: s.replace('`build/`、`artifacts/`、`reports/`', '`build/`、`outputs/`、`artifacts/`、`reports/`')
    .replace('单次正确性验证、实验报告与运行摘要统一进入 `artifacts/`；', '新训练成果统一进入 `outputs/<config-stem>/<run-id>/`，其中集中保存 checkpoint、TensorBoard、eval、导出与日志；`artifacts/` 只承载可清理研究产物和原地保留的旧 viewer 图像，不是正式功能的运行依赖；')
    .replace('- 需要导入 Falcor Python 模块时，Windows 统一通过 `scripts/run_falcor_python.ps1` 启动，Ubuntu 统一通过 `scripts/run_falcor_python.sh` 启动；两个脚本都设置锁定构建的 `PATH`/`PYTHONPATH` 后继续使用 `neural-shading` 环境。Ubuntu 只承载 headless Falcor/Vulkan reference 采集，不承载 Windows/D3D12 viewer。', '- Windows/Linux 统一使用 `python -m ncls train GPU_LIST --config YAML`；Linux 多卡自动 DDP，GPU 只指定一次。入口在导入 Torch/Falcor 前准备当前平台环境。其他需要 Falcor 的 Python 工具/测试通过 `python -m ncls.runtime --device N -- <python-args>` 启动。Linux 保留数值 validation，图像 eval 在相同接口绑定空实现；Windows 图像在本 run 的 TensorBoard 显示，reference 默认 128 spp，可只改 YAML。')
    .replace('- 新增长期依赖时同步更新', '- 新训练不读取旧 checkpoint，不保留 importer、转换工具或旧入口。旧视觉证据留在原 artifacts 位置，不迁移成果；训练身份与运行设置分离，诊断标签和完整配置/源码 hash 不作为恢复或导出门禁。\n- 新增长期依赖时同步更新'))
edit('CLAUDE.md', lambda s: s + '\n训练入口、outputs/artifacts 生命周期及统一图像接口以 AGENTS.md 和 docs/architecture.md 为准；不维护另一套平台启动或旧权重兼容规则。\n')
edit('docs/repository_policy.md', lambda s: s.replace('`build/`、`artifacts/`', '`build/`、`outputs/`、`artifacts/`').replace('训练run、checkpoint、ScatteringPackage、capture、benchmark与验证报告进入`artifacts/`。', '训练 run、checkpoint、TensorBoard、eval、导出和运行日志集中进入 `outputs/<config-stem>/<run-id>/`。`artifacts/` 保存可清理的临时研究、benchmark 与独立验证报告；旧 viewer PNG/EXR 按用户要求原地保留。本次不迁移或删除旧成果，新训练和默认部署不依赖 artifacts；用户以后可以自行清理。'))
edit('docs/migration_plan.md', lambda s: '# 架构重置状态\n\n2026-09-05 起采用统一 Python 入口、按 config/run 聚合的 outputs、直接 Method 接口和单一 checkpoint。旧 checkpoint importer、旧训练脚本、full Metal 及跨机视觉队列已删除。旧成果不迁移，旧 viewer 图像原地保留。当前使用说明见 [architecture.md](architecture.md) 和 [learning.md](learning.md)，执行合同见 `.trellis/spec/project/unified-pipeline.md`。\n')
edit('.trellis/spec/learning/index.md', lambda s: '# Learning 层\n\n当前公开方法为 `nvidia`、`metal`，真实模型/数据适配/编译位于各自方法目录。公共 `Method` 直接提供实现，`TrainingEngine`、在线 session、checkpoint 和图像接口共用；不保留旧 facet、状态转换或平台专用训练路径。\n\n开发前读 [统一 pipeline](../project/unified-pipeline.md)、[online 训练](online-training.md)、[方法与 package](pipeline-and-evaluation.md)；部署读 [deployment.md](deployment.md)，数据调度读 [online-pipeline.md](../data/online-pipeline.md)。质量检查运行相关 unit、当前模型 GPU 回归与真实短流程；按 TESTING.md 区分 Windows 已执行和 Linux 待实机验证。\n')
edit('.trellis/spec/data/online-pipeline.md', lambda s: s.replace('MethodDataFacet.', 'Method.').replace('devices: [0]\n', '').replace('YAML `execution` 必须精确包含：', 'YAML `execution` 配置队列与预算，GPU 列表来自命令：').replace('| resume 的 data plan/query/source identity 漂移 | 恢复拒绝，不装载 cursor |', '| resume 的逻辑 query/source/rank partition 不同 | 无法精确恢复，建立新 run；执行设置变化不阻止恢复 |').replace('`DataExecutionPlan.identity` 进入 checkpoint，rank partition 规则进入公共 identity，具体 rank 进入 session identity。', '`DataExecutionPlan.identity` 只作运行记录；逻辑 query 身份包含 source、route、seed 和 world size，不包含预取设置或物理 GPU 编号。'))

def package_spec(s):
    s = s.replace('MethodDefinition', 'Method').replace('method.compile_asset(checkpoint, source_snapshot)', 'method.compile_asset(source_snapshot, checkpoint)')
    lines = []
    for line in s.splitlines():
        if 'Metal `metal-fused-neural-material@1`' in line:
            line = '- 当前 Metal budgeted 方法按实际 shader 能力导出 prepare/evaluate/sample/pdf；普通导出不要求 formal、complete 或梯度覆盖，当前训练 step 和诊断写入 metadata。'
        elif '| Metal checkpoint未完成' in line:
            line = '| 初始化或短训 checkpoint | 正常导出当前模型，记录实际诊断，不增加 readiness 门禁 |'
        elif '- Good：Metal 120k' in line:
            line = '- Good：初始化模型与短训模型使用同一个 compiler，能生成与当前结构对应的 package。'
        elif 'formal/diagnostic readiness分别' in line:
            line = '- integration：当前 Metal checkpoint 依次检查量化 Python、Slang package 与 viewer，能力与研究标签分离。'
        lines.append(line)
    return '\n'.join(lines) + '\n'
edit('.trellis/spec/learning/pipeline-and-evaluation.md', package_spec)

def viewer_spec(s):
    lines = []
    for line in s.splitlines():
        if 'prepare_metal_catalog.py' in line:
            line = 'python -m ncls export <checkpoint> [--output <目录>]' if line.startswith('tools/') else '- 当前 checkpoint 的导出与 source 准备共用 `ncls export`，图像使用同一个 `prepare_source_reference`。'
        elif line.startswith('scripts/prepare_metal_viewer.ps1'):
            continue
        elif line.startswith('scripts/launch_metal_viewer.ps1'):
            line = 'scripts/launch_viewer.ps1 -Package <package> -Material <source/catalog>'
        elif '`ncls.metal-budgeted-viewer-handoff@2`' in line or 'handoff 的 `checkpoint_compatibility' in line:
            continue
        elif '| checkpoint 未完成' in line:
            line = '| tensor 名称/shape/dtype 不符 | 在模型加载边界拒绝；短训或初始化不阻止部署 |'
        elif 'test_training_checkpoint_new.py' in line:
            line = '- `test_training_checkpoint.py`：同一当前 reader、optimizer 状态保留和实际 tensor 结构检查。'
        line = line.replace('checkpoint 可为 null；', 'catalog 不携带 checkpoint/compatibility 字段；')
        line = line.replace('旧 catalog/handoff reader 已移除；', '旧 catalog/handoff reader 已移除；')
        lines.append(line)
    return '\n'.join(lines) + '\n'
edit('.trellis/spec/viewer/mdl-reference.md', viewer_spec)

def capture_spec(s):
    s = s.replace('1024', '128').replace('headless 用 64 次 PT dispatch', 'headless 用 8 次 PT dispatch')
    s = s.replace('项目正式视觉基线和 training diagnostic 的 reference 均固定为 128 spp；', '默认 reference 为 128 spp，实际采样数从 YAML/replay 获取，可任意调整合法正整数；')
    s = s.replace('  formal path-tracing slot spp = replay.reference_spp（正式基线为 128）', '  每个 path-tracing slot spp = slots[i].target_spp（reference 默认 128）')
    s = s.replace('  training-diagnostic slot 0 = reference path tracing 128 spp', '  training-diagnostic slot 0 = reference path tracing，spp 来自 YAML')
    s = s.replace('`comparison_purpose=formal` 时，所有 ready path-tracing slot 必须恰好累计到共同 `reference_spp`。`comparison_purpose=training-diagnostic` 时，', '')
    s = s.replace('| formal 中任一 ready PT slot `< reference_spp` | headless 继续累计，不提前导出 |', '| PT slot `< target_spp` | headless 继续累计，不提前导出 |')
    s = s.replace('- Good：formal 双 PT comparison 的两个 slot 都为 128 spp；共同 target 语义保持不变。', '- Good：YAML 将 reference 改为 33 spp，dispatch 16+16+1；双方 PT 的 target 独立，不限制大小关系。')
    s = '\n'.join(line for line in s.splitlines() if 'test_visual_eval_worker.py' not in line) + '\n'
    s = s.replace('断言 formal matched target、training-diagnostic per-slot target、default reference 128', '检查 per-slot target 的透传与独立调度')
    return s
edit('.trellis/spec/viewer/capture-harness.md', capture_spec)

def viewer_readme(s):
    start = s.index('查看同一固定MDL source')
    end = s.index('\n## ', start)
    return s[:start] + '''新模型统一从 checkpoint 导出；输出路径由 run 管理。导出命令会打印 package 与 source/catalog 路径：

```powershell
conda run -n neural-shading python -m ncls export outputs/<config>/<run>/checkpoints/latest.pt
.\\scripts\\launch_viewer.ps1 -Package <package目录> -Material <source文件或catalog>
```

两侧模式用 `-ReferenceMode` / `-NeuralMode` 独立选择，默认 reference PT 与 neural deferred。训练中图像由公共 eval hook 调用同一 viewer，reference 默认 128 spp，实际值只由 YAML 决定。旧视觉证据留在 artifacts 原位置，历史 checkpoint handoff 已删除。
''' + s[end:]
edit('apps/viewer/README.md', viewer_readme)

# 稳定说明中的旧包装脚本与模块路径同步到当前 Python 入口。
for folder in ('docs', 'references', 'tools', '.trellis/spec'):
    for path in (root / folder).rglob('*.md'):
        if folder == 'docs' and 'research' in path.parts:
            continue
        edit(path.relative_to(root), lambda s: s.replace('.\\scripts\\run_falcor_python.ps1 -m ncls.cli', 'conda run -n neural-shading python -m ncls')
             .replace('.\\scripts\\run_falcor_python.ps1 -m ncls', 'conda run -n neural-shading python -m ncls')
             .replace('.\\scripts\\run_falcor_python.ps1', 'conda run -n neural-shading python -m ncls.runtime --')
             .replace('MethodDataFacet.', 'Method.'))

for name in ('docs/research/experiment_log.md', 'docs/research/p1_audit.md', 'docs/research/p1_v2_plan.md'):
    edit(name, lambda s: s.split('\n', 1)[0] + '\n\n> 历史记录：2026-09-05 架构重置前的命令、格式和结果只用于理解当时实验，当前接口见 docs/learning.md。旧权重不迁移、不兼容读取，旧 viewer 图像原地保留。\n' + s.split('\n', 1)[1])
