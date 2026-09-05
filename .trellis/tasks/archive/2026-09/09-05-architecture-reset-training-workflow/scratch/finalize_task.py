from pathlib import Path
import json

task = Path(__file__).resolve().parents[1]

def edit(name, transform):
    path = task / name
    value = transform(path.read_text(encoding='utf-8'))
    path.write_bytes(value.encode('utf-8'))

edit('prd.md', lambda text: text.replace(
    '本文件已按本轮全部范围修正收敛，等待最终规划审阅；已批准实施。',
    '随后明确要求“开始执行”。代码和本机验证已完成，Linux 实机项待目标机执行；证据见 research/validation.md。'
).replace('## 已确认事实', '## 实施前盘点（以下位置与行号指改造前）')
    .replace('- [ ] AC1', '- [x] AC1').replace('- [ ] AC3', '- [x] AC3')
    .replace('- [ ] AC4', '- [x] AC4').replace('- [ ] AC6', '- [x] AC6')
    .replace('- [ ] AC7', '- [x] AC7').replace('- [ ] AC8', '- [x] AC8')
    .replace('## 范围之外', 'AC2、AC5 的实现与公共行为测试已完成；保留未勾选状态以显式交接 Linux 实机验证，不能以 Windows 结果代替。\n\n## 范围之外'))

edit('implement.md', lambda text: text.replace(
    '> 状态：in_progress，用户已于 2026-09-05 审阅最终规划后明确要求“开始执行”。',
    '> 状态：review，代码与本机验证完成；Linux 实机检查见 TESTING.md，验收证据见 research/validation.md。'
).replace('- [ ]', '- [x]').replace(
    '规划期只做源码/文件盘点；以下是实施后执行的检查，不能当作本轮已通过。',
    '下列检查方案已按本机能力执行，具体通过结果和 Linux 待执行项见 research/validation.md。'
).replace('示例命令（新入口实现后）：', '当前入口命令：'))

edit('design.md', lambda text: text.replace(
    '> 状态：in_progress，用户已于 2026-09-05 审阅最终规划后明确要求“开始执行”。',
    '> 状态：review，用户已批准实施，当前代码与本机验证完成。'
).replace('完成最终规划审阅并得到用户实施批准后，开始产品实现。',
    '用户已批准并完成本次实现，本机验证与 Linux 交接见 research/validation.md。'))

edit('research/architecture-audit.md', lambda text: text.replace(
    '## 2. 已确认的结构性问题', '## 2. 实施前确认的结构性问题（路径和行号为当时状态）'
).replace('## 3. 历史方法与共享实现', '## 3. 实施前的历史方法与共享实现')
    .replace('## 6. 规划状态', '## 6. 实施状态').replace(
    'prd/design/implement 已同步收敛，等待最终规划审阅。尚未开始产品实施，也没有移动、复制或删除旧成果。',
    '代码、配置和文档已切换到新架构，检查结果见 validation.md。当前方法的必要 helper 已并入自身目录；旧 proposal 链只有退役方法/测试消费者，已整体删除。旧成果没有移动、复制或删除。'
))

metadata = json.loads((task / 'task.json').read_text(encoding='utf-8'))
metadata['status'] = 'review'
metadata['notes'] = '用户已批准并执行。新架构代码与 Windows 验证完成：314 unit、19 GPU、Release viewer、新训练/续训、128/33 spp、TensorBoard 与初始化 MDL 导出/出图。Linux 单卡/NCCL 实机交接见 TESTING.md；AC2/AC5 保留待实机标记。旧视觉证据原地保留；未迁移旧成果、未提交或归档。完整证据见 research/validation.md。'
(task / 'task.json').write_bytes((json.dumps(metadata, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
print('任务进入 review；Linux 实机验证单独标记。')
