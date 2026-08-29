# 规划复核与连续执行授权

## 复核结论

2026-08-30 已完成父任务 `prd.md`、`design.md`、`implement.md` 与六个当前 child 的规划复核。用户在看到最终规划摘要后明确批准进入实现，并进一步要求：

- 创建长期 goal，按依赖顺序完成全部子任务后再统一报告；
- 每个 child 通过质量门后及时创建 scoped local commit 并归档；
- full method 不得以粗糙、缩减、disabled、mock、placeholder 或静默遗漏的实现代替；
- 验证梯度下降正确性时，在保持 full shape、真实 online reference、loss、optimizer 与 phase data flow 的前提下，优先使用覆盖全部 required components/groups 和关键 source 语义的最小 stratified 子集提高效率。

上述细化没有改变已冻结的方法、产品、兼容性或交付边界。`08-30-metal-formal-evaluation` 与 `08-30-metal-compact-ablation` 仍保持解除状态，不属于连续执行树。

## 连续执行元数据

父任务与六个参与 child 的 `task.json.meta` 统一记录：

- `execution_mode=continuous`
- `continuous_authorized=true`
- `authorization_parent=08-30-vmaterial-metal-neural-system`
- `commit_policy=preauthorized-scoped-local-no-push`

因此每个 child 在规划与质量门闭合后可以直接启动、创建只包含已识别范围的本地提交并归档，不再逐项等待确认；不 amend、不 push、不纳入无关 dirty files。范围扩张、破坏性动作、新外部权限或真实 blocker 仍必须暂停。

## 静态检查

- 父任务和当前/延后 child 均具备 `prd.md`、`design.md`、`implement.md` 与合法 `task.json`；
- acceptance criteria 均带类型与来源；
- Markdown fence、列表、trailing whitespace 与 JSON 解析检查通过；
- `git diff --check` 通过；
- 旧 `TrainingConfig/Checkpoint@3`、`ScatteringPackage@1` 与 `NativeFeaturePyramid` 只在缺口、删除合同和 denylist 中出现；
- inline 模式按工作流跳过 `implement.jsonl` / `check.jsonl` curation。

下一执行目标是 `08-30-metal-canonical-architecture`。

## Scoped planning commit

提交计划：以一个 `docs(task)` commit 版本化父任务、六个参与 child 和两个已解除 future task 的规划工件。明确排除既有 `.trellis/config.yaml` 改动、旧任务 `scratch/`、根目录论文/PDF/图片/字体与 `tmp/`；后续 child commit 同样不得纳入这些路径。
