# 06 旧方法彻底归零

## Goal

在新方法、runtime 和 viewer 全部验收后，删除旧方法身份、错误模型、专属旁路、失效数据入口和过时稳定文档，使生产可达路径只剩最终 neural method 与显式 analytic control。

## Scope And Dependencies

- 前置任务：`01` 至 `05` 已完成、提交并归档，替代路径能够从 source offline cook 到 viewer deferred/PT 重建。
- 本任务不得提前删除仍被前序迁移使用的数学或数据来源。
- 本任务是复杂任务；启动前以当时全仓 reachability 扫描结果补全并审阅三件套与 context manifests。

## Requirements

- 删除 Film M1 pipeline/model/exporter/backend/viewer 硬编码、配置和专属测试。
- 删除未完成的 `lobe_residual` 方法身份、注册、配置和 TODO；正确原语只能保留在 `01` 的公共组件中。
- 删除 `legacy_ltc_k2` identity；只保留新命名 analytic control 和通用数学模块。
- 删除或改写过时稳定文档、spec、命令、schema 字符串、fallback 和 dead code。
- 对 tracked 内容直接完成可审计删除；对未版本化大数据给用户精确目录、重建来源、删除时机和可恢复性说明。
- 保留合法 v5/`02` 冻结的新 corpus、source assets、reference registry 和锁定 external clones。

## Acceptance Criteria

- [ ] 生产入口只保留最终 neural realtime method 与 analytic control；diagnostic/未胜出候选只保留实验 provenance。
- [ ] 全仓旧 ID/reachability 扫描只剩明确允许的历史复现文本，不存在静默 fallback。
- [ ] 全量 unit/GPU/viewer/capture 回归通过。
- [ ] repository policy、reference registry、corpus manifest、MethodBundle provenance 和稳定 spec 一致。
- [ ] 从干净 runtime assets 可重新执行 offline cook、bundle export 和两个 viewer 模式。
- [ ] 子任务完成质量检查、提交并归档后，父任务执行最终跨层验收；父任务验收完成后再提交并归档。

## Out Of Scope

- 删除合法 reference/source assets、修改锁定 external clones 或以清理名义改变最终方法语义。
