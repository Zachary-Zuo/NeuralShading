# 04 通用 MethodBundle Runtime

## Goal

把 `03` 冻结的方法通过 backend-agnostic Slang specialization 导出为 MethodBundle，并建立无 method-ID 分支的通用 exporter、loader、ABI reflection 与 parity gate。

## Scope And Dependencies

- 前置任务：`01`、`02`、`03` 已完成、提交并归档。
- 本任务消费 `03` 的最终方法 identity/layout；不得重新选择模型或 sampler。
- 本任务是复杂任务；启动前根据最终 Slang module、state 和 asset layout 补全并审阅三件套与 context manifests。

## Requirements

- manifest/schema 完整声明 module、concrete backend type、contract version、resource layout、state stride、entry capability、runtime class、cost 和内容 hash。
- exporter 从 Slang reflection 和冻结 layout 生成资产，不保留手写 Film 权重偏移。
- loader 按 schema/hash/platform/contract/layout/cost/capability 顺序校验，再创建通用 specialization。
- `prepare/evaluate/sample/pdf` 只通过 `INclsScatteringBackend` adapter；loader/shader 不按 backend ID 调用自由函数。
- `diagnostic` 与 `realtime` bundle 使用同一语义接口，只有后者通过 realtime 成本门。
- analytic control 和 neural method 可以通过同一 loader 加载，互不 fallback。

## Acceptance Criteria

- [ ] export → validate → load → parity 的 headless 链路通过。
- [ ] Slang reflection、Python manifest、C++ loader 对 layout/cost/capability 判定一致。
- [ ] 篡改 shader、weight、layout、contract 或 hash 均明确拒绝。
- [ ] loader 与 shader 中没有 Film、新 method ID 或 analytic control ID 的硬编码分支。
- [ ] bundle identity 和 replay 所需内容 hash 稳定。
- [ ] 子任务完成质量检查、提交并归档后，父任务才允许进入 `05`。

## Out Of Scope

- viewer UI/render lifecycle、方法重新训练和旧路径删除。
