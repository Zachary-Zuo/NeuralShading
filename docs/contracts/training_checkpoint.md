# TrainingCheckpoint@1 合同

新训练只写 `ncls.training-checkpoint@1`。checkpoint 包含公开 method key、内部 implementation key、descriptor/implementation hash、六个 facet identity、完整 `ResolvedTrainingPlan@1` 及其 hash。用户可见 key 不携带 `@版本`；版本变化由结构化 identity 检出。

data identity 同时保存 `DataExecutionPlan`、reference program/plan、native asset collection、query stream、source contract 与 source snapshot identity。运行游标由 `global_step + phase_index/name/step` 表达；逐 rank RNG 与 data session cursor 在 DDP envelope 中保存，checkpoint 前必须 drain，不能让未消费 batch 或活跃 lease 跨越边界。DDP只把这些小型rank-local状态收集到rank 0；完整model/optimizer CPU snapshot与durable write只在rank 0构造，其他rank等待显式commit status，不复制完整checkpoint，也不在rank-0写入时提前销毁process group。

优化状态属于当前 phase，包含 named optimizer state、scheduler 与 precision/scaler；跨 phase 只按 `optimizer_state_policy=carry-overlap` 传递同名重叠参数。checkpoint 同时保存 model tensor、每个 parameter group 的 finite/nonzero-gradient/actual-update coverage、validation/selection evidence、hook cursor 与已经发布的 visual probe identity。完成态不携带失效的 optimization state。

resume 必须匹配同一 resolved plan、method/facet implementation、source、reference、asset、query stream 与 data execution identity；任一漂移都拒绝。读取先验证 `.pt.sha256`，再严格解析字段，并由 method checkpoint facet 恢复 tensor。

旧 `TrainingCheckpoint@4` 只有一个隔离的只读 importer。它先执行原有 descriptor、shape、identity 和 readiness 校验，再产生 `EvaluationSnapshot`，仅供 `validate`、允许的 export 与 visual eval 使用。v4 不能 resume；项目不提供旧 JSON config reader、converter、method alias 或训练 fallback。
