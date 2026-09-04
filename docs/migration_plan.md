# 统一 pipeline 迁移状态

2026-09-04 训练架构迁移把 YAML composition、`MethodPlugin`、`DataExecutionPlan/OnlineDataSession`、固定 `TrainingEngine`、`TrainingCheckpoint@1`、TensorBoard/visual hooks、三段 package 与 viewer 消费者接入同一合同。旧 offline/HDF5、family-specific producer、旧 JSON config、`ncls learn` 和旧 runner 入口已删除。`TrainingCheckpoint@4` 只保留隔离的只读 evaluation importer，不得用于继续训练。历史由 Git 与仓库外 artifacts 追溯；后续新增 source 或 method 直接遵循 `.trellis/spec/project/unified-pipeline.md`。
