# 统一 pipeline 迁移状态

2026-08-28架构重整已经将source、canonical reference query、typed online batch、method registry、`TrainingCheckpoint@3`、package与viewer切换到单一合同。旧offline/HDF5、family-specific producer和专用query shader均已删除，不保留reader、converter或兼容层；历史由Git与仓库外artifacts追溯。后续新增source或method直接遵循`.trellis/spec/project/unified-pipeline.md`。
