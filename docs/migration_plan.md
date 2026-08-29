# 统一 pipeline 迁移状态

2026-08-30 canonical migration已经将source、grouped reference plan、multi-asset collection、typed online batch、method component registry、`TrainingCheckpoint@4`、三段package与viewer切换到单一合同。旧offline/HDF5、family-specific producer、固定双route lifecycle和旧package reader均已删除，不保留reader、converter或兼容层；历史由Git与仓库外artifacts追溯。后续新增source或method直接遵循`.trellis/spec/project/unified-pipeline.md`。
