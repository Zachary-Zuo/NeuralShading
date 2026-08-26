# 统一 pipeline 迁移状态

2026-08-27 架构重整已经将 source、reference query、offline/live batch、method registry、training/checkpoint、package 与 viewer 切换到单一合同。旧并行实现不保留兼容层；历史由 Git 与仓库外 artifacts 追溯。后续新增 source 或 method 直接遵循 `.trellis/spec/project/unified-pipeline.md`。
