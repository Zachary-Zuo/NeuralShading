# 项目规范入口

依赖方向：core 合同由 data、learning、viewer 单向消费；data 与 learning 通过 `TrainingBatch@1`，deployment 与 viewer 通过 `ScatteringPackage@1`。先读 `unified-pipeline.md`，再读目标层 index。viewer 不依赖 PyTorch，pbrt 不进入产品 discovery。
