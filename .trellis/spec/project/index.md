# 项目规范入口

依赖方向：core 合同由 source/reference、learning、bundle 与 viewer 单向消费；reference 与 learning 通过 GPU online query 和 typed route batch 连接，deployment 与 viewer 通过 `ScatteringPackage@1` 连接。先读 `unified-pipeline.md`，再读目标层 index。viewer 不依赖 PyTorch，pbrt 不进入产品 discovery。
