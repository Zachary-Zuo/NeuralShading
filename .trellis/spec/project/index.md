# 项目规范入口

依赖方向：core合同由source/reference、learning、bundle与viewer单向消费；reference与learning通过`ReferenceExecutionPlan@1`、GPU online query、`NativeAssetCollection@1`和typed route batch连接，deployment与viewer通过`ScatteringPackage@2`的program/asset/instance binding连接。先读`unified-pipeline.md`，再读目标层index。viewer不依赖PyTorch，pbrt不进入产品discovery。
