# Learning 层

learning 层拥有产品 `MethodDefinition` registry、唯一 `TrainingRunner`、`TrainingCheckpoint@2`、评测和 deployment compiler。产品 registry 当前只有 NVIDIA；contract fixture 只位于 tests。

开发与质量合同见 `../project/unified-pipeline.md`。新增方法不得增加专用 runner、CLI、checkpoint、exporter、Slang session 或 viewer 分支。
