# Learning 层

learning 层拥有产品 `MethodDefinition` registry、method/source adaptation、唯一 `OnlineTrainingProducer`、`TrainingRunner`、`TrainingCheckpoint@3`、评测和 deployment compiler。产品 registry 当前只有 NVIDIA；contract fixture只位于tests。typed route与NVIDIA loss合同见 [online-training.md](online-training.md)。

开发与质量合同见 `../project/unified-pipeline.md`。新增方法不得增加专用runner、CLI、checkpoint、exporter、Slang session、viewer分支或磁盘batch reader。
