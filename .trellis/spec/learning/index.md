# Learning 层

learning层拥有产品`MethodDefinition` registry、`MethodDescriptor@2` component contracts、method/source adaptation、唯一`OnlineTrainingProducer`、phase `TrainingRunner`、`TrainingCheckpoint@4`、generic conformance、评测和program/asset/instance compilers。产品registry包含NVIDIA完整部署方法与Metal full-profile evaluator切片；后者的matched sampler/runtime能力按冻结layout在后续任务补齐，未完成前fail closed。typed route、Metal source/asset cook与loss合同见[online-training.md](online-training.md)。

开发与质量合同见 `../project/unified-pipeline.md`。新增方法不得增加专用runner、CLI、checkpoint、exporter、Slang session、viewer分支或磁盘batch reader。
