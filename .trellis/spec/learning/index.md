# Learning 层

learning层拥有产品`MethodDefinition` registry、`MethodDescriptor@2` component contracts、method/source adaptation、唯一`OnlineTrainingProducer`、phase `TrainingRunner`、`TrainingCheckpoint@4`、generic conformance、评测和program/asset/instance compilers。产品registry包含NVIDIA完整部署方法，以及具备`prepare/evaluate/sample/pdf`数学实现的Metal full-profile evaluator与matched sampler；Metal的program/asset/instance package编译仍由runtime任务补齐，在此之前只对package export fail closed。typed route、Metal source/asset cook、evaluator与proposal loss合同见[online-training.md](online-training.md)。

开发与质量合同见 `../project/unified-pipeline.md`。新增方法不得增加专用runner、CLI、checkpoint、exporter、Slang session、viewer分支或磁盘batch reader。
