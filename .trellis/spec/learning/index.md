# Learning 层

learning层拥有产品`MethodDefinition` registry、`MethodDescriptor@2` component contracts、method/source adaptation、唯一`OnlineTrainingProducer`、phase `TrainingRunner`、`TrainingCheckpoint@4`、generic conformance、评测和program/asset/instance compilers。产品registry包含NVIDIA完整部署方法，以及具备`prepare/evaluate/sample/pdf`、full-profile asset cook、typed compiler与`ScatteringPackage@2`编译的Metal完整部署方法。typed route、Metal source/asset cook、evaluator/proposal loss、四phase runtime-weight QAT、Windows fixed-stream验证与Linux 692-source单GPU交接合同见[online-training.md](online-training.md)。

开发与质量合同见 `../project/unified-pipeline.md`。新增方法不得增加专用runner、CLI、checkpoint、exporter、Slang session、viewer分支或磁盘batch reader。
