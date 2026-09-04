# Learning 层

learning 层拥有显式 `MethodPlugin` registry、`MethodDescriptor@2` component contracts、method source adaptation、唯一 `OnlineTrainingProducer`、固定 `TrainingEngine`、`TrainingCheckpoint@1`、legacy v4 只读 evaluation importer、generic conformance、hooks 以及 program/asset/instance compiler。产品 registry 的公开短 key 为 `nvidia` 与 `metal`；版本和 implementation identity 进入 resolved plan/checkpoint，不进入名称。typed route、Metal source/asset cook、step-1 evaluator/proposal 联合目标、两 phase runtime-weight QAT、checkpoint readiness、TensorBoard/visual eval 和 Linux 692-source 交接合同见 [online-training.md](online-training.md)。共享 CPU/GPU/reference 调度见 `../data/online-pipeline.md`。

开发与质量合同见 `../project/unified-pipeline.md`。新增方法只实现 model/data/objective/lifecycle/checkpoint/deployment facet；不得增加专用 runner、CLI、checkpoint、exporter、Slang session、viewer 分支或磁盘 batch reader。
