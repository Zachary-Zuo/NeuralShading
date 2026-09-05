# Learning 层

当前公开方法为 `nvidia`、`metal`，真实模型/数据适配/编译位于各自方法目录。公共 `Method` 直接提供实现，`TrainingEngine`、在线 session、checkpoint 和图像接口共用；不保留旧 facet、状态转换或平台专用训练路径。

开发前读 [统一 pipeline](../project/unified-pipeline.md)、[online 训练](online-training.md)、[方法与 package](pipeline-and-evaluation.md)；部署读 [deployment.md](deployment.md)，数据调度读 [online-pipeline.md](../data/online-pipeline.md)。质量检查运行相关 unit、当前模型 GPU 回归与真实短流程；按 TESTING.md 区分 Windows 已执行和 Linux 待实机验证。
