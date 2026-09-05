# 训练 checkpoint

当前只有 `learning/training/checkpoint.py` 的 `TrainingCheckpoint`，使用 `save_checkpoint(path, checkpoint)` 和 `load_checkpoint(path)`，eval/export 共用它。文件是一个原子写入的 `.pt`，含 `format: ncls.checkpoint`，没有额外 checksum sidecar 或版本转换 reader。

模型配置和 `model_state` 由方法恢复；optimizer、precision/scaler、phase/global step、RNG 与 query cursor 用于精确续训。DDP checkpoint 先 drain 所有 rank，再收集小型 rank state，由 rank 0 保存完整模型和 optimizer。完成态继续保留 optimizer。

`resolved_plan`、source/resource identity、代码来源、梯度覆盖和 validation 信息随文件保存，供追溯与诊断。续训比较模型、优化阶段、训练数据与 rank partition；不比较完整 plan、源码 hash 或运行设置。source 资源不符、实际 tensor 名称/shape/dtype 不符、cursor 无法对应时在加载边界报错。

日志频率、TensorBoard、图像 spp、预取和同卡数下物理 GPU 编号可变。改变模型/训练 batch/phase/卡数时建立新 run，不做弹性状态迁移。初始化 checkpoint 可以预览和导出，普通部署不受 formal/complete/coverage 标签限制。

新训练默认位于 `outputs/<config>/<run>/checkpoints/`。旧成果保持原地，不提供 importer、转换工具或兼容读取。
