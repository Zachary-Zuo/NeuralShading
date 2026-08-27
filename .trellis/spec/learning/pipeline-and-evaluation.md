# 方法、训练与评测

产品方法只实现 `MethodDefinition`；registry 当前只有 NVIDIA。公共 `TrainingRunner` 调用 definition 的 route、objective、lifecycle 与 compiler，不做 concrete type 分支。checkpoint 固定为 v2 严格 envelope；部署固定为 `ScatteringPackage@1`。NVIDIA 的 formal recipe 保留独立 evaluator/sampler online batch、encoder→hierarchical latent materialization→finetune、matched sampler 与 packed-FP16 runtime；smoke、profile 和历史 25k run 使用独立 identity，不能冒充 formal。质量比较继续使用 matched 数据与 bootstrap CI。详见 `../project/unified-pipeline.md`。
