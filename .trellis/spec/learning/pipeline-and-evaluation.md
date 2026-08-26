# 方法、训练与评测

产品方法只实现 `MethodDefinition`；registry 当前只有 NVIDIA。公共 `TrainingRunner` 调用 definition 的 phase/objective/state/compiler，不做 concrete type 分支。checkpoint 固定为 v2 严格 envelope；部署固定为 `ScatteringPackage@1`。质量比较继续使用 matched 数据与 bootstrap CI；当前 NVIDIA 是预算适配诊断。详见 `../project/unified-pipeline.md`。
