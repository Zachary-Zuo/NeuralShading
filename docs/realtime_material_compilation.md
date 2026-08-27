# 实时材质编译

目标运行时是静态有界 neural scattering program：`prepare()` 获取/过滤 latent 并复用 view-conditioned state，`evaluate(wo,wi)` 直接输出线性 `f`，需要方向采样时提供匹配的 `sample/pdf`。source snapshot、program runtime、material asset 与 package 身份分离。解析 closure 可作为 reference core 或 proposal，不是目标输出词汇。

原生 MDL 是 source/reference 侧的新增入口，不改变 neural runtime 合同。项目用 MDL SDK 编译原生 program，再由当前 Falcor 8 生成方向响应 GT；训练和 compiler 消费统一 `EvaluatedBlock` / `TrainingBatch@1`，不依赖 falcor2。V1 的 source reference 固定 surface evaluate 与 `ExplicitLod(0)`，后续若加入 derivative filtering 或 matched sampler，必须分别扩展版本化 query capability，并证明 evaluator/sampler 与 source reference 的语义一致。
