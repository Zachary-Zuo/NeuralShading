# 实时材质编译

目标运行时是静态有界 neural scattering program：`prepare()` 获取/过滤 latent 并复用 view-conditioned state，`evaluate(wo,wi)` 直接输出线性 `f`，需要方向采样时提供匹配的 `sample/pdf`。source snapshot、program runtime、material asset 与 package 身份分离。解析 closure 可作为 reference core 或 proposal，不是目标输出词汇。

原生 MDL 是 source/reference 侧的新增入口，不改变 neural runtime 合同。项目用 MDL SDK 编译原生 program，再由当前 Falcor 8 生成方向响应 GT；训练和 compiler 消费统一 `EvaluatedBlock` / `TrainingBatch@1`，不依赖 falcor2。

runtime reference 与 neural runtime 共享 canonical `prepare/evaluate/sample/pdf` scattering 合同，但各自保留 backend 实现和交付身份。MDL V1 的四个 runtime 入口直接调用同一 compiled target code；方向响应 query 则固定 surface evaluate 与 `ExplicitLod(0)`，因为 `EvaluatedBlock` / `TrainingBatch@1` 当前不传 sampler。后续若加入 derivative filtering 或让训练消费 sampler，必须扩展对应 query schema/capability；不能用兼容层、generic proposal 或 viewer 私有入口补齐。
