# 实时材质编译

目标运行时是静态有界 neural scattering program：`prepare()` 获取/过滤 latent 并复用 view-conditioned state，`evaluate(wo,wi)` 直接输出线性 `f`，需要方向采样时提供匹配的 `sample/pdf`。source snapshot、program runtime、material asset 与 package 身份分离。解析 closure 可作为 reference core 或 proposal，不是目标输出词汇。
