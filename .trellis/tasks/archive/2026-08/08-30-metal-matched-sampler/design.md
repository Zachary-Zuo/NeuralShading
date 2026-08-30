# 设计

## Proposal 表示

proposal 是固定容量混合分布：已存在的 6 个 analytic core slots、4 个 positive residual lobe slots，以及 1 个保证全半球支持的 cosine/uniform fallback。`PreparedState` 保存 component active mask、归一化 mixture logits、frame、roughness/anisotropy、support 与必要的 normalization 数据；inactive slots 仍占 ABI 位置但权重为零。

每个 component 采用可解析采样且可求密度的有界分布。core component 沿用对应 GGX/Beckmann/diffuse proposal；residual component 使用具有固定参数布局的各向异性正值 lobe proposal。fallback 保证 evaluator free positive tail 在其他 component 漏覆盖时仍有非零 PDF。所有 component 的 sample/PDF 都在同一 shading-frame 与 hemisphere convention 下定义。

## 训练

`proposal-fit`phase按config冻结或低学习率保持evaluator/codec/compiler，并训练proposal head及bounded modulation。目标密度与`luminance(f) * abs(cos(theta_i))`成比例；使用reference/evaluator directions、proposal samples和显式grazing/peak strata的mixture，联合优化density fit、mode coverage与weight-tail risk。analytic-only、source proposal只作control，不提供teacher labels。

## Runtime 数据流

`prepare()` 复用既有 state 并生成最终 mixture。`sample()` 用一个离散 component 选择随机数和两个 component 随机数产生 `wi`，随后对全部 active component 累加 mixture PDF，并调用一次 directional evaluator形成 weight。独立 `pdf()` 只计算 component densities，不运行 texture decoder、typed compiler或完整 evaluator。

Python oracle与Slang通过生成的component enum/layout共享顺序。child冻结完整`PREPARE/EVALUATE/SAMPLE/PDF` method artifact；Package@2 capability与viewer PT由runtime child消费，不提供无声fallback。

## 验证证据

验证包括component normalization、采样直方图、sample→pdf与极端参数；GPU parity覆盖随机state/direction/random tuple；短phase验证proposal gradients/updates和density下降。正式viewer bias/variance/time等待runtime及Linux结果审阅。所有报告绑定evaluator/proposal checkpoint、layout与query identity。
