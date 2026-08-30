# Metal matched sampler 与 PDF

## 目标

在不改变evaluator语义的前提下，利用其`PreparedState`中的analytic/residual lobes与proposal hints，完整实现静态有界、可训练、可部署且与evaluator条件一致的`sample()/pdf()`。本child验证数学、梯度和短程optimization flow，不等待full evaluator convergence或formal PT variance。

## 显式依赖

- 依赖`08-30-metal-canonical-architecture`的proposal batch、phase和component conformance合同；
- 依赖`08-30-metal-fused-full-method`的evaluator-side correctness checkpoint、固定`PreparedState` layout与proposal reservation；
- 输出由`08-30-metal-runtime-deployment`打包并接入viewer；
- 不依赖Linux long checkpoint、formal evaluation或compact task。

## 需求

- proposal 使用 `PreparedState` 中同一组 frame、roughness/anisotropy、active mask、mixture clues 与 typed compiler hints，不再次解码 texture 或运行 material compiler；
- fixed-capacity proposal mixture 覆盖 analytic core lobes、positive residual lobes 与能保证全半球支持的 fallback component；所有 component 都必须具有一致的有界 sample/PDF 实现；
- proposal 训练目标基于 evaluator 的 `luminance(f) * abs(cos(theta_i))`，不模仿 source sampler，也不把 source closure vocabulary 作为目标表示；
- `sample()` 使用固定数量随机数，返回 `wi`、forward PDF、有效标志及与公共 evaluator 一致的 throughput weight；每个有效 sample 至多执行一次 directional evaluator；
- 独立 `pdf()` 对相同 state/direction 求同一 mixture density，不依赖此前调用 `sample()`，并在 inactive、grazing、退化 frame 与零能量状态下 fail closed；
- Python oracle、Slang backend、package ABI 与 viewer PT 使用同一 component 顺序、normalization、hemisphere convention 和 precision policy；
- sampler 的质量报告包含 sample→pdf、forward/reverse PDF、weight identity、有限性、归一化、方向分布拟合和 matched PT variance；
- 此 child 不以 reference/analytic-only proposal 冒充最终实现，但可保留它们作为 matched controls。
- `proposal-fit`phase执行full-shape真实optimizer steps，proposal required groups必须有execution、finite非零gradient、update和Python/Slang artifact coverage。

## 不在范围

- 改变 full evaluator、texture codec 或 typed compiler 的目标语义；
- 通过新增 source-specific renderer 分支实现 sampling；
- compact profile、component 删除或任意效率 hard gate；
- full PT variance、formal泛化matrix或long-run sampler quality结论；
- 域外 recipe 组合、cutout 与任意外部 texture 导入。

## 验收标准

- [x] [数值正确性｜material interface] `sample()` 返回的 PDF 与对返回方向重新调用 `pdf()` 一致，throughput 满足公共 weight identity；
- [x] [物理正确性｜project contract] PDF 非负、归一化误差受控、有效方向位于支持域，退化输入无 NaN/Inf；
- [x] [实现正确性｜bounded runtime] component 数、循环、随机数、state、reads 与 MAC 静态有界，sample 后不重复 texture decode/material compile；
- [x] [数值正确性｜backend parity] Python与generated-layout Slang kernels在冻结probes上满足precision-derived tolerance；
- [x] [数值实现正确性｜parent completeness] proposal components/groups在proposal phase有execution、finite非零gradient、optimizer update及Python/Slang artifact coverage；
- [x] [需求交付｜parent full method] full identity同时具有`prepare/evaluate/sample/pdf`实现；analytic-only或reserved-only路径无法通过conformance；
- [x] [研究交付｜report-only] 短run density fit、support coverage、weight-tail和成本只作Linux long前流程诊断，不形成quality/variance hard gate；
- [x] [实现正确性｜项目回归合同] evaluator correctness checkpoint、canonical sampler tests与Falcor clean不回归。

## 阻塞问题

无；proposal component数和精度沿用full profile固定上限。evaluator未full convergence不阻塞数学/gradient实现验证，只有静态部署不可行才返回parent创建新profile identity。
