# Metal matched ablation 与 compact deployment

## 目标

在Linux long checkpoint先经用户效果审阅、formal baseline另行完成且用户再次批准后，以冻结full candidate为唯一基线，通过预注册matched消融与压缩识别无贡献复杂度。该任务已从`08-30-vmaterial-metal-neural-system`父树解除，不在long run后自动启动。

## 激活门

- 当前保持独立`planning`；
- Linux long run后先看效果，不自动formal或ablation；
- 只有用户明确批准formal结果足以作为baseline并确认消融范围/预算后才start。

## 显式依赖

- 依赖已完成parent、用户选定的Linux long checkpoint和`08-30-metal-formal-evaluation`完整基线报告；
- sampler/evaluator/texture/compiler所有减法都等待上述激活门，不提前开始sampler-independent sweep；
- 复用 runtime child 的 package/profile identity、静态账本与 viewer benchmark；
- 产品阈值和聚合口径只在观察到 Pareto 后与用户另行对齐。

## 需求

- 在运行前预注册 ablation matrix、训练预算、seed policy、checkpoint selection、precision、query/workload与统计方法；
- 单机制消融覆盖 texture high/low grids、per-mip/two-mip、role conditioning/asset adapter、typed compiler FiLM/LoRA、analytic core、multiplicative correction、positive residual lobes、free tail、四路方向表示与 angular bank；
- compact sweep覆盖grid channels/resolution/bit-width、decoder/evaluator width/depth、adapter/rank、lobe/frame/field数量、read count、FP16/INT8敏感路径、蒸馏和冻结latent refinement；
- 每个变体具有新的 method/profile/checkpoint/package identity，不在同一身份下静默改变 shape、precision或feature集合；
- matched 对照保持source/split/query、训练步骤与sample budget、selection policy和backend workload一致；训练成本不一致时必须显式报告；
- evaluator结果沿用formal四层质量/语义/成本指标和source-state bootstrap CI；sampler变体另报告PDF正确性、bias/variance与path cost；
- compact候选必须重新经过Python→quantized→Slang→package→viewer parity，不能只凭离线模型大小命名为deployment profile；
- 结果只登记 observed Pareto、组件净贡献与失败类别，不自动反复修改模型/seed/预算追逐阈值。

## 不在范围

- 重新定义 source cohort、GT、typed semantics 或任意组合边界；
- 把 unbounded/offline teacher 当作可部署 compact 方法；
- 在得到实测前冻结单资产、working set或全cohort聚合成功口径；
- 自动替换产品默认 profile、删除 full baseline或宣称产品价值成立。

## 验收标准

- [ ] [研究正确性｜matched protocol] ablation manifest在运行前冻结，每个变体只改变已登记轴且 identity 可恢复；
- [ ] [统计正确性｜experiment framework] full 与变体的质量/语义/time/bytes差异均含matched bootstrap CI和真实训练成本说明；
- [ ] [需求交付｜parent R5] 每个full组件都有保留/删除/交互不确定的证据结论，不以单次点估计做删除决定；
- [ ] [部署正确性｜runtime contract] 所有Pareto候选满足固定state/read/shape、package校验、Slang/viewer parity与Falcor clean；
- [ ] [需求交付｜parent R7] 报告同时展示`B_shared/B_asset`、单资产/working set/cohort residency与`C_prepare/C_eval/C_sample/C_pdf`；
- [ ] [研究交付｜report-only] 输出non-dominated profiles及其能力边界，不在本child中创造事后hard gate；
- [ ] [需求交付｜parent research contract] configs、checkpoints、packages、raw metrics、reports和experiment-log记录全部可定位且写入`artifacts/`。

## 阻塞问题

当前阻塞是Linux long效果尚未审阅、formal baseline尚未获批/完成。产品默认profile、成功阈值和聚合权重仍延后到未来Pareto事实后的用户决策。
