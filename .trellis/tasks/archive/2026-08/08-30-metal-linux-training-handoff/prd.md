# Metal Windows validation 与 Linux 单GPU长训练交付

## 目标

在canonical architecture、Metal reference、full evaluator/sampler和runtime全部完成后，用Windows/RTX 4090证明完整profile能够稳定进行online gradient descent、短程收敛、resume和部署；随后冻结platform-neutral Linux单GPU长训练版本及操作说明。当前child不在Windows执行full convergence，也不自动启动Linux formal/ablation/Pareto。

## 显式依赖

- 依赖`metal-canonical-architecture`的新合同和跨平台backend；
- 依赖`metal-reference-foundation`冻结的registry/plan/assets/query；
- 依赖`metal-fused-full-method`、`metal-matched-sampler`和`metal-runtime-deployment`全部正确性证据；
- 是父任务当前交付的最终quality gate；future formal/compact只在Linux长训结果经用户审阅后另行批准。

## 需求

- Windows使用`metal_fused_full_v1`完整shape和required components；smoke只缩短step/batch/validation cadence；
- preflight覆盖692 exports、178 execution groups、52 assets、64 schemas以及全部component/parameter responsibility/role/recipe/proposal activation；
- 真实optimizer run使用由registry生成的最小stratified子集，只减少source/query/budget并保持full shape、online reference、loss、optimizer和phase data flow；
- `codec-warmup/joint-appearance/proposal-fit/qat-refine`各执行真实optimizer steps并至少跨一个phase checkpoint/resume；
- component execution、gradient/update和artifact coverage全部闭合，无orphan parameter或未导出required artifact；
- response target始终GPU-resident，无host readback或持久化batch；
- deterministic online micro-overfit/固定query-stream短run以预冻结统计方法证明末段loss低于初段；
- profile reference、asset encode、forward、backward、optimizer、validation/checkpoint、memory和sync热点；
- 生成短训diagnostic `ScatteringPackage@2`并完成Python/BF16-QAT/Slang/viewer、bundle replacement和typed edit smoke；
- 冻结同一method/profile/phase graph的Linux smoke与long configs，差异只限step/batch/cadence；
- 交付Linux source检查、deploy/preflight/train/resume/monitor/stop/failure-recovery命令；
- 训练固定单进程单GPU，不加入DDP或distributed state；
- Linux long run结束后只生成首轮效果审阅包，等待用户决定后续。

## 不在范围

- Windows full-quality convergence或最终checkpoint selection；
- 本child替用户监督Linux长训练；
- Linux DDP或多GPU测试；
- 自动formal matrix、追加seed/预算、消融、蒸馏、compact或Pareto；
- 旧config/checkpoint/package handoff。

## 验收标准

- [x] [数值实现正确性｜父任务Windows gate] full profile四phase无NaN/Inf、负f、非法PDF或silent clamp，所有required groups有finite非零gradient和update；
- [x] [数值实现正确性｜optimization contract] 预冻结window/bootstrap判据显示短run loss统计可信下降，phase resume后query/optimizer/precision state连续；
- [x] [需求交付｜用户高效正确性验证] 最小stratified训练子集覆盖全部required components/groups与关键source语义，且没有tiny/disabled/mock替代；full cohort preflight独立闭合；
- [x] [需求交付｜父任务online contract] target无host response readback，训练不读写batch corpus，热路径无逐parameter/metric同步；
- [x] [需求交付｜父任务完整性] 全cohort preflight、component execution/gradient/artifact coverage和program/asset/instance export闭合；
- [x] [数值实现正确性｜deployment oracle] diagnostic package的Python→QAT→Slang→viewer与sample/PDF invariants通过预冻结tolerance；
- [x] [需求交付｜用户Linux handoff] platform-neutral smoke/long configs、hashes、commands、VRAM/ETA、resume/monitor/stop/recovery说明齐全；
- [x] [需求交付｜用户单GPU边界] Linux命令只接受一个可见GPU，无DDP旁路；实际Linux Metal smoke是长训启动前的handoff gate，不由Windows结果冒充；
- [x] [需求交付｜用户结果审阅顺序] long run完成后的自动产物只包含首轮效果/健康/基础成本摘要，不排队formal/ablation/Pareto；
- [x] [工程正确性｜项目回归合同] unit/GPU/viewer Release、Linux shell/static config检查和Falcor clean通过。

## 阻塞问题

无；Linux实际Metal smoke由接收方在长训前执行，失败则按handoff说明返回implementation/protocol/resource分类，不能直接开始长训。
