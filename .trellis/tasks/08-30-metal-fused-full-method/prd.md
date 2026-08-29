# Metal quality-first 融合 neural method

## 目标

实现parent `metal_fused_full_v1`的codec、typed compiler、prepare与hybrid evaluator完整切片：全部required branches同时存在、端到端可训练、可恢复、静态有界，并向sampler/runtime提供冻结layout和proposal state。该child不是删减版方法；parent只有再完成matched sampler与runtime后才算full method交付。

## 显式依赖

- 依赖`08-30-metal-canonical-architecture`的`MethodDescriptor@2`、v4 phase runner/checkpoint和asset/evaluator batches；
- 必须消费`08-30-metal-reference-foundation`冻结的registry、execution plan、asset collection、typed-state/footprint query和split identity；
- 遵守parent `design.md`的full profile、三部分组合与互斥选择；
- runtime package/viewer由`08-30-metal-runtime-deployment`承接，本child冻结其layout/profile输入。

## 需求

- 一个新`metal-fused-neural-material@1` MethodDefinition，只使用canonical runner，不修改NVIDIA数学identity或增加专用runner；
- 最多9 source slots的role-aware shared encoder、per-mip high/low grids、shared decoder双head和bounded asset adapter；
- `RecipeBase ⊕ MetalOpticalState ⊕ FinishAssetState ⊕ TypedParameterDelta` pure compiler；
- raw + stable half/diff + 3 learned frames + shared warped angular bank；
- 6 core lobes、4 positive residual lobes、multiplicative correction和free positive tail全部启用；
- typed edit只重编material state，bundle replacement只换asset state；
- texture-set identity级`encoder-only`、bounded refinement、direct optimized control严格分开；
- adjacent two-mip decoded-state interpolation、连续footprint和source-awareloss；
- descriptor为每个codec/compiler/direction/evaluator component登记required contract、parameter groups、active phases和Python/runtime artifacts；
- `codec-warmup`与`joint-appearance`使用full shape执行，所有required groups通过stratified execution/gradient/update coverage；
- source texture tiles经shared encoder/decoder端到端反向，不能把预计算/frozen grids冒充joint encoder training；
- 全程记录但不以observed quality/time/memory作为实现完成门槛。

## 不在范围

- viewer/Slang最终交付、交互UI；
- matched proposal objective与`sample/pdf`实现，由紧随其后的sampler child完成；
- 完整候选后的组件删除或compact profile；
- 外部texture import、cutout或任意overlay组合。

## 验收标准

- [ ] [需求交付｜parent R2–R5] evaluator-side full candidate所有确认分支同时存在，shape/precision/reads/state静态有界并登记；proposal/sample/pdf由显式下游child补齐，不能被隐式跳过；
- [ ] [语义正确性｜parent R3] authored presets与连续typed edits共用pure compiler，缺失/离散参数语义正确；
- [ ] [数值正确性｜model contract] `f`有限、非负，invalid/chart seam/grazing有显式处理，无signed-clamp死区；
- [ ] [需求交付｜parent R2] 52 source texture sets可编译为独立bundle state，新资产三路径身份与split无泄漏；
- [ ] [实现正确性｜TrainingCheckpoint@4] codec/joint phases可smoke、resume、export tensor state，registry/plan/asset/query/profile/component identity严格恢复；
- [ ] [数值实现正确性｜parent completeness contract] 所有required evaluator-side components有非恒等execution、finite非零gradient、optimizer update和Python artifact，无orphan/placeholder；
- [ ] [需求交付｜parent joint encoder contract] active source tiles对shared encoder/decoder产生端到端gradient，未使用frozen/precomputed grids绕过；
- [ ] [研究报告｜report-only] semantic、local/energy/peak、parameter/footprint与成本指标被记录，不据此自动改方法或预算。

## 阻塞问题

无用户决策缺口；architecture/reference未冻结时只能开发独立数学fixture，不能复制旧runner/pyramid接口或启动Metal long training。
