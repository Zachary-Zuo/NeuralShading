# Metal source registry 与 grouped reference

## 目标

为 parent `08-30-vmaterial-metal-neural-system` 建立可供 full neural method 正式训练的权威输入层：版本化全量 opaque Metal registry、可执行多 graph/多 typed state 的 grouped MDL reference，以及连续 footprint 的 filtered response GT。

## 显式依赖

- 继承 parent `prd.md` 的全部 source 范围与原生语义要求；
- 设计依据 parent `design.md` 和 `research/architecture-gap-audit.md`；
- 依赖`08-30-metal-canonical-architecture`冻结的`ReferenceExecutionPlan@1`、`NativeAssetCollection@1`、typed batches和`TrainingCheckpoint@4`；
- 本child完成并冻结registry/query/asset identities后，`08-30-metal-fused-full-method`才能实现full training。

## 需求

- registry 覆盖 692 opaque authored exports，去重登记 178 opaque graphs、52 texture sets、64 schemas；145 cutout exports显式拒绝；
- 保存 exact locator、typed descriptor、recipe/metal/finish/bundle compatibility、最多9个 texture slots与完整 provenance；
- reference 以 generated module + RO + texture bindings 划分 execution groups，同 group聚合 argument states，跨 group batch-homogeneous调度；
- typed state recipe只通过公共 source editor生成连续/离散 edit states，train/validation identity隔离且可恢复；
- filtered footprint GT对完整 source response做固定上限UV积分，和source stochastic samples分账；
- 52个texture-set identities编成role/schema/mip/tile+halo的`NativeAssetCollection`，training与cook共享同一asset identity；
- 当前 LOD0 integration、prepare-hoisted/PDF-reuse optimized source保留为对照，不冒充filtered GT；
- 所有扩展实现canonical execution-plan/collection合同，不建立Metal专用query API、producer或single-session兼容旁路。

## 不在范围

- neural model、texture encoder、compiler/evaluator训练；
- cutout、任意外部textures或任意overlay authoring；
- 把MDL graph改写成LayerStack；
- 本阶段以throughput/显存observed值设置效率hard gate。

## 验收标准

- [ ] [需求交付｜parent R1/R7] registry数量、identity、schema与资源hash可复现，unknown/missing/cutout fail closed；
- [ ] [语义正确性｜MDL source contract] authored default和sampled typed states都由same exact class-compiled program执行；
- [ ] [实现正确性｜公共reference ABI] 同group多argument states与跨group routing的evaluate/sample/pdf有限、lease安全、resume identity稳定；
- [ ] [语义正确性｜parent mip合同] footprint query真实改变filtered response，filter samples与stochastic samples不混淆；
- [ ] [需求交付｜parent R7] 输出group、compile/load/query/storage diagnostic账本但不形成hard gate；
- [ ] [需求交付｜parent R3/R6] 52 assets的role/schema/mip/tile collection可训练采样、cook和恢复，不展开全量host tensor；
- [ ] [实现正确性｜project canonical contract] 只使用child 1的新plan/collection/checkpoint合同，无single-session/pyramid兼容路径，现有五种source和NVIDIA回归成立。

## 阻塞问题

无。child 1未完成时只允许registry生成器和独立MDL provider fixture开发，不得复制旧session/asset接口形成临时产品路径。
