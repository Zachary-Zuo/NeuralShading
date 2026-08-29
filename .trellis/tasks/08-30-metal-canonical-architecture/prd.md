# Metal canonical architecture migration

## 目标

把现有source→reference→online training→checkpoint→package→viewer pipeline一次性迁移到能够自然表达grouped reference、multi-asset joint training、任意phase graph、完整method conformance和program/asset/instance deployment的唯一最新接口。迁移覆盖所有现有正式调用方，并删除旧接口与兼容层，为Metal full method提供整洁、source/method-neutral的底座。

## 显式依赖

- 依赖父任务冻结的`research/training-delivery-boundary-audit.md`和`design.md`；
- 是`metal-reference-foundation`、`metal-fused-full-method`、`metal-matched-sampler`、`metal-runtime-deployment`和handoff的前置依赖；
- 不依赖Metal checkpoint或长训练结果。

## 需求

- 新canonical pipeline采用`ReferenceExecutionPlan@1`、`NativeAssetCollection@1`、typed asset/evaluator/sampler batches、`MethodDescriptor@2`、`TrainingConfig@4`、`TrainingCheckpoint@4`和`ScatteringPackage@2`；
- `TrainingConfig@4`表达versioned phase graph、任意typed routes、parameter groups、optimizer/schedule/precision与phase checkpoint boundary，不硬编码双route或bootstrap/finetune；
- `MethodDescriptor@2`登记required components、parameter groups、active phases、batch dependencies、Python/runtime artifacts与Slang entry points；
- runner支持phase-local trainable groups、mixed precision、GPU聚合finite/gradient checks、低同步metrics和安全route prefetch；
- checkpoint保存phase graph/component manifest、phase-local optimization state、query/asset/plan identities和gradient coverage；
- package显式分离program、asset、instance identities/resources，viewer相应分离runtime cache与bindings；
- LayerStack、MERL、OpenPBR、MaterialX、MDL、NVIDIA method、四个现有training configs、CLI、tests、spec/docs和viewer全部递归迁移；
- 迁移完成后删除`TrainingConfig/Checkpoint@3`、`ScatteringPackage@1`、`NativeFeaturePyramid`及旧batch/runner/package schema的reader、alias、converter、probe和fallback；
- source/reference/method差异只存在于definition/provider/adapter，平台差异只存在于现有backend/launcher/provider；
- 不增加Metal-specific runner、CLI、exporter、session或viewer分支，也不预埋DDP路径。

## 不在范围

- Metal registry、52 assets或full method数学实现；
- 旧checkpoint/package转换工具或兼容加载；
- multi-GPU DDP和distributed checkpoint；
- Linux viewer、UE或新的source family。

## 验收标准

- [ ] [需求交付｜父任务R4] 新canonical接口覆盖reference plan、asset collection、phase training、component conformance和三部分package，公共数据流只有一条；
- [ ] [工程正确性｜项目迁移合同] 五个reference、NVIDIA、configs/tests/spec/docs/package/viewer全部使用新接口，旧public symbols/format/schema/reader/alias/converter/fallback静态扫描为零；
- [ ] [数值实现正确性｜现有oracle] 迁移后的NVIDIA Python/Slang/package与五个reference不放宽既有parity、sample/PDF、finite和lease判据；
- [ ] [需求交付｜父任务R6] runner具备phase-local groups、mixed precision、低同步gradient audit、metrics cadence和multi-slot prefetch合同；
- [ ] [需求交付｜父任务跨平台合同] upper modules无OS/device/build-path分支，同一v4 config可被Windows D3D12/CUDA与Linux Vulkan/CUDA backend解析；
- [ ] [实现正确性｜package v2合同] program/asset/instance hashes、typed resources、atomic binding与tamper rejection完整，旧v1 package明确拒绝；
- [ ] [回归正确性｜项目测试合同] unit、GPU/reference、NVIDIA training/package、viewer Release build与Falcor clean通过。

## 阻塞问题

无；这是父任务已经确认的破坏式迁移，不保留兼容层。
