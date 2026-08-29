# Metal-v1 与现有正式架构的差距审计

## 1. 结论

现有项目具备可复用的语义主干：原生`SourceSnapshot`/typed editor、统一GPU online reference、method registry、共享Slang scattering ABI和package viewer。后续训练交付审计与用户“唯一最新接口、无兼容层”决定已经推翻本审计早期的“沿用TrainingCheckpoint@3/ScatteringPackage@1并由method内部隐藏阶段”建议：当前设计采用canonical phase runner、`TrainingConfig/Checkpoint@4`和三层`ScatteringPackage@2`，并递归迁移全部调用方。

当前闭环是为“单snapshot、单latent pyramid、固定NVIDIA方法”验证生命周期而搭建，不能直接承载692 exports、52 texture sets、连续typed state、多个MDL generated modules和可替换bundle。必须把公共点从单资产形态迁移为execution-plan/multi-asset/phase-graph/three-part-package形态；不能把692个graph/preset变成shader、CLI或renderer分支，也不能保留旧pipeline兼容层。

## 2. 可直接复用的合同

### 2.1 Source 与 reference

- `MdlFamilyDefinition` 已能从 exact locator 构造 class-compiled snapshot，`describe_parameters()` 与 `apply_edit()` 保留原生 typed 类型、range、enum 和资源语义；Metal 无需新 source family。
- `ReferenceProgramDefinition`/`ReferenceBackendSession` 已统一 `prepare/evaluate/sample/pdf`、typed payload、CUDA/Falcor interop、valid-row 压实和 lease 生命周期。
- `ScatteringQuery` 已包含 position、frame、UV 与 UV derivatives，公共 query surface 不需要 Metal 专用字段。

### 2.2 Learning

- method registry与generic CLI原则可复用，但`MethodDescriptor@2`必须新增required component/phase/artifact合同。
- 公共batch保持method-neutral，但要升级为asset/evaluator/sampler三类typed schemas，显式承载asset/tile与footprint身份。
- runner的route RNG、GPU lease、resume和进度可复用；fixed双route/单optimizer/两阶段合同必须升级为canonical phase graph，不能由method按`global_step`隐藏。
- flattened tensors仍可保存multi-asset state，但checkpoint必须升级到v4以保存phase-local optimizer/precision、execution plan、asset collection和coverage state。

### 2.3 Deployment 与 viewer

- Slang `INclsScatteringBackend/State` 已把 concrete `PreparedState` 保留为方法私有布局，正好允许 Metal 扩大 per-hit structured state，而不污染公共 ABI。
- package的typed payload与hash原则可复用，但v1的`material`section混合asset/instance，必须由v2显式拆成program/asset/instance并删除v1 reader。
- viewer 已按 package 自身 module path 编译，并用 usage 名绑定权重、compiled material 与 texture/sampler；无需增加 Metal 渲染路径或 C++ program enum。

## 3. 必须补齐的 source registry

### 3.1 当前缺口

`references/mdl-vmaterials2-v1/families.json` 是先前冻结的 11-family/172-preset cohort，只含两个 Metal family。它不能代表本任务审计得到的 127 modules、692 opaque exports、178 opaque graphs、52 opaque texture sets 与 64 typed schemas。

### 3.2 调整

在现有 MDL reference package 下新增生成式 `metal-opaque` registry 和 JSON Schema，记录：

- exact locator、module/export、authored preset 与 opaque capability；
- graph/schema/texture-set/recipe/metal/finish identity；
- 逐参数 descriptor、默认值、硬/软 range、enum choices 与责任分组；
- 最多 9 个 source texture slots 的 path、role、packed channel、transfer、normal、filter/address 与 spatial-domain identity；
- 标准矩阵 compatibility 与特殊 family-local compatibility；
- source package、MDL SDK、bridge 和生成器 identity。

registry 从锁定 SDK inspection 生成并与审计数量交叉验证，不把原始 assets 写入根仓库。它是 source selector 与 neural compiler manifest，不扩张现有 viewer catalog，也不改变 vMaterials 上游包。

建议新增/调整：

- `src/ncls/source_materials/mdl_metal.py`：registry model、生成/校验与 identity；
- `references/mdl-vmaterials2-v1/schemas/mdl-metal-opaque-v1.schema.json`；
- `references/mdl-vmaterials2-v1/metal-opaque-v1.json`；
- `scripts/build_mdl_metal_registry.ps1`：只负责调用项目 Python/锁定 bridge，不复制审计逻辑。

## 4. Reference execution：当前单 snapshot 限制必须解除

### 4.1 证据

- `src/ncls/references/query.py:171` 在 material payload 含 generated module source 时明确拒绝多个 snapshots；
- `shaders/ncls/reference_backends/mdl.slang:47` 固定 `arg_block_offset = 0`；
- `NclsMdlCompiledMaterial` 当前只有 reserved 字段，尚不能选择 packed argument/RO offsets；
- 纹理 usage 是 graph-local 全局绑定，不允许在一个 shader dispatch 中混用不同 generated module/texture set。

因此把 692 snapshots 一次传给当前 `backend.open()` 不可行；即使去掉 Python 检查，也会产生 module/resource binding 冲突。

### 4.2 通用 execution-group 方案

新增 source-neutral 的 `ReferenceExecutionGroup`/routed session：

```text
execution_group_key =
  runtime module identity
  + generated module identity
  + RO layout/content identity
  + texture binding-set identity
```

- 每个 group 只编译一个合法 shader/resource specialization；
- 同 group 的 authored preset 与连续 parameter states 只增加 packed argument blocks 和 material records；
- producer 以 batch-homogeneous group 调度，避免一个 wave 或一个 dispatch 混入不同 generated programs；
- session pool lazy open，并按显存预算做显式 LRU；query identity 记录 group partition 和 cache policy，cache 命中不改变数学语义；
- group 内 `source_index` 选择 material record，record 提供 argument/RO offset；MDL `prepare()` 使用 compiled material，而不是固定 offset 0。

这一层属于 generic reference capability/session，不允许在 `OnlineTrainingProducer` 中写 `if family == MDL/Metal`。

建议调整：

- `src/ncls/core/scattering/program.py`：为 material payload 提供稳定 execution-group identity/aggregation metadata；
- `src/ncls/references/query.py`：单 specialization leaf session + routed/session-pool 组合；
- `src/ncls/references/programs/mdl.py`：可聚合 argument block/material record 与 group identity；
- `shaders/ncls/reference_backends/mdl.slang`、`mdl_runtime.slangh`：真正使用 per-material argument/RO offset；
- reference backend unit/GPU tests：同 graph 多 parameter states、跨 group 路由、lease、resume identity 和 cache eviction。

### 4.3 连续 typed state pool

当前 config 只列 source locators，producer 只在这些固定 snapshots 中采样。Metal 需要 source-backed 的未见参数状态，但仍不保存 response batch。推荐在 `online_query` 中增加版本化 `typed_state_recipe`：

- 以 base snapshot 的 `SourceParameterView` 为唯一参数域；
- 连续量用冻结的 stratified/Sobol recipe，bool/enum/index 只在离散域采样；
- 通过公共 `SourceFamilyDefinition.apply_edit()` 生成 deterministic edited snapshots；
- edited snapshots 的 argument blocks/material records 是 reference runtime state，可以缓存；它们不是离线训练响应数据；
- train/validation 使用独立 state recipe identity；resume 重建并验证 state-pool/group identities。

为控制 compile/load 开销，state pool 按 execution group 分块预备并轮换；不要求每个 direction query 创建一个新 MDL compilation。author preset、sampled edit state 和 held-out state 都走同一 exact reference program。

### 4.4 Footprint GT

当前 MDL package 明确是 ExplicitLod(0)，虽然 query 传递 UV derivatives，但普通 generated texture lookup 不形成连续 footprint GT。Metal full model 的 per-mip/footprint 需要一个版本化 reference recipe，不能假装现有 LOD0 已覆盖。

首选 source-neutral 方案是在 reference query 中增加固定上限的 footprint integration：以 `uv/uvDx/uvDy` 定义 filter support，在 GPU 上对 source `prepare/evaluate` 做确定性/低差异 UV 子样本平均；source stochastic evaluation samples 与 footprint samples 分开记账。它直接过滤完整 source response，能够覆盖 nonlinear graph 和 normal/mixture。硬件 derivative-aware MDL 与 conventional mip path作为 optimized control，不取代 filtered-response GT。

## 5. Online producer 与 source adapter

### 5.1 当前缺口

- `OnlineTrainingProducer._conditioning()` 把 direction proposal 固定为两种 NVIDIA recipe，并强制 `direction_count=1`；
- `NvidiaMaterialXSourceAdapter` 和 `NvidiaMdlFixedSourceAdapter` 都要求单 snapshot；后者拒绝空间纹理并使用固定 64-slot representation；
- `NativeFeaturePyramid` 只描述一个规则 pyramid，尚无 multi-bundle identity/slot metadata。

### 5.2 调整

保持唯一 producer，但把可变行为下沉到 registry-selected adapter/query recipe：

- 通用 direction proposal registry 支持 raw/uniform、half-difference、near-specular/peak、grazing 与 method-sampler conditioning；
- adapter 选择 batch 的 execution group、base source、typed state、asset/recipe，并返回 query surface 与方法 conditioning；
- 新增 `MetalMdlSourceAdapter`，输出 source/graph/schema/recipe/metal/finish/asset indices、packed typed tokens/presence、deterministic access fields、UV/footprint 与 semantic source features；
- 新增 `NativeAssetCollection` 协议，tile 化提供 52 bundle 的 role-aware mip tensors、asset offsets 和 schema metadata，避免把所有原图展开成一个 host tensor；
- batch 中仍只保存 tensor 与 identity/provenance，不增加 family-specific batch class。

建议调整：

- `src/ncls/learning/source_adaptation.py`：`NativeAssetCollection` 与 role/schema metadata；
- `src/ncls/learning/source_adapters.py`：拆成可发现 adapter modules，并新增 Metal MDL adapter；
- `src/ncls/learning/producer.py`：通用 proposal registry、adapter-owned source/state selection、routed reference session；
- `src/ncls/learning/batches.py`：继续保留通用 tensor mapping，只增加必要的 shape/identity validation helper。

## 6. 新方法接入，不改 NVIDIA identity

新增 `metal-fused-neural-material@1` 的独立 `MethodDefinition`，不修改 `nvidia-neural-appearance@3`：

- `src/ncls/learning/methods/metal_fused.py`：descriptor、full objective、lifecycle、checkpoint export/restore、runtime/material compile；
- `src/ncls/learning/models/metal_fused.py` 及按组件拆分的 codec/compiler/direction/evaluator modules；
- `src/ncls/learning/abi/metal_fused_layout_v1.json` 与生成的 Python/Slang layout；
- `shaders/ncls/backends/metal_fused/`：shared packed MLP、grid decode、access program、lobe bank、angular bank、hybrid evaluator 和 package wrapper；
- `configs/learning/metal-fused-*.json`：Windows smoke与Linux long使用同一method/profile/phase identity，只改变预算/cadence。

不增加Metal concrete runner，但必须先升级通用runner：v4 phase graph显式组织codec warmup、joint appearance、proposal fit和QAT refinement，每phase声明routes、parameter groups、optimizer/schedule与precision。禁止method内部根据`global_step`隐藏阶段或把所有参数塞进一个global optimizer。

evaluator child冻结proposal state，紧随其后的sampler child完整实现proposal phase与sample/pdf；Metal full package必须一次性具有`PREPARE/EVALUATE/SAMPLE/PDF`，不交付evaluator-only package。

## 7. Checkpoint、评测与 export

### 7.1 Checkpoint

`TrainingCheckpoint@3`不能继续作为canonical合同；迁移到v4且不提供reader/converter。v4保存：

- shared weights、compiler、flattened multi-asset grids、adapter tensors、offset/shape tables 都进入 method tensor schema；
- registry/plan/asset/profile/state-pool identities进入training config、query identity和phase/coverage/validation state；
- encoder-only、bounded refinement、direct optimized control 使用不同 config/method-run identity，不覆盖 full checkpoint。

### 7.2 评测

当前 `ncls learn evaluate` 只重新调用 training objective 并打印 mean loss，不足以表达 `G_asset/G_metal/G_finish/G_pair/G_param/G_recipe`、semantic reconstruction、parameter/footprint sweep 和 bootstrap CI。

应扩展通用 evaluation config/runner，输出到 `artifacts/`：

- 冻结 split/query identity 与分层 metric rows；
- local transformed/linear error、energy、peak、reciprocity、semantic role metrics；
- continuous parameter 与 footprint boundary sweeps；
- fixed reads、MAC/bytes metadata、GPU-only 与 viewer timing linkage；
- matched controls 和 ≥1,000 次 source-state bootstrap CI。

它仍通过 `MethodDefinition` evaluation hooks 和同一 online producer，不增加 Metal evaluate CLI。

### 7.3 Export

通用`ncls learn export`入口继续存在，但编译器拆为program/asset/instance三层，并从registry composition找到recipe、bundle和typed state。批量692 exports由通用manifest-driven选项完成，不创建Metal CLI。

## 8. `ScatteringPackage@2` 迁移

### 8.1 三层格式

Metal-v1需要`ScatteringPackage@2`。每个package表示一个可加载的composed material instance，但manifest/binding显式分层：

- program section：共享decoder/compiler/angular/evaluator/proposal权重和同一Slang module；
- asset section：bounded adapter、grid descriptors、量化high/low grid mip chains与sampler；
- instance section：recipe/optical/raw typed state与compiled program state；
- provenance：registry、bundle、schema、source texture 和 checkpoint identity。

多个packages共享同一个`program_runtime_id`；bundle replacement切换`texture_asset_id`并重编instance，不切换方法ABI。`B_shared/B_asset/B_instance`分开统计。v1 schema/reader/converter删除。

### 8.2 Canonical loader

当前 viewer 只识别 `gNclsRuntimeWeights`、`gNclsCompiledMaterials`、RGBA16F DDS 与一个 sampler dtype。Metal 需要：

- pack 所有 method-shared weights/angular data 进统一 weight buffer，或把 runtime blob binding改为完全按 usage 驱动；
- 为量化 grids 增加明确的 R/RG/RGBA 8/16-bit typed DDS 或 buffer format；
- instance blob必须打包成一个固定stride record，asset resources独立绑定；
- loader/resource validation 从 dtype whitelist 变为注册的 typed-resource factory，未知 dtype fail closed；
- viewer按`program_runtime_id`缓存pass/shared weights，分别建立asset/instance bindings，并在组合失败时保留旧三层binding。

建议调整：

- `src/ncls/bundle/typed_texture.py`、`writer.py` 与 package tests；
- `apps/viewer/ScatteringPackage.*`：host-only metadata、通用 typed resources与单 record验证；
- `apps/viewer/NclsViewer.*`：program-runtime cache、asset binding、atomic swap；
- package parity GPU tests：Python FP32、packed/quantized Python、Slang、viewer四方一致。

## 9. 交互 typed edit

仅靠重新导出 package 可以验证 compiler，但不满足材质实例级交互语义。Metal package 需要一个可选、方法无关的 editable-material capability：

- package provenance/resource 中携带 `SourceParameterView@1` 的 UI-safe schema、typed default buffer 和 recipe compatibility；
- viewer 继续使用原生 parameter path/type/range/enum，不把它改写成统一 PBR sliders；
- package module 可提供固定入口的 material-compiler compute pass，在参数改变时写入 `gNclsCompiledMaterials`；
- compiled buffer 以 SRV|UAV 创建，渲染前显式同步；每次 edit 只运行一次 compiler，不进入 per-pixel `prepare/evaluate`；
- asset switch 后用新 bundle identity 与保留/重置规则重新编译 state；不兼容参数显式提示，不能以零补齐；
- runtime compiler 与 Python compiler 做 tensor/Slang parity。

这需要给共享 package host ABI 增加 optional capability/entry，而不是在 viewer 中识别 Metal。若第一实现阶段先交付 Python material compile 与 package reload，必须标记为 compiler integration milestone，不能宣称 viewer interactive edit 已完成。

## 10. Viewer surface 与 frame

公共 `PathSurface.slang` 已统一 position、geometric/shading frame、UV 和 footprint，neural package 应直接消费。还需核对/补齐：

- deferred 与 PT 都把 normalized UV derivatives传入 Metal `prepare()`；
- spatial access program需要的 object/world position 与 coordinate-mode flags；
- rounded-corner 的 renderer-side final frame来源和与 source reference 的 matched control；
- decoded normal/frame只在 renderer/base frame之后组合；
- 失败、缺 UV、domain invalid、bundle不兼容均保持有限并 fail closed。

这些调整进入共享 surface/helper 或 package `prepare()`，不能复制一套 Metal path-surface 构造。

## 11. 推荐的实现分解与依赖

本系统包含数个可独立验证的 deliverable，当前 task 适合作为 parent：

1. **Canonical architecture migration**：先升级grouped plan、multi-asset/phase training、Config/Checkpoint@4和Package@2，并递归删除旧接口；
2. **Metal source registry与reference foundation**：依赖1，冻结GT、typed states、assets与footprint query；
3. **shared codec + typed compiler + hybrid evaluator**：依赖1–2，完整实现evaluator-side required components；
4. **matched sampler/pdf**：依赖3的正确性checkpoint，不等待formal convergence；
5. **Package@2/Slang/viewer**：依赖3–4，交付program/asset/instance和full capability；
6. **Windows gate与Linux单GPUhandoff**：依赖1–5，验证四phase短程优化并交付long config。

formal evaluation与compact/ablation已经从parent解除。Linux long run后先看效果，只有用户新的批准才启动。

parent 负责需求、方法身份、跨 deliverable 验收与最终集成；每个 child 的依赖必须写入自身计划，不能只依赖目录树顺序。
