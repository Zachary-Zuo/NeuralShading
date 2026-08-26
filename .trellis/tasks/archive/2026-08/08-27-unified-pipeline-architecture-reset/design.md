# 统一 Pipeline 架构设计

## 设计目标

本设计只保留一条从 source material 到训练、部署与 viewer 的路径。当前 NVIDIA 方法是唯一产品级 neural method；LayerStack random-walk、OpenPBR 1.1.1、MERL 与 MaterialX/Poly Haven 是四个进入该 pipeline/viewer 的 ground-truth reference。pbrt coated probe 只是 LayerStack 两界面 coated slice 的外部交叉验证工具，保持在 `external/`、`tools/reference/`、`references/` 和 `artifacts/` 边界，不成为该 pipeline 的 source/reference runtime。Falcor 是四个正式 reference 与 viewer 的 GPU 执行后端。方法、材质语义、外部 oracle 和执行后端不能互相渗透为公共接口。

设计完成后的核心性质是：

1. 新 neural method 只实现一个 `MethodDefinition`，不修改数据采集、训练 runner、checkpoint writer、部署 writer、viewer C++/CMake/RenderGraph/UI。
2. 新 source family 只实现 source state/reference program，不复制 offline collector、online trainer 或 viewer integrator。
3. 四个 viewer-ready reference 与 neural method 都被 viewer 加载为同一种 `ScatteringBinding`，并实现同一 `prepare/evaluate/sample/pdf` 合同。
4. viewer 只有一个 PT renderer source 和一个 deferred renderer source；左右 slot 只是两次独立 specialization。
5. 旧方法、旧 schema 和重复 runner/exporter 在同一任务内递归迁移后删除，不保留兼容层。

## 总体数据流

```text
SourceAsset
  │
  ├─ SourceFamilyDefinition ── ReferenceProgram ── ReferenceExecutionBackend
  │                                   │                         │
  │                                   └──── ReferenceQueryBatch ┘
  │                                                   │
  │                        ┌──────────────────────────┴─────────────────────────┐
  │                        │                                                    │
  │              Offline shard sink                                  Live GPU tensor view
  │                        │                                                    │
  │              OfflineBatchSource ───────────────┐                            │
  │                                                ├─ TrainingBatch ── TrainingRunner
  │              LiveReferenceBatchSource ─────────┘                            │
  │                                                                             │
  │                                                        MethodDefinition (NVIDIA)
  │                                                                             │
  │                                                        TrainingCheckpoint@2
  │                                                                             │
  └──────────────────────────────────────────────────────── DeploymentCompiler
                                                                                │
                                              ┌─────────────────────────────────┴─────────┐
                                              │                                           │
                                       MethodRuntime                         CompiledMaterialAsset
                                              └────────────────────┬──────────────────────┘
                                                                   │
                                                     ScatteringPackage@1
                                                                   │
                                                    common loader → ScatteringBinding
                                                                   │
                                        ┌──────────────────────────┴──────────────────────┐
                                        │                                                 │
                                   Slot 0 PT/deferred                               Slot 1 PT/deferred
```

## 合同与所有权

### 1. Source family、reference program 与 execution backend

`SourceFamilyDefinition` 拥有：

- 原生 source asset schema、资源解析、参数编辑与 canonical identity；
- source state/query 的生成与验证；
- 对应 `ReferenceProgram`；
- 把原生 source asset 编译/绑定为 reference runtime 私有 material payload 的逻辑。

`ReferenceProgram` 拥有 GT 的数学与随机语义。当前实例包括 LayerStack random-walk、OpenPBR、MERL 与 MaterialX/Poly Haven；公共 query、renderer 和 training runner 不理解层数、OpenPBR 参数、MERL table layout、MaterialX 图结构或其他族私有语义。这不意味着参数被隐藏；可编辑表面由 source family 通过下述公共 editor contract 对外描述。

现有 reference 的迁移边界如下：

| Registry role | Source/reference | 新架构中的职责 |
| --- | --- | --- |
| ground-truth + viewer-ready | LayerStack / random-walk | `SourceFamilyDefinition`、`ReferenceProgramDefinition`、统一 batch producer、reference `ScatteringPackage` |
| ground-truth + viewer-ready | OpenPBR 1.1.1 | 保留原生 resolved inputs/编辑/资源与独立 parity，迁入相同 definition/package/binding |
| ground-truth + viewer-ready | MERL measured BRDF | 保留原始测量表与插值/标定，迁入相同 definition/package/binding |
| ground-truth + viewer-ready | MaterialX/Poly Haven | 保留原生图、纹理、颜色空间和 upstream image parity，迁入相同 definition/package/binding |
| external crosscheck oracle | pbrt coated probe | 不进新架构。上游保持在 `external/pbrt-v4/`，长期 probe/compare 保持在 `tools/reference/`，身份/适用范围记录保持在 `references/pbrt-coated-crosscheck-v1/`，结果进 `artifacts/` |

四个 ground-truth reference 不是 neural candidate，不能出现在 neural method 删除清单中。它们的迁移允许替换旧 provider/viewer dispatch，但必须保留原生语义、资源、registry 身份和验证证据。pbrt 也不是清理对象，但原因是它是位于 pipeline 之外的锁定对照工具，而不是待迁入的 reference runtime。

#### pbrt coated probe 与 LayerStack 的边界

`references/registry.json` 用 `independent-validation` 记录 pbrt probe 对 `ncls.layer-stack@1` 的适用关系。这个 manifest 是 provenance/discovery 记录，不把 pbrt 提升为新 pipeline participant。

- LayerStack random-walk 是该族的正式 Falcor ground-truth implementation；
- pbrt-v4 `CoatedDiffuseBxDF`/`CoatedConductorBxDF` 是独立上游实现，只用来对照 rough-dielectric top + homogeneous slab + diffuse/rough-conductor base 的 `N=2` slice；
- LayerStack random-walk 不是 pbrt 代码的 Falcor port，只是在该重叠构造上应有一致物理语义；
- pbrt 不实现新架构的 `SourceFamilyDefinition`、`ReferenceProgramDefinition`、`TrainingBatch`、`ScatteringPackage`、`ScatteringBinding` 或 viewer mode；
- `tools/reference/pbrt_compare.py` 作为长期专用 crosscheck tool 是合法的语义单元，不因公共 pipeline 统一而被抽象成新 runner。若 LayerStack API 迁移使其无法调用，只修正 import/调用点和 manifest hash。

因此本任务对 pbrt 的处理是“边界保护”：不删除、不复制、不抽象、不接入新架构，只在必要时保持现有对照工具能够调用迁移后的 LayerStack 正式 reference。

`ReferenceExecutionBackend` 只负责任务提交、buffer 生命周期、同步和 tensor transport。当前 GPU 实例使用 Falcor；它不能产生或修改 source-family 语义。

offline 与 live 的组合关系为：

```text
ReferenceQueryStream + ReferenceExecutionBackend
  ├─ + CorpusShardSink  → offline corpus
  └─ + DeviceBatchView  → online training
```

不得再为 mollification、某个方法或某个训练阶段新增 collector/manifest/reader。方向扰动、query 密度、target estimator 和 curriculum 是版本化 `QueryRecipe`/`TrainingRecipe` 组件，进入同一 stream。

#### SourceParameterView@1 与 SourceEditPatch@1

`SourceFamilyDefinition.describe_parameters(snapshot)` 返回当前 source snapshot 的自描述 `SourceParameterView@1`。它是编辑表面，不是全族统一的材质 IR，也不是 neural 方法必须直接使用的 feature vector。

```text
SourceParameterView@1
  family_id / source_contract_version / snapshot_id
  root: ParameterNode
    kind: group | list | variant | value | resource | read-only
    path / stable element_id / label / group / order
    value_type / current / default
    enum choices / unit / range / step / UI hint
    binding: constant | texture | graph | geometry | measurement | derived
    editable / read_only_reason
    allowed operations / family constraint identity

SourceEditPatch@1
  base_snapshot_id
  operations[]: set | insert | remove | move | replace-variant
    target path or stable element_id
    typed value / insertion payload / destination
```

公共 UI 只根据 node kind 生成控件并提交 patch，不存在 `if LayerStack/OpenPBR/MaterialX`。path 是 family contract 内稳定的语义标识；列表成员使用稳定 element ID，不用会在重排后改变的数组 offset。label/range/widget 是显示元数据，canonical identity 只由 source 语义、资源和值决定。

`SourceFamilyDefinition.apply_edit(snapshot, patch)` 是唯一写入入口，完成 optimistic base-ID 检查、类型/跨字段约束验证、资源解析和 canonicalization，返回新 immutable `SourceSnapshot`、`source_state_id`、changed paths、invalidation 和 diagnostics。

四个已有 family 的映射为：

- LayerStack：`interfaces`/`media` 为可变长 list/variant，支持 coat 增删重排、base kind 替换与参数编辑；不对外暴露固定 `LayerStackIR` packet offset。
- OpenPBR：使用 OpenPBR 原生名称/类型，并显式暴露 Constant/Texture/Graph/Geometry binding provenance；只有 family 允许的 binding 可编辑。
- MaterialX：可编辑带 value 且未连接的 constant input；connected input 以 read-only node 暴露连接来源和原因，普通 value patch 不得改写图结构。
- MERL：返回空的连续参数编辑面/测量 provenance；切换 measurement table 是 source asset selection，不伪造 roughness/metalness 等参数。

scene/corpus/package provenance 保存 canonical source snapshot 或其可验证引用；`SourceParameterView` 由 snapshot 重建，不是另一份 source truth。

### 2. 统一 TrainingBatch

训练 runner 只接收一个 typed `TrainingBatch`。至少包含：

- `source_family_id`、`source_state_id` 与 batch-local state handle/features；
- surface/view/query 方向与坐标系身份；
- target 的量、measure、sample count、PDF/importance weight 与随机 seed identity；
- role/split/provenance；
- 全部 tensor 的 dtype、shape、device。

字段按能力/语义分组，不按方法名增加分支。offline 与 live producer 必须产生同一 logical schema；offline producer 在读取后一次性搬到目标 device，live producer 直接返回 CUDA tensor。

#### GPU-resident online 路径

锁定 Falcor 的 `Buffer.to_torch()` 通过 CUDA external memory/DLPack 返回 device tensor；`Buffer.from_torch()` 可进行 device-to-device 输入。实现要求：

- Falcor input/output buffer 使用 `ResourceBindFlags.Shared`；
- output 不调用 `to_numpy()`、`getBlob()` 或 HDF5 writer；
- batch 返回前用明确的 Falcor/CUDA 同步保证 torch 可见性；先允许正确但同步的 `device.wait()`，随后只在 profile 证明需要时改为 ring buffer/fence；
- tensor 的所有权/lease 覆盖 loss 与 backward，下一批不能提前覆写仍在使用的共享 buffer；
- smoke 同时断言 tensor `is_cuda`、data pointer/device 不经过 host、offline/live schema 完全相同；
- Windows D3D12/CUDA 与 Ubuntu Vulkan/CUDA 分别使用已有 `scripts/run_falcor_python.*` 入口，不修改上游 Falcor。

### 3. MethodDefinition：唯一 neural method 扩展点

方法目录只导出一个 `MethodDefinition`，注册器从 `ncls.methods` 包目录发现，不维护第二份名字表。定义包含：

```text
descriptor
  method_key / version / implementation hash
  supported source contracts
  source adaptation contracts
  training batch requirements
  runtime ABI / capabilities / static cost claims

create_trainable(context)
training_recipe(model, batch, phase) → losses + metrics
export_training_state(model) → validated TensorMapping
restore_training_state(model, TensorMapping)
compile_runtime(checkpoint) → RuntimePayload
compile_material(source_asset, checkpoint) → MaterialPayload
```

`SourceAdaptationContract` 以 `family_id + source contract/schema version` 为键，声明 method 可接受的 parameter domain。对一个已验证 patch，它返回 `unchanged`、`runtime-patch`、`recompile` 或 `unsupported`：分别表示改动不影响 payload、可转成 method-private blob 更新、必须从完整新 snapshot 重新产生 `CompiledMaterialAsset`，或超出该 method/checkpoint 的支持域。

`compile_material()` 始终消费完整 immutable source snapshot 和已验证的 source contract，而不消费 viewer 控件或通用 float 数组。method 可在自身定义内提供 family-specific feature/compiler adapter，但 registry、runner、package writer 和 viewer 都只调用公共面。当前 NVIDIA 可对未编译状态诚实声明 `unsupported` 或离线 `recompile`，无需伪造可直接调参的 latent。

`TrainingBatch` 只携带 source snapshot/state identity 与 recipe 约定的 batch-local source handle/features。方法如需把可编辑参数作为 conditioning，由其 source adapter 按稳定 path 读取并生成 feature；公共 runner 不将所有 family 的 editor schema 展平成固定向量。

方法不拥有：

- 文件路径、原子写、checkpoint 顶层 schema 或 hash；
- corpus/live source 的打开方式；
- optimizer loop、checkpoint selection、评测调度或 CLI；
- deployment package 目录、manifest 或内容复制；
- viewer pass、CMake、UI 与 capture schema。

evaluator、matched sampler 和可选训练 head 属于同一个 trainable/recipe，可用 phase 与 parameter group 表达；不再存在 `sampler_runner.py`。PT capability 只有在 runtime module 的 `sample/pdf` 与 evaluator 通过 correctness 后才声明。

当前 NVIDIA 实现迁入该定义。第二个“表达结构”只使用 test-scoped contract fixture，放在 `tests/fixtures/`，不进入产品 registry、CLI、配置或 viewer 默认列表；它用不同 tensor key、state stride 和 blob layout 证明公共层没有 NVIDIA 分支。

### 4. TrainingCheckpoint@2

训练 checkpoint 与 viewer 部署格式严格分离。当前项目仍以 PyTorch 训练，因此 checkpoint 可继续由公共 writer 使用 `torch.save()` 写入单个 `.pt`，但顶层 envelope 固定且版本化：

```text
format_name / format_version
method_key / method_descriptor_sha256 / implementation_identity
training_config / training_config_sha256
data_source_identity / source_contracts
step / phase / selection evidence
model_state        validated tensor mapping
optimizer_state / scheduler_state / scaler_state
rng_state
```

只有公共 `CheckpointWriter/Reader` 执行文件 I/O、sidecar hash、atomic replace 与恢复校验。方法只提供/消费 tensor mapping；不能把 Python callable、module object 或方法自定义 reader 放入 checkpoint。viewer 永不读取 `.pt/.pth`。

### 5. ScatteringPackage@1：统一 viewer 部署格式

废弃 `MethodBundle v1` 和 method-specific compiled set。统一部署容器暂定 `ncls.scattering-package@1`，使用可直接审计的目录格式：

```text
scattering-package/
  manifest.json
  runtime/
    program.slang
    modules/...
    blobs/...
  materials/
    <material-asset-id>/
      blobs/...
      resources/...
  validation/
    parity.json
  provenance/
    source.json
    training.json          # reference package 可省略
```

所有方法和 reference 使用相同 manifest/blob schema。私有 layout 通过 schema ID、dtype/stride/alignment/shape/usage 描述，不通过方法名解释。

身份独立计算：

- `program_runtime_id`：runtime descriptor、ABI、module、共享权重与 capability 的 canonical hash；
- `material_asset_id`：source asset identity、compiler identity 与私有 material payload 的 canonical hash；
- `package_id`：runtime ID、所含 material asset ID、validation/provenance 的 canonical hash。

换材质不能改变 `program_runtime_id`。同一 runtime 可以包含一个或多个 compiled material asset，不重复共享权重。

package 内 method-specific Slang module 是真实加载源，不再与仓库预编 shader 作 hash 对照。Falcor `ProgramDesc` 可接受 package 内绝对 shader 路径；module 可以包含 package-local 文件，只依赖版本化的稳定 host scattering ABI。viewer CMake 只编译 host/integrator，不列出 NVIDIA 或未来方法源码。

### 6. ScatteringBinding：reference 与 neural 的等价替换

公共 loader 对 LayerStack/OpenPBR/MERL/MaterialX reference package 与 neural package 执行完全相同的步骤：

1. schema、safe relative URI、大小和全部 SHA-256；
2. platform、Slang、host ABI 与 capabilities；
3. runtime module 从 package 路径编译；
4. 根据通用 blob descriptor 创建 buffer/texture；
5. 按同一 source material identity 选择 material asset；
6. 构造 `ScatteringBinding`；
7. 执行 package parity probe；
8. 成功后交给任一 viewer slot。

`ScatteringBinding` 只暴露公共 host 信息：program/runtime/material identity、capabilities、resource descriptors 和 specialization handle。`CompiledMaterial` 与 `ScatteringState` 仍是 Slang associated type/私有 blob，host 不解释字段。

reference 与 neural runtime 都实现现有公共语义：

```text
Backend.prepare(context, CompiledMaterial) → State
State.evaluate(wi, rng) → linear f
State.sample(rng) → direction, f/weight, pdf, event
State.pdf(wi) → forward/reverse density in declared measure
```

LayerStack、OpenPBR、MERL、MaterialX 等 source-family 分派只存在于各自 reference program/registry，不进入 integrator、loader 或 slot。

### 7. Viewer renderer 与 slot

viewer 持有两个相同的 `ComparisonSlot`：

```text
ComparisonSlot
  selection: ScatteringBinding
  renderer_mode: PT | deferred
  status: unloaded | compiling | ready | unsupported | unavailable | error
  output/raw/accumulation/timing
```

每个 slot 使用同一 host controller。默认 preset 可以是 `reference/PT | NVIDIA/PT`，另提供 `reference/PT | NVIDIA/deferred`；默认值不产生左右角色。

#### 单一 PT integrator

把 `ReferencePathTracer.cs.slang` 的 scene transport 拆成一个 `ScenePathIntegrator.cs.slang`，对 slot 的 binding 编译 specialization。只允许这一份 path loop，至少统一：

- primary rays/camera、scene intersection、emission；
- direct-light visibility、environment、bounce depth 与 Russian roulette；
- 线性 radiance 输出、sample accumulation 与 RNG identity；
- 对 scattering `evaluate/sample/pdf` 的调用和 measure 转换。

reference 的随机性、方法 matched sampler 或不支持 MIS 的情况由 capability/sample event 表达，不能按 implementation ID 分支。PT mode 要求 `prepare/evaluate/sample/pdf` 与 PT-compatible capability；能力不足时 slot 显示不可用，不能借用另一 binding 的 sampler。

#### 单一 deferred renderer

把现有 `Prepare.cs.slang`/`Approximation.cs.slang` 收敛为一个公共 deferred renderer 生命周期，可按语义拆成 prepare/evaluate 两个 pass，但代码只依赖 scattering ABI。任何 slot/binding 都可实例化；模式至少要求 `prepare/evaluate`。

#### 固定尺寸合同

- 删除 draggable split 及其 UI、state、replay/capture 字段和 shader remap。
- `panel_width = floor(output_width / 2)`；两侧都渲染为完全相同的 `(panel_width, output_height)`。奇数总宽度剩余的一个像素作为固定 divider/background，不分配给任一 viewport。
- camera aspect 永远为 `panel_width / output_height`，只由窗口尺寸决定，与 selection、mode、capability、编译/加载状态无关。
- 任一 slot 失败时在原 viewport 内显示稳定错误卡/overlay；另一侧不 resize、不铺满、不改变 projection。
- composite 只做 1:1 texel 对应与 tone mapping，不做非等比 UV 重映射。

capture/replay 以 `slots[2]` 记录 package/runtime/material ID、mode、capability、status、integrator、raw output 与 timing；不再使用 `reference`/`approximation` 和 split 字段。difference 只在两个 slot 均 ready、source identity/measure/display contract 可比较时生成。

#### Source 编辑事务与 slot 失效

viewer 只持有一个当前 canonical source snapshot，两个 slot 的 reference/neural binding 都从它派生。通用 editor 提交 patch 后，family 先原子产生新 snapshot/state ID，再由 slot controller 独立调用每个 binding definition 的 adaptation contract。reference 对所有合法 family edit 重绑定；neural binding 按 `unchanged/runtime-patch/recompile/unsupported` 处理。

用户已确认 source edit 不受当前 method 能力阻塞。需要重编译且已启动可用 compiler 的 slot 进入 `compiling`；不支持或当前只能离线重编译时进入带原因的 `unsupported`。旧 neural 图像可选作为明确标记的 stale preview 保留，但不得进入有效比较、difference 或统计；只有 source identity/schema/hash/parity 验证通过的新 asset 才能原子换入并恢复 `ready`。

## 迁移与删除

### Pipeline 保留并迁移

- LayerStack random-walk、OpenPBR、MERL、MaterialX 四个 ground-truth reference 的原生语义、资源、source asset identity、registry/provenance 与既有 parity 证据；
- NVIDIA neural appearance 的表达模型、论文对应信息和当前 diagnostic checkpoint 历史 provenance；不为旧 checkpoint 保留 reader。
- 通用 canonical JSON/hash/atomic write、FP16 packing、loss 与采样数学中真正共享的函数；
- Falcor overlay/build 边界与锁定上游提交。

### Pipeline 外 pbrt 边界保护

- `external/pbrt-v4/` 保留锁定上游且维持 clean；
- `tools/reference/pbrt_probe/` 和 `pbrt_compare.py` 保留为长期两界面 crosscheck 工具，不抽成公共 runner；
- `references/pbrt-coated-crosscheck-v1/` 只保留身份、锁定版本、查询语义和适用范围记录，不是 deployment package；
- 单次 crosscheck 输出仍进 `artifacts/`；
- 本任务只在新 LayerStack 正式 API 破坏现有 compare tool 调用时做最小边界更新，不改其 role、schema 或执行模型。

### 递归删除

- Film、analytic residual、per-state teacher、lobe residual、unified candidate 等废弃 neural method 的产品注册、模型、配置、CLI、shader、测试和稳定文档；
- legacy LTC/analytic control 等旧 approximation/backend 身份与接线；若其底层数学仍被 NVIDIA/reference 调用，只把通用函数迁到中性模块，不保留旧方法目录或产品 identity；
- 独立 sampler config/runner/CLI；
- unified/NVIDIA 专用 exporter、Slang session、parity 入口中可由公共合同替代的部分；
- mollification 专用 collector/budget/lock/store/reader/CLI 和 v2…v8 配置链；有用的 query/target 语义迁成统一 recipe 后删除原实现；
- `MethodBundle v1`、compiled-set wrapper、LayerStack preview 固定字段和 C++ method shader 枚举；
- 旧 checkpoint/MethodBundle/CLI/capture/replay 的全部 schema、writer/reader、alias、converter、version/format 探测、fallback 和兼容测试；旧调用方必须先迁到新合同，不留隐藏双轨；
- reference-only scene path loop、按 family 硬编码的 viewer dispatch、right-only approximation controller、split 相关全部状态；reference 实现本身迁入 package/binding 而不是删除；
- stable specs/docs 中把旧方法或旧 bundle 写作当前主线的内容。

不改写已归档历史报告的原字段；旧活动任务在本任务落地时标为 superseded，不再作为执行入口。Git/仓库外 artifacts 是唯一历史追溯机制，产品代码不为读取历史产物保留兼容层。

## 测试策略

### 静态合同

- 产品 method registry 只有 NVIDIA；test registry 可注入 contract fixture。
- 对通用 runner/exporter/viewer 源码做禁止 method ID/LayerStack 字段的结构检查。
- 对通用 viewer editor 做禁止 source family ID、固定 LayerStack/OpenPBR/MaterialX field/offset 的结构检查。
- package/checkpoint schema、identity、safe URI、hash、ABI、capability 和私有 blob descriptor 测试。
- 全仓库 reachability 清单确认旧方法没有注册、CLI、构建或稳定文档残留。

### 数据与训练

- offline/live producer 对同一固定 query 的字段、shape、dtype、measure 与 seed identity parity。
- Falcor shared output `to_torch()` 返回 CUDA tensor；训练一步中不存在 `to_numpy()`/HDF5；loss、gradient、optimizer step 有限且可恢复。
- 同一 `TrainingRunner` 分别跑 offline smoke 与 live smoke；NVIDIA 和 test fixture 都不要求改 runner。
- checkpoint roundtrip、tamper rejection、method tensor schema rejection。

### runtime 与 viewer

- LayerStack/OpenPBR/MERL/MaterialX reference、NVIDIA 与 test fixture package 由同一 writer/loader 通过；package 内 shader 在 viewer 未预编该 program 源码的情况下加载。
- 四个 viewer-ready reference 与 NVIDIA 按各自权威 oracle 对固定方向通过 Python/Slang/Falcor packed parity 或既有 image parity；数值 tolerance 在 formal 前依据 dtype/oracle 冻结。
- 同一 PT shader source 分别实例化四个 pipeline reference 与 NVIDIA，验证 `sample/pdf/evaluate` 有限性和 capability。pbrt 不进该 matrix；若本任务修改了它的最小调用边界，另行运行现有 pbrt compare smoke 确认工具未被破坏。
- slot 组合：reference/reference、reference/NVIDIA、NVIDIA/reference、method/method；mode 组合：PT/PT、PT/deferred、deferred/PT。
- 通用 editor 在无 family-specific UI 分支下覆盖 LayerStack 列表/变体编辑、OpenPBR typed 参数、MaterialX constant/connected 边界和 MERL 空参数面；验证 canonical state ID、patch conflict/约束拒绝、reference rebind 和 neural adaptation 状态。
- 加载失败、包被移除、hash/ABI/shader 错误、能力不足、窗口 resize、奇数宽度时验证 extent、camera aspect 与 1:1 composite。
- capture/replay roundtrip 验证两个 slot 的独立 selection/mode/status，旧 schema 被拒绝而非兼容读取。

## 风险与 rollback point

1. **Falcor/CUDA interop**：`to_torch()` 已存在，但 Shared flag、同步和 tensor lifetime 必须先用 task-local smoke 证明。若锁定构建缺 CUDA，任务回到 planning，不以 CPU copy 冒充 online。
2. **package shader 动态加载**：Falcor 接受绝对 shader 路径，但 package-local include 与稳定 host ABI 必须做最小 probe。失败时允许调整 package module closure，不恢复 CMake 枚举。
3. **reference scattering contract**：LayerStack 多界面随机游走的 marginal PDF 目前不完整；通用 sample event/capability 必须准确表达无 MIS 路径，不能伪造 PDF。若无法满足 PT correctness，应修 reference sampling 合同，而不是复制 integrator。
4. **大规模删除**：先建立新合同 vertical slice 和 reachability matrix，再一次性切入口并删除旧路径；每个阶段保留可回退提交点，但最终工作树不保留双轨。
5. **NVIDIA 忠实性**：本任务只保证其迁移后的结构与现有 diagnostic 语义不被静默改变，并交付真实 online lifecycle smoke；论文规模训练与最终质量结论仍属后续研究。
