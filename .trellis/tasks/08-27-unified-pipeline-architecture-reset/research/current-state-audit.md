# 当前架构审计：统一 Pipeline 重整

## 结论

当前仓库已经具备“某些方法可以从训练走到 viewer”的端到端能力，但尚未形成可稳定扩展的单一 pipeline。新增方法仍需要同时修改 Python 注册、训练配置、训练 runner、compiled-set 导出器、Slang session、MethodBundle 内容、viewer CMake 和 viewer runtime shader 树。数据侧还存在一套独立的 mollification 采集、manifest、reader 与训练入口。它们不是统一接口下的策略实现，而是并行 pipeline。

因此，本次重整不能只做命名整理或把两个 exporter 合并。需要先冻结跨层合同，再把现有方法迁入唯一实现，最后递归删除旧入口与旧语义。NVIDIA 方法只作为合同回归样例；其论文忠实性不作为架构验收的替代品。

viewer 的长宽比问题也不是单点 shader 错误，而有两个独立根因：

1. 方法扫描失败后，活动状态是在替换方法列表之后计算的，旧的 half-width 资源可能不会触发 resize；随后半宽 reference 被铺满全窗，必然拉伸。
2. 当前 composite 允许拖动 split，但左右渲染目标始终按固定半窗宽度生成；当 panel 宽度偏离 50% 时，shader 直接重映射 UV，同样会拉伸。

## 已确认的当前结构

### 方法注册与训练

- `src/ncls/learning/pipelines/__init__.py:3` 同时注册 lobe residual、P1 evaluator、unified neural 与 NVIDIA 四组 pipeline。
- `src/ncls/learning/pipelines/p1_evaluator.py:324` 注册七个旧实验身份；`src/ncls/learning/pipelines/lobe_residual.py:132` 注册三个旧占位身份。这些仍是可达公共入口，不只是归档代码。
- `src/ncls/learning/runner.py:102` 的 implementation identity 只认识 `unified-slang-core-v1` 与 `nvidia-neural-appearance-slang-v1`，并为两者维护两份源码清单。新增方法必须修改通用 runner。
- 普通训练使用 `TrainingConfig`/runner；matched sampler 使用独立的 `SamplerTrainingConfig`、`sampler_runner.py` 和 CLI。`src/ncls/learning/sampler_config.py:10` 直接枚举 NVIDIA 与 LTC sampler，`sampler_runner.py` 又依赖具体模型类型。
- `src/ncls/cli.py` 暴露 method-specific 的 `export-unified-compiled`、`export-nvidia-compiled` 及若干单方法 audit 命令。CLI 仍在表达内部实现身份，而不是统一能力。

### 导出、bundle 与 Slang runtime

- `src/ncls/learning/unified_artifacts.py:187` 与 `src/ncls/learning/nvidia_neural_artifacts.py:191` 分别实现 compiled-set 导出、布局、打包与 runtime adapter。
- `src/ncls/learning/slang/session.py` 直接硬编码 unified、NVIDIA core 与 NVIDIA matched-LTC shader 路径，并提供三套专用 session/hash 逻辑。
- `src/ncls/bundle/compiled_set.py:11` 直接依赖 `MaterialProgram`、`canonicalize_layer_stack` 与 `pack_layer_stack`；`src/ncls/bundle/compiled_set.py:96` 又把 preview 固定解析为 LayerStack。因此所谓公共 compiled set 实际绑定了当前源材质族。
- `apps/viewer/CMakeLists.txt:28` 显式列出 NVIDIA、unified 与 analytic backend 源文件。新增方法仍需要改 CMake 并重编 viewer。
- `apps/viewer/MethodBundle.cpp:133` 要求 bundle 中声明的 shader module 必须已存在于 viewer runtime shader tree，并校验两者 hash 相等。当前 bundle 不能把未知新方法作为自包含模块交给通用 loader。

### 数据采集与未来 online training

- `src/ncls/data/contract.py:302` 已有可复用的 `ReferenceProvider` 语义合同：source state、surface sample、query plan 与 evaluate。
- `src/ncls/data/collector.py:192` 已有通用 `collect_reference_dataset()`，但 corpus orchestration 在 `src/ncls/data/corpus.py:202` 等位置仍是 LayerStack-specific。
- `src/ncls/data/mollification.py`、`mollification_collection.py`、`src/ncls/cli.py:139` 到 `:241`、`src/ncls/learning/data.py:196` 等位置形成了独立的 mollification 采集、预算 schema、collection lock、reader、curriculum store 与训练入口。
- 当前 Falcor reference evaluator 通过 GPU buffer `to_numpy()` 再进入 HDF5/CPU 数据路径。它可以作为 offline collector 的执行器，但不是实际的 GPU-resident online training transport。

这说明公共语义合同可以保留，但执行与传输合同必须补齐：训练应只消费一种 batch 语义；持久化 corpus reader 与 live reference executor 是该合同的两个 producer；offline 采集则是同一 stream 接一个 shard sink，而不是第三套采样实现。

用户已确认本次必须交付真实的 GPU-resident online training 最小链路。LayerStack 与 Falcor 必须按正交层次建模，不能写成“LayerStack/Falcor adapter”：

- LayerStack 是当前已实现的 **source family**；其原生状态、资源与随机游走 reference 定义 GT 语义。
- Falcor 是承载 GPU reference pass、buffer 和调度的 **execution backend**；它不定义材质语义。
- live batch source 组合一个 source-family reference program 与一个 execution backend。未来换 source family 不应复制 trainer；未来换执行后端也不应改 source-family 语义。

验收链路应表述为：

```text
SourceFamilyReferenceProgram（当前实例：LayerStack random-walk reference）
  + ReferenceExecutionBackend（当前 GPU 实例：Falcor）
  → LiveReferenceBatchSource
  → device tensor batch
  → 统一 TrainingRunner
```

该链路不得经过 `Buffer.to_numpy()`、HDF5 或 CPU replay。未来新增 source family 只实现 reference adapter，不新增 online trainer。

### checkpoint、runtime 与材质资产

当前文档把 `MethodBundle` 定义为 viewer 的部署交付物，这个目标有必要：C++ viewer 不应依赖 Python、PyTorch、optimizer state 或训练目录。但当前实现把三个不同身份混在一个包里：

1. `TrainingCheckpoint`：训练恢复、选择和评测所需的 model/optimizer/RNG/config/provenance。
2. `MethodRuntime`：某个冻结方法 checkpoint 共享的 shader module、共享权重、ABI、capabilities 与 cost contract。
3. `CompiledMaterialAsset`：某个源材质由该方法编译出的 latent、参数表、纹理/资源与 source provenance。

具体证据：

- `src/ncls/learning/training/checkpoint.py` 用 `torch.save(dict(payload))` 保存训练 checkpoint。`.pt/.pth` 只是 PyTorch/Pickle 容器，不是跨语言部署 ABI；它可以承载 optimizer 和任意 Python 对象，C++ viewer 不应直接解释。
- `src/ncls/bundle/compiled_set.py:114` 同时复制共享权重、compiled material table、backend shader、单个 preview material 和 parity。
- `src/ncls/bundle/compiled_set.py:137` 把 `compiled_state_id` 与所有内容 hash 纳入 `method_id`，导致同一训练方法换一个材质也成为另一个 method identity。
- `apps/viewer/MethodBundle.cpp:90` 把 loader 固定到 LayerStackIR；这不符合跨 source-family 边界。
- `src/ncls/bundle/compiled_set.py:117` 把 shader 复制进 bundle，但 `apps/viewer/MethodBundle.cpp:133` 又不从该副本加载，而是要求 viewer 源码树已有同名 shader 并 hash 相等。这个 shader 副本在当前实现中确实冗余，而且新增方法仍需重编 viewer。

第一性原理下应保留的是“统一、可校验、viewer 可直接读取的部署容器”，而不是当前混合语义的 `MethodBundle v1`：

- 所有文件 I/O 由公共 writer/reader 拥有。方法实现只把训练状态导出为 tensor mapping，并把 runtime/material 私有数据编译成 typed byte blobs；方法不得各写一套路径、JSON 和复制逻辑。
- 训练 checkpoint 使用一个版本化 envelope。方法的 tensor key/shape 可以私有，但必须由 method descriptor/schema 校验；viewer 永不读取它。
- 部署容器使用一个版本化 manifest 与 blob 格式，内部明确分出 `runtime` 和一个或多个 `material asset`。viewer 只看公共 manifest、ABI 与 blob descriptors，不解释 tensor 名和方法私有布局。
- `method_id` 只标识 method runtime；`material_asset_id` 单独标识 compiled material；最外层 `package_id` 可以组合二者，方便一次选取和完整性校验。
- backend shader/module 必须真正由部署容器提供并由 viewer 动态加载，或完全属于稳定 host；不能一边复制、一边要求预编进 viewer。

因此，`bundle` 不是必须保留的术语。若继续使用，它只表示上述统一部署容器；当前 `MethodBundle v1` schema/exporter/loader 需要被替换，而不是在其上继续加字段。

用户已确认采用上述三层产物边界，并明确 viewer 不直接读取 `.pt/.pth`。

### viewer 的 PT、deferred 与尺寸合同

- `apps/viewer/NclsViewer.h:192` 只有一个 `mpReferencePathPass`。右侧只有 `mpPreparePass` 和 `mpApproximationPass`，没有 method PT pass 或对应资源。
- `apps/viewer/NclsViewer.cpp` 的 frame render 顺序是 reference PT，然后右侧 prepare/evaluate deferred，再 composite。现有 viewer 不能满足“左右都是完整 PT，右侧另有 deferred”。
- `apps/viewer/NclsViewer.cpp:506` 在方法活动时把 `mViewWidth` 设为输出宽度一半；reference 与右侧纹理共用这个尺寸。
- `apps/viewer/NclsViewer.cpp:681` 用 `mViewWidth / mOutputHeight` 设置 scene camera aspect，投影与当前方法是否可用耦合。
- `apps/viewer/NclsViewer.cpp:831` 扫描时先整体替换方法列表；`selectMethod()` 到 `:915` 才读取 `hasActiveMethod()`。旧方法被拒绝后，函数看不到替换前的活动状态，可能跳过必要 resize。
- `apps/viewer/shaders/Composite.cs.slang:38` 按拖动后的 split 重映射固定半宽纹理，没有 fit/crop/letterbox 合同，因此非 50% split 即使方法正常也会拉伸。
- 当前 viewer 测试没有覆盖活动方法失效、bundle 重扫、窗口 resize、split 改变、右侧无输出与 camera aspect 的组合。

### source 参数编辑当前也是平行实现

- `apps/viewer/NclsViewer.cpp:1224` 与 `:1341` 分别手写 OpenPBR/MaterialX UI；OpenPBR 直接按 77-float offset 改写，MaterialX 直接按 24-float offset 改写并用固定 flag 判断 texture connection。
- `apps/viewer/NclsViewer.cpp:1399` 中 LayerStack 又是第三套 UI，直接操作 `LayerStackIR`；它不仅编辑 scalar/color，还增删、重排 coat 并替换 base variant。
- `apps/viewer/ReferenceSource.h:23` 把 LayerStackIR、MERL table、77-float OpenPBR 和 24-float MaterialX 全部并在一个 host struct，viewer 因此同时知道 family identity 和私有 runtime layout。
- Python source 层已有可迁移的原生语义：`OpenPBRMaterial.with_parameter()` 保留 Constant/Texture/Graph/Geometry binding；`LoadedMaterialX.editable_inputs()`/`set_input_value()` 只允许未连接的 value input，显式拒绝静默替换 connected graph input。
- `docs/contracts/viewer_scene.md` 已要求 LayerStack 保存完整可编辑 material program、OpenPBR 保存原生命名 resolved inputs、MaterialX 只保存可编辑 constant override。新架构应把这些 family-owned 语义收敛到统一 editor lifecycle，而不丢掉它们。

因此“公共层不看到族私有字段”应精确表达为：公共 runner/renderer/UI 不理解这些字段的物理语义或 runtime offset；source family 仍必须通过版本化 typed parameter tree/patch contract 把原生可编辑面完整对外暴露。该树可以描述 LayerStack 列表/variant、OpenPBR binding provenance、MaterialX constant/connected 状态和 MERL 空编辑面，但不是所有 source 归约到的 material IR。

### pbrt coated probe 是 LayerStack pipeline 之外的独立 oracle

- `references/registry.json` 用 `source_material_family_id = ncls.layer-stack@1` 和 `independent-validation` 记录 pbrt probe 对 LayerStack 的适用关系。这个 provenance manifest 不意味着 pbrt 是训练/viewer pipeline 的另一 runtime implementation。
- `references/pbrt-coated-crosscheck-v1/README.md:15` 和 `tools/reference/pbrt_compare.py:75` 证明其原生对应是两界面 slice：rough-dielectric top + homogeneous slab + diffuse/rough-conductor base。它可独立检查界面透射、介质吸收/散射、多次反射与方位语义，但不是任意 N 层 GT。
- `tools/reference/README.md` 与 `.trellis/spec/project/code-organization.md` 明确 `tools/reference/` 就是长期 reference 验证工具的正式位置。`pbrt_compare.py` 的 `ProbeCase`、direction/batch/seed、统计和 CLI 是这个专用 oracle 的封闭语义，不是应当迁入通用训练/viewer pipeline 的平行基础设施。
- 现有路径已符合仓库所有权：上游在 `external/pbrt-v4/`，project probe/compare 在 `tools/reference/`，身份与适用范围在 `references/pbrt-coated-crosscheck-v1/`，输出在 `artifacts/`。本任务不应新增 adapter/runner/package/viewer 接线；只在 LayerStack 正式 API 迁移破坏其调用时做最小边界维护。

用户进一步明确：source-family reference material 与 neural method 对 viewer 必须实现相同接口，neural method 是 reference 的“等价替换”。当前实现不满足：

- neural backend 已实现 `shaders/ncls/contracts/scattering_backend.slang` 的 `INclsScatteringBackend`/`INclsScatteringState`，调用面为 `prepare/evaluate/sample/pdf`。
- reference 没有实现这个 backend 合同。`apps/viewer/shaders/ReferencePathTracer.cs.slang:300` 起直接按 source family 分派 evaluate，`:523` 起另行分派 sample，`:587` 起另行分派 pdf，并在同一文件中实现完整 direct lighting 与 path loop。
- 因此 reference PT 不是“同一 integrator + 不同 scattering implementation”，而是一条独立 renderer。若再新增 method PT pass，会形成第二/第三套 scene transport，实现必然漂移。

正确的 viewer 边界应是：

```text
Source material identity
  ├─ Reference compiler/loader → Reference material binding
  └─ Neural compiler           → Compiled neural material binding

两种 binding 都实现 IScatteringProgram
  prepare(context, material) → private state
  state.evaluate(wi)
  state.sample(rng)
  state.pdf(wi)

同一个 ScenePathIntegrator(program binding)
  ├─ 左槽默认绑定 reference
  └─ 右槽默认绑定 neural method
```

“相同接口”指 host ABI、方向/measure、生命周期和 renderer 调用完全相同，不要求 reference 与 neural method 使用相同的 `CompiledMaterial` 或 `ScatteringState` 字段。二者私有布局仍由 associated type/blob descriptor 隔离。

等价替换还要求两侧从同一个 source material identity 派生：reference binding 持有原生 source asset，neural binding 持有该 source asset 的编译结果；viewer 必须校验 provenance 对应，而不是用两个无关预览材质。

用户已确认 viewer 的两个 comparison slot 在 UI 和 host 层也完全对称：每侧都可选择 reference 或任意 neural method。默认左 reference、右 NVIDIA 只是 preset，不是硬编码角色。这使 reference/reference、reference/method 与 method/method 都成为统一 renderer 的直接回归证据。

用户已确认删除可拖动 split，固定为严格 50/50 comparison layout。实现时应同步删除 split UI、replay/capture schema 字段和 composite UV remap；它们不能作为隐藏兼容路径保留。方法加载失败时仍保留两个 viewport 与原 camera projection，失败侧显示错误状态，不把成功侧铺满全窗。

用户已确认 renderer mode 也按 slot 对称：每侧独立选择公共 PT 或公共 deferred renderer，由 binding capability 决定模式是否可用。实现中只允许存在一个 PT source 和一个 deferred source，它们分别对任意 `IScatteringProgram` specialization 实例化；不保留 right-only approximation pass。

## 新方法当前需要修改的位置

| 层 | 当前接线 | 统一后应保留的扩展点 |
| --- | --- | --- |
| 方法语义 | Python pipeline 注册、具体 model 类型判断 | 一个 method descriptor/factory，声明训练表达、runtime capabilities 与版本身份 |
| 数据 | 普通 corpus 与 mollification/未来 online 各自组织 | 一个 query/batch 合同；offline reader 与 live executor 只是 producer |
| 训练 | 普通 runner 与 sampler runner 分离 | 一个训练 orchestration；objective、可选 sampler head 等由方法组件声明 |
| 导出 | 每方法 exporter 与 packing | 一个 bundle compiler；方法只提供参数 schema/packer 和 shader module |
| Slang | 每方法 session class 与硬编码路径 | 由 descriptor/module manifest 创建通用 compile/load session |
| viewer build | CMake 枚举 backend 源码 | viewer 只链接稳定 host ABI；方法模块由 bundle/registry 装载 |
| viewer render | reference PT 与 method-specific deferred | 一套 scene PT 骨架，两种 scattering adapter；deferred 是单独模式 |
| UI | 按方法接能力与资源 | UI 根据 capability 自动启用 PT/deferred，不识别具体方法名 |

## 应冻结的统一合同

### 1. Source family 与 reference

- source family 拥有原生 source state、资源、参数编辑与权威 reference；公共层不得要求所有材质先归约为 LayerStack 或 closure。
- reference query 的方向、measure、线性输出与随机性必须显式；同一合同可由 offline collector 和 live executor 使用。
- source-family preview/编辑 schema 属于 source adapter，不属于 method bundle 的固定 LayerStack 字段。公共 editor 只看 typed parameter tree 与 patch operation，family 拥有约束验证、canonical source state 和资源/图连接语义。

### 2. Training batch source

- 训练 runner 只消费统一的 typed batch，不能知道 batch 来自 HDF5、live Falcor 或未来其他 reference。
- batch 必须携带 source state/latent 输入、surface/view/query、target、measure、采样 PDF/权重及可追溯身份；具体字段按能力分组，不能靠方法名分支。
- persisted producer 与 live producer 必须共享确定性、sharding、seed 和 sample identity 语义。
- 本次必须实现真实 GPU-resident producer，首个执行 adapter 使用当前 LayerStack/Falcor reference；公共合同与 runner 不得依赖 LayerStack。

### 3. Method plugin

- 新方法的最小实现应限于：训练侧 expression/objective 组件、参数与 state schema、runtime shader module，以及 capabilities 声明。
- pipeline registry、CLI、训练循环、评测、bundle 编译、viewer loader 和 UI 不得按方法身份分支。
- `prepare/evaluate` 是 deferred/evaluator 的基本能力；PT 模式另要求与 evaluator 匹配的 `sample/pdf`。能力不足的方法可以进入 deferred，但不能伪装成 PT。
- method 可选实现 source parameter adaptation，但只能通过公共 `unchanged/runtime-patch/recompile/unsupported` 结果进入 pipeline；不得为它在 viewer 新增方法专用 UI 或把私有参数展平成全局固定向量。
- 静态执行预算、固定读取数和 backend-specific `ScatteringState` 仍是部署合同；不能为了统一而暴露某个 backend 的 state layout。

### 4. Bundle 与 runtime

- 训练 checkpoint、共享 method runtime 与 per-material compiled asset 必须具有不同身份和生命周期；可以由一个统一部署容器共同交付，但不能混成一个 `method_id`。
- 部署 manifest 应描述 method module、capabilities、ABI/version、参数/blob schema、material asset、资源与完整 hash；公共 loader 不解析具体方法或 LayerStack 内部结构。
- host ABI 与方法 module 分离。viewer 的新增方法流程不得修改 C++、CMake、RenderGraph 或 UI；包内 shader 不能只是仓库 shader 的冗余副本。
- 所有 checkpoint 与部署文件 I/O 都由公共 writer/reader 拥有；方法自定义部分只返回受 schema 约束的 tensor mapping/typed blobs，不是第二个 exporter。

### 5. Viewer render modes

- source-family reference 与 neural method 必须实现同一个 renderer-facing scattering program 合同；neural method 是 reference binding 的等价替换，不能拥有 method-specific pass。
- 主比较模式：左侧 reference PT，右侧 method PT。两侧使用同一个 scene integrator 实现、camera、scene、采样预算、tone mapping 与尺寸合同；只替换 material/scattering binding。
- 次级模式：左侧 reference PT，右侧 method deferred。deferred 不冒充 PT，capture/telemetry 明确记录模式和 integrator。
- 方法缺失、能力不足、加载失败或 shader 错误时，右侧显示稳定的错误/不可用状态；不得改变左侧渲染尺寸、camera aspect 或 composite 采样几何。
- panel layout、render extent 与 camera projection 必须解耦。固定 50/50 layout 下，两侧使用相同 extent；窗口 resize 不能产生非等比缩放。

## 递归清理范围

用户进一步明确：这里的“其它乱七八糟的方法”指废弃 neural/approximation 方法轨道，不包括 source-family reference。当前 reference registry 的迁移分类为：

| 对象 | 当前角色 | 本任务处理 |
| --- | --- | --- |
| `ncls.layer-stack-random-walk@1` | ground-truth，viewer-ready | 保留原生 random-walk 语义，迁入统一 source/reference/scattering/package/viewer |
| `ncls.openpbr@1.1.1` | ground-truth，viewer-ready | 保留 OpenPBR 原生参数/资源与 parity，迁入同一路径 |
| `ncls.merl-brdf@1` | ground-truth，viewer-ready | 保留测量表/参数化/标定，迁入同一路径 |
| `ncls.materialx-polyhaven@1` | ground-truth，viewer-ready | 保留原生图/纹理/颜色空间与 image parity，迁入同一路径 |
| `ncls.pbrt-coated-crosscheck@1` | LayerStack 两界面外部 oracle，viewer not-applicable | 不进新架构。保持既有 external/tool/manifest/artifact 边界，不误删、不误注册；仅必要时维护对迁移后 LayerStack API 的最小调用 |
| NVIDIA neural appearance | 唯一产品 neural method | 保留方法语义与诚实的 diagnostic 身份，完整迁入新 method/training/package/viewer 架构 |

以下废弃 neural/approximation 对象不能仅从默认配置隐藏，必须从注册、CLI、构建、源码、测试夹具和稳定文档中递归清除，除非某个底层数学组件确有 NVIDIA/reference 调用方；此时只迁移该中性组件，不保留旧产品身份：

- `film-evaluator-*`、`analytic-residual-*`、`per-state-teacher-*` 与 `lobe-residual-*` pipeline 身份及其配置、测试和文档。
- `src/ncls/core/representations/legacy_ltc_k2/`、`shaders/ncls/backends/legacy_ltc_k2/`、`lobe_residual/` 和只为旧方法存在的 control/backend 身份与接线；若 NVIDIA 的 matched sampler 仍复用某个数学函数，函数迁至 NVIDIA 或公共 sampling 模块后再删除旧目录。
- method-specific exporter、session、CLI、CMake/source 枚举与 runtime adapter。
- 旧 checkpoint、MethodBundle/compiled-set、CLI、capture/replay 的 schema、writer/reader、alias、converter、格式探测、fallback 和兼容测试；不保留双轨。
- mollification 专用采集/预算/lock/reader/CLI 的平行基础设施；仍有研究价值的采样分布或 target transform 迁为统一 pipeline 的策略组件。
- 稳定 spec/docs 中把 lobe residual、Film 或 legacy LTC 写成当前主线的陈旧规则。
- 已被新任务取代的旧任务文档不改写历史，但必须在任务状态和新设计中标明 superseded，避免继续作为执行入口。

## 架构验收证据

- 用至少两个表达结构不同的方法实例或一个方法加一个最小 contract fixture，证明新增方法不修改通用 runner、CLI、exporter、viewer C++/CMake/UI。
- 同一训练命令可切换 offline 与 live batch producer；除 producer 配置外，objective、runner、评测与导出路径相同。
- 同一 bundle loader 根据 capabilities 自动支持 deferred 和 PT；缺少 `sample/pdf` 时 PT 明确不可用。
- reference 与 neural method 分别实例化同一个 renderer-facing contract 和同一个 PT integrator；仓库不存在 reference-only scene transport loop 或 method-specific PT loop。
- 用同一 source material identity 构造 reference binding 与 compiled neural binding，替换 binding 后不修改 scene、camera、light、integrator、RenderGraph 或 display path。
- viewer 自动测试覆盖：无方法、有效方法、deployment package 被移除、hash/ABI/shader 失败、两侧独立 PT/deferred 切换、窗口 resize，以及任一侧输出缺失。每种情况下都验证 camera aspect 和显示像素不被非等比缩放。
- 同一通用 source editor 覆盖 LayerStack 结构编辑、OpenPBR typed input、MaterialX connected read-only 与 MERL 无连续控件；新 source state 使 reference 正确 rebind，neural method 按 adaptation contract 进入可验证状态。
- 删除清单以全仓库可达性搜索、注册表快照、构建输入和文档检查共同验收，不能只依赖单元测试通过。

## 产品决策状态

source parameter editor/adaptation 已纳入公共合同。用户已确认 source edit 不被 neural 能力阻塞；reference 立即改用新 state，neural slot 按 `compiling/unsupported` 显式失效，旧 output 不进入有效比较，新 asset 验证后原子换入。用户拥有的范围、产物边界与 UX 决策已全部收敛。
