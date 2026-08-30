# vMaterial Metal 神经材质系统设计

## 1. 设计目标

本设计把 Metal-v1 编译成同一个静态有界的 neural material program，同时保留三类彼此独立的用户语义：

```text
MetalOpticalIdentity
× FinishNeuralTextureBundle
× NativeTypedParameterState
→ MetalMaterialInstance
```

系统以 vMaterials 2 Metal 的原生 MDL program、typed arguments 和 source textures 为权威 reference。目标不是把 692 个 opaque exports 反演成统一层模型，也不是为每个 preset 训练一个黑盒；目标是在 source-backed 组合域内，用共享方法表达 13×7 标准矩阵与 family-local 特殊 recipe，并允许替换兼容的 neural texture bundle 和编辑原生 typed 参数。

首个研究方法采用“质量优先完整形态”：把可互补的 texture codec、typed compiler、analytic prior、warped direction field、residual evaluator 与 matched sampler 同时完整实现。本任务在 Windows 上验证完整方法的执行、梯度、短程收敛与部署闭环，并交付 Linux 单 GPU 长训练版本；长训结束后先审阅效果，不自动进入 formal、消融或 Pareto。

## 2. 设计边界

### 2.1 纳入范围

- 692 个 opaque authored exports、178 个 opaque compiled graph identities 和 52 个 opaque texture-set identities；
- 标准 13 metal × 7 finish 的 source-backed 组合；
- 特殊 patina、aging、paint、crack、anodization、structure 等 recipe 的 native family-local 参数域；
- source-derived texture bundle 的随机访问压缩、替换、mip/filtering 和 provenance；
- 64 套 family-local typed schema 中 opaque 参数的完整编辑语义；
- 完整 `prepare/evaluate/sample/pdf` 与对应训练、checkpoint、package、Slang 和 viewer 路径。

### 2.2 不纳入范围

- 145 个使用 cutout opacity 的 exports；
- 任意外部 texture set 导入、自动 channel-role 推断或未知 schema 的 zero-shot authoring；
- 无 source reference 的任意 overlay、coating、structure 重排与跨 family 参数搬运；
- 把 source graph 归约成 LayerStack、Principled 或固定 closure vocabulary 后再作为 GT；
- 用每个 preset 的自由 latent 替代 pure typed compiler；
- Windows 上的 full convergence、Linux multi-GPU DDP、环境积分、UE 工作流以及长训后的自动 formal/消融/Pareto。

## 3. 身份、兼容性与交付物

### 3.1 `RecipeSignature`

`RecipeSignature` 是内部兼容性身份，不是第四个用户可组合轴。它至少包含：

- source package、module 与 exact export family；
- graph identity、typed schema identity 与 bundle schema identity；
- spatial access program identity；
- 支持的 metal/finish/recipe 域与 capability mask；
- source locator、资源依赖和版本指纹。

它负责 fail closed：标准矩阵允许有 reference 依据的 metal/finish 配对；特殊 recipe 只允许其 native family-local 组合。模型可以对域外组合产生数值，但 runtime 和评测不得把它标记为受支持语义。

### 3.2 交付物分层

系统交付物分为四层：

1. `MetalSourceRegistry`：692 个 opaque export 的 locator、recipe、schema、graph、texture-set 与资源清单；
2. `MetalMethodPackage`：共享 texture decoder、typed compiler、angular feature bank、hybrid evaluator、profile 和 schema/recipe tables；
3. `FinishNeuralTextureBundle`：某个 texture-set identity 的量化 multiscale grids、bounded asset adapter、bundle metadata 与 provenance；
4. `MetalMaterialInstance`：`MetalOpticalIdentity`、native typed arguments、选中的 bundle identity，以及 compiler 生成的 `MaterialProgramState`。

共享方法和资产必须分开统计与寻址。更换兼容 bundle 只进行 compatibility check、资源绑定和 material-state 重新生成，不重新训练 evaluator、不重新编译 shader，也不重新编码未变化的资产。

### 3.3 ABI 与 profile

每个可部署 profile 冻结以下静态上限，并写入方法身份：

- texture domain/level 数、每级 high/low grid channel 数和固定 gather 数；
- structured spatial state 维度、learned frame 数和 lobe slot 数；
- typed token 上限、compiler 层数、FiLM/LoRA rank；
- angular-bank level/rank/read count；
- evaluator 层数、宽度、precision 和输出 head；
- `PreparedState`、`MaterialProgramState`、sampler reservation 的 byte layout。

首个 profile 使用 `metal_fused_full_v1` 身份。具体维度是研究 config，而不是产品成功门槛；注册候选时必须选择有限值并由 manifest 自动计算真实参数量、MAC、reads、`B_shared` 和 `B_asset`，不能在同一方法身份下随资产改变 shape。

首个质量优先 profile 冻结为以下最大形态；若实现 preflight 证明某个 shape 无法合法编译或驻留，必须发布新的 profile identity，不能在同一 identity 下静默缩模：

| 组件 | `metal_fused_full_v1` |
|---|---|
| opaque source texture slots | 最多 9 个；不足使用显式 slot mask，不以零纹理冒充存在 |
| typed parameter tokens | 最多 32 个，覆盖审计最大 31；token width 64，4 个 4-head set-attention blocks |
| training/cook encoder | role stem 32；shared residual U-Net widths 64/128/192/256；bundle set-attention bottleneck |
| per-domain grids | high 8 channels、约 source linear extent 的 1/2；low 8 channels、约 1/8；每 mip 独立，INT8 QAT + per-channel scale |
| runtime decoder | width 128、4 个 residual blocks；asset LoRA/FiLM rank 上限 8；structured state 64 floats |
| footprint | 相邻 2 个 mip；每 active domain/mip 固定 2×2 high gather + 1 个 low filtered sample |
| learned frames | 3 个，加 renderer/base frame |
| analytic bank | 6 个 source-aware core slots + 4 个 positive residual lobe slots |
| sampler reservation | 上述 10 个 lobe proposal + 1 个全半球 cosine/uniform fallback；固定 component 顺序与 state slots |
| angular bank | 4 个 warped half/slope levels，每级 8 channels；difference factor rank 16 |
| residual evaluator | width 128、4 个 FiLM/LoRA residual blocks；输出 multiplicative correction 与 free positive RGB tail |
| deployment precision | shared/adapter weights FP16 master pack，grid INT8 QAT；敏感 normalization/lobe arithmetic允许 FP32 accumulation |
| state/read envelope | `PreparedState ≤ 4096 B`，typed/material state独立；descriptor 先登记 `maximum_reads=192`，再由生成 layout 与 shader 静态计数收紧 |

这里的 9 slots 和 32 tokens 直接来自 opaque 审计上限；其余尺寸选择的是 RTX 4090 上仍具部署可能性的高容量形态，并显著大于当前 8-latent/width-64 NVIDIA reproduction。Linux结果审阅后若用户批准compact研究，才通过蒸馏、量化、rank/width、active slot/lobe/frame和mip read消融形成新profile；full profile本身不是最终性能承诺。

## 4. Source registry 与 typed schema

### 4.1 Registry 生成

从锁定 MDL SDK 的 metadata-only compilation 和资源 inspection 生成版本化 registry，不手写 692 个条目。registry 以 exact export 为叶节点，同时去重 graph、schema 和 texture-set identity。生成过程必须保存输入 package 指纹和审计摘要，发现 cutout、缺失资源、未知类型或 identity 漂移时停止而不是静默降级。

### 4.2 Typed 参数表示

不建立 154D 零填充向量。每套 schema 保存有序 descriptor：

```text
ParameterDescriptor {
  semantic_id
  mdl_name
  value_type          // float, color, bool, enum, float2, int
  source_range/domain
  default_value
  responsibility      // access, frame, optical, finish, aging, composite
  normalization
  presence
}
```

运行时输入由 common canonical fields、family-local typed tokens、presence/type encoding 和 graph/schema tokens 组成。连续量保留 source range；color 保留线性 RGB；bool、enum 与 index 使用离散 embedding，不承诺域间连续插值。

参数按职责分流：

- UV、projection、tiling、scale、translation、rotation 进入确定性的 spatial access program；
- rounded-corner 与 object-space frame 参数由 renderer/`prepare()` 产生最终 frame；
- optical、finish、aging、contamination、coating 与 mixture 参数进入 typed compiler，并可同时影响 analytic core、texture modulation、mixture mask、residual 和 sampler hints。

### 4.3 Typed source compiler

编译器采用集合式 typed token encoder，而不是参数槽拼接：

```text
[recipe, graph, schema, metal, finish tokens]
+ typed parameter token set
+ source-computable canonical optical fields
→ TypedSourceCompiler
→ MaterialProgramState
```

编译器共享 token/type encoder，用 schema/family adapter 保留局部语义，并通过 masked pooling/attention 聚合可变参数集合。输出为固定 shape：

- core/lobe type、active mask、颜色、roughness、anisotropy、IOR/Fresnel clue 与 mixture 权重；
- texture structured-state 的 bounded modulation；
- evaluator 各 block 的 FiLM 与低秩 LoRA 系数；
- free-tail/correction 的尺度界；
- 后续 sampler 使用的 mixture/proposal hints。

它不生成任意大小的完整网络权重。材质实例创建或 typed edit 时只重跑编译器；同一着色点或每方向查询不重复编译。训练期可用 target-visible optimized state 作为 teacher/control，但正式 editable 路径必须是只读 typed source 的 pure compiler。

## 5. Finish neural texture codec

### 5.1 Bundle schema 与 spatial domains

texture-set 先编译成注册的 `BundleSchema`。schema 描述 source channel roles、packed mapping、transfer function、normal convention、address/filter 规则，以及 recipe 所需的 canonical spatial domains。opaque 审计每个 export 最多引用 9 个 source textures，因此 full profile 预留 9 个 slot；共享相同 coordinate program 的 slots 可以在 bundle compile 时归并为一个 domain，不能归并的仍保持独立。

每个 recipe 还编译一个数据驱动、固定指令上限的 `SpatialAccessProgram`，其 op 只执行 source-computable 的 UV set/object projection、scale/translate/rotate、wrap/crop、registered non-repeat transform、slot/domain routing 和 footprint propagation。它根据 surface、typed coordinate parameters 与 footprint 计算每个 active domain 的 UV、导数/LOD 和 validity；同一 Slang interpreter 执行所有 recipe，不为 178 个 graph 生成 renderer 分支。模型不使用 view-dependent UV warp。

同一方法 ABI 可以绑定不同 bundle schema，但只有 registry 声明兼容的 recipe/bundle 才能组合。标准 finish replacement 可以伴随 recipe/schema state 更新；“换 texture”不要求所有 finish 原始通道数量完全相同。

### 5.2 Shared encoder

资产编译期 encoder 使用以下共享层级：

1. color、normal、scalar/mask/AO、packed correlated channels 的 role-specific stems；
2. 对同一 bundle 多张纹理进行 masked set aggregation；
3. corpus-shared multiscale spatial trunk；
4. role/schema/finish/recipe tokens；
5. grid heads 与固定 shape 的 asset-adapter head。

全部 52 个 source texture-set identities 联合训练共享 trunk。split 单位是 texture-set identity，复用同一 texture set 的 modules/exports 不得跨 train/test 泄漏。由于精确 finish 的独立资产有限，`encoder-only` 只形成 source-derived、compatible-schema 范围内的研究结论。

### 5.3 Hierarchical grids 与 mip

每个 canonical spatial domain、每个 source mip 保存 high/low-frequency local grids：

- high-frequency grid 使用固定邻域 gather，保留 cell 内高频；
- low-frequency grid 使用硬件友好的过滤读取；
- grids 和 asset adapter 进入 QAT，实际 bundle 保存量化表示；
- 每个 mip 有独立监督，不从最高分辨率 latent 自动下采样派生。

`prepare()` 根据连续 footprint 选择相邻两个 mip，分别读取和解码，再对 structured states 做确定性插值。首个完整形态不使用 stochastic one-level 或非线性 decoder 前的 latent interpolation；二者只有在Linux结果审阅后经用户批准，才可能成为matched成本消融。

### 5.4 Shared decoder 与双 head

decoder 使用 shared trunk、schema token 和 bounded asset-local FiLM/LoRA modulation。训练有两个输出族：

- semantic heads：按 color、normal、scalar、mask/AO 和 packed role 重建 source mip，仅用于训练、codec 验证与资产审计；
- runtime structured head：输出固定 shape 的 normal/frame delta、roughness/anisotropy clue、mixture/mask clue、optical modulation 和 learned residual spatial features。

两类 head 共享 grids/trunk，并同时接受 semantic reconstruction 与 online appearance loss。这样 bundle 可审计，但 runtime 不需要重新构造 family-variable raw texture tensor，也不需要再次解释完整 source graph。

### 5.5 新资产三条路径

同一 frozen decoder 与量化 profile 下分别保留：

- `encoder-only`：共享 encoder 一次前向得到 grids/adapter，不优化资产；
- `encoder + bounded refinement`：以 encoder 输出初始化，只优化该资产 grids/adapter，shared decoder 冻结；
- `direct optimized control`：在相同 decoder、bytes 和优化预算合同下直接优化 grids/adapter，作为 best observed codec control。

新资产不产生新完整 decoder。若出现未知 channel role、bundle schema 或 graph recipe，则需要新方法/schema 版本，不冒充同语义 zero-shot replacement。

## 6. Runtime `prepare()`

`prepare(surface, footprint, wo)` 的职责按顺序为：

1. 应用 renderer 提供的 rounded-corner/base frame；
2. 执行 recipe 的 deterministic spatial access program，得到各 domain 的 UV、footprint、相邻 mip 与插值权重；
3. 以 profile 固定的读取数访问量化 high/low grids；
4. 用 shared decoder 和 asset modulation 分别解码两层 structured state，再插值；
5. 将 decoded normal/frame delta 组合到最终 shading frame；
6. 将 `MaterialProgramState` 与 spatial state 合成为 per-hit core/lobe、mixture、learned frames 和 residual condition；
7. 只对 `wo` 计算一次 view-conditioned token；
8. 填充完整 matched sampler 使用的 proposal state。

`PreparedState` 是 backend-specific 的固定 layout，不提升为跨 backend 公共序列化接口。它必须包含 explicit validity/mode flags，处理缺失 UV、half-vector 退化、grazing、inactive domain/lobe 和 chart seam。`evaluate()` 不再次解码 texture、不重新运行 typed compiler。

## 7. Directional representation

每次 `evaluate(prepared, wi)` 并联四路固定上限特征：

1. raw local Cartesian `wo/wi`、dot/cosine、hemisphere 与 validity flags；
2. 数值稳定的 half/difference chart，包括退化 fallback；
3. structured texture head 产生的有限个 learned lobe frames 中的局部方向；
4. method-shared multiscale warped angular feature bank。

angular bank 在驻定后的 half/slope plane 上提供多尺度窄峰特征，并用低秩 difference-direction vector/plane factor 补充相关性。它是共享 feature bank，不是每资产保存完整 BRDF table；每级读取、level 数与 rank 都由 profile 固定。raw/direct path 始终保留，以覆盖 chart seam、有限 rank 和先验不适配区域。

## 8. Source-aware hybrid evaluator

### 8.1 Analytic lobe bank

固定上限 lobe slots 覆盖 source-aware 的 conductor GGX/Beckmann、coat/specular、diffuse/contamination 和 broad scatter 类型。`RecipeSignature`、typed compiler 与 decoded spatial state共同决定 type、active mask 与参数。lobe bank 是内部 prior、correction anchor 和 sampler proposal，不是 source GT vocabulary。

### 8.2 Positive residual parameterization

最终输出为线性 RGB `f`：

```text
f_hat = f_core * exp(clamp(delta_log, -b, b))
      + sum_k f_positive_residual_lobe_k
      + softplus(f_free_tail)
```

- multiplicative branch 修正 core 的颜色、强度和峰形；
- nonnegative residual lobes 添加 core 未覆盖的峰，并可直接进入后续 proposal mixture；
- free positive tail 为复杂 graph、污染层与非单峰效应保留表达出口；
- 不使用 `clamp(core + signed_residual, 0)`，避免负区间死梯度。

residual trunk 读取 directional features、prepared view token、structured spatial state 和 compiler condition，并由逐 block FiLM/LoRA 调制。所有 branch 都有 active mask 与独立能量/容量记账，使后续可以在同一训练/数据预算下做 matched 消融；首个质量形态不提前删除 free tail 或 analytic branch。

### 8.3 物理与数值合同

- 输出在有效上半球有限、非负且为线性 RGB；
- invalid/unsupported state fail closed，不以 NaN 或静默 fallback 继续；
- reciprocity 通过 source-aware paired-query loss 和独立报告约束，不假设所有 view-conditioned network 自动满足；
- grazing、窄峰、极低 roughness 与 chart seam 在 query recipe 中显式过采样；
- core preservation loss 防止 free tail 在训练早期完全吞掉可解释 optical controls，但 core 质量不单独成为组件准入门。

## 9. `sample()/pdf()` 完整方法合同

`PreparedState` 从方法实现开始就包含：

- analytic 与 residual lobe mixture weights；
- lobe frame、roughness/anisotropy 和 support bounds；
- proposal condition 与 normalization state；
- evaluator 已可复用的 PDF/mixture 中间量。

matched sampler 使用固定容量 mixture：6 个 analytic core proposals、4 个 positive residual lobe proposals 与 1 个保证全半球支持的 cosine/uniform fallback。所有 component 都有一致的有界 sample/PDF；inactive slots 仍占 ABI 位置但权重为零。proposal head 以 evaluator 的 `luminance(f) * abs(cos(theta_i))` 形成训练目标，不模仿 source sampler。

`sample()` 返回方向、权重与 forward PDF，对有效 sample 至多运行一次 directional evaluator；`pdf()` 与同一 mixture/state 对应且不运行 evaluator。两者都不得重新运行 texture decoder或 typed compiler。reference/analytic-only proposal只标记为开发期control；full package只有在sample→pdf、normalization、weight identity与backend parity成立后才能导出，不能交付evaluator-only的Metal full方法。

## 10. Canonical 架构迁移

### 10.1 唯一正式数据流

迁移后的唯一pipeline为：

```text
SourceSnapshot set
  → ReferenceExecutionPlan@1
  → ReferenceBackendSession
  → NativeAssetCollection@1 + OnlineTrainingPipeline
       ├─ AssetTileBatch@1
       ├─ EvaluatorBatch@3
       └─ MethodSamplerBatch@3
  → MethodDefinition / MethodDescriptor@2
  → TrainingConfig@4 / TrainingCheckpoint@4
  → program + asset + instance compilers
  → ScatteringPackage@2
  → ProgramRuntimeCache + AssetBinding + InstanceBinding
  → ComparisonSlot[2]
```

这是破坏式canonical migration，不是给现有v3/v1合同增加optional Metal字段。根仓库的NVIDIA method、五个reference、adapters、configs、tests、CLI、checkpoint、exporter、Slang、package、viewer、spec和稳定文档全部切换后，删除旧config/checkpoint/package reader、schema、alias、converter、auto-probe与双路径。历史artifact只保留provenance，不可由产品代码加载。

`SourceSnapshot`、`ReferenceBackendCapability`、canonical scattering ABI和GPU-resident online target语义继续成立，但它们的调用签名可以随新execution plan递归更新。是否保留某个类型名取决于新合同是否自然表达目标，不以“兼容现有调用方”为理由保留旧shape。

### 10.2 `ReferenceExecutionPlan@1`

reference program把全cohort snapshots编成versioned plan：

- `ReferenceExecutionGroup` key覆盖generated/runtime module、RO data、resource table、graph/capability与backend build identity；
- group内保存material records、argument/RO offsets、typed-state pool与global source-index映射；
- session pool按group持有runtime/resources，batch必须group-homogeneous；producer在step间轮换group，而不是在一个shader launch中混合178个graph；
- query recipe登记direction proposal、typed-state sampling、footprint sub-sampling与valid-row policy；
- source/reference公共层不识别Metal名称，Metal registry只提供plan input。

MDL provider必须真正pack多个argument blocks/RO offsets；`state.arg_block_offset=0`和“每session只能一个material-specific module”的限制被删除，不保留single-snapshot adapter。

### 10.3 `NativeAssetCollection@1`

`NativeFeaturePyramid`由multi-asset collection替代，所有既有source adapters一起迁移。collection以asset/domain/mip/role/schema为identity，提供：

- tile + halo的source tensor读取和role-aware target；
- asset/tile-coherent随机采样、UV/footprint与level mapping；
- GPU working-set cache和显式lease，不在host展开全部52 assets；
- cook时遍历全asset tiles并stitch量化grids；
- 训练时只编码当前active tiles，使encoder/decoder端到端反向而无需每step重算整张4K纹理。

source textures是source资产输入，可以由collection解码和缓存；禁止持久化的是reference response batch，不是权威source texture本身。

## 11. 完整方法与机械化conformance

### 11.1 `MethodDescriptor@2`

descriptor除identity、source contracts和bounded execution外，还必须登记：

```text
ComponentContract {
  component_id
  required
  parameter_groups
  active_phases
  batch_dependencies
  python_outputs
  runtime_artifacts
  slang_entry_points
}
```

`metal_fused_full_v1`的required list逐项覆盖PRD R5。full config不提供关闭required component的flag；消融只能创建另一method/profile identity。模型创建后parameter registry必须与component contracts双向一致：每个trainable parameter恰属一个group，每个required trainable component至少有一个group，禁止orphan parameters。

### 11.2 三类完整性证据

1. **Execution coverage**：registered activation batches让每个required component产生非空、非恒等的trace/output；inactive source slot只由显式mask解释。
2. **Gradient/update coverage**：在component声明的phase内累计stratified audit window，所有required groups有finite非零gradient与optimizer state/update；未选中的embedding row可以稀疏，但对应role/schema/recipe必须在audit set中被激活。
3. **Artifact coverage**：checkpoint、program、asset、instance和Slang symbols与component list交叉校验；缺任一tensor/resource/entry point时export失败。

测试禁止以mock branch或constant oracle代替full path。analytic-only、direct-only、encoder-disabled、sampler-reserved-only等均使用不同diagnostic identity，不能通过`metal_fused_full_v1`验证。

## 12. `TrainingConfig@4` 与单GPU优化生命周期

### 12.1 Phase graph

`TrainingConfig@4`不再硬编码双route和一次materialization，而是保存有序phase graph。每个phase冻结routes、trainable groups、loss terms、optimizer、schedule、precision、step budget和checkpoint boundary：

1. `codec-warmup`：`AssetTileBatch`联合训练role stems、shared encoder/decoder、semantic/structured heads和asset modulation；
2. `joint-appearance`：`AssetTileBatch + EvaluatorBatch`同时启用codec、pure typed compiler、analytic/angular/residual evaluator，使用online MDL `f`；
3. `proposal-fit`：保留full evaluator路径，用detached `luminance(f)|cosθi|`训练10-lobe+fallback proposal，并执行sample/PDF correctness losses；
4. `qat-refine`：把INT8 grid simulation、FP16 runtime weights和敏感FP32 accumulation放入forward，refine shared/asset/compiler/evaluator/proposal后冻结可导出state。

bootstrap只是同一method identity的optimization schedule，不是删减组件。Windows smoke和Linux long config共享phase graph、model profile、loss定义与precision；Windows只缩短step/batch/validation数量。

### 12.2 Loss分账

- role-aware texture reconstruction、normal angular、scalar/mask、mip/footprint consistency；
- transformed-domain robust response与linear-domain energy；
- peak/top-energy support与grazing/roughness strata；
- source-aware reciprocity和analytic-core preservation；
- pure typed compiler的functional/state distillation；
- proposal density、support coverage、sample→pdf与weight-tail risk；
- QAT error与`B_shared/B_asset/B_instance`记账。

target-visible optimized state仍只作teacher/control。authored presets和连续edits必须走同一pure compiler，encoder-only path必须直接承受loss，不能由free asset code绕过。

### 12.3 高效梯度下降热路径

- 单进程、单GPU；不实现或预埋DDP/per-rank旁路；
- `ReferenceExecutionPlan`调度group-homogeneous batch，`NativeAssetCollection`调度asset/tile-coherent batch；
- 利用backend多slot与CUDA events预取下一route，batch lease跨forward/backward安全持有；
- 默认BF16 autocast + FP32 master/sensitive accumulation，QAT阶段模拟最终INT8/FP16；precision oracle证明不适用的op显式退出autocast；
- 每phase使用显式parameter groups和fused optimizer；不把所有参数无差别塞进单一global optimizer；
- finite/gradient检查在GPU聚合并异步断言，完整parameter coverage只在audit interval运行；
- loss/metrics在GPU累积，到log interval批量转host；`tqdm`不强制每step `.item()`；
- validation/checkpoint/source cook按独立可观察阶段运行，不在普通step插入高频I/O；
- profile拆分reference、asset encode、forward、backward、optimizer、validation/checkpoint、peak memory和host sync。

若prefetch因Falcor/CUDA资源互斥不能安全重叠，保留同一pipeline的同步调度并记录profile证据；不得另写Metal trainer。

### 12.4 `TrainingCheckpoint@4`

checkpoint保存method/component manifest、config/phase graph、program plan、asset collection、source/query identities、phase/step、model state、phase-local optimizer/scheduler/precision state、RNG、query streams、gradient coverage和validation summary。resume严格恢复当前phase与online query序列。

`TrainingCheckpoint@3` reader和converter删除；四个NVIDIA configs和method一起迁移到v4。跨Windows/Linux不默认resume同一checkpoint：两侧使用同一semantic config，但Linux long run生成自己的backend build identity和checkpoint。

## 13. `ScatteringPackage@2` 与运行时

### 13.1 三部分package

`ScatteringPackage@2`直接表达方法的三部分组合：

- `program`：shared decoder、typed compiler、angular/evaluator/proposal weights、Slang closure、ABI/profile与`program_runtime_id`；
- `asset`：finish grids、bounded adapter、role/schema/mip/filter metadata、source provenance与`texture_asset_id`；
- `instance`：metal optical identity、recipe signature、raw typed buffer、compiled `MaterialProgramState`与`material_instance_id`。

`package_id`覆盖三者、validation与provenance。一个package可以交付一个composed instance，但viewer binding必须保持program/asset/instance边界，不能再把asset和instance合并为旧`material`section。

旧`ScatteringPackage@1` schema、Python/C++ reader和viewer loader删除；NVIDIA package exporter/fixture迁移到v2，不保留兼容加载。旧artifact目录只在历史报告中以hash引用。

### 13.2 Viewer与typed edit

viewer建立`ProgramRuntimeCache`、`AssetBinding`和`InstanceBinding`：

- 相同`program_runtime_id`只编译/上传一次shared program；
- bundle replacement只验证并原子交换asset + recompiled instance；
- typed editor按UI-safe native schema写raw buffer，并dispatch通用material-compiler entry一次；
- resource factory按typed descriptor/usage创建INT8/FP16 grids、buffers与samplers，不判断Metal/module/preset名称；
- candidate验证、compile或resource失败时保留旧三层binding；
- full Metal package必须同时声明并通过`PREPARE/EVALUATE/SAMPLE/PDF`。

Python FP32、BF16/QAT、packed oracle、Slang package和viewer使用生成的layout/component enum，避免手写重复ABI。

## 14. Windows gate 与 Linux handoff

### 14.1 Windows完整性验证

Windows/RTX 4090运行full shape，不运行tiny替代模型：

1. 692 exports、178 groups、52 assets、64 schemas的registry/plan/asset preflight；
2. 每个group/asset/schema/role/recipe至少完成对应的compile/query/activation检查；
3. 从registry机械生成覆盖全部required components、parameter groups、texture roles、direction charts、recipe类别和proposal components的最小stratified训练子集；在full model shape和真实online reference路径上执行四个phase的optimizer steps并跨phase checkpoint/resume；
4. component execution、gradient/update和artifact coverage全部闭合；
5. forward/loss/gradient/state/PDF全程finite且无silent clamp；
6. deterministic online micro-overfit或固定query-stream短run以预冻结window/bootstrap方法证明loss下降；
7. 输出短训diagnostic package并完成Python→QAT→Slang→viewer、bundle swap和typed edit smoke；
8. profile训练step breakdown、memory与同步热点，证明没有host target readback或磁盘batch。

Windows checkpoint只证明实现和optimization flow，不作为Metal最终质量checkpoint。

训练子集优化只减少source/query/batch/step工作量，不允许使用tiny model、冻结required branch、替换loss/reference、跳过encoder反向或用mock target。全cohort仍完成registry、compile、resource和representative query preflight；短run结果不用于最终泛化质量判断。

### 14.2 Linux单GPU长训练交付

交付一个platform-neutral long config和一个短smoke config。二者只在step/batch/validation/checkpoint cadence上不同，不改变method/component/profile。Linux使用现有：

```bash
CUDA_VISIBLE_DEVICES=<one-gpu> bash scripts/deploy_reference_linux.sh
CUDA_VISIBLE_DEVICES=<one-gpu> bash scripts/run_falcor_python.sh \
  -m ncls.cli learn train <metal-full-long-config> <linux-checkpoint>
```

handoff还包括source资产检查、registry/config/checkpoint hashes、预计VRAM与Windows profile推导的ETA、resume命令、日志/`tqdm`、磁盘空间、failure分类和停止条件。新Metal method的Linux短smoke是接收方启动long run前的第一道gate，不由当前Windows任务代跑或冒充；既有fixed-MDL Linux证据只证明公共backend可用。

### 14.3 长训后的决策门

Linux long run完成后只自动生成首轮审阅包：训练/validation曲线、component/gradient健康、代表性13×7与特殊recipe渲染、typed parameter sweep、bundle replacement对比、sample/PDF基础统计、`B_shared/B_asset/B_instance`和step/runtime摘要。

用户先审阅实际效果。没有新的规划批准，不启动六类formal matrix、额外seed/预算、matched ablation、蒸馏、compact profile或Pareto选择。

## 15. 互斥选择、失败与回滚

| 设计点 | 当前选择 | 不进入当前full identity的替代 |
|---|---|---|
| texture输出 | shared grids + semantic/structured dual heads | raw-only、structured-only |
| decoder共享 | shared trunk + schema adapter + bounded asset modulation | global-only、per-asset full decoder |
| mip | independent per-mip + adjacent decoded-state interpolation | derived mip、stochastic one-level、latent interpolation |
| parameter state | pure typed compiler | preset free code、target-visible runtime state |
| direction | raw + half/diff + learned frames + angular bank | 逐分支删除 |
| evaluator | analytic + multiplicative + positive residual lobes + free tail | core-only、direct-only |
| sampler | 10 lobes + full-support fallback | analytic-only、source sampler imitation |
| training | single-GPU phase graph | DDP、offline response corpus |

implementation defect、protocol defect或静态部署不可能允许回到设计并创建新identity；正常低quality/低throughput是observed result，不触发自动变体。回滚发生在提交/模块边界，不通过保留旧public API实现。迁移中的调用方必须在同一child完成后全部位于新合同，禁止长期双路径。

上游MDL/Falcor保持固定commit和clean。若确需upstream修改，按根仓库patch/overlay规则另行规划，不能把未说明修改留在`external/`。
