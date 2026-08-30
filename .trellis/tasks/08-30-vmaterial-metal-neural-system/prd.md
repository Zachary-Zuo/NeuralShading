# vMaterial Metal 神经材质系统

## 目标与用户价值

以 NVIDIA vMaterials 2.4.0 的 Metal 材质为权威源语义，交付一套统一、随机访问、运行成本静态有界的 neural material program及其跨平台online training系统。方法保留金属身份、可替换finish neural texture bundle与原生typed参数编辑，并完整实现训练、`prepare/evaluate/sample/pdf`、checkpoint、package、Slang和viewer路径。

本任务的完成边界是“新架构与完整方法已经正确实现，可高效做梯度下降，并可移交Linux长训练”，不是在Windows上完成692个opaque exports的full formal convergence。Windows负责完整shape/组件/phase的数值、梯度、短程收敛与部署流程验证；正式长训练、最终泛化质量和Pareto结果使用交付后的Linux版本产生。

## 已确认事实

### Metal source审计

- Metal目录包含127个MDL module、837个authored exports、64套parameter schema、193个compiled graph identities、57个texture-set identities和140条唯一source texture paths。
- Metal-v1纳入692个opaque exports，对应178个opaque compiled graph identities与52个opaque texture-set identities；145个cutout exports使用`geometry.cutout_opacity`并fail closed。
- opaque exports共出现154个唯一editable参数名，每个export有9–31个typed参数。
- 主体是13种metal identity × 7种finish的91-module矩阵；特殊patina、aging、coating、paint、crack与structure recipe保持native family-local适用域。
- source texture是module内部固定资源，不是editable `texture_2d` argument；“换neural texture”是本项目新增的编译后资产能力。

### 跨平台online training基础

- 2026-08-29已在Windows/RTX 4090和Ubuntu 22.04.5/RTX A6000上实机验证统一`ReferenceBackendCapability/Session`；Linux固定Vulkan，Windows固定D3D12，上层不分平台。
- 五个canonical reference program已在Linux完成representative `evaluate/sample/pdf`、same-device CUDA tensor与lease验证；固定MDL state使用同一config/CLI在两平台完成GPU-resident online training smoke。
- target从Falcor shared GPU output直接进入CUDA tensor/loss，不保存或读取训练batch/corpus；Linux初步验证不需要在本任务中重做。
- 既有Linux证据只覆盖公共backend与固定MDL/NVIDIA smoke，不能替代新Metal full method的最终Linux启动验证。

### 当前架构缺口

- `TrainingConfig@3`与runner固定双route、单optimizer和`bootstrap→finetune`两阶段，不能自然表达codec、joint appearance、proposal和QAT/refinement lifecycle。
- 当前`MethodDefinition`没有required-component、parameter-group gradient/update和compiled-artifact completeness合同，无法防止只实现部分方法。
- `NativeFeaturePyramid`和MDL session是单资产/单generated-module导向，不能高效承载52 assets、178 graph groups与连续typed states。
- `ScatteringPackage@1`把finish asset与compiled instance state合并为material section，不符合三部分组合和通用runtime cache边界。
- 项目`code-organization.md`已经要求接口迁移必须递归切换调用方并删除旧reader、alias、converter、schema probe和fallback；本任务严格执行该规则。

完整证据见`research/vmaterials2-metal-audit.md`、`research/architecture-gap-audit.md`与`research/training-delivery-boundary-audit.md`。

## 需求

### R1 Source语义与范围

- 以原生MDL program、typed arguments、资源和图结构生成GT；不得先反演为LayerStack、Principled或固定closure vocabulary。
- registry完整追溯692 opaque exports、178 graphs、52 texture sets、64 schemas及其resource/capability/source identities。
- 145 cutout exports不得登记到Metal-v1 neural catalog，也不得忽略opacity后按opaque近似。
- full decoded artifact、execution group、typed state、filtered footprint与representative GPU query必须经过实施期preflight；metadata audit不能代替runtime证据。

### R2 三部分组合与可编辑范围

```text
MetalOpticalIdentity
× FinishNeuralTextureBundle
× NativeTypedParameterState
→ composed neural material instance
```

- 标准13×7 matrix支持source-backed的metal/finish/parameter合法新组合。
- 特殊recipe只在registered native module/family schema内编辑；域外overlay/layer重排明确拒绝。
- bundle replacement检查graph/schema/channel/recipe compatibility，不依赖neural extrapolation宣称不受支持的组合。
- 保留全部opaque native typed editability；缺失参数与零值分离，离散值不承诺无source依据的连续插值。
- typed edit只重新生成bounded instance state，不重训evaluator或重新编码未变化的finish bundle；Metal-v1不新增逐像素parameter maps。

### R3 随机访问neural texture

- 52个source-derived texture sets联合训练role-specific stems/heads与shared multiscale encoder/decoder trunk；不为每个finish或资产训练完整decoder。
- 新compatible asset由shared encoder产生per-mip high/low grids和bounded asset modulation，runtime继续使用既有shared decoder。分别报告`encoder-only`、`encoder + bounded refinement`和`direct optimized control`。
- bundle保存channel role、transfer、normal convention、mip/filter、compatibility与provenance。
- codec使用独立监督per-mip grids、相邻两层确定性filtering、QAT、training-only semantic heads和runtime structured head。
- 训练以asset/tile/role/schema-aware batch联合优化encoder+decoder；不能预烘全部response tensor，也不能在appearance阶段把未实现encoder反向路径伪装成“已联合训练”。

### R4 唯一最新架构与递归迁移

- 设计一套能表达multi-asset collection、grouped reference、任意typed training phases、三部分deployment和完整方法合同的canonical接口。
- 根仓库全部正式调用方一次性迁移：五个reference、NVIDIA method、source adapters、producer/runner、configs、checkpoint、exporter、package、Slang、viewer、tests、spec与稳定文档。
- 迁移结束后删除旧public symbols、format readers、aliases、converters、schema probes、双registry和fallback；不保留新旧两条pipeline。
- 旧checkpoint/package只作为历史artifact provenance，不提供runtime加载或转换。正常source/method多态不是兼容层。
- Windows/Linux调用同一source、method、config、runner与checkpoint/package合同；platform/device/toolchain差异只存在于现有公共backend/launcher/provider。
- 新增Metal不得产生family-specific runner、CLI、checkpoint、exporter、session或viewer renderer分支。

### R5 完整实现`metal_fused_full_v1`

full profile必须同时包含并实际执行：

- role-specific texture stems、shared encoder trunk、per-mip high/low grids、semantic/structured heads与bounded asset adapter；
- typed token compiler、spatial access program、core/texture/FiLM/LoRA/proposal state generation；
- raw Cartesian、stable half/difference、learned lobe frames与multiscale angular bank；
- 6个source-aware analytic core slots、multiplicative correction、4个positive residual lobes与free positive RGB tail；
- 10-lobe proposal mixture、全半球fallback以及一致的`sample/pdf`；
- quantization-aware state、program/asset/instance packing与Python/Slang runtime。

方法descriptor必须把上述项登记为required components，并为每项声明parameter group、训练phase、runtime presence和artifact outputs。以下行为均视为未完成：空实现、恒等/零分支、只创建参数但不执行、无梯度参数组、配置默认关闭、只实现Python不实现Slang/package、用analytic-only proposal代替matched sampler。

### R6 可高效进行梯度下降

- 所有reference response保持GPU-resident；正式训练不调用host response readback，不保存/读取HDF5、shard或recorded batch。
- 以`ReferenceExecutionPlan`生成program/resource-homogeneous batches，以`NativeAssetCollection`生成asset/tile-coherent codec与appearance batches，避免逐export Python循环和全量纹理每step展开。
- lifecycle显式覆盖codec warmup、joint appearance、proposal fit和QAT/refinement；各phase声明active routes、trainable parameter groups、optimizer/schedule、precision与resume边界。
- 支持GPU适合的mixed precision、FP32敏感累加、fused optimizer和异步/低同步gradient finite检查；日志、validation和checkpoint不得强迫每step读取GPU scalar。
- 利用现有multi-slot lease实现安全prefetch/overlap；若profiling证明某stage不适合overlap，保留统一同步实现但记录原因，不增加旁路trainer。
- gradient conformance audit必须证明每个required trainable group在其registered activation batch上有finite非零gradient和实际optimizer update；允许mask导致的单batch稀疏，但不能存在整个audit window不可达的参数。
- 记录reference query、asset encode、forward、backward、optimizer、validation/checkpoint、peak memory与host sync breakdown。这些数据用于发现热点，不在本轮预设效率hard gate。

### R7 Windows正确性与短程收敛验证

- Windows使用与Linux长训练相同的full method/profile；smoke只能缩短steps、batch/query数量和validation频率，不能关闭required components或替换为tiny model。
- registry/asset/schema/execution-group preflight覆盖整个opaque cohort；训练activation set覆盖所有component、parameter responsibility、texture role、direction chart、recipe类别与proposal component。
- 梯度下降正确性与短程收敛优先使用能够闭合上述coverage的最小stratified source/query子集；full cohort只执行静态/compile/query preflight。该优化不得改变full model shape、method identity、真实online reference、loss、optimizer或phase data flow，也不据此宣称最终质量。
- codec、joint appearance、proposal与QAT/refinement各执行真实optimizer steps，并跨至少一个phase/resume boundary恢复。
- forward、loss、state、gradient、optimizer、checkpoint tensors全程finite，无NaN/Inf、非法PDF、负`f`或silent clamp。
- deterministic online micro-overfit/固定query-stream短训练的末段loss相对初段具有统计可信下降，同时每个required parameter group有update；不要求full-quality convergence。
- 完成Python FP32→mixed/QAT→Slang package→viewer的`prepare/evaluate/sample/pdf`、sample→pdf、normalization、weight identity和atomic loading smoke。

### R8 Linux长训练交付

- 交付platform-neutral full long-run config、source registry/locator、phase budget、checkpoint/resume、failure recovery、monitoring与正式评测入口。
- Linux通过现有`deploy_reference_linux.sh`和`run_falcor_python.sh`启动同一CLI；不得创建Metal或Linux专用trainer/config schema。
- Windows checkpoint不冒充Linux完成证据；Linux long run生成自己的backend build identity、checkpoint和artifacts。
- 交付时提供Linux启动preflight和full method短smoke命令；新Metal method的实际Linux smoke是接收方开始long run前的第一道handoff gate，不属于当前Windows完成证据，已有fixed-MDL结果只证明底层可用。
- 当前训练合同固定为单进程、单GPU；Windows和Linux都通过一个可见CUDA设备运行。multi-GPU DDP、per-rank reference、distributed sharding/RNG/checkpoint不进入本任务，也不在当前接口中预埋未经验证的分布式旁路。

### R9 后续研究与成本

- Linux full long run结束后先输出checkpoint、训练曲线、代表性材质/参数/texture replacement效果与基础成本摘要，等待用户审阅。未经新的用户决策，不自动排队六类泛化formal matrix、matched bootstrap CI、消融、蒸馏或compact Pareto。
- 当前实现从首次可运行起记录`B_shared/B_asset/B_instance`、state/read/MAC和训练/runtime profile，为后续Linux formal与compact保留可比身份。
- 最终产品价值仍要求在用户认可质量/能力范围内，time与storage/residency不能同时不占优；数值门槛和aggregation等实测后再对齐。

## 不在范围

- 在Windows上完成692 exports的长时间full convergence或最终checkpoint selection；
- 当前交付中产生完整`G_asset/G_metal/G_finish/G_pair/G_param/G_recipe` formal结果、全量matched ablation或产品Pareto结论；
- 145个cutout exports及opacity/coverage/visibility/silhouette合同；
- 任意外部texture import、自动channel authoring或无source reference的跨family overlay；
- UE集成或把环境积分/多灯scaling写成evaluator实现hard gate；
- multi-GPU DDP、per-rank reference/session、distributed sampler与distributed checkpoint；
- 为旧config/checkpoint/package保留reader、converter、alias或兼容fallback。

## 验收标准

- [x] [需求交付｜用户架构要求] canonical新接口覆盖grouped reference、multi-asset online training、完整method phases与program/asset/instance deployment；全部正式调用方递归迁移，旧API/reader/alias/converter/fallback经静态负向测试证明已删除；
- [x] [语义正确性｜`docs/material_scope.md`] 692 opaque registry与GT保持原生MDL语义，145 cutout fail closed，特殊recipe不越出native适用域；
- [x] [需求交付｜用户三部分组合] metal identity、finish bundle与native typed instance state独立可追溯、可组合、可编辑，兼容切换不重训或重编码未变化资产；
- [x] [需求交付｜用户“完整实现”] required-component manifest覆盖R5全部组件；construction、training、checkpoint和export均拒绝缺失、disabled、zero/identity placeholder或无artifact分支；
- [x] [数值实现正确性｜gradient conformance] stratified activation audit中每个required trainable group具有finite非零gradient和optimizer update，所有phase可resume且没有orphan trainable parameter；
- [x] [数值实现正确性｜online optimization] Windows full-profile短训练无host target readback、无持久化batch、无NaN/Inf，并以预冻结统计方法证明loss末段低于初段；
- [x] [需求交付｜用户高效正确性验证] Windows以覆盖全部required components、parameter groups和关键source语义的最小stratified子集执行真实online优化；除budget/cadence与cohort缩减外不改变full方法，并保留全cohort preflight；
- [x] [数值实现正确性｜scattering/package oracle] Python、mixed/QAT、Slang与viewer的`prepare/evaluate/sample/pdf`、sample→pdf、normalization与weight identity tolerance在观察正式结果前由precision/oracle推导；
- [x] [需求交付｜用户“高效梯度下降”] 训练profile分解关键stage、memory与sync，热路径不逐step同步全部parameter/metrics，group/tile batching和prefetch合同有实现及回归证据；observed throughput保持report-only；
- [x] [需求交付｜用户跨平台要求] 同一full method/config/CLI可由Windows D3D12/CUDA与Linux Vulkan/CUDA backend消费，无平台或Metal专用upper-layer分支；
- [x] [需求交付｜用户Linux handoff] long-run config、source identity、resume/monitor/failure说明和Linux preflight/smoke命令齐全，可在已验证Ubuntu/A6000环境开始长训练；
- [x] [需求交付｜用户单GPU边界] Windows/Linux使用同一单GPU训练合同，不包含未验证DDP旁路；Linux长训后只生成首轮效果审阅材料，不自动启动formal/ablation/Pareto；
- [x] [工程正确性｜项目统一性合同] NVIDIA、五个reference、configs/tests/spec/docs/package/viewer均迁入唯一新接口，上游external保持锁定且干净。

## 延后决策

- Linux长训练完成后，先由用户审阅实际效果、训练稳定性和基础成本；是否执行完整formal matrix、追加训练、做matched ablation/Pareto或开始compact deployment，均作为新的规划决定。
- multi-GPU DDP不在当前任务范围；若以后需要，必须单独定义distributed reference、sharding、RNG、checkpoint和Linux实机验收合同。
