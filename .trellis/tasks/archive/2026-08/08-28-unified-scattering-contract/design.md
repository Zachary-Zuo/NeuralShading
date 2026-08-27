# 设计：统一 source / neural scattering contract

## 1. 设计目标与不变量

本任务不再发明一套 reference-only ABI。已有 `INclsScatteringBackend` / `INclsScatteringState` 就是唯一公共 shader 合同，source reference 与 neural package 都必须遵守：

```slang
State prepare(NclsScatteringContext context, CompiledMaterial material);
NclsScatteringEval State.evaluate(float3 wiWorld, inout SampleGenerator sg);
bool State.sample(out NclsScatteringSample result, inout SampleGenerator sg);
NclsScatteringPdf State.pdf(float3 wiWorld);
```

必须同时成立的 estimator 不变量：

1. `evaluate()` 返回纯 BSDF `f`；renderer 只在一个公共 adapter 中乘 `abs(dot(Ns, wi))`。
2. 连续事件的 `sample()`、`pdf()`、`evaluate().pdf` 必须属于同一数学 proposal。project-owned proposal 逐点验证；source-native API 的 sample direction/event/PDF/weight 是不可拆分的权威 tuple，不能用已舍入方向的 independent query 重建。
3. 连续事件的数学关系是 `sample.weight = f * abs(dot(Ns, wi)) / pdf.forward`；source 原生 sampler 直接返回等价 throughput weight 时，以 source oracle identity 为数值门，并单独验证 independent evaluate↔pdf。
4. 环境 NEE/MIS 只调用同一个 state 的 `pdf()`；不得另造 renderer-side PDF。
5. delta 事件显式携带 `Delta` flag，可具有 0 solid-angle PDF；不能除以 epsilon PDF。
6. source 的 native payload、图、纹理、测量表和参数仍是 GT。派生 sampling proposal 只影响 Monte Carlo 方差，不改变 evaluator 或 source identity。

## 2. 架构边界

```text
Falcor PathSurface
        │
        ▼
NclsScatteringContext
        │
        ├── source slot ── SceneReferenceProgram ─┬─ LayerStack backend/state
        │          （heterogeneous sum type）      ├─ MERL backend/state
        │                                         ├─ OpenPBR state
        │                                         ├─ MaterialX state
        │                                         └─ MDL generated state
        │
        └── neural slot ── NclsPackageBackend ────── method-private state
                         
两条 PT 在命中后都只执行：prepare → evaluate/pdf 或 sample
```

新增 `apps/viewer/shaders/SceneReferenceProgram.slang` 作为 heterogeneous source scene 的正式组合程序。它不是兼容层：五个 `shaders/ncls/reference_backends/*.slang` 是 canonical implementation，直接实现公共 contract；组合程序只表达 scene material 可能属于不同 concrete backend 这一必要的 discriminated sum type。

- 每个 source-private module 直接实现自己的 `Backend : INclsScatteringBackend` 与 `State : INclsScatteringState`；formal runtime 和 viewer scene composition 引用同一个实现，不复制 evaluator/sampler/PDF。
- scene composition 的 `CompiledMaterial` 是带 family tag 的 native binding。受 Slang/DXC 对 resource-handle aggregate 的 lowering 限制，composer state 保存 resource-free prepared values，并在 canonical 方法调用点用全局绑定重建对应 concrete state；它不能调用旧 `nclsEvalReferencePath/nclsSample*Path/nclsReferencePdfPath`，这些符号会删除。这是 sum-type 的存储实现，不是第二套 scattering ABI。
- family dispatch 只允许存在于 sum type 的 `prepare/evaluate/sample/pdf/emission`；`ReferencePathTracer.cs.slang` 不再包含 family switch、generic proposal 或 source 数学。
- `NCLS_REFERENCE_FAMILY_MASK` 继续由 C++ 根据 scene resource binding 生成，用于静态裁剪未使用 family。这是 host 资源 specialization，不参与 estimator。
- MDL 仍允许同一 scene specialization 只有一个 generated target-code module；project-owned MDL backend 直接实现 scattering contract 并由 formal/viewer 共用。现有 `MdlViewerAdapter.slang` 的 viewer-only API 被迁移后删除，不作为 shim 保留。

`ReferencePathTracer` 会采用和 `PackagePathTracer` 相同的调用形状：构造公共 context、`backend.prepare()`、直接光 `state.evaluate()/state.pdf()`、续路径 `state.sample()`。source-specific working-space/emission 处理也移入 `NclsReferenceState` 的私有辅助方法，积分器不读取 `surface.family`。

## 3. Source-private sampling 实现

### 3.1 公共数学 primitive

新增一个只提供数学 primitive 的 bounded reflection mixture：一个 cosine support lobe 加最多四个 rotated anisotropic GGX VNDF lobe。它负责：

- 对有限非负 component weights 做确定性归一化；
- 用一维随机数选 component、二维随机数采样该 component；
- 对任意方向计算完整 mixture PDF；
- sampled GGX 产生下半球方向时返回显式 null event，使连续方向 PDF 的缺失质量与 null probability 一致，而不是偷换 cosine fallback。

共享 primitive 不决定任何材质参数。每个 source backend 自己从 native state 构造 mixture，因而 estimator 所有权仍在材质族。

### 3.2 LayerStack

- 单界面：继续调用现有 interface-native sampler/PDF；diffuse/sheen、rough conductor、rough dielectric reflection 各自保持原生分布。
- 多界面：不再把 analog random-walk exterior direction 配 `pdf=0`。backend 从 top coat 与各有效反射界面的 roughness、rotation、Fresnel/color proxy 构造 bounded mixture，并保留 cosine floor 覆盖整个上半球。
- `sample()` 先从该 proposal 取 exterior direction，再调用同一个 random-walk `evaluate()` 得到 stochastic `f`，返回 `f·cos/pdf`；`pdf()` 计算同一 mixture。这样环境 NEE/MIS 可恢复，且不需要伪造 layered random-walk marginal PDF。
- `gMaxLayerWalkDepth` 作为 viewer backend 的私有 prepare 参数；正式 reference program 继续使用冻结上限 64。两者使用同一散射数学，只允许执行预算不同并显式记录。

### 3.3 MERL

MERL 数据没有原生 sampler。backend 在 `prepare()` 中针对当前 `wo` 对原始测量表做固定次数的 peak/off-peak probes，给 cosine + 多尺度 isotropic GGX components 分配权重：

- roughness support 固定覆盖极窄到宽峰；
- probes 与 component 数均编译期有界，不生成新 GT 表，也不修改 MERL measured evaluator；
- cosine component 保留严格正的 support floor；
- forward/reverse PDF 分别用对应 outgoing direction 重建同一 view-conditioned proposal。

这比固定 roughness `0.2` 或纯 cosine 能覆盖 chrome/specular phenolic 等高动态范围材料，同时 sample 与 PDF 保持解析一致。

### 3.4 MaterialX

当前锁定的 MaterialX backend 是项目已支持的 1.39.4 `standard_surface` 反射子集。它由 diffuse/Oren–Nayar 与共享主 GGX distribution 的 dielectric/conductor response 组成。本任务：

- 从 resolved base color、metalness、specular weight/color、roughness、anisotropy 与 rotation 构造 diffuse + rotated anisotropic GGX mixture；
- component weights 由当前 surface resolved inputs 决定，并保留 support floor；
- sample 与 PDF 都在旋转后的 tangent frame 中使用同一 alpha/rotation；
- normal map 与 UV footprint 的解析留在 backend `prepare()`，不再由 path tracer 特判。

不在本任务中扩展该 evaluator 子集到尚未支持的 MaterialX closure；OpenPBR 的 coat/fuzz/thin-film 由 OpenPBR 自己覆盖。

### 3.5 OpenPBR

继续使用锁定 `openpbr_sample/openpbr_pdf/openpbr_eval`。只做公共 state 包装、event/working-space/absolute-cosine 对齐，不替换其原生 lobe selection。`openpbr_sample` 返回的 direction/event/accumulated PDF/weight 原样进入公共 sample；独立 `openpbr_pdf` 用于 NEE。极窄掠射反射不得从已舍入 `wi` 重建 sample tuple。

### 3.6 MDL

继续使用同一 generated target code 的 `surface_scattering_evaluate/sample/pdf`：

- canonical MDL backend 把 `Shading_state_material`、event、forward/reverse PDF、`bsdf_over_pdf` 直接实现为 `NclsScattering*` contract；
- formal `ReferenceProgramDescriptor` 声明 `Prepare|Evaluate|Sample|Pdf|ReversePdf`，并按 target code 能力声明 delta/transmission；
- offline query provider 可继续只批量消费 evaluate，并把 PDF 用于诊断，但其 composed runtime 必须包含完整 state；query-plane capability 与 scattering runtime capability 在文档中分开命名，避免再次混淆。

## 4. Response measure 与 transmission

`NclsScatteringEval.f` 是纯 BSDF，因此 response adapter 改为：

```text
response = valid ? f * abs(dot(Ns, wi)) : 0
```

Python 与 Slang 同步改名为 `absolute_light_cosine` / `nclsAbsoluteLightCosine`，删除 positive-cosine 旧入口。reflection backend 自己拒绝下半球方向；transmission backend 通过 event/domain 保留正确的绝对余弦。该改动与 `reference_dataset_v5` 的 `rgb-bsdf-times-absolute-shading-normal-light-cosine` 一致。

## 5. Capability 与 fail-closed

- 新增公共 `REQUIRED_PATH_TRACING_CAPABILITIES = PREPARE|EVALUATE|SAMPLE|PDF`。
- 所有 `ReferenceProgramDescriptor` 构造时必须满足该集合；当前五个 source family 无例外。
- viewer source pass 的 backend 编译失败、sample/PDF 不完整或动态 MDL module 不完整时明确失败并保留上一有效 binding；不恢复 generic proposal。
- data collection provider 的 query-plane descriptor 不冒充 scattering capability；稳定文档明确两者是不同接口。

## 6. 验证设计

### 6.1 数学与 GPU contract

五个 source backend 与 neural package 都验证：

- `evaluate(wi).pdf == pdf(wi)`；
- project-owned 连续 proposal 满足 `sample.pdf == pdf(sample.wi)`；
- project-owned deterministic evaluator 或冻结同一 RNG stream 满足 `sample.weight == evaluate(sample.wi).f * absCos / sample.pdf`；
- source-native API 的 sample tuple 与 source oracle identity，并分别验证 independent `evaluate.pdf == pdf`；极窄掠射方向追加 native identity 与 tail 回归；
- forward/reverse PDF、event、方向与 weight finite/nonnegative；delta/null 走显式例外；
- mixture 的数值积分加 null probability 在预先冻结的统计置信区间内为 1。

float32 点对点容差由已有 MDL/OpenPBR oracle 与运算次数推导并在正式 run 前固定；统计检查的 sample 数、seed 与置信度写入测试源码，不根据结果回调。

### 6.2 高动态范围与 viewer

- MDL：car paint、glazed ceramic 1024 spp 回归。
- MERL：chrome 与 specular phenolic 代表尖锐 measured BRDF。
- MaterialX：低 roughness/metal 与 anisotropy rotation 的 synthetic contract fixture，加一个真实纹理资产 capture。
- LayerStack：单界面窄 roughness 与多界面 coat/slab。
- OpenPBR：coat/thin-film 与 transmission/delta 代表资产。

tail/max/isolated-neighborhood 作为诊断报告；用户要求的“随 spp 不持续增加随机孤立白点”是视觉验收。除数学一致性外，不从历史 capture 数值反推新的任意 hard threshold。

### 6.3 回归与构建

运行相关 unit、Falcor/D3D12 GPU、headless capture、Release viewer build；最后检查六个锁定 upstream worktree clean。所有运行产物进入 `artifacts/`，临时诊断脚本只放本任务 `scratch/`。

## 7. 兼容与回滚

- source snapshot、reference identity、viewer scene/capture schema 和 UI material selection 不变；这类产品兼容不意味着保留旧 shader implementation。
- shader 内部符号与 reference runtime implementation identity 会改变；已有基于旧 implementation hash 的衍生产物必须 fail closed 并重建，不伪装兼容。
- 不修改 `external/`。若新 backend 不能满足任一现有 family 的完整合同，任务回到 planning；不能以 optional capability 或 generic fallback 发布半统一状态。
- 不设置旧路径 runtime fallback。回滚只通过本任务 scoped Git commit 恢复整个迁移前版本；不 amend、不 push、不包含用户现有 dirty files。
