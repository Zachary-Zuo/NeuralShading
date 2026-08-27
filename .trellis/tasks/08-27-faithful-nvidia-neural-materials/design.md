# 忠实复现 NVIDIA Neural Materials：技术设计

## 1. 设计结论

本任务交付的是作者公开方法的 **functional reproduction**：公开的 encoder、z8 hierarchy、learned-frame evaluator、two-lobe analytic sampler、训练规模与 runtime fetch 都进入可执行合同；作者未发布的资产、stage 切换点和 estimator 细节以显式 recipe choice 冻结。它不会把当前 MaterialX 或 LayerStack 伪装成论文原始 shader graph，也不会把短程 smoke 标成 formal result。

实现继续只有一条统一路径：

```text
SourceSnapshot + native feature layout
  → ReferenceQueryStream
  → role-aware TrainingBatch@1 set
  → TrainingRunner
  → TrainingCheckpoint@2
  → NvidiaMethodDefinition
  → ScatteringPackage@1 (FP16 weights + 2×RGBA16F mip chains)
  → ScatteringBinding
  → ComparisonSlot[2] (PT | deferred)
```

不新增 NVIDIA runner/exporter/viewer 分支。公共层只获得能被 fixture/reference program 复用的 role request、typed texture resource、binding 与双 slot 能力；NVIDIA 的网络、lifecycle、feature interpretation 和 sampler 数学仍由 `NvidiaMethodDefinition` 私有拥有。

## 2. 一手 correspondence 与复现身份

### 2.1 四种差异标签

| 标签 | 含义 | 本任务典型项 |
| --- | --- | --- |
| `faithful` | 与论文/补充材料公开定义相同 | z8、`64×4→8` encoder、2 learned frames、3×64 evaluator、3×32 sampler、300k/2×65k、mollification、two-lobe warp |
| `source-domain adaptation` | 保留 source 原生 GT，但 source 不是作者资产 | MaterialX standard_surface spatial route、LayerStack 1×1 route |
| `runtime-contract adapter` | 只桥接项目 ABI，不改变方法分布 | response-cos → linear `f`、reverse PDF、null/error 语义、`prepare` cache |
| `author-underspecified` | 作者没有公开足以逐 bit 复现的信息 | stage boundary、log offset、完整 KL estimator、latent container、初始化 seed |

formal identity 使用三层字段，全部进入 config/checkpoint/provenance hash：

- `correspondence_id = nvidia-rta2024-functional@1`；
- `source_adaptation_id = materialx-standard-surface-spatial@1 | layer-stack-uniform-1x1@1`；
- formal 使用 `recipe_id = nvidia-rta2024-materialx-formal-300k-stage100k@1`；它覆盖网络 variant、总步数、双 route batch、optimizer、mollification、filter、stage boundary、log loss、KL estimator与所有 seed。smoke 使用不同 identity。

`smoke/profile/adapted/formal` 是互斥 run class。实现满足 correspondence、数学、lifecycle 和部署合同时可称为 `functional reproduction`；经验结果必须另行声明实际 checkpoint/budget。本次用户冻结的 200k 结果不得使用“300k formal protocol完成”标签。

### 2.2 作者未公开项的冻结策略

- **stage boundary**：2024 论文没有公开精确切换点。后续作者公开实现的默认 BSDF encoder phase 为 100k steps，因此本项目把 `materialization_step=100000` 作为 author-code-informed choice 冻结进 formal recipe；它不是对 2024 论文隐藏训练代码的事实宣称，formal run 中不动态改值。
- **log loss**：先保留 `L1(log1p(response), log1p(target))` 作为 `log1p-l1@1`，明确是 reproduction choice，不写成作者公式。
- **sampler KL**：2024 论文没有公开完整估计式。后续作者公开训练源码从当前 learned proposal `pθ` 取样，并使用 score-function 形式 `-stopgrad(L(fθ·cos)/pθ)·log pθ`；本项目采用同一估计结构，sample direction、learned response、分母 PDF 和 latent 全部 detach，只有 `log pθ` 向 sampler head 传梯度。identity 为 `learned-sampler-forward-kl-score@1`，这是 author-code-informed choice；若新一手证据否定它，必须升 recipe 版本。
- **shader path**：补充材料 regular FP16 pseudocode 是 correctness 基准；不声称复现未公开 tensor-core intrinsic。

## 3. Source-native feature 与 spatial 训练

### 3.1 NativeFeatureLayout

为 reference query 增加 source-owned `NativeFeatureLayout` 描述：字段名、原生路径、通道数、filter rule、normal/tangent 表示和 layout hash。它只描述“怎样在某个 snapshot 的 UV/LOD 处取原生参数”，不是公共材质编辑 schema，也不把不同 source 归约成 LayerStack。

`TrainingBatch@1.tensors` 保留当前基础查询字段，并允许 route request 声明附加字段：

- `uv [B,2]`、`uv_dx/uv_dy [B,2]`、`mip_level [B]`；
- `native_features [B,K]` 与 `native_feature_layout_id`（identity 在 provenance/config 中，tensor 只存值）；
- formal route 使用 `B=65_000, direction=1`，每条记录拥有独立 UV、固定方向和 queried direction，不再用一个 `wo` 共享 64 个方向伪装 65k samples；
- evaluator 与 sampler route 的 `stream_id/request_id/rng_seed` 必须不同。单 route fixture 继续证明 runner 没有硬编码 NVIDIA。

`TrainingRouteRequest` 由 MethodDefinition 产生，包含 route name、batch size、global step、direction proposal、target estimator、filter recipe、mollification参数和需要的 tensor fields。`BatchSource.next_batch(request)` 是唯一 producer 入口；offline/live 返回同一 `TrainingBatch@1` envelope。公共 runner 只遍历命名 route 并把 batch mapping 交回 method objective。

### 3.2 MaterialX spatial route

MaterialX 采用已有 resolved standard_surface 参数、纹理与官方 reference：

- UV 均匀采样；方向按 Rusinkiewicz half/difference vectors 均匀采样；
- MIP level 按冻结的指数分布采样；在对应 Gaussian footprint 内按面积比例取 spatial samples并在线平均 reference；
- coarse level 的 native parameters 使用 source adapter 的预过滤结果；normal/tangent 使用 LEAN-style 一、二阶矩，不把 normal RGB 直接平均后当作作者精确实现；
- directional mollification在前 20k global steps按补充材料 cosine 公式从 10° 到 0°，每个 target 用 256 cone directions；
- Falcor live executor 使用 flat GPU query buffer和至少两个 in-flight lease slot；UV、方向、mip、Gaussian footprint、native-feature filter 与 mollification 都在 CUDA 生成，reference 通过 shared buffer 返回，两个 65k route 都不经 host/NumPy response readback。

MaterialX standard_surface 不是论文复杂 layered asset，因此只证明公开方法在真实 spatial/texture/footprint 路径上的功能忠实性。其资源和参数继续属于 MaterialX snapshot identity。

### 3.3 LayerStack uniform route

LayerStack adapter从 canonical native interfaces/media/masks构造固定 layout，random-walk reference不变。空间域明确为 1×1、所有 footprint fetch 同一个 z8；仍执行 encoder bootstrap、materialization 和 finetune，以证明 source-native/生命周期合同没有 LayerStack 特例。该 route 永远标为 `layer-stack-uniform-1x1@1`，不能作为 spatial filtering 证据。

## 4. 模型与训练 lifecycle

### 4.1 模型

`NvidiaNeuralAppearanceModel` 变为每次 run 对一个 snapshot/layout：

- encoder：`K→64→64→64→64→8`，hidden ReLU；
- frame projection：`8→12`，无 bias/activation；
- evaluator：论文正式最大形态 `20→64→64→64→3`，hidden ReLU，输出 `exp(raw-3)`；
- sampler：`11→32→32→32→9`，保持补充材料 raw order/warp；
- latent hierarchy：语义上每层两个 trainable RGBA FP32 master texel plane；训练实现把所有 mip texel 平铺为一个 FP32 parameter 并用 mip offset 做四点 gather，以减少 Python/kernel 开销，部署时仍还原为两张 RGBA16F mip chain。该平铺只改变存储容器，不改变 texel、filter 或 Adam 更新语义。

训练 core 用 FP32 master。Phase 1 直接把 prefiltered `native_features` 送进 encoder并绕过 texture；边界 materialization 对所有 mip texels运行 encoder；Phase 2 删除/冻结 encoder，按 discrete mip + bilinear fetch 直接优化 latent texels。两个阶段都同时优化 evaluator 与 sampler，不再出现 evaluator-only/sampler-only phase或重置 optimizer。

### 4.2 Runner、optimizer 与恢复

公共 runner 维护一个 Adam 和一个 global cosine scheduler贯穿两阶段：`β=(0.9,0.999)`、`eps=1e-7`、weight decay 0、`lr 1e-3→1e-4`、total 300k。每步顺序固定为：

1. 生成 evaluator/sampler 两个独立 route request；
2. 获取两个 live/offline batch；
3. 计算 evaluator log loss和 detached-latent sampler loss；
4. 一次 backward/finite-gradient check/optimizer step/scheduler step；
5. 更新 validation、checkpoint、work units、吞吐、显存与 ETA。

checkpoint 增加 lifecycle state、optimizer、scheduler、所有 Python/NumPy/Torch/CUDA RNG、每条 BatchSource stream RNG/request counter、latent materialization identity和 validation state。resume 必须在 route sequence、LR、stage 与 batch seed 上与未中断轨迹一致。原子写与 SHA-256 sidecar 保持不变。

## 5. Runtime 与 package

### 5.1 Latent resource

material compiler不再写单个 z8 record，而是写：

- 一个小 `CompiledMaterial`：latent extent、mip count、resource slots、layout version；
- `latent0.dds`、`latent1.dds`：完整 RGBA16F mip chain；
- descriptor `dtype=texture2d-rgba16float-dds@1`、`shape=[width,height,mips,4]`、`stride=8`、`alignment=16`、`usage=<shader binding>`。

DDS 是通用 typed texture container；writer只保存并 hash bytes，C++ loader按 descriptor vocabulary创建 texture，不按 NVIDIA method ID分支。坏 header、extent/mip/format/usage不符全部拒绝 binding。

### 5.2 Shader fetch 与精度

`prepare(context, material)`：

1. 由 `uvDx/uvDy` 和 base extent计算 supplemental Listing 1 的 isotropic LOD；
2. 用 sample generator 的一个 `u` 在相邻 mip 间 stochastic selection并 clamp；
3. 在选中 mip 对两张 texture 做 bilinear fetch得到 z8；
4. 计算 learned frames、固定 view 投影与 sampler raw，打包为可复用 state。

运行时权重、latent与 MLP accumulator/intermediate 均走 functional FP16 identity，公共输出转回 float。`evaluate()` 把网络 response-cos 适配成线性 `f`；`sample()` 精确使用补充材料的 2D `u` mixture select/remap 和 two-lobe proposal；`pdf()` 返回同一完整 mixture。项目需要的 reverse PDF通过交换方向重新 prepare sampler参数，是 `runtime-contract adapter`。

formal sampler不包含固定 `1/32` cosine safety lobe。若未来需要 robust variant，必须使用不同 runtime/recipe identity，本任务不保留双重含义。

## 6. Generic ScatteringBinding 与 neural PT

### 6.1 Binding

`ScatteringPackage` C++ loader完整校验 schema、三个 identity、hash、typed descriptors、validation/provenance和 module closure，产生 CPU `ViewerProgram`；`ScatteringBinding` 再通用创建：

- runtime/material structured buffers；
- material textures与 sampler；
- package absolute module specialization；
- parity、deferred 和 PT passes；
- 独立 state/accumulation/timing/capability status。

资源按 descriptor `usage`/shader reflection绑定。viewer CMake不列 NVIDIA 文件，package module closure是唯一 method shader来源。

### 6.2 两个对称 slot

`NclsViewer` 使用真正的 `ComparisonSlot[2]`：每侧独立选择已验证 package或显式 `source-reference` 请求，以及 capability支持的 `path-tracing|deferred` mode。`source-reference` 保持内建权威 source transport边界，不虚构磁盘 package身份。panel固定 `floor(W/2)`；slot失败只影响自身。capture/replay使用 `ncls.viewer-capture@4 slots[2]`。

### 6.3 Package path tracer

新增/重构一个只依赖 scattering ABI 的 package PT：

- 每次 surface hit从 Falcor scene vertex得到 position、shading/geometric frame、UV和 material instance；ray cone传播得到 `uvDx/uvDy`；
- 调用当前 slot binding的 `prepare`，直接光用 `evaluate/pdf`，续路径用该 state 的 `sample`；
- MIS、throughput和 null/invalid 只读公共 `NclsScatteringEval/Sample/Pdf`，不解释 source family、closure或 NVIDIA state；
- 所有 neural/method package走同一 package transport代码，差异只在 binding；内建 `source-reference` 是用于GT对照的独立权威 transport请求，不把它误报成 package。

deferred路径从已有 `texCoord/texCoordGrad` 填完整 context并调用同一 package backend。受控测试会给 reference 和 neural sampler不同的确定性行为，证明 neural PT不是 reference PT的 UI 别名。

## 7. 验证设计

### 7.1 CPU/unit

- exact-vector：encoder、frame、input order、3×64 evaluator、3×32 sampler、warp、2D remap、response adapter；
- lifecycle：bootstrap/materialize/finetune、joint gradient ownership、双 stream independence、global scheduler、resume trajectory；
- TrainingBatch role/flat 65k shape、native layout identity、offline/live一致性；
- DDS/typed resource roundtrip与 tamper matrix；single-route fixture保证公共 runner无 NVIDIA branch；
- checkpoint formal/smoke identity不可互换。

### 7.2 GPU/Slang/Falcor

- PyTorch FP32 core、packed FP16 SlangPy与 Falcor parity，tolerance由固定 calibration记录；
- sampler normalization、sample→pdf、hemisphere/null/invalid、finite-difference gradient oracle；
- MaterialX spatial和 LayerStack 1×1各一条 live step，无 host readback；
- latent LOD/stochastic adjacent mip使用人工 mip colors做 exact distribution测试；
- package shader从绝对 package路径编译，typed resources实际绑定。

### 7.3 Viewer/训练证据

- Release viewer双 slot覆盖 reference/neural × PT/deferred；camera/scene固定时 neural PT触发 package sampler；capture/replay identities一致；
- formal前跑显存/吞吐 preflight和短程 smoke；冻结的 300k recipe 保存初始化、阶段边界与周期 validation。本次按用户决定以 step 200k checkpoint 导出最终记录 package，并显式排除更晚日志；
- 报告 directional/energy/visual quality、sampler correctness、单次 query/prepare成本、显存、package bytes与实际训练耗时，不设未经用户确认的绝对 quality kill gate。

## 8. 风险与回滚边界

- **65k×2 + 256 cone显存/吞吐**：先实现 flat tiled dispatch与双 buffer pool，preflight只调整 tile/dispatch实现，不改变 logical batch/recipe；若仍不可承载，保留证据并停止 formal claim。
- **MaterialX GPU native feature过滤**：先锁定最小现有 textured standard_surface snapshot；任何 LEAN approximation都进入 layout/recipe identity。不得 CPU/NumPy fallback。
- **viewer PT重构风险**：先用 fixture package使 generic PT通过，再接 NVIDIA；每阶段保持 reference-only viewer可构建，禁止修改 `external/Falcor`。
- **作者未公开 KL/stage**：通过 correspondence + recipe version隔离。新证据只能升版本，不能改写已完成 artifact。
- **长训练中断或用户冻结记录点**：checkpoint必须先用 interrupted-vs-uninterrupted测试证明恢复，再允许 formal run；若用户改变记录预算，保留原 config/checkpoint identity，并把实际停止点、排除尾段和 protocol completion边界写入结果，不伪造完成态。
