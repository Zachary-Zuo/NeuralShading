# 设计：公共 PT continuation 的法线域与 environment MIS 修复

## 1. 设计目标

本任务修的是 viewer transport，不修改任何材质族的 closure 或 native sampler。最终调用形状保持：

```text
PathSurface → common path-frame policy → backend.prepare(context, material)
            → state.evaluate()/pdf()  （direct lighting）
            → state.sample()          （direct-BSDF pool / indirect continuation）
            → common geometric-event validation / MIS / visibility
```

统一接口的边界是：材质 owner 决定 `evaluate/sample/pdf`；renderer 决定几何可见性、路径事件是否能穿过 geometric surface、light-strategy MIS 与 ray continuation。公共 transport 不读取 source family，也不重建 backend sample tuple。

## 2. 先诊断、后选择已冻结分支

### 2.1 Contribution AOV

在 task-scoped 诊断 build 中把每个 sample 的 radiance 分为：

1. primary/当前 hit 的 environment NEE；
2. BSDF-sampled direction 直接命中 environment；
3. continuation 命中 secondary surface 后的 environment NEE；
4. depth ≥ 2 的剩余 indirect；
5. emission/其它 light。

同时记录 `dot(Ns,Ng)`、`dot(wo,Ng)`、`dot(wi,Ng)`、event、`p_bsdf`、`n_light·p_light`、MIS weight 与 throughput luminance 的固定 histogram。AOV 分量和必须在 float32 累计误差内还原 beauty。诊断资源不进入 `INclsScatteringState`，完成归因后从产品 shader 移除，只把脚本、截图和报告保存在任务/artifacts。

### 2.2 判别门

- H1 成立：视觉离群点主要落在 sample event 与 geometric hemisphere 冲突、或 `Ns/Ng` 掠射偏差显著的像素；实施 3.1。
- H2 成立：离群点的 geometry/event 合法，但主要来自 BSDF-hit environment 或 secondary environment direct 长尾；实施 3.2。
- H1/H2 可同时成立，允许同时交付两项；若都不能解释，停止并回到 planning，不把其它猜测自动扩成实现范围。

## 3. 正式实现

### 3.1 公共 path-frame 与几何事件域

新增/扩展 viewer 公共 math，而不是修改每个 backend：

```slang
NclsShadingFrame nclsViewerAdjustPathShadingFrame(
    NclsShadingFrame frame, float3 geometricNormal, float3 woWorld);

bool nclsViewerIsGeometryConsistentEvent(
    NclsSurfaceInteraction surface,
    float3 woWorld,
    float3 wiWorld,
    uint eventFlags);
```

`nclsViewerAdjustPathShadingFrame()` 对照锁定 Falcor `ShadingUtils.slang::adjustShadingNormal()`：把 `Ns` 朝向 oriented `Ng`，当 `dot(wo,Ns) ≤ 0.1` 时用同一阈值向 `Ng` 平滑混合，并重新正交化 tangent。它在 source/package 两条 PT 构造 `NclsScatteringContext` 前执行；MDL/material-private normal 的适配仍由其 owner 使用同一几何输入完成。

`nclsViewerIsGeometryConsistentEvent()` 对 reflection 要求 `wo/wi` 位于 geometric surface 同侧，对 transmission 要求异侧；同时保留 shading-frame 的本地半球要求。backend 返回的 direction/event/PDF/weight 不改写：合法 sample 原样进入 throughput，不合法 sample 作为 null path 终止。ray origin 继续只按实际 `wi` 与 `Ng` 符号选侧，避免把 event label 当偏移方向。

### 3.2 对称的 environment MIS 与 primary path pool

H2 首次实现为“每个 hit 使用 4 个 light direct samples、4 个独立 BSDF direct samples，再加 1 条 continuation”。该版本修正了 `n_light=4, n_bsdf=1` 的 MIS 不对称，但新的 strategy AOV 证明底座残留亮点有 94.36% 来自第一条 continuation 命中几何后的 secondary direct；继续拆分后，其中 81.34% 来自 secondary BSDF strategy、18.66% 来自 secondary light strategy，二者相关系数为 0.919。问题不是 secondary MIS 单侧有错，而是两侧都继承了同一条罕见 primary continuation 的大 throughput。于是正式 estimator 细化为：

- primary hit 固定取 `n_light=4` 个 environment-CDF samples；
- 同一 primary hit 固定取 `n_bsdf=4` 个 `state.sample()`，但每个样本都是完整 path sample：直接 miss environment 时作为 BSDF direct strategy，命中几何时继续追踪自己的 path suffix；
- primary BSDF path 的 throughput 使用 native `f·cos/p_bsdf` weight 并除以 4；直接 miss environment 时再乘以比较 `4·p_bsdf` 与 `4·p_light` 的 power-MIS weight；
- delta BSDF direct sample 使用离散 measure，MIS weight 为 1；连续 light strategy 不冒充 delta；
- secondary 及更深 hit 固定使用 4 light + 4 BSDF 的 environment-direct MIS pool，另取一条独立 continuation。该 continuation 若直接 miss environment 不再累计 radiance，避免 direct environment 双计；
- 不在每个 bounce 继续分裂 4 条完整 suffix，因此运行上界保持静态，不形成指数 path tree。

这不是材质兼容层，而是公共 Monte Carlo estimator 的根本迁移。所有材质仍只通过 canonical `prepare/evaluate/sample/pdf` 进入 renderer，source-native sample tuple 原样使用。path sample 的随机数由 Falcor 官方 `UniformSampleGenerator` 提供，公共 wrapper 只分配稳定的 pixel/path/depth/stream identity。`ReferencePathTracer` 与 `PackagePathTracer` 共享环境 CDF、PDF、power heuristic、sample-count 常量和 sample generator；两者只保留必要的颜色空间/illumination adapter 差异。

环境 CDF 同时改为对 GPU 双线性 radiance lookup 的 cell-integrated reconstruction 建表。单个 texel 对相邻 cell 的积分权重为可分离的 `[1/8, 3/4, 1/8]`；这样亮 texel 过滤到邻格时，light PDF 不再仍按暗邻格报告。

### 3.3 交互累积与 headless capture 的调度边界

用户在性能复盘中明确纠正了原设计：交互式渲染不是可丢弃的低质量 preview，也不拥有 capture spp cap。source/package PT 使用同一条连续 sample sequence：

- 交互模式每次 dispatch 固定追加 1 spp；只要场景、相机和材质状态不变，就持续累积，不在 1024 spp 停止；
- `globalSample = accumulatedSpp + sampleIndex` 仍是唯一 sample identity，4 条 primary path suffix 继续由该 identity 派生；不创建 preview/final 两套 estimator；
- 状态变化由 `resetReference()` 把累计计数和图像一起清零；reset 后当前帧的 1 spp 是新状态的 sample 0，必须保留；
- headless capture 才读取 replay 的 batch 大小，按固定 1024 spp 目标计算 remaining，并在最后一个 dispatch 截断；
- UI 和 viewer-scene 不再持有 `Samples per frame`。capture batch 属于 headless 执行参数，不是交互材质/场景状态；
- `ReferencePathTracer` 与 `PackagePathTracer` 删除 `gAccumulate` 非累计分支，避免算完再把同一状态的 sample 丢弃。

这是一条 host scheduling 合同，不改变 4×4 MIS、native tuple、path suffix 或 scattering ABI。

## 4. 文件边界

预计正式修改：

- `apps/viewer/shaders/PathSurfaceMath.slang` / `PathSurface.slang`：公共 path-frame policy。
- `apps/viewer/shaders/PathEnvironment.slang`、`PathEnvironmentMath.slang`：共享环境 CDF、PDF 与 multiple-sample MIS。
- `apps/viewer/shaders/PathSampleGenerator.slang`：Falcor `UniformSampleGenerator` 的 canonical `ISampleGenerator` 包装。
- `apps/viewer/shaders/ReferencePathTracer.cs.slang`、`PackagePathTracer.cs.slang`：只保留通用 state 调用与各自 illumination adapter。
- `apps/viewer/CMakeLists.txt`：登记新 shared shader。
- `tests/unit/test_viewer_slots.py`：静态架构边界与两条 PT 同源检查。
- `tests/gpu/kernels/viewer_path_surface.cs.slang`、`tests/gpu/test_viewer_path_surface.py`：normal adjustment 与 event-domain oracle。
- 新增 focused GPU MIS 与 sample-generator kernel；不修改 `external/`。

## 5. 数值与视觉验证

### Hard correctness

- normal adjustment 的 normal incidence、threshold 两侧、切线正交与 finite oracle；
- reflection/transmission geometric-side truth table；
- accepted native tuple 逐字段不变，invalid event 显式 null；
- 4×4 multiple-sample MIS 在常量 environment + Lambertian/GGX fixture 上与独立积分 oracle 一致，delta 不走 continuous PDF；
- contribution AOV sum 与 beauty 一致；
- 既有五类 source 与 neural package contract suite 全部继续通过。

### Visual / report-only

- 冻结 960×540、1024 spp 的 MDL car paint、MDL ceramic、OpenPBR car paint before/after；
- ceramic bounce 0/1/2 用于确认 first-continuation signature 消失而 direct 图不漂移；
- R7 全材质矩阵记录 local-median residual、RSE、tail 与耗时，全部 report-only；
- 不以降低 max 或某个事后阈值替代用户视觉验收。

## 6. 回滚点

- 诊断 AOV 不证明 H1/H2：不进入对应正式分支，回到 planning。
- 任一正式分支破坏 native tuple identity、MIS 数学或 source/package estimator 对称性：回滚该分支，不以 fallback/clamp 发布。
- 性能变化只作为 observed cost 报告；除非固定 sample 数导致工程上无法运行，否则不根据一次 timing 自动更改预算。
