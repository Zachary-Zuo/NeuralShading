# Viewer 可执行合同

## 1. Scope / Trigger

修改 viewer slot、source reference、package binding、path tracer、deferred renderer、环境采样、MIS、capture 或动态 shader module 时适用。详见 `../project/unified-pipeline.md` 与 `../core/shared-slang-backend.md`。

## 2. Signatures

```text
ComparisonSlot[2] = {package/source binding, mode, status, resources, timing}
SceneReferenceProgram.prepare(context, sceneMaterial) -> ReferenceState
ScatteringBinding.prepare(context, packageMaterial) -> PackageState
State.evaluate/sample/pdf(...)
ncls.viewer-capture@4 -> slots[2] + single-panel EXR outputs
```

## 3. Contracts

- viewer 只有一个 source scene path tracer、一个 package path tracer 与一个 deferred renderer；source scene composer 只负责选择 concrete canonical backend，不在 integrator 中暴露 `surface.family` 或材质专用入口。
- source reference 与 package path tracer 收到同一 `PathSurface` 和 scattering context，并只调用 prepared state 的 `evaluate/sample/pdf`。package 与 source 可以有不同 transport pass，但不得有不同 scattering ABI。
- 环境 direct estimator 的 light strategy 固定取 `NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT = 4` 个样本，BSDF strategy 固定取 `NCLS_VIEWER_ENVIRONMENT_BSDF_SAMPLE_COUNT = 4` 个样本。power heuristic 两侧必须比较同一组 `n_light * p_light` 与 `n_bsdf * p_bsdf`；light estimator 除以 `n_light * p_light`，BSDF estimator 使用 native `f*cos/p_bsdf` weight 再除以 `n_bsdf`。改变任一样本数时必须同时更新两侧。
- primary hit 的 BSDF strategy pool 同时是固定 4 条完整 path samples：sample 直接 miss environment 时累计带 MIS 的 direct contribution，sample 命中几何时继续自己的 path suffix。不得把 4 个 BSDF direct samples 与额外 1 条 primary continuation 分开，否则 downstream 的多个合法 strategy 仍会共同继承单条 continuation 的长 throughput 尾部。
- secondary 及更深 hit 使用 4 light + 4 BSDF 的 environment-direct pool与 1 条独立 continuation；continuation 直接 miss environment 时不得再次累计 environment radiance。这个边界避免 direct 双计，也避免 path tree 按 bounce 指数分裂。
- path material sample 统一由锁定 Falcor 的 `UniformSampleGenerator` 通过项目 `ISampleGenerator` wrapper 产生。wrapper 只分配 pixel/path/depth/stream identity，不改变材质消费随机数的方式，也不重建 native direction/event/PDF/weight tuple。
- environment CDF 必须按 GPU radiance lookup 的重建核积分。当前线性 sampler 对一个 cell 的 texel 权重是可分离的 `[1/8, 3/4, 1/8]`；不能用 point-sampled texel luminance 建 CDF、再用 bilinear radiance 与该 PDF 做 MIS。
- 不能用 radiance、throughput 或 sample-weight clamp 修复白点。高动态范围环境的真实窄高光应通过 PDF/weight、空间邻域和随 spp 收敛诊断，与孤立 firefly 区分。
- 续路径 ray origin 根据实际 sampled direction 与 geometric normal 的符号选择法线侧，不根据 reflection/transmission event label 猜测偏移侧。
- MDL 动态 module 只装载 SDK target-code types、项目 runtime callback 与 generated target code；canonical `mdl.slang`、scene composer 和 query 是静态项目 module。
- 每个 slot 独立 package/mode/status/resource/timing；panel 为相同 `floor(W/2)×H`，奇数像素是 divider。错误只影响当前 slot。
- source editor 只解释 typed parameter tree。capture v4 保存 `slots[2]`，不含左右角色或可变分割位置。

## 4. Validation & Error Matrix

| 条件 | 必须行为 |
|---|---|
| source/package backend 缺完整 PT capability | 当前 binding unsupported/error，另一 slot 不变 |
| integrator 出现 source family/material-specific 分支 | 静态边界测试失败 |
| light/BSDF 样本数与任一 MIS PDF multiplier 不一致 | 数值/静态测试失败 |
| primary direct BSDF pool 与 primary continuation 使用不同 sample pool | 静态边界测试失败；必须以同一组 BSDF samples 拥有 miss 与 hit suffix |
| environment CDF reconstruction 与 GPU radiance filtering 不同 | host/static 与 GPU estimator 测试失败 |
| sample 为 null、非有限或连续 PDF 非正 | 终止当前路径，不 fallback |
| shader/resource 创建失败 | 保留上一有效 binding，报告当前 slot error |
| capture 未达到 path-tracing 目标 spp | 不导出正式 EXR |
| Falcor upstream worktree 被 overlay 留脏 | build gate 失败 |

## 5. Good / Base / Bad Cases

- Good：MDL、OpenPBR、MaterialX、MERL 与 LayerStack 都经 `SceneReferenceProgram` 进入同一 reference integrator，各自 state 仍执行自己的 proposal。
- Base：常量 Lambertian source 与同语义 package 在相同 scene surface 上给出一致 transport。
- Bad：`ReferencePathTracer` 按 source family 调不同自由函数；把 `4*p_light` 只用于 light side、遗漏 `4*p_bsdf`；保留额外的单条 primary continuation；用 clamp 隐藏 HDR 环境尾部。

## 6. Tests Required

- unit/static：slot 对称性、canonical source routing、无 family branch、4+4 MIS 两侧 multiplier、primary path-pool ownership、双线性 CDF reconstruction、实际方向 ray-origin 选择、capture schema。
- GPU：公共 `PathSurface`、source backend sample→pdf/weight、package ABI、continuous/delta MIS math 与 Falcor path sample generator。
- Release：只用 `scripts/build_viewer.ps1 -Configuration Release` 编译真实 scene specialization，结束后 Falcor clean。
- headless：MDL car paint/ceramic 及其他 source 做 1024 spp capture，检查 finite、identity、RSE/high quantile 与局部 firefly 结构。

## 7. Wrong vs Correct

```slang
// 错：event label 决定 ray offset，可能与 smooth-shading 下实际方向相反。
origin = isTransmission ? position - geometricNormal * epsilon
                        : position + geometricNormal * epsilon;

// 对：实际 sampled direction 决定离开表面的哪一侧。
origin = nclsViewerDirectOrigin(position, geometricNormal, scatter.directionWorld);
```

```slang
// 错：light-sampled 一侧使用 n_light*p_light，BSDF-hit 一侧仍使用 p_bsdf。
// 对：两个 heuristic 比较同一对 technique PDF。
float lightTechniquePdf = float(NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT) * pLight;
float bsdfTechniquePdf = float(NCLS_VIEWER_ENVIRONMENT_BSDF_SAMPLE_COUNT) * pBsdf;
```
