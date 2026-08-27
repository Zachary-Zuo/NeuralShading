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
- 环境 next-event estimation 每个 surface hit 固定取 `NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT = 4` 个独立样本。light-sampled MIS 使用 `n * p_light`，BSDF-hit 环境 MIS 同样使用 `n * p_light`；改变样本数时必须同时更新两侧，保持 multiple-sample MIS 无偏。
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
| NEE 样本数与任一 MIS PDF multiplier 不一致 | 数值/静态测试失败 |
| sample 为 null、非有限或连续 PDF 非正 | 终止当前路径，不 fallback |
| shader/resource 创建失败 | 保留上一有效 binding，报告当前 slot error |
| capture 未达到 path-tracing 目标 spp | 不导出正式 EXR |
| Falcor upstream worktree 被 overlay 留脏 | build gate 失败 |

## 5. Good / Base / Bad Cases

- Good：MDL、OpenPBR、MaterialX、MERL 与 LayerStack 都经 `SceneReferenceProgram` 进入同一 reference integrator，各自 state 仍执行自己的 proposal。
- Base：常量 Lambertian source 与同语义 package 在相同 scene surface 上给出一致 transport。
- Bad：`ReferencePathTracer` 按 source family 调不同自由函数；把 4-sample light PDF 只用于 direct MIS、遗漏 BSDF-hit MIS；用 clamp 隐藏 HDR 环境尾部。

## 6. Tests Required

- unit/static：slot 对称性、canonical source routing、无 family branch、4-sample MIS 两侧 multiplier、实际方向 ray-origin 选择、capture schema。
- GPU：公共 `PathSurface`、source backend sample→pdf/weight、package ABI。
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
// 错：light-sampled 一侧使用 n*p_light，BSDF-hit 一侧仍使用 p_light。
// 对：multiple-sample MIS 的两个 heuristic 都使用同一个 n*p_light。
float lightTechniquePdf = float(NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT) * pLight;
```
