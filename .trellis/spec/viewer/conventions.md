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
ncls.viewer-studio@2 -> scene + slots[2] + camera + lighting + display
NclsViewer --bundle-root DIR --slot0-package ID --slot0-mode MODE --slot1-package ID --slot1-mode MODE
ncls.viewer-scene@2 -> scene/material/camera/lighting + reference bounce limits（无 spp target/batch）
interactive PT dispatch -> exactly 1 new globalSample per ready PT slot
headless PT dispatch -> min(reference_samples_per_frame, reference_spp - slot.spp)
```

## 3. Contracts

- viewer 只有 `PathTracer.cs.slang` 与 `DeferredRenderer.cs.slang` 两个 renderer。`SceneScattering` 包装 canonical source 或 package prepared state，保留 native surface/emission/颜色适配；integrator 不包含 source family 或 method key 分支。
- scene composer 只在 `materialInstanceId == activeMaterialId` 时替换为 package；其他 primary/secondary/raster hit 使用原 source。host 只验证 active binding 的 source identity。
- 两模式均整 panel 单次 dispatch，无 tile/stride/逐 tile submit。deferred 完成后缓存，变化后失效，保持 0 spp。source/package deferred 共用原 neural 的 1 个 environment、1 个 rectangle 查询预算，不通过提高预算逼近 PT；sun/point 同样执行。
- GPU timestamp 使用环形 timer；连续 dispatch 回收旧样本，停止 dispatch 后在 `Device::kInFlightFrameCount + 1` 个已结束帧后回收最后样本，不增加同步等待。capture 在已有 device wait 后回收剩余 timer，不能把缓存 deferred 显示成未计时的 0 ms。headless 最少帧数默认 1，由实际 PT spp target 决定完成，避免目标后空帧污染 wall timing。
- PT capability 必须包含 prepare/evaluate/sample/pdf（mask 15），deferred 必须包含 prepare/evaluate（mask 3）；缺失明确 Unsupported，不能隐式切模式。
- 两侧常驻标题来自已提交 binding：Reference/Neural、family/profile、PT/Deferred、spp/状态。失败保留旧 binding 时另示 Request failed；Swap sides 交换完整 slot。标题为 GUI overlay，不改变 panel 或进入线性 EXR/difference。
- scene/source specialization 变化必须先编译候选 pass，成功后更新缓存和 slot 指针；失败恢复原 scene/source/pass/metadata。程序资源可复用，编译后的 scene pass 不可跨不匹配的 specialization 复用。
- 环境 direct estimator 的 light strategy 固定取 `NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT = 4` 个样本，BSDF strategy 固定取 `NCLS_VIEWER_ENVIRONMENT_BSDF_SAMPLE_COUNT = 4` 个样本。power heuristic 两侧必须比较同一组 `n_light * p_light` 与 `n_bsdf * p_bsdf`；light estimator 除以 `n_light * p_light`，BSDF estimator 使用 native `f*cos/p_bsdf` weight 再除以 `n_bsdf`。改变任一样本数时必须同时更新两侧。
- primary hit 的 BSDF strategy pool 同时是固定 4 条完整 path samples：sample 直接 miss environment 时累计带 MIS 的 direct contribution，sample 命中几何时继续自己的 path suffix。不得把 4 个 BSDF direct samples 与额外 1 条 primary continuation 分开，否则 downstream 的多个合法 strategy 仍会共同继承单条 continuation 的长 throughput 尾部。
- secondary 及更深 hit 使用 4 light + 4 BSDF 的 environment-direct pool与 1 条独立 continuation；continuation 直接 miss environment 时不得再次累计 environment radiance。这个边界避免 direct 双计，也避免 path tree 按 bounce 指数分裂。
- path material sample 统一由锁定 Falcor 的 `UniformSampleGenerator` 通过项目 `ISampleGenerator` wrapper 产生。wrapper 只分配 pixel/path/depth/stream identity，不改变材质消费随机数的方式，也不重建 native direction/event/PDF/weight tuple。
- environment CDF 必须按 GPU radiance lookup 的重建核积分。当前线性 sampler 对一个 cell 的 texel 权重是可分离的 `[1/8, 3/4, 1/8]`；不能用 point-sampled texel luminance 建 CDF、再用 bilinear radiance 与该 PDF 做 MIS。
- 不能用 radiance、throughput 或 sample-weight clamp 修复白点。高动态范围环境的真实窄高光应通过 PDF/weight、空间邻域和随 spp 收敛诊断，与孤立 firefly 区分。
- 续路径 ray origin 根据实际 sampled direction 与 geometric normal 的符号选择法线侧，不根据 reflection/transmission event label 猜测偏移侧。
- MDL 动态 module 只装载 SDK target-code types、项目 runtime callback 与 generated target code；canonical `mdl.slang`、scene composer 和 query 是静态项目 module。
- shader 必须直接 include 自己调用的公共工具；不能依赖某个 source backend 的传递 include。纯 MDL deferred 与 PT-only 启动分别检查实际需要的 pass，未使用的模式按需创建。标题内容用显式 text 绘制，不能依赖未启用的 GUI title bar。
- 每个 slot 独立 package/mode/status/resource/timing；panel 为相同 `floor(W/2)×H`，奇数像素是 divider。错误只影响当前 slot。
- `studio-v2.json`是干净构建唯一默认preset；CMake把scene/environment/material复制到runtime data目录，loader按preset URI的filename重定位并验证hash。CLI slot选择传package ID，`--bundle-root`单独传扫描根；不得把package目录冒充ID。
- package v2的program/asset typed blob与sampler均按`usage`绑定。sampler descriptor的filter/address是运行语义，不得从resource dtype猜测或补默认值；program/asset usage冲突由loader拒绝。
- source editor 只解释 typed parameter tree。capture v4 保存 `slots[2]`，不含左右角色或可变分割位置。
- 交互 PT 不拥有 capture spp cap 或可调 batch：每次 dispatch 恰好追加 1 spp，状态不变时持续累积。`globalSample = slot.spp + sampleIndex` 是 source/package 共用的 sample identity；相机、材质或灯光实际变化才 reset 到 0，reset 后当前 dispatch 的 sample 0 必须进入 accumulation，不能因处于拖动状态而计算后丢弃。
- `reference_spp` target 与 `reference_samples_per_frame` batch 只属于 headless capture。viewer UI 与 `ncls.viewer-scene` 不保存它们；headless 最后一个 dispatch 必须按 remaining 截断。交互手工 capture 记录当前 matched slot spp，不把 1024 伪装成当前值。
- viewer scene 只接受 `ncls.viewer-scene@2`，reference 只保存 bounce/layer-walk limit；capture 只接受 v4。CLI 只接受显式 slot 参数，不保留 `--method`、capture v3、scene v1 或 lighting override 别名。旧产物用当前入口重建。

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
| 交互 slot 达到或超过 headless `reference_spp` | 继续每 dispatch 追加 1 spp，不停止、不回绕 |
| 相机拖动但当前姿态已 dispatch | 保留新姿态的 1 spp；下一次实际相机变化再 reset |
| source/package ready PT slot 在交互 capture 时 spp 不一致 | 拒绝 comparison capture，不能伪造 matched spp |
| Falcor upstream worktree 被 overlay 留脏 | build gate 失败 |
| runtime只有`studio-v2.json`而viewer读取旧preset名 | Release/headless启动失败；修正canonical runtime文件名，不依赖旧build残留 |
| slot CLI收到目录而非manifest中的package ID | 选包失败；benchmark必须同时传bundle root与ID |

## 5. Good / Base / Bad Cases

- Good：MDL、OpenPBR、MaterialX、MERL 与 LayerStack 都经 `SceneReferenceProgram` 进入同一 reference integrator，各自 state 仍执行自己的 proposal。
- Base：常量 Lambertian source 与同语义 package 在相同 scene surface 上给出一致 transport。
- Bad：`PathTracer` 按 source family 调不同自由函数；把 `4*p_light` 只用于 light side、遗漏 `4*p_bsdf`；保留额外的单条 primary continuation；用 clamp 隐藏 HDR 环境尾部。
- Bad：把 replay 的 capture batch 接到交互 UI；交互到 1024 spp 后停止；拖动时仍计算完整 batch、再通过 `gAccumulate=false` 丢弃。

## 6. Tests Required

- unit/static：slot 对称性、canonical source routing、无 family branch、4+4 MIS 两侧 multiplier、primary path-pool ownership、双线性 CDF reconstruction、实际方向 ray-origin 选择、capture schema。
- unit/static：studio v2 canonical字段、CLI slot合同、package typed blob/sampler按usage绑定与旧preset文件名denylist。
- unit/static：还必须断言交互固定返回 1 spp、host 没有 `mSamplesPerFrame`、viewer scene 没有 batch 字段、source/package shader 没有 `gAccumulate`，并保留 `globalSample = accumulatedSpp + sampleIndex`。
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

```cpp
// 错：capture cap/batch 泄漏进交互，并在拖动时丢弃已经计算的 sample。
samples = min(uiSamplesPerFrame, captureTarget - slot.spp);
accumulate = !cameraDragging;

// 对：交互是一条持续增长的正式 sample sequence；只有 headless 计算 remaining。
if (!options.headless) return 1u;
return min(options.captureSamplesPerDispatch, options.captureTargetSpp - slot.spp);
```

```powershell
# 错：把package目录传给只接受identity的slot参数。
NclsViewer --slot0-package artifacts/package-a

# 对：扫描根和manifest package_id是两个独立参数。
NclsViewer --bundle-root artifacts/exports --slot0-package <package_id>
```
