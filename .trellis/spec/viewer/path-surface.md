# Viewer path surface 合同

## 1. Scope / Trigger

凡是修改 viewer 的 scene PT 命中解码、UV、frame、front-facing、ray cone、texture/latent footprint，或新增 source/package PT renderer，都必须遵守本合同。目标是让 source reference PT 与任意 package PT 在材质私有 `prepare()` 之前收到同一份 scene surface 数据；deferred 则以 raster `ddx/ddy` 提供同单位的 normalized UV gradient。

## 2. Signatures

公共实现位于 `apps/viewer/shaders/PathSurface.slang`、`PathSurfaceMath.slang` 与 `SceneSurface.slang`：

```slang
bool nclsViewerLoadPathVertexData(
    HitInfo hit, Ray ray, out VertexData vertex, out uint materialId);

NclsViewerPathSurface nclsViewerPreparePathSurface(
    VertexData vertex, uint materialId, float3 viewWorld,
    float3 rayDirection, float rayConeWidth, uint flipTexCoordV);

NclsViewerPathSurface nclsViewerPreparePathSurfaceFields(
    float3 position, float3 faceNormal, float3 normal, float3 tangent,
    float2 uv, float triangleUvWorldLog2Scale, uint materialId,
    float3 viewWorld, float3 rayDirection, float rayConeWidth,
    uint flipTexCoordV);

float nclsViewerPrimaryRayConeSpreadAngle(
    float3 cameraV, float3 cameraW, uint frameHeight);

NclsViewerPathUvFootprint nclsViewerPathUvFootprint(
    float triangleUvWorldLog2Scale, float rayConeWidth,
    float3 rayDirection, float3 geometricNormal);
```

reference/package PT 只能调用这些入口，不能在各自 shader 中复制 `getVertexDataRayCones()`、frame 整理或 footprint 公式。

## 3. Contracts

- `nclsViewerSurfaceContext(surface, wo, filterRandom)` 是 PT/raster 共用的最终 context 构造。`nclsViewerDecodeRasterSurface` 在读取边界解码：material buffer 的 0 是背景，真实 ID 为 stored-1；depth<0 同样无效。
- raster 的 geometric normal 存入 normal.w/tangent.w/viewDirection.w，front-facing 存入 texCoord.z。normal/tangent/frame/V flip 采用同一 `PathSurfaceMath`；不能用 shading normal 假装 geometric normal。
- deferred 的环境方向以 prepared state 的 shading frame 构造，保留 source normal-map 的语义。PT 的 ray cone 与 raster ddx/ddy 可以有不同 footprint；同输入 witness 才是严格材质 parity。


- `cameraU/cameraV/cameraW` 共享 Falcor 的 `focalDistance` 尺度。primary ray 会归一化方向，ray-cone spread 必须使用 `length(cameraV) / length(cameraW)`；共同缩放三个 camera basis vector 不得改变 footprint。
- `rayConeWidth` 是世界空间长度；`coneTexLODValue` 是三角形 UV/world Jacobian 的 `log2` 尺度；`NclsViewerPathSurface.uvDx/uvDy` 是 normalized UV derivative，调用方不得再次除以或乘以纹理尺寸。
- triangle mesh 必须用 Falcor `getVertexDataRayCones()`。displaced triangle、curve 与 SDF 当前没有同精度 Jacobian，显式标记 invalid differential，再由公共 math 降级到 full-UV footprint；不得消费未初始化值。
- 无效/退化 differential 产生一个完整 UV period 的有限 footprint，而不是 NaN、固定 texel 或任意 sharp mip。
- GLB UV 不翻转；只有 scene format policy 令 `flipTexCoordV != 0` 时才翻 V。frame 先按 view orientation 处理 geometric/shading normal，再把 tangent 正交化；退化 tangent 使用 `nclsViewerTangent()`。
- 续路径 origin 必须由 sampled direction 与 geometric normal 的符号选择偏移侧；不得根据 backend 报告的 reflection/transmission event label 猜测偏移侧。event 仍用于 MIS/delta 等 transport 语义，不承担几何自相交修正。
- deferred 的 `SceneVisibility.3d.slang` 继续输出 raster UV 与 `ddx/ddy`；这些梯度与 PT helper 输出采用同一 normalized UV 单位，但 transport 不互相冒充。

## 4. Validation & Error Matrix

| 条件 | 必须行为 |
|---|---|
| camera basis 共同缩放 | spread、LOD、UV gradient 不变 |
| 分辨率升高 | 同视角/距离 footprint 变细 |
| 距离增加或投影更掠射 | footprint 允许变粗，结果保持有限 |
| triangle differential 有限 | 使用 Falcor triangle Jacobian |
| 非 triangle 或 differential 非有限 | 使用有限、保守的 full-UV fallback |
| source/package PT 出现不同 UV/frame 构造 | 构建或 source-contract 单测失败，不接受复制修补 |
| event label 与实际方向所在几何半球不一致 | ray origin 仍按实际方向选侧，禁止 event-driven offset |
| slot `ready` 但真实纹理为平均色 | 视觉验收失败，必须检查 raw UV/LOD，不以 ready/parity 代替 |

## 5. Good / Base / Bad Cases

- Good：`american_walnut_veneer` 在 source/neural PT 中显示同位置的木纹，neural PT/deferred 在匹配局部光下保留同一空间布局。
- Base：常量材质不产生人为纹理；非 triangle geometry 使用明确 fallback，仍保持有限和可运行。
- Bad：只用 `2 * length(cameraV) / height` 计算 spread。它会把 Falcor 默认 `focalDistance` 约一万的共同尺度误当成角度，使所有采样退化到最粗 mip。

## 6. Tests Required

- `tests/unit/test_viewer_slots.py`：唯一 PT 与 raster decode 共用 helper，且不直接调用 `getVertexDataRayCones()`；续路径 origin 使用实际 sampled direction，不用 event label 选侧。
- `tests/gpu/test_viewer_path_surface.py`：在实际 Slang/GPU 上断言 UV/V flip、frame/front-facing、camera basis scale invariance、分辨率/距离/掠射单调性、有限 fallback 与明确数值 oracle。
- `scripts/build_viewer.ps1 -Configuration Release`：编译 reference/package PT 的真实 scene specialization。
- headless local-light capture：用明显空间结构和真实 walnut/denim 资产检查 raw EXR 与 display；环境 PT/deferred 的 transport 差异不能被误判成 surface contract 差异。
- 构建后 `external/Falcor` 必须干净。

## 7. Wrong vs Correct

```slang
// Wrong: cameraV 包含 focalDistance 的任意共同尺度。
float spread = 2.0f * length(cameraV) / float(frameHeight);

// Correct: 比值恢复 image-plane slope，再换算到单像素。
float spread = 2.0f * length(cameraV) / max(length(cameraW), 1e-8f)
    / float(max(frameHeight, 1u));
```

Wrong：reference/package 各复制一次 vertex/frame/LOD 逻辑，再靠截图发现漂移。

Correct：source/package 都经 `nclsViewerLoadPathVertexData()` 与 `nclsViewerPreparePathSurface()`，材质私有逻辑从公共 surface 之后开始。

Wrong：`sample.eventFlags` 标为 transmission 就固定沿 `-geometricNormal` 偏移，即使实际 sampled direction 在另一侧。

Correct：用 `dot(scatter.directionWorld, geometricNormal)` 的符号选择偏移侧；frame/event 只描述散射语义。
