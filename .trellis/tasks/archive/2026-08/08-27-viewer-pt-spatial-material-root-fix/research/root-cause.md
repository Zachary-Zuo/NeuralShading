# 根因审计记录

## 已确认事实

- `american_walnut_veneer` 的 base color、roughness、normal 纹理均成功加载为 4096×4096、13 mip；MaterialX source 参数中的纹理启用标志也正确。
- walnut 与 `denim_fabric` 的 source-reference PT 在高 spp 下仍只呈现平均色，denim 的大尺度接缝也消失，因此不是单纯的纹理对比度或采样不足。
- 同一个 200k neural package 在 deferred 路径保留明显空间变化，而 PT 路径退化为近似常量。这把问题定位到 reference/package PT 共享的命中 surface 数据路径，而不是训练、package 导出或 latent 内容。
- 场景 GLB 的两个 mesh 都含 `TEXCOORD_0`，UV 覆盖接近完整 `[0,1]^2`；离线计算的三角形 UV/world Jacobian 有限，且不存在退化 UV 三角形。
- 两个 PT shader 都通过 `getVertexDataRayCones()` 读取 `vertex.texC` 与 `coneTexLODValue`，并用同一公式构造各向同性 `SampleGrad` footprint；当前自动测试没有覆盖真实 ray hit 后的 UV/LOD 空间变化。

## 待判定分支

首命中 probe 直接输出 `(uv.x, uv.y, texture-independent lambda)`，绕过 BSDF、灯光、tone mapping 和材质纹理：

1. 若 UV 在命中区域内退化为常量，则修复 ray-hit vertex/attribute 读取合同；
2. 若 UV 正常而 `lambda` 异常偏大，则修复 ray-cone footprint 的单位或传播；
3. 若二者都正常，再检查 MaterialX/package 的 `SampleGrad` 消费合同。

probe 仅用于形成可复核的根因证据；最终产品代码会改为共享的 PT surface helper，并由自动回归覆盖。

## Probe 结果与直接根因

raw probe 产物已按仓库政策保存到 `artifacts/nvidia-faithful/materialx-recorded-200k/viewer-reference-neural/root-cause-probes/`，task scratch 只保留可复现脚本与 replay。

第一轮 raw EXR 首命中 probe 的 UV 在可见区域覆盖约 `[0.003, 0.96]`，两个方向标准差分别约 `0.29/0.25`，因此 ray-hit UV 没有退化为常量。texture-independent LOD 却落在约 `1.18..8.58`，中位数 `4.55`，足以让 4K MaterialX 纹理和 neural latent 都选择最粗 mip。

第二轮把 LOD 拆为三项：

| 项 | 中位数 | 判定 |
|---|---:|---|
| `coneTexLODValue` | -2.79 | triangle UV/world Jacobian 正常 |
| `log2(rayConeWidth)` | 6.73 | 异常，ray cone 宽约 106 个世界单位 |
| `log2(normalProjection)` | -0.60 | 视角投影正常 |

直接根因是 primary spread 使用了 `2 * length(cameraV) / height`。Falcor 的 `cameraU/cameraV/cameraW` 共同乘有 `focalDistance`（默认量级约一万）；ray direction 的 `normalize()` 隐藏了该尺度，而 ray cone 没有归一化，导致 footprint 被放大约一万倍。正确的 image-plane slope 是 `length(cameraV) / length(cameraW)`。

source reference PT 与 neural package PT 复制了同一错误公式，所以 MaterialX texture 与 NVIDIA latent 同时只读最粗 mip。deferred 使用 raster `ddx/ddy`，不经过该公式，因而一直保留空间变化。训练、checkpoint、package identity 与纹理加载均不是本次根因。

## 根本修复

- `PathSurface.slang` 统一 triangle/displaced/curve/SDF 命中解码、UV/V flip、frame、front-facing 与 footprint 构造；reference/package PT 不再保留各自副本。
- `PathSurfaceMath.slang` 用 camera basis 长度比构造 primary spread，并明确 normalized UV derivative 单位。
- triangle 使用 Falcor `getVertexDataRayCones()`；无 triangle Jacobian 或非有限 differential 时走有限、保守的 full-UV fallback。
- GPU oracle 锁定 camera basis 共同缩放不变量，以及分辨率、距离、掠射角的变化方向；强制最粗 mip或恢复旧公式会直接失败。

## 修复后证据

- `denim-fixed-32spp.json`：reference PT 恢复大尺度接缝、织物 base color 与 normal 细节；局部 gradient RMS 从旧 capture 的约 `0.00160` 增至 `0.00658`（report-only）。
- `walnut-direct-fixed-64spp.json`：1280×720 local-light 下，source reference 与 200k neural PT 都显示稳定木纹；亮度布局相关约 `0.993`、梯度相关约 `0.939`（report-only）。
- `walnut-neural-pt-deferred-fixed-64spp.json`：同一 neural package 的 PT/deferred 亮度布局相关约 `0.995`、梯度相关约 `0.903`（report-only）；边缘与阴影差异来自 transport/raster coverage，不作为 evaluator hard gate。

## 旧验收缺口

旧证据验证了纹理加载、package hash/ABI、direction parity、slot `ready` 与平均输出，却没有让真实 scene hit 的 UV/LOD 进入硬门。低 spp walnut 又把“最粗 mip 的平均色”误解成噪声或资产本身低对比。以后每次涉及 spatial material 的 viewer 收尾必须同时包含 shader 数值 oracle、明显低频结构的 local-light capture 和真实资产视觉证据；单纯 ready、平均色或 tone-mapped screenshot 均不足以证明空间语义成立。
