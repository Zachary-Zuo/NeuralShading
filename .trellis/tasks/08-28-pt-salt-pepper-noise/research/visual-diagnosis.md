# 视觉诊断：PT 细碎亮点

## 1. 观察方法

所有对照保持相同 shaderball、相机、`studio_small_03_1k.exr`、`exposure_ev=-0.5` 与 1024 spp。为了避免旧 capture 的单 slot 只有 160×240 像素掩盖空间结构，本次把 composite 提升到 960×540、单 slot 提升到 480×540；display PNG 只用于视觉分类，float32 `capture-slot-0.exr` 仍是权威线性结果。

## 2. 视觉证据

| 材质 / 对照 | 产物 | 观察 |
|---|---|---|
| OpenPBR car paint，bounce 4 | `artifacts/diagnostics/pt-salt-pepper-noise/highres-960x540-1024spp/` | 单像素/极小簇亮点密集在底座蓝色掠射区，球体轮廓附近零星出现；连续主高光正常。 |
| MDL car paint，bounce 2 | `artifacts/diagnostics/pt-salt-pepper-noise/mdl-carpaint-960x540-1024spp/` | 与 OpenPBR 相同的底座空间签名；不能只解释为 OpenPBR sampler。 |
| MDL ceramic，bounce 2 | `artifacts/diagnostics/pt-salt-pepper-noise/mdl-ceramic-960x540-1024spp/` | ceramic 没有 flakes closure，仍出现同类亮点，排除 car-paint flake texture。 |
| MDL ceramic，bounce 0 | `artifacts/diagnostics/pt-salt-pepper-noise/mdl-ceramic-bounce0-960x540-1024spp/` | primary-hit environment NEE 基本干净，底座没有对应密集亮点。 |
| MDL ceramic，bounce 1 | `artifacts/diagnostics/pt-salt-pepper-noise/mdl-ceramic-bounce1-960x540-1024spp/` | 一开放第一条 BSDF continuation，底座亮点立即出现。 |

放大使用 nearest-neighbor，只复制原像素而不平滑：

- `artifacts/diagnostics/pt-salt-pepper-noise/carpaint-slot0-nearest-4x.png`
- `artifacts/diagnostics/pt-salt-pepper-noise/carpaint-lower-nearest-8x.png`
- `artifacts/diagnostics/pt-salt-pepper-noise/highres-960x540-1024spp/base-nearest-3x.png`
- `artifacts/diagnostics/pt-salt-pepper-noise/highres-960x540-1024spp/sphere-nearest-2x.png`

## 3. 当前分类

这不是严格意义上的黑白椒盐，而是以正向亮离群值为主的 firefly。它不是 half EXR 溢出：当前权威 EXR 是 float32 且 finite；也不是 display tone mapping 产生，因为原像素在空间上与 glossy continuation 区域一致。旧任务只统计 `>100` component，会漏掉大量低于该阈值但经 tone mapping 后肉眼明显的局部亮点。

## 4. 竞争假设

1. **H1：shading normal 与 geometric normal 的路径域不一致。** 项目当前把 backend 放在 shading frame 中 sample/evaluate，却只在 ray-origin offset 使用 geometric normal；复杂底座处缺少 Falcor PathTracer 已有的 shading-normal grazing adjustment 与 reflection/transmission geometric hemisphere check。空间位置和 bounce 0/1 分界支持该假设。
2. **H2：几何有效的 one-bounce environment estimator 方差。** 现有 primary hit 使用 4 个 environment-light samples，却只有一条 BSDF continuation 同时承担 environment hit 与 indirect path；极窄 glossy lobe、HDR texel 与遮挡组合仍可能留下长尾。bounce 1 分界同样支持该假设。
3. **H3：RNG 或显示端问题。** 跨材质空间稳定、float32 raw 与 bounce 隔离使其优先级较低，但在 H1/H2 证据不足时仍需检查 sample 维度相关性。

## 5. 下一步判别

实施阶段先输出 task-scoped contribution/normal AOV：`BSDF-hit environment`、`secondary-hit NEE`、`deeper path`、`1-dot(Ns,Ng)`、event 的 geometric-side validity，以及对应 PDF/MIS/throughput 尾部。若离群点与 invalid hemisphere 高度重合，先完成 H1 的公共 transport 修复；若离群点主要来自几何有效的 environment direct 分量，再启用设计中冻结的 4×4 multiple-sample MIS。两者都不成立时停止并回到 planning，不自动尝试 clamp/denoiser。
