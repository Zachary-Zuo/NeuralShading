# NclsViewer

`NclsViewer` 是 Windows/D3D12 原生材质查看器。左侧现在由独立的 Falcor Scene path tracer 生成源材质 reference：从相机射线开始，经过场景相交、可见性/阴影、环境或解析光直接采样、材质散射、跨物体间接反弹和 Russian roulette，再逐帧累计 raw Monte Carlo 均值。它不再是只在首个可见表面计算局部光照的预览 pass。

当前接入 LayerStack、MERL、OpenPBR 和 MaterialX 四种源材质族。每一次 path 命中都按 Falcor material slot 分派到该 slot 自己的源材质 reference，不会先改写成 LayerStack 或简单 PBR。LayerStack 在层内继续使用原生随机游走；MaterialX 纹理和 normal map 使用由 path ray cone 推导的 mip LOD。

“完整 path tracing”指 viewer 覆盖了上述完整场景传输路径，不表示计算无限深度。capture 会明确记录 `reference_scene_max_bounces` 与 `reference_layer_walk_max_depth`；raw 结果是相对于这两个有限上限的 Monte Carlo 估计。单界面、MERL、OpenPBR 和 MaterialX 的 HDRI 采样使用亮度 × `sin(theta)` importance sampling 与 MIS，可较快压低噪声。深层 LayerStack 暂无可直接求值的完整方向 PDF，因此环境路径仍保持 raw 有限深度估计，但收敛会慢于单界面材质。

默认勾选的去噪结果只用于观看。raw reference 始终单独累计并作为 comparison、difference、噪声估计和 `*-reference.exr` 的权威输入；a-trous cross-bilateral 去噪预览写入 `*-reference-denoised.exr`，manifest 明确标记它有偏且不具权威性。

右侧只有在用户显式选择通过完整性检查与 GPU parity 的 `MethodBundle` 后才出现，仍是固定成本 deferred 实时方法。因而 viewer 的左右差图是“完整 path-traced reference 与实时系统”的视觉系统差异，会同时包含材质近似和全局传输差异；它不能替代方向响应数据集上的 closure 误差指标。

## 固定 studio-v1 场景

无 replay、也没有显式 `--reference-geometry` 时，viewer 固定加载 `configs/viewer-studio-v1.json`：

- 几何：锁定 MaterialX 1.39.4 commit 中的 `shaderball.glb`，Apache-2.0；
- 环境：Poly Haven `studio_small_03_1k.exr`，CC0；
- 默认材质：`configs/viewer-studio-material-v1.json` 中的各向异性粗糙导体；
- 相机、曝光、环境旋转、积分深度和所有资产 SHA-256 均由 preset 固定。

`scripts/fetch_viewer_assets.ps1` 把 shaderball 放到 `data/source-materials/viewer-scenes/studio-v1/`，并验证 `data/hdris/polyhaven_1k/` 中的 HDRI；构建再把固定资源复制到 viewer runtime 的 `data/ncls-viewer/`。因此默认启动、无窗口捕获和 benchmark 看到的是同一个标准场景，而不是依赖当前工作目录的临时资产。

## 构建与运行

只使用项目构建脚本。它先验证固定资产，再临时把 `apps/viewer/` overlay 到锁定的 Falcor 8.0 Samples 树，结束后反向应用补丁并验证 `external/Falcor` 恢复干净：

```powershell
.\scripts\build_viewer.ps1 -Configuration Release
.\scripts\build_viewer.ps1 -Configuration Release -Run --bundle-root artifacts\exports

.\scripts\build_viewer.ps1 -Configuration Release -Run `
  --reference-geometry data\source-materials\scenes\example.glb `
  --material data\source-materials\example.mtlx `
  --environment data\hdris\example.exr
```

常用交互：单击选择物体/material slot；左键拖动 orbit；中键或右键拖动 pan；滚轮 dolly；拖动分割线；`Space` 暂停/继续 raw reference 累积；`R` 重置相机。UI 中可切换 raw/denoised preview，但该开关不重置或修改 raw 累积。

## 无窗口捕获和回放

```powershell
external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --bundle-root artifacts\exports --headless --frames 256 `
  --capture artifacts\captures\example

external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --replay artifacts\captures\example\capture.json --headless `
  --capture artifacts\captures\replayed
```

capture v3 记录固定场景、环境、相机、材质、积分上限、raw 噪声估计、显示去噪语义、方法和文件角色。单材质场景可以完整回放；交互式多 slot 场景虽会逐 slot 记录来源，但在形成稳定的多 slot 序列化合同前仍标记为不可完整回放。

## 固定路径 benchmark

```powershell
.\scripts\benchmark_viewer.ps1 `
  -BundleRoot artifacts\exports `
  -Preset configs\viewer-benchmark-v1.json `
  -OutputDirectory artifacts\benchmarks\viewer
```

benchmark preset 固定 studio-v1 几何/HDRI/默认材质的路径与哈希，以及分辨率、相机路径、reference 上限和帧数。输出只进入被 Git 忽略的 `artifacts/`。
