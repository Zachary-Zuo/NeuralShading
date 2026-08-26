# NclsViewer

`NclsViewer` 是 Windows/D3D12 原生材质查看器。左侧现在由独立的 Falcor Scene path tracer 生成源材质 reference：从相机射线开始，经过场景相交、可见性/阴影、环境或解析光直接采样、材质散射、跨物体间接反弹和 Russian roulette，再逐帧累计 raw Monte Carlo 均值。它不再是只在首个可见表面计算局部光照的预览 pass。

当前接入 LayerStack、MERL、OpenPBR 和 MaterialX 四种源材质族。每一次 path 命中都按 Falcor material slot 分派到该 slot 自己的源材质 reference，不会先改写成 LayerStack 或简单 PBR。LayerStack 在层内继续使用原生随机游走；MaterialX 纹理和 normal map 使用由 path ray cone 推导的 mip LOD。

“完整 path tracing”指 viewer 覆盖了上述完整场景传输路径，不表示计算无限深度。capture 会明确记录 `reference_scene_max_bounces` 与 `reference_layer_walk_max_depth`；raw 结果是相对于这两个有限上限的 Monte Carlo 估计。单界面、MERL、OpenPBR 和 MaterialX 的 HDRI 采样使用亮度 × `sin(theta)` importance sampling 与 MIS，可较快压低噪声。深层 LayerStack 暂无可直接求值的完整方向 PDF，因此环境路径仍保持 raw 有限深度估计，但收敛会慢于单界面材质。

默认勾选的去噪结果只用于观看。raw reference 始终单独累计并作为 comparison、difference、噪声估计和 `*-reference.exr` 的权威输入；a-trous cross-bilateral 去噪预览写入 `*-reference-denoised.exr`，manifest 明确标记它有偏且不具权威性。

右侧只有在用户显式选择通过完整性检查与 GPU parity 的 `ScatteringPackage` 后才出现。viewer 不识别具体方法名：bundle 声明 shader module、反射生成的权重 offset、`CompiledMaterial` stride 与私有 state stride；通用 pass 通过 `INclsScatteringBackend` 调用 `prepare/evaluate/sample/pdf`。初始化只负责读取和绑定这些资源，不包含 method-specific renderer 分支。

当前 03 部署轨道可同时加载原规模 NVIDIA paper baseline 与 core-frame candidate。两者都保留完整 matched GGX9 sampler；baseline 按真实成本标为 `diagnostic`，candidate 标为 `realtime`，均未缩小训练或部署形态。它们只接受 bundle 声明的精确 corpus state，任意材质不会被错误映射到这个 latent。普通模式的左右差图会同时包含材质表示、实时积分和全局传输差异；`--evaluator-preview-lighting` 改用单方向光并令 scene bounce cap 为 0，用于更直接观察局部 evaluator 外观，但仍不能替代方向响应数据集指标。

viewer 不是当前 neural 模型结构的搜索工具。先在完整 `wo × wi` 监督上确定单材质 evaluator、共享 decoder/latent 和 compiler，再导出 Slang ScatteringPackage 进入这里做 GPU 与系统验证；matched sampler 和环境积分同样在 evaluator 成形后接入。

## 固定 studio-v1 场景

无 replay、也没有显式 `--reference-geometry` 时，viewer 固定加载 `configs/viewer-studio-v1.json`：

- 几何：锁定 MaterialX 1.39.4 commit 中的 `shaderball.glb`，Apache-2.0；
- 环境：Poly Haven `studio_small_03_1k.exr`，CC0；
- 默认材质：`configs/viewer-studio-material-v1.json` 中的各向异性粗糙导体；
- 相机、曝光、环境旋转、积分深度和所有资产 SHA-256 均由 preset 固定。

`scripts/fetch_viewer_assets.ps1` 把 shaderball 放到 `assets/viewer/scenes/studio-v1/`，并验证 `assets/viewer/environments/polyhaven-1k/` 中的 HDRI；构建再把固定资源复制到 viewer runtime 的 `data/ncls-viewer/`。因此默认启动、无窗口捕获和 benchmark 看到的是同一个标准场景，而不是依赖当前工作目录的临时资产。

## 构建与运行

只使用项目构建脚本。它先验证固定资产，再临时把 `apps/viewer/` overlay 到锁定的 Falcor 8.0 Samples 树，结束后反向应用补丁并验证 `external/Falcor` 恢复干净：

```powershell
.\scripts\build_viewer.ps1 -Configuration Release
.\scripts\build_viewer.ps1 -Configuration Release -Run --bundle-root artifacts\exports

conda run -n neural-shading python -m ncls.cli bundle export-compiled-set `
  --compiled-set artifacts\compiled-materials\example `
  --preview-material artifacts\inputs\preview-material.json `
  --parity artifacts\inputs\parity.json `
  --output artifacts\exports\example `
  --display-name "Example method" --state-id <SHA256>

$bundle = "artifacts\exports\example"
$method = (Get-Content -LiteralPath "$bundle\manifest.json" -Encoding UTF8 -Raw | ConvertFrom-Json).method_id
.\scripts\build_viewer.ps1 -Configuration Release -Run `
  --bundle-root $bundle `
  --material "$bundle\resources\preview-material.json" `
  --method $method --evaluator-preview-lighting --width 640 --height 360

.\scripts\build_viewer.ps1 -Configuration Release -Run `
  --reference-geometry assets\viewer\scenes\example.glb `
  --material assets\source-materials\example.mtlx `
  --environment assets\viewer\environments\example.exr
```

常用交互：单击选择物体/material slot；左键拖动 orbit；中键或右键拖动 pan；滚轮 dolly；拖动分割线；`Space` 暂停/继续 raw reference 累积；`R` 重置相机。UI 中可切换 raw/denoised preview，但该开关不重置或修改 raw 累积。

UI 按职责分为 `Scene and camera`、`Material`、`Lighting`、`Reference and display`、`Realtime method`、`Capture` 与 `Performance and status`。`Lighting` 内再按环境光、方向光、点光和矩形光分组；未启用的灯只显示禁用说明，不再暴露看似可改但不会参与图像的颜色和强度。颜色控件编辑线性 RGB。材质或光照参数一旦变化，viewer 会自动解除 `Freeze reference`、清空旧累积并立即用新状态出图。

`Source material family` 可以在当前 slot 上明确切换 LayerStack、MERL、OpenPBR 与 MaterialX。LayerStack 可直接建立规范化默认实例；MERL 必须选择 `.binary` 测量表，OpenPBR 必须选择记录原始 `.mtlx` provenance 的 resolved adapter，MaterialX 必须选择 `.mtlx` 及其原生纹理资源。OpenPBR 可另存 resolved native parameter JSON；MaterialX 的图/纹理和 MERL 测量表不由 viewer 改写，编辑后的 override 随 viewer scene 保存。

`Save viewer scene` 写出 `ncls.viewer-scene@1` sidecar。它逐 material slot 保存 family 与状态：LayerStack 内嵌 `MaterialProgram`，OpenPBR 保存具名参数，MERL 保存测量表 URI/hash，MaterialX 保存文档/纹理 identity 与可编辑 constant override；同时保存几何、HDRI、相机、物理光照和 reference 上限。`Load viewer scene` 会按 URI、hash 和 slot 覆盖关系验证并重建 GPU 资源。详细合同见 [viewer scene 合同](../../docs/contracts/viewer_scene.md)。

光照方向也有明确约定：方向光向量是 surface-to-light；矩形灯法线为 `normalize(cross(U,V))`。固定 studio preset 的资产 hash 只用于验证默认启动，不会再拦截交互式加载另一份 scene 或 HDRI。

交互式启动默认把 Falcor/Slang 的详细 shader diagnostics 写入 exe 同目录的 `NclsViewer*.log`，避免把大量锁定上游 warning 刷到控制台；未捕获的 fatal error 仍会直接输出。需要调试完整控制台日志时加 `--verbose-console`。headless 模式始终保留控制台日志，便于自动化任务诊断。

## 无窗口捕获和回放

```powershell
external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --bundle-root artifacts\exports --headless --frames 256 `
  --capture artifacts\captures\example

external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --replay artifacts\captures\example\capture.json --headless `
  --capture artifacts\captures\replayed

external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --viewer-scene artifacts\captures\example\capture-scene.json
```

capture v3 记录固定场景、环境、相机、材质、积分上限、raw 噪声估计、显示去噪语义、方法和文件角色，并写出 `*-scene.json`。capture 的 `viewer_scene` 字段让单材质和多 slot scene 都从同一份逐 slot 状态回放；`source_material_sha256` 保留源资产 identity，`source_material_state_sha256` 单独标识 UI 编辑后的实际状态。

## 固定路径 benchmark

```powershell
.\scripts\benchmark_viewer.ps1 `
  -BundleRoot artifacts\exports `
  -Preset configs\viewer-benchmark-v1.json `
  -OutputDirectory artifacts\benchmarks\viewer
```

benchmark preset 固定 studio-v1 几何/HDRI/默认材质的路径与哈希，以及分辨率、相机路径、reference 上限和帧数。输出只进入被 Git 忽略的 `artifacts/`。
