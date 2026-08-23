# NclsViewer

`NclsViewer` 是 Windows/D3D12 原生材质查看器。对于当前统一方法已支持的 LayerStack，左侧累积多层随机游走 reference，右侧运行通过完整性检查和 GPU parity 的 realtime `MethodBundle`；两侧共享相机、主可见性、材质、灯光、HDRI、曝光和 tone mapping。

`--material` 也可以直接加载 MERL 原始二进制表、OpenPBR 原生参数文档和 MaterialX `.mtlx`/纹理。此时两侧显示同一个 Falcor reference，直到统一方法声明支持该源材质族；viewer 不会把源材质偷偷转换为 LayerStack 或简单 PBR，也不会拿空白的 approximation 冒充结果。MaterialX parity replay 还可通过 `reference_geometry`（命令行对应 `--reference-geometry OBJ`）指定原始网格；Falcor 会实际加载并光栅化该 OBJ，而不是用解析 SDF 替代它。

## 构建与运行

构建脚本会临时把根仓库的 `apps/viewer/` overlay 到锁定的 Falcor 8.0 构建中，结束后反向应用补丁并验证 `external/Falcor` 恢复干净：

```powershell
.\scripts\build_viewer.ps1 -Configuration Release
.\scripts\build_viewer.ps1 -Configuration Release -Run --bundle-root artifacts\exports
```

常用交互：左键 orbit，中键或右键 pan，滚轮 dolly，拖动分割线，`Space` 暂停/继续 reference 累积，`R` 重置相机。材质编辑器操作的是 `MaterialProgram`，不会暴露某个拟提方法的内部 packet。

## 无窗口捕获和回放

```powershell
external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --bundle-root artifacts\exports --headless --frames 256 `
  --capture artifacts\captures\example

external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --replay artifacts\captures\example\capture.json --headless `
  --capture artifacts\captures\replayed
```

捕获包含左右线性 EXR、共同显示 PNG、difference EXR/PNG、材质快照、GPU 时间 CSV 和完整 manifest。若使用 reference OBJ，manifest 同时记录绝对路径和 SHA-256。`--replay` 会锁定方法 ID；对应 bundle 未通过 hash、平台、合同或 parity 时直接失败，不会静默换方法。

## 固定路径 benchmark

```powershell
.\scripts\benchmark_viewer.ps1 `
  -BundleRoot artifacts\exports `
  -Preset configs\viewer-benchmark-v1.json `
  -OutputDirectory artifacts\benchmarks\viewer
```

preset 固定分辨率、场景、灯光、参考设置、预热帧和相机路径。脚本对每个相机复用 viewer 的无窗口 replay/capture 路径，输出 `summary.json`、`metrics.csv`、日志和被 Git 忽略的图像。当前时间语义是每个相机捕获前的最后一次 GPU 时间戳；`visibility`/`prepare` 是相机状态建立时的一次性成本。
