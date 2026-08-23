# NclsViewer

`NclsViewer` 是 Windows/D3D12 原生材质查看器。左侧累积项目自己的多层随机游走参考解，右侧运行通过完整性检查和 GPU parity 的 realtime `MethodBundle`；两侧共享相机、主可见性、材质、灯光、HDRI、曝光和 tone mapping。

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

捕获包含左右线性 EXR、共同显示 PNG、difference EXR/PNG、材质快照、GPU 时间 CSV 和完整 manifest。`--replay` 会锁定方法 ID；对应 bundle 未通过 hash、平台、合同或 parity 时直接失败，不会静默换方法。

## 固定路径 benchmark

```powershell
.\scripts\benchmark_viewer.ps1 `
  -BundleRoot artifacts\exports `
  -Preset configs\viewer-benchmark-v1.json `
  -OutputDirectory artifacts\benchmarks\viewer
```

preset 固定分辨率、场景、灯光、参考设置、预热帧和相机路径。脚本对每个相机复用 viewer 的无窗口 replay/capture 路径，输出 `summary.json`、`metrics.csv`、日志和被 Git 忽略的图像。当前时间语义是每个相机捕获前的最后一次 GPU 时间戳；`visibility`/`prepare` 是相机状态建立时的一次性成本。
