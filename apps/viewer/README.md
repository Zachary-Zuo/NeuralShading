# NclsViewer

`NclsViewer` 是 Windows/D3D12 的 `ScatteringPackage@1` 部署验证器。界面固定包含两个宽度相同的 `ComparisonSlot`；每个 slot 独立绑定一个 package，并选择 `path-tracing` 或 `deferred`。slot 加载、ABI、资源或 capability 失败只影响本侧，camera aspect 和另一侧 extent 不变。

viewer 不解释 NVIDIA、LayerStack 或其他 program 的私有结构。C++ loader 校验 package 的三个 identity、module closure、typed buffer/texture/sampler descriptor 与 content hash，随后创建通用 `ScatteringBinding`。NVIDIA latent 的两张 RGBA16F DDS mip chain由 descriptor绑定，不需要把 method shader预编入 viewer。

## 两条 renderer 路径

package path tracer 在每个 scene hit构造完整 scattering context，包括 position、shading/geometric frame、outgoing direction、material instance、UV 与 ray-cone footprint。续路径直接调用当前 slot binding 的 `prepare/sample/pdf`，直接光调用同一 state 的 `evaluate/pdf`；因此 neural PT不是 source reference PT的显示别名。

deferred renderer从 G-buffer传入相同的 UV/gradient 与 frame，再调用同一 package `prepare/evaluate`。两种 mode只改变 transport，不改变 package math或资源。

两个 PT 都通过 `PathSurface.slang` 构造 scene surface。Falcor camera basis 含共同 focal-distance 尺度，primary ray-cone spread 使用 `cameraV/cameraW` 的长度比；输出 footprint 是 normalized UV derivative，可直接交给 MaterialX `SampleGrad` 或 neural latent filter。修改这一链路时必须运行 `tests/gpu/test_viewer_path_surface.py`，并用具有明显空间结构的 local-light capture验证，不能只检查 slot `ready` 或平均颜色。

## 构建

只使用项目脚本。它验证锁定 Falcor提交和干净 worktree，临时应用 `patches/falcor-viewer-overlay.patch`，构建结束后反向应用 overlay；`external/Falcor` 最终必须保持干净。

```powershell
.\scripts\build_viewer.ps1 -Configuration Release
```

构建产物位于锁定 Falcor Release bin。交互式启动可直接指定 package root；`--method` 是 package ID：

```powershell
external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --bundle-root artifacts\exports\example `
  --method <package-id> --width 1280 --height 720
```

## capture v4 与回放

自动化比较使用 `ncls.viewer-capture@4`。manifest 的 `slots[2]` 分别记录 `package_id`、`mode`、status 与 runtime/material/source identity；输出图像固定为 `*-slot-0.exr` 与 `*-slot-1.exr`。`source-reference` 是内建的权威 source transport请求，其余值必须对应 `bundle_root` 下通过验证的 package。验证 neural mode 对称性时，可让两个 slot绑定同一 neural package并分别选择 PT/deferred。

最小 slot 片段如下：

```json
{
  "format_name": "ncls.viewer-capture",
  "format_version": 4,
  "reference_integrator": "ncls.scene-path-tracer@1",
  "bundle_root": "../exports/example",
  "slots": [
    {"package_id": "source-reference", "mode": "path-tracing"},
    {"package_id": "<package-id>", "mode": "path-tracing"}
  ],
  "resolution": [640, 360],
  "reference_spp": 16,
  "reference_samples_per_frame": 4
}
```

无窗口回放与捕获：

```powershell
external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --replay artifacts\captures\request.json `
  --headless --capture artifacts\captures\result
```

同一 replay 把 slot 1 的 mode改为 `deferred`即可验证 deferred 路径。capture只报告实际 ready/unsupported/error 状态，不 fallback 到另一 program或 transport。

## 操作与诊断

交互模式保留 orbit、pan、dolly、材质/source编辑与 viewer scene保存。任一相机、场景、材质、package或 mode变化都会清空对应 slot accumulation。普通运行把详细 shader diagnostics写入 `NclsViewer*.log`；需要完整控制台输出时使用 `--verbose-console`。headless 始终保留控制台日志，便于 CI 和 artifact审计。

固定路径 benchmark仍由项目脚本执行：

```powershell
.\scripts\benchmark_viewer.ps1 `
  -BundleRoot artifacts\exports `
  -Preset configs\viewer-benchmark-v2.json `
  -OutputDirectory artifacts\benchmarks\viewer
```
