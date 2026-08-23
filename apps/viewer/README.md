# NclsViewer

`NclsViewer` 是 Windows/D3D12 原生材质查看器。界面统一使用英文和 Falcor 默认字体。左侧显示当前源材质族的 reference；只有用户显式选择通过完整性检查与 GPU parity 的 `MethodBundle` 后，右侧才显示该方法的实时结果。方法为空时 reference 使用整个视口，不再生成空白或伪造的右侧图像。

当前接入的源材质族包括 LayerStack、MERL、OpenPBR 和 MaterialX。切换源文件后，UI 会按材质族显示其原生可调参数：LayerStack 编辑界面/介质，OpenPBR 编辑 resolved native inputs，MaterialX 编辑当前 reference subset 中未被纹理连接占用的 `standard_surface` 输入；MERL 测量表通过切换原始文件更换。源材质不会先被改写成 LayerStack 或简单 PBR。

viewer 使用 Falcor `Scene` 导入器加载场景，不再维护只支持单个 OBJ 的自建 VAO 路径。文件对话框展示当前 Falcor 插件实际支持的格式（通常包括 OBJ、glTF/GLB、FBX 等）；命令行保留兼容参数 `--reference-geometry SCENE`。主可见性 pass 输出 instance/material ID，单击物体会选择其 Falcor material slot；各 slot 独立保存源材质族、原生参数、纹理与 GPU reference 资源，后续编辑只作用于选中的 slot。

需要明确当前积分边界：左侧是共享主可见性表面上的源材质局部 reference 光照积分，不是包含物体间遮挡、间接反弹和穿物体传输的完整场景 path tracer。右侧是固定成本 deferred prepare + lighting；它没有 spp、跨帧样本累计或随帧变化的噪声。当前环境光使用固定的确定性方向集合，解析灯直接求值。场景阴影、全局间接光以及更完整的 Falcor RenderGraph 仍是后续工作。

## 构建与运行

构建脚本临时把根仓库的 `apps/viewer/` overlay 到锁定的 Falcor 8.0 Samples 树，结束后在 `finally` 中反向应用补丁并验证 `external/Falcor` 恢复干净：

```powershell
.\scripts\build_viewer.ps1 -Configuration Release
.\scripts\build_viewer.ps1 -Configuration Release -Run --bundle-root artifacts\exports

.\scripts\build_viewer.ps1 -Configuration Release -Run `
  --reference-geometry data\source-materials\scenes\example.glb
```

常用交互：单击选择物体/material slot；左键拖动 orbit；中键或右键拖动 pan；滚轮 dolly；拖动分割线；`Space` 暂停/继续 reference 累积；`R` 重置相机。

## 无窗口捕获和回放

```powershell
external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --bundle-root artifacts\exports --headless --frames 256 `
  --capture artifacts\captures\example

external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --replay artifacts\captures\example\capture.json --headless `
  --capture artifacts\captures\replayed
```

方法为空时 capture 只写 reference、共同显示图、当前选中材质快照、GPU 时间和 manifest；存在方法时才额外写 approximation 与 difference。当前 manifest 仍以选中的源材质 slot 作为回放入口，多 slot 的完整编辑状态尚未形成稳定序列化合同，因此不能把交互式多材质场景 capture 描述为逐 slot 完整回放。

## 固定路径 benchmark

```powershell
.\scripts\benchmark_viewer.ps1 `
  -BundleRoot artifacts\exports `
  -Preset configs\viewer-benchmark-v1.json `
  -OutputDirectory artifacts\benchmarks\viewer
```

preset 固定分辨率、场景、灯光、reference 设置、预热帧和相机路径。输出位于被 Git 忽略的 `artifacts/`，不把单次 benchmark 或正确性验证结果持久化到根仓库。
