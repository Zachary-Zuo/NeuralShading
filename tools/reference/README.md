# Reference 验证与原始材质预览

这里的工具直接调用各源材质族自己的 reference，不经过统一近似表示或拟合 backend。

## pbrt 两界面交叉验证

`pbrt_compare.py` 同时比较 coated diffuse 与 coated conductor，默认覆盖 clear、吸收介质和散射介质，以及有方位差异的粗糙各向异性 conductor。它验证的是 LayerStack reference 的真实两界面分支，不把 pbrt probe 扩成按 `N` 枚举的材质系统。

```powershell
.\scripts\run_falcor_python.ps1 tools\reference\pbrt_compare.py `
  --pbrt-exe build\pbrt-probe-current\Release\ncls_pbrt_probe.exe `
  --samples 65536 --batches 8 --max-depth 32
```

脚本报告逐 RGB 通道和逐方向相对误差，以及 Falcor Monte Carlo standard error。pbrt 自身也有采样噪声，不能把单次 probe 当作无噪声解析真值。

## OpenPBR 与 MERL 离线预览

`analytic_material_preview.py` 用 Adobe OpenPBR CPU reference 或原始 MERL 表渲染球体。OpenPBR 预览保留官方 MaterialX 参数和模型颜色空间，仅在写 PNG 时把 ACEScg 转成显示 linear sRGB；MERL 预览直接使用官方 half/difference 参数化和 RGB scale。

```powershell
conda run -n neural-shading python tools\reference\analytic_material_preview.py openpbr
conda run -n neural-shading python tools\reference\analytic_material_preview.py merl
```

OpenPBR 默认展示 car paint、brushed aluminum、velvet 和 pearl；MERL 默认展示 alum-bronze、blue metallic paint、beige fabric 和 red plastic。输出位于 `artifacts/reference-previews/`。

## MaterialX / Poly Haven 原生预览与 Falcor 验收

构建脚本同时生成锁定 MaterialX 1.39.4 的官方 viewer、安装形式的上游 runtime，以及独立的 float parity probe：

```powershell
.\scripts\build_materialx_reference.ps1 -Configuration Release
```

官方 viewer 只承担原生预览。下面的命令直接加载原始 `.mtlx`、4K 纹理、OpenGL tangent normal、displacement 图和 MaterialX 标准库，输出 shader-ball PNG：

```powershell
conda run -n neural-shading python tools\reference\materialx_preview.py
```

正式的统一呈现不是上述预览器，而是 Falcor `NclsViewer`。它从同一个原始 `.mtlx` 解析 `standard_surface` 和纹理连接，在 Falcor/D3D12 中直接求值，不经过 `LayerStackIR`、OpenPBR 或项目统一拟合表示。完整图像验收运行：

```powershell
.\scripts\build_viewer.ps1 -Configuration Release
conda run -n neural-shading python tools\reference\materialx_parity.py --suite
```

suite 只生成一次高精度 `common-sphere.obj`，把它的绝对路径写入 Falcor replay，并让上游 MaterialX float renderer 与 Falcor 光栅管线加载同一个文件；报告和 Falcor capture 都记录该 OBJ 的 SHA-256。两端还共享相机、方向光和线性 HDR 输出。无纹理核心 probe 使用严格公式门槛；8 个原始材质另用保留两端原生 mip/16x 各向异性 footprint 的纹理门槛。完整图像和逐材质报告写到 `artifacts/validation/materialx-parity/suite/`。

当前 Falcor surface-response 路径保留源文档的 displacement graph，但验证场景没有移动几何；这只是当前几何 capability 的范围，不是对原始材质表达的删减要求。
