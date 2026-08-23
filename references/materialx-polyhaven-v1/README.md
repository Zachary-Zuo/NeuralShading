# MaterialX / Poly Haven 高分辨率纹理 reference package

这一族保留每个 CC0 Poly Haven 资产的原生 MaterialX 文档和随包纹理，不把 map 集合预先改写为 `LayerStackIR` 或 OpenPBR GT。MaterialX 1.39.4 提供图解析、标准库和 shader generation reference。

项目固定 8 个 4K 代表材质：`american_walnut_veneer`、`bark_brown_02`、`denim_fabric`、`curly_teddy_natural`、`rusty_metal_02`、`metal_plate`、`lichen_rock` 和 `monastery_stone_floor`，覆盖木材、石材、织物、金属与混合表面。`assets.json` 逐文件记录 API 返回的 URL、大小、MD5、物理尺寸以及纹理和位移语义；下载通过项目脚本完成并校验。使用 Poly Haven API 的项目入口明确标注 “Powered by Poly Haven”。

依赖清单以原始 `.mtlx` 的 filename 连接为权威，而不是盲信 API 的 `mtlx.include`：接入时实际发现 3 个 API include 把文档引用的 EXR roughness 列成了 JPG，锁定脚本已改为解析原始图后匹配精确文件。8 个文档均已通过 MaterialX 1.39.4 validate、GLSL generation、原生 literal 参数编辑 round-trip 和官方 `MaterialXView` shader-ball 渲染。

## Falcor 直接呈现

正式运行时由 `NclsViewer` 直接读取原始 `.mtlx` 和原始纹理，解析 `standard_surface` 的 literal/连接、base color、roughness、metalness、切线 normal 与源颜色空间，并在 Falcor/D3D12 reference shader 中求值。项目没有先把它们烘焙成简单 PBR、`LayerStackIR` 或任何拟合 backend；Python adapter 只负责登记、检查和编辑原生图，不是最终 renderer。

独立验收使用上游 MaterialX 1.39.4 `GlslShaderGenerator` + `GlslRenderer` float framebuffer 作为另一条实现。在共同的 256×128 分段 UV sphere、240×240 相机和方向光下：

| 范围 | p95 相对 L1 | linear PSNR | 绝对 MAE | 门槛 | 结论 |
|---|---:|---:|---:|---|---|
| 无纹理核心 `standard_surface` | 0.498% | 79.91 dB | 0.0000492 | ≤2%、≥40 dB、≤0.0001 | 通过 |
| 8 个原始 4K 材质 | 最高 9.510% | 最低 54.95 dB | 最高 0.001392 | ≤10%、≥50 dB、≤0.002 | 全部通过 |

两种门槛刻意分开：核心 probe 用严格门槛约束 closure 公式；原始纹理组保留 MaterialX/OpenGL 与 Falcor/D3D12 各自的 mip 构建和 16x 各向异性采样 footprint，因此只放宽纹理足迹差异，不能用它掩盖公式误差。逐材质证据保存在 `falcor-parity.json`，完整可再生图像位于被 Git 忽略的 `artifacts/validation/materialx-parity/suite/`。

源 `.mtlx` 中的 displacement 图仍属于原始 GT，并被资产清单保留；当前本地 surface-response 验收不移动球体几何。这表示 Falcor 接入的几何 capability 尚未覆盖 displacement，不表示原始材质不应包含或编辑该参数。

```powershell
.\scripts\build_materialx_reference.ps1 -Configuration Release
.\scripts\build_viewer.ps1 -Configuration Release
conda run -n neural-shading python tools\reference\materialx_parity.py --suite
```
