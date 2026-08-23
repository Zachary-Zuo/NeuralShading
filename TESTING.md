# 测试说明

所有 Python 命令使用唯一 Conda 环境 `neural-shading`。需要导入 Falcor Python 模块的测试只能通过 `scripts/run_falcor_python.ps1` 启动。

## CPU 单元测试

```powershell
conda run -n neural-shading python -m pytest tests\unit -q
```

覆盖 `MaterialProgram`/`LayerStackIR`、旧数据一次性转换、散射合同、`legacy-ltc-k2` 私有状态、训练/评测/checkpoint/TensorBoard、MethodBundle 导出与内容哈希。

## Slang/GPU 与随机游走参考

```powershell
.\scripts\run_falcor_python.ps1 -m pytest `
  tests\gpu tests\integration\reference -q
```

覆盖 Python/Slang ABI、方向和余弦语义、`evaluate/sample/pdf`、各向异性、P1 compiler 的 PyTorch/Slang parity、解析 diffuse、互易性、八层执行、统计量和数据生成 smoke。

## pbrt-v4 外部交叉验证

首次配置和构建 probe：

```powershell
cmake -S tools\reference\pbrt_probe -B build\pbrt-probe-current `
  -G "Visual Studio 17 2022" -A x64
cmake --build build\pbrt-probe-current --config Release `
  --target ncls_pbrt_probe --parallel 12
```

比较锁定 pbrt-v4 与 Falcor 随机游走参考解：

```powershell
.\scripts\run_falcor_python.ps1 tools\reference\pbrt_compare.py `
  --pbrt-exe build\pbrt-probe-current\Release\ncls_pbrt_probe.exe `
  --samples 65536 --batches 8 --max-depth 32
```

默认 suite 同时覆盖 diffuse-clear、conductor-clear、conductor-absorbing 和 conductor-scattering，并包含不同方位的各向异性 conductor 切片。

## OpenPBR、MERL 与 MaterialX 源材质

获取固定上游和原始资产：

```powershell
.\scripts\fetch_reference_sources.ps1 -All
conda run -n neural-shading python scripts\fetch_source_materials.py merl
conda run -n neural-shading python scripts\fetch_source_materials.py polyhaven
```

构建 OpenPBR CPU probe，以及 MaterialX 官方 viewer、上游 runtime 和独立 float parity probe：

```powershell
cmake -S tools\reference\openpbr_probe -B build\openpbr-probe `
  -G "Visual Studio 17 2022" -A x64
cmake --build build\openpbr-probe --config Release `
  --target ncls_openpbr_probe --parallel 12
.\scripts\build_materialx_reference.ps1 -Configuration Release
```

运行原生身份、参数编辑、实表查表、图依赖和 shader generation 回归，以及三个材质族的离线呈现：

```powershell
conda run -n neural-shading python -m pytest `
  tests\unit\test_reference_registry.py `
  tests\unit\test_openpbr_material.py `
  tests\unit\test_merl_material.py `
  tests\unit\test_materialx_catalog.py `
  tests\integration\reference\test_source_material_references.py -q

conda run -n neural-shading python tools\reference\analytic_material_preview.py openpbr
conda run -n neural-shading python tools\reference\analytic_material_preview.py merl
conda run -n neural-shading python tools\reference\materialx_preview.py
```

三个新增源材质族都由 Falcor viewer 直接呈现，而不是只停留在 Python adapter。MERL 与 OpenPBR 做逐方向数值 parity；MaterialX 的空间纹理契约使用共同相机线性 HDR 图像 parity：

```powershell
.\scripts\build_viewer.ps1 -Configuration Release
conda run -n neural-shading python tools\reference\materialx_parity.py --suite
```

MaterialX suite 先生成一次 `common-sphere.obj`，让上游 renderer 与 Falcor 光栅管线加载同一路径，并在报告/capture 中核对几何 SHA-256；随后用无纹理核心 probe 检查 closure 公式，再验证全部 8 个原始 4K 材质。报告位于 `artifacts/validation/materialx-parity/suite/report.json`；验收门槛由 `references/acceptance.json` 版本化，逐材质通过证据保存在 `references/materialx-polyhaven-v1/falcor-parity.json`。

## Windows viewer

```powershell
.\scripts\build_viewer.ps1 -Configuration Release

external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --bundle-root artifacts\exports --headless --frames 32 `
  --width 320 --height 240 --capture artifacts\captures\smoke
```

固定相机路径 benchmark：

```powershell
.\scripts\benchmark_viewer.ps1 `
  -BundleRoot artifacts\exports `
  -Preset configs\viewer-benchmark-v1.json `
  -OutputDirectory artifacts\benchmarks\viewer
```

验收时还应将 capture manifest 传回 `--replay`，确认左右 EXR、显示 PNG、difference 和材质快照逐字节一致；篡改任一 bundle 内容哈希后，锁定方法 ID 的 headless replay 必须以非零状态失败。

## 仓库边界与静态检查

```powershell
conda run -n neural-shading python -m compileall -q src tests tools
git diff --check
git -C external\Falcor status --short
git -C external\pbrt-v4 status --short
git -C external\OpenPBR status --short
git -C external\openpbr-bsdf status --short
git -C external\glm status --short
git -C external\MaterialX status --short
```

所有上游工作树必须为空。`build/`、`data/`、`artifacts/`、`external/` 和缓存不得进入根仓库。
