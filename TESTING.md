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
cmake -S tools\reference\pbrt_probe -B build\pbrt-probe `
  -G "Visual Studio 17 2022" -A x64
cmake --build build\pbrt-probe --config Release `
  --target ncls_pbrt_probe --parallel 12
```

比较锁定 pbrt-v4 与 Falcor 随机游走参考解：

```powershell
.\scripts\run_falcor_python.ps1 tools\reference\pbrt_compare.py `
  --pbrt-exe build\pbrt-probe\Release\ncls_pbrt_probe.exe `
  --samples 65536 --batches 8 --optical-thickness 0.4 `
  --medium-albedo 0.5 --g 0.3
```

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
```

两个上游工作树必须为空。`build/`、`data/`、`artifacts/`、`external/` 和缓存不得进入根仓库。
