# 测试说明

所有 Python 命令使用 Conda 环境 `neural-shading`。

## Python 与层栈数据布局

```powershell
conda run -n neural-shading python -m pytest
```

期望：`tests/test_stack_schema.py` 全部通过。

## 自包含界面与两层随机游走参考（GPU）

```powershell
./scripts/run_falcor_python.ps1 -m pytest -m falcor -q
```

覆盖粗糙介电界面的双向透射、强制透射/内反射期望、Lambert 白炉、采样与概率密度一致性、确定性、互易性，以及关闭俄罗斯轮盘赌（RR）时与 Falcor 参考实现的一致性。

## pbrt-v4 两层 CPU 探针

首次配置与构建：

```powershell
cmake -S teacher/xval/pbrt_probe -B build/pbrt-probe -G "Visual Studio 17 2022" -A x64
cmake --build build/pbrt-probe --config Release --target ncls_pbrt_probe --parallel 12
```

运行灰色“粗糙涂层 + 漫反射基底”方向切片：

```powershell
./build/pbrt-probe/Release/ncls_pbrt_probe.exe 262144 20 32
./scripts/run_falcor_python.ps1 ./teacher/xval/falcor_two_layer.py --samples 262144 --max-depth 4
```

其中 pbrt 探针参数依次为 `nSamples viewAngle maxDepth opticalThickness seed mediumAlbedo g`。三方/介质对比可运行：

```powershell
./scripts/run_falcor_python.ps1 ./teacher/xval/pbrt_compare.py --pbrt-exe ./build/pbrt-probe/Release/ncls_pbrt_probe.exe --samples 65536 --batches 8 --optical-thickness 0.4
./scripts/run_falcor_python.ps1 ./teacher/xval/pbrt_compare.py --pbrt-exe ./build/pbrt-probe/Release/ncls_pbrt_probe.exe --samples 65536 --batches 8 --optical-thickness 0.4 --medium-albedo 0.5 --g 0.3
```

## Falcor 8.0 Release 构建

```powershell
Set-Location external/Falcor
./setup_vs2022.bat
./tools/.packman/cmake/bin/cmake.exe --build build/windows-vs2022 --config Release --target FalcorPython -- /m
```

## CPU/GPU 层栈布局

```powershell
./scripts/run_falcor_python.ps1 ./datagen/validate_layout.py
```

期望输出：

```text
LayerStack CPU/GPU layout OK: 752 bytes
```
