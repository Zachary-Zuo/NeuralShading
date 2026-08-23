# 参考解外部交叉验证

`pbrt_compare.py` 用同一组灰色 coated diffuse 参数，对比锁定 pbrt-v4 probe 与新的 Falcor 随机游走参考 shader。它验证的是参考实现，不属于训练或拟合后端。

已有 probe 可直接运行：

```powershell
.\scripts\run_falcor_python.ps1 tools\reference\pbrt_compare.py `
  --pbrt-exe build\pbrt-probe-v3\Release\ncls_pbrt_probe.exe `
  --samples 262144 --batches 4
```

脚本报告逐方向相对误差、Falcor Monte Carlo standard error，以及两者差值相对 standard error 的倍数。pbrt 自身也有采样噪声，因此不能把单次 probe 当作无噪声解析真值。
