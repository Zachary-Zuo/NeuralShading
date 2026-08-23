# 随机游走参考解迁移验证（2026-08-23）

## 它验证什么

本次验证确认 `shaders/ncls/reference/` 使用新 `LayerStackIR` 后没有改变已有随机游走算法的数值语义，并且新的正式采集路径仍与锁定 pbrt-v4 的独立 layered BxDF 实现一致。它只证明参考解迁移正确，不代表固定成本拟合表示已经解决。

## 迁移前后 GPU 回归

同一份 legacy v0 材质分别打包为旧 `LayerStack` 和新 `LayerStackIR`，使用相同方向、seed、stream key、样本数和 max depth 执行两个 shader。两界面和三界面材质的 A/B 均值、A/B 二阶矩逐项通过 `rtol=1e-6, atol=1e-7`。

对应自动测试：

```powershell
.\scripts\run_falcor_python.ps1 -m pytest tests\gpu\test_reference_migration_gpu.py -q
```

## pbrt-v4 交叉验证

使用 `build/pbrt-probe-v3/Release/ncls_pbrt_probe.exe`，每个方向 32768 samples、max depth 32。Falcor 一次生成两个独立随机流，因此其报告的总样本数为 65536。

无体散射、近零 optical thickness 的 coated diffuse 切片：

| 入射光角度 | pbrt response_cos | Falcor reference | 相对误差 |
|---:|---:|---:|---:|
| -55° | 0.05670166 | 0.05671647 | 0.026% |
| -20° | 0.32255653 | 0.32276672 | 0.065% |
| 0° | 0.11644415 | 0.11657036 | 0.108% |
| 35° | 0.07620538 | 0.07600161 | 0.268% |
| 60° | 0.04464751 | 0.04478133 | 0.299% |

mean 相对误差 0.153%，max 0.299%。

均匀介质参数为 optical thickness 0.4、scattering albedo 0.5、`g=0.3` 时：

| 入射光角度 | pbrt response_cos | Falcor reference | 相对误差 |
|---:|---:|---:|---:|
| -55° | 0.03141055 | 0.03145046 | 0.127% |
| -20° | 0.28163508 | 0.28189006 | 0.090% |
| 0° | 0.07345186 | 0.07317020 | 0.384% |
| 35° | 0.03989308 | 0.03988722 | 0.015% |
| 60° | 0.02256921 | 0.02262400 | 0.242% |

mean 相对误差 0.172%，max 0.384%。差值均不超过本次 Falcor standard error 的 2.70 倍；pbrt 结果自身也有采样噪声。

复现命令：

```powershell
.\scripts\run_falcor_python.ps1 tools\reference\pbrt_compare.py `
  --pbrt-exe build\pbrt-probe-v3\Release\ncls_pbrt_probe.exe `
  --samples 32768 --batches 1 --max-depth 32 --optical-thickness 0.000001

.\scripts\run_falcor_python.ps1 tools\reference\pbrt_compare.py `
  --pbrt-exe build\pbrt-probe-v3\Release\ncls_pbrt_probe.exe `
  --samples 32768 --batches 1 --max-depth 32 `
  --optical-thickness 0.4 --medium-albedo 0.5 --g 0.3
```

## 其他物理门槛

新的 integration tests 还覆盖：单界面 Lambert 解析响应、三界面互易性、八界面各向异性路径执行，以及对“有体散射但 RGB 总消光不同”这一 v1 不支持状态的显式拒绝。
