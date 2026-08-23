# Reference 响应数据

## 它是什么

`ncls.reference-dataset@2` 是训练、逐样本直接拟合和回归测试共用的方向响应数据合同。每个 tile 对应一个源材质状态与一个观察方向，保存全部入射光方向上的 `response_cos = f(wo, wi) * max(dot(Ns, wi), 0)`、统计方差、总样本数和可选的独立随机流均值。

它不包含 K2、LTC、latent 或网络参数。拟合表示发生变化时，只要 reference 和源材质语义没有变化，就不需要重新采集。

它也不是源材质资产合同。源材质的原生参数、图、程序、纹理、测量表和其他资源必须独立保存并可追溯；本数据集只记录 reference 对这些源材质状态执行查询后得到的监督。完整边界见 `docs/material_scope.md`。

## 生成

下面的命令只生成当前 `LayerStackIR` 材质族的随机游走 reference 数据，不是所有源材质族必须共用的生成器：

Falcor Python 只能通过锁定环境启动：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli data generate-reference `
  --output data\reference-v2 `
  --families 128 `
  --local-states 8 `
  --views 16 `
  --lights 128 `
  --adaptive `
  --batch-samples 256 `
  --min-samples 512 `
  --max-samples 16384 `
  --resume
```

`--resume` 只复用具有 `complete.json`、且 index/response 内容哈希都正确的完整分片。公共材质、方向和 split 文件也必须与本次 resolved config 完全一致；不一致时会停止，不会把两次配置的数据拼在一起。

固定采样适合确定性 smoke 和快速回归：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli data generate-reference `
  --output data\reference-smoke `
  --families 1 --local-states 1 --views 1 --lights 16 `
  --samples-per-replica 16 --max-depth 16
```

## 验证与读取

完整验证包括 manifest、公共文件与分片哈希、dtype/shape、连续 tile ID、family split、一致的 state/index 关系，以及半球立体角权重：

```powershell
conda run -n neural-shading python -m ncls.cli data validate data\reference-v2
```

Python 代码只从 manifest 入口读取：

```python
from ncls.data import ReferenceDataset

dataset = ReferenceDataset.open("data/reference-v2")
statistics = dataset.statistics(0)
print(statistics.mean.shape)
print(statistics.standard_error.shape)
```

新采集数据的 `variance` 是逐样本总体方差，reader 会用总样本数计算 standard error。v0 转换数据的 manifest 会标记 `replica-mean-variance`，reader 不会把它伪装成逐样本方差。

## v0 转换

旧 `ncls-direction-tiles@1` 数据只作为迁移证据保留，通过一次性 adapter 转换：

```powershell
conda run -n neural-shading python -m ncls.cli data convert-legacy-v0 `
  data\pilot_v0_batched data\reference-v2-converted --resume
```

转换后的 `mean` 是原 A/B 均值的平均，replica 字段逐值保留。由于旧格式没有二阶矩，`variance = 0.5 * (mean_a - mean_b)^2` 只表示两个 replica 均值之间的不确定性估计；这个限制同时写入 manifest。

## 当前随机游走 reference 的适用范围

v1 随机游走参考解处理局部反射、至多八个界面、各向异性粗糙度、各层切线旋转、均匀吸收/散射介质和不透明基底。为了让 RGB 共用一次自由飞行采样，有体散射时仍要求三个通道的总消光系数相同；各通道散射反照率可以不同。这只是当前实现约束，不是一般介质的物理性质。
