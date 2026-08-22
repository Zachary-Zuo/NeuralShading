# 解析着色表示上界实验

`baselines/` 用每个 tile 的随机游走参考响应直接优化解析 closure 参数。这里没有预测网络，因此测到的是 closure 函数族本身能达到的最好结果，不包含“网络学不会”的误差。

第一轮实验比较 GGX、LTC、球面高斯（SG）和共享字典。小规模运行命令：

```powershell
conda run -n neural-shading python -m baselines.oracle_fit --dataset ./data/pilot_v0_batched --output ./reports/oracle_pilot
```

每个 tile 独立优化，并使用多次随机初始化降低局部最优的影响。报告同时给出方向域 SMAPE、relative-L1、A/B 参考噪声和按材质类型、层数、视角划分的误差。

512-family 的主要结论见 `reports/oracle_ceiling_v0.md`。当前三槽基线由一个精确顶层界面项和两个 LTC 残差瓣组成。它明显优于纯 GGX-K3，但方向域 median/p90 relative-L1 仍为 6.73%/31.20%，因此只是当前基线，还不能称为最终表示。

导出的 LTC 参数必须使用与拟合求值完全相同的约束。逆尺度保存为 `exp(clamp(log_scale, -3, 3))` 之后的值，从而保证 PyTorch 拟合、二进制 packet 和 Falcor 求值解释的是同一组参数。
