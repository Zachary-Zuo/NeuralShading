# 真实 HDRI 环境光表示上界

数据集：`data\v0_oracle_512`；48 个光照探针；2048 个 tiles。

A/B 环境光 relative-L1 噪声：median 0.29%，p90 2.99%。

| closure | 环境光 median relative-L1 | p90 | median log-RGB L1 |
|---|---:|---:|---:|
| ggx-k3 | 7.50% | 24.13% | 0.0064 |
| ltc-k3 | 3.84% | 21.31% | 0.0028 |
| sg-k8 | 2.59% | 12.15% | 0.0020 |

真实探针使用固定的 Poly Haven CC0 清单，共 12 个 HDRI，每个取四个方位旋转。
