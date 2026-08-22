# 解析环境光探针表示上界

数据集：`data\v0_oracle_512`；26 个光照探针；2048 个 tiles。

A/B 环境光 relative-L1 噪声：median 0.47%，p90 3.92%。

| closure | 环境光 median relative-L1 | p90 | median log-RGB L1 |
|---|---:|---:|---:|
| ggx-k3 | 8.60% | 25.40% | 0.0075 |
| ltc-k3 | 4.11% | 18.58% | 0.0028 |
| sg-k8 | 2.63% | 10.95% | 0.0021 |

探针组包含均匀光、阴天、太阳与天空以及多软箱光。真实 HDRI 的结果见 `ibl_real.md`。
