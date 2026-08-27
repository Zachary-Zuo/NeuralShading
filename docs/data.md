# 数据与 reference

四个正式 source family 通过 `SourceFamilyDefinition` 产生 canonical `SourceSnapshot`，再由对应 `ReferenceProgramDefinition` 求值。LayerStack 使用随机游走，OpenPBR、MERL、MaterialX 使用各自权威实现；pbrt 不进入 discovery。

采集与在线训练共享 query/batch 合同。离线入口读 `reference-shard v5` / `reference-corpus`；在线入口返回同设备的 CUDA `TrainingBatch@1`。MaterialX formal route 在 GPU 上生成独立 UV、half/difference 或 conditioning direction、指数 mip、Gaussian footprint 与 cone mollification，native-feature mip 在 CUDA 上 bilinear 过滤，Falcor reference 通过 shared buffer直接写回 CUDA。response 不经过 CPU/NumPy readback。

`TrainingRouteRequest` 明确 route name、global step、batch/direction shape、proposal、target estimator、filter recipe 与 seed。producer identity 覆盖 source snapshot、reference implementation 与 live producer implementation；recipe 的完整 JSON另由 config hash锁定。

正式 corpus 仍遵循 role 隔离、矩形 shard、canonical identity、原子写入与 hash 校验。源材质参数与资源都参与 snapshot identity。
