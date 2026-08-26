# 数据与 reference

四个正式 source family 通过 `SourceFamilyDefinition` 产生 canonical `SourceSnapshot`，再由对应 `ReferenceProgramDefinition` 求值。LayerStack 使用随机游走，OpenPBR、MERL、MaterialX 使用各自权威实现；pbrt 不进入 discovery。

采集与在线训练共享 query/batch 合同。离线入口读 `reference-shard v5` / `reference-corpus`，在线入口使用 Falcor shared buffer 并返回 CUDA `TrainingBatch@1`。方向 proposal、target estimator、seed 与 shard 规则属于 recipe，不拥有独立 collector、manifest 或 reader。

正式 corpus 仍遵循 role 隔离、矩形 shard、canonical identity、原子写入与 hash 校验。源材质参数与资源都参与 snapshot identity。
