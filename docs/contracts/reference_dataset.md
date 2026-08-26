# Reference 数据合同

离线 `reference-shard v5` 固定 source family、role 与 direction count；`reference-corpus` 组合矩形 shards 并冻结 plan/selection identity。online reference 不写 shard，而是直接产生同 schema 的 CUDA `TrainingBatch@1`。五个 query role、state metadata、direction proposal、response/PDF/sample count 与 hash 均严格校验。
