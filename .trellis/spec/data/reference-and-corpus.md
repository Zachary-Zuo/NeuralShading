# Reference 与 batch

四个正式 family 共享 query stream 和 `TrainingBatch@1`。offline reader 与 live Falcor executor 只是 producer；live tensor 必须驻留 CUDA、显式同步并受 lease 保护。proposal、target estimator、seed 与 sharding 是 recipe。HDF5 只由 offline corpus sink 写入。旧平行采集/reader 不存在。详见 `../project/unified-pipeline.md`。
