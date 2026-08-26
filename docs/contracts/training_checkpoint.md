# TrainingCheckpoint@2 合同

format 为 `ncls.training-checkpoint@2`。envelope 精确保存 method key/descriptor identity、implementation identity、完整 training config/hash、data source identity、source contracts/state IDs、step/phase、selection evidence、model/optimizer/scheduler/scaler/RNG state。

读取时先验证 `.pt.sha256` sidecar，再严格解析字段并按 `MethodDescriptor.tensor_state_schema` 校验 tensor key、dtype、固定/符号 shape 与有限性。旧格式没有 reader、alias、converter 或自动探测。
