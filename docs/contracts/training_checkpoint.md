# TrainingCheckpoint@3 合同

format为`ncls.training-checkpoint@3`。envelope精确保存method key/descriptor/implementation identity、完整`TrainingConfig@3`及hash、source contracts与snapshot IDs、reference program identity、query stream identity/state、step/phase、selection evidence、model/optimizer/scheduler/scaler/RNG、lifecycle与validation state。

query stream state包含每条typed route的generator state与request count；resume必须继续同一online query序列。source locator解析结果、reference实现、method/source adapter、query recipe或route定义任一漂移都改变identity并拒绝恢复。

读取时先验证`.pt.sha256` sidecar，再严格解析字段，并按`MethodDescriptor.tensor_state_schema`校验tensor key、dtype、固定/符号shape与有限性。旧格式没有reader、alias、converter或自动探测。
