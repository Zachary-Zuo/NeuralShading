# TrainingCheckpoint@4 合同

format为`ncls.training-checkpoint@4`。checkpoint精确保存method descriptor与component manifest、完整`TrainingConfig@4`及phase graph hash、reference program/plan、native asset collection、query stream和source snapshot identity，以及`global_step + phase_index/name/step`游标。

优化状态属于当前phase，包含named optimizer state、phase scheduler和precision/scaler状态；跨phase只按`optimizer_state_policy=carry-overlap`传递同名重叠参数。checkpoint同时保存model tensor、RNG、query stream、每个parameter group的finite/nonzero-gradient/actual-update coverage、validation与selection evidence。完成态不携带已失效的optimizer状态。

resume必须恢复同一config、phase graph、method implementation、component manifest、source snapshots、reference plan、asset collection和query stream；任一identity漂移都拒绝。读取先验证`.pt.sha256`，再严格解析字段，并按`MethodDescriptor.tensor_state_schema`校验tensor key、dtype、固定/符号shape与有限性。没有旧格式reader、alias、converter或自动探测。
