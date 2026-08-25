# 02 Directional Mollification 数据充分性

## Goal

以可复现实验证明现有 `layer-stack-p1-v1` v5 corpus 是否能忠实支持 NVIDIA directional mollification；若不能，在训练 `03` 前完成最小、版本化的数据合同和 reference-response corpus。

## Scope And Dependencies

- 前置任务：`01-reusable-scattering-math` 已完成、提交并归档。
- 本任务输出唯一的数据决定与可用 corpus identity；`03` 不得自行重新解释旧数据或绕过该决定。
- 本任务是复杂任务；启动前必须根据 `01` 的最终公共方向/measure 语义补全并审阅三件套与 context manifests。

## Requirements

- 在运行对比前冻结 representative states、`wo`、cone distribution/radius、reference sample count、误差指标和通过阈值。
- 覆盖 diffuse、窄导体峰、grazing 和四个既有尾部 state，将现有 peak-aware 邻域重建与同 anchor 的新鲜 cone-averaged reference queries 做 matched 比较。
- learned frames、sampler KL、解析 `sample/pdf` 本身不得作为重采理由；本任务只处理 directional mollification 的数据语义。
- 若 adequacy 通过，保留现有 v5 corpus 并记录可复现构造方法和误差证据。
- 若 adequacy 未通过，先版本化定义 cone radius、anchor/group、jitter distribution、sample count 与 curriculum level；随后生成并验证能供 `03` 直接训练的最小新 corpus identity。
- 不删除或覆盖现有合法 v5 corpus；所有运行摘要进入 `artifacts/`，reference response 只进入 `data/reference-responses/`。

## Acceptance Criteria

- [ ] adequacy protocol、阈值与代表性覆盖在查询结果产生前冻结。
- [ ] matched 结果能明确得出“复用现有 v5”或“使用新版本 corpus”二选一结论。
- [ ] 复用路径包含确定性的邻域/权重构造与重复运行证据。
- [ ] 新数据路径包含 schema/manifest、完整生成验证、reference provenance 和 corpus identity，不存在局部静默重解释。
- [ ] `03` 获得一个明确、可读取、可追溯的训练数据入口。
- [ ] 子任务完成质量检查、提交并归档后，父任务才允许进入 `03`。

## Out Of Scope

- neural 模型训练、sampler family 选择、MethodBundle 和 viewer。
