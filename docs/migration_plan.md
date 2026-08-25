# P0 工程迁移记录

## 当前结论

P0 已把数据采集、学习入口和评测协议迁到一条正式架构：

```text
CorpusPlan
  → 按 family / role / density 拆分的 reference-shard v5
  → reference-corpus manifest
  → LearningPipeline + TrainingConfig v1
  → quality-v1 report
  → matched state-block paired bootstrap
```

正式入口只有可读名称和结构字段；精确身份由规范化配置或 descriptor 的 SHA-256 给出。迁移前的数据格式、训练配置、pipeline、gate、reader 和转换器不在当前工作区保留，历史证据由 Git 提交追溯。

## 已冻结的边界

- 源材质保留原生状态和族专属 reference；`LayerStackIR` 只属于 LayerStack，不是公共 GT 表示。
- `CorpusPlan` 同时声明 state 分布、采样密度、source split、reference 预算和 shard 策略。
- 一个 `reference-shard` 只有一个 query role 和一个方向数；训练 batch 始终为矩形。
- `reference-corpus` manifest 内嵌完整计划、计划 hash、语义 dataset ID 和文件 hash。语义 corpus ID 不依赖时间、路径或容器字节布局。
- candidate 只输出线性 RGB `f`；cosine 由 `quality-v1` harness 乘一次。
- 只有 sanity 会使结果无效。方向域与能量域是主指标；峰形、互易性和分组长尾属于 scorecard；reference SE 只作诊断。
- 正式比较要求相同 data ID、完全相同的至少 50 个 test state、matched 训练预算，并以 state 为 block 做不少于 1,000 次的 95% 配对 bootstrap。
- `CompiledMaterial` 和 `ScatteringState` 由 backend 私有；公共运行时仍只依赖 `prepare/evaluate` 及声明过的可选 capability。

## P0 实现状态

- `configs/corpus/layer-stack-v1.json` 冻结 28×10 LayerStack 状态、W/G/S 密度、+T/+M 采样、G1/G2/G2s split 和 adaptive reference 预算。
- dense slice 默认 `4×8192`；实际 response 审计只把峰邻域不足的 state 晋升到 `4×16384`，并保持独立矩形 shard。
- HDF5 v5 保存原始和 reciprocal paired response；corpus validator 检查 hash、角色完整性、state 元数据稳定性及 split/source 泄漏。
- learning registry 在 P0 保持为空。P1 候选必须用 `LearningPipelineDescriptor` 的 `data/model/fitting/runtime` 结构注册。
- `configs/evaluation/quality-v1.json` 是固定评测协议；报告记录其精确 hash。能量 bootstrap 对完整 `(state, wo)` 行重采样，不使用 state 摘要近似。
- 已有解析实现继续作为部署回归 fixture、成本对照或 sampling proposal；它不注册成研究候选，也不形成第二套训练/评测入口。

## 后续顺序

1. 生成并审计 LayerStack v1 corpus，冻结 manifest；
2. 在 P1 注册首批 evaluator 候选，做 matched 比较；
3. 稳定共享 decoder、latent 与 compiler；
4. 每个研究阶段收尾时执行一次 MethodBundle、Slang parity、成本与 viewer 证据；
5. evaluator/compiler 稳定后再扩展 matched sampler、环境积分和 Falcor/UE 式工作流。

迁移记录只描述当前仍有效的边界和执行顺序，不保存单次测试结果；运行报告统一进入 `artifacts/`。
