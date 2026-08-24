# Learning pipeline 与评测

当前 P0 先冻结语料和评测协议，候选 registry 保持为空。P1 开始实现 M1–M6 时，每个候选再显式注册；仓库不为迁移前候选保留 registry 别名。

## Pipeline 身份

候选使用短而可读的名称，例如 `film-evaluator-s-v1`。descriptor 不再把所有选择拼进一个长 ID，而是分成四组结构字段：

- `data`：reader、partition、source adapter；
- `model`：representation、architecture、latent；
- `fitting`：gradient/direct-fit/hybrid 路径和 loss；
- `runtime`：compiler 与 exporter。

名称用于人读，descriptor 的 SHA-256 才是精确实现身份。容量档位固定写成 `S/M/L`。schema 见 [`learning_pipeline_v1.schema.json`](../src/ncls/learning/schemas/learning_pipeline_v1.schema.json) 和 [`training_config_v1.schema.json`](../src/ncls/learning/schemas/training_config_v1.schema.json)。

## 数据读取

训练和评测的 `--data` 可以指向单个 `reference-shard`，也可以指向完整 `reference-corpus` manifest。corpus reader 建立全局 state 索引，但每个实际 batch 只从一个矩形 shard 读取，因此 W/G/S、普通 role 和 dense slice 的不同方向数不会被混装或 padding。

`parametric-v1` 使用 source split 与同名 query role 的交集；`target-visible-v1` 只按 query role 切分，允许 encoder 或 bounded refinement 读取 source-test state 的 train response；`workflow-v1` 用于资产式族的固定工作流批量评测。三者必须在候选 descriptor 中明确选择。

## 固定四层 quality-v1

候选只返回线性 RGB `f`。harness 在指标内部乘一次 `|cos θi|`，与 HDF5 的 response measure 对齐；candidate 不能替换 metric suite。报告以可读名称 `quality-v1` 和 [`quality-v1.json`](../configs/evaluation/quality-v1.json) 的 SHA-256 共同标识协议，比较器会拒绝 suite hash 不一致的报告。

1. sanity：数据/hash、role 完整性、输出 finite、颜色范围、checkpoint 可恢复、fitted state 只来自 train。只有这一层会把报告标成无效。
2. 主指标：solid-angle weighted normalized L1，先聚合到 state 再报告 median/p95；半球或整球积分能量相对误差按 `(state, wo)` 报告 median/p95。checkpoint 固定按 validation 的 directional state-median 选择，同值用 state-p95 决胜。
3. scorecard：log-domain error、95% 峰高支持集角距、峰高比、top-5% energy recall、source-aware reciprocity deviation，以及 difficulty/tag/family/structure/cohort/state breakdown。它们解释长尾，不单独否决候选。
4. diagnostics：绝对误差、reference SE 与两者比值。`model error / reference SE` 不再是 kill gate。

reciprocity 使用语料中落盘的 reciprocal paired response，比较“模型互易偏差与 source 自身互易偏差之差”；因此非严格互易的 source 不会被错误地强制为零。

每份 checkpoint report 另记录 `B_asset`、`B_shared`、`C_prepare`、`C_eval` 和参数数目。成本用于 Pareto，不在研究期提前淘汰。

## 命令

注册候选后，训练只接受完整 `training-config-v1`：

```powershell
conda run -n neural-shading python -m ncls.cli learn train `
  --data artifacts/corpus/layer-stack-v1.json `
  --config configs/learning/film-evaluator-s-v1.json `
  --run artifacts/runs/film-evaluator-s-v1-seed-1
```

显式读取 validation、test、adversarial probe 或 dense slice：

```powershell
conda run -n neural-shading python -m ncls.cli learn evaluate `
  --data artifacts/corpus/layer-stack-v1.json `
  --checkpoint artifacts/runs/film-evaluator-s-v1-seed-1/checkpoints/best.pt `
  --split test `
  --output artifacts/runs/film-evaluator-s-v1-seed-1/quality-test.json
```

matched 对照必须使用同一 `data_id` 和完全相同的 test state；比较入口以 state 为 block 重采样每个 state 的完整方向主指标和 `(state, wo)` 能量行，做至少 1,000 次、95% 置信度的配对 bootstrap：

```powershell
conda run -n neural-shading python -m ncls.cli learn compare `
  --baseline artifacts/runs/baseline/quality-test.json `
  --candidate artifacts/runs/candidate/quality-test.json `
  --output artifacts/comparisons/baseline-vs-candidate.json
```

历史 profile、训练 schema、gate 和 pipeline 不设 reader、converter 或 registry 别名。历史证据只从对应 Git 提交读取；正式新 run 必须使用当前 corpus 与 quality-v1。
