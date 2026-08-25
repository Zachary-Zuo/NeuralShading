---
name: learning-pipeline-and-evaluation
description: pipeline 注册与训练配置、quality-v1 四层指标与 sanity 唯一否决、compare 的 matched 规则、checkpoint tail guard、experiment_log 登记格式、MethodBundle 导出边界
paths:
  - src/ncls/learning/**
  - src/ncls/bundle/**
  - configs/learning/**
  - configs/evaluation/**
  - docs/research/experiment_log.md
---

# pipeline、评测与导出规则

## 候选身份（`src/ncls/learning/pipelines/base.py`）

- `LearningPipelineDescriptor` 字段精确为：`data = {reader, partition, source_adapter}`、`model = {representation, architecture, latent}`、`fitting = {path ∈ gradient|direct-fit|hybrid, loss}`、`runtime = {compiler, exporter, deployment_candidate: bool}`，外加 `name / stage / supported_families / scope`。多一个或少一个字段都会 `ValueError`。
- 名称给人读（`lobe-residual-k2-v1`），descriptor 的 SHA-256 才是实现身份；`deployment_candidate` 是研究期分类，**不进 hash**，P1 v1 checkpoint 仍可评测。
- `register_pipeline(factory)` 在 `pipelines/__init__.py` 统一调用；重复名报错；没有 alias。
- partition 三选一并显式声明：`parametric-v1`（source split ∩ query role）、`target-visible-v1`（只按 query role，encoder / refinement 可读 source-test 的 train response）、`workflow-v1`（资产式族批量评测）。
- `parameter_costs()` 返回的 key 与现有 pipeline 一致（`test_deployment_budget.py` 的 `SOFT_BUDGET` 集合 + `B_shared`、`parameter_count`、`analytic_core_state_bytes`、`C_eval_excludes_analytic_core`），按部署实际 bytes 记账。

## 训练配置（`training/config.py`）

- 只接受完整 `training-config-v1`；`capacity ∈ {S, M, L}`；默认 `seed=20260824`、`steps=25000`、`checkpoint_selection="median_then_p95"`（旧默认不进 hash）。新配置用 `"tail_guard"`：先剔除 validation p95 > 该 run 至今最小 p95 × 1.25 的 checkpoint，再取 median 最小。
- `dataset_selection` 只允许 `state_ids / asset_ids / family_ids`。
- 预算档位：快速档（≤ 30 min GPU，只做 smoke，不入注册表）、标准档（全量阶段数据、共同 seed、4,000 步后 validation patience 早停）、冲刺档（×5–10）。P1 主搜索只用一个 deterministic seed；只有差距接近或轨迹异常才追加 seed。
- `run_manifest.json` 记录解析后配置、Git 提交、reference 实现 hash、合同版本、seed、依赖版本、输入产物 ID；命令行只能覆盖配置里已声明的字段。

## quality-v1（`evaluation/quality.py`，`configs/evaluation/quality-v1.json`）

- 候选只返回线性 RGB `f`；harness 在指标内乘一次 `|cos θi|`；candidate 不能替换 metric suite；报告记录 suite 的 SHA-256，`compare` 拒绝 suite hash 不一致的报告。
- 四层：① sanity（唯一会把报告标无效的层）；② 主指标：solid-angle weighted normalized L1 的 state median / p95，半球能量相对误差的 `(state, wo)` median / p95；③ scorecard：log-domain error、峰位角、峰高比、top-energy recall、source-aware reciprocity、分组 breakdown——解释长尾，不否决；④ diagnostics：绝对误差、reference SE 及比值——`model error / reference SE` 不是 kill gate。
- 当前参考线（可随证据修订）：state-median ≤ 0.05 且 p95 ≤ 0.15，能量 median ≤ 0.03；这是"值得进下一阶段"的参考，不是单指标 kill gate。
- `quality-v2` 与 v1 只差 `checkpoint_selection` 块；比较器接受两者。

## compare（`evaluation/comparison.py`）

- 只接受 `valid=True`、`evaluation_role=test`、hash 自洽的 quality 报告；baseline 与 candidate 必须同 `data_id`、完全相同的 test state，≥ 20 个 matched state，≥ 1,000 次 95% state-block paired bootstrap。
- 只允许声明的差异：容量曲线加 `--vary capacity`；其余未声明的训练字段差异直接拒绝。
- CI 跨零记"无显著差异"，不写"X 优于 Y"。30-state p95 的 CI 很宽，只作 selection 诊断；正式长尾结论用 ≥ 50 个 test state 并附 p90/p95、最差 state 清单、CI 与 leave-one-state-out。
- 有 core 的候选始终与 direct 候选配对报告；跨族汇总分层报告。

## experiment_log 登记（`docs/research/experiment_log.md`）

每个正式 run 一行：日期、run ID、候选+档位、数据版本（`data_id`）、预算档、seeds、方向 L1 med/p95、能量误差 med/p95、一句话结论（含是否部署候选、超了哪条线）、artifacts 路径。详细数值留 `artifacts/`；阶段结论写在表下方小节。

## 导出（`src/ncls/bundle/`）

- 从不可变 checkpoint 导出全新 bundle，计算全部内容哈希，记录源 run / checkpoint；bundle 只含推理必需内容与验证证据，不含 optimizer / TensorBoard。
- `runtime_class=realtime` 需要 descriptor `is_complete_realtime_backend`（`Prepare|Evaluate|Sample|Pdf|AnisotropicFrame`）且 `cost_claims` 满足硬线；否则 `diagnostic`。
- 通用 exporter（`bundle/exporter.py`，`p1_v2_plan.md` P4.1）抽 manifest 组装、写文件、hash、parity probe；各方法只提供 descriptor、params/layout、`compiled_materials/` 与 `cost_claims`。当前 `bundle/film_m1.py` 的手写序列化是待迁移债务。

## 反例

- 训练循环里按 test 结果选 checkpoint 或超参。
- 为一个候选单独调 loss 权重后再与别的候选"同表比较"。
- 把 PyTorch `benchmark` 的 kernel 时间当最终 viewer 时间。
- 报告里用"上界"描述有限预算下的结果。
