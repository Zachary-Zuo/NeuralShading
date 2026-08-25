---
name: learning-index
description: 训练、评测与导出层入口：LearningPipelineDescriptor 注册、TrainingConfig、quality-v1 四层指标、learn compare 的 matched bootstrap、experiment_log 登记、MethodBundle 导出；开发前检查与质量检查
paths:
  - src/ncls/learning/**
  - src/ncls/bundle/**
  - configs/learning/**
  - configs/evaluation/**
  - docs/learning.md
  - docs/research/**
  - tests/unit/test_training_config.py
  - tests/unit/test_pipeline_contract.py
  - tests/unit/test_quality_evaluation.py
  - tests/unit/test_deployment_budget.py
  - tests/unit/test_p1_audit.py
---

# 训练、评测与导出

> 这一块的职责：候选注册、梯度 / 直接 / 混合拟合、validation checkpoint 选择、固定 quality 评测、TensorBoard 与方法导出。它读 `reference-corpus`，产出 checkpoint 与 `MethodBundle`；它不碰 viewer 代码。

## 正式入口

```text
configs/learning/<name>.json（training-config-v1）
  → ncls learn train    → artifacts/runs/<run>/{checkpoints/best.pt, run_manifest.json}
  → ncls learn evaluate → quality-v1 报告（test / adversarial / dense）
  → ncls learn compare  → matched state-block paired bootstrap
  → ncls learn benchmark / audit-p1 / oracle-m3（诊断）
  → ncls bundle export-* → artifacts/exports/<bundle>/
```

代码：`src/ncls/learning/{data,pipelines,training,evaluation,direct_fit,source_adapters}/`、`src/ncls/bundle/`。研究路线与判据在 `docs/research/experiment_framework.md`，候选设计在 `model_candidates.md`，当前计划在 `p1_v2_plan.md`。

详细规则见 `pipeline-and-evaluation.md`；方法约束见 `project/method-constraints.md`。

## 开发前检查清单

- [ ] 已读 `project/method-constraints.md`，新候选通过注册时静态检查。
- [ ] 新候选用 `LearningPipelineDescriptor` 的 `data / model / fitting / runtime` 四组字段注册，`capacity` 字段省略，`deployment_candidate` 明确。
- [ ] 模型前向是 Slang（`core/shared-slang-backend.md`）；Torch 只有 loss、optimizer 与 parity oracle。
- [ ] 复用 `pipelines/appearance_loss.py` 与现有 `source_adapters/`，不复制。
- [ ] 改 `TrainingConfig` / descriptor / quality suite 字段时同步 `schemas/*.json`、对应测试，并确认旧 hash 不变（`test_training_config.py`）。
- [ ] 已判定开发机状态：训练与 GPU 测试只在"完整 / 仅 GPU"状态可做；静态状态把命令写进 `TESTING.md`。

## 质量检查

- [ ] 训练只读 train / validation；test 只由独立 evaluate 命令在配置冻结后读取一次。
- [ ] checkpoint 选择走 `training/selection.py` 的策略（新配置用 `tail_guard`）。
- [ ] 报告含成本字段（`B_asset`、`B_shared`、`C_prepare`、`C_eval`、state bytes、参数量）与 signed 能量比等诊断。
- [ ] 结论用 `learn compare` 的 paired bootstrap 支撑，CI 跨零记"无显著差异"。
- [ ] 正式 run 在 `docs/research/experiment_log.md` 登记一行并标注是否部署候选；快速档 smoke 不入表。
- [ ] 导出的 bundle 通过 `ncls bundle validate` 与 Python/Slang parity；`runtime_class` 如实标注。
