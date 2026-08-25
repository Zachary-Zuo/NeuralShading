---
name: project-code-organization
description: 按语义划分文件、不设行数硬限、临时诊断代码放 task 目录、复用与第一性原理、不留兼容层并递归迁移清理旧代码
paths:
  - src/**
  - shaders/**
  - apps/**
  - tools/**
  - scripts/**
  - .trellis/tasks/**
---

# 代码组织

## 按语义划分，不按行数

- 一个文件承担一个语义单元：一个合同、一个 provider、一个 pipeline、一个 pass、一个 backend core。不设行数硬限；触发拆分的是"文件里出现了第二种职责"，不是某个数字。
- 已知债务（不作为新代码的模板；触碰时按语义拆，不新增同类文件）：`apps/viewer/NclsViewer.cpp`（约 2.5k 行，UI、pass 调度、capture、场景加载混在一起）、`apps/viewer/shaders/ReferencePathTracer.cs.slang`（约 1k 行，四族 reference 分派）、`src/ncls/learning/evaluation/p1_audit.py`（约 0.9k 行）。
- 目录按功能块组织（见 `project/index.md` 表）；新目录先在 `docs/architecture.md`「目标目录」登记。
- 不为未来阶段预留抽象、基类或配置轴；只暴露当前需要的轴。范例：`src/ncls/learning/pipelines/lobe_residual.py` 只暴露 `K ∈ {2, 3}` 与 `correction ∈ {none, log32}`。

## 复用与第一性原理

- 写新功能前先搜现成件：界面 evaluate / pdf / sample 三件套在 `shaders/ncls/reference/interfaces.slang`，各向异性 VNDF 在 `sampling.slang`，appearance loss 在 `src/ncls/learning/pipelines/appearance_loss.py`，canonical JSON / hash / 原子写在 `src/ncls/data/profiles.py`、`src/ncls/learning/training/runner.py`。抽函数共用，不复制。
- 从问题本身推导实现，不从"上一版是这么写的"出发；旧实现只作对照或 parity oracle。
- 反例：为新 pipeline 复制一份 loss；为新 backend 再写一套手工偏移的权重序列化；为"以后可能的族"预留空 adapter。

## 临时诊断代码

- 一次性诊断脚本、spike、对照绘图放在当前任务目录 `.trellis/tasks/<task>/scratch/`；运行输出进 `artifacts/`。不进 `src/`、`scripts/`、`tools/`。
- `scripts/` 只放长期入口（构建、获取资产、benchmark、环境）；`tools/reference/` 只放长期 reference 验证工具。
- 需要长期保留的诊断（如 `ncls learn audit-p1`）作为正式 CLI 子命令进 `src/`，并配测试。
- 现存 `scripts/spike_slangpy_autodiff.py`、`spike_slangpy_checks.py`、`spike_slangpy_autodiff.slang` 是 P1.0 一次性 spike：远程跑完并把结果回填 `docs/research/p1_v2_plan.md` 后删除。

## 不留兼容层，递归迁移

- 仓库不保留 alias、legacy reader、converter、旧格式探测或"兼容分支"（`docs/repository_policy.md`；`docs/learning.md`「不设 reader、converter 或 registry 别名」；`ReferenceDataset.open()` 只接受 v5）。历史证据由 Git 提交追溯。
- 改一个接口、字段名或合同时，在同一任务内递归迁移全部调用方、shader、测试、schema JSON 与文档，然后删除旧路径。不能"先加新的，旧的以后再删"。
- 实现者在改动中发现 alias、`legacy` / `old` 命名、兼容 fallback 或重复实现时，必须在报告里列出「待迁移清单」并提醒用户决定；不得默默保留。
- 已登记待迁移：
  - `src/ncls/core/representations/legacy_ltc_k2/`：作 `sample/pdf` parity oracle 与 `INclsScatteringBackend` 合同范例保留到 `lobe_residual` 接入完成，之后删除；
  - `film_m1` 的四处实现（`src/ncls/learning/models/p1_evaluator.py`、`shaders/ncls/backends/film_m1/`、`src/ncls/bundle/film_m1.py`、`apps/viewer/MethodBundle.cpp` 硬编码）：`p1_v2_plan.md` Phase 4 泛型 pass 与通用 exporter 落地后删除。
- 命名不含时间、作者或 `new` / `v2` 式相对词；版本进 schema、descriptor 或 `@N` 后缀。

## 产物与仓库边界

单次运行报告、capture、checkpoint、MethodBundle 一律进 `artifacts/`；根仓库只收源码、合同、配置、测试、中文结论和轻量 JSON 指标（`docs/repository_policy.md`）。
