---
name: data-reference-and-corpus
description: reference provider 与 corpus 的执行规则：ReferenceProvider 只暴露原生 state / query / response、矩形 shard、五个 query role、噪声预算分层、reciprocal 只作 scorecard、reference shader 单一源、不保留旧 reader
paths:
  - src/ncls/data/**
  - shaders/ncls/reference/**
  - shaders/ncls/data/**
  - configs/corpus/**
  - references/**
---

# reference 与 corpus 执行规则

## ReferenceProvider

- 公共 `BaseProvider`（`src/ncls/data/providers/base.py`）只暴露原生 state、surface query、direction query 和 reference response；新族实现自己的 state 解析、难度分级（`W/G/S` + `T/M`）和 proposal adapter，再加对应 `CorpusPlan`。
- `ReferenceDescriptor`（`src/ncls/data/contract.py`）声明 `family_id`、`reference_id`、`native_schema_id`、`incident_domain`（`upper-hemisphere` / `full-sphere`）、`deterministic`、`capabilities`、`implementation_sha256`（`implementation_hash()` 对实现文件求 hash）。
- 参数式族（LayerStack、OpenPBR 常量资产）从源参数推难度；资产式族（MERL、MaterialX）用 `difficulty-probe-v1` 实测分级，结果随资产 manifest 版本化。
- 每个 GPU dispatch ≤ 4,096 条 query；更多时 provider 自动切 tile。

## shard 与 corpus（`docs/contracts/reference_dataset.md`）

- 一个 `reference-shard` v5 固定一个结构 family、一个 query role、一个 `direction_count`——训练 batch 因此永远是矩形，不 padding。`ReferenceDataset.open()` 只接受 v5，没有旧格式探测或转换。
- 五个 role：`train / validation / test / adversarial_probe / dense_slice`；每个 state 必须五个 role 齐全，各 role 独立 seed 与方向表，validator 拒绝同一 state 跨 role 的方向 hash 碰撞。
- `responses/` 保存 `mean / variance / replica_mean_a|b / sample_count / valid / event_flags / reference_pdf` 和 reciprocal 三元组；`queries/` 逐 query 落盘 proposal PDF 与 `1/(N·pdf)` 权重，训练分布与均匀立体角指标因此能分开算。
- `reference-corpus` v1 内嵌完整 `CorpusPlan` 与 `plan_sha256`；v2 再内嵌版本化 `corpus-selection`（P1 的 30-state selection：`configs/corpus/layer-stack-p1-v1.selection.json`）。validator 从基础计划重新枚举 state 集合，不信任 manifest 自报。
- `corpus_id` 由语义内容计算，不依赖文件名、时间或容器字节布局。

## 噪声预算分层（`docs/data.md`「Reference 噪声与 reciprocal pair」）

| 用途 | 目标 SE p95 | 最终 group 门 | cap | 达 cap 未达标 |
|---|---|---|---|---|
| validation / test 主 response | 0.04 | 0.10（个别 state 在 CorpusPlan 显式晋升） | 262,144（晋升可到 1,048,576） | shard 失败，不写入 |
| train 主 response | 0.06 | 0.25 | 262,144 | 失败 |
| adversarial / dense | 0.08 | 0.50 只作参考 | 262,144 | 无条件落盘 |
| reciprocal（全部 role） | 0.20（train 0.50） | 0.999 只作参考 | 65,536（train 4,096） | 无条件落盘 |

- 只有排名 GT 的门是硬失败；有实测失败证据的 state 才在 `reference_budget.state_sample_promotions` 逐 state、逐 role 晋升，不整体翻倍。
- reciprocal pair 只进 source-aware reciprocity scorecard，不进训练、方向 L1 或能量主指标；高噪声 reciprocal 不得支撑质量结论。
- dense slice 默认 `4×8192`，只把 dense audit 证明峰邻域不足的 state 晋升到 `4×16384` 并独立成 shard；这两个值之外的密度不属于 v1。

## reference shader（`shaders/ncls/reference/`）

- 随机游走 reference 已是单一源：`random_walk_reference.slang` 同时服务采集（`shaders/ncls/data/reference_layer_stack.cs.slang`）与 viewer。新族的 reference shader 同样只写一份，采集入口与 viewer 都 `#include` 它。
- `interfaces.slang` 提供四种界面的 evaluate / pdf / sample 三件套；纯 frame、Fresnel、cosine、GGX/VNDF 在 `shaders/ncls/scattering/` 单一拥有。`sampling.slang` 只保留 PCG/reference RNG、体相函数，以及“恰好一次 `nextFloat2()` 后转调公共 VNDF”的薄包装。改这些文件必须保持现有 reference GPU 测试、固定 seed 随机数消费与已采集结果的 hash 不变（`p1_v2_plan.md` S2.2 的验收）。
- reference 可以在 `evaluate()` 内部用随机样本、逐像素累积；它不参与 MethodBundle 成本比较，也不生成通用 packet。

## 反例

- 给采集器加 `--profile` 之类可拼接参数绕过 `CorpusPlan`。
- 把两个 `direction_count` 混进同一 shard。
- 用 train 的放宽噪声数据当排名 GT。
- 为"整体提升质量"把全 corpus 的 cap 翻倍。
- 保留迁移前 HDF5 的 reader / converter；需要历史结果时用对应 Git 提交。
