---
paper_id: "{stable-paper-id}"
title: "{official-title}"
authors: "{official-author-list}"
year: "{year}"
venue: "{venue}"
doi: "{doi-or-not-reported}"
report_status: "draft"
main_source: "{official-url-or-local-locator}"
supplemental_status: "available|unavailable|not-applicable"
official_code_status: "audited|available-not-audited|unavailable"
official_code_commit: "{commit-or-not-applicable}"
author_worker: "{worker-name}"
reviewer: "unassigned"
last_verified: "YYYY-MM-DD"
---

# {论文正式标题}

> 本模板的占位符只用于新建报告。任何适用但来源未披露的字段写“未报告”，不要删除章节或用常见设置补全。

## 1. 研究对象与报告边界

- 论文解决什么问题；
- 本报告覆盖哪个正式版本；
- 与 local material、scene transport、volume transport 或 load-bearing related 的关系；
- 明确不把哪些相邻问题算入该论文的方法。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` |  |  |  |  |
| Supplemental `S` |  |  |  |  |
| Official code/config/data `C` |  |  |  |  |
| Author page/talk/correction `A` |  |  |  |  |
| NeuralShading evidence `N` |  |  |  |  |

记录无法获得的来源、检索过的第一方入口及其影响。

## 3. 原论文的问题、假设与贡献边界

先按作者自己的定义说明目标、输入条件、输出能力和贡献。不要在本节评价新颖性或迁移价值。

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material/scene input |  |  |  |
| Runtime query |  |  |  |
| Direction/space/time/light coordinates |  |  |  |
| Output quantity and measure |  |  |  |
| Validity/domain restrictions |  |  |  |

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

用文字和必要公式重建从 source 到 runtime output 的完整路径。

### 5.2 持久化表示

记录 latent、texture/plane/grid、analytic parameters、weights、mip/LOD、quantization 和 per-asset/shared 分界。

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |

### 5.4 条件化、坐标变换与物理先验

分别记录 learned frame、half/difference、offset、warp、analytic core、visibility/transport decomposition 等机制；未使用的机制不凭类比补写。

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset/source assets/scenes |  |  |
| GT/reference renderer or measurement |  |  |
| Train/validation/test split |  |  |
| Spatial/directional/light/time sampling |  |  |
| Filtering/LOD/footprint |  |  |
| Augmentation/distillation/teacher |  |  |
| Online/offline generation |  |  |

## 7. Loss、optimizer 与训练 lifecycle

| 项 | 正式配置 | locator |
|---|---|---|
| Target transform/output transform |  |  |
| Loss terms and weights |  |  |
| Optimizer and hyperparameters |  |  |
| LR schedule |  |  |
| Batch/query count |  |  |
| Steps/epochs/stages |  |  |
| Initialization/seed/model selection |  |  |
| Hardware/training time |  |  |

代码 default、example 和 paper/formal 配置分别登记。

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path and frequency |  |  |
| Parameter count/MAC/FLOP |  |  |
| Shared/per-asset/state bytes |  |  |
| Texture/feature fetches |  |  |
| Precision/quantization |  |  |
| Hardware/backend/coherence |  |  |
| Time/FPS/latency |  |  |
| Precompute/prepare/amortization included? |  |  |

## 9. 实验 protocol、baseline、指标与结果

每个关键结果保留 dataset/scene、输入、模型版本、指标定义、hardware 和统计聚合；不可比结果不横向排名。

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `author-negative` / `ablation-inferior` |  |  |  |  |  |

没有作者负结果时明确写“在已获得第一方材料中未报告”，不得从最终方法推测失败历史。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Architecture |  |  |  |  |
| Data/query |  |  |  |  |
| Loss/training lifecycle |  |  |  |  |
| Runtime/export |  |  |  |  |
| Assets/evaluation |  |  |  |  |

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

逐项区分作者声明与由方法 domain 直接推出的边界。

### 12.2 未报告/材料不可得

列出对复现或解释有影响、但第一方来源没有披露的内容。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

### 13.2 成功所依赖的假设

### 13.3 可迁移机制与不能迁移的部分

### 13.4 与本项目 runtime contract 的关系

说明是否静态有界，以及更适合作为产品候选、teacher、proposal、compiler、prefilter 或 capacity diagnostic。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

只引用当前仓库中真实存在的 correspondence、配置、实现或 artifact；标记 `faithful`、`author-underspecified`、`interface-adaptation`、`budget-adaptation`、`intentional-deviation`、`suspected-defect` 或 `not-applicable`。

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |

## 16. 证据索引

按 `P/S/C/A/N/I` 汇总正文中使用的 locator，便于 reviewer 复核；不重复粘贴长段原文。

## Evidence review

```text
author_worker: {name}
reviewer: unassigned
reviewed_at: not-reviewed
sources_rechecked: []
findings_closed: []
remaining_evidence_gaps: []
review_status: changes-requested
```

### 完成检查

- [ ] main paper 已完整阅读，关键公式/图/表/图注/脚注已视觉核对；
- [ ] supplemental/appendix/勘误的可用性已检查；
- [ ] official code/config/data 的可用性与 commit 已检查；
- [ ] architecture、training、runtime 和主要结果均有 locator；
- [ ] 失败尝试与较差消融正确分类；
- [ ] paper/code gap 和“未报告”保留；
- [ ] `I` 分析晚于事实层，没有改写作者结论；
- [ ] NVIDIA 影响引用真实 `N` 证据；
- [ ] 假设包含 matched control、部署类别和证伪条件；
- [ ] 独立 evidence review 已完成。
