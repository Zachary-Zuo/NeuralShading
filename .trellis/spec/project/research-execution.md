---
name: project-research-execution
description: 研究任务的验收来源、需求保真、数据与实验冻结、失败分类、长运行进度与性能及收尾边界
paths:
  - .trellis/tasks/**
  - configs/learning/**
  - configs/evaluation/**
  - docs/research/**
---

# 研究任务规划与执行

> 本规则约束“如何把研究需求变成可执行任务”。它防止把观察值写成硬门、把 pilot 当正式实验、把运行日志回写成事后需求，以及用连续执行授权扩张范围。

## 验收标准只来自需求与正确性

每条 acceptance criterion 必须标明类型与来源：

| 类型 | 能否决定任务完成 | 合法来源 |
|---|---|---|
| 需求交付 | 可以 | 用户明确要求、已批准父任务合同 |
| 理论 / 语义正确性 | 可以 | 权威方法定义、项目接口、数学不变量、数据角色隔离 |
| 数值实现正确性 | 可以 | 浮点误差分析、独立 oracle、统计置信度设计；必须在正式结果前冻结 |
| observed quality / time / memory | 默认不可以 | 实验结果，只用于报告与相对比较 |
| 研究软线 | 不可以 | 版本化参考、成本分类或晋级排序 |

- AI 不得把某次历史 run、已有候选或直觉产生的数值写成任务 hard gate。一个数值若会决定“任务成功 / 失败”，必须记录 `source / scope / why_hard / failure_action`；缺任一项就降为 report-only 指标。没有权威来源而又确实需要产品目标时，先请用户确认，不能代替用户编一个数。
- parity、PDF 归一化、sample→pdf、有限性等 tolerance 验证的是同一数学实现；其容差必须由数据类型、独立 oracle 或预先隔离的 calibration 推导，不能在看过正式 test 结果后反向包住当前误差。
- “质量较差”不等于“实现错误”。忠实实现且稳定收敛后，低 quality 是研究结论，不自动触发改结构、换 seed 或继续训练。

## 需求保真：既不简化，也不扩张

- PRD 中每个验收项都要能反向映射到用户需求或已批准的上层合同。无法映射的候选、ablation、额外 backend、额外 seed、部署变体或统计表不得成为收尾前置。
- 复现 prior art 时，先完成 method correspondence：原结构、规模、输入输出、loss、训练 lifecycle 和 sampler 必须逐项对应。接口适配与研究预算适配要显式登记；缩模、替代 loss、替代特征或另一 sampler 使用独立 identity，不能为了容易跑而沿用原方法名称。
- 不用较小的、较安全的或容易通过现有测试的实现替代需求；也不为“也许以后有用”增加需求没有要求的轴。最小实现指完整满足需求的最小边界，不是功能缩水。
- 连续执行授权只免去已授权范围内的重复确认，不授权新增 hard gate、扩大数据 / 训练预算、追加 seed、排队新方法变体或修改冻结协议。

## 规划文档不是运行日志

- `prd.md` 保存需求、边界和验收；`design.md` 保存实现合同与取舍；`implement.md` 保存执行清单与 rollback point。单次 run 的 step、metric、失败值、v1→vN 过程写入 `research/` 或 `artifacts/`，不得不断追加到 PRD 来事后合理化执行。
- 正式结果产生后只能因两类原因修改规划合同：用户明确改变需求，或证据证明原合同在理论 / 实现上错误。修改时记录 `trigger / invalidated evidence / scope impact / rerun required`。
- 数据量、训练量、方法身份、验收门或计算成本发生实质扩大时，回到 planning 并取得用户确认；不得把“下一版 config 有新 hash”当作已获得范围授权。

## Pilot、formal 与失败分类

1. **Pilot** 只回答协议、正确性、内存和吞吐是否可行；输出标为 diagnostic，不进入正式比较，也不直接晋升为正式 shard/checkpoint。
2. **Freeze** 在 formal 前冻结方法 correspondence、数据/训练 config、随机种子策略、预算、选择规则和验收来源。
3. **Formal** 只执行冻结合同。实现 bug 修复后用新 implementation/run identity 重跑；不能覆盖旧结果。
4. **Interpret** 把失败分成：implementation defect、protocol/design defect、resource/throughput defect、正常的 empirical outcome。只有前三类允许回到实现或规划；最后一类直接登记结果。

禁止自动 `v1 → v2 → ...` 循环直到过门。达到冻结 cap、训练没有通过 correctness/convergence、或吞吐偏离 preflight 时，应先停止并分类原因；扩大预算或改变规则是新的 planning 决策。

## 长运行的进度与性能

任何预计明显长于 smoke 的采集、训练或评测都应保持简单、直接可观察：

- Python 长循环使用 `tqdm` 展示真实工作单元的 `completed / total`、elapsed、rate 与 ETA；训练可在 postfix 中显示当前 phase 和必要指标。长 validation、reference collection 等独立阶段也要有自己的进度，不能只在阶段结束时打印一行。
- 按 batch / chunk 更新进度，避免为了刷新进度在每个 sample 上引入 GPU 同步或高频 I/O。进度条反映已经完成的工作，不用“进程存活”或定时空转伪造进度。
- 启动前只需用 smoke 确认正确性、显存和基本数据流，并由短时实测速率给出大致 ETA；不得把覆盖所有阶段的详尽 profile、heartbeat、watcher、PID 状态机或 liveness timeout 设成普通长运行的统一准入门。
- 当 `tqdm` 显示的吞吐 / ETA 明显不合理、长时间没有实际 work unit 完成，或 GPU 利用方式与预期不符时，暂停盲目长跑，对代表性慢段做有针对性的 profile。根据证据检查 batch 几何、data/I/O、host→device、forward、backward、optimizer 或 validation/checkpoint，并优先优化占主导的热点。
- 只有任务本身确实要求无人值守调度、跨进程恢复或外部作业管理时，才按该运行模型增加 heartbeat、watcher、进程身份和超时回收；这些是按需基础设施，不是研究训练默认必须实现的功能。

## 收尾判断

任务完成的证据是：需求逐项交付、方法 / 数据语义忠实、理论与数值正确性成立、必需生命周期可运行且产物可恢复。未被用户要求的更好 quality、更多 seed、更多对照、更多集成和“顺手完善”不阻塞收尾。

```text
错误：旧 run p95=0.12 → 写 hard gate p95≤0.10 → 不过就改模型/数据/seed继续跑。
正确：先证明方法对应、梯度与数学正确、validation 相对初始化改善；p95=0.12 作为 observed result 报告。

错误：正式采集达 cap → 根据失败值提高 cap → 自动发布 v2/v3/... 直到过门。
正确：pilot 推导并确认一次 frozen plan；formal 只按该 plan 自适应采集，达 cap 未满足就停止并返回 planning。
```
