# 服务器实施清单

当前只完成任务规划。以下复选框仅在服务器有实际证据时勾选，不把计划当已实现。

## S0：建立可独立运行的监管基础

- [ ] 读项目/learning/data spec 与本任务，检查服务器 Codex CLI/认证可用性、Conda、锁定 Linux reference、资产 manifest；不输出凭证。
- [ ] 在任务 `scratch/` 做环境诊断；可复用监管器放 `tools/research/`，仅使用标准库起步，不另建训练引擎。创建小型可测试模块管理事件、deadline、进程/GPU 租约、台账与 CLI 决策适配。
- [ ] 实现结构化决策 schema、事件去重、空响应处理、状态原子落盘、固定 T0 的恢复、独立 deadline watchdog；单个串行 Codex 决策工作树，独立训练 checkout。
- [ ] 用假作业/假 Codex 和虚拟时钟测试：正常完成、失败/重复事件、无输出、token 解析、决策崩溃、supervisor 重启、排队时到期、Codex 挂住时到期、PID 重用、未授权 GPU、stdout 大量输出。测试使用秒级时钟，不消耗 24h 或真实五卡训练。
- [ ] 真实短 smoke 验证一个终端事件唤醒 Codex、输出可解析、检查后 scoped commit、终端断开后托管存活、watchdog 能终止仅属于本 campaign 的假作业。记录已实现启动命令和状态查询命令到 `server-handoff.md`；在此之前不得宣称服务已启动。

S0 是用户明确要求无人值守的必要基础设施，不把复杂 watcher 变成所有普通训练的通用 gate。不要顺手改 Trellis runtime。若 CLI/托管能力不可用，提交具体阻塞与可运行部分，不用当前聊天的长 sleep 假装独立服务器监管。

## S1：启动时间窗口并冻结研究协议

- [ ] 从干净且已提交的任务实现创建 campaign；保存 T0/deadline、物理 GPU 5–9 的 UUID、有效源码/资产/环境身份。开始后 D0、候选实现和调试均计时。
- [ ] 执行 E00；在看效果前完成独立 train/selection/final RF、主指标/CI、配置/seed/预算冻结。实现真实 summary-control 和测试，登记与旧版本的区别。
- [ ] 建立 `research/experiment-ledger.md`、`research/decisions.md`、`research/final-report.md`，只写真实结果；失败/未执行显式保留。不要先生成虚构 run 或指标。
- [ ] 首次 D1 提交配置和源代码后再启动，训练日志/大产物进入 outputs。相同 query 的独立两臂应有 recipe witness。

## S2：事件驱动的实验与分析

- [ ] 依 E01/E02 的观测选择 E03–E10 或预登记 R-###，每组冻结预算和停止规则；按 [实验表](experiments.md) 的触发条件优先归因与确认。
- [ ] 每次新形态读 method constraints；核算真实 MAC/reads/state/latent，明确自然成本变化。代码修复通过相关 unit/GPU/生成文件检查后 commit，不把不相关 dirty path 带入。
- [ ] 用事件账本证明训练等待区间无模型轮询；记录每次模型调用与 token。出现共用 reference 或 split 缺陷则暂停相关比较，以新身份修复并 fresh rerun。

## S3：到期交付

- [ ] 22h 收束，补齐可完成的 pairs/确认，最终少量候选做盲测、成本和阶段 export/parity；23h 进入报告，24h 停止探索。
- [ ] final report 包含 T0/deadline/实际执行时长、停机/失败、GPU-hours、调用/token、逐条结论/证据/CI、negative/未判定/未执行项、真实产物索引、下一步。
- [ ] 核对所有授权 GPU 的本任务作业已结束、无遗留子进程，最终 scoped commit，不 push；正常/degraded 明确区分。按 Trellis 记录 journal 并归档这个 24h 任务，不把“还有可能提高质量”作为延期理由。

## 回退点与质量边界

保留前继提交作为可部署基线。每个 candidate 有独立源码 commit/config identity；失败不覆盖原始输出。代码、生成的 ABI/Slang 和测试保持同步。snapshot/GT/native semantics 不通过时只能交付平台/语义诊断，不能输出模型胜负。

本机此前的 Windows unit/GPU/build/step 2 验证只覆盖前继实现；S0 监管器、Linux E00、corrected summary、冻结 D1 和 24h 生命周期目前均未验证。服务器执行前依据 `dev-environment.md` 报告本机状态，所有 Python 使用 `neural-shading`。
