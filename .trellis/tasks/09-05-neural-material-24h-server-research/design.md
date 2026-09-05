# 事件监管与时间合同

## 运行结构

这是服务器上的独立研究进程：普通 Python 监管器管理 GPU 作业、时间、事件和恢复；一个串行的 Codex 决策进程在必要事件发生后分析证据、实现变化并提交。训练可并行，模型决策不并发修改同一仓库。当前文档定义待实现合同，仓库尚没有本任务的 24h 监管器。

采用已有 `codex exec` 非交互入口。它支持 JSONL 事件、末条回复输出、结构化输出和指定会话恢复；这些是脚本接口，24h 调度与可靠终止仍由本任务实现。[官方非交互文档](https://developers.openai.com/codex/noninteractive/)；本机已检查 `codex-cli 0.153.4` 的 help，服务器须重新检查实际版本与认证状态。

```text
监管器：await 作业退出 / 合并事件 / 单次 deadline timer
    → 紧凑事件包 → codex exec → 分析、代码检查、commit、结构化决策
    → 校验决策、登记 run、分配 GPU → 公共 ncls train/validate/export
作业日志持续落盘；普通等待期间不调用 Codex。
```

不依赖当前聊天窗口保持打开。服务器须有可持久运行的进程托管（已有 user service 或 tmux 等），并验证断开终端后仍在运行；不安装系统服务或使用 sudo。认证、上下文额度和服务器可用性不是调度器能保证的条件，异常进入有界恢复与 degraded 报告。

## 时钟与状态

- supervisor smoke 完成后，实际研究开始时写入 T0、UTC deadline=T0+86,400s、主机/进程身份和 campaign ID。D0、正式候选实现、调试、训练与报告都计入窗口；任务规划与通用监管器实现不伪计为已运行实验。
- 进程内用 monotonic 时间等待；重启以已保存的 UTC deadline 恢复，不能再给 24h。记录时钟跳变与不可用时间。
- 状态为 `planned → active → consolidating → finalizing → completed/degraded`。任何状态到期都禁止新作业；提前得到好结果不能直接 completed。
- 0–2h 优先 D0/summary/protocol；2–8h 优先 D1 和信号诊断；8–18h 依证据选分支；18–22h 补齐对照、种子和最终考核。时间段是优先级，D0 未过时不强行开始 D1。
- 22h 触发收束：不新增探索分支；只启动预计能在 23h 前完成的确认、评测或导出。23h 进入最终分析，23.5h 前停止 GPU 新工作并整理已有证据；24h 是探索硬截止。报告持续增量写入，避免最后一分钟才生成。
- 截止不依赖 Codex 在线。独立 watchdog 持有 deadline、作业 ownership 和状态账本，到期终止本 campaign 的进程组并写出最终事实摘要。若 Codex 最终分析失败，标记 degraded、保存最近已提交报告和缺失部分，不声称已完成高质量分析。
- 到期清理最多保留 120s 的终止/落盘宽限，不能用于继续训练或新增模型调用；报告记录实际停止时间。该宽限只服务资源回收。

## 事件与 token

| 事件 | 动作 | 模型调用 |
|---|---|---|
| 作业成功退出且产物可解析 | 聚合同波已到事件、补齐 matched pair、更新指标 | 需要选择或解释时调用一次 |
| 作业失败/超时/产物缺失 | 固化 exit code、尾部日志、资源快照、失败指纹 | 首次新故障调用；相同故障按既定规则处理 |
| 完整对照波次结束 | 更新比较、CI、假设状态和下一波 | 一次 |
| 22h/23h/24h 单次计时事件 | 收束/报告/终止 | 22h、23h 可调用；24h 使用已完成报告或事实兜底 |
| 常规 step、tqdm、GPU utilization、heartbeat | 写日志或系统指标 | 不调用 |

普通进程用 subprocess wait/asyncio 退出回调与单次 timer，不靠 LLM sleep 循环。若平台没有文件通知，低成本本地状态检查也不得自动转为模型轮询。合并已到事件，最多一个分析进程；每个事件有稳定 ID、处理水位与决策 ID，崩溃重放不能重复启动 run。

每次只传入当前假设、资源账本、受影响 run 的指标摘要/差异和必要日志尾部。先读 [研究摘要](research/research-digest.md)，按当前分支加载具体论文，禁止每次注入全部历史对话/PDF。保存 CLI JSONL，汇总调用数、input/cached/output tokens、分析耗时与触发原因；不把无模型的等待计作 token。用户未设费用/token 硬上限，记录实际模型与配置，不擅自切模型或增加子代理。

## 决策与失败隔离

Codex 可以修改本任务代码、配置、测试与研究文档并及时 commit；回复结构包含 `event_ids`、`summary`、`hypothesis_updates`、`requested_experiments`、`commit_refs`、`retry_or_stop`、`next_expected_event`。所有实验先登记再运行。

监管器校验结构、事件去重、deadline、GPU 租约、配置与源码 commit、预算及产物位置，依据白名单操作组装 argv；不把模型 JSON 中的任意字符串当 shell 执行。未提交/检查失败的变更不能成为正式 run。Codex 退出但无有效决策算失败，不能标记实验分析成功。

相同失败指纹最多自动重试 2 次，第 3 次触发该分支熔断；新修复必须有原因与检查。临时服务失败可以用普通计时器做有界退避，不反复让 LLM 解释相同日志。把科学负结果、实现错误、资源不足、认证/限流、进程失联分开登记。某分支失败时优先运行独立且已通过 gate 的分支；共用 reference 失效时停止依赖它的全部质量比较。

每个作业记录 PID、process group、启动时间、GPU UUID 和 run ID。只向本 campaign 创建且身份仍匹配的进程发送信号，不按全局进程名杀 Python/Codex，也不清空显卡。stdout/stderr 直接写文件，不能因 pipe 未消费阻塞作业。

## GPU 与不可变实验

用户授权物理 GPU 5、6、7、8、9 独占。服务器启动时核对 index→UUID，禁止无意把容器逻辑编号当物理编号。公共入口只指定一次卡号：`python -m ncls train 5 --config ...`；不再叠加 CUDA_VISIBLE_DEVICES/torchrun。先用短 probe 确认 Torch、Falcor、SlangPy 选同卡。

默认五个单卡槽。先运行三个 source 的 raw 和两个 corrected-summary；空出下一槽补第三个 summary。优先补齐同 source 的两臂，其次确认种子，不能只保留较快/较好的臂。DDP 必须先有吞吐/显存理由与 Linux NCCL smoke，保持 global batch/query/optimizer step 的比较口径；允许不用满五张卡等待必要依赖。

每个 trial 从已提交源码创建独立 detached worktree/checkout。Codex 在控制工作树修改下一版，运行中的 checkout 保持不变；共享只读 assets/reference 源码，构建与可写产物显式隔离。共享路径是否只读、native backend 是否复用兼容构建由启动 probe 确认，不能用符号链接掩盖不同源码与二进制。无需自动删除历史 worktree 或成果。

训练仍走现有公共 engine，使用有限 step 分段运行和正常 checkpoint 完成点作为事件；具体 stop/resume 行为先测。不得假设 SIGINT 自动保存 checkpoint。每段在 deadline 前留出评测与退出裕量，独立 watchdog 兜底。源码 revision 改变时，只有未改变模型/数据/optimizer 身份的受测恢复才可续跑；方法变化 fresh run，不强行加载旧 checkpoint。

## 落盘与报告

监管产物放 `outputs/neural-material-24h-server-research/<campaign-id>/`：campaign 状态、事件/占卡账本、Codex JSONL、决策、日志与产物索引。各训练 run 仍由公共入口写 `outputs/<config-stem>/<run-id>/`；以入口返回的真实路径关联，禁止扫描“最新目录”猜测 run。

根仓库保存本任务 `research/experiment-ledger.md`、`research/decisions.md`、`research/final-report.md` 及必要配置/实现。台账记录 source locator、texture-set、query/split identity、源码/config、seed、global batch/steps、unique reference queries、network evaluations、训练/评测/编译时间、GPU-hours、checkpoint 路径和失败状态；不提交 checkpoint、PDF、raw batch 或大日志。

关键修复/候选实现通过相关检查后立即 scoped commit；每一完整分析波次提交精简结论。提交频率由可审阅工作触发，禁止固定每分钟空提交。最终报告明确支持/反驳/未判定、canonical 指标与 CI、成本测量范围、负结果、适用 source/seed、未完成项和下一步。Linux 可做数值/Slang parity；Windows viewer 属阶段后续部署证据，不成为服务器 24h 必须远程实现的隐藏 gate。
