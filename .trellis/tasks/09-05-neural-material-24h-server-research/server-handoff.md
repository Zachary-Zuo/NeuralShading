# 服务器执行交接

## 当前状态

本任务已获得约 24h 内自主试验、迭代、分析、及时本地 commit 的授权；可独占物理 GPU 5、6、7、8、9。当前是提交到仓库的规划，**没有连接服务器、没有运行监管器、T0 尚未开始**。服务器地址/仓库路径由服务器会话现场使用，不在本任务猜测。

前继实现和归档见 [closure](../archive/2026-09/09-05-neural-material-spatial-optimization-research/closure.md)：实现基线 `e044156d9031f186551e05ff19436fbabfbd2b59`，归档提交 `2594d7a97e054dbba2b5eb07e63ea4107f704dfc`。同步包含本任务的后续提交后，从该版本创建服务器研究分支；不复制旧 checkpoint 作为新方法训练起点。

Codex CLI 非交互能力已经查过本机 help 和 [官方文档](https://developers.openai.com/codex/noninteractive/)，服务器须验证自身版本/认证。使用环境中已有的模型与配置，事件决策可显式恢复指定 session ID；禁止 `resume --last` 意外串入其它 campaign。上下文膨胀时使用持久摘要开启新决策会话，保留关联 ID。

## 给服务器 Codex 的启动指令

下面内容可作为服务器 Codex 会话的用户指令。它要求先实现和验证尚缺的监管器，再启动真实 24h；不能只在终端开训练便称已监管。

```text
继续 .trellis/tasks/09-05-neural-material-24h-server-research。
按 AGENTS.md 和该任务 prd/design/experiments/implement/server-handoff 执行；
先读 research/research-digest.md，再按当前候选读取对应深入论文报告。
我已授权独占物理 GPU 5、6、7、8、9，研究窗口约 24h。
请先实现并验证缺少的事件监管器和独立截止控制，随后实际启动 campaign，
保存唯一 T0/deadline。研究启动后的 D0、summary 实现、修复、训练、分析都计入 24h。
普通进程等待作业/截止事件，仅在需要分析或决策时调用 Codex，不用 LLM 高频轮询。
按证据自主实现或重排任务表内实验，也可以预登记新假设实验；每组运行前冻结，
在同一 24h/五卡边界内选择预算。每个合格实现与重要分析及时 scoped commit，不 push。
沿用公共 Method/engine/CLI、GPU online reference 和 native 材质语义；不持久化训练 batch。
优先 D0、真实 raw/summary matched D1、分段信号诊断，然后根据结果安排后续。
final test 留到收束，22h 起补齐对照和确认，24h 到期停止探索并提交成型结论，
负结果或系统故障如实报告，不因没有收益自动延期，也不因提早有收益提前结束。
单个 Codex 决策进程，GPU 训练可以并行；不要增加子代理。
启动成功后先报告实际 T0/deadline、托管方式、物理卡 UUID、run/事件账本位置，
之后保持独立监管，到期保存最终报告与本地提交。不要把文档计划当作已运行证据。
```

## 现场检查与实际启动命令

以下只有已有命令，不包含尚未实现的假启动入口。服务器执行 `task.py start` 后按 S0 开发；S0 完成时必须把真实监管器的 CLI、恢复、单次状态查询与停止命令补在本节，并用短 smoke 验证。

```bash
git status --short
codex --version
codex exec --help
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
conda run -n neural-shading python .trellis/scripts/task.py start .trellis/tasks/09-05-neural-material-24h-server-research
conda run -n neural-shading python -m ncls reference probe --device 5
```

先按 spec 确认 Linux reference 能力再执行最后的 probe；另四卡也须确认物理映射/可用性。现有依赖或资产缺失要给出具体诊断，按项目部署边界处理，不安装驱动、sudo 或自动租算力。日常用户状态请求读取 ledger/报告即可，不需要后台让 LLM 连续查 GPU。

## 授权范围与到期通知

`research-execution.md` 的默认连续执行不自动授权扩展实验预算；本任务的用户请求明确包含“根据过程中发现的问题实现新实验或安排已有实验”，因此允许在已授权 24h、GPU 5–9 和既有 Metal 源族内自主重新登记小型实验、确认 seed 与预算。每个已冻结 run 仍不可事后改变规则，超出上述窗口/资源/源族须另行取得授权。

最终报告和 commit 保存在服务器仓库，可供下一次连接直接读取。Codex CLI 不自动保证把消息送回本桌面聊天；若服务器已有获授权的交付渠道可沿用，否则以本地最终报告和会话末条输出交付，不擅自发送邮件/Slack。认证/配额不足时普通监管器仍应落实截止和事实报告，不能承诺模型无限可用。
