# 服务器 24 小时神经材质自主研究

## 目标

在服务器正式启动监管后约 24 小时内，使用独占 GPU 5–9，由 Codex 根据事件自主实现、选择、运行、分析和提交实验，形成可追溯的神经材质研究结论。到时按证据收尾；任务不以达到某个质量分数、跑完全部候选或不断改到有收益为完成条件。

用户已授权创建本任务、准备既有研究与候选，并明确可独占 GPU 5、6、7、8、9。当前交付是服务器执行用的任务包；尚未连接服务器、启动实验或开始 24 小时计时。

## 基线与已知事实

- 前继任务为 [原生多 UV 实现阶段](../archive/2026-09/09-05-neural-material-spatial-optimization-research/closure.md)。已完成 raw 多 UV encoder、GPU online 资源、流式 cook、176 B prepared state、新 Slang ABI 和 Windows step 2 smoke；没有 matched 质量结论。
- `metal_spatial_hybrid_v1` 的 prepare/evaluate 为 7,664/11,392 dense MAC，最多 9 UV 组；每组保留自身 affine/nonrepeat/address 语义。实际青铜为 12 reads，上限 54，实际 latent 约 181.33 MiB。
- `metal_spatial_summary_control_v1` 当前明确报错，尚未实现。必须先做真正的修正 summary，不能用 raw CNN 的别名作为对照。
- 先前 GPU 5–9 的记录为 RTX A6000；型号、UUID、显存与空闲状态均须在服务器重新确认。Windows smoke 不证明当前 Linux reference/NCCL 正常。
- 历史论文研究和旧 Metal 实验保留为机制、负结果与错误线索；旧入口、旧 checkpoint、旧 GT 语义和旧成本不能作为当前 matched baseline。证据入口见 [研究摘要](research/research-digest.md)。

## 需求

- **R1 时间边界**：监管器记录一次 T0 与 `deadline=T0+24h`；计时包括 D0、实现、修复、训练、分析和最终报告。重启不得重置 deadline；有效 wall time、停机和 GPU-hours 均记录。临近截止提前收束，24h 终止继续探索并产出最终报告，不自动延期。
- **R2 事件驱动**：普通进程负责等待训练退出、失败、预定分析里程碑与截止事件；只有这些事件才唤醒一个 Codex 决策进程。禁止让 LLM 每几秒/分钟查看进度。训练 tqdm/日志可以持续写磁盘而不触发模型调用。
- **R3 自主权限**：窗口内可根据证据修复本任务代码、实现候选、增减/重排预列实验、创建有明确理由的新实验、进行同预算确认和本地 commit；不逐实验等待用户确认。所有变体在运行前登记假设、唯一变化、比较对象、source/query/split、预算和停止规则。
- **R4 资源边界**：只使用物理 GPU 5–9，最大并发占卡数为 5；默认单卡独立实验，DDP 为有证据需求的独立调度选择。同一比较不能因卡数不同改变 global batch 却仍称 iso-work。禁止自动租新算力、占用其它卡、改驱动/系统环境、上游源码或删除其它任务产物。
- **R5 科学比较**：先 D0，后 raw/summary matched D1，再依证据进入信号读取、bounded code/refinement、correction/loss、filtering、坐标/activation、泛化及成本轴。保留 canonical 线性 f 指标和块级 CI，不能只用自身 training loss 比较不同 loss。负结果、实现失败、资源问题和未执行项都可成为结论。
- **R6 防止选择泄漏**：冻结 train、用于自适应选择的 validation 和最后一次 blind test 的完整 raw RF；final test 不给日常决策使用。若 test 结果用于再设计，该集合已成为探索数据；须记录污染并采用新的最终考核，不能继续宣称盲测。
- **R7 语义与架构**：原生 source/reference、参数编辑、不同 UV 分组、固定绝对解码、GPU online GT、真实量化读取和随机访问有界 runtime 保持。主路径为 encoder-only；free-code/refinement 只作明确对照。所有模型训练仍走公共 Method/engine/CLI，不建立第二套训练系统，不持久化 batch/GT/replay tensor。
- **R8 提交与恢复**：每个可审阅且通过对应检查的修复/实验实现及时 commit；每波分析独立提交中文报告。实验固定到已提交源码，不在运行中修改它的工作树。outputs/log/checkpoint/Codex token 明细进入 ignored run 目录；根仓库仅保留源码、配置、研究决策与证据索引。不 push。
- **R9 最终交付**：24h 后给出成型报告，至少明确输入/表示/优化中哪些解释得到支持、最有用与无收益的实验、证据不确定性、质量—时间—内存观察和建议下一步；每条重要结论绑定具体 matched run/代码/指标/CI/适用范围。没有质量收益时交付负面结论，不捏造成功。

## 验收

| 编号 | 可观察结果 | 类型与来源 |
|---|---|---|
| A1 | T0/deadline 持久化且重启不延长；到期不再启动实验，报告和资源回收有时间证据 | 用户时间要求与运行正确性 |
| A2 | 正常等待区间无重复 LLM 调用；每次调用对应事件 ID，记录调用数/token/事件合并 | 用户节省 token、事件监管要求 |
| A3 | 全部作业只使用 GPU 5–9，源码/配置/运行路径和占卡账本可追溯 | 用户资源授权与实验可复现性 |
| A4 | 既有论文与历史实验证据转为有优先级、触发条件、对照和边界的实验表 | 用户预先安排研究的要求 |
| A5 | 运行前登记、失败分类、检查后 commit、在线 GT 和恢复合同得到执行 | 用户自主迭代/及时提交及项目工程合同 |
| A6 | 截止时提交 final report、实验台账、有效/无效/未完成项、代表 checkpoint/config 和后续建议 | 用户 24h 成型结论要求 |

**时间到并不自动证明研究质量。** 正常完成是时间窗口已用尽或按截止计划收束，且证据报告交付；外部故障令计算无法继续时仍按原 deadline 交付故障/可行性结论，状态明确为 degraded，不伪称跑满 24h。不能因短期得到好结果或候选表跑完而提前停止：余时用于确认、归因、成本或新证据驱动实验。

## 范围与执行交接

首轮 source 为 Tungsten、划痕青铜、开裂涂漆钢；泛化分支可在已支持 Metal schema 内按 texture-set 分组选取小型 cohort。场景级 GI、新源材质族、UE/多灯/PT 质量项目、大规模论文原配方复现、外部发布均不自动进入本任务。

详细 [运行设计](design.md)、[实验候选表](experiments.md)、[实施顺序](implement.md) 与 [服务器启动交接](server-handoff.md) 共同定义研究窗口。具体服务器路径、账号和 CLI 版本由目标机现场记录；不把本机认证文件复制到任务或仓库。
