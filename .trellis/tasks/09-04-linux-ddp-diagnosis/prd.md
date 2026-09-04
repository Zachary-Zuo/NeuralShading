# 诊断 Linux DDP 利用率与超时

## Goal

对近期训练架构重构后的 Linux/NCCL 多卡路径做证据驱动审计与整改：既让 phase-local DDP reducer、控制面和 checkpoint 生命周期正确，也把已有但尚未接入生产的 host pipeline、GPU residency、reference scheduler 与 ready ring 收敛成唯一的通用 `OnlineDataSession`。最终在原生 Linux 的 GPU5–9 上证明多卡不会因数据长尾或不对称退出形成无来源 timeout，并能相对单卡产生真实吞吐收益；模型、method、数据需求与训练引擎不感知 Windows/Linux，平台差异只由 launcher、device/reference backend 与 capability 提供。

## Background

- 用户观察到先前 Linux DDP 没有充分提效，并且容易 timeout，怀疑存在设计与实现问题。
- 当前正式多卡路径是单机 Linux `torchrun` + NCCL；每个 rank 独占一张物理 GPU，并把 Torch/SlangPy 映射为进程内 `cuda:0`，Falcor 仍按物理序号选 Vulkan adapter。
- 当前会话已迁到原生 Linux reference 环境：10 张 RTX A6000、`neural-shading` 环境与锁定的 Linux Falcor Python 构建均存在；用户授权本任务使用物理 GPU5–9，GPU0 的外部负载不在本任务范围。
- `HostPipeline`、`GpuResidencyManager`、`ReferenceScheduler` 与 `OnlineDataSession` 协议已经存在，Metal asset miss 也使用部分 host/residency primitive；但生产 CLI 仍固定构造 `SynchronousOnlineDataSession`，engine 在主线程同步填充 step queue，`ready_batches`/`reference_batch_steps` 尚未形成统一的生产 data plane。
- 与本任务无关的未跟踪 scratch、PDF、图片、字体和其他活动任务目录已经存在，必须保持不动。

## Requirements

- R1：审计 launcher、rank/device 映射、process-group 生命周期、模型/梯度同步、online data/reference 调度、validation、checkpoint、hook 与退出清理的完整控制流。
- R2：使用仓库内已有 Linux 单卡与 3 卡 DDP 产物做 matched 对比，量化吞吐、并行效率以及 prepare/forward/backward/optimizer+sync 分解；指出指标定义错误或证据局限。
- R3：定位 timeout 的具体等待边界与不对称失败路径，覆盖 rank 间数据长尾、惰性 reference group 构建、collective 顺序、rank-0 I/O、checkpoint state gather 和 worker teardown。
- R4：核对 PyTorch 2.11 官方 DDP/NCCL 合同，判断当前实现是否真正使用 `DistributedDataParallel`，并设计兼容 phase graph、动态 parameter group、unused parameter 与 checkpoint resume 的整改结构。
- R5：为 Linux 目标机定义可复现的 before/after 验证矩阵，包括 rank-local stage trace、DDP reducer 统计、collective flight recorder、故障注入、1/2/3/4 卡 scaling 与正确性对照。
- R6：结论以严重级别和证据强度排序；修复计划先解决错误同步与 timeout 诊断，再处理 reference/data overlap，不用单纯增大 timeout 掩盖长尾或 collective desync。
- R7：在用户批准后实施阶段A+B：由phase-local PyTorch DDP reducer拥有objective forward/backward，删除逐parameter同步；NCCL data group与Gloo control group分责。
- R8：对metrics、逐rank stage、checkpoint/final artifact与teardown建立固定顺序协议；非rank 0不得构造完整model/optimizer snapshot，rank-0失败必须显式传播。
- R9：以一个 pipeline-backed `OnlineDataSession` 替换生产固定同步入口；`reference_batch_steps=1` 只是同一实现的 correctness 配置，不保留第二套兼容 runner/session。session 统一拥有 step request、host-only prefetch、GPU residency、group-homogeneous reference dispatch、ready ring、lease、drain 与 checkpoint cursor。
- R10：Vulkan `global-sync` 能力下不虚构同卡 Falcor/Torch 异步重叠；通过提前提交 host work、group boundary 预热、同 group packed query、预分配/复用和有界 ready ring 减少 consumer starvation 与 barrier。只有 backend 明确提供 stream-fence capability 时才允许真实 inflight。
- R11：`TrainingEngine`、模型、method facet、source adapter 与数据语义不得读取 OS、物理 GPU 或 backend 名称；Windows/Linux 差异只收敛在 launcher、device/reference backend 与 capability，并向上提供相同的 plan/session/batch/checkpoint 合同。
- R12：在 GPU5–9 上执行 1/2/3/4/5 卡 weak scaling；强 scaling 只对当前 route batch 可精确等分且不改变 loss 权重的 1/2/4 卡执行。正式性能 run 不启用 debug wrapper，diagnostic/fault run 与性能结果分开。
- R13：用户明确要求“DDP 真正提效”。目标机 hard gate 的适用范围固定为 RTX A6000 GPU5–9、同一 source/query/model/precision 与 post-warmup steady window；至少 2 卡和 5 卡的 global work/s 必须高于单卡 matched baseline，失败时继续定位 pipeline/communication/resource defect，不以增大 timeout 或改训练语义放行。GPU activity、并行效率和强 scaling 数值仍作为 observed report，不另编脱离机器的阈值。

## Out of Scope

- 不修改 `external/Falcor`，不引入独立 reference GPU/P2P service，不改变 method/source/reference 公共语义。
- 不以 Windows 单卡结果替代 Linux 多卡 scaling、Vulkan barrier 或 NCCL 证据。
- 不承诺 Vulkan `global-sync` backend 上 reference 与 model 的同卡时间线重叠，也不把瞬时 100% GPU utilization 写成普适硬门。
- 不引入按 OS 分叉的 runner、data session、model 或数据配置，不用 compatibility alias 保留被替换的生产入口。

## Acceptance Criteria

- [x] AC1（需求交付，来源：用户问题与当前源码）：形成带 `file:line` 锚点的 DDP 控制流与风险清单，清楚标记“已确认 / 历史产物支持 / 待 Linux 验证”。
- [x] AC2（观察性报告，来源：仓库内已有 Linux 产物；数值不是 hard gate）：给出同配置单卡与 3 卡历史产物的真实 work throughput、speedup、并行效率和关键 stage 均值，并核对当前 throughput 字段公式。
- [x] AC3（需求交付，来源：用户报告与控制流推导）：解释至少一种高概率 timeout 因果链，明确哪一 rank 在何种边界等待、为何 300 秒阈值会被触发，以及现有日志为何不足以定位 culprit rank。
- [x] AC4（设计交付，来源：当前架构合同与 PyTorch 2.11 官方接口）：形成分阶段整改设计，明确真实 PyTorch DDP reducer、phase 切换、batched metrics、checkpoint 协调、fail-fast/flight recorder 与 data/reference scheduler 的责任边界。
- [x] AC5（验证方案，来源：目标 Linux/NCCL 运行约束）：形成 Linux 验证命令与产物清单，能够区分通信瓶颈、reference/data 长尾、Falcor Vulkan 全局同步和 rank-0 I/O。
- [x] AC6（用户决策门，来源：本轮只读诊断边界）：最终向用户给出简明结论、优先级和下一步决策点；未经后续明确批准不进入产品代码实现。
- [x] AC7（实现与数值正确性，来源：用户批准、PyTorch DDP语义）：production engine不再包含逐parameter gradient collective；两rank Gloo测试证明DDP平均梯度与一步optimizer结果一致，并覆盖phase-local active parameter重构。
- [x] AC8（控制面，来源：设计合同）：metric一次pack聚合；checkpoint只gather小型rank state、仅rank 0编码/写入并广播状态；run lifecycle、checkpoint rank state与teardown错误可跨rank传播。
- [x] AC9（本机回归，来源：完整Windows环境）：全部unit测试通过，16-step Windows Metal共享路径完成且checkpoint/summary有效；该结果只证明公共单卡路径未回归。
- [x] AC10（目标机硬门，来源：Linux/NCCL部署约束）：两卡NCCL integration、跨两phase smoke、checkpoint/resume/failure/teardown在原生Linux目标机通过；正常运行无 timeout，故障注入能给出 stage/request/rank 并在有界时间内全rank退出。
- [x] AC11（数据面正确性，来源：用户明确要求与通用online pipeline合同）：生产 CLI 只构造统一 pipeline-backed session；`num_workers`、`host_prefetch`、`ready_batches`、`reference_batch_steps`、`reference_inflight` 与 residency budget 均有真实生产调用点。单步基线与 packed 模式逐 logical step 的 request/RNG/top-up/target、release、drain 和 resume identity 一致。
- [x] AC12（跨平台边界，来源：用户明确要求与统一pipeline合同）：静态与单元测试证明 engine/model/method/data contract 不按 Windows/Linux 或 backend key 分支；Linux launcher/Falcor/NCCL 与 Windows launcher/D3D12 差异停留在 backend/capability 层，两个平台消费同一 session/batch/checkpoint API。
- [x] AC13（性能硬门，来源：用户于2026-09-04明确要求DDP真正提效；scope/why/failure action见R13）：GPU5–9 的冻结配置完成1/2/3/4/5卡weak scaling，2卡与5卡global work/s均高于单卡；post-warmup trace无可避免之外的consumer production wait，并报告rank skew、GPU activity未单独采样这一证据边界、显存、bucket与barrier。1/2/4卡strong scaling独立报告，不与weak scaling混用。

## Approved Implementation Scope

- 用户于 2026-09-04 审阅诊断、设计与实施计划后，明确批准继续实施阶段 A+B：真实 DDP reducer，以及指标、stage trace、checkpoint/finalization 对称化。
- 用户随后明确把良好数据管道、DDP多卡实际提效、无timeout、避免平台兼容层以及Windows/Linux底层能力内聚纳入同一任务，并于2026-09-04批准扩展后的Phase 4/5实施范围。
