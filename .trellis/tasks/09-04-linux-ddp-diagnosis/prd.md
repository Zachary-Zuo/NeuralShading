# 诊断 Linux DDP 利用率与超时

## Goal

对近期训练架构重构后的 Linux/NCCL 多卡路径做证据驱动审计，解释为何多卡吞吐扩展不足、为何容易表现为 collective timeout，并形成可执行的整改与目标机验证方案。诊断必须区分当前代码可直接确认的问题、历史 Linux 产物支持的观察，以及只有真实 Linux 多卡 trace 才能确认的假设。

## Background

- 用户观察到先前 Linux DDP 没有充分提效，并且容易 timeout，怀疑存在设计与实现问题。
- 当前正式多卡路径是单机 Linux `torchrun` + NCCL；每个 rank 独占一张物理 GPU，并把 Torch/SlangPy 映射为进程内 `cuda:0`，Falcor 仍按物理序号选 Vulkan adapter。
- 当前会话运行在完整 Windows 开发环境：RTX 4090、`neural-shading` 环境和 Windows Falcor 构建均存在；可以做代码、unit 与本机 GPU 检查，但不能把它们当作 Linux/Vulkan/NCCL 运行证据。
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

## Out of Scope

- 已批准批次只实施阶段A+B；不宣称已经在 Linux 目标机复现timeout、完成NCCL修复验收或取得新的scaling结论。
- 不修改 `external/Falcor`，不引入独立 reference GPU/P2P service，不改变 method/source/reference 公共语义。
- 不以 Windows 单卡结果替代 Linux 多卡 scaling、Vulkan barrier 或 NCCL 证据。

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
- [ ] AC10（目标机硬门，来源：Linux/NCCL部署约束）：两卡NCCL integration、跨两phase smoke、checkpoint/failure/teardown与1/2/3/4卡matched scaling在原生Linux目标机通过并回写证据。

## Approved Implementation Scope

- 用户于 2026-09-04 审阅诊断、设计与实施计划后，明确批准继续实施阶段 A+B：真实 DDP reducer，以及指标、stage trace、checkpoint/finalization 对称化。
- data/reference scheduler 优化仍留在后续阶段，本批不实施。
