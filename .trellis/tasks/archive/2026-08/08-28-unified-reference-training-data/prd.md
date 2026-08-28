# 统一材质 Reference 训练数据回路

## Goal

删除旧 HDF5/offline replay 与按材质族分裂的 live batch producer，让所有 source material 通过同一套 GPU online scattering query 回路产生训练输入。统一回路以 backend-specific `ScatteringState` 为唯一求值边界，完整提供 `prepare/evaluate/sample/pdf`；NVIDIA evaluator 直接拟合 `evaluate().f`，sampler 按论文语义从 learned proposal 取样并以 `luminance(f)·|cosθi|` 构造目标密度，不再用 dummy target、`f·cos` evaluator adapter 或另一套 offline GT 语义。

## Background

- NVIDIA《Real-Time Neural Appearance Models》的 evaluator batch 在线采样位置、LOD、`wo/wi`，直接求 reference `f`；不先导出固定 HDF5。论文每步另取独立 sampler batch。
- NVIDIA sampler 从当前 learned proposal 取样并计算该 proposal 的 PDF，以当前 learned BRDF response 构造 forward-KL score loss；source reference 的 `sample/pdf` 不是该 sampler objective 的监督数据。当前实现见 `src/ncls/learning/methods/nvidia.py:489-526`，论文证据见本地作者 paper §5 的 training 段落。
- NVIDIA 作者论文把 evaluation decoder 的输出定义为 BRDF value；项目公共 ABI 也规定 `evaluate()` 返回不含 cosine 的线性 RGB `f`。当前 Python core 拟合 `f·|cosθi|`，runtime 再除以 cosine，是旧 HDF5 response measure 遗留的 adapter，本任务必须删除。
- 公共 Slang ABI 已在 `shaders/ncls/contracts/scattering_backend.slang:8-29` 要求每个 `INclsScatteringState` 实现 `evaluate/sample/pdf`，但 data 层仍有 `LiveReferenceBatchSource`、`MaterialXLiveReferenceBatchSource`、`MdlLiveReferenceBatchSource` 与 `OfflineBatchSource`，CLI 也按 source option shape 分支。
- `TrainingBatch@1` 强制每条 route 都带 `target/wi/reference_pdf/sample_count`；sampler route 因而制造零 `target`，与“该 route 只提供 conditioning + `sample_u`，method 自己执行 sample/pdf/evaluate”的真实语义不一致。
- `data/reference-responses/` 下旧 LayerStack/P1 HDF5 属于被替代的离线训练架构。用户明确要求不保留向后兼容，本任务也不新增 recorded-batch 能力；如以后确有固定诊断样本需求，另立任务从 generic online 回路设计新 schema。
- LayerStack source family、原生参数、reference 与统一 scattering backend 继续保留；只删除 LayerStack 专用 data/training producer、配置和 HDF5 路径。

## Requirements

- R1：所有正式 source family 都编译/绑定到同一个完整 scattering backend 合同；数据、训练、viewer 与 package 不得各自维护另一份 reference math。
- R2：generic online producer 只按 route 生成 context/query/RNG，并通过统一 backend dispatcher 执行所需 scattering operation；不得按 LayerStack、MaterialX、MDL、OpenPBR、MERL 增加 producer 类或 CLI 分支。
- R3：evaluator route 调 source `prepare/evaluate`，以线性 RGB `f` 作为唯一 reference target；空间过滤、方向 mollification 或 stochastic-reference averaging 可以聚合多个相同合同的 evaluate query，但不能改写 response measure 或形成另一种 GT。
- R4：NVIDIA sampler route 只生成 source conditioning 与独立 `sample_u`；method backend 对当前 learned proposal 执行 `sample/pdf`，再用 detached learned evaluator `f` 显式乘 `|cosθi|` 后计算论文 forward-KL。不得调用 source sampler 冒充 NVIDIA sampler teacher，也不得制造 dummy reference target。
- R5：source backend 的 `sample/pdf` 对所有材质族仍为强制能力，用于 reference PT、sample→pdf/weight 恒等式与 proposal 质量审计；`sample()` 返回的 direction/event/PDF/weight tuple 不可拆分或重建。
- R6：Training batch schema 必须显式区分 evaluator batch 与 method-sampler conditioning batch；每种 route 只携带有语义的 tensor，不用零值占位满足旧 schema。source `sample/pdf` 属于统一 dispatcher/query API 与验证路径，不为当前 NVIDIA 训练虚构第三条 route。
- R7：删除 `OfflineBatchSource`、旧 collector/corpus/store/HDF5 schema、旧 offline 配置、旧 CLI 与测试入口，不保留 reader、migration、compatibility shim 或新的 recorded-batch sink。
- R8：实施开始前先向用户报告 `data/reference-responses/` 的精确只读清单；由用户批量删除其中全部过时数据，代码变更不得隐式删除用户数据，也不得以旧文件仍在磁盘为理由保留 reader。
- R9：删除 source-specific data provider 的 capability 分歧；正式注册的每个 source 必须在构造/注册时证明 `prepare/evaluate/sample/pdf` 完整，否则 fail closed。

## Acceptance Criteria

- [ ] **需求交付｜来源：用户本轮决策与项目根合同｜映射：R1、R2、R9**：同一个 backend dispatcher 无需修改通用实现即可执行所有正式 source family；同一个 `TrainingRunner` 与 generic online producer 可切换 method 已声明支持的 source，未声明的 adaptation 必须 fail closed。LayerStack 只保留 source/reference/backend，不再拥有专用 data/training 路径。
- [ ] **理论 / 语义正确性｜来源：NVIDIA 作者论文 §5 与 `ncls.scattering-backend@1`｜映射：R3、R4、R6**：evaluator route 的 GPU `target_f` 由 source backend `evaluate().f` 直接产生且无 host readback/HDF5 中转；NVIDIA evaluator 不再预测 `f·cos`，sampler route 无 dummy target，并只通过 method 的 learned `sample/pdf/evaluate` 形成 loss。
- [ ] **理论 / 数值实现正确性｜来源：公共 scattering ABI、路径采样恒等式与 response-measure 合同｜映射：R1、R5、R9**：每个 source family 都通过统一 `evaluate`、sample tuple、independent `pdf(wi)`、finite/event/measure 测试；project-owned 确定性连续事件验证 `weight=f·|cos|/pdf`，stochastic/native tuple 分别按独立 oracle 或预冻结统计协议验证，不要求用舍入后的 `wi` 重建 native tuple。无 family-specific fallback。容差须在正式验证前按浮点类型、独立 oracle 或隔离 calibration 写入 design/test，失败视为 implementation defect。
- [ ] **需求交付｜来源：用户明确要求不保留旧 offline/HDF5 兼容｜映射：R7**：仓库中不存在 `OfflineBatchSource`、旧 corpus/HDF5 reader/writer/schema、旧 offline 配置、按 source 分支的 live producer或相关用户入口；不存在 recorded/offline data sink、reader 或 CLI。
- [ ] **需求交付｜来源：用户明确指定磁盘数据由其自行删除｜映射：R8**：`task.py start` 前向用户交付 `data/reference-responses/` 精确只读清单并等待其确认；用户批量删除后，验证目录中无旧 HDF5。此项只决定旧数据清理是否完成，不授权代码执行删除。
- [ ] **理论 / 语义正确性｜来源：NVIDIA 作者论文 formal recipe 与既有 correspondence 合同｜映射：R3、R4**：formal 配置保持两条独立 65k route、300k lifecycle、evaluator log-L1 与 sampler forward-KL。这里的数值只约束 NVIDIA formal correspondence，`source=作者论文`、`scope=nvidia-rta2024 formal recipe`、`why_hard=方法身份保真`、`failure_action=回到 planning 修正 correspondence，不通过缩小 recipe 冒充 formal`。
- [ ] **数值实现正确性｜来源：公共 ABI、作者论文与项目质量门｜映射：R1—R6、R9**：固定 seed focused test 证明 source `evaluate().f → target_f → learned evaluate().f` 的 measure 一致；sampler test 证明 KL target 显式包含 cosine，且不依赖 reference target。旧 `f·cos` tensor 不作为 parity oracle。相关 unit、GPU、integration、source-module query 与 learned-package query parity 通过，锁定 upstream 保持 clean。

## Out of Scope

- 不把 viewer 的 environment MIS、primary path pool 或场景 PT estimator 引入 reference response 采集。
- 不用 clamp、去异常值或删合法窄峰替代 reference/scattering 合同修复。
- 不修改 NVIDIA 论文冻结的 evaluator/sampler loss、网络规模或训练 lifecycle；若证据证明现有复现偏离论文，先回到 planning 更新 correspondence 决策。
- 不设计或实现新的 recorded-batch schema、sink、reader 或固定样本训练模式。
