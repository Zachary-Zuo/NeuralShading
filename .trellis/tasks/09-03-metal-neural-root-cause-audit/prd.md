# Metal 神经训练与部署根因审计

## Goal

找出并消除 Metal 神经方法从 GPU online reference query、训练、checkpoint、MethodBundle 编译到 viewer `prepare/evaluate` 执行链路中的根本性正确性与性能问题，使训练结果具有可证实的学习效果、部署后不再退化为白模，并使长程训练成本保持在可接受且可解释的范围内。同时对同一链路做系统性风险审计，发现尚未显性暴露但会让实验结论或部署结果失真的问题。本任务不止提交审计报告，还必须实现根本修复并完成分层回归验证。

## Background

- 用户报告：先前训练 20,000 step 的 Metal `.pt` 权重在 viewer 中表现为完全白模，视觉上等同于没有学习。
- 用户报告：先前复现的 NVIDIA 模型至少能够训练，因此当前 Metal 路径很可能存在 Metal 特有或新统一链路引入的根本问题。
- 用户报告：Linux 训练达到 20,000 step 后，每个 step 约需 4 分钟，无法接受。
- 规划阶段的初始证据见 `research/initial-evidence.md`；单次 run 的数值和后续 profile 继续写入 `research/` 或 `artifacts/`，不作为事后改写需求的依据。

## Confirmed Facts

- C1：20k checkpoint 只完成 `codec-warmup`；散射相关 `typed_compiler/prepared/angular/analytic/hybrid/proposal` tensor 从 5k 到 20k 逐位不变，gradient/update coverage 全为 false。它不是已经训练 20k 的 evaluator。
- C2：Metal viewer 准备脚本把这个 checkpoint 设为默认值，并在未检查 runtime component 训练 coverage 的情况下允许编译 package；catalog 又把 resume cursor `joint-appearance@0` 显示成 `joint-appearance`。
- C3：该 checkpoint 与当前 runtime 的 descriptor 和 implementation identity 均不同；现行 `state-schema-compatible-preview` 只比较 tensor schema，仍会用当前代码解释旧权重。
- C4：既有 headless capture 的 reference 左侧有金属外观，neural 右侧是近白无信息材质，但两个 slot 都被标为 `ready`；finite/ready 与 Python↔Slang parity 都不能证明模型学到了 reference。
- C5：20k 正好启用 reference evaluator route。producer 每步轮转 178 个 execution group，而 backend 只 resident 8 个，steady state 形成近 100% LRU miss/evict/recreate；group 构造还同时创建训练不使用的 `sample/pdf` pass。
- C6：现有 metrics 每 10 步只保留最后一个 step 的分项时间，并报告 run-global 累计吞吐，无法定位中间 cold materialization 尖峰。现有 phase-1 记录出现约 45.06 与 24.76 秒/step 的 10-step interval，分项 row 却只显示约 2 秒 preparation。
- C7：20,010–21,010 的局部窗口出现部分 loss 下降，也有 linear-energy 恶化；当前证据说明 appearance 路径开始优化，但不足以证明稳定收敛、泛化或部署正确。
- C8：当前 0–20k 不是端到端的 coarse-to-fine 优化。`codec-warmup` 只接收 `asset-tile`，只优化 codec 六组参数及纹理/结构重建损失；reference evaluator route、散射 loss 与所有 evaluator/proposal 参数均从 20,001 step 才启用。viewer 不展示 codec 重建，因此 20k 对材质散射外观等价于初始化 evaluator。
- C9：项目中的 NVIDIA reproduction 并非 codec-only 分阶段训练。其 `bootstrap` 从第 1 步就以 online reference 同时训练 source-parameter encoder、evaluator 和 sampler；`materialize-assets` 只把 encoder 结果烘成 latent texture，随后 `finetune` 继续训练 asset latent、evaluator 和 sampler。NVIDIA recipe 中的 20k 是方向 mollification 区间，不是无 appearance 监督的 warmup。

## User Decisions

- D1：交付边界包含全面根因审计、根本优化、修复实现和回归验证，不以只给报告或临时绕过结束任务。
- D2：Windows 负责证明共享训练路径能够正常、正确且高效地运行；120,000-step 完整训练仍在 Linux 目标机执行。
- D3：Windows 与 Linux 对上层保持同一 config/producer/runner/checkpoint/method/package 语义。修复必须位于共享实现或平台 capability 内部的合法边界，不得增加 Windows-only 兼容训练路径，也不得以 Windows 特例替代 Linux 修复。
- D4：允许严格分级的显式 `diagnostic preview`。正式 viewer/package 只接受完整训练、readiness 与 identity 均成立的 checkpoint；中间 checkpoint 只能暴露已完成训练并通过 coverage/audit 的 capability，必须标记为 diagnostic，不能冒充正式 `ready` 结果。
- D5：Metal 训练必须改为面向最终 runtime 目标的端到端训练；空间/纹理表示不能先用 20k codec-only 重建目标独立定型，而应从早期 step 就由 online reference appearance loss 共同约束 compiler、prepare 与 evaluator。阶段用于训练难度、materialization 或精度逐步变化，不得让目标 evaluator 在长阶段内保持初始化状态。
- D6：proposal sampler 从第 1 步与 appearance path 共同训练，使用 detached evaluator 构造目标，并按冻结、确定且可恢复的 schedule 渐进增强 proposal loss；不再把独立 15k `proposal-fit` 作为 sampler 首次接受训练的阶段。末期 QAT 仍可作为部署精度 refinement。
- D7：没有预设的绝对秒/step、120k 总时长或显存 hard gate。性能优化以正确性和冻结语义为前提，按 matched profile 中的主导热点尽可能优化；结构性退化、steady-state 重建、无界资源增长和随 step 恶化的吞吐是 hard failure，绝对 quality/time/memory 为 observed result。

## Requirements

- R1：建立 phase/component-aware checkpoint readiness，明确区分 resume cursor、已完成 phase、已训练 parameter group 和可部署 capability；不可用 checkpoint 必须在 package/catalog 生成前 fail closed。
- R2：以固定 seed/source/query 的最小实验确认 Metal evaluator 是否实际学习，逐层检查 target、loss、梯度、optimizer update、validation 与 checkpoint round-trip；训练信号与 holdout 行为分开报告。
- R3：端到端追踪同一严格兼容 checkpoint 的 eager Python、部署量化 Python、Slang package 与 viewer 输出，分别证明实现 parity 和 reference-neural 学习效果，定位白模首次出现的边界。
- R4：checkpoint/runtime identity 漂移不得仅凭 tensor shape 静默放行；若保留诊断 preview，必须有显式兼容合同、能力限制和不可误认为正式结果的标记。
- R5：剖析 Linux 端到端训练阶段的完整 step wall time，消除 178-group round-robin 与 8-resident LRU 造成的稳态 thrash，并保持随机访问、显存有界、query stream 可恢复和 source/query recipe 语义不漂移。
- R6：补齐 phase-local/rolling step wall、group ID、cache hit/miss/evict、session compile、resource verify/upload、reference dispatch、rejection、forward/backward/optimizer、validation/checkpoint 等低扰动观测；普通 step 不得为 profile 强制高频 GPU host sync。
- R7：检查 20k 前后训练状态和资源增长，并审计其它静默失败模式：采样分布偏差、schema/单位/frame/UV/颜色空间不一致、输出尺度或激活错误、梯度断开、参数未注册、权重错配、fallback、hash/identity、NaN/Inf 掩盖与 viewer 错误隔离。
- R8：所有根因结论必须可重复；每个已修复根因必须有回归测试或版本化验收脚本。不得以旧 checkpoint 或单一 finite/ready 指标作为正确性标准。
- R9：保持统一 `OnlineTrainingProducer`、`TrainingCheckpoint@4`、`ScatteringPackage@2` 和 viewer 公共合同，不引入 Metal 专用 runner、磁盘训练 batch、family-specific backend 或 viewer 分支。
- R10：Windows 验证必须运行 full-shape Metal 模型和与 Linux long 同语义的 source/query/phase 路径；允许缩短 step、选择有覆盖性的诊断 cohort 和控制运行预算，但不得缩模型、替换 loss、改变 sampler、绕过 reference 或使用另一套 checkpoint/exporter。
- R11：Linux full-cohort long run 使用与 Windows 已验证的同一共享实现和上层合同；平台差异只允许存在于既有 `ReferenceBackendCapability`、Falcor device API、launcher/toolchain 和硬件资源参数边界。
- R12：Windows 验证产物必须足以在 Linux 长训前拒绝明显错误，包括 phase readiness、学习信号、严格 checkpoint round-trip、部署 parity、viewer 语义、稳态调度、资源上界和性能观测完整性；Linux 仍负责 full cohort、完整训练预算和目标机真实性能证据。
- R13：重构 Metal phase/loss/parameter-group 合同，使 source adaptation、空间/纹理表示、typed compiler、prepare 与 `evaluate()` 从训练早期就在同一 online reference appearance 目标下优化；若保留 reconstruction、teacher、mollification、proposal 或 QAT 辅助目标，必须服务于最终 runtime 目标，并以 matched ablation 或明确数学依赖说明其阶段位置。
- R14：`sample()/pdf()` 的 proposal 参数从首个正式训练阶段开始更新；proposal target 对 evaluator/latent 的反向路径必须按合同 detach，proposal loss 权重采用由 global step 决定、可由 checkpoint 精确恢复的冻结 schedule，不能依赖平台、wall clock 或运行时观测结果临时改写。

## Acceptance Criteria

- [ ] AC1 `[理论/语义正确性｜来源：TrainingCheckpoint@4 与 method component 合同]`：checkpoint 明确携带并验证 completed phase、resume cursor、每个 runtime capability 所需 parameter group 的 finite/nonzero/update coverage；不满足者在正式 export/catalog 前被拒绝。
- [ ] AC2 `[需求交付｜来源：用户要求确认 Metal 是否真正学习]`：固定 seed/source/query 的 evaluator 可学习性实验记录初始化与训练后 prediction/target、loss、梯度和参数变化；独立 holdout query 的结果单独报告，失败能定位最早失效层。
- [ ] AC3 `[数值实现正确性｜来源：统一 scattering/package 合同]`：同一严格 identity checkpoint 经 save/load 后，eager Python、部署量化 Python 与 Slang 在预先冻结的 dtype/oracle 容差内一致；容差来源与 failure action 在正式结果前记录。
- [ ] AC4 `[需求交付｜来源：用户报告 viewer 白模]`：代表性 Metal reference 与 neural viewer capture 不再出现“reference 有材质而 neural 为无信息白模”却仍判 ready 的情况；capture 同时报告执行状态、checkpoint readiness、reference-neural 误差和输出塌缩诊断。
- [ ] AC5 `[理论/语义正确性｜来源：identity 与 package fail-closed 合同]`：method implementation、descriptor、model/asset schema、phase/readiness、source/material identity 或 runtime ABI 不匹配时显式拒绝；不允许仅凭 tensor shape 以正式结果身份加载。
- [ ] AC6 `[需求交付｜来源：用户报告 20k 后约 4 分钟/step]`：Linux trace 能把每个异常 step 归因到 route/group/cache/compile/resource/query/model/validation/checkpoint 中的具体阶段，并确认用户现场口径；不再只有 run-global 累计速度。
- [ ] AC7 `[理论/语义正确性｜来源：有界 online query 合同]`：修复后的 group 调度/缓存保持 GPU online target、group-homogeneous dispatch、有界 residency、确定性 resume 和冻结 query recipe，不再产生稳态逐 step session 重建；绝对 time/memory 作为 observed 指标报告，除非用户另行确认 hard budget。
- [ ] AC8 `[需求交付｜来源：用户要求发现其它问题]`：NaN/Inf、零梯度/无更新、输出塌缩、显存/主存增长、group/source visitation 偏差、错误 fallback 和 identity 漂移均有审计结论；确认的问题有自动回归或版本化验收命令。
- [ ] AC9 `[需求交付｜来源：用户要求全面根因调查]`：形成根因报告；每项发现包含证据、根因分类、影响范围、修复、回归门、observed quality/time/memory 与仍存风险。
- [ ] AC10 `[需求交付｜来源：用户明确决定 D1]`：所有已确认的正确性与性能根因均完成共享路径修复和自动回归，不以诊断报告、增加等待、降低模型或平台特例作为交付。
- [ ] AC11 `[理论/语义正确性｜来源：统一 pipeline 与用户决定 D2/D3]`：同一上层 `TrainingConfig`、producer、runner、checkpoint、method 和 package 流程可在 Windows D3D12 与 Linux Vulkan capability 下执行；静态检查和双平台身份记录证明 upper layer 不按 OS/device API 分支。
- [ ] AC12 `[需求交付｜来源：用户决定 D2]`：Windows full-shape、真实 online reference 的完整 lifecycle 验证证明 evaluator 能学习、checkpoint 可严格恢复/部署、viewer neural 输出保留材质信息，并证明修复后的热循环不存在 execution-group 每 step 重编译/重建；绝对 throughput 作为 observed 结果报告。
- [ ] AC13 `[需求交付｜来源：用户决定 D2/D3]`：Linux 交接以同一 commit/config semantic fingerprint 执行 full-cohort smoke、resume 和 120,000-step long run；Linux 结果包含完整性能 trace、checkpoint/review/package，并由 Windows 对同一最终 checkpoint 完成部署/viewer 验证。
- [ ] AC14 `[需求交付｜来源：用户决定 D5]`：新的 Metal 训练从首个正式优化阶段就产生 evaluator reference loss，并对空间/纹理表示、compiler、prepare 与 evaluator 全链路形成有限、非零、可更新的梯度；早期 diagnostic checkpoint 在 holdout/viewer 中呈现可辨识的材质响应，后续阶段在不改变上层跨平台合同的前提下逐步细化。
- [ ] AC15 `[需求交付｜来源：用户决定 D6]`：首个正式训练阶段对 proposal sampler 产生有限、非零、可更新的梯度，detached evaluator 边界有自动测试，proposal 权重 schedule 在 uninterrupted 与 save/resume 运行中逐 step 一致；最终 `sample()/pdf()` identity 与有界支持合同继续成立。
- [ ] AC16 `[需求交付｜来源：用户决定 D7]`：在同一 source/query/model/work 配置下保存修复前后 matched profile，按 wall time 占比从高到低处理主导热点；修复后 steady-state 不发生逐 step session compile/materialization，CPU/GPU/显存资源保持有界，phase-local rolling throughput 不随已完成 step 系统性恶化。绝对吞吐、总 ETA 与峰值显存只报告实测值，不决定任务成败。

## Out of Scope

- 在根因未确认前更换 Metal 候选架构或通过单纯增大模型/训练步数掩盖问题。
- 把持久化训练 batch/corpus 重新引入正式架构。
- 与本故障证据无关的多灯 scaling、路径追踪方差或 UE 集成扩展。
- Windows-only runner、Windows-only checkpoint 修补器、Linux-only query 语义或任何由上层按平台选择的备用训练实现。

## Open Questions

- 无。后续若 profile 证明必须改变 frozen source/query recipe、模型 identity 或训练预算，必须回到 planning，不在实现阶段自行扩张。
