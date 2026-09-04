# Metal 神经材质预算重设与模型重构

## Goal

不再把 `metal_fused_full_v1` 视为可继续压缩的目标结构，而是在明确的部署预算下重新分析并实现 Metal neural evaluator。新模型的主 profile 必须同时具备可验证的局部散射质量与有意义的单次查询成本，使主 profile 上得到的结论能够直接回答“这种结构是否适合实时 neural material program”，而不是依赖一个远超目标预算的 full profile 间接推断 compact 形态。

本任务同时完成上一轮诊断路线中的三项前置工作：单材质短程过拟合、loss/质量观测修正、以及与 reference 和 NVIDIA 方法口径一致的 runtime 测量。它们既是新模型设计的证据来源，也是开始下一轮正式长训之前的准入门槛。

## Requirements

### 1. 预算与结论边界

- 新模型主 profile 采用用户于 2026-09-04 确认的 NVIDIA-class 原则：`evaluate ≤ 20,000 MAC/direction`、prepared state `≤ 192 B`。
- 上述两个 hard 数值的 `source` 是用户确认；`scope` 是目标 evaluator 的主 profile；`why_hard` 是让该 profile 的质量结果能够直接支持部署形态判断；`failure_action` 是把超额候选降为 diagnostic/teacher 并回到 planning，不静默扩宽预算或继续以主 profile 身份训练。
- `prepare` MAC、共享权重、材质资产、固定随机读取数和实测 GPU 延迟先按 matched 口径测量；随后在正式候选冻结前记录目标值与来源。它们在获得用户确认或权威实现约束前均为 report-only，不能事后反写成完成门。
- 主 profile 的预算必须以 NVIDIA neural material 与 optimized-code reference 的 matched 测量为依据，而不是从现有 `metal_fused_full_v1` 的规模按比例缩小。
- 可以保留不超过主 profile 神经网络 MAC 约 4 倍的诊断性 teacher/profile，用于判断数据与训练瓶颈；该倍数是研究资源 cap，不是质量保证，也不能作为目标方法的质量—成本结论或替代主 profile 结果。若确需扩大该 cap，必须重新取得用户确认。
- `metal_fused_full_v1` 只作为历史回归和消融对照；不得通过静默改形状或复用旧 method/profile 身份把新结构伪装成兼容升级。

### 2. 单材质可表达性与训练诊断

- 建立固定 Metal 材质、固定参数状态和固定方向/空间探针的短程过拟合实验，至少覆盖微小划痕、亮高光及高光颜色。
- 对 eager、量化 Python、Slang/package 输出分阶段比较，从模型表达能力、量化误差和部署 parity 三个层面定位问题。
- 单材质结果先按预登记的相对选择规则和图像/数值证据报告；在该结果完成失败分类前不扩大到多材质训练。observed quality 不作为任务事后 hard gate，失败时也不得靠自动增加训练步数或扩宽模型掩盖。

### 3. Loss 与质量观测

- 训练进度和报告分别展示 appearance loss、proposal loss 及其权重；不再把可能为负的连续密度 NLL 总和当作唯一质量指标。
- 保留 proposal objective 的数学语义，并明确记录其为负时的原因；如需改变归一化或显示方式，必须与训练梯度语义解耦并有回归测试。
- 新增能够暴露当前失真的指标和固定 probe：逐通道误差、亮度/色度误差、峰值与高能尾部误差、微细节或空间频率保真度。
- validation 必须使用稳定的 source locator 与 query recipe，并能分辨训练拟合、未见参数状态和未见材质上的变化。

### 4. Matched runtime 基线

- 在相同 GPU、精度、batch/packet、方向布局、同步和预热口径下，分别测量 reference、NVIDIA faithful baseline、旧 full profile 与新候选。
- 分开报告 `prepare`、单次 `evaluate`、首次 `prepare+evaluate` 以及同一着色点复用多方向时的摊销成本；同时报告静态 MAC、状态、权重、资产与读取数。
- benchmark 必须能解释“neural 比 reference 慢”的来源，不能只给 viewer 整帧时间或把训练吞吐当作运行时成本。

### 5. 新模型重新设计

- 从运行时合同和已观察到的失败出发重新选择表示，不要求继承现有 typed attention、宽 U-Net、四级 latent bank、十 lobe head 或当前聚合方式。
- 空间编码必须能保留源材质中的高频方向性结构；法线/语义特征若参与 runtime，应由直接监督或等价可验证信号约束，不能仅监督一个未被 evaluator 实际消费的旁路 head。
- RGB/光谱相关的高光颜色必须由与 reference 一致的目标和输入条件约束，避免仅靠 luminance 权重决定峰值与 proposal 行为。
- `prepare()` 只承担同一着色点可复用、且能证明具有摊销收益的工作；`evaluate()` 保持固定成本和固定读取数。模型候选在登记前通过 `.trellis/spec/project/method-constraints.md` 的静态部署检查。
- 新结构应先实现最小可判定版本，再通过 matched 消融证明每个新增模块对质量—时间—内存 Pareto 的净贡献。
- 实现继续服从统一 `prepare/evaluate/sample/pdf` dispatcher、线性 `f` 输出和 MethodBundle/Slang 生命周期；本阶段以 evaluator 为中心，sampler 只做保持合同或与 evaluator 对齐所必需的工作。

### 6. 执行边界

- DDP 基础设施已有独立验收证据；2026-09-05 用户进一步授权在 GPU 5–9 上执行本任务的五卡训练。若真实 budgeted objective 暴露 reducer、rank-local reference、checkpoint 或 teardown 缺陷，本任务允许做范围明确的基础设施修复并补回归，但不得靠放宽 `static_graph`、启用 unused-parameter 扫描或提高 NCCL timeout 掩盖根因。
- 在单材质表达力、观测完整性和 matched runtime 结果完成审阅前，不恢复旧 v4 checkpoint 的长训，也不启动旧 `metal-compact-ablation` 的盲目 sweep。
- 不修改固定提交的 `external/` 上游源码，不把临时实验产物提交到根仓库。
- 这是复杂任务；在实现前补齐 `design.md` 与 `implement.md`，并在需求总结获得用户明确确认后才进入开发阶段。

### 7. 2026-09-05 连续训练与交付增补

- 用户已授权约 12 小时的连续监督、范围内修复和分段本地 commit；不得 push。监督必须以进程退出、checkpoint/metric 事件和低频 heartbeat 驱动，完整 stdout/stderr 写入 `artifacts/`，不在对话中做秒级轮询或转储逐 step 日志。
- 使用物理 GPU 5–9 运行一个五卡 DDP 作业；hybrid 与 direct 不并发争抢卡，而是在共同的 stop/resume 里程碑间交替执行。两侧必须使用相同 world size、per-rank batch、source/query、loss、optimizer、schedule、precision和验证 cadence。
- 五卡保持冻结配置的 per-rank batch 64，因此 global batch 为 320、每 step 全局 work units 为单卡协议的五倍。该运行使用独立 `ddp5` artifact/plan identity；hybrid/direct 之间可比较，但不能冒充原单卡 2048-step pilot或与其按 step 直接合并统计。
- 先完成 DDP step-0/短步 smoke 与 checkpoint/resume，再按 `128/256/512/1024/2048` 共同里程碑交替推进。若实现修复改变 resolved plan、method/data/query identity或训练数值语义，旧 checkpoint 不继续混用，两侧从相同有效边界重跑。
- 对每个里程碑分析 appearance、proposal、chroma、逐通道 peak、spatial gradient、required-group gradient/update、rank straggler、reference/cache 与显存；有限且下降的 total loss 本身不足以判定模型有效。
- 最终至少保留 hybrid 与 direct 两个 exact-identity checkpoint，并为两者生成 Windows viewer 可消费的 diagnostic evaluator package/catalog。未完成 formal readiness 或缺少 `sample/pdf` 的候选必须明确标注 diagnostic，不能显示为 formal ready。
- 约 12 小时是本轮执行 timebox，不是伪造成功的验收门；到时按已完成的共同里程碑给出证据结论、失败分类、可恢复位置与下一步方向。
- 只有 hybrid/direct训练、比较、双Windows diagnostic package与必要正确性检查已经交付且12小时时间仍有余额，才进入后置探索。探索先研究少量机制鲜明的单材质，再决定是否值得扩到小型混合cohort；它只形成设计方向与候选优先级，不阻塞主任务完成。

### 8. 2026-09-05 吞吐优先与自适应训练增补

- 用户进一步明确执行优先级为：先依据真实 profile 优化通用训练架构与同步/验证调度，再审查模型结构和 loss，随后成对推进 hybrid/direct，最后才使用剩余时间做额外实验。性能修改必须优先消除占主导的公共热点，不能为了提高瞬时 GPU utilization 改写 reference、query 或 loss 语义。
- `128/256 step` 是正确性、吞吐与早期趋势探针，不是质量结论 cap。两侧在实现正确时继续到共同的 `512/1024/2048` 里程碑；若 2048 后结构选择仍不确定、至少一侧在冻结 validation 上仍有统计支持的改善趋势且 timebox 足够，可按设计中的预登记扩展阶段成对继续到最多 4096 step。扩展保持同一 seed、source/query、静态模型预算和 matched 轴，不自动新增 seed 或 sweep。
- 任何影响训练数值、batch 几何、数据顺序、loss、optimizer、schedule、precision 或 resolved identity 的效率改动都产生新 recipe/checkpoint identity，hybrid/direct 从同一 fresh 边界重跑。只改变等价 metric 聚合、进度频率或不参与梯度的 profile 实现时，可保留旧结果作为性能对照，但不得覆盖旧 artifact。
- 在 GPU 5–9 上允许把per-rank batch从旧基线64提高到128/256/512做有界吞吐profile；以稳定段global work units/s、step wall、GPU利用和peak memory的Pareto选取共同batch，不按单个`nvidia-smi`快照选择。新pair两侧必须使用同一入选batch，并同时报告global batch和累计work units；大batch run不得与batch64旧证据按step直接比较。
- 12 小时内不因某个早期里程碑信息不足而向用户请求是否继续；只在安全、权限或任务范围出现新的实质阻塞时才停下。任务、执行顺序、checkpoint、failure classification 和本地 commit 由当前 Trellis task 与 continuous goal 持久化，不依赖单个对话上下文。

## Acceptance Criteria

- [ ] 【需求交付｜来源：用户本轮要求】冻结经用户确认的主 profile 预算表，清楚区分硬性静态边界、实测目标和诊断性 teacher 额度，并为每个 hard 数值记录 `source / scope / why_hard / failure_action`。
- [ ] 【需求交付｜来源：用户明确要求完成前置项 3】固定单材质短程过拟合可复现，并以数值 probe 和图像证据报告主 profile 对微小划痕、亮高光与高光颜色的表达结果；观察质量只作研究结论，不以事后改门保证通过。
- [ ] 【需求交付 + 数学语义正确性｜来源：用户明确要求完成前置项 4；连续 PDF NLL 定义】训练/验证日志分开展示 appearance 与 proposal，解释负 proposal NLL，新增逐通道、色度、峰值与空间细节指标并提供测试覆盖。
- [ ] 【需求交付｜来源：用户明确要求完成前置项 5】形成 matched runtime 报告，至少包括 reference、NVIDIA faithful baseline、旧 full profile 和入选新候选的 `prepare/evaluate` 分解与静态资源表。
- [ ] 【需求交付｜来源：用户本轮“重新全面分析”】完成候选结构分析与淘汰理由；入选结构的每个主要模块都有对应失败假设、预算增量和预先登记的消融或正确性检查。
- [ ] 【需求交付 + 接口正确性｜来源：用户本轮“实现为新的模型”；项目 runtime 合同】以新的 method/profile 身份实现入选主模型，训练、checkpoint、量化、MethodBundle 与 Slang 路径不依赖旧 full profile 的静默兼容。
- [ ] 【需求交付｜来源：用户本轮质量—成本可推断性要求】在用户确认的预算内完成冻结 validation 和质量—时间—内存报告；observed quality/time/memory 只用于相对比较和失败分类，不由任务执行过程反写成完成门，也不能用 teacher 结果替代主 profile 结果。
- [ ] 【理论 / 数值 / 接口正确性｜来源：项目合同与本任务涉及生命周期】通过相关单元测试、静态成本验证、单 GPU smoke、DDP 回归和最终部署 parity；容差在正式结果前由 dtype、oracle 或隔离 calibration 冻结，正式环境与运行证据按项目规则记录。
- [ ] 【连续执行交付｜来源：用户 2026-09-05 要求】GPU 5–9 的五卡 DDP 以共同里程碑交替完成 hybrid/direct 的可恢复训练；监督日志、异常处置、commit 与比较身份可追溯。
- [ ] 【Windows 检视交付｜来源：用户 2026-09-05 要求】至少产出 hybrid/direct 两个训练 checkpoint 与两个可由 Windows viewer 加载的 exact-identity diagnostic package/catalog，并明确 formal/diagnostic capability 边界。
- [ ] 【条件性交付｜来源：用户 2026-09-05 补充】若主交付完成后仍有时间，对少量有代表性机制的材质完成有界专项probe；仅在证据支持时扩到小型普适性检查，并报告哪些结构机制值得下一轮正式候选。

## Notes

- 初始证据与历史观察记录在 `research/initial-evidence.md`，不把单次运行结果写成事后需求。
- 预算决策已完成：主 profile 使用 NVIDIA-class 约束；旧 full profile 与至多约 4 倍主 profile MAC 的模型只进入诊断轨道。
- 2026-09-04：用户已明确批准 `prd.md`、`design.md`、`implement.md` 与最终规划摘要，可以进入实现；后续若实质扩大模型、训练预算、formal cohort 或验收门，必须回到 planning。
