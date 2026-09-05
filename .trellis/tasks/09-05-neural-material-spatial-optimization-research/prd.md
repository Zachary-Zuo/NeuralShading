# Neural material 原始纹理编码、读取对齐与模型修复

## 目标与用户价值

把纹理资产主路径落实为“原始纹理 → 按语义共享的空间 encoder → 联合 latent → prepare/evaluate decoder”，让邻域信号、绝对语义数值和实际部署读法贯穿训练与编译。复用已有研究解释旧问题，用有界诊断区分表示、采样/目标和 loss/correction 的影响，同时修复已确认的模型错误。

用户已在规划审阅后明确要求「开始实施」，任务进入 `in_progress`。实际实施与验证记录见 [progress.md](progress.md)。随后用户澄清：**只融合相同 UV 的纹理，不同 UV 的纹理分开处理**；这替换原先所有 slot 合成同一双平面的假设，具体合同见 design §9。

## 已确认背景

**最新收尾边界**：用户已要求提交并归档本实现阶段，把后续实验迁到独立的服务器 24 小时任务。验收完成情况和未完成项去向以 [closure.md](closure.md) 为准；下文原需求/清单保留历史，不以归档状态冒充所有实验已完成。

- 用户先要求创建研究任务、复用先前资料并查看 viewer 图像，随后追加模型审计；又明确 R8 原始纹理 encoder 与 R9 同语义统一数值尺度是必须满足的结构合同。2026-09-05 最新指示说明另一会话已全部提交，要求据当前任务更新 PRD/design，规划具体代码修改；不再以等待架构为规划阻塞。
- 当前读取 HEAD 为 `e3f1c216401a1156c288a9e735a690bc446dca2b`，架构提交为 `ea2d743`。归档任务是 [architecture-reset-training-workflow](../archive/2026-09/09-05-architecture-reset-training-workflow/prd.md)。代码须接入当前 `Method`、共享 online session/engine、checkpoint、package 与 `outputs/`，不恢复旧训练入口。
- 历史证据基线为 `cc4d76bf4df089b725ad91b2a2673ca177edff86`。用户引用的四点总结定位于 Codex `01a07002-caaa-7c23-9b31-c6095ab05a97` turn 7，实验依据为旧任务的 `research/final-conclusions.md` 与 `characteristic-probes.md`。有限 hybrid/v4–v6/fixed-batch/mixed 结果未定位唯一根因，亦未证明 692-source 泛化或实时帧率。
- 前期 [texture encoder 共享研究](../archive/2026-08/08-30-vmaterial-metal-neural-system/research/texture-encoder-sharing.md:29) 已提出分组输入、共享空间主干与跨图融合。当前 summary MLP 未满足这一输入合同；架构迁移保留了相关数学实现，当前证据和文件锚点见 [提交后代码核对](research/post-architecture-code-plan.md:15)。
- 历史 [证据分析](research/evidence-and-diagnosis.md)、[viewer 观察](research/viewer-evidence.md) 和 [模型审计](research/model-design-audit.md) 保留原始结果、单位、源码锚点与限制；它们不作为本轮已运行新模型的证据。具体技术合同和执行步骤分别以 [design.md](design.md) 与 [implement.md](implement.md) 为准。

## 需求与代码范围

- **R1 证据复用**：保留原总结、旧实验和前期调研的可追溯表，区分原始观察、历史解释、当前静态发现与待检验假设。
- **R2 完整数据链**：覆盖 source → tile → encoder → latent → prepare → evaluate → loss。修正坐标/LOD、主/paired query、过滤 target 和量化/bilinear 的不一致；通过通用 conditioning 资源关联传递共享 raw tile，不引入 Metal 专用训练调度器。落点与接口证据：代码核对 §2、design §8.1/§8.3。
- **R3 视觉证据**：使用能识别材质、左右侧和 renderer 的历史 viewer 图像；阶段收尾再以新模型同场景图像补充数值观察，不把 progressive tile 或不同模式的截图当帧率/空间质量证明。
- **R4 可区分原因的诊断**：先 D0 语义/读法 witness，再 D1 原始 encoder 与修正后的 summary matched 对照，最后按证据进入 D1b refinement/code control 或 D2 loss/correction。固定项、工作量、统计单元和停止条件须在运行前登记；质量无收益也是可交付结论。
- **R5 部署和原生语义边界**：保留 source-native reference、参数编辑、GPU online GT 与有界运行时。每个兼容 UV 组分别编码为 Detail/Context RGBA SNORM plane，Context 每轴 1/4；读取数按 UV 组及其原生 lookup 次数登记。保留 11,392 evaluate dense MAC、C5 独立 proposal frame 与 176 B packed state；prepare 的输入/成本按多组读取重新推算。真实时间/内存另报，静态推算不等于实时性能。
- **R6 架构对齐与变更隔离**：在 `ea2d743` 后的当前架构上实施；旧 checkpoint/ABI/CLI 不要求兼容。源/target/模型数学改变后 fresh run，不将旧数值直接拼入新 matched 表，不修改其它会话的成果。
- **R7 模型错误**：C1 control 初值再次 tanh、C2 Slang softplus 尾部消减、C3 无效结果成为有效零、C4 frame 接缝、C5 reverse PDF view-conditioning 缺口均给出确定修复。其触发条件、严重性、历史文件:行号及影响范围由模型审计 C1–C5 持有；当前修复与阻塞关系见 design §3、§8.5/§8.6。Beckmann G、secondary 类型及 view-conditioned core 的物理解释按近似报告，不自动扩大 lobe/模型族。
- **R8 原始纹理 encoder 必修结构**：五类语义 stem 在各图原生 mip0 上学习，保留完整二维数据与有效通道/缺图 mask，随后对齐 learned feature、跨 slot 融合并生成 hierarchy。默认 encoder-only，不允许中心值/均值/少量导数取代唯一输入，也不允许新资产自由 latent 或 asset-ID 学习表成为隐藏步骤。去掉 `variant_scale_bias` 与固定资产数；新增 C6 处理部署的训练 source 名单限制（[method.py:847](../../../src/ncls/learning/methods/metal/method.py:847)）。固定 E/D 可为已支持 schema 的未见 snapshot 直接 cook，decoder 输出线性 `f`，保留原生图/参数条件，不必先重建完整纹理或层模型。
- **R9 绝对数值语义**：同语义定义与单位使用固定映射；禁止逐图/通道/tile/batch 的 min-max、均值方差、直方图或曝光自适应输入变换。UNORM/声明的颜色和 normal 解码合法，normal 单位化必须核对原生意义及插值顺序；不裁剪合法 HDR/height，不删 authored scale/bias。当前读取链未发现逐图统计归一化，不将这一禁令写成已定位旧根因。边界与反例保留于 design §2.1、证据分析 §3.3。

## 验收标准

AC1–AC8 保留原研究交付编号，均只验收规划材料；AC9–AC15 是本次细化的代码验收。每项来源明确，未实施的项目不打勾。

| 状态 | 验收项与需求映射 | 类型、来源及证据 |
|---|---|---|
| [x] | AC1（R1）：总结、旧实验、前期纹理调研可追溯，指标单位明确 | 需求交付；用户要求；证据分析 §2/§6/§7 |
| [x] | AC2（R2）：解释有 encoder 仍丢失邻域的条件，包含采样/loss 替代原因 | 需求交付；用户提问；证据分析 §3–§5 |
| [x] | AC3（R3）：实际查看五张原图，记录身份与限制 | 需求交付；用户要求；viewer-evidence |
| [x] | AC4（R4/R5）：D0/D1/D2、matched 项、统计单元和停止条件明确 | 需求交付；用户研究要求；design §3–§6、implement §3 |
| [x] | AC5（R6）：交付 PRD/design/implement 与研究笔记，只修改本任务，保持 planning | 需求交付；用户本轮规划范围与 Trellis 规划合同 |
| [x] | AC6（R7）：C1–C5 有修复方案、独立 witness 与阶段依赖 | 需求交付；用户追加审计要求；模型审计与 design §8.5/§8.6；不代表已修复 |
| [x] | AC7（R8）：原始 encoder 是必修结构，取消先证明收益才实施的条件 | 需求交付；用户明确结构要求；design §2/§8.2/§8.4 |
| [x] | AC8（R9）：固定语义尺度、范围碰撞反例与图外统计不变 witness 明确 | 需求交付；用户明确数值要求；design §2.1；不代表 witness 已运行 |
| [ ] | AC9（R2/R8/R9）：学习层直接消费固定解码的 raw mip0；完整图/tile 对应区域等价；改变图外统计/batch 不改变解码值；response 梯度能到各有效语义分支 | 需求交付＋语义/数值正确性；R8/R9、卷积和地址数学；tests 计划见 implement P1/P2 |
| [ ] | AC10（R2）：量化四邻点读取、UV/LOD/Jacobian/paired random、program/prepared pack 在训练/cook/Slang 一致；point 与 filtered GT 分开 | 语义/数值正确性；统一 runtime/reference 合同；D0 独立 oracle，容差在质量比较前冻结 |
| [ ] | AC11（R8）：支持 schema 内的未训练 snapshot 只用冻结 E/D 和原始资源编译；不增加 learned asset state、不优化；相同内容更换 locator 不改变编码 | 需求交付＋角色隔离；用户 R8；C6 与 encoder-only 生命周期 witness |
| [ ] | AC12（R7）：C1 初值等价、C2 小值/单调数值、C3 零/无效区分、C4 连续性与原生轴、C5 独立反向 prepare 的真实密度及 capability 均通过 | 语义/数值正确性；模型审计对应数学与公共接口；独立 witness，不以双方同错 parity 替代 |
| [ ] | AC13（R2/R5/R6）：共享资源经 concat/rejection/select/release 不错配或泄漏；当前 train→checkpoint/resume→validate→export/eval 闭环成立；Nvidia 空资源路径回归通过 | 需求交付＋资源/生命周期正确性；当前通用 pipeline、lease 和 checkpoint 合同 |
| [ ] | AC14（R3/R4/R5）：按冻结 diagnostic 配方交付空间/峰值/分支贡献、分组 CI、真实成本和阶段末 viewer 证据，并记录无收益/未执行分支 | 需求交付；用户研究目标与当前研究流程；质量/time/memory 均 report-only，不要求超过旧指标 |
| [ ] | AC15（R5）：包 inventory 与 shader 实际结构一致，encoder 不在 runtime evaluate 执行；真实 latent bytes/packed state/read/MAC 完整登记 | 工程合同＋需求交付；method-constraints、当前 profile 有界部署合同；超过所选形态先停止并回规划 |

数值门槛只用于同一数学实现的正确性。每项 tolerance 在独立构造/calibration 上记录 `source/scope/why_hard/failure_action`，不能在看过正式质量后调整。研究质量/时间/内存结果默认不能决定任务失败，不追加训练直到“过门”。

## 范围之外与已解决决定

- 另一会话已完成的架构工作。
- 任意未知 MDL 图的普适 compiler、所有源族强制提供纹理或层参数、每 finish 独立网络。
- 自动追加 seed、模型、步数、692-source formal long，及 PT 方差/多灯/UE 集成准入门。
- 迁移、删除或重新生成旧资产/图像；旧 checkpoint/包转换器。
- 将 8 个 latent 通道强加为 source 的公共物理语义，或将有限 code control 称为质量上界。

没有阻塞本轮规划的用户决策。具体实现形态与代价已经落到 design，诊断扩展的分支边界落到 implement；实际资源表现、normal 原生顺序及质量影响由预定 witness/实验回答，不再以等待架构或泛泛“继续调研”为交付条件。
