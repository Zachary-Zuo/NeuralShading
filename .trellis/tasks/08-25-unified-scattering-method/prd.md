# 统一散射方法与路径追踪验证

## Goal

建立一条遵守现有架构规范、可在通用 Slang backend 中部署的完整散射方法链路。该方法需要提供相互一致的 `evaluate()`、`sample()` 与 `pdf()`，能够在 path tracing 中检查局部求值、采样与概率密度的正确性，并在 deferred 路径中正确执行 `evaluate()`。

这项工作的价值不是保留现有端到端演示，而是让 neural material program 首次具备可验证的 evaluator/sampler 闭环，并为后续方法研究、数据生成和 viewer 集成提供唯一可信的基线。

## Background

- `p1_audit.md` 与 `p1_v2_plan.md` 含有可复用判断，但部分内容已经过时，不能直接作为当前执行计划。
- 当前需要先根据仓库代码、测试、合同与架构规范核实现状，再由用户确认仍属于产品/研究取舍的事项。
- 本任务只处于 planning；创建任务不代表已经批准实现。

## Confirmed Facts

- 右侧 viewer 当前不是通用 backend，也不能 path trace MethodBundle：loader 和三个 shader pass 硬编码 `film-m1-direct-neural`，右侧只执行 deferred approximation；左侧 path tracer 只分派四种 source reference。证据与行号见 `research/current-state.md`。
- 当前 `lobe_residual` 只是未完成骨架：训练入口抛 `NotImplementedError`，`sample()` 固定失败，`pdf()` 固定为 0，也没有合同包装。因此它不能作为“已有方法继续补 viewer”的起点。
- 公共散射语义已经存在；本任务应补齐真实 backend、MethodBundle/runtime specialization 和 viewer 两种 renderer path，而不是新造一套 method-specific 公共接口。
- `reference-shard` v5 已保存训练 evaluator 与 tractable proposal 所需的方向 response、采集 proposal PDF 和 solid-angle weight。对当前 LayerStack upper-hemisphere 反射域，没有证据要求先改合同或重采 P1 v1。
- 当前 LayerStack reference 不提供可蒸馏的完整 `reference_pdf`，且 P1 v1 不含外部透射/delta。若首个方法把这些事件纳入范围，则必须扩展合同和重采。
- 目标方法必须保留逐方向小型 `EvaluateMLP`；解析 core/proposal 可以复用，但“`prepare()` 只输出 lobe，`evaluate()` 纯解析”只能作为 optimized-code control，不能冒充目标 neural method。

## Requirements

- 首个方法采用 target-visible offline compile：给定任意受支持的 LayerStack `MaterialProgram`，通过该材质的 reference response 离线优化材质 latent/compiled asset；编辑源参数后允许重新 cook。本任务不要求 feed-forward source compiler 对未见状态即时生成 latent。
- 首个方法的事件域固定为 LayerStack upper hemisphere、reflection-only、non-delta；不声明或部分实现 transmission、delta、volume boundary capability。
- 方法必须实现语义一致的 `evaluate(wo, wi)`、`sample(wo, u)` 与 `pdf(wo, wi)`；三者的方向约定、测度、余弦归属、事件支持域和数值边界必须有明确合同。
- sampler 必须是共享 `ScatteringState h` 的 learned tractable distribution head：神经网络预测已知归一化、可采样且密度可计算的分布参数；`sample()` 与 `pdf()` 使用同一组参数和同一 proposal，不使用固定 proxy，也不采用无法计算 Jacobian/PDF 的裸方向 MLP。
- 首个反射域 proposal 必须含严格正的 full-support 基础分量；其余 learned lobe 的参数约束、混合权重与方向变换必须保证“有效连续方向密度 + 显式 null-event 概率质量”归一化、有限值和静态有界执行。非 delta 有效样本的路径权重只能由同一个 neural `evaluate()` 按 `f·|cos|/pdf` 得到，null event 返回零贡献且不能被静默重采样。
- path tracing 必须通过通用方法接口调用该方法，用于检查 evaluator、sampler 与 PDF 的一致性和渲染正确性。
- deferred 必须通过同一通用方法接口执行正确的 `evaluate()`；不能维护一套只为 deferred 存在的旁路语义。
- viewer 必须能实际显示该 MethodBundle：至少提供“左侧 source reference PT / 右侧 method deferred”和“左侧 source reference PT / 右侧 method PT”两个明确命名的比较模式；两种右侧路径都通过同一通用 backend specialization，不直接调用 backend 自由函数。
- 实现必须遵守项目架构规范，并以通用 Slang backend 为唯一方法前向实现，不引入 backend-specific 公共接口或重复的 Torch/C++ 前向。
- 必须核实并按需修改 viewer：包括当前右侧视图是否已经能够在通用接口下执行 path tracing、若不能还缺少哪些 capability、状态或 pass 生命周期支持。
- 必须核实现有样本采集合同是否足以训练和验证完整散射方法；若合同语义或字段不足，需要版本化修改并重新采集受影响语料，不能把旧数据静默解释成新合同。
- 任务完成时，旧的混乱实现、错误模型与失效数据必须退出可达路径并归零；可由用户预先安全删除的未版本化数据，应先给出按目录分类、说明重建来源与删除时机的清单。
- `legacy_ltc_k2` 不保留旧方法身份或兼容入口；其中已验证的 LTC、VNDF、sample/PDF 与方向变换公式应迁移、重命名并封装为 Falcor-free 通用 Slang 数学组件，供目标 sampler、optimized-code control 和合同测试复用。该 control 必须走同一通用 backend 接口，不能成为 fallback。
- 在进入实现前必须冻结首个具体方法的完整设计：evaluator 数学参数化、physical core 的职责、latent/`prepare`/逐方向 MLP 划分、learned proposal family、训练 loss、容量与精度、`CompiledMaterial`/`ScatteringState` 布局、offline cook、MethodBundle 导出及失败判据。设计必须吸收 `p1_audit.md`、`p1_v2_plan.md` 和现有 `lobe_residual` 中仍成立的机制证据，不能为了缩短实现路径选择已知表达能力不足的粗糙方法。
- NVIDIA Real-Time Neural Appearance Models 的核心结构必须成为首个 matched baseline：two learned shading frames + direct BRDF MLP + neural head 预测的 tilted-cosine diffuse / non-centered anisotropic GGX specular 两项解析 proposal；训练复用 log-space L1、directional mollification 可行性、sampler 相对当前 evaluator 的 KL，以及 sampler loss 对 latent detach 的证据。
- 目标 evaluator 冻结为 `core-frame-neural-v1`：在同样 learned-frame direct MLP 主体上增加 exact top-interface reflection core，并用必选的 positive neural residual 补全其余散射。sampler 以 NVIDIA 9 参数 proposal 和 K=2 LTC proposal 为 matched 配置轴；只有经过相同数据、预算、训练与统计协议比较后才选择部署 bundle，不能预设 LTC 必然更好。
- 当前 Slang 2024.1.34 路径没有本项目可验证的 cooperative-vector 加速能力；这不妨碍原规模 `64×64×64` evaluator 以普通 Slang 标量矩阵乘法导出 MethodBundle、由通用 viewer 加载并在 deferred/PT 中显示。原规模形态就是正式 baseline：其实际成本和 runtime class 如实登记，超过软线不构成复现失败，也不触发缩模替换。缩模只能在原规模复现完成后以独立方法身份研究，不能作为 baseline 验收前置。
- 不修改锁定的 `external/` 上游源码；viewer 变更继续位于根仓库自有代码和既定 overlay 边界内。

## Constraints

- 保持源材质族的原生 reference 语义；当前层栈随机游走是该材质族的 reference，不能因为 neural backend 需要而把 GT 改写为其他表示。
- 单次执行、状态和访存必须静态有界；候选方法仍需满足着色器部署可能性。
- `evaluate()` 输出线性 `f`；`prepare()` 承担可复用的 view-conditioned 编码与 latent 获取/过滤；需要材质驱动方向采样时提供匹配的 `sample()/pdf()`。
- 根仓库不接纳 `data/`、`artifacts/`、`reports/`、`build/` 等运行产物；数据迁移和清理必须遵守 registry/package/manifest 追溯规则。

## Acceptance Criteria

- [ ] 需求、设计与执行计划明确说明 `evaluate/sample/pdf` 的统一数学合同、ABI 映射、错误处理和测试 oracle。
- [ ] 对每个测试 state/`wo`，proposal 的解析 PDF 归一化；`sample()` 返回的 PDF 与重新调用 `pdf(sample.wi)` 一致；采样直方图与 PDF 一致；所有支持域内 PDF 严格为正且 grazing/各向异性输入无 NaN/Inf。
- [ ] 对同一 neural evaluator，method PT 的 Monte Carlo 估计与独立高样本/确定性方向积分在预设置信区间内一致；该判据证明 sampler/PDF 正确性，不把 evaluator 相对 source reference 的表示误差混入 sampler 结论。
- [ ] viewer 现状有代码证据；最终方案明确 deferred 与 path tracing 各自通过哪条通用接口执行，以及二者共享和特有的 capability。
- [ ] 至少一个符合着色器预算的真实方法能在通用 Slang backend 中导出并加载，且不依赖旧方法旁路。
- [ ] 该方法在 deferred 中正确执行 `evaluate()`，并在 path tracing 中使用匹配的 `sample()/pdf()` 完成可重复的正确性验证。
- [ ] viewer 可以加载本任务导出的 bundle，并在交互/headless capture 中显示 method deferred 与 method PT；capture manifest 明确记录比较模式、两侧 integrator、spp/反弹上限和 MethodBundle identity。
- [ ] 数据合同是否变更、哪些语料必须重采、哪些旧 shard/模型/产物不可继续使用，都有版本化结论和可执行迁移门槛；不能仅因引入 NVIDIA sampler/KL 就重采，也不能在没有数据证据时宣称现有离散点响应已忠实覆盖 directional mollification。
- [ ] 旧实现、错误模型和失效数据的清理清单完整；版本化路径由任务内删除，未版本化大数据由用户按明确清单删除或重建，最终不存在静默 fallback。
- [ ] LTC/VNDF/sample/PDF 公共数学原语只有一份 Falcor-free Slang 实现；目标 method、analytic control、GPU 测试和 viewer 不复制公式。
- [ ] `nvidia-frame-two-lobe-v1` baseline 对论文结构、训练和 sampler 的复用项与项目适配逐条登记；原规模实现正确、训练稳定收敛并通过同一通用 MethodBundle/viewer 路径显示。收敛后的绝对质量和是否满足软成本线都不决定复现成功。
- [ ] evaluator 的 `{NVIDIA direct, exact-core positive residual}` 与 sampler 的 `{NVIDIA diffuse+GGX9, LTC-K2}` 形成 matched 2×2 对照；implementation/convergence 与 quality/cost comparison 分开报告，后者按材质结构分组并由 paired bootstrap 支撑，不设跨材质统一质量门，也不强制产出一个全局 winner。
- [ ] `design.md` 在实现前给出首个方法的可执行数学设计与逐项成本自检，并逐条标注旧审计/计划中的证据是“复用、修正或淘汰”；不存在仅为 lifecycle smoke 而选择的降级方法。
- [ ] Python、Slang、C++ loader、MethodBundle、viewer 与文档/测试的跨层合同一致，完成规定的质量检查与部署验证。

## Out of Scope（初稿）

- 不在需求尚未收敛时直接实现或启动任务。
- 不把多灯 scaling、完整 PT 方差优化或 UE 集成提升为本任务前置 kill test；本任务只做支撑正确性闭环所必需的 viewer/path-tracing 集成。
- 不在本任务中扩展新的源材质族，除非后续证据表明它是验证通用接口不可替代的最小需求。
- 不把外部透射、delta、volume boundary 纳入首个 LayerStack 方法；这些能力后续必须以独立 capability、数据合同和重采任务扩展。
- 不在本任务中实现 feed-forward source compiler 或用 G2/G2s 证明即时参数编辑；本任务保留编辑后重新运行 offline cook 的正确工作流。

## Execution Structure

- 父任务已建立六个按数字前缀排序的子任务：公共数学 → 数据充分性 → neural baseline/候选 → 通用 runtime → viewer deferred/PT → 旧方法归零。
- 用户已对这个任务树授予 task-scoped continuous execution：每个子任务在启动前仍须根据前序真实产物完整更新并审阅自己的 `prd.md`、`design.md`、`implement.md` 与 context manifests，但只要细化没有越出本父任务冻结的需求/设计边界，就无需再次等待 planning approval 或 commit confirmation，可依次启动、实现、检查、创建仅包含已识别任务文件的本地 commit 并归档；禁止 amend、push 或夹带无关 dirty files。
- 连续授权不允许擅自扩大范围。只有出现新的产品/兼容性/风险选择、未经授权的不可恢复删除、外部权限要求或真实 blocker 时才暂停请求用户；普通技术细节必须依靠仓库证据自行推进。
- path-tracing 场景与数值阈值由 `05` 在产生验证结果前冻结；directional-mollification adequacy 阈值由 `02` 在查询结果产生前冻结。这些是子任务内的防偏置研究参数，不改变父任务产品范围。

## Notes

- 这是跨 core/data/learning/viewer 的复杂任务，最终必须有 `design.md` 与 `implement.md`。
- 用户原始表述中的“彻底归零”按“旧路径不可达、旧资产不可误用、删除范围可审计且可按 reference/manifest 重建”解释；具体物理删除范围仍需在审阅后由用户确认。
