# 忠实复现 NVIDIA Neural Materials

## 目标与用户价值

以作者版论文《Real-Time Neural Appearance Models》及其补充材料为一手规范，审查当前 `nvidia-neural-appearance` 在表达结构、数据生成、训练生命周期、sampler、部署表示和 renderer 接入上的偏差；在不破坏最新统一 pipeline 合同的前提下消除未登记的简化，使项目能够用可追溯证据区分“论文忠实实现”“针对当前 source domain 的显式 adaptation”和普通预算诊断。

## 已确认背景

- 用户已确认“真正忠实复现”包含论文公开的完整方法：encoder、hierarchical latent texture、filter footprint、encoder bootstrap → latent finetune、独立 evaluator/sampler batch、matched two-lobe sampler 与 neural path tracing；本任务按持续执行推进到实现、验证、归档和 scoped local commit，不再等待中途批准。
- 正式训练进入缓慢收敛区间后，用户于 2026-08-27 明确要求停止等待并按 step 200k 记录。本次经验结果以 200k checkpoint 为边界，保留原 300k recipe 身份但不宣称完成 300k 协议；详见 `research/200k-recording-decision.md`。
- 最新架构只保留一条 `SourceSnapshot → TrainingBatch@1 → TrainingRunner → TrainingCheckpoint@2 → MethodDefinition → ScatteringPackage@1 → ComparisonSlot[2]` 路径；NVIDIA 是唯一产品 neural method。新增能力不得恢复 method-specific runner、checkpoint、exporter、Slang session 或 viewer 分支。
- 论文方法由 hierarchical 8D latent texture、训练期 native-parameter encoder、learned-frame BRDF decoder 和 analytic-proxy sampler 组成；encoder 预训练后生成 latent pyramid，再移除 encoder并直接 finetune latent texels。
- 论文训练为 GPU online reference generation，共 300k iterations；每步使用彼此独立的 evaluator/sampler 两个 65k batch。前 20k steps 对 outgoing 方向 `ωo` 使用 `10°→0°` cosine mollification、每个 target 256 个 cone samples；Adam 参数为 `β=(0.9,0.999)`、`eps=1e-7`、zero weight decay，学习率按 cosine 从 `1e-3` 降到 `1e-4`。evaluator 与 sampler 联合优化，KL target 来自当前 learned BRDF，KL 路径 detach latent。
- 任务启动时的实现只保留了 z8、两个 learned frames、`3×64` evaluator、`3×32→9` sampler、`exp(raw-3)` 和 analytic two-lobe proposal 的主要 decoder 算术；训练资产仍是每个 source state 一个直接优化 latent、25k steps、batch size 16、分段常数学习率、AdamW 配置，且 evaluator/sampler objective 复用同一个 batch。
- 任务启动时的 runtime material 只有一个 32-byte z8 record，不含 hierarchical latent texture、LOD footprint 或 stochastic adjacent-MIP selection；viewer package C++ loader 会校验但不绑定 `material.resources`，deferred `prepare` 也未把已有 G-buffer UV/gradient 写入 `NclsScatteringContext`。
- 架构重置前的根因报告已经证明旧配置只是 LayerStack 离线预算适配：其名义方向查询量比论文约少 760 倍，不能称为论文忠实复现。

## 需求

### R1. 一手 correspondence 与身份

- 对论文正文 §4–5、补充材料 §1–2 建立逐项可执行 correspondence；每个差异必须分类为 `faithful`、`source-domain adaptation`、`runtime-contract adapter` 或 `unfaithful`，不得用重命名掩盖实现偏差。
- 完整实现的正式身份限定为“作者公开方法的 functional reproduction”；它不声称拥有作者未公开的 shader graph、训练资产或精确论文图像。formal、短程 smoke 和预算适配实验使用不同、不可混淆的 recipe/checkpoint identity；只有完整满足对应合同的产物才可标注为 functional-faithful。
- encoder 阶段切换点、log-space offset/epsilon、sampler KL estimator、texture container 与未优化 FP16 shader 路径等作者未公开细节，必须作为 `author-underspecified` 的冻结选择进入 recipe/runtime identity，而不是写成作者事实。

### R2. 表达与 source adaptation

- 保留作者定义的 learned-frame evaluator、输出 measure、sampler MLP、参数 warp 与 analytic proposal；公共 `evaluate()` 继续按项目 ABI 输出线性 `f`，cosine-weighted response 转换必须有 exact-vector parity。
- native source 参数到 z8 的 encoder、hierarchical latent representation、encoder bootstrap 与 latent finetune lifecycle 不得被 per-state free latent 静默替代。
- 当前 LayerStack source family 的原生参数和 random-walk reference 保持 GT；不得为了 NVIDIA 方法反演或改写 reference 语义。uniform source 若退化成 1×1 spatial domain，必须作为显式、可验证的 source-domain adaptation，而不是删除论文结构。
- 为真实覆盖 spatial hierarchy，正式功能复现至少使用一个当前 registry 内已有纹理与 UV footprint 的 source snapshot；优先使用现有 MaterialX `standard_surface` reference，按它自己的原生参数/纹理生成 GT。MaterialX 不是作者论文资产，必须登记为 source-domain adaptation；LayerStack 继续提供独立的 1×1 uniform adaptation 证据。

### R3. 论文训练 lifecycle

- online 与 offline 仍消费统一 `TrainingBatch`/runner，但论文 formal recipe 必须真实表达独立 evaluator/sampler batch、online reference、300k iteration、65k batch、连续 mollification、Adam/epsilon/cosine LR 和 joint detach 边界。
- smoke/profile 可以缩短 steps/batch，但只能证明链路与性能可行，不能晋升为 formal result。正式训练前必须先做显存/吞吐 preflight，并在长循环显示 work units、吞吐和 ETA。
- 训练恢复必须保存 scheduler、两条 query stream、阶段、RNG、encoder/latent lifecycle 与 optimizer 状态，恢复后与未中断轨迹语义一致。

### R4. 统一架构内的部署

- 所有变化通过 NVIDIA `MethodDefinition` 私有表达和必要的通用合同扩展进入现有 runner/checkpoint/package/viewer；不得恢复第二套训练、导出或 renderer 路径。
- `ScatteringPackage` 必须能够承载并由通用 loader 绑定 latent texture/resource；`prepare(context, material)` 使用 UV footprint 选择/过滤 latent，并复用 view-conditioned frame/sampler state。
- 同一 packed FP16 weights/latent asset 在 SlangPy、Falcor package parity、PT 和 deferred 中执行同一数学实现；NVIDIA 的 matched `sample/pdf` 不能借用 reference sampler，也不能加入未登记的 cosine safety lobe。
- viewer 必须落实当前双 `ComparisonSlot` 架构。neural slot 的 path-tracing mode 在每个命中点通过同一 package binding 调用 `prepare/sample/pdf/evaluate`，不得继续用 source reference PT 冒充 neural PT；deferred 与 PT 都必须把真实 UV footprint 传给 `prepare`。

### R5. 验证与报告

- 验收首先判断 correspondence、数学、梯度、lifecycle、恢复和跨后端 parity，不使用用户未确认的绝对质量数值作为 kill gate。
- 报告训练相对初始化的收敛轨迹以及 directional/energy/visual quality、sample/pdf correctness、时间、显存、package bytes 和 shader cost；observed quality 不反向改写方法合同。
- 无作者训练资产/精确 reference graph 的论文图像不作为可伪造的数值复现结论；本项目只对实际 source snapshots 与 reference 负责。

## 验收标准

- [x] A1（需求交付；来源：用户“审查哪里不忠实”与一手论文）：correspondence 表逐项关联论文页/补充材料 listing、实现符号、测试和显式 adaptation；不存在未分类差异。
- [x] A2（理论/语义正确性；来源：论文正文 §4、补充材料 Listings 1–4）：exact-vector tests 锁定 z8、learned frames、网络输入顺序、层数/宽度、activation/output、sampler raw layout/warp 和 response-measure adapter。
- [x] A3（需求交付；来源：用户确认完整方法与论文正文 §5/补充材料 §1）：formal recipe 静态校验并实际驱动独立的 65k evaluator/sampler online batches、300k global schedule、前 20k 连续 mollification和规定 Adam 参数；smoke identity 不能冒充 formal。
- [x] A4（理论/语义正确性；来源：论文正文 §5.2 的 simultaneous optimization/detach 语义）：梯度测试证明 evaluator loss 更新 encoder/latent/evaluator，sampler KL 只更新 sampler head且不改变共享 latent；两条 batch stream 不复用 query。
- [x] A5（需求交付；来源：用户确认完整方法与论文正文 §5.1）：encoder bootstrap → latent materialization → latent finetune 可恢复；uniform LayerStack adaptation 与 spatial/hierarchical runtime contract 都有测试。
- [x] A6（需求交付；来源：用户指定当前架构与补充材料 Listing 1）：package roundtrip 和 viewer loader 真实绑定 latent resources；UV/gradient/LOD 与 stochastic MIP selection 在 PT/deferred 可达，失败按 package 合同拒绝而非 fallback。
- [x] A7（数值实现正确性；来源：补充材料 functional FP16 pseudocode、scattering ABI与数学不变量）：FP32 training core、FP16-packed Slang/Falcor、`evaluate/sample/pdf` 与 checkpoint restore parity 通过；容差在 formal 前由数据类型/独立 oracle calibration 冻结；sample→pdf、null event、hemisphere support 与 estimator weight 正确。
- [x] A8（需求交付；来源：用户“持续推进到做完”、论文 300k protocol 与 2026-08-27 的 200k 记录决定）：短程 live GPU smoke 有有限 loss/gradient并显示吞吐/ETA；正式 recipe 保持 300k 配置身份，但本次经验结果只使用并登记 step 200k checkpoint。报告必须排除更晚日志、明确不声称完成 300k，并完整登记 200k 训练轨迹、耗时、吞吐、显存和产物身份；quality/time/memory 观察值只报告，不决定完成。
- [x] A9（需求交付；来源：用户明确要求 neural 方法走 PT 与当前 `viewer_spec.md`）：同一 neural package 在两个对称 slot 中可分别运行 PT/deferred；PT 实际命中 NVIDIA `sample/pdf/evaluate`，并用受控 scene 证明不是 reference transport。capture/replay 分别记录两个 slot 的 package/mode/status。
- [x] A10（理论/工程正确性；来源：Trellis 统一 pipeline 与 repository/viewer 合同）：全量 unit/GPU/viewer Release build、package tamper matrix、`git diff --check` 与 Falcor clean 通过；开发机无法覆盖的验证明确交接，不宣称已完成。

## 范围外

- 不新增第二个产品 neural candidate，不恢复已删除的旧 runner/方法/格式或兼容 alias。
- 不把 NVIDIA 原论文 MaterialX 资产、作者未公开的训练实现或论文图像数值伪装成本项目可复现资产。
- 不以本任务为由扩展 UE、多灯 scaling、PT 方差研究或修改锁定的 Falcor 上游源码。
- 不复现作者自定义 DXC tensor-core intrinsic 的未公开实现；以补充材料公开的 regular FP16 functional path 为 correctness 目标，并只报告本项目实际测得的 shader cost。
