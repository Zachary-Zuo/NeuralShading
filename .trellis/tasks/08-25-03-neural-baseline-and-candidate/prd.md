# 03 Neural Baseline 与候选选择

## 它是什么

本任务在 `01` 的公共散射数学和 `02` 的冻结训练数据入口上，建立第一个可信的 neural evaluator / learned sampler 研究闭环。原始目标是先忠实复现 NVIDIA learned-frame direct evaluator、9 参数解析 proposal 及论文训练 lifecycle，再实现 exact top-interface core + positive neural residual 候选；随后在相同数据与评测协议下比较各方法。

> 状态更正（2026-08-27）：下文 R2 仍是未完成的需求，不是既有实现的完成声明。现有实现只对齐主要网络形态，训练采用冻结 LayerStack HDF5、25k steps、每条 route 每 step 1,024 个方向查询；论文采用 GPU online reference、300k iterations、每次两个 65k batch。现有 run 必须登记为 `nvidia-frame-two-lobe-layer-stack-budget-adapted-v1` 离线诊断，不能称为忠实复现、原训练协议或 paper-scale baseline。

“复现成功”和“方法选择”是两个独立结论。前者只回答实现是否对应原方法、训练是否稳定收敛，不由收敛后的绝对质量决定；后者才使用按材质结构分组的质量、时间、内存与 sampler 方差证据描述 Pareto。若不同材质结构上的优劣不一致，结论必须保留这种条件性，不能用一个全局阈值或单一分数抹平。

## 前置事实与冻结输入

- `01` 已归档于 `.trellis/tasks/archive/2026-08/08-25-01-reusable-scattering-math`；公共 cosine/LTC/GGX/tilted-cosine/non-centered-GGX/mixture、三态 sample 结果和 solid-angle measure 是唯一数学实现。
- `02` 已归档于 `.trellis/tasks/archive/2026-08/08-25-02-mollification-data-adequacy`。旧 v5 adequacy 决定固定为 `use-mollification-supplement-v1`，不能由本任务重解释。
- base corpus ID 固定为 `0513d0c837b109f74cbf6fd4f811e05c6bc68c02226bd6d443f3225ef5dd64b7`；mollification composite corpus ID 固定为 `f6931474890ab7642f244b84df2736e2a5fc1f9e169b5f7a620494184d99e4f3`；training entry ID 固定为 `47ef20138007703f2d1b644bcb4ca4b084001da4ec975f1b712587d3e7e35a89`。
- training entry 的有限离线 curriculum 固定为四个正半径 level；`t<0.875` 取最近 level，`t≥0.875` 读取 shard 内的 base-v5 0° `source_response`。本任务不得扫描目录猜测数据或自行插值连续半径。
- 首个事件域固定为 LayerStack upper hemisphere、reflection-only、non-delta；response 监督量为线性 RGB `f·|cos(wi)|`，运行时 `evaluate()` 输出线性 `f`。
- 当前机器是完整开发机：RTX 4090、`neural-shading` 环境、Falcor build 与原生 Windows 均存在。`slangpy==0.43.1` 已可在统一环境导入；启动时 D3D12 Agility SDK warning 不影响 SlangPy import，但 Falcor/viewer 路径仍由各自测试验证。六个锁定 upstream 在本任务正式收尾前重新检查干净状态。

## 需求

### R1 单一 Slang 前向

- learned frame、prepare/evaluate、exact top core 组合、NVIDIA/LTC proposal decode 与 sampler `sample/pdf` 的生产前向只写一份 Falcor-free Slang。
- SlangPy 训练与 Falcor GPU 测试编译同一 core；Torch 只承载 loss、optimizer、统计与独立 oracle，不实现可被导出或运行的第二套模型前向。
- 首先验证 SlangPy 0.43.1 的 autodiff、Torch interop、Falcor 2024.1.34 双编译和固定网络吞吐；语法差异用稳定测试锁定。无法安全修复的双编译失败才构成 blocker，不静默回退。

### R2 方法对应关系与原规模 NVIDIA baseline

- 每个方法先建立 `method correspondence`（方法对应关系）清单：一手论文/补充材料中的结构与训练定义、项目实现位置、方向/余弦约定、允许的接口适配、实际偏差及其独立方法 ID。`D:\01_Workspace\Real-Time Neural Appearance Models` 只读作为二手复现对照，不作为正确性权威，也不得复制其中未经验证的 GT、loss 或采样实现。
- NVIDIA baseline 的主实现使用一手 supplemental 定义：8 维 latent；无 bias/activation 的 `8→12` frame layer；两个只由 latent 决定、按 `N=normalize(s.xy,s.z+1)`、`T=normalize(t.x+1,t.yz)`、`B=cross(N,T)` 构造且明确不做 Gram–Schmidt 的 learned frame；`z + T_j wi + T_j wo` 的 20 维 decoder 输入；论文报告的最大原规模 `3×64` hidden ReLU MLP；`exp(raw-3)` 的 cosine-weighted response 输出；以及以 `z + wo` 为条件的 `3×32 → 9` sampler head。不得用自定义 17 维特征、view-conditioned/正交 frame、`softplus`、output floor 或缩小网络继续沿用 baseline ID。
- sampler 必须预测 9 参数 `{w_d, mu_d.xy, w_s, alpha.xy, rho, mu_s.xy}`，解析执行 tilted-cosine diffuse 与 non-centered anisotropic GGX specular mixture。
- evaluator 复现原方法对 cosine-weighted response 的 log-space L1 与 directional mollification；公共 backend 通过显式 adapter 满足 `evaluate()` 返回裸 `f`，并锁定 `evaluate()*cos == response`。为适应本项目 noisy reference 而增加的 loss 项必须作为显式 adaptation 做 matched ablation，不能混入“原方法复现”身份。原 baseline 的 evaluator 与 sampler 联合训练，sampler 相对当前 evaluator response 做 KL，且 KL 对 latent detach；“冻结 evaluator 后再训练 sampler”只属于后续 matched sampler comparison，不替代原方法复现。
- 原规模 baseline 是正式 baseline 和后续 MethodBundle/viewer 的输入，不是只跑一次的低优先级 diagnostic。标量 Slang 上的实际成本与 runtime class 如实记录；速度慢、超过研究软线或暂不具备 cooperative-vector 加速，都不构成复现失败。
- 缩模只能在原规模复现完成后作为独立方法身份和独立实验提出；不得替代原规模实现、训练或 viewer 证据，也不是本任务的必做交付。

### R3 exact-core positive-residual 候选

- evaluator ID 固定为 `core-frame-neural-v1`：复用同样 learned-frame direct MLP 主体，加入 exact top-interface reflection core，并以必选 positive neural residual 表达其余路径。
- 预测形式不得出现 `clamp(core + signed residual, 0)`、prediction clamp、`correction=none` 或 lobe-only fallback；loss 作用于最终 `f_hat·cos`，不把 noisy `reference-core` 截断成监督标签。
- CompiledMaterial 仍只保存固定 top-interface core、16 维 latent 与 normalization/flags；源参数编辑后通过 target-visible offline cook 重拟合，不实现 feed-forward source compiler。

### R4 matched sampler 轴

- sampler 轴固定为 `nvidia-diffuse-ggx9` 与 `ltc-k2`；二者都加入固定 `epsilon=1/32` cosine safety component，保证有效上半球 full support。
- sample 与 pdf 必须使用同一解析 proposal 和同一 prepare state；非 delta 权重只由 `evaluate(wi)·wi.z/pdf(wi)` 形成。
- NVIDIA reflection proposal 的 below-surface 结果必须进入显式 null mass；LTC 不允许制造隐式 rejection/resampling。两种 sampler 都必须通过连续 PDF + null 质量归一化、sample→pdf 重求值、直方图和同 evaluator MC 无偏性门。

### R5 训练稳定性与 matched 比较

- 每个 evaluator 使用自己声明且经过审计的方法形态；matched 指相同的 data entry、训练/validation/test 划分、方向 convention、reference target、训练预算类别、checkpoint 规则和评测 protocol，不要求不同方法具有相同 latent bytes、MLP 宽度或运行成本。
- “稳定收敛”使用与绝对材质质量无关的证据判定：全程 loss/梯度/权重有限；验证目标相对初始化产生有统计支持的改善；后期验证轨迹没有可信发散；best checkpoint 可恢复并复算相同结果。默认每个方法只运行预先冻结的主 seed；额外 seed 只用于轨迹异常诊断或用户明确要求的置信度补充，不阻塞尽快生成可见结果。已经完成的第二个 baseline seed 只作额外佐证，不要求 candidate 与 sampler 重复同样次数。
- checkpoint 只由 validation 选择，test 不参与训练、早停、超参修改或复现状态判定。某方法稳定收敛但最终质量较低时，复现状态仍为成功，质量结论记录为该方法在当前数据/预算下的观察结果。
- NVIDIA 复现先完成原方法的 evaluator/sampler joint stage；随后为了比较 sampler 轴，另从冻结 evaluator 训练 matched GGX9/LTC heads，二者使用独立 adaptation identity。exact-core evaluator 同样在冻结后训练两种 matched heads。所有 sampler stage 都必须提交独立的有限梯度、训练下降与后期不发散证据。
- 最终报告必须分别记录 `implementation_status`、`convergence_status` 与 `comparison_outcome`，并追溯 checkpoint/compiled-asset/Slang implementation hash、质量、sampler 方差、静态成本、实测单 query 时间和 paired bootstrap CI。任何一个字段都不得由另一个字段代替。

### R6 质量、成本与工作流证据

- 不设置跨材质统一的绝对质量通过线。directional L1、signed energy ratio、`E_core/E_ref`、峰值/尾部、最差 state、state paired bootstrap、leave-one-state-out、reference SE 和视觉差异必须完整报告，但只用于描述质量与相对比较，不决定 baseline 是否复现成功。
- 质量报告按源表示和结构特征分组；当前 LayerStack 至少区分单界面/多层、core coverage、粗糙度与 grazing 等已有 breakdown。不同组出现不同 Pareto 前沿时保留分组结论，不强制生成一个全局 winner。
- 成本合同只要求执行、状态和访存静态有界；`C_prepare/C_eval`、state、资产、共享权重和 viewer 实测时间如实记录。超过研究软线只改变 `deployment_candidate/runtime_class` 声明，不阻止导出、viewer 加载或复现验收。
- 数学正确性仍使用必要数值容差，例如 PDF 归一化、sample→pdf 重求值、SlangPy/Falcor/half-packed parity。它们验证同一数学实现，不是材质质量阈值。
- 验证 target-visible offline cook：shared weights 冻结后，给受支持 state 的 train/reference queries 只拟合 latent/compiled asset，再在未见 query role 上评测；不得读取 test target 做拟合。

### R7 交付边界

- 本任务交付训练/评测框架、唯一 Slang method core、原规模 baseline 与候选 checkpoint、packed compiled-material set、复现报告和比较报告。
- `04` 拥有通用 MethodBundle exporter/loader/specialization，`05` 拥有 viewer deferred/PT 与 capture。03 必须交付可由这条通用路径直接消费的原规模资产；父任务收尾时必须在 viewer 中真实显示原规模 baseline 和保留的候选，不能因其成本分类而改用缩模替身。
- 运行 checkpoint/report/compiled assets 进入 `artifacts/`；根仓库只提交源码、测试、配置、schema、稳定中文文档与实验登记，不提交 checkpoint/HDF5。

## 验收标准

- [ ] SlangPy/Falcor 双编译与 autodiff spike 通过；同一 Slang evaluator 前向、梯度和 half-packed parity 有固定测试，仓库中不存在第二套 Torch 生产前向。
- [ ] NVIDIA baseline 的 method-correspondence 清单逐项闭合：原规模网络、learned frame 条件、方向输入、输出激活、loss/mollification、9 参数 sampler 与 detach 边界均有一手来源、唯一 Slang 实现和独立 oracle/测试；项目适配使用独立身份或显式登记。
- [ ] 原规模 NVIDIA evaluator/sampler 和 `core-frame-neural-v1` 候选均完成主 seed 真实训练；loss、梯度、权重全程有限，validation 相对初始化改善且后期无可信发散，checkpoint 可恢复复算。验收不限制收敛后的绝对质量，也不以追加 seed 阻塞可见结果。
- [ ] `core-frame-neural-v1` 无 clamp/lobe-only fallback；其实现正确性与收敛状态独立于是否在质量比较中胜出。
- [ ] 两个 sampler 对两个 evaluator 均完成 matched 2×2；PDF/null 归一化、sample/pdf 重求值、histogram 和同 evaluator MC 无偏性全部通过。
- [ ] 报告分开给出 implementation、convergence、quality/cost comparison；质量部分含按材质结构分组的 paired bootstrap、leave-one-state-out、signed energy、core coverage、sampler 方差和成本，不使用统一绝对质量 kill gate。
- [ ] training entry→curriculum batch→Slang forward→checkpoint→compiled-material→Falcor parity 的端到端 identity 可追溯，0° 明确回到 base v5。
- [ ] 原规模资产可由 `04/05` 的通用 MethodBundle/viewer 路径加载和显示；成本分类只影响声明和排序，不影响可见性，也不允许用缩模资产冒充。
- [ ] 完整 unit、SlangPy、Falcor GPU/reference、训练 smoke、正式 run validator、repository policy 与 upstream cleanliness gate 通过。
- [ ] 完成独立 `trellis-check`、长期 spec 判断、scoped local commits 和归档后，父任务才进入 `04`。

## 不在范围内

- 通用 MethodBundle loader/exporter、viewer renderer path、capture/replay 和旧方法删除。
- transmission、delta、volume boundary、新 source family、P2 全量 250+ state 泛化或 feed-forward source compiler。
- cooperative-vector 性能复现；当前先在标量 Slang 上实现、显示和如实测量原规模网络。
- 为满足软成本线而缩小 latent、网络宽度、层数或特征的派生 baseline；它只能在原规模复现后另立方法身份。
- 多灯 scaling、环境积分、最终 PT 场景方差和 UE 集成；本任务只验证局部 proposal 数学和单次查询成本。

## Planning 结论

2026-08-26 根据用户反馈在原任务内修正规划：撤销由旧实验观察值晋升而来的 Q1 绝对质量门，撤销“先缩到 deployment-matched 规模再算 baseline”的路线。后续正式运行以方法对应关系和稳定收敛为复现门，以分组质量—时间—内存证据作比较结论；已经完成的旧 run 保留为诊断证据，但不能继续驱动重跑或冒充原方法复现。
