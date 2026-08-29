---
paper_id: "bitterli-2026-taming-optimization-variance"
title: "Taming Optimization Variance in Compact Neural Shading Networks"
authors: "Benedikt Bitterli, Petrik Clarberg, Chris Cummings, Aaron Lefohn, Steve Marschner, Jan Novák, Fabrice Rousselle, Andrea Weidlich, Tizian Zeltner"
year: "2026"
venue: "ACM SIGGRAPH Conference Papers '26"
doi: "10.1145/3799902.3811231"
report_status: "evidence-reviewed"
main_source: "https://research.nvidia.com/labs/rtr/publication/bitterli2026taming/bitterli2026taming.pdf"
supplemental_status: "available"
official_code_status: "audited"
official_code_commit: "305b4b9c12e679398c487603dd8245c3f348526c"
author_worker: "/root/taming2026"
reviewer: "/root/nbrdf2021"
last_verified: "2026-08-29"
---

# Taming Optimization Variance in Compact Neural Shading Networks

- Query：完整重建论文的 multi-instance/pruning/batch-resize 算法、compact neural material 配置、正式实验、负结果与对当前 NVIDIA 复现的影响。
- Scope：mixed（正文/补充/作者页/官方代码的 external evidence，加 NeuralShading 当前 correspondence 的 internal evidence）。
- Date：2026-08-29。
- Related specs：本任务 `research/evidence-policy.md`、`research/report-template.md`、`research/dispatch-brief.md` 与 `.trellis/spec/project/research-execution.md`；来源文件与代码 pattern 集中列于 §2、§11、§16。

## 1. 研究对象与报告边界

这篇论文研究的首要问题不是发明一种更大的 neural material representation，而是让**几乎没有冗余容量的小型网络更可靠地训练成功**。作者观察到，当网络刚好有能力拟合目标材质时，不同初始化与训练数据顺序可能把相同结构送入质量差异很大的局部极小值；这使单次训练结果不可预测，妨碍批量烘焙、资产压缩和 look-development 迭代。[P: Abstract；§1，pp.1–2]

论文的主贡献是一种 training-only 的 multi-instance schedule：先以不同初始化并行训练多个同构候选，按阶段淘汰 loss 较差者，同时把淘汰后释放的固定 batch budget 分给幸存者。作者把它与 successive halving 联系起来，但与常见做法不同，不给幸存者增加训练 step，而是增大每个幸存实例的 batch size，使每一步所有实例合计处理的 sample 数保持常数。[P: §2–§3，pp.2–4]

论文随后以 2024 年 *Real-Time Neural Appearance Models* 的 encoder–latent–decoder 材质表示为 case study，并另外提出三项会改变拟合条件或表示行为的设计：稳定的 shortest-arc half/difference 坐标、power-mapped L1 loss，以及 LeakySmeLU activation。[P: §4–§5，pp.5–8] 因而本报告严格区分：

- **优化方差降低**：相同部署表示、相同总 network-evaluation budget 下，改变候选的训练与选择过程；
- **representation / conditioning quality 改变**：改变 direction inputs、activation 或 loss，使单个候选的可表达函数、梯度场或可见伪影改变；
- **runtime representation**：训练结束仍只部署一个 latent + 小 decoder；64 个候选不是运行时 ensemble。[P: Fig.1；§3；§6]

本文属于 `local material`，目标是给定着色点及一对方向求 RGB reflectance/BRDF。它不学习 scene-level global illumination、visibility、light transport、体传输或随时间变化的场景状态，也不把 matched `sample()/pdf()` 作为本文实验对象。官方仓库继承了 sampler/auxiliary decoder 等更大系统能力，但本文关于 optimization variance 的正式结果只支撑 compact reflectance model 的 BSDF fitting。[P: §4–§6；C: `configs/default.json`]

本报告覆盖作者发布的 SIGGRAPH 2026 正文、7 页 supplemental、NVIDIA Research/作者项目页与 NVLabs 官方代码快照；不会把公开代码当前 default 自动等同于论文全部正式配置，也不会把当前 NeuralShading 项目的 NVIDIA 2024 复现结果倒灌为论文事实。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---:|---|---|
| Main paper `P` | [NVIDIA Research PDF](https://research.nvidia.com/labs/rtr/publication/bitterli2026taming/bitterli2026taming.pdf)，9 页；DOI `10.1145/3799902.3811231` | 2026-08-29 | SHA-256 `EB979B03328353C03B7606DD03462DE9F10873F777764E965C70C21DB61A6F5C` | 方法、正式实验、公式、图表、作者结论；已逐页渲染并视觉核对 |
| Supplemental `S` | [NVIDIA Research supplemental PDF](https://research.nvidia.com/labs/rtr/publication/bitterli2026taming/bitterli2026taming_supplemental.pdf)，7 页 | 2026-08-29 | SHA-256 `93BA59966A61628B6451FA9CEB582B8614673A50D6EAD7EC8985267836389F55` | pruning factor、shared data、optimizer 与 loss/activation/input 的定量消融；已逐页渲染并视觉核对 |
| Official code/config/data `C` | [NVlabs/neuralappearance](https://github.com/NVlabs/neuralappearance)，Apache-2.0 | 2026-08-29 | commit `305b4b9c12e679398c487603dd8245c3f348526c`；提交时间 2026-08-05；父提交 `b59ac4c` | 算法实现、default 与 paper example config、数值类型、data generation、export；仓库不含论文 15 个正式材质 |
| Author page/talk/correction `A` | [NVIDIA project page](https://research.nvidia.com/labs/rtr/publication/bitterli2026taming/)、[Jan Novák project page](https://www.jannovak.info/publications/NAP-smallnets/index.html)、[SIGGRAPH schedule](https://s2026.conference-schedule.org/presentation/?id=papers_1751&sess=sess139) | 2026-08-29 | web locator | 书目信息、官方入口、外部摘要；没有发现正式勘误 |
| NeuralShading evidence `N` | `.trellis/tasks/archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md`、`current-fidelity-audit.md`；`.trellis/tasks/archive/2026-08/08-27-reference-material-candidates/research/nvidia-2026-materialx.md`、`report.md` | 2026-08-29 | repository files | 仅用于 §14 判断当前 2024 复现与本文的差异、公开资产边界；不作为论文事实 |

代码锁定说明：审计日的 `HEAD` 相对公开快照 `b59ac4c`（author date 2026-07-13；commit date 2026-07-17）只修正 7 个 Slang 文件名大小写，没有实质算法变更。因此本报告固定 `305b4b9c…`，并同时记录首次公开快照以便长期追溯。[C: GitHub commit metadata/diff `b59ac4c..305b4b9c`]

第一方页面还给出“run-to-run variance 降低 88%、average loss 降低 38%”的宣传摘要。[A: SIGGRAPH schedule, presentation description] 正文与 supplemental 没有定义能直接复算这两个 aggregate 的公式或对应表项；本报告不把它们替换为正文结果，并将其列作 evidence gap。

没有找到独立 author talk、slides、勘误或正式 material asset download。官方代码仓库仅带 Bark、FauxLeather、PatternedMetal 示例；正文使用的 9 个 da Vinci 烘焙材质与 6 个内部 layered material 不在仓库中。[C: repository `assets/materials/`; N: `nvidia-2026-materialx.md`]

## 3. 原论文的问题、假设与贡献边界

### 3.1 作者定义的问题

作者把适用场景限定为 baking/compression：网络本来就要专门拟合已有训练资产，因此这里优先追求“任意一次训练都有高概率得到好拟合”，而不是未见数据 generalization。[P: §1, p.2] 小网络的 loss landscape 被作者描述为比 over-parameterized network 更不平滑、更不宽容；可行的高质量 basin 较少，结果更受初始化与数据顺序影响。[P: §1]

### 3.2 主假设

1. 同一结构的候选优劣排名在训练早期 warm-up 后就足够稳定，故可以在结束前淘汰差候选。[P: §3.1, Fig.3]
2. 若每一步所有候选的总 sample budget 固定，则候选数和单候选 batch size 可以反向配平：早期小 batch/多初始化承担 exploration，后期大 batch/少候选承担 exploitation/convergence。[P: §3.2]
3. 初始权重对 run-to-run variation 的影响比不同 training data 大，因此多个实例共享同一批 query 既能节省数据生成，还可能让候选 loss 更可比较。[P: §4, pp.6–7；S: §B]

### 3.3 作者声称的贡献

- fixed network-evaluation budget 下的 multi-instance + greedy pruning + batch-size annealing；
- 对 15 个材质和 64-initialization 规模的排名、保留率、最终 loss 与 wall-clock 分析；
- stable shortest-arc half/difference direction parameterization；
- power mapping `M_pow(x)=n(x^(1/n)-1)`，正式取 `n=3`；
- 用带负半轴小斜率的 LeakySmeLU 避免 compact ReLU decoder 的 specular faceting。[P: Abstract；§3–§6]

### 3.4 贡献不覆盖的相邻问题

- 算法不提高最终单个网络的参数容量；即使 run-to-run variance 接近大型网络，小网络的最低 loss 仍可能更高。[P: Fig.1]
- 不是用 ensemble 平均提升推理质量；训练结束只保留一例。[P: §2–§3]
- 不是超参数搜索或 architecture search；正式对照冻结结构、optimizer 和其余 hyperparameters，只改变实例/批量 schedule。[P: §4.2]
- 不保证 greedy culling 一定保留全局最佳初始化；论文直接测量其误删概率。[P: §4.3, Table 1]
- 不证明 batch noise 关于 local minima 的因果机制；“小 batch 易探索、大 batch 易收敛”是设计动机，实验证据是 schedule 对照而非 loss-landscape 因果实验。[P: §3.2；§4.3]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | 空间位置 `x` 对应的原生表面属性；每层包含 albedo、normal、tangent、roughness、weight。正式资产是单 UV tile 的 4K SVBRDF 或多层材质 | P 未给总 feature count；C 每层打包 `3+3+3+2+3=14` 个标量，另带 mip scalar | P: §4.1, Fig.4；C: `neuralappearance/datagen/reference_materials.py:15–17` |
| Training query | `(x, ω_i, ω_o, native attributes, f(x,ω_i,ω_o))` | `ω_i,ω_o` 在着色局部坐标；`f` 为 RGB | P: §4.1, Fig.2 |
| Runtime query | latent code 与局部 `ω_i,ω_o` 进入 decoder | 两个 learned shading frame；方向特征随 parameterization 而变 | P: Fig.2；§5 |
| Direction coordinates | direct `(ω_i,ω_o)`；Rusinkiewicz `(ω_h,ω_d)`；稳定 `(ω_h,ω_d')`；最终论文建议同时输入 `(ω_i,ω_o,ω_h,ω_d')` | 单位向量；只在有效反射半球的 BRDF query 上评估 | P: Eq.1–2, Fig.6；S: Table 6–7 |
| Output quantity | 非负 RGB BRDF/reflectance value `f̂` | 正文定义 `f̂=exp(z)`；没有声称输出已乘 cosine、PDF 或入射辐射 | P: §5 “Loss function”, Eq.3–4 |
| Validity/domain restrictions | spatially varying local surface reflectance；position 已烘焙到 latent；无 visibility、shadow、interreflection 或 participating media | local material domain | P: §4–§6 |

### 4.1 方向变换的精确定义

旧 Rusinkiewicz difference vector 为：

```math
\omega_d=R_y(-\theta_h)R_z(-\phi_h)\omega_i .
```

当 perfect reflection 使 `ω_h=[0,0,1]^T` 时，`φ_h` 未定义，输入在该点附近快速变化，然而目标 BRDF 可能几乎相同。[P: Eq.1, Fig.6 top]

作者改用把 `ω_h` 以 shortest arc 旋到 z 轴的 closed form：

```math
\omega_d'=\omega_i h_z+(v\times\omega_i)+\frac{(v\cdot\omega_i)v}{1+h_z},
```

其中代码将 `v` 构造成 `(h_y,-h_x,0)`。[P: Eq.2, Fig.6 bottom；C: `neuralappearance/model/half_diff_parameterization.slang:58–75`] 这消除了 perfect-reflection singularity，但作者也观察到 normal mapping 会把实际 lobe 旋离 canonical half/difference alignment；所以最终不是只用 `ω_d'`，而是同时给 decoder direct 与 stable half/difference 特征。[P: §5, pp.7–8]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

论文继承 Zeltner et al. 2024 的两段式材质压缩：

1. 训练期，从 `x` 读取 native surface attributes；encoder `E_φ` 把这些属性映射成低维 latent `z_x`。
2. latent 另经一个线性 projection 预测两个 local shading frames；每个 frame 把 `ω_i,ω_o` 变换到材质自适应坐标。
3. decoder `D_θ` 接收 `z_x` 与每个 frame 的 direction features，输出 RGB reflectance。
4. 对 online reference query 的 `f(x,ω_i,ω_o)` 计算 mapped L1 loss，同时更新 encoder、frame projection 与 decoder。
5. 训练结束将 encoder 的输出烘焙成 spatial latent texture；部署时 encoder 与 native source graph 不再执行，只保存 latent、frame projection 与一个 decoder。[P: Fig.2；§4.1；C: training lifecycle]

Multi-instance 只复制 trainable encoder/decoder 候选。每个实例独立初始化，但正式实验各候选共享相同 query；P 只说在 schedule boundary 按 loss 排序，C current 具体使用最新完成 iteration block（通常最多 64 steps）内各实例的 mean training loss，再只复制/保留最小 loss 的候选索引并扩大 batch。最终烘焙一套 latent 并部署一例。[P: Fig.2–3；§3–§4；C: `train.py:445–519`, `neural_material_model.py:304–324`]

### 5.2 持久化表示

| 项 | 论文正式描述 | 代码快照 | 证据边界 |
|---|---|---|---|
| Spatial latent | encoder 输出被 baked，运行时替代 native attributes | 1 个 mip、8 channels、归一化到 `(-1,1)` | P: Fig.2；C: `configs/default.json:93–104`；P 未报告 mip 数与存储格式 |
| Learned frames | 每个 latent 产生 2 个 shading frames | 8→12 无 bias 的线性层；分别 normalize normal/tangent，以 cross product 得到 bitangent | P: §4.1；C: `model/rotation.py:22–66`, `model/rotation.slang:56–121` |
| Decoder weights | 每个资产最终只保留 1 个 compact decoder | Cooperative Vector / half network weight path | P: §3, §6；C: network/export source |
| Encoder | 只存在于 material encoding/training | 训练后先 bake latent，随后可 direct finetune | P: Fig.2；C: `train.py` |
| Quantization | 未报告 | trainable network weights 为 half，optimizer 保留 float copy；latent texture的最终序列化格式不能由论文确认 | C: `train.py:272–313`, `full_precision_optimizer.slang:46–60` |
| Sampler/aux | 不属于本文正式 optimization-variance 结果 | 仓库能另训 GGX-mixture sampler 和 auxiliary decoder | C: `configs/default.json:141–160`；本文不可据此补写性能 |

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Encoder（正式 §4 实验） | 每点 native surface attributes | hidden `64→64→64`；输入/输出层尺寸随 source feature count | P 只说 Glorot 初始化；C 为 hidden LeakyReLU | 8D latent | per-material training network，部署丢弃 | P: §4.1；C: `configs/default.json:106–113` |
| Frame projection | latent 8D | linear `8→12`，无 bias；12 数值形成 2×(normal,tangent) | normalize `n,t`，`b=n×t` | 2 learned frames | 与部署 decoder 一起 per material | P: §4.1；C: frame implementation |
| Small decoder（正式） | 8D latent + 两 frame 的方向特征 | hidden `[16,16]`，RGB output | 正式最终主张 LeakySmeLU；正文 output `exp` | RGB `f̂` | per-material | P: Fig.1, §4.1, §5, Fig.9 |
| Large decoder 对照 | 同上 | hidden `[128,128,128,128]` | 与比较配置匹配 | RGB `f̂` | per-material | P: Fig.1；C: `configs/single_instance_training_large.json` |
| Released default decoder | 8D latent + `WhWdZiZo` | hidden `[16,16]` | `SmeLU`; `ScaledSigmoid(scale=1e5)` | RGB | per-material | C: `configs/default.json:115–138` |

Figure 1 报告 large decoder 约 54k parameters、small decoder 精确 947 parameters。[P: Fig.1 caption/labels] 论文没有逐项展开参数计数；但用其最终 full direction input 会得到与标注唯一自然匹配的算术：两 frame 的 `(ω_i,ω_o,ω_h,ω_d')` 共 24 scalars，加 8 latent 得 32D；MLP 参数为 `(32×16+16)+(16×16+16)+(16×3+3)=851`，无 bias 的 frame projection 为 `8×12=96`，合计 `947`。[I，基于 P/C 结构算术] 大网络相同算法得 `54,243`，与 “54k” 一致。

这个算术也是重要 correspondence 证据：released paper examples 继承 default 的 `WhWdZiZo`，每 frame 只给 `ω_h,ω_d,z_i,z_o` 共 8 scalars，按同一语义参数计数 small/large 会变成 `819/53,219`，不能精确对应 Figure 1 的 `947/54k`。因此 README 所称 Figure 1 configs 能重现**训练形态**，不能在没有额外 override 的情况下重现论文标注的精确网络。[C/I，详见 §11]

### 5.4 条件化、坐标变换与物理先验

- learned frames 是 latent-conditioned 坐标变换，不是解析 BRDF core；network 仍直接回归 RGB reflectance。[P: Fig.2；§4.1]
- half/difference 是输入 parameterization，不是 sampling distribution。stable variant 修复坐标 singularity，但不保证 normal-mapped lobe 对齐。[P: §5]
- power mapping 只改变 loss domain；不增加 runtime input 或参数。[P: Eq.3–4]
- LeakySmeLU 以 `C1` 二次过渡换掉 ReLU 的导数突变，目的之一是减少锐利高光的 piecewise-linear facet；它改变 decoder function family，不应归入纯 training schedule。[P: Eq.5, Fig.9]
- 论文没有 analytic closure、reciprocity/energy-conservation penalty、Fresnel/lobe decomposition、visibility factorization 或 transport decomposition。不能从 layered asset 名称推断网络显式表示 layer physics。

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset/source assets/scenes | 9 个 da Vinci objects：Birdcage、Chandelier、Hammer、Lantern、Mirror、Palette、Chair、Scales、Table。每个 unwrap 成单 UV tile，bake 4K maps 得一个 reference SVBRDF，再把 MDL 翻译为 USDPreviewSurface | P: §4.1, Fig.4 |
| Layered materials | Scratched steel、Bumpy plastic、Oxydized metal（保留作者拼写）、Gold and ceramic、Brushed brass、Glazed ceramic。base 上叠 glazing/stain/dust 等层，各层有参数、纹理和多张 normal map | P: §4.1, Fig.4 |
| GT/reference renderer or measurement | 正文定义 query target 为 source material 的 RGB `f(x,ω_i,ω_o)`，但未报告 renderer/backend、每 query MC spp 或 error tolerance | P: §4.1 |
| Train/validation/test split | 主 optimization 实验以每个固定材质拟合为目标；未报告 spatial/directional holdout split。Supplemental 图像指标使用共享的 sphere + plane test scene | P: §4；S: §D |
| Spatial sampling | P 未报告分布。C 在 UV 上用平方网格 stratification，并 snap 到 texel center；batch 必须为 perfect square | C: `configs/default.json:164–168` 与 data-generation source |
| Directional sampling | P 未报告正式 mixture。C default：95% `UniformRusinkiewicz`，5% `UniformWiImportanceWo`，后者的 `ω_o` 由 reference importance sampler 产生 | C: `configs/default.json:174–187` |
| Filtering/LOD/footprint | P 未报告。C default 只学习 1 mip；若多 mip，prefilter 64 samples，reference diagnostics 256 | C: `configs/default.json:200–206` |
| Augmentation | P 未报告。C default 在 encoder training 中对 material parameters 与 target 做 RGB permutation/duplication，概率用 cosine 从 1 降到 0，历时 100k | C: `configs/default.json:188–198` |
| Distillation/teacher | 无 | P: §2 将 distillation 作为相关方法讨论，并未用于方法 |
| Online/offline generation | P 只说 samples/records，没有披露是否预生成。C 用 Falcor reference material 在 GPU 上在线生成并以 64 optimizer batches 为 block 排队 | C: `train.py:341–407` |

正式论文的两个资产族承担的是**已见材质拟合与优化稳定性**，不是跨材质 train/test generalization。每个网络被允许专门拟合自身材质，这是作者在 §1 明说的目标；因而把 Table 1–3 解读为 unseen-material performance 会改变实验语义。

代码中的 MaterialX/MDL provider 与三个公开 sample asset 证明管线可运行，但不能补上论文正式资产。由于 6 个 layered materials 未公开，主结果无法只依靠仓库做 source-identical reproduction。[C/N]

## 7. Loss、optimizer 与训练 lifecycle

### 7.1 正文正式配置

| 项 | 正式配置 | locator |
|---|---|---|
| Target/output transform | 正文令 `f̂=exp(z)`；target 与 prediction 均经 `M` 后比较 | P: §5, Eq.3–4 |
| Loss | `L=|M(f)-M(f̂)|`；proposed `M_pow(x)=n(x^(1/n)-1)`, `n=3` | P: Eq.3–4, Fig.7–8 |
| Optimizer | Adam；正文未报告 LR、β、ε、weight decay 等数值 | P: §4.1 |
| LR schedule | 未报告；只说明所有比较的 learning rate、momentum 等相同 | P: §4.2 |
| Initialization | Glorot；不同 run 只改变 model initialization seed | P: §4.1 |
| Steps | `N=100k` optimization steps，分 `P=4` 个等长 phase，即每 phase 25k | P: §4.2 |
| Instances | `64→16→4→1` | P: §4.2 |
| Per-instance batch | `1k→4k→16k→64k`；这里 `k` 在官方实现中是 1024 | P: §4.2；C scheduler parser |
| Total network evaluations | baseline 与各对照完全相同。每一步 `K×B=65,536`，总计 `6,553,600,000` 次 sample–network evaluations | P: §4.2；后一个绝对数为 I 对 P schedule 的算术展开 |
| Model selection | phase boundary 按 loss greedy cull；最终只保留 1 个 | P: §3；C implementation |
| Repetitions | Fig.1: 64 initializations；Fig.3: 100 次独立 repetition 观察 rank；Fig.5: 64 runs/curve；Table 2: 15 trials | P: Fig.1, Fig.3, Fig.5, Table 2 prose |

### 7.2 Multi-instance/pruning/batch-resize 算法重建

设 baseline 为 `N` 步、batch `B`。brute-force 训练 `K` 个完整实例会把 cost 放大 `K` 倍；若单例失败概率为 `p` 且独立，至少有一个成功的失败概率可降为 `p^K`，但这不是作者采用的最终 cost model。[P: §3]

作者将 `N` 步均分成 `P` 个 phase，初始化 `K` 个实例。一般 schedule 在 phase boundary 将实例数按 `K^(1/(P-1))` 的因子减少，使最后为 1；同时反向增大单实例 batch，令 `K_t B_t = B_baseline`。正式 `K=64,P=4` 给出淘汰因子 4。[P: §3.2, Fig.2]

```text
phase 1: 64 instances × 1,024 samples × 25,000 steps
  -> loss 排序，保留 16
phase 2: 16 × 4,096 × 25,000
  -> 保留 4
phase 3: 4 × 16,384 × 25,000
  -> 保留 1
phase 4: 1 × 65,536 × 25,000
  -> 输出该实例
```

只有 early culling 而不缩放 batch 并不能 cost-match baseline；作者推导其相对 baseline 的下界约为 `K/log(K)`（`P→∞`），且正式 4-phase schedule 的第一阶段本身已相当于 baseline 全程 cost 的 16 倍。[P: §3.2] 论文的关键不是“多训若干小 batch 网络然后免费挑一个”，而是**从一开始就把固定 sample budget 在候选之间拆分**。

官方 scheduler helper 把 `instance_schedule` 与 `batch_size_schedule` 映射到理想的等长 period；但 `train.py` 每次排入最多 64 个 optimizer step，`get_num_iterations_for_next_block()` 不会在下一个 schedule boundary 截断。按 default 100k/64-block 配置，实际切换发生在 block 结束的 iteration `25,024 / 50,048 / 75,008`，四段实际长度为 `25,024 / 25,024 / 24,960 / 24,992`，而不是恰好四段 25k。[C/I: `training/batch_instance_scheduler.py:44–56`; `train.py:390–405,685–727`] 这不会改变当前实现的总 network-evaluation 算术，因为四种状态始终满足 `K_tB_t=65,536`，100k 步仍为 `6,553,600,000` 次 sample–network evaluations。

跨过理想 boundary 的 block 完成后，代码会 drain 所有 in-flight blocks；reporter 将最新已完成 block 的逐实例 mean training loss 写入 `latest_instance_losses`，下一轮再以 `np.argsort(instance_losses)[:keep]` 对当时 active 的 encoder/decoder 同步 prune 并 resize buffers。第一次用于淘汰的 64-step block 因而覆盖 paper 理想边界前 40 步和边界后 24 步，但整块仍使用切换前的 instance/batch 状态。[C/I: `train.py:419–426,452–524,625–646,729–747`; `model/neural_material_model.py:307–336`] 共享数据路径不是“每次只生成一个 optimizer batch”，而是每个 generation block 生成 `num_batches_per_generation=64` 个 batch，各 batch 仅生成一份 instance slice，再 broadcast 到 `[64,num_instances,batch_size]`。[C: `train.py:348–379`; `configs/default.json:164–174`]

### 7.3 Power loss 的梯度设计

对 mapped L1 忽略 sign 后，output logit 的梯度尺度为：

```math
\left|\frac{\partial L}{\partial z}\right|=M'(\hat f)\hat f .
```

当 `f̂=exp(z)` 且 `M_log(x)=log(1+x)`，梯度是 sigmoid：低值附近趋近 0，高值饱和到 1。作者认为这使 diffuse stripe/stain 等低值细节被忽略。改用 `M_pow` 后梯度为 `exp(z/n)`，低值仍非零且高值继续增长；`n→∞` 时趋于 logarithmic mapping，论文取 `n=3`，并指出 Xue et al. 2024 用过相关的 fourth-root loss。[P: Eq.3–4, Fig.7–8]

### 7.4 LeakySmeLU

正文定义：

```math
\operatorname{LeakySmeLU}(x;\beta,\epsilon)=
\begin{cases}
\epsilon x,&x\le-\beta\\
(1-\epsilon)\frac{(x+\beta)^2}{4\beta}+\epsilon x,&|x|<\beta\\
x,&x\ge\beta.
\end{cases}
```

它在过渡区是二次函数、整体 `C1` 连续，负区保留小斜率避免完全 dying。[P: Eq.5] 正文没有披露 `β,ε` 数值；官方实现 config 名为 `SmeLU`，实际常量 `alpha=0.5`（对应正文半宽 `β`）、negative slope `beta=0.01`（对应正文 `ε`）。[C: `neuralnetworks/components/smelu.slang:10–29`] 代码变量名与论文符号相反，但函数语义一致。

### 7.5 公开 default/example 配置，不能自动冒充 paper formal

官方 commit 的 default 是：

- BSDF encoding 100k steps；direct finetune 0；sampler 50k；aux 50k；
- Adam `lr=1e-3`, cosine final multiplier `0.01`, gradient scale `128`, `β1=.9`, `β2=.999`, `ε=1e-7`, L2=0，跳过 non-finite gradient；
- deterministic gradient accumulation；half trainable weights + float optimizer state；
- `[64,16,4,1]`、`[1k,4k,16k,64k]`、每次生成 64 batches、跨实例共享 query；
- power-mapped L1 `n=3`；latent 8 channels/1 mip；encoder `[64,64,64]` LeakyReLU；decoder `[16,16]` `SmeLU`；
- **output 为 `ScaledSigmoid(1e5)`，不是正文的 `exp`**；direction inputs 为 **`WhWdZiZo`，不是正文最终 full tuple**。[C: `configs/default.json:29–198`]

三个 derived examples 分别将 decoder 设成 `[128,128,128,128]` + single `64k`、`[16,16]` + single `64k`、`[16,16]` + `64→16→4→1`；它们关闭 aux 但仍继承 50k sampler phase。[C: `configs/single_instance_training_large.json`, `single_instance_training_small.json`, `multi_instance_training_small.json`] README 将其关联到 Figure 1，但 source material、output activation、direction feature 与精确参数数目并不完全对应正文，故只能作为 algorithm example。

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path | 每 shading query：取 baked latent → 预测两 learned frames → 构造 direction features → 一个 compact decoder 输出 RGB `f̂` | P: Fig.2；§4.1 |
| Runtime multiplicity | 最终 1 个实例；64/16/4 只存在于训练 | P: §3, §6 |
| Parameter count | 正式 small `947`，large `~54k`；包括 8→12 frame projection 后可精确复核为 947/54,243 | P: Fig.1；I 算术，见 §5.3 |
| MAC/FLOP | 未报告；从正式 947 结构可推静态矩阵规模，但本文不把推导值冒充作者 benchmark | P: 未报告 |
| Shared/per-asset/state bytes | 未报告。论文没有 latent resolution、storage precision 或 mip memory 表 | P: 未报告 |
| Texture/feature fetches | 未报告。只知 runtime 读取 baked latent | P: Fig.2 |
| Precision/quantization | 正文未报告；C 训练/inference network path 使用 half/CoopVec，但不能确认论文部署资产的完整序列化精度 | C: network source |
| Hardware/backend | training kernels 为 SlangPy fully fused；wall-clock 在单 NVIDIA RTX 5090 测量。没有 shader inference backend benchmark | P: §4.3, Table 3 |
| Time/FPS/latency | 只报告 optimization seconds；未报告单次 BRDF evaluation ns、frame FPS 或 end-to-end renderer latency | P: Table 3 |
| Precompute/prepare/amortization | encoder baking 属于 offline；是否把 compilation/checkpoint/render diagnostics 纳入 Table 3 未报告 | P/C gap |

Multi-instance schedule 对部署成本原则上是零增量：所有候选结构相同，最终仍只有一个模型。因此它与本项目“单次 `evaluate()` 静态有界”的 runtime contract 没有结构冲突，但会显著提高 peak training state 和并行实例管理需求。论文展示在 fully fused backend 中“并行网络数对 training performance 影响 marginal”，这是特定实现与 GPU 的结论，不应外推为任意 framework 都免费；作者也明确提示其他 framework overhead 可能不同。[P: §4.3]

## 9. 实验 protocol、baseline、指标与结果

### 9.1 实验总览

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| 容量与初始化方差 | large 4×128、small 2×16 各 64 个初始化；再以同 cost 的 multi-instance 训练 small | large single；small single；small ours | training loss trajectories、渲染 | large run-to-run 一致且 loss 更低；small single 方差大；ours 让 small 的不同 run 更一致，但没有消除相对 large 的容量差距 | P: Fig.1 |
| 早期 rank 稳定性 | `K=64,B=1k`，对 100 个独立 repetition 追踪最终 best/worst 在训练各时刻的 rank | 无 culling 的全程观测 | rank trajectory | warm-up 后最终好/坏候选开始分离，为阶段淘汰提供经验依据；未给固定 rank-correlation 数值 | P: §3.1, Fig.3 |
| fixed-cost schedule | 4 个难度递增材质；每条 curve 是 64 次优化的 mean，band 为 1 std；100k steps | `1@64k`, `4@16k`, `16@4k`, `64@1k`, ours | training loss over steps | 简单材质差异小；复杂材质中多初始化改善 exploration，但固定小 batch 妨碍最终 settle；ours 常优于第二好固定配置 | P: Fig.5, §4.2–4.3 |
| culling retention | 15 材质；统计 survivor 相对于 64 candidates 的最终 rank；为得到完整 rank 必须保留或恢复未淘汰候选，但 Table 1 未像 Table 2 caption 那样明说具体执行方式 | 无 selector baseline | best/top-3/top-6/top-12 retention | Table 1 表格平均 79.1/95.6/98.7/99.6%；邻接 prose 报 74.4/未报/99.4/未报，样本数与表不一致 | P: Table 1 与 pp.6 prose |
| surviving loss | 关闭实际 culling，仅模拟选择路径，才能看见全部 64 的最终 best | random candidate | `L_b`, `L_o/L_b`, `L_r/L_b` | 平均 best loss `.081±.089`；ours/best `1.010±.017`；random/best `1.946±.658` | P: Table 2 |
| data sharing | all-main shared；supplement 另做 independent batches | shared vs independent | retention、best/relative loss | no-share best retention 72.4% vs shared 79.1%；两者 best loss 均 .081；no-share ours/best 1.015 | S: §B, Tables 1–2 |
| pruning factor | 同 Fig.5 四材质，2×/4×/8× | 三种 phase schedule | loss curve | 作者称影响 negligible；无数值表/显著性检验 | S: §A, Fig.1 |
| optimizer | 与 main Fig.1(c) 相同 small config，Adam/Muon/SOAP 各 64 初始化 | 三 optimizer | loss trajectories | convergence rate 不同，但三者都保留显著 trajectory spread；换 optimizer 本身未解决 variance | S: §C, Fig.2 |
| loss mapping | 15 材质，每项 5 次初始化，共享 sphere+plane scene | log vs power | HDR FLIP | average `.0847±.0340` → `.0583±.0273`，15/15 材质 power 数值更低 | S: §D, Table 3 |
| activation | 同上，隔离 activation | ReLU/LeakyReLU/LeakySmeLU | HDR FLIP、training loss、faceting | average HDR `.0601/.0582/.0583`；LeakySmeLU 不具平均数值优势，但 Fig.9 的小网高光更平滑 | P: Fig.9；S: Tables 4–5 |
| direction input | 同上，隔离 input parameterization | direct、old half/diff、stable half/diff-only、full | HDR FLIP、training loss | full 平均 HDR `.0583` 最低，loss `.0769` 最低；但若干单材质由 direct/old 更优 | S: Tables 6–7 |
| optimization wall-clock | single RTX 5090，fully fused SlangPy；4 个 layered materials | `1@64k` vs ours shared data | seconds | ours 54.40–59.00s，baseline 78.05–86.50s；降低 28–32% | P: Table 3 |

### 9.2 Figure 1：不能把“稳定”误写成“容量提升”

Figure 1(b) 的 large 54k decoder 在 64 个初始化上高度一致；Figure 1(c) 的 947-param small decoder 出现明显 spread；Figure 1(d) 在与 single-small 相同总 network-evaluation cost 下采用本文 schedule，使 small runs 的结果聚拢；Figure 1(a) 只是 reference rendering。[P: Fig.1] 但图中 large 的 loss 仍低于 small+ours。作者的因果主张是“更可靠地找到小网络已有的好 basin”，不是“训练算法让 947 参数具备 54k 参数的表示能力”。

### 9.3 Figure 5：fixed-K 对照与 exploration/exploitation 解释

所有对照都有 `K×B=64k`，运行 100k steps：

| 名称 | K | B | 作用 |
|---|---:|---:|---|
| Baseline | 1 | 64k | 单例大 batch |
| `4@16k` | 4 | 16k | 少量候选 |
| `16@4k` | 16 | 4k | 中等候选 |
| `64@1k` | 64 | 1k | 最大初始化覆盖、全程 noisy batch |
| Ours | `64→16→4→1` | `1k→4k→16k→64k` | exploration 到 exploitation |

四列材质依次为 Oxydized metal、Brushed brass、Gold and ceramic、Glazed ceramic。[P: Fig.5] 对 rough/simple Oxydized metal，各法已稳定，multi-instance 只有 marginal improvement；对更复杂 layered material，增大 `K` 往往能找到更好 basin，但 `64@1k` 的持续噪声又妨碍 loss 后期下降。作者以此解释 ours 为何能在同 cost 下兼顾两端。[P: §4.3] 图只给 loss curves，没有公开每列最终数值或 bootstrap CI。

### 9.4 Table 1：保留率全表与正文内部冲突

| Material | Best rank 1 | Top 5% / top 3 | Top 10% / top 6 | Top 20% / top 12 |
|---|---:|---:|---:|---:|
| Birdcage | 93.3% | 100.0% | 100.0% | 100.0% |
| Chandelier | 73.3% | 80.0% | 93.3% | 100.0% |
| Hammer | 86.7% | 100.0% | 100.0% | 100.0% |
| Lantern | 86.7% | 100.0% | 100.0% | 100.0% |
| Mirror | 60.0% | 93.3% | 100.0% | 100.0% |
| Palette | 66.7% | 93.3% | 100.0% | 100.0% |
| Chair | 93.3% | 100.0% | 100.0% | 100.0% |
| Scales | 80.0% | 100.0% | 100.0% | 100.0% |
| Table | 60.0% | 80.0% | 86.7% | 93.3% |
| Scratched steel | 100.0% | 100.0% | 100.0% | 100.0% |
| Bumpy plastic | 93.3% | 100.0% | 100.0% | 100.0% |
| Oxydized metal | 66.7% | 100.0% | 100.0% | 100.0% |
| Gold and ceramic | 66.7% | 93.3% | 100.0% | 100.0% |
| Brushed brass | 86.7% | 93.3% | 100.0% | 100.0% |
| Glazed ceramic | 73.3% | 100.0% | 100.0% | 100.0% |
| **Average（表中）** | **79.1%** | **95.6%** | **98.7%** | **99.6%** |

[P: Table 1, p.6]

这是本次审计最重要的未解析论文内部冲突。表格全部以 6.7% 为基本增量，且其平均正好对应每材质 15 trials；紧邻正文却写“24 trials per material、all 360 trials”，并报告 best 74.4%、top-10 99.4%。15 个材质×24 的确是 360，但这些数值无法由表格 15-trial proportions 得到。[P: p.6, lines after Table 1] 没有发现勘误；因此不得选择其中一组而不标注冲突。本报告后文使用“Table 1 表格值”和“adjacent prose 值”分别称呼。

### 9.5 Table 2：幸存者与 counterfactual best

| Material | Best loss `L_b` | Ours `L_o/L_b` | Random `L_r/L_b` |
|---|---:|---:|---:|
| Birdcage | `.054±.003` | `1.001±.002` | `1.872±.120` |
| Chandelier | `.038±.003` | `1.024±.060` | `1.814±.123` |
| Hammer | `.032±.003` | `1.003±.009` | `1.686±.130` |
| Lantern | `.022±.002` | `1.001±.003` | `1.769±.148` |
| Mirror | `.081±.005` | `1.010±.017` | `1.808±.120` |
| Palette | `.260±.014` | `1.006±.016` | `1.576±.091` |
| Chair | `.044±.005` | `1.000±.001` | `3.279±.473` |
| Scales | `.046±.002` | `1.005±.015` | `1.704±.071` |
| Table | `.019±.001` | `1.067±.131` | `1.558±.117` |
| Scratched steel | `.056±.005` | `1.000±.000` | `2.450±.235` |
| Bumpy plastic | `.018±.001` | `1.001±.005` | `1.360±.062` |
| Oxydized metal | `.022±.001` | `1.004±.006` | `1.416±.065` |
| Gold and ceramic | `.283±.007` | `1.005±.013` | `1.444±.063` |
| Brushed brass | `.045±.004` | `1.015±.041` | `3.575±.303` |
| Glazed ceramic | `.201±.021` | `1.009±.018` | `1.870±.198` |
| **Average** | **`.081±.089`** | **`1.010±.017`** | **`1.946±.658`** |

[P: Table 2] 每项是 15 optimization trials 的 mean±std。为知道“如果没有淘汰，64 例中谁最终最好”，此实验关闭真实 culling，只模拟选择路径；所以它验证 ranking fidelity，不是正式 culling run 的独立 wall-clock benchmark。[P: Table 2 caption]

### 9.6 Shared-data 与 wall-clock

在前 3 phases 中，shared data 相对每实例独立生成分别减少 64×、16×、4× query generation；跨 4 个等长 phase 合计减少 3.01×。[P: §4.3] Supplemental 的 no-sharing 对照给出：

| 配置 | Best rank 1 | Top 5% | Top 10% | Top 20% | Best loss | Ours/best | Random/best |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shared（main Table 1/2 表格） | 79.1% | 95.6% | 98.7% | 99.6% | `.081±.089` | `1.010±.017` | `1.946±.658` |
| Independent | 72.4% | 94.2% | 99.1% | 100.0% | `.081±.088` | `1.015±.017` | `1.953±.668` |

[S: §B, Tables 1–2] 作者把较高 best-retention 解释为共享 query 使 loss 更 correlated、更易排序；best achievable loss 没有显著差异。Supplemental 没给统计检验，故“没有显著 downside”是作者定性结论，不是报告中可复核的 p-value。

RTX 5090 优化时间如下：

| Material | Baseline `1@64k` | Ours |
|---|---:|---:|
| Oxydized metal | 86.50 s | 59.00 s |
| Gold and ceramic | 78.05 s | 55.80 s |
| Glazed ceramic | 81.02 s | 57.11 s |
| Brushed brass | 79.48 s | 54.40 s |

[P: Table 3] 这比 baseline 快 28–32%。这里“same training cost”首先指相同 network evaluations；wall-clock 反而更低是 shared data + fused implementation 的结果。两种 cost 定义不能混为一项普遍硬件结论。

### 9.7 Power mapping 定量结果

所有值为 5 个初始化在同一 sphere+plane test scene 的 HDR FLIP mean±std；由于两列 loss definition 本身不同，supplemental 刻意不比较它们的 training-loss 数值。[S: §D]

| Material | Log mapping | Power mapping `n=3` |
|---|---:|---:|
| Birdcage | `.0541±.0097` | `.0370±.0029` |
| Chandelier | `.0537±.0045` | `.0345±.0013` |
| Hammer | `.0862±.0128` | `.0412±.0024` |
| Lantern | `.0608±.0016` | `.0346±.0007` |
| Mirror | `.1196±.0121` | `.0810±.0039` |
| Palette | `.1134±.0135` | `.0991±.0144` |
| Chair | `.1265±.0096` | `.0776±.0134` |
| Scales | `.1379±.0154` | `.0652±.0012` |
| Table | `.0707±.0059` | `.0425±.0024` |
| Glazed ceramic | `.1224±.0138` | `.1063±.0081` |
| Brushed brass | `.0541±.0014` | `.0404±.0028` |
| Scratched steel | `.0468±.0011` | `.0353±.0012` |
| Bumpy plastic | `.0524±.0025` | `.0391±.0006` |
| Oxydized metal | `.0536±.0038` | `.0400±.0012` |
| Gold and ceramic | `.1177±.0113` | `.1014±.0100` |
| **Average** | **`.0847±.0340`** | **`.0583±.0273`** |

[S: Table 3] 15/15 rows power 数值更低。Figure 8 的视觉诊断与 error map 显示 log 能覆盖 specular peak，却漏掉 diffuse stripes/stains；power 在两类幅度之间取得更均衡梯度。[P: Fig.7–8]

### 9.8 Activation 定量结果与视觉目标

| Material | HDR ReLU | HDR LeakyReLU | HDR LeakySmeLU | Loss ReLU | Loss LeakyReLU | Loss LeakySmeLU |
|---|---:|---:|---:|---:|---:|---:|
| Birdcage | `.0417±.0041` | `.0407±.0026` | `.0370±.0029` | `.0591±.0014` | `.0602±.0027` | `.0528±.0039` |
| Chandelier | `.0364±.0017` | `.0376±.0034` | `.0345±.0013` | `.0407±.0039` | `.0410±.0024` | `.0326±.0015` |
| Hammer | `.0421±.0026` | `.0419±.0016` | `.0412±.0024` | `.0389±.0026` | `.0377±.0017` | `.0298±.0007` |
| Lantern | `.0379±.0027` | `.0361±.0010` | `.0346±.0007` | `.0276±.0006` | `.0272±.0010` | `.0195±.0006` |
| Mirror | `.0847±.0012` | `.0803±.0026` | `.0810±.0039` | `.0690±.0032` | `.0703±.0019` | `.0738±.0065` |
| Palette | `.0990±.0210` | `.0929±.0080` | `.0991±.0144` | `.2932±.0130` | `.2918±.0067` | `.3017±.0122` |
| Chair | `.0865±.0056` | `.0756±.0127` | `.0776±.0134` | `.0590±.0016` | `.0516±.0064` | `.0512±.0090` |
| Scales | `.0672±.0023` | `.0671±.0033` | `.0652±.0012` | `.0545±.0068` | `.0502±.0023` | `.0448±.0017` |
| Table | `.0441±.0027` | `.0438±.0007` | `.0425±.0024` | `.0278±.0020` | `.0255±.0020` | `.0190±.0013` |
| Glazed ceramic | `.0865±.0037` | `.0832±.0027` | `.1063±.0081` | `.1450±.0063` | `.1352±.0058` | `.2141±.0137` |
| Brushed brass | `.0424±.0034` | `.0415±.0031` | `.0404±.0028` | `.0432±.0020` | `.0440±.0017` | `.0388±.0037` |
| Scratched steel | `.0373±.0018` | `.0373±.0019` | `.0353±.0012` | `.0533±.0029` | `.0524±.0031` | `.0455±.0036` |
| Bumpy plastic | `.0398±.0004` | `.0400±.0009` | `.0391±.0006` | `.0206±.0004` | `.0208±.0008` | `.0159±.0011` |
| Oxydized metal | `.0438±.0032` | `.0432±.0017` | `.0400±.0012` | `.0247±.0014` | `.0242±.0010` | `.0209±.0008` |
| Gold and ceramic | `.1123±.0022` | `.1110±.0040` | `.1014±.0100` | `.1856±.0176` | `.1824±.0171` | `.1925±.0111` |
| **Average** | **`.0601±.0264`** | **`.0582±.0246`** | **`.0583±.0273`** | **`.0761±.0754`** | **`.0743±.0746`** | **`.0769±.0867`** |

[S: Tables 4–5] LeakySmeLU 的平均 HDR FLIP 与 training loss 都不是最低；LeakyReLU 分别以极小差距和明确差距更低。作者仍采用平滑 activation 的理由是 Figure 9 的结构性伪影：2×16 ReLU 在 sharp highlight 上出现 facet，甚至 2×32 仍可见，而 2×16 SmeLU 平滑。[P: Fig.9] 这是“视觉连续性与 compact capacity 的 tradeoff”，不是被平均表格证明的全面数值胜利。Glazed ceramic 是强负例：LeakySmeLU HDR `.1063` 对 ReLU `.0865`，loss `.2141` 对 `.1450`。

### 9.9 Direction input 定量结果

下表四列是 direct `(ω_i,ω_o)`、old Rusinkiewicz `(ω_h,ω_d)`、stable-only `(ω_h,ω_d')`、full `(ω_i,ω_o,ω_h,ω_d')`。

| Material | HDR direct | HDR old | HDR stable-only | HDR full | Loss direct | Loss old | Loss stable-only | Loss full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Birdcage | `.0410±.0018` | `.0381±.0027` | `.0434±.0026` | `.0370±.0029` | `.0947±.0028` | `.0720±.0024` | `.0708±.0046` | `.0528±.0039` |
| Chandelier | `.0358±.0019` | `.0346±.0010` | `.0388±.0034` | `.0345±.0013` | `.0548±.0017` | `.0471±.0009` | `.0502±.0049` | `.0326±.0015` |
| Hammer | `.0411±.0014` | `.0444±.0021` | `.0476±.0017` | `.0412±.0024` | `.0491±.0022` | `.0569±.0024` | `.0506±.0024` | `.0298±.0007` |
| Lantern | `.0367±.0010` | `.0383±.0015` | `.0396±.0024` | `.0346±.0007` | `.0328±.0011` | `.0351±.0016` | `.0374±.0021` | `.0195±.0006` |
| Mirror | `.0986±.0030` | `.0865±.0034` | `.0911±.0025` | `.0810±.0039` | `.1272±.0032` | `.1017±.0072` | `.1030±.0044` | `.0738±.0065` |
| Palette | `.0839±.0046` | `.0837±.0116` | `.0854±.0051` | `.0991±.0144` | `.3282±.0049` | `.3233±.0117` | `.3247±.0100` | `.3017±.0122` |
| Chair | `.0739±.0083` | `.0691±.0064` | `.0832±.0160` | `.0776±.0134` | `.0593±.0029` | `.0735±.0033` | `.0667±.0056` | `.0512±.0090` |
| Scales | `.0686±.0043` | `.0674±.0014` | `.0703±.0015` | `.0652±.0012` | `.0725±.0031` | `.0621±.0027` | `.0641±.0030` | `.0448±.0017` |
| Table | `.0434±.0032` | `.0439±.0026` | `.0504±.0029` | `.0425±.0024` | `.0314±.0016` | `.0369±.0011` | `.0335±.0035` | `.0190±.0013` |
| Glazed ceramic | `.1357±.0046` | `.1105±.0090` | `.1205±.0082` | `.1063±.0081` | `.4221±.0129` | `.2193±.0104` | `.2371±.0156` | `.2141±.0137` |
| Brushed brass | `.0485±.0041` | `.0456±.0041` | `.0490±.0042` | `.0404±.0028` | `.0945±.0022` | `.0507±.0038` | `.0560±.0044` | `.0388±.0037` |
| Scratched steel | `.0399±.0009` | `.0416±.0013` | `.0395±.0019` | `.0353±.0012` | `.0997±.0072` | `.0973±.0041` | `.0774±.0043` | `.0455±.0036` |
| Bumpy plastic | `.0403±.0003` | `.0456±.0015` | `.0433±.0024` | `.0391±.0006` | `.0274±.0013` | `.0559±.0023` | `.0292±.0030` | `.0159±.0011` |
| Oxydized metal | `.0409±.0009` | `.0588±.0081` | `.0532±.0056` | `.0400±.0012` | `.0289±.0005` | `.0542±.0021` | `.0318±.0035` | `.0209±.0008` |
| Gold and ceramic | `.1298±.0063` | `.1096±.0069` | `.1096±.0039` | `.1014±.0100` | `.3852±.0328` | `.2608±.0125` | `.2473±.0181` | `.1925±.0111` |
| **Average** | **`.0639±.0339`** | **`.0612±.0257`** | **`.0643±.0272`** | **`.0583±.0273`** | **`.1272±.1345`** | **`.1031±.0895`** | **`.0986±.0924`** | **`.0769±.0867`** |

[S: Tables 6–7] Full input 的 average 两项最低，且 training loss 在 15/15 rows 最低；但 HDR 不是逐材质全胜：Hammer 略偏 direct，Palette/Chair 偏 old half/difference。Stable-only 的平均 HDR 还略差于 direct 和 old；所以成功机制不是“修掉 singularity 即可”，而是保留 direct/full redundancy 让网络按材质选择。

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `author-negative` | small 947-param single instance，64 初始化 | run-to-run loss 与视觉质量差异大 | compact network 的 good minima 少，初始化/数据顺序敏感 | 这是优化方差问题本体，不等价于 representation 不可用 | P: Fig.1(c), §1 |
| `ablation-inferior` | `64@1k` 全程不淘汰/不增 batch | exploration 充分，但后期 noisy gradient 难 settle 到低 loss | 小 batch 适合早期，大 batch 适合后期 | 固定总 evaluations 下，候选数与梯度估计精度存在真实竞争 | P: Fig.5, §4.3 |
| `ablation-inferior` | `1@64k` baseline | complex layered materials 上结果分散/经常高 loss | 单初始化探索不足 | 多 seed 是 distributional control，单 seed 不能代表方法 | P: Fig.5, Table 2 |
| `known-risk` | greedy culling | 不总保留最终 rank-1；Table material 仅 60% | early rank 有信息但不是完美预测 | 对新 source family 必须先校准 retention，不应直接把 79.1% 当常数 | P: Table 1 |
| `ablation-neutral` | pruning factor 2×/4×/8× | 四材质测试中影响 negligible | phase 粒度不敏感 | 只有曲线与 4 个材质，不能宣称普适 | S: §A, Fig.1 |
| `ablation-inferior` | independent data per instance | rank-1 retention 72.4%，低于 shared 79.1%；best loss同为 .081 | correlated batches 有利于排序，且省生成成本 | 不支持“更多数据多样性必然更好” | S: §B |
| `author-negative` | 仅换 Adam→Muon/SOAP | convergence rate 改变，trajectory spread 仍大 | optimizer 不是 variance 的唯一来源 | 论文没给精确 hyperparameters，结论限于图示设置 | S: §C, Fig.2 |
| `author-negative` | old Rusinkiewicz half/difference | perfect-reflection singularity，输入急剧变化 | `φ_h` 在 `ω_h=z` 未定义 | 属 coordinate conditioning defect，不是网络容量问题 | P: Eq.1, Fig.6 |
| `ablation-inferior` | stable half/difference-only | average HDR `.0643`，差于 old `.0612` 与 full `.0583` | normal mapping 可破坏 canonical lobe alignment | singularity 修复不保证单独用该坐标更可表达 | P: §5；S: Table 6 |
| `author-negative` | logarithmic mapped L1 | specular peaks 尚可，但 diffuse stripes/stains 被忽略；15/15 HDR rows 差于 power | `exp` output 下低值梯度趋零、高值饱和 | 若 output activation 改成 scaled sigmoid，梯度推导必须重做 | P: Fig.7–8；S: Table 3；C gap |
| `author-negative` | ReLU compact decoder | dying neurons；sharp highlight 出现 facets，2×32 仍可见 | piecewise linear activation 与容量瓶颈 | 这是结构性伪影，单看平均 FLIP 可能漏检 | P: Fig.9 |
| `ablation-inferior` | LeakySmeLU 对 Glazed ceramic | HDR/loss 都明显劣于 ReLU/LeakyReLU | 作者未单独解释该材料 | LeakySmeLU 是 smoothness tradeoff，不能无条件作为 quality win | S: Tables 4–5 |
| `ablation-inferior` | LeakySmeLU average | HDR `.0583` 略差于 LeakyReLU `.0582`；loss `.0769` 差于 `.0743` | 作者明确说 LeakyReLU 有时 loss 更低，SmeLU 用于避免 facet | 应以 artifact-aware metric 与 Pareto 而非单一 average 选择 | P: §5；S: Tables 4–5 |
| `representation-limit` | small+ours vs large | small 训练更稳定但最终 loss 仍高于 large | schedule 不改变模型容量 | 不能把 variance reduction 报成 representation quality 等价 | P: Fig.1 |

作者没有公开逐次开发日志，故不能从最终 schedule 反推出“曾失败的 pruning threshold、loss exponent、activation 参数或网络宽度”。除了正文/supplemental 明确展示的 inferior ablation，上述未披露开发历史一律记为未报告。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Multi-instance schedule | `64→16→4→1`，batch `1k→4k→16k→64k`，4 等长 phases | 另测 2×/8× pruning | helper 定义等长 period，但 64-step submission block 令 default 实际切换落在 25,024/50,048/75,008；boundary 后按 loss `argsort` 同步 prune | **算法语义一致、lifecycle 有 block 量化差异**；P 未规定实现边界处理 |
| Cost matching | 每一步总 batch 固定，100k steps、同 network evaluations | 无变化 | batch 字符串解析为 1024 倍，`K×B` 固定 65,536 | **一致**；但 wall-clock 等价只在特定 fused backend 上测 |
| Shared data | main 所有实验共享；整体 datagen 降 3.01× | no-share retention/relative loss 对照 | 单 batch broadcast 到全部 instances | **一致** |
| Candidate selection | greedy cull by loss；Table 1 未披露是否实际禁用 pruning | Table 2 明确关闭真实 culling，仅模拟 survivor | 最新 iteration block 的 mean loss 经 `np.argsort(instance_losses)[:keep]`；encoder/decoder 采用相同 indices | **主语义一致**；P 未精确定义 averaging window，C current 通常以最多 64 steps 的 block 排名，且 boundary block 可跨过理想 phase 边界 |
| Encoder/decoder width | encoder 3×64，decoder 2×16；large 4×128 | 定量表隔离其它设计 | default/derived configs 同 widths | **宽度一致** |
| Figure 1 parameter count | small 947、large约54k；与 final full directions 相容 | 无 | derived configs 继承 `WhWdZiZo`，实算 819/53,219 | **实质冲突**；公开 examples 不能精确对应图中网络数目 |
| Direction inputs | 最终 full `(ω_i,ω_o,ω_h,ω_d')` | Table 6/7 full average 最佳 | default/figure examples 为 `WhWdZiZo` | **冲突**；C 支持 `WhWdWiWo`，但 configs 未选 |
| Half/difference formula | shortest-arc Eq.2 | stable-only/full 消融 | `StableRusinkiewiczParameterization` 与公式一致 | **一致** |
| Output activation | `f̂=exp(z)`，loss 梯度推导依赖它 | 表格未重述 | default/derived examples 为 `ScaledSigmoid(scale=1e5)`；代码也支持 `Exp` | **实质冲突**；正文梯度分析不能原样套到 released default |
| Power loss | mapped L1，`n=3` | 15 材质 HDR FLIP | `L1WithPowerLog`, power 3 | **一致**，但与 ScaledSigmoid 组合的 logit 梯度不是 P Eq.4 |
| Activation | 正文称 LeakySmeLU；未给 `β,ε` | Table 4/5 | config 名 `SmeLU`，实现 `alpha=.5`,`beta=.01` 且负区有斜率 | **语义一致、命名/超参披露 gap** |
| Optimizer | Adam、Glorot；精确超参未报告 | Adam/Muon/SOAP 曲线；仍未给其超参 | 只实现/配置 Adam；`lr=.001`, cosine `.01`, β等齐全 | **C 可给当前 recipe，不足以证明 paper exact**；Muon/SOAP 不可复现 |
| Data/query | 15 个正式材质；P 只定义 record 内容 | sphere+plane HDR scene | default 是 FauxLeather；方向 95/5 mixture、UV stratification、online GPU | **正式 source/query 缺失**；C recipe 不能自动填 P |
| Runtime/export | bake latent、保留 decoder | 无 runtime benchmark | 完整 checkpoint/export，并带 sampler/aux phases | **C 能运行更大系统，但 sampler/aux 不属于本论文结果** |
| Precision | 未报告 | 未报告 | half trainable weights + float optimizer state；CoopVec | **仅 C 事实** |
| Formal assets | 9 da Vinci baked + 6 internal layered | 同一 15 材质 | 只有 Bark/FauxLeather/PatternedMetal examples | **缺失**；无法 source-identical reproduction |
| Retention statistics | Table 1 为 79.1/98.7%，prose 为 74.4/99.4%；表像 15 trials，prose 称 24 | no-share 表沿用 15-trial increments，并以 shared 79.1 作对照 | 无原始 trial log | **P↔P 未解析冲突**，S 实际支持 Table 1 表格版本 |

### 11.1 公开代码的算法保真边界

可以高置信标为 code-faithful 的是 scheduler、pruning、shared data、batch resize、power mapping 与 stable shortest-arc 公式。不能标为 paper-formal-reproduced 的是 Figure 1 精确网络、正文 `exp` output 下的 loss-gradient实验、15 个正式材质、正式 query distribution、Muon/SOAP 与所有表格的 raw seeds/logs。

### 11.2 为什么 output mismatch 是 load-bearing

P 的 loss 论证使用 `d exp(z)/dz=exp(z)=f̂`，由此得到 log mapping 的 sigmoid 梯度与 power mapping 的 `exp(z/n)`。[P: Eq.4] C 的 `ScaledSigmoid(s)` 有 `d f̂/dz=f̂(1-f̂/s)`；接近 upper bound 时又会衰减。因此 C default 虽使用同一个 target mapping，优化景观并不是正文公式分析的那一个。复现时必须选择：

- 精确复现 P：用 `Exp` 并登记 numerical safeguards；或
- 复现 C current recipe：保留 ScaledSigmoid，但把它标成 code-current variant，并重新测 loss ablation。

两者都可能是合理工程选择，但不能共用一个“paper exact” identity。[I]

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. **网络越大，收益越小。** 作者在脚注明确说 multi-instance 的影响随 network size 增大而减弱，因为大网络本来更可靠；他们仍默认开启，是因为实验中“不伤且偶有改善”。这个陈述不是“所有规模均有统计显著收益”。[P: §4.2 footnote 2]
2. **greedy pruning 可误删真正最好者。** Table 1 的 rank-1 retention 远非 100%，个别材质表格值只有 60%。作者把目标放在“高概率保留 top performer”而非绝对最优。[P: §4.3]
3. **backend dependence。** Fully fused SlangPy 下并行候选开销 marginal，shared datagen 让总时间更快；作者明确提示别的 framework 会有不同 overhead。[P: §4.3]
4. **half/difference 不是普适最佳坐标。** normal mapping 可把 lobe 旋离预期 alignment，故作者保留 direct directions。[P: §5]
5. **LeakySmeLU 不是最低 loss 的保证。** 作者明确说 standard LeakyReLU 有时更低；选择平滑 activation 是为了避免 facet。[P: §5]
6. **小网容量仍受限。** Fig.1 的 ours 改善稳定性但没有达到 large 的最低 loss。[P: Fig.1]
7. **方法只选择 candidate，不改进 candidate 本身。** 作者把 stochastic weight averaging 留作 orthogonal future work。[P: §6]

### 12.2 未报告/材料不可得

- Table 1 的正确 trial 数与正确 aggregate；没有勘误或 raw log。
- project-page “88% variance reduction / 38% average loss reduction”的精确定义、baseline、聚合与对应图表。
- 正式 15 个材质文件、转换脚本、纹理/license manifest 与其 source hash。
- 正文主实验的 exact reference renderer/backend、query generator、方向与 UV 分布、train/validation query count、held-out split、loss-evaluation batch。
- pruning boundary 用 instantaneous、phase average 还是 validation loss；P 也未说明 Table 1 是否像 Table 2 一样关闭真实 culling 后离线模拟。C current 使用最新最多 64-step block 的 mean training loss，不能证明这就是 paper exact。
- 正式 Adam 的 learning rate、β、ε、cosine schedule、gradient scale、weight decay、deterministic mode；C default 不能倒推为 P exact。
- Muon/SOAP 的版本、hyperparameters、schedule 与数值结果。
- 正式 Figure 1/3/5/Table 1/2 的 seed list、raw trajectories、confidence interval 或 hypothesis test。
- encoder 的 native input dimensionality；它会随材质 graph/layer count 变化。
- 正式 latent texture resolution、mip 数、filtering、format、bytes、network precision、compiled artifact format。
- 单次 runtime `evaluate` 的 MAC/FLOP、texture fetches、latency、register/shared-memory 占用、renderer FPS。
- Table 3 timing 是否包括 JIT compile、data initialization、checkpoint、reference rendering 或只计 optimizer loop。
- Figure 5 曲线的 per-material final numeric values；supplemental pruning factor 只有图，没有原始数表。
- activation 的论文 `β,ε`；C 常量只能描述 released implementation。
- 结果是否对透射、各向异性、极端 grazing、displacement、multi-UDIM 或多 mip 稳健。
- power exponent `n=3` 的系统 sweep；正文只解释与 `n→∞`、Xue `n=4` 的关系。
- optimizer variance 与 training-data-order variance 的独立因果分解；main runs 据称只改 initialization seed。
- multi-instance peak VRAM、host state 与 compilation scaling；只报告 wall-clock。
- matched sampler 的质量与训练方差；正式论文没有 sampler 实验。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

本文同时动用了三种不同“容量”，必须拆开：

1. **部署函数容量**仍由 latent、two-frame projection 与最终 decoder 决定。Multi-instance 不增加它；947-param candidate 最终还是 947 parameters。
2. **训练期搜索容量**来自 64 个独立初始化。作者在早期把固定 query-evaluation budget 分散到 64 个 basin，再逐步集中到一个 basin。这提高找到既有 good solution 的概率，但 peak model/optimizer state 约随实例数增加。
3. **输入/函数族容量**由 full direction features 和 LeakySmeLU 改变。Full tuple 把 decoder input 从只含 direct 或 half/diff 扩成两者并存，947 的精确参数数目本身已包含这部分额外 input weights；LeakySmeLU 把 piecewise-linear family 改成含二次平滑段。它们不是 optimization schedule 的效果。

Power mapping 又是第四类：不改变部署函数或参数数目，只改变训练时不同 radiance/BRDF magnitude 的梯度权重。因此未来实验不能用一个“本文完整方法 vs baseline”就判断哪个组件在起作用；至少应把 schedule、loss、direction 与 activation 分开 matched。

### 13.2 成功所依赖的假设

- 同一 source/query/architecture 内，early training loss rank 对 final rank 有可利用的相关性；
- loss 是与最终质量足够一致的 selector。Palette/Chair 的 input 消融已表明 lower training loss 不保证该 parameterization 的 HDR FLIP 最优，故 selector 的 validity 需要单独检查；
- small-batch noise 能提供有益 exploration，而后期 large batch 能可靠 exploitation；
- online reference query 可共享，且不同实例无需独立数据多样性才能找到不同 basin；
- GPU/back-end 能同时容纳并高效执行多个 small MLP，data generation 是可被共享削减的显著成本；
- network 确实处在 optimization-limited 而非纯 capacity-limited 区域。若所有 64 seeds 都收敛到同样坏的 approximation，schedule 无法创造新函数容量；
- source asset 不需要跨未见材质 generalization；算法选择的是对当前 material 专门化的候选。

### 13.3 可迁移机制与不能直接迁移的部分

**可迁移、但要重新验证的机制：**

- fixed-budget `K_t×B_t=const` scheduler 与 phase-boundary pruning；
- shared queries 让 candidate loss 同条件比较，并降低 expensive online reference generation；
- 把 full candidate 训练保留作小规模 audit，以校准 early-rank/top-k retention；
- 对 `exp` HDR evaluator 用 power mapping重新分配 diffuse/peak 梯度；
- stable half/difference 与 direct direction 并存，作为材质自适应输入；
- artifact-aware smooth activation evaluation，而非只看 scalar loss。

**不能直接搬用的量：**

- `64→16→4→1`、每 phase 25k、79.1% retention 不是跨 source family 常数；
- RTX 5090 的 28–32% wall-time improvement 取决于 kernel fusion、reference generator 与内存布局；
- six layered materials 没有开放，不能拿当前 LayerStack 的名字相似就声称 matched；
- 论文的 `exp` loss 分析不能直接证明 `ScaledSigmoid` default 的行为；
- LeakySmeLU 不能作为无条件的平均 quality upgrade；其强项是连续性/高光 facet；
- 2026 论文没有验证 joint evaluator–sampler ranking，故不能未经实验就用包含 sampler KL 的混合 loss 淘汰候选。

### 13.4 与本项目 runtime contract 的关系

Multi-instance 是 **compiler/training policy**，部署后完全移除，最适合作为 compact evaluator 的 optimization-robustness mechanism。它不会让 `evaluate(wo,wi)` 的执行次数、模型数或随机访问失去静态上界。Power loss同样是 training-only。

Full direction input 与 LeakySmeLU 会进入 runtime：前者增加固定 decoder input/第一层权重，后者增加固定 activation arithmetic；两者仍静态有界，但必须重新登记 parameter/MAC、Slang parity 与单次查询成本。它们属于新 evaluator candidate/recipe，而不是“免费训练技巧”。

本文不提供 transport、visibility 或 matched sampler 证据。对于本项目，它不是 scene-level teacher 或 sampling proposal，而是：

- compact neural evaluator 的训练 schedule 候选；
- HDR loss/coordinate/activation 的 capacity diagnostic 来源；
- 在方法成形后可进入 compiler recipe 的 robustness 组件。

### 13.5 Load-bearing related work 建议

| Related work | 本文依赖 | 建议状态与 promotion trigger |
|---|---|---|
| Zeltner et al. 2024, *Real-Time Neural Appearance Models* | 本文 case-study representation 的直接母体 | **load-bearing / 必须 full report**；任何把 2026 设计迁入当前 NVIDIA reproduction 前都要与 2024 correspondence 并读 |
| Xu et al. 2025, *Improving Angular Parameterization for Compact Neural Materials*，DOI `10.1145/3757374.3771447` | 作者直接以其说明 direction parameterization 对 compact model 的影响 | **建议升为 load-bearing**；trigger：计划实现/比较 full direction 或 stable half/diff |
| Xue et al. 2024, *A Hierarchical Architecture for Neural Materials* | `n=4` related root loss 的直接 precedent | **建议升为 load-bearing-related**；trigger：power exponent/loss family 成为候选轴 |
| Jamieson & Talwalkar 2015；Li et al. 2018 successive halving/Hyperband | pruning 预算分配的直接算法谱系 | catalog 为 load-bearing-related；只有要推导 phase/budget 理论或自适应 schedule 时才需 full report |
| Smith et al. 2018, *Don't Decay the Learning Rate, Increase the Batch Size* | batch annealing 的直接优化动机 | 保持 discovery；若要把 exploration/exploitation 写成方法机制而不只经验设置，再升 full report |
| Shamir et al. 2020, SmeLU | activation 的直接定义来源 | 若 LeakySmeLU 成为正式 candidate，再升 load-bearing；否则本论文已足够说明材质消融 |
| Muon / SOAP | 只用于说明换 optimizer 不能消除 spread | 暂不升；只有 optimizer-specific reproduction 成为任务轴时再研究 |

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

### 14.1 当前真实状态

当前仓库 formal identity 是 `nvidia-rta2024-functional-f@2`，recipe 是 `nvidia-rta2024-materialx-formal-300k-stage100k@1`。[N: `configs/learning/nvidia-rta2024-materialx-formal.json:6–7`; archived `correspondence.md:8–12`] 它对应 2024 方法，而非本文 2026 方法：

- native parameters 经 `K→64→64→64→64→8` encoder，随后 materialize hierarchical z8 latent；
- z8 投影两个 learned frames；两个 frame 内 direct fixed/query directions 共 12D，再接 z8，evaluator input 为 20D；
- evaluator `20→64→64→64→3`，hidden ReLU，输出 `exp(raw-3)`；
- formal lifecycle 为 300k steps、materialization at 100k，每步 evaluator/sampler 两个独立约65k batch，Adam/cosine；
- evaluator loss identity 冻结为 `log1p-l1@1`，因为 2024 论文只写 log-space L1 而未给 offset。[N: archived `correspondence.md:19–36`; current config `:80`]

该 route 对 2024 architecture/correspondence 标为 `faithful`，但这并不等于已实现 2026 改进。对当前 `src/`、`configs/`、`docs/` 搜索没有找到 2026 的 `instance_schedule`/culling scheduler、power-log loss、StableRusinkiewicz 或 SmeLU evaluator path。[N: repository audit 2026-08-29]

### 14.2 逐项影响分类

| 2026 组件 | 当前状态 | 分类 | 影响 |
|---|---|---|---|
| Multi-instance/pruning/batch resize | 未实现 | `not-applicable` 于当前 2024 faithful identity；若新增是新 recipe | 最终 runtime architecture 可不变，但 training recipe、seed policy、peak memory 与 selection rule 必须新登记 |
| Shared query generation | 当前 formal 已 online reference，但不是 64 instances broadcast | `not-applicable` | 可复用昂贵 reference query；需证明 candidate batch shape 与 GPU-resident source compatible |
| Power mapping `n=3` | 当前为 `log1p-l1@1` | `intentional-deviation` 若直接替换；应建新 loss identity | 当前 output 仍是 exponential，故 P 的梯度动机比 C default 更直接适用；但必须做 matched loss ablation |
| Full stable direction | 当前 two-frame direct directions，20D | `intentional-deviation` / 新 representation identity | 变成每 frame direct+half/diff，输入与第一层参数增大；需更新 shader、pack、parity、cost |
| LeakySmeLU | 当前 evaluator ReLU | `intentional-deviation` / 新 representation identity | 可针对 highlight facet，但不能以 S average 声称普遍 quality win |
| 2×16 compact decoder | 当前正式最大 3×64 | `budget/architecture variant`，不能冒充当前 identity | 2026 明说 method gain 随网络变大而下降；要检验 schedule 应先建 2×16 matched candidate，而非假定 3×64 同幅受益 |
| 2026 official MaterialX examples | 当前 source 为冻结的 MaterialX formal route | `source-domain adaptation` | 公开 FauxLeather/Bark/PatternedMetal 与论文 15 材质不匹配；当前 `american_walnut_veneer` 也不是 P asset |

### 14.3 不应静默修改当前复现

最安全的研究解释是保留 `nvidia-rta2024-functional-f@2` 不变，新增至少两个正交身份：

1. **training-only schedule recipe**：结构仍是指定 2024 evaluator，只改变 candidate population、batch geometry 与 selector；
2. **2026 representation-side candidate**：full stable directions + power loss + LeakySmeLU，可再与 schedule 做 factorial。

如果两者一次性全部替换，结果改善无法归因于 variance reduction 还是 representation/loss 改变，也破坏已建立的 correspondence。特别是当前 2024 formal 同时训练 evaluator 与 sampler，而本文 2026 主实验以 BSDF fitting loss 选 candidate；是否复制/prune sampler、用何种混合 loss 排名均未被 P/S 验证，必须保持为未报告的新设计问题。[I]

## 15. 可证伪的迁移假设 `[I]`

下列是候选实验假设，不是任务 hard gate；任何 numerical threshold 都应在 formal 前按项目协议另行冻结。

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：fixed-budget multi-instance 能降低当前 compact evaluator 的 run-to-run spread | P Fig.1/5、Table 2 | 当前 source family 的 2×16 candidate 同样 optimization-limited，且 early ranks 可预测 | 同 source/query、2×16、相同总 `sample×network` evaluations；`1@B` vs geometric `K→1/B→B`；预先固定 top-level seeds 与 selector | source locator、query recipe、model、loss、optimizer、steps、总 evaluations、validation | per-seed local metrics、median/p90/IQR/variance、failure-tail、best observed、wall-time/VRAM，bootstrap CI | training-only；部署仍单 evaluator | matched CI 不支持 spread/tail 降低，或质量分布恶化；若仅 best 改善而 variance 不降，也否定“taming variance”表述 |
| H2：shared queries 在当前 GPU online reference 中改善 ranking 且降低 datagen cost | S §B；P 3.01×/Table 3 | 不同实例的多样性主要来自初始化，共享 query 不会使 basin collapse | 同一 multi-instance schedule，shared broadcast vs per-instance independent query；各实例 network evaluations 相同 | candidate seeds、query distributions、optimizer、selection checkpoints | top-k retention、survivor/best、unique reference evaluations、wall-time、peak memory | training-only | shared 的 retention/quality劣化且 datagen收益不足，或候选梯度高度同质化导致 best observed 下降 |
| H3：在当前 exponential-output evaluator 上 `n=3` power loss 改善低值材质细节而不牺牲 peak/energy Pareto | P Eq.3–4/Fig.8；S Table3 15/15 | 当前 `exp(raw-3)` 保留同类 gradient pathology；reference dynamic range相近 | 单实例、同架构/seed/query/budget，`log1p-l1@1` vs power mapped L1；不得同时换 activation/directions | output transform、optimizer、direction features、training schedule、source split | HDR FLIP、linear-f/local relative errors、diffuse/peak strata、energy、median/p90 与 CI | loss training-only | diffuse strata无改善，或 peak/energy恶化使整体 Pareto 不占优；若只 training loss变小但 image/local metrics不变亦不支持 |
| H4：full direct+stable half/diff 的收益超过其额外第一层成本 | P Fig.6；S Table6–7 | 当前 learned frames 后仍存在 grazing/singularity或 alignment ambiguity | 四项 raw-paper对照 direct/old/stable/full；另做 iso-parameter control（调宽度或 projection）分离“坐标”与“参数更多” | loss/activation/schedule/source/query、两 learned frames | local angular strata、HDR FLIP、grazing/normal-map slices、params/MAC/latency | runtime evaluator改变但静态有界 | full 的质量增益在 iso-parameter control 消失，或增加成本后不改善 Pareto；出现 material-specific退化且无可预测选择规则 |
| H5：LeakySmeLU 能以小固定成本减少 sharp-highlight facets | P Eq.5/Fig.9；S Tables4–5 | 当前 small decoder的 ReLU facet是关键 error mode | ReLU/LeakyReLU/LeakySmeLU；先用 C constants `.5/.01`，不同时换其它轴；同参数数 | architecture、loss、directions、seeds、budget | highlight angular derivative/curvature、artifact-focused renders、HDR FLIP、loss、latency | runtime activation改变但静态有界 | artifact metric/盲评没有改善，或像 Glazed ceramic 一样的质量损失超过任何平滑收益；LeakyReLU在相同成本支配 |
| H6：early rank calibration 能在新 source family 上给出安全 pruning time | P Fig.3/Table1；P内部统计冲突提示需自校准 | 新材质 early-to-final rank 相关性不会自动等于 P | pilot 中保留全部 candidates 到终点，仅离线模拟不同 boundary/factor；与 random selector比较 | source strata、candidate seeds、query stream、full-run budget | Spearman/rank retention、top-k recall、survivor regret，按 material bootstrap | diagnostic/training-only | 在计划 boundary 前 rank近随机，top-k recall不稳定，或不同材质需要相反 schedule；此时静态 P schedule 不应进入 formal |

这些假设的执行顺序应先 H6（只诊断 ranking），再 H1/H2（训练策略），最后分别 H3/H4/H5（loss/representation）。顺序用于避免归因混淆，不把后续轴排除出本研究范围。[I]

## 16. 证据索引

### `P` Main paper

- `P-1`：Abstract、§1（pp.1–2）——compact network variance、baking/compression domain、贡献边界。
- `P-2`：§2（pp.2–3）——parallel training、successive halving、distillation/pruning、batch adjustment related work。
- `P-3`：§3、Fig.2–3（pp.3–4）——cost model、early rank、general `K/P` pruning 与 batch annealing。
- `P-4`：Fig.1（pp.1–2）——54k/947 parameters、64 initializations、大/小/ours 的容量与稳定性区别。
- `P-5`：§4.1、Fig.4（pp.4–5）——record fields、encoder/decoder、15 个材质、Glorot/Adam。
- `P-6`：§4.2–4.3、Fig.5（pp.5–6）——100k、fixed-cost baselines、64 runs、复杂度解释。
- `P-7`：Table 1–2 与 adjacent prose（p.6）——retention/relative loss，以及 15-vs-24 trial、79.1-vs-74.4 内部冲突。
- `P-8`：Table 3 与 §4.3（pp.6–7）——shared data 3.01×、RTX 5090 timing、28–32%。
- `P-9`：§5 Eq.1–2、Fig.6（p.7）——old singularity、shortest-arc stable difference、normal-map caveat、full input。
- `P-10`：§5 Eq.3–4、Fig.7–8（pp.7–8）——exp output、mapped L1、log/power gradient与视觉现象。
- `P-11`：§5 Eq.5、Fig.9（p.8）——LeakySmeLU、faceting、LeakyReLU tradeoff。
- `P-12`：§6（p.8）——只选择 candidate、SWA future、runtime adoption claim。

### `S` Supplemental

- `S-1`：§A, Fig.1（p.1）——2×/4×/8× pruning schedules 与 negligible-impact 定性结论。
- `S-2`：§B, Tables 1–2（pp.2–3）——shared vs independent data 全表。
- `S-3`：§C, Fig.2（pp.3–4）——Adam/Muon/SOAP 的 64 trajectories。
- `S-4`：§D, Table 3（pp.4–5）——log/power HDR FLIP，5 runs，shared scene。
- `S-5`：§D, Tables 4–5（pp.5–6）——activation HDR FLIP 与 loss。
- `S-6`：§D, Tables 6–7（pp.6–7）——direction parameterization HDR FLIP 与 loss。

### `C` Official code at `305b4b9c…`

- `C-1`：[default config](https://github.com/NVlabs/neuralappearance/blob/305b4b9c12e679398c487603dd8245c3f348526c/configs/default.json)——训练 phases、Adam、instance/batch schedules、loss、network、direction sampling、augmentation。
- `C-2`：`configs/single_instance_training_large.json`、`single_instance_training_small.json`、`multi_instance_training_small.json`——README 的 Figure 1 examples。
- `C-3`：`neuralappearance/training/batch_instance_scheduler.py:7–55`——等长 period 与 batch parsing。
- `C-4`：`neuralappearance/train.py:248–524,625–747`——FP16/FP32 optimizer wrapper、loss scaling、64-batch data blocks、shared broadcast、block-quantized boundary handling 与 latest-block loss reporting。
- `C-5`：`neuralappearance/model/neural_material_model.py:112,304–324`——instance construction 与 `argsort` pruning。
- `C-6`：`neuralappearance/model/half_diff_parameterization.slang:58–75`——stable shortest-arc exact implementation。
- `C-7`：`neuralappearance/neuralnetworks/losses/l1_with_power_log.slang:10–44`——power mapping exact implementation。
- `C-8`：`neuralappearance/neuralnetworks/components/smelu.slang:10–29`——leaky SmeLU constants/forward。
- `C-9`：`neuralappearance/model/bsdf_decoder_flattener.py:16–68`——`WiWo/WhWd/WhWdZiZo/WhWdWiWo` layouts。
- `C-10`：Git `b59ac4c..305b4b9c`——只修 Slang filename casing。

### `A` Author/project pages

- `A-1`：[NVIDIA Research publication page](https://research.nvidia.com/labs/rtr/publication/bitterli2026taming/)——正式 PDF、supplemental、code、DOI 入口。
- `A-2`：[Jan Novák page](https://www.jannovak.info/publications/NAP-smallnets/index.html)——作者侧书目与入口。
- `A-3`：[SIGGRAPH schedule](https://s2026.conference-schedule.org/presentation/?id=papers_1751&sess=sess139)——88%/38% 摘要；与 P/S metric 对应未报告。

### `N` NeuralShading

- `N-1`：`.trellis/tasks/archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md:8–36`——当前 2024 formal identity、architecture、loss、lifecycle、faithfulness。
- `N-2`：`configs/learning/nvidia-rta2024-materialx-formal.json:6–7,80`——当前 correspondence/recipe/loss identity。
- `N-3`：`.trellis/tasks/archive/2026-08/08-27-reference-material-candidates/research/nvidia-2026-materialx.md`——官方 2026 repo/material asset 范围审计。
- `N-4`：2026-08-29 repository `rg` audit——当前 `src/configs/docs` 未实现 2026 multi-instance/power/stable-half-diff/SmeLU path。

### `I` 本报告推导

- `I-1`：由 P full input、two frames、z8、2×16/4×128 与 C frame projection 复核 947/54,243 参数。
- `I-2`：由 P schedule 复核每步 65,536 与总 6.5536B sample–network evaluations。
- `I-3`：根据 output derivative 区分 P Exp 与 C ScaledSigmoid 的 loss-gradient landscape。
- `I-4`：把 schedule、loss、direction、activation 分成 training search、objective 与 runtime function-family 三类容量。
- `I-5`：§15 的迁移假设、matched controls 与 falsification conditions。

## Evidence review

```text
author_worker: /root/taming2026
reviewer: /root/nbrdf2021
reviewed_at: 2026-08-29
sources_rechecked:
  - NVIDIA main PDF，9 页，SHA-256 EB979B03328353C03B7606DD03462DE9F10873F777764E965C70C21DB61A6F5C；逐页渲染复核 Fig.1–9、Eq.1–5、Table 1–3、图注与脚注
  - NVIDIA supplemental PDF，7 页，SHA-256 93BA59966A61628B6451FA9CEB582B8614673A50D6EAD7EC8985267836389F55；逐页渲染复核 Fig.1–2、Table 1–7
  - NVIDIA Research 与 Jan Novák 第一方项目页、SIGGRAPH 2026 schedule；未发现勘误或额外 talk/slides
  - NVlabs/neuralappearance commit 305b4b9c12e679398c487603dd8245c3f348526c；复核 commit metadata、README、default/derived configs、scheduler、training loop、pruning、direction、loss、activation 与 decoder layout
  - NeuralShading 当前 formal config 与 archived 2024 correspondence/material-asset audit
findings_closed:
  - 修正 Figure 1 panel locator：reference/large/small/ours 分别为 (a)/(b)/(c)/(d)
  - 明确 paper 25k 等长 phase 与 code 64-step block lifecycle 的差异，并复核总 network-evaluation 预算仍恒定
  - 修正 shared-data 实现描述：每个 generation block 生成 64 个共享 batch，而非只生成一个 batch
  - 将 Table 1 的实际执行方式保持为未报告；仅 Table 2 有明确 disable-culling counterfactual caption
  - 保留并交叉验证 Table 1 的 15-vs-24 trials、79.1-vs-74.4、98.7-vs-99.4 内部冲突，supplemental 支持表格版本但不能构成勘误
  - 复核 power mapping/Exp 梯度、LeakySmeLU 公式与代码常量、stable shortest-arc 公式和 full-direction 消融
  - 复核 947/54,243 与 released WhWdZiZo 819/53,219 的语义参数算术，保留为 P/C formal-config gap
  - 确认 multi-instance 仅为 training/compiler policy，部署 checkpoint 只保留单个 latent/decoder，不把训练候选写成 runtime ensemble
  - 复核 N/I 晚于 P/S/C/A、当前 2024 formal identity 未被 2026 recipe 静默改写，promotion trigger 保持条件化
remaining_evidence_gaps:
  - Paper Table 1 表格与 prose 的 trial count/aggregate 冲突没有官方勘误
  - 88% variance reduction 与 38% average loss reduction 的定义未映射到 P/S
  - 正式 15 个材质、query recipe、seeds/raw logs 未公开
  - P Exp/full-direction/947 与 released default ScaledSigmoid/WhWdZiZo/819 不一致
  - Muon/SOAP 与 paper exact Adam hyperparameters 未报告
  - runtime latency/memory/fetch/precision 未报告
review_status: evidence-reviewed
```

### 完成检查

- [x] main paper 已完整阅读，全部 9 页已渲染；关键公式/图/表/图注/脚注已视觉核对；
- [x] supplemental 的全部 7 页、appendix 与公开勘误可用性已检查；
- [x] official code/config/data 已审计并固定 commit；
- [x] architecture、training、runtime 和主要结果均有 locator；
- [x] 失败尝试与较差消融已分类，未从最终方法虚构开发失败；
- [x] paper/code gap、P内部冲突和“未报告”已保留；
- [x] `I` 分析晚于 P/S/C/A 事实层，没有改写作者结论；
- [x] NVIDIA 影响引用当前真实 `N` evidence，并保持 2024 identity 边界；
- [x] 假设包含 matched control、冻结轴、部署类别和证伪条件；
- [x] 独立 evidence review 已完成；报告保持 `evidence-reviewed`，未越权声明 `complete`。
