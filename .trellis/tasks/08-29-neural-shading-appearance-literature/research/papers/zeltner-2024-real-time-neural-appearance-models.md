---
paper_id: "zeltner-2024-real-time-neural-appearance-models"
title: "Real-Time Neural Appearance Models"
authors: "Tizian Zeltner, Fabrice Rousselle, Andrea Weidlich, Petrik Clarberg, Jan Novák, Benedikt Bitterli, Alex Evans, Tomáš Davidovič, Simon Kallweit, Aaron Lefohn"
year: "2024"
venue: "ACM Transactions on Graphics 43(3), Article 33; presented at SIGGRAPH 2024"
doi: "10.1145/3659577"
report_status: "evidence-reviewed"
main_source: "https://research.nvidia.com/labs/rtr/neural_appearance_models/"
supplemental_status: "available"
official_code_status: "audited"
official_code_commit: "305b4b9c12e679398c487603dd8245c3f348526c"
author_worker: "rta2024"
reviewer: "/root/nbrdf2021"
last_verified: "2026-08-29"
---

# Real-Time Neural Appearance Models

> 证据标签：`P` 为 2024 正文，`S` 为 2024 supplemental，`C` 为作者团队 2026 年公开、同时说明 2024 工作为“additional details”的后续官方代码，`A` 为 NVIDIA 官方项目页，`N` 为 NeuralShading 项目内证据，`I` 为本项目分析。除明确标注为 `I` 的段落外，不把后续代码默认值、当前项目适配或常见实践倒灌为 2024 论文事实。

## 1. 研究对象与报告边界

- Query：完整重建 *Real-Time Neural Appearance Models* 的表示、训练、采样、过滤、部署、成功/负结果，并分析其对当前 NeuralShading NVIDIA 复现与候选改进的含义。
- Scope：mixed（2024 first-party paper/supplemental/author page、后续 official code，以及当前项目内 correspondence）。
- Date：2026-08-29。

本文研究的是一种 **local、reflective、spatially varying neural material**：把一个昂贵的参考 SVBRDF 材质图离线烘焙为层级 latent texture、BRDF evaluator 和 analytic importance sampler，并把小型 MLP 直接内联进实时 path tracer 的 shader。运行时查询对象是表面点、方向与 footprint/LOD，不包含场景级 visibility、照明传输或体传输；场景级 path tracing 只是论文验证材质调用是否能在真实随机访问、分支发散和多 bounce 条件下运行的承载环境。[P §3–§4 p4–6; P §7–§8 p10–16]

本报告覆盖正式发表的 TOG 43(3) author PDF（17 页）、配套 supplemental（33 页）以及 NVIDIA 官方项目页。正文与 supplemental 已完整阅读；正文的公式 (1)–(9)、图 1–18、表 1–5、图注和 limitation 段落，以及 supplemental 的 Listing 1–6、表 1–11、reference-material graph、Stage scene 与方向切片图均做过页面视觉核对。`NVlabs/neuralappearance` 是 2026 年发布、主要对应 *Taming optimization variance in compact neural shading networks* 的后续官方实现；它能提供实现家族的强证据，但不是 2024 artifact，因而本报告把它单列为 paper-code gap，而不称作 2024 formal code。[A project page; C `README.md`:9–27]

本文的边界如下：

- **属于本文**：源材质的 local reflective BSDF 压缩，空间层级表示，learned shading-frame prior，BRDF/方向反照率预测，匹配 evaluator 的 importance proposal，LOD filtering，FP16 shader 内联，以及多材质/相干与发散执行分析。[P §3 p4–5]
- **不属于本文**：运行时材质编辑、未见材质泛化、delta 分量、可靠 refraction、displacement 的成功神经表示、texture compression、能量/互易硬约束、去噪器设计，以及完整光输运的学习。[P §3 p5; P §7.1 p10; P §9 p15–16]
- **与本项目关系**：它是当前 NVIDIA neural material 复现的直接方法来源，也是 local evaluator、`prepare/evaluate/sample/pdf` 分工和部署预算的 load-bearing 论文；它不是场景级 transport 方法。[N `correspondence.md` §1–§7]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---:|---|---|
| Main paper `P` | NVIDIA author PDF；项目页 `Paper`；DOI `10.1145/3659577` | 2026-08-29 | SHA-256 `E709C1B5C4F0F16EB7EDF848D29079E007E3546DEDB8B5DFE4EA6BF44D9D1002` | 2024 formal 方法、实验、限制的主证据；本地 locator 为 `.trellis/tasks/08-25-03-neural-baseline-and-candidate/scratch/nvidia-neural-materials-author-paper.pdf` |
| Supplemental `S` | NVIDIA 项目页 `Supplemental`，33 页 | 2026-08-29 | SHA-256 `4AFADFF6A6F0A0E6CA8B2FF92927DE7E7DF4350A20CBC260FFEFC8920BA08376` | optimizer、mollification、functional pseudocode、scene/material graph、逐材质误差和方向切片；本地 locator 为 `.trellis/tasks/08-25-03-neural-baseline-and-candidate/scratch/nvidia-neural-materials-author-supplemental.pdf` |
| Official code/config/data `C` | `https://github.com/NVlabs/neuralappearance`，`main` 固定到 `305b4b9c12e679398c487603dd8245c3f348526c`；falcor2 submodule pin `d629c967fa800af81cf5c916bfb2a825b012f473`；Apache-2.0 | 2026-08-29 | commit 如左；owned audit copy 的 `.git/HEAD` 与 `refs/heads/main` 均为该 commit | 后续官方实现家族证据；README 明确主要对应 2026 论文，**不是** 2024 artifact。audit copy：`scratch/workers/zeltner-2024-real-time-neural-appearance-models/official-code/` |
| Author page/talk/correction `A` | NVIDIA 项目页 `https://research.nvidia.com/labs/rtr/neural_appearance_models/`；NVIDIA Research publication record `https://research.nvidia.com/publication/2023-05_real-time-neural-appearance-models` | 2026-08-29 | 网页，无本地 hash | 作者、venue、paper/supp/video/image viewer 入口和高层方法说明；publication record 的 uploaded files 也只有 paper/supplemental。两页均未列 2024 code/data/checkpoint 或 correction/erratum |
| NeuralShading evidence `N` | 当前 `configs/learning/nvidia-rta2024-materialx-formal.json`、`src/ncls/learning/{methods/nvidia.py,models/nvidia_neural_appearance.py}`、NVIDIA runtime shaders/tests；归档任务 `08-27-faithful-nvidia-neural-materials` 的 `correspondence.md`/`prd.md`/`design.md`；`artifacts/nvidia-faithful/materialx-recorded-200k/formal-report.json` | 2026-08-29 | 当前源码/配置与归档 artifact 各自按 identity 读取 | 当前 `functional-f@2` 方法身份与 bare-`f` contract；旧 `functional@1` correspondence/200k 结果；二者不能混作同一 evidence |

检索结果与材料缺口：

- 作者项目页提供 paper、supplemental、video 和 image viewer；NVIDIA Research publication record 的 uploaded files 也只有 paper 与 supplemental。两个 2024 第一方入口都没有 source code、训练 config、原始五材质资产、checkpoint、训练 log、逐 seed 统计或勘误链接。[A project page; A NVIDIA Research publication record]
- 官方 GitHub 仓库 README 明确称其伴随 2026 *Taming optimization variance...*，把 2024 工作列为 additional details；因此 commit 虽由同一研究团队发布，也只能解释后续实现选择，不能填充 2024 “未报告”字段。[C `README.md`:9–27]
- supplemental 的模型切片是强定性证据，但图中 PDF 为便于对比按常数缩放；它不是采样分布绝对归一化误差的量化验证。[S §7 p15]

## 3. 原论文的问题、假设与贡献边界

作者把目标定义为：给定现有 target/reference SVBRDF (f(\mathbf{x},\omega_i,\omega_o))，构造可实时求值的 (g\approx f)，同时保留高分辨率空间变化、低 roughness layer、glint、污渍和各向异性，并能在低样本数下进行 LOD filtering 和 importance sampling。[P §3 Eq. (1)–(3) p4–5]

论文的设计假设是：

1. 目标 appearance 可由 local reflective SVBRDF 描述，且源材质能在训练时在线输出 BSDF 与各层 surface parameters。[P §3 p4–5; P §5.2 p7]
2. 运行时可牺牲 editability，接受“编辑后重新 bake”，换取统一、紧凑且固定结构的 evaluator/sampler。[P §3 p5]
3. 受实时预算限制，直接增加 MLP 宽深不是主要解法；应通过 learned shading frame 和 analytic sampling family 把 graphics prior 放进网络。[P §4.2–§4.3 p5–7]
4. 纹理尺度变化不应只在输出后处理，而应让不同 MIP level 的 latent 直接表示对应 footprint 下过滤后的 BRDF。[P §4.1 p5]
5. shader 性能不仅取决于网络 MAC，还取决于 wave/warp 内 material coherence；同一模型需要 coherent tensor-core path 和 divergent packed-FMA path。[P §7.2–§7.3 p11–13]

作者声称的主要贡献可拆成四组：

- **表示**：8-channel hierarchical latent texture 加小型 neural decoder，用两组 latent-conditioned shading frames 表达复杂空间变化。[P §4.1–§4.2 p5–6]
- **采样**：另一个小 MLP 输出 tilted diffuse 与 non-centered anisotropic correlated GGX 的混合参数，保持 closed-form sampling/PDF。[P §4.3 p6; P Appendix A p16–17]
- **训练**：先用源材质参数 encoder amortize 高分辨率 latent 学习，再 materialize texture 并直接 finetune；数据全部由 reference 在 GPU 上在线生成。[P §5 p6–8]
- **系统**：为 ray-tracing shader 生成 FP16 网络代码，按 coherence 在 tensor core 和 packed FMA 实现间选择，展示完整 path-traced scenes 的质量—时间权衡。[P §7–§8 p10–16]

这些贡献不等于“任意材质编译器”：作者只在五个手工制作的 layered materials 与 MERL measured BRDF 定性切片上展示，未给未见材质或未见参数状态的 holdout 泛化实验。[P Table 1 p3; S §7.1 p15]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | 可查询的 reference SVBRDF；训练 sample 还导出每层 normal、tangent、albedo、roughness、layer weight 等 (\mathbf{k}(\mathbf{x})) | layered graph 可含 20–54 nodes、2–5 layers、43–143 parameters、3–16 textures；单 sample 可超过 100 floats | P Table 1 p3; P §5.1–§5.2 p6–7 |
| Spatial input | source UV parameterization 与 surface point (\mathbf{x})；finest latent resolution 对齐源纹理 | 2D UV；支持多个 4k tiles 的示例 | P §4.1 p5; P §5 p6 |
| Runtime query | (\mathbf{x},\omega_i,\omega_o) 加用于 LOD 的 UV-space projected footprint；sampler 给定 (\mathbf{x},\omega_i,\mathbf{u}) 产生 (\omega_o) | 方向在上半球；论文 sampler 的随机数 (\mathbf{u}\in[0,1)^2) | P Eq. (1)–(4) p4–6; P §4.3 p6 |
| Direction coordinates | 正文 Eq. (1) 约定 (\omega_i) 为 incident、(\omega_o) 为 outgoing；两者均投影到两组 learned shading frame。supplemental renderer Listing 5 却把 view 方向传给 `wi`、light/sample 方向传给 `wo`，所以复现必须按 call semantics 显式映射，不能只按变量名 | 每 frame 产生 3D local coordinates，2 frame × 2 directions = 12 scalars | P §3 p4; P Eq. (4) p6; S Listing 2/5 p3–4,8–9 |
| Evaluator output | 正文以 BRDF `f` 叙述；supplemental functional decoder 返回 RGB `f(wi,wo) * dot(n,wo)`，其中 Listing 5 的 `wo` 是 light/sample query direction；raw transform 为 `exp(raw - 3)`。可选另预测 RGB directional albedo | 3 或 6 scalars；非负；linear response，但 P 的数学量、S 的调用命名与 shader-return measure 必须分开登记 | P §3 Eq. (1)–(3) p4; P Fig. 4 p4; S Listing 2/5 p3–4,8–9 |
| Directional albedo | (\alpha(\mathbf{x},\omega_o)=\int_{H^2} f(\mathbf{x},\omega_i,\omega_o)\cos\theta_i d\omega_i) | RGB，半球积分 | P Eq. (2) p4 |
| Sampler output | analytic mixture 参数、sample transform (W)、full mixture PDF (p(\omega_o;\mathbf{x},\omega_i)) | 9 raw scalars；2-lobe density | P §4.3 p6; S Listing 3–4 p5–8 |
| Validity/domain restrictions | reflective local surface scattering；无 delta；rough refraction 仅 preliminary；不保证 reciprocity/energy conservation | 反射半球；不覆盖 transmission/delta/volume | P §3 p5; P §9 p15–16 |

一个容易造成接口误读的细节是：论文公式用传统 `f` 描述 reference，但 supplemental Listing 2 明确把 cosine 烘焙进网络返回值。项目若要输出线性 `f`，必须显式选择并版本化两类适配之一：对 supplemental response 做 cosine conversion，或像当前 `functional-f@2` 那样直接以 bare `target_f` 训练同一 `exp(raw-3)` head；不能把二者混成同一输出语义。[S Listing 2 p3–4; N `configs/learning/nvidia-rta2024-materialx-formal.json`:6–7; N `src/ncls/learning/methods/nvidia.py`:523–534]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

训练期第一阶段：

\[
  (\mathbf{x},\text{LOD})
  \xrightarrow{\text{reference parameter query}}
  \mathbf{k}(\mathbf{x},\text{LOD})
  \xrightarrow{E_\phi}
  \mathbf{z}\in\mathbb{R}^8
  \xrightarrow{\text{frame prior}+D_\theta}
  \widehat{f(\omega_i,\omega_o)\,\mathrm{dot}(n,\omega_o)}.
\]

同一 latent 还与 (\omega_i) 输入 sampler MLP，生成两叶 analytic density 参数。论文每步为 BRDF decoder 与 sampler 各处理一批 65k samples，并同时优化两者；它没有披露这两批的 RNG/stream independence。sampler 的 KL 路径对 latent `detach`，避免 sampling loss 扭曲 evaluator 表示。[P §5.1–§5.2 p7]

当 encoder “sufficiently trained” 后，作者遍历所有 texel/MIP，把 encoder 输出 materialize 成层级 latent texture，丢弃 encoder，再直接优化 latent texels 和 decoder。论文没有给切换 step 或充分训练判据。运行时只保留 latent hierarchy、frame projection、evaluator 和 sampler，不携带 source graph 或 encoder。[P Fig. 6 and §5.1 p6–7]

运行时：由 UV footprint 面积得到 fractional LOD；以 Russian roulette 在相邻整数层中选一个，在选中层内做 bilinear latent fetch；从 z 线性解码两组 frame，把方向投影后与 z 拼接进 evaluator；需要采样时由 z 与 incident direction 解码 analytic mixture，一次解码出的 sampler parameters 可供 sample 和 PDF 使用。[P §4.1–§4.3 p5–6; S Listing 1–4 p3–8]

### 5.2 持久化表示

- 每个空间位置/尺度使用 (z=8)；正式 runtime 以两张 RGBA FP16 mipmapped textures 存储。[P §4.1 p5; P §7.1 p10]
- finest level 跟随 source texture resolution 与 UV texel density；每个 coarse level 学的是该 footprint 下 filtered BRDF 的 latent，不是简单对 level 0 latent 做 downsample。[P §4.1 p5]
- 对高分辨率 Teapot ceramic，作者称源材质为 14 个 4k×4k tiles、235M texels；若把每 texel 的 8 latent 都独立计入，达到约 2.5B latent parameters，解释了 direct optimization 的稀疏更新问题。[P §5 p6]
- 论文报告 8-channel FP16 latent 对一个 4k tile 的存储为 256 MB，network weights 约 9.3 kB（2×16 evaluator）到 37 kB（3×64 evaluator）。该 256 MB 数值是否已把完整 mip-chain overhead 计入，正文措辞没有进一步拆分，故不擅自加 (4/3) 系数。[P §8.3 p15]
- network weights 和 latent 从 FP32 master post-training 转成 FP16；未使用 texture compression，未探索 INT8。[P §5.2 p7; P §7.1 p10]
- 通常每个 large latent texture 与自己的 encoder 独立训练；作者说 encoder 理论上可跨多个材质或整个 reference parameter space，但没有把这点作为主要五材质实验的共同设置。[P §5.1 p7]

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Encoder（训练期） | source parameters (K=\dim\mathbf{k}) | `K→64→64→64→64→8` | 4 个 hidden ReLU；输出 activation 未报告 | z8 | 实践中 per trained latent/material；运行时丢弃 | P Fig. 6 p6; P §5.1 p7 |
| Frame projection | z8 | 单个无 bias、无 activation 的 linear layer `8→12` | 每组 (n,t) 分别加 canonical (n_0=(0,0,1),t_0=(1,0,0)) 后 normalize；不做正交化。`b` 的 normalization 存在 P↔S 冲突，不能在此合并 | 2×(n,t)，即 12 | per baked material/asset，跨其 texel/MIP 共享；z per texel | P §4.2 Eq. (4) p5–6; P §9 p15; S Listing 2 p3–4 |
| BRDF evaluator，small | z8 + 2 frame 下 wi/wo，共 20 | `20→16→16→3` | hidden ReLU；输出 `exp(raw-3)` | supplemental functional measure 为 RGB `f(wi,wo) * dot(n,wo)` | per baked material/asset，跨其 texel/MIP 共享 | P Fig. 4 p4; P §8.1 p13–14; S Listing 2 p3–4 |
| BRDF evaluator，medium | 同上 | `20→32→32→3` | 同上 | 同上 | 同上 | P §8.1/Table 3 p13–14 |
| BRDF evaluator，large | 同上 | `20→64→64→64→3` | 同上 | 同上 | 同上 | P Fig. 4 p4; P §8.1/Table 3 p13–14 |
| 可选 albedo head | evaluator hidden trunk | 正文 Figure 4 把最后输出扩为 BRDF RGB + albedo RGB，两个分支均画有 `exp`；exact albedo offset 未披露 | albedo 分支为 exponential nonnegative output；用 L2 监督 | 额外 RGB (\alpha) | per baked material/asset，和 BRDF 共用 trunk | P Fig. 4 p4; P §5.2 p7; P §6.4 p10 |
| Importance sampler | z8 + wi3 = 11 | `11→32→32→32→9` | hidden ReLU，raw linear output，再做 analytic parameter transform | 9 个 raw parameters | per baked material/asset，跨其 texel/MIP 共享；z per texel | P Fig. 4 p4; S Listing 3 p4–5 |

正文 Figure 4 给出最大 evaluator 的 `3×64` 图；supplemental Listing 2 是 **2×32 functional example**，不是所有实验模型统一宽度。三种 evaluator 都在 §8 明确比较，sampler 始终为 3×32。[P §8.1 p14; S §2.1 p3–8]

### 5.4 条件化、坐标变换与物理先验

**Learned shading frames。** 每个 z 生成两组随空间变化的 (n,t)，将 wi/wo 分别投影到两个基底。作者的理由不是让 frame 本身成为可解释 normal map，而是把昂贵的乘法式旋转交给显式 prior，避免小 MLP 用其非线性容量重新学习方向旋转。n/t 均 normalize 但不正交；正文 §9 把第三轴写成归一化的 (n\times t)/\lVert n\times t\rVert，而 supplemental Listing 2 明确使用未 normalize 的 `cross(N,T)`，并注释 resulting bitangent 可能既非 unit length 也不与 tangent 正交。后续代码也选择后一实现。故“两个 learned projections、不正交”是稳定拓扑，而 bitangent normalization 是未解析的第一方冲突，不能把整条 frame 算术无条件称为唯一 formal TBN。[P §4.2 p5–6; P §9 p15; S Listing 2 p3–4; C `rotation.slang`:82–105]

**Hierarchical filtering。** fractional LOD 不做 trilinear latent interpolation，而在相邻 level 间随机选择一个，并只读取一个 level。作者观察其质量更高，推测原因是 trilinear 会迫使语义可能差异很大的 level 间线性路径也解码为合理 BRDF。该选择把跨层插值 bias 换成“小而有界”的方差。[P §4.1 p5]

**Analytic sampling prior。** sampler 输出 tilted cosine diffuse 与 non-centered anisotropic GGX specular mixture。9 个输出次序为：

1. (\alpha_x,\alpha_y)：`1e-4 + 0.5(1+tanhApprox(raw))`；
2. correlation (\rho=\text{tanhApprox}(raw))；
3. specular mean slopes ((\mu_{sx},\mu_{sy})=\text{sinhApprox}(raw))；
4. diffuse mean slopes ((\mu_{dx},\mu_{dy})=\text{sinhApprox}(raw))；
5. specular/diffuse mixture logits 经 exp 后归一化。

其中 `tanhApprox(x)=x/sqrt(1+x^2)`，`sinhApprox(x)=x*sqrt(1+x^2)`。正文 Appendix A 给出 mixture、slope-space transforms 和 Jacobian；supplemental Listing 4 用同一 `float2 u`，以 `u.x` 选择 mixture component 并在选中区间内重映射，返回完整 mixture PDF。[P Appendix A Eq. (5)–(9) p16–17; S Listing 3–4 p5–8]

本文没有 visibility、transport decomposition、light encoding 或 scene grid；不要从其 path-traced scene 实验推断这些机制存在。

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset/source assets/scenes | 五个高质量手工 layered material：Teapot ceramic/handle、Slicer handle/blade、Inkwell；统计见 Table 1。另在 supplemental 展示 100 个 MERL measured BRDF 方向切片 | P Fig. 2/Table 1 p2–3; S §7.1 p15 |
| GT/reference | layered material graphs 直接求值；过滤 GT 在 target footprint 内做 Gaussian spatial sampling/averaging；图像 reference 使用原材质 path tracing | P §5.2 p7; P Fig. 5 p5; P §8 p13–15 |
| Train/validation/test split | 五材质没有报告 formal train/validation/test split；MERL 实验把 100 个 material index 以 one-hot 输入 encoder，但未报告 holdout split，故不能解释为未见材质泛化 | S §7.1 p15 |
| Spatial sampling | uniform UV sampling | P Fig. 6 p6; P §5.2 p7 |
| Direction sampling | uniform sampling half vector 与 difference vector，再转换为 wi/wo | P §5.2 p7 |
| MIP sampling | 离散采样 level，使用偏向 fine levels 的 exponential distribution；具体分布参数未报告 | P §5.2 p7 |
| Filtering/footprint | level 对应 Gaussian footprint；空间 sample 数与 filter area 成比例；coarse encoder inputs 使用 LEAN 预过滤 | P §5.1–§5.2 p6–7 |
| Directional mollification | 训练前 20k iterations；围绕 wo 的 cone 从 10° 以 cosine schedule 降至 0°；每 target 平均 256 个 wo samples | S §1 Eq. (2) p2 |
| Sampler training queries | 由当前 learned sampler 自身采出 wo，PDF 以当前 learned BRDF 为 target 做 KL optimization；2024 没披露 KL 方向、target normalization 或 estimator | P §5.2 p7 |
| Online/offline generation | 所有训练 sample 在单 GPU 上 online 生成，不保存预生成 corpus | P §5.2 p7 |

五个 source 的规模为：

| Material | Nodes | Layers | Parameters | Textures (channels) | RGB MTexels |
|---|---:|---:|---:|---:|---:|
| Teapot ceramic | 37 | 5 | 121 | 5 (11) | 1174 |
| Teapot handle | 41 | 2 | 91 | 11 (19) | 152 |
| Slicer handle | 20 | 5 | 43 | 3 (7) | 201 |
| Slicer blade | 54 | 3 | 114 | 16 (40) | 324 |
| Inkwell | 49 | 5 | 143 | 4 (11) | 201 |

[P Table 1 p3]

作者没有公开五个 source assets，因此上述 graph 复杂度能说明 target 难度，却不足以独立复现同分布的训练查询。supplemental 给出 graph 截图，但不是可执行参数、纹理或材质包。[S §3 p11]

## 7. Loss、optimizer 与训练 lifecycle

| 项 | 正式配置 | locator |
|---|---|---|
| Target/output transform | evaluator functional output为 `exp(raw-3)`，学习 cosine-weighted RGB BRDF | S Listing 2 p4–5 |
| BRDF loss | log space L1；2024 没披露 exact log formula、epsilon/offset 或 RGB aggregation | P §5.2 p7 |
| Sampler loss | 用 KL divergence 让 learned PDF 逼近当前 learned BRDF；latent 从 KL 路径 detach。2024 正文没有给 KL 的方向、归一化或 gradient estimator | P §5.2 p7 |
| Albedo loss | 对 Equation (2) 的 one-sample MC estimate 做 L2 | P §5.2 p7 |
| Optimizer | Adam，β1=0.9、β2=0.999、ε=1e-7、weight decay=0 | S §1 p2 |
| LR schedule | cosine，从 (10^{-3}) 到 (10^{-4})；supplemental Eq. (1) 给出形式 | S §1 Eq. (1) p2 |
| Batch/query count | 每 iteration 两个各 65k 的 respective batches：一个 evaluator，一个 sampler；论文没有披露二者是否使用独立 RNG stream | P §5.2 p7 |
| Steps | 总计 300k iterations，约 40B online material samples | P §5.2 p7 |
| Stages | 第一阶段 encoder output 直接送 decoder；encoder sufficiently trained 后 materialize 全 texel/MIP、drop encoder，第二阶段直接 finetune latent+decoder；论文中 evaluator 与 sampler 同时训练 | P Fig. 6/§5.1–§5.2 p6–7 |
| Initialization/seed | **未报告**：网络/latent initialization、seed 数与选择规则均未披露 | P/S 全文核对 |
| Stage switch/checkpoint | **未报告**：encoder→latent 的 exact iteration、“converges”的判据、checkpoint selection 未披露 | P §5.1 p7 |
| Hardware/time | 单张 RTX 4090，约 4–5 h/material；direct optimization 可近乎翻倍并达到 10 h | P §5.2 p7; P §6.2 p9 |
| Training precision | FP32 master；未研究 mixed precision；load time 转 FP16 | P §5.2 p7 |

训练 lifecycle 必须按论文事实理解为“共享 latent、两个 respective batches、simultaneous evaluator/sampler optimization”，但不能把未披露的 stream independence 写成论文事实。后续官方代码的 lifecycle 已发生变化：BSDF encoding、可选 direct-opt finetune、sampler、auxiliary 按阶段顺序执行，sampler 对 frozen neural BSDF 训练。[C `neuralappearance/train.py`:857, 880–1052, 1077–1147; C `neuralappearance/model/sampler.py`:12–17]

### 2024 paper/formal 与 2026 code/default 的分层登记

| 层级 | 可确认内容 | 不能据此声称的内容 |
|---|---|---|
| 2024 paper/formal `P/S` | 300k；两批各 65k；encoder→latent；simultaneous evaluator/sampler；ReLU；Cartesian wi/wo 经两个 learned frames；3×32 sampler；LR 1e-3→1e-4 | exact log transform、stage switch、KL 方向/归一化/estimator、两批 RNG 关系、seed/model selection |
| 2026 code paper configs `C` | repo README 称 Figure 1 configs 比较 large/small single-instance 与 small multi-instance optimization；属于 2026 paper experiment | 不是 2024 Figure 15/16 的 formal config |
| 2026 `configs/default.json` `C` | example `FauxLeather.mtlx`；BSDF 100k、finetune 0、sampler 50k、aux 50k；batch 1k→4k→16k→64k；deterministic；z8、1 mip；encoder `[64,64,64]` LeakyReLU；decoder `[16,16]` SmeLU、ScaledSigmoid、StableRusinkiewicz/WhWdZiZo；sampler `[16,16]` LeakyReLU | 不能回填成 2024 paper defaults，也不能视为论文五材质 recipe |

[C `README.md`:92–107; C `configs/default.json`:31–50, 77–122, 144–193]

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path | hit→UV footprint/LOD→一次 latent fetch→learned frames→BRDF eval；需要 importance sampling 时，另解码 analytic params，再做 sample/PDF。path tracing 中通常对 sampled direction 与 NEE direction 分别求 evaluator/PDF | P §4 p5–6; S Listing 5 p8–10 |
| Call frequency | 每个 non-delta surface hit 至少一次 sampler decode；evaluator/PDF 次数取决于 continuation 与 NEE/MIS。论文 sampler MLP 参数可在同一 hit 内供 sample/PDF 复用 | S Listing 5 p8–10 |
| Evaluator sizes | 2×16、2×32、3×64；论文未逐项报告 MAC/FLOP，不能以粗略参数数代替实测 shader cost | P §8.1/Table 3 p14 |
| Sampler size | 固定 3×32→9；normalizing-flow baseline 通常需每 hit 4 次 MLP evaluation，analytic proxy 只需 1 次 sampler decoder | P §6.3/Fig. 12 p9–10 |
| Shared/per-asset bytes | 作者只按 network configuration 报告总 network weights 约 9.3 kB（2×16）至 37 kB（3×64），没有把 frame/evaluator/sampler 逐模块拆账；z8 FP16 为 256 MB/4k texture tile | P §8.3 p15 |
| Texture/feature fetches | z8 以两张 RGBA FP16 texture 表示；stochastic adjacent-MIP 只选一层，再 bilinear。具体硬件采样指令数、cache hit 与 texture hierarchy overhead 未报告 | P §4.1 p5; P §7.1 p10 |
| Precision | runtime weights/latent FP16；coherent path 使用 tensor-core 16×16 blocks；divergent path 对两个 packed 16-bit weights 做展开 FMA，并使用 128-bit vector loads | P §5.2 p7; P §7.1–§7.2 p10–12 |
| Backend | Slang code generation、Falcor、D3D12/DXR、NVIDIA RTX 4090 | P §7 p10–13; P §8 p13–15 |
| Coherence policy | wave/warp 同 material 时使用 tensor core；发散时用 packed-FMA；运行时按 warp coherence 动态选路径；SER 用于局部重排 path work | P §7.2–§7.3 p11–13 |
| Image-time protocol | 1920×1080，1 SPP timing，NEE+MIS，无 denoising，max path 6 vertices（含 camera/light），timing 时关闭 Russian roulette | P §8 p13 |
| Precompute/amortization | offline training 4–5h/material 不包含在 frame time；encoder 不部署。latent fetch、frame projection、sampler decode 和 path integration 均包含在 renderer path；shader compilation/codegen 时间未报告 | P §5.1–§5.2 p7; P §7.1 p10; P §8 p13 |

### 8.1 Shader 生成与两条执行路径

系统从 neural material description 为每个材质生成带 evaluation、sampling、PDF entry points 的 Slang shader module。作者没有调用通用推理 runtime，而是把 layer 展开成 shader math。发散路径把权重打包成 16-bit pair，并用 128-bit loads 降低访存/指令开销；相干路径通过给 DXC 增加的 intrinsics 暴露 NVIDIA tensor cores，以 16×16 block 计算。两者的 break-even 取决于一个 warp 中 hit 的 material coherence，故系统动态选择。[P §7.1–§7.3 p10–13]

这说明论文的“real-time”不是单一 MLP latency 声称：它建立在 NVIDIA GPU、定制 shader compiler path、FP16、warp coherence、SER 和完整 path distribution 上。换 backend 时，网络结构仍然静态有界，但原速度比不自动成立。[I，由 P §7–§8 推出]

### 8.2 正式图像级性能

Table 4 的完整数字如下；括号为相对 reference material 的 speedup，均为 1 SPP frame time：

| Scene/view | 2×16 | 2×32 | 3×64 | Reference |
|---|---:|---:|---:|---:|
| Inkwell view 1 | 3.64 ms (4.01×) | 4.36 ms (3.34×) | 9.94 ms (1.47×) | 14.58 ms |
| Inkwell view 2 | 3.26 ms (4.71×) | 4.16 ms (3.69×) | 10.93 ms (1.41×) | 15.36 ms |
| Stage view 1 | 3.15 ms (4.21×) | 3.71 ms (3.57×) | 6.31 ms (2.10×) | 13.25 ms |
| Stage view 2 | 3.30 ms (4.33×) | 4.32 ms (3.31×) | 7.67 ms (1.86×) | 14.29 ms |
| Stage view 3 | 4.29 ms (4.66×) | 5.73 ms (3.49×) | 11.02 ms (1.81×) | 19.98 ms |
| Stage view 4 | 3.49 ms (4.74×) | 4.39 ms (3.77×) | 8.68 ms (1.90×) | 16.53 ms |
| Stage view 5 | 3.45 ms (2.26×) | 4.12 ms (1.89×) | 7.68 ms (1.01×) | 7.78 ms |
| 平均 | **3.51 ms (4.14×)** | **4.40 ms (3.31×)** | **8.89 ms (1.64×)** | **14.54 ms** |

[P Table 4 p14]

Table 5 用 dedicated single-material benchmark 估计 material shading cost：仍在 renderer 内执行而不是把 material 拆成独立 kernel，并用简单 cosine-weighted direction distribution 锁定 path distribution。四个 scene/view 的算术平均为：2×32 1.54 ms、相对 reference 9.06×；3×64 6.06 ms、2.30×；reference 13.93 ms。该结果不能与 Table 4 的七个 full-frame view 算术平均混用；正文没有给重复次数、误差条或 timing dispersion。[P Table 5/Fig. 17 p14–15]

Cake box scalability 实验把 1–25 个不同 neural materials 放在固定 path-distribution 场景内，显示网络时间随材质数增加较平缓；图上 2×32 约在 5 ms、3×64 约在 13 ms 范围。正文没有给图中所有点的精确表格，故不把读图近似当精确数字。[P Fig. 14/§7.3/§8.3 p13,15]

## 9. 实验 protocol、baseline、指标与结果

### 9.1 Protocol 与可比性边界

- 最终质量图以 8192 SPP 输出以压低 Monte Carlo 噪声；时间测量是同一 scene 的 1 SPP frame。Stage supplemental 说明采用 unbiased path tracing、HDR distant map 加三盏 softbox local lights，最终图 4096–8192 SPP。[P §8/Fig. 15–16 p13–14; S §4 p12]
- 所有 runtime 结果在单张 NVIDIA GeForce RTX 4090、1920×1080、Falcor/D3D12-DXR、NEE+MIS、无 denoising 下得到；正文没有报告 driver、重复次数或 timing dispersion，因此不与移动 GPU、raster-only 或带 neural denoiser 的数值直接横排。[P §8 p13]
- error table 的 reference 是 authors’ original material render；FLIP、mean absolute/squared、relative absolute/squared、SMAPE 按图像像素聚合，再在 Table 2 的四张图或 Table 3 的七个 view 上取算术平均。正文没有为这些平均提供置信区间、error bars 或多 seed 分布。[P Table 2–3 p8,14; S §6 p13–14]

### 9.2 关键实验

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| 8-channel analytic approximation | 对 Teapot layered materials，使用与 z8 相同通道预算；analytic diffuse+isotropic GGX，一版在其 pipeline 数值优化、一版由 specialist 手调 | optimized analytic、manual analytic、z8 neural、reference | FLIP + visual | 两种 analytic 都无法重现 view-dependent ceramic glaze；neural 更接近 reference。未给该图精确 aggregate table | P Fig. 3 p3; S §5 p12 |
| Architecture ablation | 四个图：Inkwell、Teapot、Slicer blade/handle；vanilla latent+MLP、+encoder、+learned frame | 两个逐步消融 | FLIP、abs/sq、relative abs/sq、SMAPE | full frame model 在六项平均指标全优于两消融；详见下表 | P Fig. 8/Table 2 p8–9 |
| Filtering | 在不同距离/各 MIP 比较 unfiltered、ours、512 SPP filtered GT | unfiltered 与 supersampled GT | visual/aliasing | fine/近景匹配好；medium distance 偏软、丢细节；远景相似。无精确统一 scalar | P Fig. 5 p5; P Fig. 9/§6.1 p9 |
| Encoder resolution scaling | 512² 与 4k² latent，direct optimization 对 encoder bootstrap | direct latent vs encoder | latent visual + render crop | 512² 接近；4k direct 残留 random-init noise，因 texel 约少 64× updates；训练可到 10h | P Fig. 10/§6.2 p9 |
| Sampler anisotropy/non-centering | 4 SPP crops，full analytic proxy 对 isotropic centered variant | simplified isotropic specular sampler | inset pixel-wise std 与 mean | 两处 mean std：0.70 vs 1.73、0.22 vs 0.42，full 更低 | P Fig. 11/§6.3 p9 |
| Analytic proxy vs normalizing flow | evaluator 固定 2×32；flow 8/16 bins；Inkwell 与 Teapot | Zheng-style normalizing flow | total runtime, TTUV, visual/variance | analytic total 3.06/4.55 ms，TTUV 4.73/70.34；flow 8-bin 7.93/10.59 ms，TTUV 12.26/366.00；16-bin 14.31/17.69 ms，TTUV 22.12/330.06。flow 总 frame 慢 2–3.8×；Teapot 8-bin 漏窄峰 | P Fig. 12/§6.3 p9–10 |
| Evaluator size quality | 七个 Inkwell/Stage views，最终 8192 SPP | 2×16、2×32、3×64 | 六种 image errors | 3×64 最低平均 error，2×32 接近且明显更快；详细平均见下表 | P Table 3/Fig. 15–16 p13–14 |
| Evaluator size runtime | 同 scene path distribution，1 SPP | reference materials 与三 neural sizes | total ms/SPP、speedup；另隔离 shading | 平均 total：3.51/4.40/8.89 ms vs ref 14.54；隔离 shading 2×32 平均 9.06×、3×64 2.30× | P Table 4–5/Fig. 17 p14–15 |
| Multi-material scaling | Cake box 固定 paths、1–25 个不同 neural materials | coherent/divergent behavior | time、warp coherence distribution | 时间随 material count 近似平缓，coherence 不足时动态路径仍工作；无逐点数字表 | P Fig. 14/18, §7.3/§8.3 p13,15 |
| MERL multi-material encoding | 全 100 MERL BRDF，material identity one-hot→encoder，唯一 z8/material，共享 decoder | measured BRDF plots | direction-slice plots | 显示能够共同拟合多种 measured BRDF；无 holdout split、aggregate error 或运行时表 | S §7.1 p15–17 |

架构 ablation 的正文平均表：

| Variant | Mean FLIP | Mean abs. | Mean sq. | Mean rel. abs. | Mean rel. sq. | SMAPE |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla MLP | 0.2390 | 0.0769 | 0.0682 | 0.2177 | 0.0798 | 0.2670 |
| + encoder | 0.1956 | 0.0652 | 10.1933 | 0.3439 | 265.4018 | 0.2397 |
| + frame transform | **0.0815** | **0.0183** | **0.0057** | **0.0656** | **0.0090** | **0.0713** |

[P Table 2 p8]

`+encoder` 的 mean squared/relative squared 异常大不是排版推测：supplemental 的逐材质表显示 Slicer handle 等少量 outlier 会把均值推至 40.1738/1061.4310；作者用其他 robust/perceptual metrics 和图像共同解释，而没有声称该阶段稳定胜过 vanilla 的每一项。[S Table 1–4 p13]

三种 evaluator 的跨 view 平均质量：

| Model | Mean FLIP | Mean abs. | Mean sq. | Mean rel. abs. | Mean rel. sq. | SMAPE |
|---|---:|---:|---:|---:|---:|---:|
| 2×16 | 0.1087 | 0.0439 | 1.3855 | 0.1042 | 0.0353 | 0.1449 |
| 2×32 | 0.0551 | 0.0145 | 0.0107 | 0.0429 | 0.0056 | 0.0468 |
| 3×64 | **0.0444** | **0.0121** | **0.0101** | **0.0347** | **0.0035** | **0.0363** |

[P Table 3 p14]

这些数字是单论文 protocol 下的 best observed trade-offs，不是模型表达“上界”，也没有 bootstrap confidence interval 或 seed robustness。因此报告只保留作者观察，不把 2×32 自动选为本项目最优候选。[I]

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `ablation-inferior` | vanilla MLP + direct hierarchical latent | 空间细节失败；高分辨率 latent 留下随机噪声 | texel 数巨大，每 query 只更新少数 texel；surjective latent mapping 浪费 decoder capacity | `[I]` 是优化覆盖率与表示容量耦合，不足以证明 direct latent 形式本身不可表达 | P §6/Fig. 8–10 p8–9 |
| `ablation-inferior` | 加 encoder、不加 frame | 相对 vanilla 的纹理细节与 FLIP/abs./SMAPE 改善，但 Teapot 窄 glaze lobe 仍丢失，且 squared/relative metrics 被少量 outlier 放大；它不在每个指标上都优于 vanilla | 小 decoder 同时承担旋转与 BRDF shape，容量不足；最终 learned-frame 配置在六项平均指标全优 | `[I]` encoder 解决 spatial optimization，不等价于 directional prior；这里的 inferior 是相对最终配置，不是“所有指标均失败” | P §6/Table 2 p8; S Tables 1–4 p13 |
| `ablation-inferior` | isotropic、centered analytic sampler | normal-map shifted lobe 与 coarse-level anisotropy 下方差更高 | analytic family 缺少 mean slope、correlation/elliptical roughness | `[I]` sampler family bias 可直接进入 PT variance；应与 evaluator error 分开测 | P Fig. 11/§6.3 p9 |
| `author-negative` | normalizing flow，8/16 bins | 8-bin 漏 Teapot narrow peak；16-bin 质量更接近但 total frame 慢 2–3.8×；通常 4 MLP eval/hit | real-time budget 下 flow invert/eval 成本过高 | `[I]` 结论只针对其实现、network/bin 和 RTX 4090；不能概化为所有 flow sampler | P Fig. 12/§6.3 p9–10 |
| `author-negative` | 直接优化 4k latent | texel 留 random initialization noise；训练近翻倍并达 10 h | 每 texel 获得约 64× 更少更新，且 memory pressure 高 | `[I]` encoder 是 source-aware optimizer/preconditioner，不只是运行时表示模块 | P Fig. 10/§6.2 p9 |
| `author-negative` | coarse MIP latent finetuning | medium distance 过软、细节丢失；继续 finetune 未跳出 initial local minimum | coarse-level optimization 较难；可偏向 finer LOD 但会增加 aliasing | `[I]` 这是 bias—variance 交换：选择更细层不是无代价修复 | P §6.1 p9 |
| `author-negative` | 把 BRDF loss 从 L1/log 改为 L2 | 有时改善 grazing-angle Fresnel artifact，但其他区域下降，甚至显著 | 单纯 loss replacement 不够，可能需要新的 graphics prior | `[I]` 适合设计局部 Fresnel prior 的 matched test，不应直接把 L2 当修复 | P §8.1 p14 |
| `author-negative` | 用 GT TBN 对 learned frames 显式 supervision | glint 有时改善，但需要 extensive hyperparameter tuning，最终未纳入 | 监督权重/目标难调 | 不能把“有时改善”写成正式优越结论 | P §9 p15 |
| `author-negative` | NeuMIP-style neural displacement 与若干 geometric-prior variants | 所有神经版本都被 fixed-function ray marching 在 bandwidth/runtime 上击败，且没有一个足够快 | displacement 查询模式与网络成本不合实时要求 | `[I]` 这是部署负结果，不证明神经 displacement 在离线质量上失败 | P §9 p15 |
| `author-negative` | 小 network 的不同 initialization/hyperparameter | 可落入明显不同 local minima；最小 network 不能可靠保留 Teapot glaze，Figure 16 省略了一版无 glaze 结果 | compact optimization variance | `[I]` 需要多 seed 或 multi-instance 机制区分 capacity failure 与 optimizer failure | P §9 p15 |

作者还写明测试了从 unconstrained high-dimensional affine 到 rotation-only 的多种 geometric-prior implementation，但只披露最终 frame 形式，没有给替代结构、protocol、观察或定量结果。依据本任务证据规则，这段只证明“尝试过”，不足以把全部替代 prior 分类为 `author-negative`。[P §9 p15]

上述负结果与“最终方法为什么看起来合理”分开记录：只有作者明确给出退化/不稳定/部署失败观察的 direct optimization、flow、L2、TBN supervision、displacement、coarse filtering 和 optimization variance 才进入失败/较差尝试表；本报告没有从最终 architecture 反推未披露的搜索历史。

## 11. Paper ↔ supplemental ↔ code correspondence

`C` 的 commit 时间与 README 定位均表明它是 2026 后续实现。下表的“冲突”指不能把后续 default 当 2024 formal；它不自动表示代码错误。

| 主题 | Paper `P` | Supplemental `S` | Code/config `C` | 结论/冲突 |
|---|---|---|---|---|
| Latent | z8 hierarchical texture；finest=source resolution；stochastic adjacent MIP | Listing 1 给 footprint LOD、随机 level 与 bilinear fetch | later default z8、`num_mip_levels=1`；代码支持 per-material/UDIM/mip textures | 核心数据结构同族；default 单 mip 不是 2024 filtering formal config [C `latent_texture.slang`:10–19; `latent_texture.py`:261–308] |
| Encoder | `K→64×4→8` ReLU，先 bootstrap | 无另一 formal width | default `[64,64,64]` LeakyReLU，latent 可 normalize | later default 少一层且 activation/normalization 不同 [C `encoder.py`:12–67; `configs/default.json`:98–112] |
| Learned frames | z8→12，2 frames，方向投影；§9 写作 `b=n×t/||n×t||` | Listing 2：canonical residual，n/t normalize、不正交，`b=cross(n,t)`，没有 normalize b | 同一 core prior 保留，later code 也是未 normalize 的 cross | **未解析的 P↔S 冲突**：正文 limitation 段与 functional supplemental 不同；后续代码支持 S，但不能抹掉正文写法 [C `rotation.py`:22–33; `rotation.slang`:57, 82–105] |
| Direction parameterization | 2024 formal 是 wi/wo 在 learned frames 下的 Cartesian components | Listing 2 直接拼 transformed wi/wo | later default 先 StableRusinkiewicz half/diff，再 rotation，flattener=WhWdZiZo | 有实质 paper-code gap；later default 不可代替 2024 formal [C `bsdf_decoder.py`:34–75; `half_diff_parameterization.slang`:59; `bsdf_decoder_flattener.slang`:119] |
| Evaluator | 2×16/2×32/3×64 ReLU，`exp(raw-3)` | Listing 2 是 2×32 functional example | later default 2×16 SmeLU + ScaledSigmoid 1e5，可配置 Exp | 输出 family 和默认 activation 均变化 [C `configs/default.json`:110–134] |
| Sampler architecture | 3×32 ReLU→9，一 spec+一 diffuse | Listing 3–4 定义 exact raw transform 与 2D mixture remap | later sampler 可有多 spec lobes，输出 `3+6L`，default 1 lobe/2×16 LeakyReLU | analytic family同源；network default 不同 [C `sampler.py`:15–54] |
| Sampler random variates | (u\in[0,1)^2)，`u.x` 选 component 后重映射，并继续参与 lobe sample | Listing 4 明确 2D remap | later code用 `float3`：x 选 component，yz 独立采 lobe，不 remap；PDF clamp 到 `1e-4` | 这是明确 algorithmic gap，不应将 later path 称 2024 exact [C `sampler.slang`:104–130] |
| BRDF loss | L1 in log space，exact transform 未报告 | 未补 exact transform | default `L1WithPowerLog`, power=3 | later default 只能给候选解释，不能消除 2024未报告 [C `configs/default.json`:77–85] |
| Optimizer/schedule | Adam，300k，LR 1e-3→1e-4 | 给 exact Adam/cosine | default phase counts 100k/0/50k/50k，base LR 1e-3 且 phase-end scale=0.01（即 1e-5） | 2026 default lifecycle/终点不同 [C `configs/default.json`:31–64] |
| Training lifecycle | encoder 与 decoder；materialize+finetune；evaluator/sampler simultaneous；两批各65k | mollification exact | encoding→optional direct-opt→frozen-BSDF sampler→auxiliary | 核心阶段组件可追踪，但时序改变 [C `train.py`:857–1147; `train.slang`:60–195] |
| KL objective/estimator | 只写 learned PDF 相对当前 learned BRDF 的 KL divergence，latent detached；KL 方向、target normalization 与 estimator 均未报告 | 未补 estimator | later code 从 sampler 采样，target=decoder luminance，`-detach(target)*log(pdf)/detach(pdf)`，对应 forward-KL score-function 形态 | “forward KL”只由 later code 支持，是可审计补全选择，不是 2024 formal 明文 [C `train.slang`:169–196] |
| Data sources | 五个不可公开的 layered materials | graph screenshots/scene说明 | MaterialX/MDL GPU online query，bundled FauxLeather example | 后续 source family更开放，但不复原五材质 [C `README.md`:62–69,109–150] |
| Runtime | Slang/DXR、tensor-core/coherent 与 packed-FMA/divergent | functional pseudocode，不是 optimized kernels | 2026 repo基于 Slang/SlangPy/falcor2/Vulkan/CoopVec | backend 已从 D3D12/DXR 路径扩展/变化；不可期待数值性能 parity [C `README.md`:5–7,30–35] |

此外，supplemental Listing 1 的函数签名含 `uv,u`，某些后续 pseudocode call 省略这些实参；这更像说明性伪代码的排版/抽象省略，不能据此重建 exact RNG threading。它提示 reviewer 不应把 Listing 当可编译 artifact。[S Listing 1, 5–6 p3,8–10]

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. **无硬物理约束。** evaluator 不保证 energy conservation 或 reciprocity。作者没在例子中观察到明显问题，但承认高 albedo、多 bounce 可积累能量误差；Rusinkiewicz-style representation 可能更容易强制 reciprocity，但作者称 Cartesian input 数值更稳定且视觉更好。[P §3 p5; P §9 p15]
2. **反射域。** delta components 被排除；rough refraction 只有 preliminary evidence，正确处理 IOR 仍是问题。[P §3 p5; P §9 p16]
3. **过滤 bias。** coarse latent levels 容易过度平滑；偏 finer LOD 会重新引入 aliasing。[P §6.1 p9]
4. **优化方差。** compact models 对 initialization/hyperparameters 敏感，可能收敛到不同 local minima；最小网络不可靠保留 Teapot glaze。[P §9 p15]
5. **displacement 未成功。** neural displacement variants 在作者实现中均不够快，落后 fixed-function ray marching。[P §9 p15]
6. **无 runtime editability。** 表示是 reference 的 bake；材质编辑后需要重新训练/烘焙。[P §3 p5]
7. **存储仍大。** z8 FP16 没有 texture compression；INT8/更低精度只列 future work。[P §7.1 p10; P §8.3 p15]
8. **平台依赖。** 主要性能论证依赖 NVIDIA RTX 4090、D3D12/DXR、tensor cores、SER 和专门 shader codegen；作者没有证明跨厂商/移动端同样 speedup。[P §7–§8 p10–15]
9. **不是 unseen-material generalization。** 五材质通常各自训练；MERL 是共同拟合 100 个已见 identity，不是 holdout generalization。[P §5.1 p7; S §7.1 p15]

### 12.2 未报告/材料不可得

- 五个正式 source materials 的可执行 graph、textures、license/release locator、reference shader code 与 checkpoint：未报告/未公开。
- exact training initialization、随机 seed 数、seed selection、checkpoint selection 和失败 run 数：未报告。
- encoder→latent switch 的 iteration 或收敛判据；300k 内各阶段分配：未报告。
- log-L1 的 exact formula、offset/epsilon、color-channel reduction 与 loss normalization：未报告。
- 2024 KL 的方向、sampler target normalization、PDF floor、invalid-direction handling 和 gradient estimator：未报告。
- exponential MIP distribution 的参数、level cap；Gaussian footprint 的 sigma/截断、每 level sample-count rule；LEAN 具体过滤字段与处理：未报告。
- runtime stochastic LOD RNG 的 exact stream、clamp/boundary/address mode 与 mip-chain byte accounting：未报告。
- network/latent 的 exact parameter initialization、是否训练 bias、albedo head activation：部分未报告。
- learned-frame bitangent 是否 normalize：正文 §9 写 normalized cross，supplemental Listing 2 与后续官方代码使用未 normalize 的 `cross(n,t)`；第一方材料内部冲突未解析。
- 每个正式模型的参数数/MAC/FLOP、compiler flags、driver version、完整 shader binaries、冷/热 cache protocol：未报告。
- image metric 的全部实现参数、逐 seed/逐 material confidence interval：未报告。
- energy conservation、reciprocity、PDF normalization 与 sample↔PDF parity 的定量 oracle：未报告。
- five-material train/test split、unseen parameters/materials/texture identities 的泛化实验：未报告。
- 项目页未发现 correction/erratum 链接；这不等于不存在任何作者私下修订，只表示已检查的一方入口没有提供。

上述缺口直接影响“完全复现 2024 数值”而非“理解公开方法”。后续代码可以帮助选择一个可审计补全方案，但补全必须登记为 later-code-informed adaptation，不能伪装为作者原配置。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

这篇论文的 compactness 不是“用一个小 MLP 记住整个材质”这么简单。容量分成四层：

1. **per-texel/per-scale z8** 保存高频空间与过滤后 appearance；其总参数量远大于 MLP，并承担 material-specific memory。
2. **learned frame projection** 把空间相关的 lobe rotation 显式变成乘法投影，保留小 MLP 的非线性容量处理 lobe shape、layer mixture 与颜色。
3. **shared evaluator weights** 只做已对齐坐标下的局部回归；2×16/2×32/3×64 是质量—时间可调轴。
4. **analytic sampler family** 把 invertibility、PDF 和各向异性放在 closed-form core 内，network 只回归少量参数。

Encoder 的作用主要在 optimization geometry：它让一个 source-parameter sample 的梯度规律跨相似 texel amortize，并给 latent space 施加“相似参数→相似 code”的结构。它运行时被删除，因而不贡献 shader inference capacity；但它决定能否在有限训练预算下把巨大的 latent hierarchy 训练到合理 basin。把 encoder 仅称为压缩网络，会遗漏论文最重要的负结果解释。[I，依据 P §5.1/§6.2]

### 13.2 成功所依赖的假设

- reference 能在线、可微训练链外地稳定提供大量 (f\cos\theta) 查询与 coarse-footprint target；论文不需要 reference 本身可微，但需要足够高吞吐。[I]
- source 参数 (\mathbf{k}) 对 appearance 有可学习的局部结构。若输入只是 opaque material identity，encoder 仍可拟合已见 material（MERL one-hot），但无法自然提供未见参数状态归纳偏置。[I]
- 两个 learned frames 足以覆盖目标 layer stack 的主要 lobe orientations；如果 closure 数、离轴 peak 或 transmission topology 超出该 prior，剩余 MLP 会重新承担难度。[I]
- 两叶 diffuse+GGX mixture 足以做低方差 proposal；它不要求与 evaluator 完全相等，但要求尾部与主要峰覆盖充分，否则 MIS variance 会暴露 family bias。[I]
- material state 在 runtime 固定。论文通过“编辑后 rebake”避开 editable parameter conditioning；这与本项目保留原生可调参数的长期要求并不自动一致。[I]
- shader backend 能把小 dense MLP 映射为低 overhead code，并且材质 coherence 足以让专用路径获益；否则 network bytes 小不代表实际 latency 小。[I]

### 13.3 可迁移机制与不能迁移的部分

可直接进入本项目候选/协议的机制：

- source-native online reference query，不把源材质先反演成项目内部 closure；
- z8 latent、learned-frame prior 和 bounded MLP 作为 local evaluator candidate；
- encoder bootstrap 对高分辨率空间 latent 的优化作用；
- evaluator 与 sampler 分开建模，但让 sampler 读取同一 latent；
- analytic proposal 输出完整 PDF、且 sample/PDF 使用同一参数；
- LOD 作为训练 query domain，而不是部署后对 latent 随意 downsample；
- coherent/divergent shader cost 分开测，不用参数量代替真实单查询时间。

不能直接迁移或必须改造的部分：

- 论文把一个最终 material state 烘焙进 texture；本项目要求 source 原生可调时保留编辑能力。若要迁移，必须把 native parameters 纳入 runtime conditioning、编译成可覆盖参数域的 latent/weight 表示，或明确限定为不可编辑 asset bake。不能把一次状态 bake 称为“保留可调性”。
- 论文输出 cosine-weighted response，而项目 evaluator contract 是线性 (f)；需要经过验证的 adapter，并在 grazing angle 处理数值稳定性。
- 论文 performance 来自定制 RTX 4090/DXR path；项目 MethodBundle/Slang/viewer 需要独立测真实 backend，不能继承 1.6–4.1× speedup。
- 五材质 source 和 formal config 不公开，故当前 MaterialX/LayerStack 只能做 functional reproduction 与 transfer study，不能声称 image reproduction。
- 论文的 learned hierarchy 没有证明 unseen material/parameter generalization；本项目 G1/G2/G2s 必须独立冻结 split 与 selection protocol。

### 13.4 与本项目 runtime contract 的关系

| Contract 维度 | 论文形态 | 本项目判断 `[I]` |
|---|---|---|
| `prepare()` | 论文没有该 API 名，但 hit 时先 fetch z、解 frame，并可解 sampler params | 适合缓存 z、两个 frame、view-conditioned evaluator input 与 sampler raw；对同一 shading point 的多 light/eval query 可复用 |
| `evaluate(wo,wi)` | fixed-size frame projection + MLP，返回 (f\cos) | 静态有界；经 cosine adapter 后可作为产品 evaluator candidate |
| `sample()/pdf()` | fixed-size MLP + two-lobe analytic sample/PDF | 静态有界；属于与 evaluator 匹配的 proposal，不是 closure 输出词汇 |
| 随机访问 | 单 hit 的一个 stochastic mip + bilinear latent fetch，网络无邻域 convolution | 满足随机访问；但 texture footprint 与 cache locality 仍须计入实际成本 |
| 固定读取数 | z8 两张 RGBA texture；stochastic 只选一 level | formal 形态固定有界；若改 trilinear 或多 level ensemble 必须重新登记读取预算 |
| source 语义 | reference graph 训练时直接出 GT；运行时只留 bake | 训练侧与 native-source 原则相容；runtime editability/parameter generalization 不足 |
| 物理正确性 | 无 reciprocity/energy hard constraint；sampler analytic PDF | evaluator 需要 energy/reciprocity diagnostics；proposal 需要 normalization、sample↔PDF 和 finite tests |
| 部署类别 | local material evaluator + matched sampler + prefiltered texture compiler | 不是 scene-transport learner；可作为 product candidate，也可作为 capacity diagnostic。reference 本身仍是 GT/teacher |

总体上，本文最有价值的不是某个单独层宽，而是 **source-aware encoder bootstrap + frame prior + bounded analytic sampler + shader coherence path** 这一整条设计链。任何只复刻 evaluator MLP、却丢掉 hierarchy、encoder lifecycle 或 sampler semantics 的实现，都不足以代表论文方法。[I/N]

### 13.5 建议晋升为 load-bearing 的相关论文

以下建议只登记研究依赖，不修改 catalog：

| Related work | 建议 trigger | 为什么会改变本报告的解释 |
|---|---|---|
| *Taming optimization variance in compact neural shading networks* | `direct-inheritance` + `failure-explanation` | 2024 作者明确报告 compact network 的 seed/local-minimum 问题；2026 论文和官方代码直接处理该失败，能区分 capacity 与 optimizer variance |
| Zheng et al. 2021, *A Compact Representation of Measured BRDFs Using Neural Processes* | `key-baseline` + `direct-inheritance` | 2024 的 log-space objective、reciprocity discussion 与 normalizing-flow sampler 直接沿用/对照它；需要核对 flow 的正式次数与编码，而不是只读 2024 转述 |
| Sztrajman et al. 2021, *Neural BRDF Representation and Importance Sampling* | `direct-inheritance` + `key-baseline` | half/difference sampling、compact evaluator 与 analytic proxy 是直接方法前身 |
| Fan et al. 2022, *Neural Layered BRDFs* | `direct-inheritance` + `key-baseline` | layered neural representation 与 analytic sampling proposal 是作者主要比较对象之一 |
| *NeuMIP* | `failure-explanation` + `key-baseline` | direct latent optimization 的分辨率 scaling 与 displacement negative 都以 NeuMIP 为紧邻参照；需核对两者 query、offset 和预算差异 |
| MIPNet / LEAN mapping | `key-baseline`（若 filtering 成为决策轴） | 只有当层级 filtering 进入当前候选 kill/advance decision 时才提升；否则保持 discovery，避免把背景方法扩成无关 formal 负担 |

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

当前工作树的 formal config 与归档 correspondence 已不再使用同一 correspondence identity，必须分开登记：

本节的 `archived correspondence.md` 指 `.trellis/tasks/archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md`，`formal-report.json` 指 `artifacts/nvidia-faithful/materialx-recorded-200k/formal-report.json`。

- 当前 config/validator 强制 `correspondence_id = nvidia-rta2024-functional-f@2`；
- `recipe_id = nvidia-rta2024-materialx-formal-300k-stage100k@1`；
- `source_adaptation_id = materialx-standard-surface-spatial@1`；
- LayerStack smoke 为 `layer-stack-uniform-1x1@1`；
- archived `correspondence.md` 与 200k `formal-report.json` 仍写 `nvidia-rta2024-functional@1`，所以它们只能证明旧 identity，不能被引用成当前 `@2` 的运行结果。[N `configs/learning/nvidia-rta2024-materialx-formal.json`:6–7; N `src/ncls/learning/methods/nvidia.py`:398–400; N `tests/unit/test_nvidia_faithful_contract.py`:21; N archived `correspondence.md`:8–13; N `artifacts/nvidia-faithful/materialx-recorded-200k/formal-report.json`:17–24]

逐项影响如下：

| 分类 | 当前实现/证据 | 本报告结论 |
|---|---|---|
| `faithful` | `K→64→64→64→64→8` encoder、encoder→materialize→latent finetune lifecycle | 与 P Fig. 6/§5.1 对应；归档 correspondence 与当前 lifecycle test 均可定位 [N archived `correspondence.md`:19,27; N `tests/unit/test_nvidia_faithful_contract.py`:51–76] |
| `faithful` | z8 hierarchy、两张 RGBA16F、footprint LOD、stochastic adjacent mip、bilinear | 与 P §4.1/S Listing 1 对应；训练期扁平存储只作为 storage adapter [N archived `correspondence.md`:20; N `tests/unit/test_nvidia_faithful_contract.py`:79–105] |
| `faithful` | 两个 learned frames、canonical residual、normalize n/t、两个 frame 的 Cartesian direction projection；20→64×3→3 ReLU、`exp(raw-3)` | frame/evaluator 拓扑与 S Listing 2、正式最大 evaluator 对应；bitangent normalization 与 output measure 另行分类，不能包进本行 [N `src/ncls/learning/models/nvidia_neural_appearance.py`:252–287; N `shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_core.slang`:147–199,218–220] |
| `author-underspecified` | 当前 `bitangent=cross(normal,tangent)`，不 normalize | 匹配 S Listing 2 和 later C，但与 P §9 的 normalized cross 冲突；这是第一方冲突后的显式选择，不应写成无条件 `faithful` [N `src/ncls/learning/models/nvidia_neural_appearance.py`:252–260; N `shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_core.slang`:147–168] |
| `faithful` | 11→32×3→9 sampler、exact 9-param decode、仅 two lobes、同一 `float2 u` 与 component remap | 与 S Listing 3–4 对应；已移除旧 safety lobe/第三随机数 [N archived `correspondence.md`:24–26; N `tests/unit/test_nvidia_faithful_contract.py`:79–91] |
| `faithful` | uniform UV、half/difference directions、exponential mip、Gaussian footprint；20k mollification；两条各 65k route；Adam schedule；simultaneous optimization 与 latent detach | 公开结构与 P/S 对应；当前实现另外给两个 route 配置不同 `seed_offset`，但 stream independence 是项目冻结选择而非论文明文 [N `configs/learning/nvidia-rta2024-materialx-formal.json`:26–78; N archived `correspondence.md`:28–32] |
| `author-underspecified` | `materialization_step=100000` | 取后续作者 default BSDF encoder phase；不是 2024 隐藏事实 [N `correspondence.md`:35] |
| `author-underspecified` | `log1p-l1@1` | 论文只写 log-space L1；当前 exact transform 是项目冻结选择 [N `correspondence.md`:34] |
| `author-underspecified` | `learned-sampler-forward-kl-score@1`，从当前 proposal 采样并用 score-function estimator | later-code-informed estimator；2024 只公开 KL、current learned BRDF target 与 latent detach，连 KL 方向也未明文 [N archived `correspondence.md`:33; N `configs/learning/nvidia-rta2024-materialx-formal.json`:79–82] |
| `author-underspecified` | mip rate=1、spatial cap=64、LEAN coarse feature rules | 论文没给 rate/cap/字段；保持 recipe identity，不升级为 faithful paper detail [N `correspondence.md`:28] |
| `interface-adaptation` | 当前训练以 `target_f` 监督 `evaluate_f`，runtime 直接写 `result.f`；保留 `exp(raw-3)`，但不再执行 archived `nclsNvidiaNeuralResponseToBareF` cosine division | 这是把 supplemental 的 cosine-weighted response 改为项目 bare-`f` ABI 的重新定目标；必须使用当前 `functional-f@2` 身份，不能继续引用 `@1` adapter 文字 [N `src/ncls/learning/methods/nvidia.py`:523–534; N `shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance.slang`:100–107] |
| `interface-adaptation` | backend `prepare/evaluate/sample/pdf`、reverse PDF 交换方向后重新 prepare、null/error ABI | 不改变 forward proposal；让方法进入统一 dispatcher/renderer [N `correspondence.md`:38,40] |
| `source-domain adaptation` | MaterialX `american_walnut_veneer` snapshot；LayerStack uniform 1×1 smoke | 作者五资产未公开，项目只对自身 snapshot/reference 负责，不逐图复刻论文 [N `correspondence.md`:39] |
| `budget-adaptation` | archived `@1` run 由用户决定在 200k 冻结经验结果 | 不得标作 300k formal protocol 完成，也不得当成当前 `@2` 的验证结果；recipe 名仍保留 300k [N `formal-report.json`:5–24; N archived `correspondence.md`:13,46] |
| `suspected-defect` | 当前 config/validator 已升为 `functional-f@2`，但 archived correspondence 仍以 `functional@1` 自称“当前” | 这是证据文档/identity migration gap，不是已证明的 evaluator 算术 bug；在补迁移说明或新 `@2` artifact 前应保留 [N current config/source vs archived `correspondence.md`:9] |

archived 200k `@1` 记录仍是有效的历史 project observation：checkpoint `ee3e6fb3…12fe`、package `6950aeb2…02073`；64×4096 方向诊断的 normalized L1 median/p95 为 `0.02548/0.07018`，energy relative error median/p95 为 `0.01157/0.04272`，packed runtime 对 FP32 master 的 log1p L1 mean 为 `4.49e-5`；viewer PT/deferred 与 reference/neural slots 均 `ready`。这些数字不是论文结果、不是新 hard gate、不证明 300k formal 已完成，也不验证当前 `functional-f@2`。[N `artifacts/nvidia-faithful/materialx-recorded-200k/formal-report.json`:5–24,114–234,304–359]

本报告对当前复现最直接的约束是：继续保留 2024 formal identity 与 2026 default identity 的分离，尤其不能把 later code 的 StableRusinkiewicz、WhWdZiZo、3D sampler random、PDF floor 或 sequential frozen-BSDF sampler 静默混入当前 `nvidia-rta2024-functional-f@2`；同时应补一份 `@1→@2` output-measure/identity migration evidence，避免历史 artifact 被误读成新身份结果。[I，依据 §11 与上述 N mismatch]

## 15. 可证伪的迁移假设 `[I]`

下列是假设，不是新任务 hard gate。进入实验前仍需按项目 protocol 冻结 source locator、query recipe、seed/selection、预算和统计聚合。

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：learned-frame prior 在相同 evaluator 预算下改善 multi-lobe/normal-mapped local scattering | P Table 2/Fig. 8，full 六项平均误差均优 | 当前 LayerStack/MaterialX reference 的方向结构同样受显式 rotation prior帮助 | `z8+vanilla MLP` vs `z8+2 frames+MLP`；匹配总参数或报告 parameter/time 两种控制 | source、query、latent bytes、optimizer、work units、seeds、output transform | 局部方向 FLIP/normalized error、grazing/peak error、energy、single-query median/p90、bootstrap CI | 静态有界 evaluator | matched run 中质量无稳定改善，或同质量下真实 query time/bytes Pareto 被 vanilla 支配 |
| H2：encoder bootstrap 的收益随 spatial resolution 增长 | P Fig. 10：512² 接近、4k direct 残噪，约 64× 少更新 | source native parameters 对 texel appearance 有共享结构 | encoder→materialize→finetune vs direct latent；至少两档 resolution，decoder/latent 相同 | total logical samples、optimizer、memory cap、source/filter、seed、decoder | texel coverage、latent noise谱、方向误差、训练时间/显存、最终 query time | compiler/training mechanism；runtime不变 | resolution×method interaction 不显著，或 encoder 在高分辨率同预算不改善 error/noise/收敛 |
| H3：stochastic adjacent-MIP 比 trilinear latent interpolation 有更好 filtering quality—cost trade-off | P §4.1 作者观察；Fig. 9 同时暴露 coarse blur | 各 level latent 语义差异使跨层线性插值产生 decoder bias | 同一层级/decoder，随机单层 vs 两层 trilinear；各自允许 matched retraining，并另做 frozen-model diagnostic | footprint、level targets、fetch precision、sample count、network、scene/camera | filtered GT error、aliasing temporal spectrum、pixel variance、texture fetch/time | 一层随机读取 vs 两层确定读取 | trilinear 在相同或更低 total error/variance/cost 下不被支配，或 stochastic variance 抵消其 bias 优势 |
| H4：two-lobe analytic proposal 在本项目 shader 预算内优于 centered/isotropic proposal | P Fig. 11 的 mean std；Fig. 12 的 runtime/TTUV | 当前 reference 也包含 shifted/anisotropic peaks，且 sampler family 可覆盖主要 support | full non-centered anisotropic GGX+diffuse vs centered isotropic；同 sampler MLP size/training budget | evaluator checkpoint、queries、RNG、MIS、SPP、backend | PDF normalization/parity、TTUV/variance、firefly quantiles、sampler time/bytes | 静态有界 matched sampler | full family没有显著降低 variance/TTUV，或增加 cost 后 Pareto 不优 |
| H5：新的 Fresnel/grazing graphics prior 比单换 L2 更能修复 grazing artifact | P §8.1：L2 有时改善 grazing、但其他区域明显下降 | 错误来自小网络对角度形状的容量浪费，而非 GT/query bug | 当前 log-L1 baseline、L2-only、显式 Fresnel-conditioned residual；匹配网络/训练与必要 parameter control | source、directions、optimizer、steps、seed、runtime precision | grazing/non-grazing stratified error、energy、overall FLIP、single-query time | evaluator candidate | prior 只把误差从 grazing 移到其他角度、无 overall/Pareto 改善，或违反有界部署预算 |
| H6：multi-instance/selection 能把“小模型容量不足”与“优化方差”分离 | P §9 报告 seed/local minima；后续 2026 工作直接承接 | 当前 compact candidate 的失败也包含 basin selection，而不全是 capacity | single-instance 多 seed vs 固定总 work 的 multi-instance/prune；另有 large reliable control | 总 reference queries、wall/GPU work、data streams、init distribution、selection rule预先冻结 | success rate、loss/quality distribution、worst-case seed、训练 time/memory、最终 runtime不变 | training strategy；runtime同一小模型 | 固定总成本下 multi-instance 不提高成功率/分布，或只有事后 selection 才成立 |

H2 不能由 uniform 1×1 LayerStack smoke 验证，必须使用 spatial source；H4 属于 sampler/transport 阶段，不应在 evaluator 尚未稳定时变成当前 kill test；H6 应由 *Taming optimization variance...* 报告先完成证据审查，再冻结实验。[I]

## 16. 证据索引

### `P`：2024 main paper

- §3、Eq. (1)–(3)，p4–5：SVBRDF/方向反照率定义、目标、domain 与排除项。
- §4.1，p5：z8 hierarchy、source resolution、footprint LOD、stochastic adjacent-MIP 与 bilinear。
- §4.2、Eq. (4)，p5–6：learned shading frames 与方向投影。
- §4.3、Appendix A Eq. (5)–(9)，p6、16–17：analytic two-lobe sampler、GGX slopes、Jacobian/PDF。
- Fig. 6、§5.1–§5.2，p6–7：encoder architecture/lifecycle、online query、filter/mollification、loss、双65k、300k、FP32→FP16。
- Fig. 3、8–12，Table 2，§6，p3、8–10：analytic baseline、ablation、filtering、direct-opt、sampler 与 flow 对比。
- §7、Fig. 14，p10–13：shader codegen、packed-FMA、tensor core、coherence/SER。
- §8、Table 3–5、Fig. 15–18，p13–15：image/runtime protocol、quality/cost/scaling。
- §9，p15–16：energy/reciprocity、refraction、alternative-prior/TBN supervision、displacement、filtering 与 optimization-variance 限制。

### `S`：2024 supplemental

- §1 Eq. (1)–(2)，p2：Adam、cosine LR、20k/10°→0°/256-sample mollification。
- Listing 1–4，p3–8：latent fetch、functional evaluator、sampler parameter decode、exact 2D component remap/PDF。
- Listing 5–6，p8–10：path tracer/forward renderer integration；为说明性 pseudocode，不是 optimized artifact。
- §3–§4，p11–12：reference graph screenshots、Stage lighting/setup。
- §5–§6/Table 1–11，p12–14：8-channel baseline 扩展图与逐材质/逐 view error。
- §7，p15–33：100 MERL 与五高保真材质的 evaluator/PDF/learned-frame direction slices；PDF 仅按常数缩放比较 shape。

### `C`：后续官方代码，commit `305b4b9c12e679398c487603dd8245c3f348526c`

- `README.md`:9–27,62–69,92–115：repo 对应 2026 论文，2024 为 additional details；bundled example、paper configs 与 GPU-online MaterialX/MDL 范围。
- `configs/default.json`:31–50, 77–193：later example/default phase、loss、latent、network、sampling 配置。
- `neuralappearance/model/rotation.py`:22–33 与 `rotation.slang`:57,82–105：保留的 learned-frame core。
- `neuralappearance/model/bsdf_decoder.py`:34–75、`half_diff_parameterization.slang`:59、`bsdf_decoder_flattener.slang`:119：later StableRusinkiewicz/WhWdZiZo path。
- `neuralappearance/model/sampler.py`:15–54、`sampler.slang`:104–130：later multi-lobe generalization、3D random 与 PDF floor。
- `neuralappearance/train.py`:857–1147、`training/train.slang`:60–195：later sequential lifecycle 与 sampler score estimator。
- `neuralappearance/model/latent_texture.slang`:10–19、`latent_texture.py`:261–308：per-material/UDIM/mip texture 与 encoder materialization。

### `A`：作者页

- NVIDIA official project page：作者、TOG/SIGGRAPH 2024、paper/supp/video/image viewer links，以及 learned hierarchy、two priors、LOD/anisotropic sampling 与 in-shader execution 的作者摘要。检查日期 2026-08-29；未见 erratum/correction 链接。
- NVIDIA Research publication record：uploaded files 只有 paper 与 supplemental，并回链 project page；未提供 2024 code/data/checkpoint。

### `N`：当前项目证据

- `configs/learning/nvidia-rta2024-materialx-formal.json`:6–82、`src/ncls/learning/methods/nvidia.py`:398–534、`src/ncls/learning/models/nvidia_neural_appearance.py`:252–287：当前 `functional-f@2` identity、formal recipe、bare-`f` target/output 与 frame/evaluator 算术。
- `shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_core.slang`:147–220、`nvidia_neural_appearance.slang`:100–107、`tests/unit/test_nvidia_faithful_contract.py`:21–105：当前 unnormalized bitangent、runtime `f` contract、identity validator 与 lifecycle/resource tests。
- `.trellis/tasks/archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md`:5–46：旧 `functional@1` 方法身份、逐项 correspondence、cosine adapter、underspecified/source/budget 分类与 200k recorded-result 索引；不能证明当前 `@2`。
- 同任务 `prd.md`、`design.md`：旧 functional reproduction 边界、formal identity 与 runtime contract。
- `artifacts/nvidia-faithful/materialx-recorded-200k/formal-report.json`:5–24,114–234,304–359：旧 `functional@1` 的 200k checkpoint/package、directional/energy/parity 与 viewer evidence。
- 同任务 `research/current-fidelity-audit.md`：修复前差距的历史审计；仅解释为什么 correspondence 必须逐项记录，不用于声称当前仍有旧缺陷。

### `I`：本报告推导

- §13：容量分解、迁移边界、runtime contract 与 load-bearing promotion triggers。
- §14：基于 `N` 的当前复现影响；未把项目选择改写成论文事实。
- §15：六个可证伪假设；均需后续 freeze 后才能成为实验结论。

## Evidence review

```text
author_worker: rta2024
reviewer: /root/nbrdf2021
reviewed_at: 2026-08-29
sources_rechecked:
  - P author PDF SHA-256 E709C1B5C4F0F16EB7EDF848D29079E007E3546DEDB8B5DFE4EA6BF44D9D1002; visually rechecked Fig. 4–18, Tables 1–5, Eq. (1)–(9), §5 lifecycle, §9 limitations and Appendix A
  - S author supplemental SHA-256 4AFADFF6A6F0A0E6CA8B2FF92927DE7E7DF4350A20CBC260FFEFC8920BA08376; visually rechecked §1, Listings 1–6, reference graphs, Tables 1–11 and model-evaluation plot preface
  - A NVIDIA official project page and NVIDIA Research publication record; both expose paper/supplemental but no 2024 code/data/checkpoint/correction
  - C NVlabs/neuralappearance commit 305b4b9c12e679398c487603dd8245c3f348526c; audit copy HEAD/refs verified and README/default/rotation/decoder/sampler/train/latent paths rechecked
  - N current formal config, validator/model/shader/tests, archived correspondence and archived 200k formal-report identity
findings_closed:
  - corrected the unsupported claim that the 2024 paper specifies forward KL; KL direction and estimator remain unreported
  - corrected the stage trigger from decoder convergence to encoder sufficiently trained, and removed the unsupported paper-level RNG-independence claim for the two 65k batches
  - clarified that frame/evaluator/sampler weights are per baked material and shared across that material's texels/MIPs
  - preserved the P normalized-bitangent versus S/C unnormalized-bitangent conflict and removed the blanket faithful classification from the current reproduction
  - corrected the +encoder failure label to ablation-inferior and removed unsupported author-negative classification for unspecified alternative priors
  - added full-frame versus dedicated material benchmark scope, view averaging and missing timing/seed dispersion
  - updated the current reproduction identity to functional-f@2 and separated it from archived functional@1 correspondence/artifacts
  - corrected the current output-measure correspondence: bare-f retraining/runtime output replaces the archived cosine-division adapter
remaining_evidence_gaps:
  - 2024 exact source assets/checkpoints/config/logs unavailable
  - exact log transform, KL direction/normalization/estimator, stage split, batch-stream relation, seeds and selection unreported
  - main-paper versus supplemental bitangent normalization conflict unresolved
  - current functional-f@2 has no located migration note or formal artifact linking it to archived functional@1 200k evidence
review_status: evidence-reviewed
```

### 完成检查

- [x] main paper 已完整阅读，关键公式/图/表/图注/脚注已视觉核对；
- [x] supplemental/appendix/勘误的可用性已检查；
- [x] official code/config/data 的可用性与 commit 已检查；
- [x] architecture、training、runtime 和主要结果均有 locator；
- [x] 失败尝试与较差消融正确分类；
- [x] paper/code gap 和“未报告”保留；
- [x] `I` 分析晚于事实层，没有改写作者结论；
- [x] NVIDIA 影响引用真实 `N` 证据；
- [x] 假设包含 matched control、部署类别和证伪条件；
- [x] 独立 evidence review 已完成。
