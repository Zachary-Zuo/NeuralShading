---
paper_id: "2026-neural-material-adapter"
title: "Neural Material Adapter: Transforming Complex Materials into Efficient Analytic BRDFs"
authors: "Rajesh Sharma, Tiziano Portenier, Sebastian Weiss, Markus Gross, Marios Papas"
year: "2026"
venue: "Computer Graphics Forum 45(4), Eurographics Symposium on Rendering (EGSR) 2026"
doi: "10.1111/cgf.70549"
report_status: "evidence-reviewed"
main_source: "https://diglib.eg.org/items/887713f3-27d2-4901-b01a-26c90c7eace3"
supplemental_status: "available"
official_code_status: "unavailable"
official_code_commit: "not-applicable"
author_worker: "/root/taming2026"
reviewer: "/root/adapter2026_review"
last_verified: "2026-08-29"
---

# Neural Material Adapter: Transforming Complex Materials into Efficient Analytic BRDFs

## 1. 研究对象与报告边界

本文研究的是一个 **local material adapter**：给定复杂源材质的原生参数，以及当前 viewing direction（论文记作 `omega_i`），网络不直接输出 BRDF 数值，而是在线生成两个解析 Principled BRDF 的参数与混合权重；随后的 `evaluate` 与重要性采样由这个解析目标完成。论文的核心问题是：能否借助解析 BRDF 的物理与采样先验，把 PFMC 多层材质或测量 BRDF 的复杂外观转换为 CPU 上可用、无需逐材质 baking 的近似。

本报告覆盖 DOI `10.1111/cgf.70549` 对应的 Computer Graphics Forum / EGSR 2026 正式版本、Eurographics 官方 supplemental bundle、DisneyResearch|Studios 项目页、ETH CGL 条目以及可检索的作者公开入口。它不把以下相邻问题算入论文已经解决的内容：

- 论文没有学习场景级 illumination、visibility 或 global transport；它只近似局部反射 BRDF。
- 源材质的 native parameterization/reference 与目标解析 adapter 必须分开。PFMC 随机游走和 MERL/RGL 测量值是训练或验证 reference；两个 Principled BRDF 的混合只是近似目标，不能反过来冒充源 GT。
- “保留重要性采样”指解析近似目标内部的 evaluator/proposal 相匹配，不代表其 sampling distribution 与原 PFMC reference 完全相同。
- 论文没有公开训练、CPU renderer、checkpoint 或正式 sampler 代码；supplemental 中的 WebGL 交互 viewer 是展示/编辑 prototype，不是论文训练实现。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---:|---|---|
| Main paper `P` | [Eurographics item](https://diglib.eg.org/items/887713f3-27d2-4901-b01a-26c90c7eace3)，`cgf70549.pdf`，DOI `10.1111/cgf.70549` | 2026-08-29 | SHA-256 `E9D748C89CF2B06C09917AFE37F35BC803346369679F185F229272A1F05276BC` | 正式论文，13 页；公式、图、表、图注均逐页视觉核对。 |
| Main mirror `P` | [DisneyResearch PDF](https://studios.disneyresearch.com/app/uploads/2026/06/Neural-Material-Adapter-Transforming-Complex-Materials-into-Efficient-Analytic-BSDFs-Paper.pdf) | 2026-08-29 | SHA-256 `08D335C497B9083AF2073F788BE74EB7DADF03E30138BEF6CF432E0922C277EE` | 作者机构镜像；报告数值以 Eurographics 正式版为准。 |
| Supplemental bundle `S` | Eurographics item 的 `paper1096_mm_crc1.zip` | 2026-08-29 | SHA-256 `1007DDE49665514AF2FB83C96CD2A86E5888A0FDA97F3E61945D75971A866CA2` | 包含 AvA/tabulation 说明、W&W 图、92 个验证材质的五路 gallery 和 directional-parameter WebGL prototype。 |
| Supplemental derivation `S-AvA` | `additional-pdfs/AvA_Tabulation.pdf`，7 页 | 2026-08-29 | SHA-256 `7934A657AD9C595312621417B6534F12D992E4772185DBCE457E09E17EF5A986` | AvA、PFMC bin estimator 与离线 splatting 的补充推导；存在与正文的 loss/reciprocity/sample-budget 冲突，见 §11。 |
| Supplemental fits `S-WW` | `additional-pdfs/wandw_materials.pdf`，1 页 | 2026-08-29 | SHA-256 `CBF24D172FC72DF888148DBECC1B3BA619BAF4E098F273C5B9C0AF52DBA8D1DB` | 五个 Weidlich & Wilkie 材质的 reference/ours 定性对照；没有数值或训练配置。 |
| Supplemental gallery `S-gallery` | `nma-supplemental-viewer/` | 2026-08-29 | gallery HTML SHA-256 `0CE1BDEA6EF3FC39555E7F64DDA1C4CD84545AB13C11E7141D2BEDF2A6746A64` | `single/blended/wi/blended_wi/ref` 五个实际目录各有 92 张 JPG；UI 把 `ref` 展示为 reference。它们不是原始数据或 checkpoint。 |
| Supplemental prototype `S-dir` | `dir-param-viewer/` | 2026-08-29 | entry HTML SHA-256 `C9E66A2CA8A937F3850A0C8613013920F1A1A4DB1F0A3ED148C6A0CC662781E6` | 允许按 `cos(theta_i)` 编辑解析参数曲线；它使用 Three.js/WebGL 展示代码，不能用来补全正式 Mitsuba 网络配置。 |
| Official code/config/data `C` | 未找到公开仓库、正式 config、checkpoint、PFMC tables 或 train/validation manifest | 2026-08-29 | `not-applicable` | Disney 项目页、Eurographics item、ETH CGL 条目均未链接正式代码；官方 supplemental 只提供 viewer assets。公开入口未提供可锁定的训练/runtime repository。 |
| Author/project page `A` | [DisneyResearch|Studios 项目页](https://studios.disneyresearch.com/2026/07/01/neural-material-adapter-transforming-complex-materials-into-efficient-analytic-bsdfs/) | 2026-08-29 | HTML SHA-256 `CFB213B58ED9DA5CC3A8943F895AB1BB6BE474CC6A3204AE4B09BF0C701C1C5E` | 作者、摘要、正式 PDF 入口；没有 correction、code、data 或 talk。 |
| Author bibliography `A` | [ETH CGL publications](https://cgl.ethz.ch/publications/papers/papers.php)，条目 `Sha26a` | 2026-08-29 | HTML SHA-256 `C9D339221B191D635AFB9E5FF994E42CA893BBAEDDAC5ECE126F7B1AE7DC3E4E` | 只列 PDF/BibTeX/abstract；相邻论文有 supplemental/YouTube 时会显式列出，而本条目没有。 |
| Talk availability `A` | [MANER 2026 program](https://manerappearance.wordpress.com/maner2026/) | 2026-08-29 | HTML SHA-256 `5FAD7E26D7677152ED1061CC1B0B4FFAC7188B8FC5E02596197FA7147E929BD4` | 确认 Rajesh Sharma 报告题名；未找到第一方录像或 slides，不能作为额外技术证据。 |
| NeuralShading evidence `N` | [`docs/realtime_material_compilation.md`](../../../../../docs/realtime_material_compilation.md)、[`docs/material_scope.md`](../../../../../docs/material_scope.md)、[`docs/research/experiment_framework.md`](../../../../../docs/research/experiment_framework.md)、[`docs/learning.md`](../../../../../docs/learning.md) | 2026-08-29 | workspace current | 仅用于 §13–15 的项目映射，不反向补全论文事实。 |

官方 supplemental 是 87 MB 的静态站点 bundle，而不是传统单一 appendix。报告已视觉核对两个 PDF，并审计 gallery、曲线 CSV、WebGL entry 与 shader assets。当前没有可锁定的 official code commit；因此网络 activation、输出约束、CPU fused Eigen kernel、Mitsuba 0.5 adapter、sample/PDF 细节都只能登记为未报告，而不能从常见实现猜测。

## 3. 原论文的问题、假设与贡献边界

作者把现有方案的矛盾概括为：复杂解析/随机游走层材质准确、可编辑且能采样，但单次求值昂贵；直接神经 BRDF 容量高，却常需大网络、GPU、逐材质 latent/precompute，并且 evaluator 与 sampler 可能不一致。NMA 选择“解析 target + 小型参数 adapter”的混合路线。

论文的三个技术假设是：

1. 单个固定参数的 Principled BRDF 不足，主要瓶颈不是微表面 core 本身，而是参数对方向不变以及只有一个主要 colored lobe。
2. 把两个 Principled BRDF 以学习权重混合，并令它们的参数随 viewing direction `omega_i` 变化，可扩大 target gamut；对每个固定 `omega_i`，解析分量仍提供能量控制与可采样结构。
3. 随机 PFMC reference 无法直接给出解析 `f` 和 `pdf`，但可以把随机游走返回的 throughput splat 到 solid-angle bins；用 bin 平均能量监督 target 的 bin 平均值，能减轻窄峰被稀疏点采样漏掉的问题。

作者声明的贡献包括：

- 一个同时覆盖固定三层 PFMC 参数族与单独训练的 MERL corpus 的 hybrid material representation；
- directionally varying、two-Principled-lobe 的解析 target；
- AvA（Average-vs-Average）bin-energy supervision；
- 对同一 PFMC topology 内未见参数配置和 spatially varying 参数的 class-level zero-shot online translation，无逐材质 latent optimization、texture baking 或 preprocessing；
- CPU renderer 中比 PFMC reference 更低的 equal-spp variance 与更低 render time。

这些贡献不是一个统一网络从 PFMC 直接零样本迁移到 MERL/RGL/W&W。PFMC、MERL 和单材质 overfit 使用了不同训练范围；论文只证明“同一 adapter 设计可为不同 source class 另行训练”。

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | PFMC 三层 stack 的原生参数 `P`：上下界面 roughness `alpha_1,alpha_2`，两介质界面 IOR `eta_1,eta_2`，底层 conductor 的 RGB complex IOR `eta,kappa`，均匀 medium 的 RGB single-scattering albedo `rho` 与 scalar extinction `sigma_t`。 | 固定 topology；参数含 scalar 与 RGB。论文没有给出 flatten 顺序、归一化或网络实际 input dimension。 | `P` Table 1、§3.2–3.3 |
| Runtime query | 多材质 adapter 输入当前 shading point 的 `P` 和 viewing direction `omega_i`；single-material overfit 只输入 `omega_i`。 | 每个 shading point/view 在线调用；spatial 参数由纹理在该点给出。 | `P` Fig.4、§3.2、§4 “Textured Materials” |
| Direction coordinates | Background 采用 Mitsuba convention，并把 `omega_i` 称为 viewing direction、`omega_o` 称为 light direction：`f_r(omega_i,omega_o)=dL_o(omega_i)/dE_i(omega_o)`。PFMC adapter 的 Fig.4 实际把网络方向输入画成 `theta_i`；PFMC table 原为 4D direction pair，isotropy 去掉一个方位维后为 3D。 | 反射半球；table 具体为 supplemental 所写的 `40 x 1 x 40 x 80` solid-angle bins。正文未说明网络使用 `theta_i`、`cos(theta_i)` 还是其他归一化；§4 又把同一 `omega_i` 写成 incident direction，属于术语不一致。 | `P` Eq.(1)、Fig.4、§3.3、§4 Ablations；`S-AvA` §1.5 |
| Adapter output | 两组 Principled BRDF 参数 `p_1(omega_i),p_2(omega_i)` 与混合权重 `alpha(omega_i)`。 | 参数向量的字段、维数、range transform 未在论文/正式代码中披露。 | `P` Eq.(3)–(4)、Fig.4 |
| Runtime output | `alpha f_r(...;p_1)+(1-alpha)f_r(...;p_2)` 的 RGB BRDF 值；采样由解析 target 承担。 | 线性 RGB BRDF；不是 radiance，也不包含入射 illumination。 | `P` Eq.(1)–(4)、Conclusion |
| Validity/domain restrictions | PFMC 正式多材质实验限于 isotropic、reflection-only、固定三层 topology。对每个固定 `omega_i` 作者声明能量积分不超过 1；direction-conditioned 版本明确不满足 reciprocity。 | 不能直接用于要求互易性的 bidirectional path tracing。Discussion 推荐 blended-only 作为 reciprocal 设计点，但正文 p.6 同时写“Principled BRDF itself violates this”；在正式 target 实现不可得时，blended-only 的严格互易性保证未闭合。 | `P` Eq.(5)、p.6、Discussion p.11 |

方向不能只按变量下标解释。论文 Background/Eq.(1) 明确把 `omega_i` 定义为 **viewing direction**、`omega_o` 定义为 **light direction**，Fig.4 也把 `theta_i` 作为 adapter 输入；但 §4 Ablations 又称 `omega_i` 为 incident direction。第一方材料没有进一步消解这处术语漂移，正式 target code 也不可得。

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

PFMC class 的完整路径如下：

1. 从 Table 1 的分布随机抽取三层 native configuration `P`。
2. 对每个 `P`，均匀按 solid angle 抽 viewing direction `omega_i`；PFMC 随机游走输出配对的 `omega_o` 与 local throughput `t=f/p`。由于 `f` 与 `p` 都没有显式形式，把 `(omega_i,omega_o,t)` splat 到 4D/等价 isotropic 3D bin，形成 reference bin average `g_tilde_b`。
3. 训练时从某个 bin 内均匀抽 20 个 direction pairs，解析 target 在这些 pairs 上求值并平均，得到 `f_tilde_b`。公式上 target parameters 依赖各 pair 的 `omega_i`，但正文又称每个 minibatch sample 只做一次 network forward、只重复 target evaluation；它没有解释 20 个 pairs 是否共享同一 view-bin representative 或如何复用，缺少代码时不能进一步补全。
4. 用 log-domain absolute loss 比较 `g_tilde_b` 与 `f_tilde_b`，反向传播到 adapter 权重。
5. class training 完成后，runtime 直接读取 shading point 的 native `P`，以 `P,omega_i` 前向得到解析参数，不保存 per-material latent/table。解析 target 负责该 query 的 BRDF evaluation 和 sampling。

因此 NMA 的 learned capacity 不直接存完整 `f(P,omega_i,omega_o)`；它存的是从低维 source parameter space 到一个强解析 target family 的映射。source reference 只出现在训练 GT 生成中。

### 5.2 持久化表示

- PFMC multi-material 模型的持久化表示是 class-shared MLP weights；论文明确说 runtime 不需要 per-material latent、texture baking 或 per-material optimization。
- source 的纹理和 native parameters 仍是输入资产。每个 texel 的 `P` 独立通过同一 MLP，不先转换为 latent texture。
- `p_1,p_2,alpha` 是 query-time transient target parameters；论文未报告缓存布局、bytes 或能否跨多个 `omega_o` 重用。
- MERL 实验是例外：100 个测量材质通过 PyTorch `nn.Embedding` 学 per-material embedding，再由 3x32 MLP 解码。论文关于“无 latent code”的结论不能无条件覆盖这个 MERL branch。
- 没有 mip/LOD、quantization、weight packing 或 exported asset format。

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Single-material overfit | 正文写 `omega_i`；Fig.4 画为 `theta_i` | MLP，3 hidden layers x 32 neurons | 方向具体编码与 activation 均未报告 | `p_1,p_2,alpha` | 一个材质一套权重 | `P` §3.1、Fig.4 left |
| PFMC multi-material adapter | 正文写 `P,omega_i`；Fig.4 画为 `P,theta_i` | MLP，5 hidden layers x 64 neurons | 方向具体编码与 activation 均未报告 | `p_1,p_2,alpha` | 同一固定 topology/class 共享 | `P` §3.1、Fig.4 right、§4 setup |
| MERL multi-material model | learned material embedding 与 `omega_i` | MLP，3 hidden layers x 32 neurons；caption 报模型 7,159 parameters | 未报告；embedding dimension 未报告 | direction-conditioned blended Principled target | MLP shared，embedding per measured material | `P` Fig.12 caption、“Evaluation on MERL” |
| Analytic target | `omega_i,omega_o,p_1,p_2,alpha` | 两次 Principled BRDF evaluation 后 convex blend | 解析实现；输出参数的 clamp/transform 未报告 | RGB `f_r` | 共享解析 core | `P` Eq.(3)–(5) |

论文没有报告 input feature encoding、hidden activation、bias、parameter-vector schema、最后一层 activation、`alpha` 如何约束到 `[0,1]`、Principled 参数合法域如何保证、weight initialization 或确切 parameter count（MERL caption 的 7,159 除外）。supplemental curve viewer 暴露 `baseColor/roughness/metallic/clearcoat/specular/subsurface/specularTint/anisotropic/sheen/sheenTint/clearcoatGloss` 等手工曲线，但该 viewer 使用 WebGL/Three.js 展示代码，不能证明正式 Mitsuba NMA 的输出头就是这一字段列表。

### 5.4 条件化、坐标变换与物理先验

- **条件化**：PFMC branch 直接以 native `P` 加 `omega_i` 条件化；MERL branch 用 per-ID embedding 代替 native parametric `P`。
- **坐标/warp**：PFMC GT table 按 solid angle 均匀分 bin。supplemental 的一般讨论推荐 half-difference 与 nonlinear warp 处理窄峰，但正文正式 PFMC recipe 是 solid-angle `40x1x40x80` tabulation；Fig.4 只把 adapter 输入标成 `theta_i`，没有证据表明它采用 Rusinkiewicz/half-difference encoding，也没有给出 `theta_i` 到网络特征的精确变换。
- **解析先验**：两份 Principled BRDF 提供 microfacet/energy/sample structure；网络只选择解析参数与 blend。
- **能量与互易性**：正文 Eq.(5) 对固定 `omega_i` 声明 energy conservation，同时明确 direction-conditioned target 破坏 reciprocity。关于 base/blended-only target，p.6 的“Principled 本身不互易”与 p.11 的“blended-only 保持互易”互相冲突；没有正式 target code 时不能替作者消解。解析 prior 降低了网络容量需求，但不是“自动复原 source physics”。

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset/source assets/scenes | PFMC train：8,000 个随机三层配置；top 是 rough dielectric，下面是 homogeneous participating medium，底是 rough conductor。MERL：另训 100-material corpus。RGL Weta Brushed Steel、Mitsuba rough plastic、W&W coating materials 只做 single-material overfit/定性展示。 | `P` §3.3、Table 1、Fig.12–13；`S-WW` |
| Native parameter distributions | `alpha_1 ~ 10^{U(-3,-0.5)}`；`alpha_2 ~ 10^{U(-3,0)}`；`eta_1,eta_2 ~ U(1.05,2)`；conductor RGB `(eta,kappa)` 从 corpus range 抽样；medium RGB `rho = 1-U(0,1)^2`；`sigma_t ~ U(0,1)`。 | `P` Table 1 |
| GT/reference | PFMC/Guo et al. 随机游走是 layered source reference；它返回 stochastic outgoing direction 与 throughput。MERL/RGL 是 tabulated measurements。Principled mixture 是 target approximation，不是 GT。 | `P` §3 background、Eq.(8)–(9) |
| Train/validation/test split | 8,000 PFMC train；正文标准消融与 Fan 比较用 89 个 unseen PFMC validation materials；Table 3/4 和 supplemental gallery 使用 92 个 unseen materials。两组的关系、material IDs、抽样 seed 与独立 test set 未报告。 | `P` Fig.3、§4 setup、Table 3–5；`S-gallery` |
| PFMC reference queries | 每个 train material 抽 819,200 个 viewing directions `omega_i`，对每个执行随机游走得到 `(omega_o,t)`。`omega_i` 对半球 solid angle 均匀。bin estimator 是 `g_tilde_b ≈ (2pi N_b)/(Omega_b N) sum t`；supplemental 给出由隐式 sample weight 消去未知 PDF 的推导。 | `P` §3.3 Eq.(8)–(9)；`S-AvA` §2 |
| Table domain | 一般表是 `(theta_i,phi_i,theta_o,phi_o)`；isotropy 后使用 `40x1x40x80 = 128,000` bins。正文称 `(theta=40,phi=80)` per material。 | `P` §3.3；`S-AvA` §1.5 |
| Training query inside bin | minibatch 的每个 bin 内按 solid angle 均匀抽 `N_b=20` direction pairs，只做一次 adapter forward/样本，但对解析 target 求 20 次并平均。 | `P` §3.2、§3.4 |
| Filtering/LOD/footprint | 未报告；纹理实验依赖 renderer multisample anti-aliasing，论文明确说没有跨像素 shared state，但没有 footprint/mip filter。 | `P` “Textured Materials” |
| Augmentation/distillation/teacher | 没有 neural teacher/distillation。PFMC native random walk 是 reference；解析 target 是被训练的 student family。 | `P` §3.2–3.3 |
| Online/offline generation | class training 前形成离散 tabulation；runtime 对新 `P`/texture 不再生成 table。论文的“no precomputation”只适用于训练完后的逐材质使用，不等于训练 data preparation 在线且无持久数据。 | `P` Introduction、§3.3、Conclusion；`S-AvA` §1.5 |

Supplemental gallery 实际每种 variant 各有 92 张图；`mat_105,111,112,117,122,134,169,196` 不在 `101..200` 文件序列中。它与 Table 3/4 的 92-material protocol 对应更自然，但作者没有提供 manifest 来证明 gallery 与数值集完全同一，也没有解释 89-material 与 92-material 两套 validation 的筛选关系。

## 7. Loss、optimizer 与训练 lifecycle

| 项 | 正式配置 | locator |
|---|---|---|
| Target/output transform | 对 source bin target `g_tilde_b` 与 analytic target bin average `f_tilde_b` 使用 `log(1+y)`。 | `P` Eq.(6)–(7) |
| Loss | `L(y,y_hat)=abs(log(1+y)-log(1+y_hat))`；正文未说明 RGB channel reduction、batch reduction 或额外 regularizer。 | `P` Eq.(7) |
| Optimizer | Adam，learning rate `1e-5`；betas、epsilon、weight decay、gradient clipping 未报告。 | `P` §3.4 |
| LR schedule | single-material overfit：`ReduceLROnPlateau`；8,000-material model：`OneCycle`。OneCycle 的 max/min LR、warm-up fraction 与 plateau patience 未报告。 | `P` §3.4 |
| Batch/query count | batch size 1,000 bins；每 bin 20 target direction pairs，故每 step 做 20,000 次解析 target evaluation，但 adapter forward 的复用方式按正文是每 sample 一次。 | `P` §3.2、§3.4 |
| Steps/epochs/stages | 未报告 epoch/step 数，也未报告 table preparation 与 network training 是否流水执行。 | `P` §3.4 |
| Initialization/seed/model selection | activation、weight initialization、random seed、重复次数、checkpoint selection、early-stop rule 均未报告。 | `P/C` unavailable |
| Single-material cost | 3x32 overfit 约 5 分钟，单 RTX 3090。 | `P` §3.4 |
| Multi-material cost | 5x64、8,000 PFMC materials，单 RTX 3090 训练 2 小时；`S-AvA` 较宽松地写为 under 3 hours。PFMC table 生成时间和存储没有计入。 | `P` §3.4；`S-AvA` §1.5 |

`S-AvA` 的教学性公式把 PtP 与 AvA 都写成 squared error，而正式正文明确使用 log-domain absolute loss。报告将正文 Eq.(7) 视为正式配置；supplemental 的 squared formula 只作为 AvA “比较 bin mean” 的概念推导，不据此改写正式 loss。

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path/frequency | 每 shading point 读取 native `P`；对当前 view `omega_i` 运行一次 5x64 MLP，生成两组 target params 与 blend；对不同 light directions `omega_o` 可复用这些参数，再由解析 target 执行 evaluation/sampling。纹理是 per-texel 独立 MLP evaluation。 | `P` Eq.(4)、Fig.4、Textured Materials、Conclusion |
| Parameter count/MAC/FLOP | PFMC 5x64 的总 parameter/MAC 未报告，因为 input/output dimensionality未披露；MERL 3x32 caption 报 7,159 parameters。解析 target 还需两次 Principled evaluation 与 blend。 | `P` Fig.12 caption |
| Shared/per-asset/state bytes | PFMC weights class-shared；native texture/source params per asset。具体 bytes、transient target state bytes 未报告。MERL embeddings per material，但 embedding dimension/bytes 未报告。 | `P` Introduction、Fig.12 |
| Texture/feature fetches | 未报告固定 fetch count；native input textures 由 renderer 求值，NMA 自身没有 latent texture。 | `P` Fig.7 |
| Precision/quantization | 未报告；无 FP16/FP8/QAT/export evidence。 | `P/C` unavailable |
| Training backend | PyTorch network + Mitsuba 3 differentiable Principled target，RTX 3090。 | `P` §3.4 |
| CPU backend | 作者另写 Mitsuba 0.5 CPU-only network/target，MLP 用 fused Eigen kernels；该实现未公开。测试 CPU 型号、线程数、编译 flags 未报告。 | `P` Efficiency Comparison |
| GPU evaluation | 与 Fan decoder 的 table evaluation 在 RTX 4090；NMA 3.76 s，Fan 99.78 s。该时间不含 NMA class training，也没有明确说是否包含 Fan latent optimization；正文称其为 inference time。 | `P` Table 5、comparison text |
| CPU scene/render time | Teaser 64 spp：PFMC 186.6 s，NMA 93.1 s。92-material mean、32 spp：PFMC 9.90 s，NMA 5.75 s。图像 resolution、CPU hardware 未报告。 | `P` Fig.1、Table 4 |
| `sample()/pdf()` | 作者声明 evaluation 与 sampling 都由解析 target 完成并“built-in/accurate importance sampling”。没有公开 blend sampler、mixture PDF、reverse PDF、MIS event 或 delta handling 公式/代码。 | `P` Introduction、Conclusion；`C` unavailable |
| Precompute/amortization | runtime 对 unseen `P`/texture 无 per-material bake/optimization；一次 class training 与训练 table generation 被摊销且不计入 runtime 表。 | `P` Introduction、Conclusion |

论文的速度证据是完整 renderer 或整张 4D table 的 aggregate，不是单 query latency/MAC。5x64 MLP 与两次 Principled evaluation 的单次成本、不同 coherence/packet 宽度、shader inline 成本都没有测量。

## 9. 实验 protocol、baseline、指标与结果

作者使用 AE、MAPE、SMAPE、MRSE、PSNR、SSIM 和 HDR/LDR FLIP；Table 4 的 variance 是相对 2048 spp reference 的 MSE，Monte Carlo efficiency 定义为 `E=1/(variance*time)`。不同表的 validation 数量、渲染 spp 与硬件不同，不能跨表直接排名。

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| Target design | 89 个 unseen PFMC materials；所有 variant 同为 5x64；训练集从 100 增至 8,000，另画 validation-set overfit 虚线 | Single、`omega_i`、Blended、Blended+`omega_i` | Eq.(7) validation loss | 两个扩展各自优于 Single，组合最好；3k/5k train 已接近各自 overfit curve，8k继续收敛。图中 curve 没有附精确数值表。 | `P` Fig.3、§4 Ablations |
| Shader-ball reconstruction | PFMC validation render；正文 setup 写 89 materials | 上述四个 variants | L1/L2/MAPE/SMAPE/MRSE/SSIM/FLIP | Single：`.0055/.0003/2.6285/2.5006/.0124/.9954/.0484/.0387`；`omega_i`：`.0051/.0002/2.5037/2.3927/.0106/.9961/.0453/.0367`；Blended：`.0053/.0003/2.4593/2.3516/.0114/.9957/.0454/.0368`；完整：`.0050/.0002/2.3876/2.2867/.0103/.9963/.0432/.0352`。 | `P` Table 2、Fig.6 |
| Number of lobes | 92 unseen PFMC materials，256 spp，directional blend `N=1..4`；正文说另训 3/4/5-lobe variants，但表没有 `N=5` row | 1、2、3、4 Principled instances | 同上 | N=2 大多指标优于 N=1；N=3 非单调变差但 SSIM最高 `.9610`；N=4 的 L1 `.01004` 与 LDR FLIP `.0465` 略优于 N=2，而其他多数指标无一致增益。作者据此选 2 lobes 作为 complexity/quality tradeoff。 | `P` Table 3 |
| Texture generalization | 5x64 只在常量随机 `P` 上训练；runtime 给 conductor roughness、medium `sigma_t`、medium albedo 纹理 | constant input | 定性 render | 多纹理能揭示底层 metal/colored medium；没有 resolution、filter、指标或 held-out texture corpus。 | `P` Fig.7 |
| CPU vs PFMC | 92 unseen materials，32 spp；相对 2048 spp reference；Mitsuba 0.5 CPU implementations | Guo PFMC | time、variance、`E` | PFMC `9.90 s,1.49e-3,E=94.5`；NMA `5.75 s,1.19e-3,E=158.1`；ours/reference 为 `0.58x` time、`0.80x` variance、`1.67x` efficiency。 | `P` Table 4 |
| Teaser equal spp | 同场景 64 spp，CPU | Guo PFMC | time、SMAPE | PFMC 186.6 s、SMAPE .052；NMA 93.1 s、SMAPE .037。NMA equal-spp 既更快又更低噪；不是 equal-time protocol。 | `P` Fig.1 caption |
| Variance convergence | teaser，1–4096 spp，对 8192 spp reference | PFMC | MSE curve | NMA 前期更快下降；高 spp tail 保留略高 representation floor，作者明确称为 representation gap。 | `P` Fig.8 |
| Medium-density scaling | 同 spp；横轴 `sigma_t=0.01..0.99` | PFMC | CPU render time | NMA curve近似常量；PFMC 随 `sigma_t` 增大而变慢。图未提供精确数值表。 | `P` Fig.9 |
| Fan et al. comparison | 89 PFMC validation materials；为每材质用 Fan official implementation 优化 latent；40x80x40x80 bins，每 bin 10 uniform locations；RTX 4090 | Neural Layered BRDFs decoder | AE/PSNR/SMAPE/Val.Loss/time | NMA `.0065/45.5/.100/.0037/3.76s`；Fan `.0130/39.7/.257/.0100/99.78s`。论文说明未比较 Fan sampler，因为 release 没有 sampler weights/code。 | `P` Table 5、Fig.10–11 |
| MERL branch | 100 MERL materials，另训 3x32 + `nn.Embedding` | tabulated MERL reference | 定性 interpolation | 能 fit `color-changing-paint2`，四材质 embedding 的 bilinear interpolation 产生平滑中间外观；无 held-out measured-material 数值。 | `P` Fig.12 |
| Other source overfits | W&W coating、Mitsuba rough plastic、RGL Weta Brushed Steel，single-material fits | 各自 reference | FLIP/定性 | Fig.13 三例 FLIP `.042/.060/.065`；supplement 另给 W&W 五例图但无数值。只证明 target gamut/overfit，不证明跨 source zero-shot。 | `P` Fig.13；`S-WW` |

Fan 对照是重要但不完全对称的 representation comparison：NMA 先在同类 8,000 PFMC configurations 上 class-train；Fan 则用其公开 decoder 对每个 validation material 优化 latent。Table 5 的时间被正文称为同 GPU inference time，未明确纳入 latent optimization；同时不含 Fan sampler。它支持该 protocol 下的 decoder accuracy/time 结论，但不能单独证明完整 end-to-end asset conversion 或 path-tracing pipeline 的普遍优越性。

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 证据边界 | locator |
|---|---|---|---|---|---|
| `author-negative` | 用 gradient descent 直接把 constant single Principled BRDF overfit 到复杂 PFMC material | 无法匹配 colored/view-dependent lobes | 固定方向无关参数和单个 non-scalar lobe 的 gamut 不足 | 该实验只比较 target family，没有 optimizer/seed 对照，不能作为 optimization variance 证据。 | `P` §3.1、Fig.2 |
| `ablation-inferior` | Multi-material Single，5x64，与完整模型同层宽 | Table 2 所有主要指标落后完整模型 | layered appearance 同时需要 extra multiscattering/multiple colors 与 directional effect | 所有 variant 同层宽；作者据此把改善归因于 target extensions，而非网络层宽变化。 | `P` Table 2、Ablations prose |
| `ablation-inferior` | `omega_i` + single lobe | 明显优于 Single，但多数指标仍落后完整模型 | 方向变化解决 grazing/color shift，单 lobe 仍难表达多个 colored components | 只支持该 PFMC protocol 下的相对结果；没有单独的跨 topology 结论。 | `P` Table 2 |
| `ablation-inferior` | two-lobe Blended、无 direction conditioning | 百分比类指标优于 Single，仍落后完整模型 | 额外 lobe 捕获多个颜色/多散射，grazing angle 变化仍不足 | Discussion 把它列作 reciprocity design point，但正文对 base Principled 的互易性陈述冲突，严格保证未闭合。 | `P` Table 2、p.6、Discussion p.11 |
| `author-negative` | 固定 grid center 的 1 sample/bin point supervision | rough conductor example loss `.201`，complex layered `.076`，fit 模糊/漏峰 | 窄峰小于 bin spacing，grid 无法提供能量位置的监督 | supplemental 明确说固定中心盲点不会随训练时长消失；这仍是作者推导，未提供不同 network capacity 的实验。 | `P` Fig.5；`S-AvA` §1.1–1.4 |
| `ablation-inferior` | bin 内随机 1/2/4/8 samples | rough conductor 约 `.161/.142/.078/.051`；complex layered 约 `.059/.054/.053/.051`，增加到 4–8 后收益递减 | 多点均值降低漏峰与梯度方差 | 论文正式训练用 20 samples/bin；Fig.5 数值只来自图中示例，不是全数据统计。 | `P` Fig.5 |
| `ablation-inferior` | directionally blend 3/4 lobes | N=3 多数指标比 N=2 差；N=4 只有少数指标微胜，整体非单调 | 超过两个 lobe 收益小而增加 inference complexity | Table 3 caption 还称训练了 5-lobe variant，但没有给出 N=5 row；结论只覆盖已公开的 N=1..4 数值。 | `P` Table 3 |
| `known-limitation` | 高 spp rendering | error curve出现高 spp representation floor，略高于 PFMC reference | target approximation 存在 representation gap | Table 4 的 equal-spp variance 优势不能消除 Fig.8 显示的 approximation bias。 | `P` Fig.8、Table 4 |
| `known-limitation` | challenging validation layer configurations | Fig.11 少量 case 与 Fan 误差相近，而非一致胜出 | 配置更难 | 没有公开 IDs/参数，不能进一步归因。 | `P` Fig.11 text |

在已获得第一方材料中，没有报告以下开发历史：其他 activation、output parameterizations、normalization、不同 optimizer、失败 seed、不同 table resolution、不同 PFMC topology 的系统失败实验。不得从最终 5x64/2-lobe 配置反推这些尝试发生过。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Architecture | 3x32 single/MERL、5x64 PFMC；输出两组 `p` + `alpha` | directional curve viewer 展示单个 PBR material 的手工曲线 | 无正式 NMA code/config | viewer 是 editing prototype，不能补全 MLP activation、output schema 或 Mitsuba target。 |
| PFMC table/query | 每材质 819,200 total random `omega_i`；isotropic 3D table | `40x1x40x80=128,000` bins；§1.5 又写 practical pre-splatting 使用“millions of samples per bin” | 无 data generator | 两者不能描述同一正式 table：前者平均仅约 6.4 个 total PFMC samples/bin，而后者至少要求 `O(10^11)` samples/material。以前者正文正式 recipe 为准，后者保留为无法与正文闭合的 supplemental statement。 |
| Loss | 正式 Eq.(7) 是 log1p absolute error | §1.4 的 PtP/AvA 教学公式使用 squared error | 无 training code | 形式不一致；报告采用正文正式 loss，同时保留 gap。 |
| AvA forward | 每 bin 20 target pairs；正文同时称 adapter 每 minibatch sample 一次、target 多次 | 讨论 `N=1/15` 与预先 splat fixed target | 无 config | `p(omega_i)` 对 direction 有依赖，正文未解释多个 pair 如何共享一次 network forward；无法确认正式 data loader 的 view-bin representative、复用或 cache。 |
| Physical validity | Eq.(5) 声明 energy conservation；p.6 说 direction-conditioned NMA 破坏 reciprocity且“Principled BRDF itself violates this”；p.11 又说 blended-only remains reciprocal | `S-AvA` §1.5 泛称“physically valid target”会 by-construction 继承 reciprocity 与 energy conservation | 无 sampler/target code | direction-conditioned NMA 非互易、会使 bidirectional PT biased 是正文一致结论。base/blended-only 的互易性则 main↔main 已冲突，supplemental blanket claim 又未区分 variant；在实现不可得时不得宣称严格保证。 |
| Training time | PFMC 2 h，single overfit约5 min，RTX3090 | 8,000 materials under 3 h | 无 logs | 宽松上界不冲突；但 table generation cost 均未报告。 |
| Runtime | 自研 Mitsuba0.5 CPU + fused Eigen；Table4/Fig1/Fig9 | WebGL/Three.js prototype | CPU renderer、weights、export均无 | supplemental 不能复现实验 backend。 |
| Assets/evaluation | 89-material ablation/Fan；92-material lobe/runtime | gallery含92 materials x 5 variants；W&W额外图 | 无 manifest/raw tables | 92 张图可核对定性结果；89/92 set relationship 未报告。 |
| Sampler | 声称 target evaluation/sampling matched | prototype不含论文 path-tracing sampler | 无 `sample/pdf` code | 可采样是 architecture-level author claim；完整 mixture PDF、reverse PDF 与 renderer event correctness 无可审计证据。 |

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. **正式 PFMC domain 很窄**：isotropic、reflection-only、固定 rough-dielectric / homogeneous-medium / rough-conductor 三层 topology。作者说这不是概念上的根本限制，但没有提供跨 topology 的多材质实验。
2. **互易性破坏**：`p(omega_i)` 令 `f(omega_i,omega_o) != f(omega_o,omega_i)`；作者明确说用于 bidirectional path tracer 会有 bias。Discussion 建议要求 reciprocity 时使用质量较低的 blended-only variant，但 p.6 又称 base Principled 本身不互易；因此该替代的严格保证属于作者内部未闭合项。
3. **deeper stacks 未验证**：作者提出未来可借鉴 Fan 的 recursive layer composition 并增加 lobes；这不是当前结果。
4. **anisotropy 仅有 single-material overfit**：Fig.13 的 RGL 例子证明 target 有一定 representational capacity，不证明 5x64 PFMC class model 对未见 anisotropic materials 泛化。
5. **representation floor**：Fig.8 高 spp tail 显示 analytic target 近似误差不会随采样数消失。
6. **measured-material 范围**：MERL branch 用 100 个 per-ID embeddings 训练，没有 held-out material zero-shot protocol；RGL/W&W 是单材质 overfit。
7. **texture 证据仅定性**：参数可逐 texel 求值，但没有 footprint、mip、filtering、subpixel texture statistics 或 high-resolution quantitative result。
8. **重要性采样只匹配近似目标**：即使 target 的 evaluator/sample一致，也不消除 target 与 PFMC source distribution 的 representation gap。

### 12.2 未报告/材料不可得

- Principled parameter vector 的正式字段、维数、颜色通道展开、`alpha`/roughness/IOR 等输出约束；
- MLP direction/source input encoding、activation、normalization、initialization、exact parameter count/MAC（MERL 7,159 除外）；
- Fig.4 的 `theta_i` 究竟以角度、余弦还是其他特征进入网络，以及 §4 “incident direction” 与 Eq.(1) “viewing direction” 的术语关系；
- Adam 完整 hyperparameters、OneCycle/Plateau 配置、steps/epochs、seed、重复训练与 checkpoint selection；
- PFMC table generation time、并行硬件、存储量、空 bin 处理、RGB reduction、NaN/outlier policy；
- 89 与 92 validation set 的 manifest、关系和独立 test set；
- CPU 型号、线程数、render resolution、compiler flags、Eigen kernel 代码；
- target blend sampler 与 `pdf()` 的精确算法、reverse PDF、MIS、delta/transmission event；
- runtime precision、weight bytes、fetches、cache/state layout、export/renderer integration；
- official training/runtime code、config、checkpoint、PFMC tables、measured inputs 与 performance logs；
- source parameter 超出 Table 1 distribution 时的 behavior、deep topology、transmission、production asset editing latency；
- talk/slides/correction；截至 2026-08-29 只找到 MANER 日程，无第一方录像。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

NMA 的成功并不主要来自让 5x64 MLP 自己记住四维 BRDF。容量分布在三处：

1. **解析 target family** 承担最昂贵的 inductive bias：微表面 lobe、Fresnel、能量控制和可构造的采样分布。
2. **两个 colored Principled instances + direction-conditioned parameters** 扩大 target gamut；Table 2 的同宽消融说明输出结构比单纯增加网络容量更关键。
3. **class-level native parameter mapping** 只学习 `P,omega_i -> analytic parameters` 的光滑映射，因而能使用小 MLP。PFMC native `P` 本身已经是一份高信息、低维的 source description；对于任意 graph/texture/material family，不能假设同样存在等价低维参数轴。

AvA 把一部分难度从 network capacity 转移到 source-query/data aggregation：训练前用大量 PFMC random walks 形成 bin-energy target，再用每 bin 20 个解析 target queries 估计 student mean。报告的 2 h 只覆盖 network training，不能视为完整 compilation cost。

### 13.2 成功所依赖的假设

- source 是同一 fixed topology、连续可参数化的 family，且 unseen states 仍落在 Table 1 train distribution；
- 两个 Principled lobes 对目标局部散射具有足够近似能力；
- 对 production forward path tracing，non-reciprocity 的 bias 可接受；
- runtime 接受每个 shading-point view `omega_i` 一次 MLP，并对每个 light query 做两次 analytic BRDF，而不是只允许极小 direct evaluator；
- source authoring inputs/纹理可在 shading point 直接取得，且 per-texel MLP 没有昂贵 graph evaluation bottleneck；
- 离线 class training 和 PFMC tabulation 能被大量 assets/amended states 摊销。

### 13.3 可迁移机制与不能迁移的部分

可迁移机制：

- **结构消融顺序**：固定 single target -> two-lobe constant -> direction-conditioned single -> direction-conditioned two-lobe。它能区分 target gamut、direction conditioning 与网络宽度，而不是把改善统称为“neural capacity”。
- **bin-integrated supervision**：对 stochastic/noisy reference，比较小区域平均能量而非单点，可能保护窄峰并降低 query variance。
- **native-source compiler 思路**：若 source family 有稳定、可编辑的 typed parameters，可用 class-shared network把编辑状态前向转换为 runtime representation，无需逐状态优化。
- **analytic matched proposal**：即使 direct neural evaluator 最终保留，NMA 的解析 two-lobe target 仍可作为 sampler proposal/control。

不能直接迁移：

- 本项目要求各 source family 的 native reference 直接产生 GT；不能先把 MaterialX/MDL/OpenPBR 反演成 PFMC/Principled 参数再称为 source GT。
- 正式训练只允许 GPU-resident online reference query，不保存/读取 batch corpus。NMA 的 pretabulated `40x1x40x80` tables 不能原样进入正式 recipe；若采用 AvA，需设计 online bin query 或只把高样本 table 用作 isolated optimized-code control。
- NMA runtime 是“小 MLP 生成解析 closure 参数 + analytic evaluation”，不是小 MLP 直接输出 `f`。依据项目目标，它更适合作为 **compiler/control/analytic proposal**，而不是自动替代当前 direct neural evaluator。
- PFMC zero-shot 结论只覆盖同一三层 topology；MERL branch 甚至依赖 per-ID embeddings。不能据此宣称对任意未见材质 graph 的 universal compiler。

### 13.4 与本项目 runtime contract 的关系

- **静态有界**：5x64 MLP、两次 Principled evaluation 和固定 blend 都是静态有界，随机访问不依赖邻域或历史；这一点与工程合同相容。
- **`evaluate` 语义**：target 输出线性 BRDF `f`，不含 illumination；与项目 bare-`f` 语义相容。
- **`prepare()` 复用匹配**：按 Background/Eq.(1) 的显式定义，论文 `omega_i` 是 viewing direction，语义上对应项目 `wo`；`p_1,p_2,alpha` 只依赖 native `P` 与这个 view，可放入 `prepare(c,wo)` 并供多个项目 `wi`/论文 `omega_o` 复用。§4 又写 incident direction、Fig.4 只标 `theta_i`，所以接入必须显式冻结 convention 与 feature transform，不能交换非互易函数的两个实参，并须用 dense directional parity 验证。
- **`sample/pdf`**：解析 target 概念上能提供 matched sampling，但论文没有公开实现或 API contract。接入项目 dispatcher 前必须独立验证 `sample -> pdf/weight` 恒等式、forward/reverse PDF 与 event semantics。
- **硬件预算**：论文只证明 CPU scene aggregate 与 RTX4090 table aggregate；没有 shader MAC/fetch/bytes。5x64 本身还叠加两份 Principled BRDF，不能仅因“lightweight”就判定通过当前 shader budget。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

当前 [`docs/learning.md`](../../../../../docs/learning.md) 的 NVIDIA functional reproduction 是 native-parameter encoder/bootstrap、hierarchical latent、direct `f` evaluator 与 learned GGX9 sampler；它和 NMA 是不同 representation family。这里不把 NMA 论文当作 NVIDIA 实现规格，也没有任何 `suspected-defect` 结论。

| 主题 | 状态 | 对应关系与影响 |
|---|---|---|
| Native source input | `interface-adaptation` | 二者都证明 source parameters 可进入 learned compiler，但 NVIDIA encoder 先生成/烘焙 latent，NMA 在每 query 直接生成 analytic params。不能用 NMA 替 NVIDIA latent lifecycle 补规范。 |
| Evaluator output | `not-applicable` | NVIDIA evaluator 直接输出 `f_hat=exp(raw-3)`；NMA 输出 analytic parameters 再求两次 Principled。NMA 可成为新候选/control，不是 faithful NVIDIA variant。 |
| Training data | `intentional-deviation` | 项目 formal route 按 [`docs/research/experiment_framework.md`](../../../../../docs/research/experiment_framework.md) 在线 GPU source `evaluate().f`；NMA 用离线 PFMC bin tables。若测试 AvA，必须登记为 adaptation，并匹配 total source-query work。 |
| Direction conditioning | `author-underspecified` + `interface-adaptation` | NMA Background/Eq.(1) 把 `omega_i` 定义为 viewing direction，语义对应项目 `wo`，因此 network 放入 `prepare` 与公式数据流一致；但 §4 使用 incident wording、Fig.4 只标 `theta_i`。适配点是符号/convention 与未报告的 feature transform；不能依据下标直接接到项目 light-direction `wi`。 |
| Sampler | `not-applicable` | 当前 NVIDIA sampler 是 learned two-lobe GGX9 proposal，loss 由 learned evaluator 构造；NMA sampler来自解析 target。两者可做 matched proposal 对照，但不能互相冒充。 |
| Shader/runtime budget | `author-underspecified` | NMA 缺 MAC/bytes/fetch/precision/export；项目必须按 [`docs/realtime_material_compilation.md`](../../../../../docs/realtime_material_compilation.md) 和 method constraints 独立静态记账。 |
| Source semantics | `faithful`（原则层） | NMA 的 PFMC branch 在 runtime 保留 native `P`/textures 作为输入，没有把 target Principled 参数当 source authoring GT；这与 [`docs/material_scope.md`](../../../../../docs/material_scope.md) 的边界一致。 |

对 NVIDIA 复现最直接的价值不是替换网络，而是增加两个 control：一是同 source queries 下的 analytic two-lobe adapter；二是当前 learned sampler 相对“解析 target 自带 sampler”的 quality/time/variance 比较。两者都应作为新实验注册，而不是修补 NVIDIA correspondence。

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：direction-conditioned two-lobe analytic target 在当前 LayerStack family 上比 constant/single targets 有更好的 quality-time Pareto | `P` Table 2 的同宽四路消融 | 当前 reference 的多散射/colored grazing structure 也能被两份解析 lobes近似 | Single、constant two-lobe、direction-single、direction-two-lobe；相同 hidden layers/width、train queries、optimizer与 seed | source split、query recipe、analytic core、output range、训练预算 | local `f` log error、dense grazing tail、energy、single-query time/MAC；bootstrap CI | compiler + analytic evaluator/proposal | 完整 variant 在主要 held-out quality 指标上无显著改善，或其时间/bytes使其被任一较简单 variant Pareto dominate。 |
| H2：online AvA/bin-energy supervision 比 pointwise log-L1 更稳地保留极窄峰 | `P` Fig.5；`S-AvA` 对漏峰机制的推导 | 不持久化 table 时，online 同-bin query 仍能获得相同的 integrated supervision收益 | pointwise 与 AvA 使用完全相同的 total source reference evaluations；另设 center-grid 与 random-point control | model、optimizer、material states、方向分布、wall-clock/query work | peak energy/width、dense tail、seed variance、收敛斜率、GT query cost | training-only adaptation；runtime不变 | AvA 在 matched query work 下对峰值/长尾/seed CI 无改善，或训练成本增幅使同 wall-clock pointwise 更优。 |
| H3：native-parameter source compiler 能在 G2s 未见编辑状态上接近 response-derived/optimized code | PFMC 8,000-class model对同 topology unseen configurations/texture的 zero-shot结果 | 当前 typed source family 也存在足够平滑、信息完备的 native parameterization | 项目 M6 source compiler vs M5 target encoder vs autodecoder，三者共享同一 direct neural evaluator/latent bytes；不要把 NMA analytic target差异混入 | evaluator、latent D、asset bytes、train family/split、seed与query budget | G1/G2/G2s local quality、compile latency、edit latency、workflow W、runtime time/bytes | compiler（训练期/prepare前） | compiler 在 G2s 的 CI 明显落后 response-derived control且无法用可接受 compile latency换回，或对 native edits出现不连续/越界失败。 |
| H4：把 NMA view-conditioned analytic parameters 缓存在 `prepare(c,wo)` 可在不改变函数的前提下摊销多次 light query | `P` Eq.(4)/Fig.4：`p_1,p_2,alpha` 只依赖 source `P` 与 viewing `omega_i` | 论文 Mitsuba convention 可无歧义映射到项目 `wo/wi`，且 native `P` 在同一 shading state 内不变 | uncached 每 query 重算 adapter vs cached prepare state；相同 weights、analytic target 与 query pairs | direction convention、precision、source state、sampler | dense `f` parity、sample/PDF parity、prepare cost、amortized per-light time | prepare + analytic evaluator/proposal | cached/uncached 超出预先冻结的数值容差，或状态依赖实际含逐-light输入而无法安全复用。 |

这些假设都要求本项目重新实现并验证；论文事实不构成当前候选已经成立的证据。

## 16. 证据索引

### `P` 正文

- `P-identity`：p.1，DOI、作者、venue、Fig.1、abstract。
- `P-problem`：§1–2，解析 layer model、neural representation 与 parameter adapter 的边界。
- `P-target`：§3.1，Eq.(3)–(5)，Fig.2–4；two-lobe blend、`omega_i` conditioning、Fig.4 的 `theta_i` 输入标记、3x32/5x64。
- `P-AvA`：§3.2，Eq.(6)–(7)，Fig.5；bin average 与正式 log1p absolute loss。
- `P-data`：§3.3，Eq.(8)–(9)，Table 1；PFMC throughput estimator、8,000 configurations、819,200 directions/material。
- `P-train`：§3.4；Adam `1e-5`、batch 1,000、20 pairs/bin、scheduler、RTX3090 time。
- `P-ablation`：§4，Fig.3/6、Table 2/3；四路 target 与 lobe count。
- `P-texture-runtime`：Fig.7–9、Table 4；texture、CPU variance/time 与 representation gap。
- `P-Fan`：Fig.10–11、Table 5；89-material 4D table protocol、RTX4090 aggregate。
- `P-measured-limit`：p.6、Fig.12–13、Discussion、Conclusion；MERL embeddings、single overfits、reciprocity 的 main↔main 冲突、fixed-topology/deeper-stack 边界。

### `S` supplemental

- `S-AvA-nyquist`：`AvA_Tabulation.pdf` §1.1–1.4；point-vs-bin mean、窄峰与 sample-count 分析。
- `S-AvA-precompute`：同 PDF §1.5；`40x1x40x80`、pre-splat 描述、under-3h 声明。
- `S-PFMC-estimator`：同 PDF §2–2.1；未知 PDF 下用 PFMC sample weight 与 bin solid-angle volume 估计 mean。
- `S-gallery`：92 materials 的 `single/blended/wi/blended_wi/reference` render images。
- `S-dir`：directional parameter curve prototype；只支持编辑展示，不是 NMA training/runtime code。
- `S-WW`：五个 W&W reference/ours 定性 fits。

### `C` official code/config/data

- 截至 2026-08-29 未找到；没有可锁 commit。supplemental viewer code 属 `S`，不提升为正式 `C`。

### `A` author/project

- `A-Disney`：DisneyResearch|Studios project page，摘要与 PDF。
- `A-CGL`：ETH CGL `Sha26a`，PDF/BibTeX/abstract，无 code/supp/talk link。
- `A-MANER`：MANER 2026 日程确认报告存在；无公开 recording/slides。

### `N` 项目证据

- `N-runtime`：[`docs/realtime_material_compilation.md`](../../../../../docs/realtime_material_compilation.md)，`prepare/evaluate/sample/pdf` 与 direct neural runtime。
- `N-source`：[`docs/material_scope.md`](../../../../../docs/material_scope.md)，native source/reference 不得被 adapter target 取代。
- `N-online`：[`docs/research/experiment_framework.md`](../../../../../docs/research/experiment_framework.md)，GPU-resident online reference 与 matched controls。
- `N-nvidia`：[`docs/learning.md`](../../../../../docs/learning.md)，当前 NVIDIA evaluator/sampler/latent lifecycle。

### `I` 分析

- §13–15 中关于 compiler/control/proposal 分类、论文 `omega_i` 到项目 `prepare(wo)` 的 convention mapping、online AvA adaptation 及四项可证伪实验均为本项目推论，不是作者声明。

## Evidence review

```text
author_worker: /root/taming2026
reviewer: /root/adapter2026_review
reviewed_at: 2026-08-29
sources_rechecked:
  - Eurographics formal main PDF 13/13 pages, Disney mirror and both SHA-256 locks
  - AvA_Tabulation supplemental PDF 7/7 pages and SHA-256
  - W&W supplemental PDF 1/1 page and SHA-256
  - supplemental ZIP, five 92-image gallery directories, gallery ID set and direction-parameter prototype assets
  - Disney project page, ETH CGL Sha26a entry and MANER 2026 program HTML locks
  - current NeuralShading runtime, source-scope, online-training and NVIDIA correspondence documents
findings_closed:
  - verified all 16 report sections and kept project inference after the factual evidence layers
  - re-transcribed Eq.(3)-(9), Table 1-5, Fig.1/3/5/8/9/12/13 configurations and numerical results
  - separated Eq.(1) viewing-direction semantics, Fig.4 theta_i input and section 4 incident-direction wording
  - preserved main-vs-supp log1p-L1 versus squared-loss conflict
  - quantified 819200-total versus millions-per-bin PFMC sample-budget incompatibility
  - preserved main-vs-main and main-vs-supp reciprocity conflicts without inventing a guarantee for blended-only
  - verified 89/92 evaluation counts, the omitted N=5 table row and all five 92-image gallery sets
  - verified that official training/runtime/sample code and configs are unavailable from locked first-party entry points
remaining_evidence_gaps:
  - official training/runtime/sample code, configs, checkpoints and data are unavailable
  - formal Principled output schema, activation, direction encoding and range constraints are unreported
  - loss, reciprocity and PFMC sample-budget conflicts have no author correction or executable implementation to resolve them
  - 89-vs-92 validation split relationship and material manifest are unavailable
  - CPU hardware/render resolution and complete compilation/precompute cost are unreported
  - Table 3 claims a trained 5-lobe variant but provides no N=5 row
review_outcome: passed-with-explicit-gaps
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
