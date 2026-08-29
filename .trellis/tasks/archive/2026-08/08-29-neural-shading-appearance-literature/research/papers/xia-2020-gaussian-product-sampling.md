---
paper_id: "xia-2020-gaussian-product-sampling"
title: "Gaussian Product Sampling for Rendering Layered Materials"
authors: "Mengqi (Mandy) Xia, Bruce Walter, Christophe Hery, Steve Marschner"
year: "2020"
venue: "Computer Graphics Forum 39(1), 420–435"
doi: "10.1111/cgf.13883"
report_status: "evidence-reviewed"
main_source: "https://diglib.eg.org/bitstream/handle/10.1111/cgf13883/v39i1pp420-435.pdf"
supplemental_status: "unavailable"
official_code_status: "unavailable"
official_code_commit: "not-applicable"
author_worker: "/root"
reviewer: "/root/belcour2018_review"
last_verified: "2026-08-29"
---

# Gaussian Product Sampling for Rendering Layered Materials

## 1. 研究对象与报告边界

Xia、Walter、Hery 与 Marschner研究如何降低薄层状材质 position-free Monte Carlo 求值中的方差。Guo 2018 把 layered BSDF 写成只含深度与方向的 path integral，但固定外部入射、出射方向后，一条含 `n` 个内部方向的 path component 有 `n+1` 个相邻 BSDF 因子，而通常的 sequential proposal 只有 `n` 次方向采样：至少有一个可能很尖锐的因子没有进入 proposal。本文用 Gaussian approximation 构造相邻两个或三个 BSDF 因子的乘积分布，让 proposal 更贴近 path contribution。[P Abstract, §§1,3–6]

本文本身不是 neural shading 方法。它的正式对象是 `local-material transport` 中 stochastic layered-BSDF evaluation 的 variance reduction，直接承接 [Guo 2018](./guo-2018-position-free-layered-bsdfs.md)；与 Belcour 2018 相比，两者分别保持随机估计的 exact-convergence 语义和采用有偏闭式近似。本报告重建：

1. layered path component 为什么必然遗留一个未采样因子；
2. pair-product 与 multiple-product proposal 的数学构造和实际适用范围；
3. isotropic microfacet slice 到二维 Gaussian mixture 的离线拟合方法；
4. 正式场景、RMSE 与作者所称 `effective time reduction` 的定义；
5. 已报告的退化情形、论文内部数值冲突和未披露的复现配置；
6. 它对本项目 online reference、matched sampler 与 learned evaluator 的准确影响。

必须区分两套 sampling domain：pair/multiple product 是固定 `(ω_i,ω_o)` 时 layered-BSDF `evaluate()` 内部 path integral 的 proposal；场景 integrator 所调用的外部 BSDF `sample(ω_i)` 仍沿用 Guo 的 forward layer-stack random walk，而外部 `pdf(ω_o|ω_i)` 通常是独立随机数估计的 path-generation probability integral。后两者没有被 Gaussian product 直接替换，也不能把 parametric Gaussian fits 当作 neural material representation。[P §§3–5,7, Eqs.22–23]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | [Eurographics Digital Library 正式条目](https://diglib.eg.org/items/29f27579-086d-47b6-83c1-af7329112c79)；[正式 PDF locator](https://diglib.eg.org/bitstream/handle/10.1111/cgf13883/v39i1pp420-435.pdf)；DOI `10.1111/cgf.13883` | 2026-08-29 | 无本地 PDF hash：公开索引可读取 16/16 页正文，旧 bitstream 的匿名直接下载返回 DSpace Login/401 | 正式 CGF 39(1), pp.420–435 版本；方法、公式、图表、实验和限制 |
| Publisher record `A-Wiley` | [Wiley DOI page](https://onlinelibrary.wiley.com/doi/10.1111/cgf.13883)，first published 2019-10-31，issue 2020-02 | 2026-08-29 | 固定 URL | 核对 DOI、出版时间与卷期；不把 online-first 2019 改写成卷期年份 |
| Author/project listing `A-Cornell` | [Cornell Graphics and Vision Group 条目](https://rgb.cs.cornell.edu/papers/gaussian-product-sampling-for-rendering-layered-materials/) | 2026-08-29 | 固定 URL | 核对作者、摘要和正式论文入口 |
| Supplemental `S` | 正式条目、作者组页面和论文链接范围内未发现独立 supplemental/appendix package | 2026-08-29 | unavailable | 无法补足精确拟合网格、原始图像、seeds 或更多失败配置 |
| Official code/config/data `C` | 正式条目与作者组页面未提供 official implementation；未使用非作者 student repo 或第三方镜像 | 2026-08-29 | unavailable | 不能审计 epsilon、MIS heuristic、sample allocation、termination 或正式 scene configs |
| NeuralShading evidence `N` | [runtime contract](../../../../../docs/contracts/scattering_backend.md)、[experiment framework](../../../../../docs/research/experiment_framework.md)、[method constraints](../../../../../.trellis/spec/project/method-constraints.md)、[NVIDIA correspondence](../../../archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md) | 2026-08-29 | repo-local | 只用于 §§13–15，不回填成论文事实 |

公开 PDF 的文字、公式、table 与 caption 已按 16 页顺序阅读；由于正式 bitstream 匿名直下返回认证页面，本 author pass 没有获得可 hash、可逐页 render 的本地 PDF，故不宣称完成图像像素级视觉复核。该下载行为只作为 source gap 登记，不请求登录或凭据。误下载的登录响应保存在任务 `scratch/workers/xia-2020-gaussian-product-sampling/diglib-login-response.html`，不作为论文证据。

## 3. 原论文的问题、假设与贡献边界

### 3.1 问题形式化

作者采用 Guo 2018 的 position-free layered BSDF：界面局部平行、层足够薄、忽略入口和出口的横向位移，因此 path state 只保留深度和方向。对固定外部方向，记某一 bounce topology 的 component 为

\[
f_{j_0\ldots j_n}(\psi_0,\psi_{n+1})
=\int\!\cdots\!\int
f_{j_0}(-\psi_0,\psi_1)\cdots f_{j_n}(-\psi_n,\psi_{n+1})
\,d\psi_1^\perp\cdots d\psi_n^\perp,
\]

其中 `\psi_0=-\omega_i`、`\psi_{n+1}=\omega_o`，且 `d\psi_k^\perp=|\cos\psi_k|d\psi_k`。这就是正文 Eq.3 的完整 integrand/measure；作者按 radiance convention 省略 transmission 的 IOR factors，并没有另列 position-free propagation factor。`n` 个内部方向能由 joint density 的 `n` 个 conditional factors 生成，但 contribution 有 `n+1` 个 scattering factors。forward sampling 会遗漏靠近固定出口的一端；从出口反向则遗漏入口端。Guo 的双向 MIS 混合这些 sequential strategies，却仍没有让单一 proposal 同时贴合所有相邻因子。[P §3.1, Eqs.3–4, Fig.3]

### 3.2 作者贡献

- 把 isotropic surface BSDF 在固定入射方向下的 outgoing slope-space slice 近似为低阶二维 Gaussian mixture，并把 mixture parameters 拟合成 material parameters 与 incident polar angle 的显式 polynomial functions；[P §§5.1,6]
- `pair-product sampling`：直接从两个相邻 BSDF slice 的 Gaussian-mixture 乘积分布采样一个内部方向，并对从 path 两端开始的 strategies 做 MIS；[P §§4.3,5.2]
- `multiple-product sampling`：在第一类三因子 component 上，用 middle BSDF 的二阶局部近似构造四维 Gaussian，联合采样两个内部方向；[P §§4.4,5.3, Appendix B]
- 仍以原始 BSDF/transport contribution 和显式 proposal PDF 加权，因此 Gaussian 只改变采样分布，不把目标 layered BSDF 替换成 Gaussian approximate value；[P §§4–7, Fig.2]
- 在 reflective/transmissive、两层/三层、spatially varying roughness/albedo/IOR 的 PBRT scenes 中与 forward 和 Guo bidirectional estimator 比较。[P §8, Figs.1,5,8–12]

“任意数量 textured layers”适用于 pair-product representation/proposal 可以逐局部 pair 重复使用，不等于 multiple-product 已能对任意长 chain 做高质量联合拟合。后者正式只演示三个 BSDF 因子、两个内部方向的 component。[P §§4.3–5.3,9]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | ordered locally planar surface layers；每个界面为可精确 `eval/sample/pdf` 的 isotropic microfacet BSDF，可有纹理化 albedo、roughness、IOR | 本文实验为 surface scattering layers；不含 volumetric phase fitting | [P §§3–6,8, Fig.11] |
| Internal evaluation query | 固定外部 `(ω_i,ω_o)` 下的 layered BSDF component/full evaluation；pair/multiple product只生成该积分的内部方向 | external endpoints固定，internal directions on sphere/hemisphere，按 reflection/refraction path topology | [P §§3–6] |
| External sample/PDF query | `sample(ω_i)` forward-trace layer path直到退出并返回 `ω_o`/weight；`pdf(ω_o|ω_i)` 对所有path types的生成概率作随机估计 | 场景 integrator 的 material-direction interface；与internal product proposal不同域 | [P §7, Eqs.22–23] |
| Product proposal coordinate | outgoing direction slope `s(ω)=(ω_x/ω_z,ω_y/ω_z)` | `R²`；reflection/transmission cases 分别拟合 | [P §§5.1,6, Eq. slope Jacobian] |
| Pair-product output | 一个内部方向及 product-mixture density；与另一端 sequential/path strategy 联合 | 二维 Gaussian-mixture sample，转回 unit direction | [P §§4.3,5.2, Figs.3–4] |
| Multiple-product output | 两个内部方向的 joint proposal | 四维 Gaussian sample，由三因子局部乘积近似形成 | [P §§4.4,5.3, Appendix B] |
| Final estimator output | 原始 layered BSDF contribution 的 Monte Carlo estimate | radiometric BSDF；Gaussian approximation 只进入 PDF/proposal | [P §§4–7, Fig.2] |
| Measure conversion | slope-space density转 unit-hemisphere solid-angle density | 正文给出 Jacobian factor `1/|ω_z|^3` | [P §5.2] |
| Validity restrictions | thin/local-flat position-free assumption；正式 Gaussian model 为 isotropic surface scattering | BSDF，不保留 BSSRDF 横向位移 | [P §§3,9] |

论文 §3 声明 integrals 与 probabilities 均采用 projected solid-angle measure `dω^⊥=|cosθ|dω`，§5.2 又把 slope density乘 `1/|ω_z|^3` 称为 unit-hemisphere probability；后者是对 solid-angle density `p_ω` 的 Jacobian。若代回前面的 projected-measure estimator，还需满足 `p_⊥=p_ω/|cosθ|`。正式 code 不可得，因此本报告保留这一 paper-notation/implementation reconciliation gap，不擅自假设代码中已补 cosine。迁移到本项目时必须显式对齐 bare linear `f` 与 solid-angle `pdf()`，不能直接复制 Eqs.3–4 的 projected-measure density。[P §§3.1,5.2; C gap; N runtime contract]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

本文没有神经网络。离线阶段为每类 microfacet BSDF 构造一个小型 parametric proposal model；runtime 仍计算原始 layered transport：

```text
offline per BSDF model family
  exact isotropic BSDF slices over (η, α, θi)
    → fit 1- or 2-component bivariate Gaussian mixture in outgoing slope
    → fit Gaussian parameters as low-order multivariate polynomials

runtime fixed (ωi, ωo), path component/topology
  evaluate fitted Gaussian slice(s) for adjacent BSDF factors
    → multiply Gaussian mixtures analytically
    → sample one slope (pair product), or two slopes jointly (multiple product)
    → convert slopes to directions and query explicit proposal PDF
    → evaluate exact BSDF/path contribution
    → MIS across forward/reverse/product strategies
    → unbiased Monte Carlo estimate when support and weighting conditions hold

separate external scene-integrator interface
  sample(ωi): Guo-style forward random walk until the layer stack exits
  pdf(ωo|ωi): independent stochastic estimate of the path-generation integral
  → Gaussian product is not itself this external proposal/PDF
```

Gaussian 乘积的核心闭式关系为：

\[
\Sigma=(\Sigma_1^{-1}+\Sigma_2^{-1})^{-1},\qquad
\mu=\Sigma(\Sigma_1^{-1}\mu_1+\Sigma_2^{-1}\mu_2),
\]

乘积还带 normalization scalar

\[
s=\mathcal N(\mu_1\mid\mu_2,\Sigma_1+\Sigma_2).
\]

对 mixture，所有 component pairs 形成新的 mixture：若两侧分别有 `n_j,n_k` 个 components，乘积最多有 `n_j n_k` 个 components。论文常用每 slice 1 或 2 个 Gaussian，因此局部 product cost 小于反复 stochastic BSDF evaluation，但没有给出硬件级指令数或 cache cost。[P §§5.1–5.2, Appendix A]

### 5.2 持久化表示

每个二维 Gaussian component 由 mean `μ=(μ_x,μ_y)` 与 precision matrix `V=Σ^{-1}` 表示。为保持 positive definite，作者用 Cholesky-like factor `V=LL^T`，拟合五个标量：

\[
\mu_x,\ \mu_y,\ L_{11},\ L_{21},\ L_{22}.
\]

这些量不是逐材质 table，而是 incident polar angle `θ`、roughness `α`、IOR `η` 的多维 polynomial。对 regular-roughness 单 Gaussian case，若各轴 degree 为 `I,J,K`，五个参数的 coefficient count 是

\[
5(K+1)(J+1)(I+1).
\]

正对角项采用 Bernstein/Bezier polynomial form 与 positive coefficients 约束，以使 precision 有效。`α<0.1` 时 mean固定为 specular slope，只剩 `L_11,L_21,L_22` 三项，因此不服从上面的五项计数。Beckmann fits 使用一个 Gaussian、二阶或三阶 polynomial，论文报告每个 case 18–135 coefficients；GGX 通常需要两个 Gaussian，但正文没有给出两分量 mixture weight `q` 的完整 polynomial/normalization 配置。reflection 与 refraction、low-roughness 与 regular-roughness 区间分别拟合。[P §6, Eqs.17–21; gap]

Fresnel 不进入 reflection slice fit，以删除 IOR 这一拟合维度；runtime proposal 因此不完全匹配 exact reflected BSDF，但 final contribution 仍求 exact Fresnel，故这属于 variance-quality approximation，不是 target-value approximation。[P §6]

### 5.3 网络逐层配置（本论文为 analytic proposal 模块）

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| BSDF slice parameter model | `η, α, θ_i` 与 reflection/refraction case | 低阶 multivariate polynomial；positive diagonal 用 Bernstein form | 无 neural activation；显式正值约束 | `μ_x,μ_y,L_11,L_21,L_22`；GGX 可两组并带 mixture weights | per-BSDF-family shared；material query supplies parameters | [P §6] |
| Gaussian slice | polynomial outputs | `V=LL^T`, `Σ=V^{-1}` | positive-definite construction | outgoing-slope density | runtime temporary | [P §§5.1,6] |
| Pair-product | 两个相邻 slice mixtures | all component-pair Gaussian products + normalization weights | mixture renormalization | one internal direction proposal/PDF | runtime per query/path component | [P §§4.3,5.2] |
| Multiple-product | end slices + middle BSDF around expansion point | middle term二阶 Taylor → 4D quadratic exponent；与 end 2D Gaussians组合 | precision eigenvalue repair：negative eigenvalues 替换为小正 `ε` | joint 4D Gaussian for two internal slopes | runtime per selected three-factor component | [P §§4.4,5.3, Appendix B] |
| Exact estimator | sampled directions + native layer BSDFs | exact factor product / proposal density；strategies MIS | paper采用 MIS，具体 heuristic 配置未充分披露 | layered BSDF sample estimate | runtime | [P §§4–7] |
| External `sample/pdf` | fixed `ω_i`；或 fixed `(ω_i,ω_o)` | forward layer-stack random walk；path-generation probability integral 的独立 stochastic estimate | 与BSDF estimate独立随机流是无偏MIS条件；正文引用Guo supplemental proof | external `ω_o`/weight；外部 directional PDF estimate | runtime；不由pair-product直接给出 | [P §7, Eqs.22–23] |

### 5.4 条件化、坐标变换与物理先验

- **isotropy**：绕 normal 的旋转对称性把 conditional outgoing distribution 降为由 incident polar angle 控制；anisotropy 会增加 orientation/roughness dimensions。[P §§5–6,9]
- **slope space**：microfacet lobes在 slope coordinates 更接近 Gaussian；转回 hemisphere 必须乘 Jacobian。[P §5.1]
- **specular anchor**：`α<0.1` 时把 mean 固定为 perfect-specular direction 的 slope `μ*`，只拟合 covariance/precision，防止 sharp lobe 的 mean regression 不稳。[P §6]
- **case split**：reflection/refraction 和 low/high roughness 分开，允许参数化在边界处不连续；作者认为 proposal discontinuity 不改变 estimator exactness。[P §6]
- **multiple-product local curvature**：middle BSDF 的 value、first derivative、second derivative 在 expansion point 上被四维 Gaussian 匹配；它不是全域 approximation，远离 expansion point 或 lobe 很宽时会恶化。[P §5.3, Appendix B]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Analytic fitting source | isotropic Beckmann/GGX reflection/refraction BSDF slices；论文分别处理 case 和 roughness regime | [P §§5.1,6] |
| Fit coordinates | regular regime 在 `(η,α,θ_i)` grid 上取 BSDF slice samples；low-roughness regime 数值计算 target covariance | [P §6] |
| Exact GT/reference | rendering experiments 用原始 layered path contribution；converged/reference images 的精确 spp 与 uncertainty 未报告 | [P §§4–8] |
| Train/validation/test split | 不适用 neural split；polynomial fitting 的 parameter grid、held-out states 与 validation protocol 未完整报告 | [P §6] |
| Directional sampling | fit grid 内把 BSDF slice 离散为近似 constant cells；runtime pair/multiple product 再从 Gaussian mixture 采 slope | [P §§4–6] |
| External BSDF sample | 给定 `ω_i`，按single-layer proposals向前随机游走，直到path退出layer stack，weight见Eq.22 | [P §7, Eq.22] |
| External PDF estimate | 对所有path types的generation-probability integral作Monte Carlo估计；须与BSDF estimate统计独立，可用较近似估计降低MIS成本 | [P §7, Eq.23] |
| Scene evaluation | reflective/transmissive double/triple layers，direct illumination 与 textured car；与 forward/Guo bidirectional 对比 | [P §8, Figs.1,5,8–12] |
| Filtering/LOD/footprint | material textures可驱动 albedo/roughness/IOR；未报告 footprint filter、mip policy 或 derivative path | [P Fig.11; gap] |
| Online/offline | Gaussian polynomial fit 是一次性 per-BSDF-model offline step；layered transport query/render online | [P §§6–8] |

论文没有披露各 parameter grid 的范围、resolution、方向 slice resolution、cell quadrature、正式 held-out set 或 fits 的 numerical error distribution。因而不能从“scene/material independent”推导出它对任意 IOR/roughness 都外推稳定；正式证据只覆盖作者拟合域和展示的 material states。[P §6; gap]

## 7. Loss、optimizer 与训练 lifecycle

| 项 | 正式配置 | locator |
|---|---|---|
| Regular-roughness target | Gaussian mixture density 对离散 exact BSDF slice 的 maximum-likelihood fit；loss 为 sample/cell 上 negative log probability | [P §6, Eq.20] |
| Low-roughness target | numerical covariance → target Cholesky precision；fit loss 为 matrix/parameter 的 Frobenius norm，mean 固定为 specular slope | [P §6, Eq.21] |
| Optimizer | sequential least squares programming（SLSQP） | [P §6] |
| Initialization | 由较少变量的 partial fits 初始化更高维 polynomial fit | [P §6] |
| LR schedule / batch / epochs | 不适用 gradient-training vocabulary；function evaluations、stopping tolerance 与 restart count 未报告 | [P §6; gap] |
| Fit model selection | Beckmann 一个 Gaussian；GGX 两个；polynomial degree/case 依据 reported fit choice；系统选择标准未完整披露 | [P §§5.1,6] |
| Fit time | Beckmann proposal model 的一次性拟合约 23 分钟；作者说明与 scene/material asset 无关 | [P §6] |
| Hardware for fit | 未报告 | [P §6; gap] |
| Random seed / uncertainty | 未报告 | [P §6; gap] |

这里的 “training” 是 parametric density fitting，不应和 neural appearance 的 optimizer lifecycle 混写。正式材料缺 official code，无法确认 SLSQP bounds、tolerances、mixture-weight parameterization、input normalization、epsilon 或 exact polynomial degrees在所有 cases 上的默认配置。

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Internal evaluation call path | fixed `(ω_i,ω_o)` layered evaluation 时，在 path-component proposal 内求 polynomial → Gaussian product → internal sample/PDF → exact contribution/MIS | [P §§4–7] |
| External scene-integrator call path | `sample(ω_i)`采用forward layer random walk；`pdf(ω_o|ω_i)`通常用独立路径随机估计，无single-layer以外closed form | [P §7, Eqs.22–23] |
| Product component count | pair product 为 `n_j n_k`；正式 slice 常为 1–2 components | [P §§5.1–5.2] |
| Multiple-product algebra | 4D mean/precision、Hessian/eigenvalue repair 与 Gaussian sample | [P §5.3, Appendix B] |
| Parameter bytes | Beckmann每 case 18–135 scalar coefficients；所有 BSDF families/cases 的总 bytes 未报告 | [P §6] |
| Precision/quantization | C++ numeric precision、coefficient packing、SIMD/GPU layout 未报告；无 quantization | [P §7; gap] |
| Hardware/backend | PBRT C++，8-core Intel i7-6700K；wall-clock render time | [P §8] |
| Single-query latency | 未报告 | [P §8; gap] |
| Render time | Fig.11 textured car：1024 spp、46 min；其余主要用 RMSE reduction 与推导的 effective time reduction 汇报 | [P Fig.11, §8] |
| Precompute included? | 一次性 23 min BSDF-family fit 不计每 scene render；scene-independent，但正式总模型生成成本不完整 | [P §§6,8] |
| Path termination/bounds | 单次 local Gaussian-product algebra操作数有限；完整 evaluation/external sample/PDF 仍沿动态layer paths，正文未报告max depth、RR或固定iteration bound | [P §§3,5–7; gap] |

论文的主要 “time reduction” 不是逐项测出的 runtime speedup。作者先测 fixed-patch RMSE ratio，再用 Monte Carlo `RMSE ∝ 1/√N` 推出平方关系：若 RMSE 降低 `r` 倍，就称 `r²` 倍 effective time reduction。这在每样本成本近似相等时可估算等误差工作量，但不能替代真实 wall-clock，尤其 multiple-product 的 4D algebra 比 forward proposal 更贵。[P §8]

## 9. 实验 protocol、baseline、指标与结果

RMSE protocol 是：对某一图像 patch，以同一方法渲染四张图，计算 per-pixel RMSE，再对 pixels 求 mean。论文未报告四次的 seeds、error bars、reference image SPP/standard error，也未说明所有方法是否严格 matched wall-clock；因此下面只复述作者正式比值，不追加跨图排名。[P §8]

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| Short double-layer component | sphere、global illumination，layered evaluation截到 `f_010` component（正文为 “up to”） | forward；Guo bidirectional | RMSE ratio；平方推导 effective time | pair vs forward `5.4×` RMSE / `29.2×` effective time；vs Guo `1.8×/3.2×`。multiple vs forward `8.2×/67.2×`；vs Guo `2.8×/7.8×` | [P Fig.5, §8] |
| Double-layer reflective full path | teapot/full layered evaluation | Guo bidirectional | RMSE/effective time | pair `1.3×/1.7×`；multiple `1.4×/2.0×` | [P Fig.8] |
| Triple-layer reflective | full path | Guo bidirectional | RMSE/effective time | pair `1.2×/1.4×`；作者指出 path 较长时收益减小 | [P Fig.9] |
| Double-layer transmissive | transmissive configurations | Guo bidirectional | RMSE/effective time | caption 报 pair `1.8–5.0×/3.2–25×`；正文报 `2.6–5.1×/6.8–26×`，两者冲突，不能合并 | [P Fig.10 caption and body] |
| Triple-layer transmissive | full paths | Guo bidirectional | RMSE/effective time | pair `1.5–2.9×/2.3–8.4×` | [P Fig.12] |
| Mixed direct-light scene | multiple materials/configurations under direct lighting | Guo bidirectional | RMSE/effective time | pair `1.6–3.3×/2.6–10.9×` | [P Fig.1] |
| Spatially varying car | procedural roughness；texture albedo/roughness | qualitative render and wall time | image at 1024 spp, 46 min | 展示 pair-product 能按 shading point 查询参数ized fits；无 matched baseline numeric table | [P Fig.11] |
| Exactness/convergence | Gaussian proposal versus approximate layered analytic result | Belcour-style approximation/context；reference | convergence/visual error | product sampling 仍向 original layered integral 收敛；Gaussian fit不替换 final integrand | [P Fig.2, §§4–8] |

这些比值不能组成统一 Pareto ranking：短 component 的最大收益不代表 full layered render；reflective、transmissive、double/triple-layer的 path distributions 不同；`effective time` 是 RMSE 平方推导，不是同一硬件上所有方法逐项 measured speedup。

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `author-negative` | forward sequential proposal | 固定两端方向时总留一个 BSDF factor 未进入 proposal，sharp products 方差高 | joint density只有与内部方向数相同的 factors | 这是 proposal mismatch，不是 target representation 容量不足 | [P §§1,3–4.2] |
| `ablation-inferior` | Guo bidirectional MIS | 能从两端覆盖，但单一 strategy仍不直接采 adjacent product；短 component 上低于 pair/multiple | product proposal更贴近 contribution | 不能外推到所有 full-path/cost regimes | [P Figs.3,5,8] |
| `author-negative` | multiple-product 延长到超过三个 BSDF factors | approximation quality随 chain length下降；正式方法只改善首个 length-three component | higher-dimensional/local Gaussian approximation error累积 | 4D成功不构成任意维 joint sampler证据 | [P §§4.4,5.3,9] |
| `author-negative` | multiple-product 在高 roughness/宽 lobe 情形 | local Taylor/Gaussian approximation更差，收益下降 | expansion point附近二阶匹配不能覆盖全域形状 | 若学 proposal，应把 support/tail 而非 peak fit 作为核心审计 | [P §§5.3,9] |
| `author-negative` | pair-product on longer paths | 仍可用于任意长度，但只相当于把未匹配 chain 缩短一个 factor，整体收益递减 | path越长，一个 local pair覆盖的比例越小 | 适合 reference/control，不是深层复杂度的完整解 | [P Figs.8–12, §§8–9] |
| `robustness-repair` | multiple-product precision 非 positive definite | 二阶近似可能产生 negative eigenvalues | 把负 eigenvalues 替换为小正 `ε` | `ε` 未报告，可能影响 broad/tail proposal 和重现性 | [P §5.3, Appendix B] |
| `robustness-repair` | 用单一 regular fit覆盖 `α<0.1` sharp regime | 作者另设 specular-anchored covariance fit | mean 的小拟合误差会比高roughness产生大得多的sample variance | 说明最终分段设计来自拟合稳定性边界；不是“任意 polynomial 即可” | [P §6] |
| `reported-conflict` | Fig.10 transmissive result | caption 与正文给出不同范围 | 未提供 correction | 保留两组数字，不代作者选择 | [P Fig.10] |

第一方材料没有报告被尝试后失败的 neural model、normalizing flow、tabulated high-dimensional product、不同 Gaussian component counts 的完整 sweep，也没有 raw failed fits。不能从最终采用 1/2 components 或 polynomial degrees 反推作者尝试过哪些替代方案。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Representation | 2D Gaussian-mixture BSDF slices；polynomial parameter functions | 未发现 | official code未发现 | 只能按 paper 重建，无法核对 coefficient arrays、normalization 与 exact case count |
| Pair-product | closed-form mixture products、two-ended strategies/MIS | 未发现 | 未发现 | proposal公式有正式证据；runtime edge cases/support tests不可审计 |
| Multiple-product | 4D second-order construction、eigenvalue repair | 未发现 | 未发现 | `ε`、eigensolver、fallback和数值精度未报告 |
| External `sample/pdf` | forward layer random walk；独立 stochastic path-probability estimate | 本文只引用Guo supplemental proof | official Xia code未发现 | 是scene-integrator接口，不是pair-product internal PDF；无法核对随机流独立性或近似模式 |
| Fitting lifecycle | SLSQP、regular/low-roughness split、约23 min Beckmann | 未发现 | 未发现 | grid/ranges/tolerances/seeds/fit hardware缺失 |
| Renderer/evaluation | PBRT C++、i7-6700K、四render patch RMSE | 未发现 | scene configs/raw renders未发现 | 无法逐场景复现 sample allocation、RR、reference SPP 与 CI |
| Result consistency | Fig.10 body/caption 数值两套 | 无 correction | 无 config/raw metric | unresolved formal conflict |

由于没有 code，本报告不会把 PBRT 的常见默认值补成论文配置；也不会借用第三方实现替作者补齐 coefficient tables 或 MIS heuristic。

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

- 正式 Gaussian representation 只覆盖 isotropic surface BSDFs。anisotropy 需要额外 direction/orientation dimensions，可能需要更多 mixture components。[P §9]
- volume scattering 可原则上扩展，但需分别拟合 phase function 的上下 hemisphere interactions并改写 tracing；本文未实现或实验。[P §9]
- multiple-product 正式只对三个相邻 BSDF factors/两个内部方向构造 4D proposal；更长 chains 的 Gaussian approximation恶化。[P §§4.4,5.3,9]
- pair-product 可沿任意长 path使用，但只减少一个未匹配 factor，path增长时相对收益变小。[P §§4.3,5.2,8–9]
- position-free thin/local-flat 假设继承自 Guo：不表达横向位移、内部几何 shadow/caustic 或 BSSRDF。[P §3 and Guo domain]
- product proposal是 approximate density；它可以保持 estimator exact，但 variance收益依赖 proposal support、PDF和MIS实现正确，不能由 Gaussian fit误差小自动保证。[P §§4–9]
- 外部 `sample/pdf` 仍是Guo-style forward walk与stochastic PDF estimate；pair/multiple-product论文贡献针对固定端点evaluation，不能据此声称得到closed-form external material-direction PDF。[P §7, Eqs.22–23]

### 12.2 未报告/材料不可得

- official code、supplemental、coefficient tables、正式 scene/config/data bundle；
- 所有 `(η,α,θ_i)` parameter ranges、grid resolution、direction slice resolution、polynomial degree/case完整表；
- SLSQP bounds、stopping tolerance、restart、seed、mixture-weight parameterization和fit hardware；
- multiple-product eigenvalue clamp `ε` 与 failure/fallback rate；
- MIS heuristic、每strategy sample count、strategy selection probability、support guards；
- path termination/RR、max depth、PBRT revision/compiler/precision；
- reference image spp、standard error、四次render seeds和confidence interval；
- `effective time` 中是否校正 per-sample cost；各实验的 measured wall-clock breakdown；
- single-query latency、coherence、state bytes、coefficient fetches、GPU feasibility；
- Fig.10 正文/caption 冲突的作者 correction；
- 本地可 hash PDF：正式匿名下载网关返回 401；当前只经公开索引读取正文；
- §3 的 projected-probability convention与§5.2的`1/|ω_z|^3` solid-angle Jacobian在实现中的cosine reconciliation。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

本文没有压缩 layered appearance。材质语义和高频结构仍由 native interface BSDFs 与完整 position-free path integral承担；Gaussian polynomial只存 proposal 所需的局部 conditional density shape。其容量分成三处：

1. exact layered reference 本身的动态 path topology；
2. 每个原生 BSDF family 的低维 parametric slice fit；
3. runtime 对相邻 factors 做 exact Gaussian-product algebra与MIS。

所以它更接近“为 reference 的积分器编译 proposal”，而不是“把材质编译为固定 evaluator”。这一区分对本项目尤其重要：它可能降低 online GT variance，却不提供 `evaluate(wo,wi)` 的静态有界神经替代。[I]

### 13.2 成功所依赖的假设

1. BSDF slice 在 slope space 可由 1–2 个 Gaussian 足够近似，且 mismatch 不造成难以处理的 tails；
2. material parameterization低维且平滑，分段 polynomial能覆盖正式 query domain；
3. 邻接两个 factors 是主要 variance bottleneck，局部 product提升足以抵消 Gaussian algebra成本；
4. exact integrand/PDF/MIS保持分离，proposal approximation不进入 target value；
5. layer paths较短，或至少单个 pair 对总 chain 的控制仍占显著比例；
6. formal isotropic microfacet families与当前 source material的 native closures一致；
7. 外部 stochastic PDF/BSDF estimates在scene MIS 中满足Guo所需独立性条件；这与internal product proposal的显式direction PDF是两件事。[P; I]

### 13.3 可迁移机制与不能迁移的部分

可迁移机制：

- 将 pair-product 作为当前 LayerStack random-walk reference 的 `optimized-code control`，比较 matched wall time、reference SE 和 bias CI；
- 把 proposal approximation 与 exact target semantics严格分层：fit只决定采样，不决定 GT；
- 对 sharp `α<0.1` states使用 analytic/specular anchor，提示 learned proposal也应显式处理 singular-limit，而不是只靠平均loss；
- 在 proposal实验中单独检查 slope↔solid-angle Jacobian、support、tail weight、normalization和MIS独立随机流；
- 用“component length/path topology”而不只用平均材质误差分桶，定位收益为何在长path衰减；
- 只在另行定义了matched external `sample/pdf` adapter后，才把 product-aware proposal纳入 learned sampler baseline；pair-product本身首先是internal evaluator proposal。

不能直接迁移：

- polynomial fits绑定 isotropic microfacet parameter axes，不覆盖项目未来的 arbitrary native graph/measured/BTF/material families；
- 4D multiple product没有任意长chain、volume或anisotropy证据；
- CPU PBRT的RMSE平方收益不能预测 GPU single-query cost；
- dynamic random walk不满足shader static-bounded contract；
- product sampling改善 fixed-direction layered integral；论文的runtime external `sample(ω_i)/pdf(ω_o|ω_i)`是单独的Guo-style随机路径接口，不是Gaussian pair-product的显式internal PDF；
- 缺 official code使其不能在未经独立重建验证前直接当作“已复现 baseline”。

### 13.4 与本项目 runtime contract 的关系

当前合同要求 deployment evaluator 的执行、状态、读取数静态有界，并使 `sample()/pdf()` 与 evaluator在同一 solid-angle measure下匹配。[N runtime contract] 本文完整方法继承 dynamic path length，因此部署分类为：

- **当前优先级**：LayerStack `reference query generator` 的 variance-reduction candidate；
- **matched sampler轨**：只能作为机制来源；必须先另行实现并验证external `sample/pdf` adapter，才构成同域analytic control；
- **不适用**：直接产品 evaluator、通用 source representation、当前 neural architecture；
- **潜在后续**：把 product-aware target/teacher用于训练静态有界 sampler，但必须另行证明 support 和bounded runtime。

项目当前顺序是先稳定 evaluator，再扩展 matched sampler和环境积分；因此本文不应成为当前 evaluator 的 kill test。它可以在reference轨提前验证，因为降低GT variance不会改变源材质语义，但任何实现都必须与原 random walk oracle做独立matched对照。[N project goal/framework; I]

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

本文不是 NVIDIA Real-Time Neural Appearance Models 的架构或训练依据，直接 fidelity 分类为 `not-applicable`。它只影响当前复现所使用的 LayerStack reference 与未来 sampler control：

| 主题 | Xia 2020 | 当前 NVIDIA functional reproduction | 分类与影响 |
|---|---|---|---|
| Evaluator | dynamic stochastic layered integral + variance-reduced proposal | fixed latent + small MLP | `not-applicable`：不能据此改变 NVIDIA evaluator topology |
| Source semantics | isotropic surface layer stack | 当前以 `1×1` LayerStack 作 source-domain adaptation | `not-applicable` 于 fidelity；可用于同一 restricted source family 的 GT control |
| Internal/external proposal | pair/multiple product采layer-integral内部方向；外部接口另用Guo forward walk/stochastic PDF | learned material-direction sampler是外部 runtime路径 | `interface-adaptation`风险：两者sampling domain不同，不能把pair-product internal PDF直接当external `pdf` |
| Output/PDF measure | projected-measure推导，slope PDF需Jacobian转solid angle | runtime要求bare `f`、solid-angle `pdf` | `not-applicable`于论文复现；但可作为measure audit，防止proposal PDF接错ABI |
| Runtime budget | full method不静态有界 | MethodBundle/Slang路径固定有界 | `not-applicable`：只进reference/control，不进deployment parity |
| Training | 无 neural training | 当前functional reproduction有其自身budget adaptation | `not-applicable`：不能解释或闭合 NVIDIA optimizer variance |

若后续发现当前 NVIDIA复现的 LayerStack GT 在 sharp multi-interface states 中reference SE过高，pair-product可以作为 `optimized-code control`；这不是修复 NVIDIA 网络，而是提高 supervision/evaluation evidence quality。若它只减少GT cost而不改变mean，则应保持模型、query recipe、state split完全冻结后再比较。[N experiment framework; I]

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：pair-product能降低当前LayerStack online reference在sharp双/三界面states的GT方差 | [P Figs.1,5,8–12] | 当前native interfaces与正式isotropic microfacet domain重叠，variance同样来自adjacent-factor mismatch | original position-free random walk vs faithful pair-product；matched wall time和exact same queries | source states、RR/depth、precision、RNG quality、exact integrand、reference target、hardware | per-query SE、bias CI、time、tail latency、path-length bins | optimized-code reference control | matched time下SE/CI无改善，或mean出现不可解释偏差/support failure |
| H2：specular-anchored proposal能改善grazing/low-roughness reference而不改变GT mean | [P §6 low-roughness split] | 当前hard states包含`α<0.1`且mean regression是proposal不稳主因 | anchored covariance fit vs unconstrained polynomial fit vs original random walk | fit data/domain、components、query states、work、MIS、seeds | peak/grazing relative error、SE、normalization、tail weights、mean bias CI | reference/proposal control | anchor无稳定收益，或case boundary产生可测bias/support hole |
| H3：product-aware结构只有经external adapter后才可能成为learned matched sampler的有效analytic baseline | [P §5.2 internal PDF；§7 separate external sample/PDF] | internal adjacent-factor结构可转成对外direction proposal，但论文没有证明这一步 | Guo-style external proposal vs product-aware external adapter vs learned sampler；三者都提供matched solid-angle `sample/pdf`并共享同一evaluator | evaluator checkpoint、source split、sample count、MIS、lights、seeds、runtime budget、external support | variance/time Pareto、sample/pdf identity、normalization、support failures、G2/G2s image CI | sampler interface-adaptation research | 无法构造合法external PDF/有support hole，或adapter相对现有analytic proposal无Pareto收益 |
| H4：multiple-product的long-chain/high-roughness收益在当前domain与matched成本下会衰减 | [P §§4.4,5.3,9] | 正式long-chain/high-roughness观察会在G2s深层states重现 | pair-product vs 4D multiple-product；按path length/roughness分桶 | same exact integrand、work cap、fit domain、MIS、seeds | SE/time、fallback rate、negative-eigenvalue rate、tail/support | capacity diagnostic/reference only | multiple-product在长stack和严格成本下持续稳定胜出且无数值/支持问题 |
| H5：对reference estimator按path topology做curriculum/query allocation可减少teacher噪声 | [P短component与full-path收益差异] | online GT预算可按难度分配，且topology proxy可在线获得 | uniform evaluation_samples vs topology/roughness-adaptive allocation；same total reference work | model、optimizer、training steps、source/query distribution、seed set | target SE、training seed variance、G1/G2/G2s、wall time | query recipe | adaptive allocation未降低target/model variance，或改变有效训练分布导致泛化退化 |

H1/H2/H5属于reference/query轨；H3只在evaluator稳定且external adapter定义完成后进入matched sampler轨；H4是对formal退化观察的可证伪迁移，不是hard gate。所有假设都保留 original random walk mean作为独立oracle，不把product fit自身当GT，也不把任何假设写成已授权run。

## 16. 证据索引

### `P` Main paper

- Abstract、§§1–3、Fig.3：问题、position-free component、`n` directions versus `n+1` factors 与既有proposal缺口。
- §§5.1、6：isotropic BSDF slice、slope coordinates、Gaussian mixture、precision Cholesky、polynomial parameterization、regular/low-roughness fits、SLSQP和约23 min fit。
- §§4.3、5.2、Figs.3–5：pair-product Gaussian-mixture algebra、two-ended strategies、MIS 与短component结果。
- §§4.4、5.3、Appendix B：multiple-product 4D construction、middle-factor二阶匹配、negative-eigenvalue repair和适用范围。
- §8、Figs.1,5,8–12：PBRT/i7-6700K实验、四render patch RMSE、RMSE平方effective-time定义、反射/透射/双层/三层与textured car。
- §7、Eqs.22–23：external BSDF forward sample、stochastic PDF estimate及其与BSDF estimate独立的条件；与internal product proposal分域。
- §9：isotropy、volume extension、longer-chain/high-roughness与pair/multiple产品边界。
- Fig.2：proposal approximation与最终exact convergence边界。
- Fig.10 caption/body：transmissive数字冲突，原样保留。

公开索引中已读取 main paper 16/16 页的正文、公式、table/caption文本；因正式bitstream匿名直接下载返回401，本author pass未完成本地render的图像像素级复核，也没有可登记的PDF SHA-256。

### `S` Supplemental

- 正式Eurographics条目、Cornell author listing和论文可见链接未发现独立supplemental/appendix package；因此没有用二手材料补齐fit grid、configs或raw metrics。

### `C` Official code/config

- 正式条目与作者组入口未发现official code/config/data。第三方/student实现未进入事实层。

### `A` Author/publisher material

- Eurographics Digital Library正式条目：卷期、页码、DOI和PDF入口。
- Wiley DOI page：online-first日期与2020卷期边界。
- Cornell Graphics and Vision Group：作者列表、摘要与formal paper入口。
- 第一方公开入口未发现correction；这不能证明不存在未索引说明。

### `N` NeuralShading evidence

- `docs/contracts/scattering_backend.md`：bare linear `f`、solid-angle `pdf`、matched `sample/evaluate/pdf`与static-bounded deployment contract。
- `docs/research/experiment_framework.md`：source/query冻结、stochastic reference repeated evaluation、matched control和CI要求。
- `.trellis/spec/project/method-constraints.md`：新方法可超软线诊断表达力，但必须保留硬件部署可能性；reference control与产品candidate分轨。
- NVIDIA correspondence：当前functional reproduction、LayerStack source adaptation、runtime adapter和MethodBundle/Slang边界；只用于§14。

### `I` Derived/transfer notes

- “为reference积分器编译proposal”是本报告定位，不是作者术语。
- `effective time`不可等同measured wall-clock、internal layer-integral proposal不可直接等同external material-direction sampler、完整runtime不静态有界，均由正式方法与本项目合同对照得出。
- `p_⊥=p_ω/|cosθ|` 是从正文两种measure声明推导的迁移审计关系；正式code不可得，不能据此断言作者实现错误。
- 所有迁移假设都要求original random walk独立oracle、matched work和bootstrap/CI，不把Gaussian fit替换成GT。

### 建议提升的 load-bearing 论文/方法

- **Guo 2018 position-free layered BSDFs**：已完成 evidence review；本文的方法、无偏性和外部 sample/PDF 边界依赖它。
- **Belcour 2018 atomic decomposition/statistical operators**：已完成 evidence review；作为 fast biased layered-material control，帮助区分“proposal exact convergence”和“value approximation”。
- **Neural Layered BRDFs / MetaLayer / BSDF Importance Baking**：已各自报告；需在 sampling synthesis 中明确其 reference、learned evaluator 与 external proposal 是否真的继承本文机制。
- **支持 product-aware learned importance sampling 的后续一方论文**：仅在 synthesis 出现“analytic pair product不足以作为learned sampler baseline”的明确触发后提升，不凭引用列表扩写。

## Evidence review

```text
author_worker: /root
reviewer: /root/belcour2018_review
reviewed_at: 2026-08-29
sources_rechecked:
  - official Eurographics record and public 16/16-page paper text/formula/table/caption extraction independently rechecked
  - Wiley DOI record and Cornell author/group listings independently rechecked; no code/supplemental/correction link found
findings_closed:
  - exact Eq.3 signs, endpoint convention and projected-solid-angle measure restored
  - pair-product and multiple-product problem decomposition
  - Gaussian slice representation and fitting lifecycle reported by the paper
  - internal evaluator proposal separated from Guo-style external sample/stochastic-PDF interface
  - formal experiment ratios and effective-time definition
  - long-path, high-roughness, isotropy and volume-extension boundaries
  - internal Fig.10 body/caption conflict preserved
remaining_evidence_gaps:
  - direct official bitstream download returns 401/login page; no local PDF hash or full rendered-page visual audit
  - no official supplemental, code, coefficients, configs, raw images or correction found
  - fitting grids/ranges, optimizer tolerances/seeds and multiple-product epsilon are unreported
  - MIS/sample allocation, PBRT revision, termination, reference SPP/SE and per-scene measured timings are unreported
  - projected-probability convention versus the solid-angle slope Jacobian cannot be reconciled against unavailable code
review_status: evidence-reviewed
```

### 完成检查

- [x] main paper 16/16页公开正文、公式、table/caption抽取已完整复核；正式PDF因401/403无法本地hash或逐页视觉核对，该gap已显式保留；
- [x] supplemental/appendix/勘误的可用性已检查；
- [x] official code/config/data 的可用性与 commit 已检查；
- [x] architecture、training、runtime 和主要结果均有 locator；
- [x] 失败尝试与较差消融正确分类；
- [x] paper/code gap 和“未报告”保留；
- [x] `I` 分析晚于事实层，没有改写作者结论；
- [x] NVIDIA 影响引用真实 `N` 证据；
- [x] 假设包含 matched control、部署类别和证伪条件；
- [x] 独立 evidence review 已完成。
