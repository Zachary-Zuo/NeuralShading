---
paper_id: "bai-2023-bsdf-importance-baking"
title: "BSDF Importance Baking: A Lightweight Neural Solution to Importance Sampling General Parametric BSDFs"
authors: "Yaoyi Bai, Songyin Wu, Zheng Zeng, Beibei Wang, Ling-Qi Yan"
year: "2023"
venue: "Technical Report, arXiv:2210.13681v3"
doi: "10.48550/arXiv.2210.13681"
report_status: "evidence-reviewed"
main_source: "https://arxiv.org/abs/2210.13681"
supplemental_status: "unavailable"
official_code_status: "unavailable"
official_code_commit: "not-applicable"
author_worker: "/root"
reviewer: "/root/nbrdf2021"
last_verified: "2026-08-29"
---

# BSDF Importance Baking: A Lightweight Neural Solution to Importance Sampling General Parametric BSDFs

## 1. 研究对象与报告边界

本文研究的主要对象不是 neural evaluator，而是**一般参数化 BSDF 的 neural importance sampler**：先为固定材质参数与固定出射方向的二维 BSDF slice 离线求一张从均匀二维随机数到入射方向的 transport map，再用小型 MLP 压缩这些 map。为了接入 MIS，作者另训可选的 BSDF evaluation network 与 PDF query network。[P §1, §3.2, §4.2]

本报告严格以 2023-02-16 的 `arXiv:2210.13681v3` 为研究对象。2025-10-28，Wiley 又发布了标题改为 **BRDF Importance Baking: A Lightweight Neural Solution to Importance Sampling General Parametric BRDFs** 的 Computer Graphics Forum Version of Record（VOR，DOI `10.1111/cgf.70286`）。其 landing page/Crossref 可确认 BRDF 用词、作者顺序、47 条参考文献以及一份 53.6 MB supporting information，但当前无法取得 VOR 正文与附件；本报告因此不会把 2023 配置冒充 2025 正式版配置，也不会由 metadata 猜测修订细节。[A-VOR landing page; A-Crossref]

论文覆盖三种参数化反射材质族：[P §3.1, Fig.2]

1. anisotropic multiple-bounce GGX conductor；
2. top rough dielectric + homogeneous medium + bottom diffuse 的两层 position-free layered BSDF；
3. 只改变 metallic 与 roughness 的 Disney Principled BSDF。

它不覆盖空间变化纹理、footprint/LOD、BTDF、源材质 compiler 或跨材质族共享网络。作者声称方法对更多层/更多 BSDF 类型没有结构性限制，但正式实验只训练上述三个固定参数域；本报告不会把该声明扩成已经验证的 universal material sampler。[P §3.1, §6, §7]

分类为 `local-material`，主要价值在 `sample()/pdf()` 路线，而不是替代本项目 `evaluate(wo,wi)` 主线。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---:|---|---|
| Main paper `P` | [arXiv v3](https://arxiv.org/abs/2210.13681)，2023-02-16，14 页 | 2026-08-29 | SHA-256 `6E915EA80DF6B4D42C2B46EA295841B3DCD8C7A5168314EA6CC69A2BA1418A44` | 本报告方法、训练和实验事实的主要来源 |
| Author-hosted paper `A-P` | [UCSB 作者 PDF](https://sites.cs.ucsb.edu/~lingqi/publications/paper_impbaking.pdf) | 2026-08-29 | SHA-256 同 `P` | 与 arXiv v3 逐字节相同，不是独立版本 |
| Author publication entry `A` | [Ling-Qi Yan publication list](https://lingqiyan.github.io/) | 2026-08-29 | 固定页面；未持久化 | 将该版本列为 “Technical Report (arXiv:2210.13681), Feb 2023”；没有可用 project/code/video 链接 |
| 2025 VOR `A-VOR` | [Wiley/DOI](https://doi.org/10.1111/cgf.70286)，first published 2025-10-28 | 2026-08-29 | 未取得 PDF | 只用于登记后续正式版本身份、abstract、作者顺序与 supporting-information 存在性；不能用于恢复内部配置 |
| VOR metadata `A-Crossref` | Crossref work `10.1111/cgf.70286` | 2026-08-29 | API response 未持久化 | 确认 VOR 日期、e70286、47 条参考文献与 VOR PDF locator |
| VOR supplemental `S-VOR` | Wiley `cgf70286-sup-0001-SuppMat.pdf`，页面标称 53.6 MB | 2026-08-29 | 未取得 | publisher 页面列出，但 direct download 被 403 拒绝；不把附件内容写入报告 |
| Official code/config/data `C` | 作者主页、论文 locator、GitHub title/author 检索 | 2026-08-29 | not-applicable | 未找到官方实现、训练配置、checkpoint 或论文所称公开 importance-map database |
| NeuralShading evidence `N` | [scattering backend contract](../../../../../docs/contracts/scattering_backend.md)、[compiler contract](../../../../../docs/realtime_material_compilation.md)、[current NVIDIA formal config](../../../../../configs/learning/nvidia-rta2024-materialx-formal.json)、[method](../../../../../src/ncls/learning/methods/nvidia.py) 与 [model](../../../../../src/ncls/learning/models/nvidia_neural_appearance.py) | 2026-08-29 | repo-local；current correspondence `nvidia-rta2024-functional-f@2` | 只用于 §13–14，不回填论文事实；不再用归档 correspondence 代表当前实现 |

### 2.1 来源可得性与版本结论

- `P` 已全文读取；Eq.(1)–(8)、Table 1、Fig.1–18、图注、脚注 1 与关键结果页均以 PDF render 视觉核对。
- 2023 技术报告没有 supplemental、勘误或 code locator。作者把 importance-map database 列为贡献，但正文没有下载地址；截至本次审计未能找到第一方入口。[P §1, p.2]
- 2025 VOR 的 landing page 与 Crossref 可读，全文 PDF、full HTML 和 supplemental 下载均被 Wiley 返回 403。这里的缺口是“后续正式版本不可完整审计”，不是用 2023 预印本猜正式版。
- VOR abstract 将 `importance map` 攵称 `importance warping map (IWM)`，把对象收窄为 BRDF，并将结论从“perfect importance sampling”改写为 “high-quality approximations”。这些文字变化提示正式版可能实质修订，因此在获得 VOR 前不能把本报告升级为 VOR correspondence。[A-VOR abstract]

## 3. 原论文的问题、假设与贡献边界

参数化 BSDF 的 evaluator 可能昂贵，且其完整 `f_s(ω_i,ω_o)|n·ω_i|` 通常没有与之完全匹配的解析 sampler。只采 NDF/VNDF 会漏掉 Fresnel 或其他 lobe；multiple scattering 与 layered BSDF 往往要 random walk；normalizing flow 又可能过重。论文目标是把“如何采样”提前烘焙成可压缩的二维映射，以降低 runtime variance 与 reference sampling 成本。[P §1–2]

作者的四项贡献是：[P §1, pp.1–2]

1. 面向一般参数化 BSDF 的 importance-sampling 方案；
2. 用离散 optimal transport 在均匀点集与目标 BSDF 分布点集之间求一一映射；
3. 用 lightweight MLP point-query 该映射，并可选地增加 evaluator/PDF networks 形成 MIS 接口；
4. 声称公开 multiple-bounce、layered 与 Disney 的 importance-map database。

这里的“perfect”必须按论文脚注收窄：RGB BSDF 只有一个 scalar PDF，无法同时逐通道匹配；作者选择由 RGB luminance 得到的 grayscale target PDF，因此它至多是**对所选 scalar target 的离线离散 transport 最优**。经过有限分辨率、OT solver 和神经压缩后，正文自己也展示了 bias。[P footnote 1, p.3; §7, Fig.18]

网络不是让 normalizing flow 自行学习 density/Jacobian，而是回归已经求出的 transport samples。换言之，容量的上游主要来自昂贵 reference slice 与离线 OT，MLP 只是压缩器。[P §4.1–4.2]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | 三个固定 parametric family 的原生参数 | multiple-bounce：`αx,αy,R0_RGB`；layered：`η,α,σT,albedo_RGB` 加固定 diffuse substrate；Disney：`metallic,roughness`，其余固定 | [P §3.1, Fig.2; Table 1] |
| Sampling query | `I(ε,ωo,ξ0,ξ1)` | encoded material parameters + 2D outgoing direction + two uniform random numbers | [P Eq.(6), Fig.8] |
| Sampling output | incident direction coordinate与 RGB sampling weight | direction head 为 2 scalars；weight head 为 3 scalars，`sw=f_s|n·ω_i|/p(ω_i)` | [P Eq.(6), Fig.8; §5.3] |
| Evaluation query | `E(ε,ωo,ωi)` | encoded parameters + two 2D directions | [P Eq.(7), Fig.10] |
| Evaluation output | cosine-weighted RGB BSDF | `f_s(ωo,ωi)⟨ωi,n⟩∈R³`，不是 bare `f` | [P Eq.(7)] |
| PDF query | `P(ε,ωo,ωi)` | 与 evaluator 同输入 | [P Eq.(8), Fig.10] |
| PDF-network output | 训练 target 是对 grayscale `f_s cos` 归一化得到的 solid-angle density；runtime 网络值只被作者用于 MIS weighting query | `FC+exp` 的 positive scalar；正文不施加归一化或与 sampling map 同 proposal 的约束，故它不是经认证的 sampling PDF | [P §4.2–4.3, Eq.(8), Fig.10–11] |
| Direction parameterization | hemisphere 先重参数化到 unit disk；网络输入/输出都以两个标量表示方向 | 精确 disk↔hemisphere 公式未报告 | [P §5.1; Fig.8] |
| Validity/domain restrictions | 表面反射上半球；固定 family/参数域 | 参数精确训练上下界未报告；不含 transmission | [P §3.1, §5.1, §6] |

论文把 `ω_i` 称 incident/light direction、`ω_o` 称 camera/outgoing direction。[P Fig.2 caption] 正文没有说明切线 frame、anisotropic axis、disk parameterization 的极点/边界处理或 reciprocity convention，这些均不能从常见 Mitsuba 实现补写。

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

离线阶段：[P §4.1, §5.1]

```text
uniformly sampled material parameters ε and grazing-emphasized ωo
  → reference evaluates one 128×128 cosine-weighted RGB BSDF slice
  → luminance + solid-angle normalization gives scalar target PDF
  → row-column sampling discretizes target PDF into equal-weight point set β
  → regular unit-square grid is source point set α
  → GeomLoss initialization + SOT refinement solves a one-to-one permutation ψ
  → each source pixel stores mapped target coordinate (u,v)
  → reference f·cos / true target PDF gives RGB sampling-weight target
  → MLP compresses point queries over (ε,ωo,ξ)
```

作者把规则网格写成等权 Dirac 点集 `α=(1/n)Σ_i δ_(ξ_i^0,ξ_i^1)`，把 row-column sampling 得到的目标点写成 `β=(1/n)Σ_j δ_(u_j,v_j)`；Eq.(5) 再求 `ψ*=argmin_ψ Σ_i ||α(i)-β(ψ(i))||`。因此 row-column sampling 这里只负责把 continuous density 离散成 target points；真正从 source point 到 target point 的一一 permutation 由 OT 求，不是 row-column inverse transform 本身。[P §4.1, Eq.(4)–(5)]

runtime 有三条互相独立的路径：[P §4.2, §5.3]

```text
sample(ε,ωo,ξ)   → sampling MLP → (ωi, precomputed RGB weight)
evaluate(ε,ωo,ωi)→ evaluation MLP → RGB f·cos
pdf(ε,ωo,ωi)     → PDF MLP → scalar MIS value
```

`sample()` 不再调用 evaluator/PDF network，而直接返回训练好的 RGB throughput weight。作者把这作为降低三次 inference 与避免三个网络不一致 bias 的优化。[P §5.3]

这里必须区分三个量：[P §4.2–4.3, §5.1–5.3]

1. 构造 OT teacher 与 RGB weight target 时使用的真实 scalar target density `p_target`；
2. sampling MLP 直接回归的 RGB `sw=f_s cos/p_target`，runtime 不由 evaluator/PDF network 重算；
3. 独立 PDF MLP 给 MIS heuristic 的 positive query value。它以 density 为监督，但作者允许 MIS weighting query 不归一；这不把它变成 sampling map 的可核查 density。

### 5.2 持久化表示

- 每个 parametric family 保存一套 sampling network；复杂 layered family 使用更宽网络。[P Fig.8]
- evaluator 与 PDF 各是独立网络，不共享 decoder；正文称分别对“each material”训练，但上下文实际在三个 material family 间使用网络，措辞没有进一步澄清 checkpoint 粒度。[P §5.2]
- 完整离线 importance maps/BSDF slices 用作训练数据，不是 runtime texture；runtime 持久化的是 binary network weights。[P §4.2, §5.2–5.3]
- 参数量、binary bytes、weight precision、activation precision 与 layout 未报告；没有 quantization。

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Sampling MLP | encoded `(ε,ωo,ξ0,ξ1)` | 6 个等宽 hidden FC；multiple-bounce/Disney 宽 64，layered 宽 128；末端分成两个 head | hidden ReLU；direction head `FC+Sigmoid`；weight head `FC+exp` | 2D incident coordinate + RGB sampling weight | per family；是否进一步 per material 未报告 | [P Fig.8 caption and diagram] |
| Evaluation MLP | encoded `(ε,ωo,ωi)` | Fig.10 shared diagram：宽 `64→32→32→32→32→32`，再接 output FC | hidden ReLU；output `exp` | RGB `f_s cos` | 独立网络 | [P Fig.10 diagram; Eq.(7)] |
| PDF MLP | 与 evaluation 相同 | 同一 topology，独立参数 | hidden ReLU；output `exp`；无 sum-to-one constraint | scalar MIS PDF query | 独立网络 | [P Fig.10; §4.3] |

Figure 10 是 evaluator/PDF 层数的唯一明确来源；正文没有逐层表或参数总数。上表保留图示，不用推算值替代作者未报告的 parameter count。

### 5.4 输入 encoding

`freq(x)` 对每个 scalar 使用 `k` 个频率 `2^0…2^(k-1)` 的 sin/cos，共 `2k` channels；`ob(x)` 是 `l` bins 的 one-blob Gaussian encoding；`id` 是 identity；`nl` 把范围归一化到 `[0,1]`。[P §5.2, Table 1]

| Family | 参数 encoding | 方向/随机数 encoding | 表中可恢复的 encoded 维度 |
|---|---|---|---:|
| Multiple-bounce | `freq(αx,αy,k=4)`；`freq(R0_RGB,k=4)` | `freq(ωo,k=12)`；按路径再输入 `freq(ωi,k=12)` 或 `freq(ξ,k=12)` | 136 |
| Layered | `ob(η,4)`；`ob(nl(1-exp(-α)),4)`；`ob(σT,5)`；`id(albedo_RGB)` | `ob(ωo,12)`；按路径再输入 `ob(ωi,12)` 或 `ob(ξ,12)` | 64 |
| Disney | `ob(metallic,8)`；`ob(nl(1-exp(-α)),8)` | `ob(ωo,12)`；按路径再输入 `ob(ωi,12)` 或 `ob(ξ,12)` | 64 |

维度是按 Table 1 直接求和；`nl` 的原始上下界、one-blob Gaussian width/center 与 frequency phase convention 未报告。[P Table 1]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset/source assets/scenes | 正文写对“each material”生成 32,768 个 BSDF slices、每 slice `128×128`，即每个所称 material 有 `536,870,912` 个 pixel queries；上下文把三个 parametric family 当作三个 material 对象，但没有进一步定义 checkpoint 粒度 | [P §5.1；query 数为直接算术] |
| Multiple-bounce reference | Heitz et al. 2016 multiple-scattering Smith microfacet random walk，GGX conductors | [P §3.1, §6.1] |
| Layered reference | Guo et al. 2018 position-free layered BSDF；正式训练/实验只展示两层 rough dielectric + medium + diffuse | [P §3.1, §6.2] |
| Disney reference | Mitsuba 中完整 Disney Principled evaluator/sampler；只改变 metallic/roughness | [P §6.3] |
| Train/validation/test split | 未报告；没有 material holdout、parameter holdout 或 independent test slice 数量 | [P §5–6] |
| Material sampling | 参数在各自空间均匀采样；roughness 改在 squared space 采样 | [P §5.1] |
| View sampling | 在 polar angle 的平方根空间均匀采样，以增加 grazing cases | [P §5.1] |
| Slice target | 每 pixel 存 `f_s(ωo,ωi)⟨ωi,n⟩` RGB | [P §5.1] |
| Scalar PDF target | 对 RGB slice 取 luminance，再按 summed solid-angle measure 归一化 | [P §5.1] |
| Sampling target | hemisphere PDF 重参数化到 unit disk；row-column samples 构成 OT target points；另存 RGB `f_s cos/PDF` | [P §5.1] |
| OT solver | GeomLoss 给初值，SOT 继续优化 | [P §5.1] |
| Online/offline generation | 全部离线；数据机为 AMD Threadripper PRO 3995WX 64-core CPU | [P §5.1] |

作者解释不需要额外 Jacobian：BSDF 与 PDF 都按 solid angle 计算，unit-disk 操作只用于构造 sample locations，没有把训练权重改写为 parameter-space density。[P §5.1] 但精确 disk map、pixel solid angle、zero-density/support 处理与 OT point count 没有披露，因此这句话不足以复现实装。

## 7. Loss、optimizer 与训练 lifecycle

| 项 | 正式配置 | locator |
|---|---|---|
| Evaluator target/output transform | network 正输出；loss 比较 cosine-weighted RGB `f_s cos` | [P Eq.(7); §5.2] |
| Evaluator loss | `L_eval=||pred-gt||₁ / (sg(||pred||₁)+sg(||gt||₁)+ε)`，`ε=0.01`；分母 stop-gradient | [P §5.2] |
| PDF loss | `L_pdf=||log(1+pred)-log(1+gt)||₁` | [P §5.2] |
| Sampling loss | `||ω̂-ω_gt||₁ + λ||log(1+sŵ)-log(1+sw_gt)||₁`，`λ=0.4`，`sw` 为 RGB | [P §5.2] |
| Sampling optimizer | Adam default parameters | [P §5.2] |
| Evaluator/PDF optimizer | Ranger；具体组成/default version 未冻结 | [P §5.2] |
| LR schedule | initial LR `1e-4`；sampling 用 cosine annealing；evaluator/PDF 是否同 scheduler 的文字不明确 | [P §5.2] |
| Batch | `1,048,576` | [P §5.2] |
| Epochs | 每个网络 500 epochs | [P §5.2] |
| Initialization/seed/model selection | 未报告 | [P §5.2] |
| Hardware | Intel i9-7960X 16-core + NVIDIA RTX 3090 | [P §5.2] |
| Training time | sampling network 约 48 h；evaluation 与 PDF network 各约 12 h；正文写“on each material”，checkpoint 粒度未再定义 | [P §5.2] |

论文没有给 epoch 内 query 数、slice batching、validation、early stopping、best/last checkpoint、mixed precision、gradient clipping、Ranger 超参数、cosine end LR 或数据生成总时间。极大的 batch 数也没有说明是一次 GPU resident batch、gradient accumulation 还是 epoch 内总 queries。

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path | Mitsuba 只替换 BSDF class；sample path 一次 sampling MLP，直接返回 direction+weight；evaluate/pdf 分别调用自己的 MLP | [P §5.3] |
| Backend | C++ + Eigen dense matrix multiplication；只测 CPU inference | [P §5.3, §6] |
| Rendering hardware | Intel i9-9900K 8-core CPU | [P §6] |
| Parameter count/MAC/FLOP | 未报告 | [P Fig.8, Fig.10] |
| Shared/per-asset/state bytes | 未报告；只称 weights 存 binary files | [P §5.2] |
| Texture/feature fetches | 无 runtime texture fetch；输入是参数与方向 | [P §5.3] |
| Precision/quantization | 未报告 | [P §5.2–5.3] |
| Single-query latency | 未报告；只给整图分钟/秒 | [P §6, Fig.1, 13–17] |
| Amortization | offline BSDF/OT generation与 12–48 h training 不计入 rendering time | [P §5–6] |

作者明确承认 inline CPU inference 未优化，并提出 GPU/TensorRT 或工程优化可能降低成本；这只是 future direction，不是已经测量的 GPU runtime。[P §7]

## 9. 实验 protocol、baseline、指标与结果

所有 rendering time 来自 i9-9900K CPU。指标为 relMSE，但论文明确写道不同方法可能收敛到不同 ground truth，因此“use their own converged results as the ground truth (GT)”。这意味着表中每个 relMSE 是该方法相对其自身收敛图像的 convergence error；不同方法未必共享同一权威 reference，不应把跨方法数值或比值解释成统一 GT 下的严格 error ratio。[P §6]

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| Conductor kitchen shelf | full three-network MIS；256 spp；area light；四物体粗糙度 `0.2,0.6,0.03,(αx=.08,αy=.3)` | Heitz et al. 2016 | crop relMSE + visual；整图 time 未在 Fig.1 标出 | 三个 crop：Heitz `5.4e-3/2.7e-3/1.5e-2`，ours `3.1e-3/1.5e-3/7.9e-3`；正文称 ours 约 2× 慢 | [P Fig.1 top; §6.1] |
| Vase equal quality / RealNVP | direct light、environment、`αx=.3,αy=.1`；sampling only | Heitz 2016；Xie 2019 RealNVP width 32、相近参数量和相同训练时间 | time、spp、relMSE | Heitz 2048 spp：3.99 min / `8.6e-3`；ours 256 spp：46.74 s / `8.6e-3`；Xie 256 spp：1.37 min / `0.13`，4096 spp：21.57 min / `8.2e-2` | [P Fig.13; §6.1] |
| Low-roughness teapot | `α=.01`，256 spp，sampling only | Heitz 2016 | visual + time | Heitz 48.76 s；ours 1.27 min；authors report correct highlights/lower variance with 0.5× extra cost | [P Fig.14; §6.1] |
| Ginkgo full MIS | 256 spp，roughness `.3` | Heitz；ours sampler+GT eval/PDF；ours three networks | relMSE + time | Heitz 41.12 s / `9.5e-3`；ours sample+GT 1.11 min / `4.9e-3`；full neural 1.24 min / `3.5e-3` | [P Fig.15; §6.1] |
| Layered shoes | 256 spp、environment、sampling only；top `α=.08,η=1.5`，medium `σT=.8` | Guo 2018 | four crop relMSE + time | Guo 8.4 min vs ours 5.12 min；crop pairs `1.6e-2→4.9e-3`,`4.2e-3→1.3e-3`,`6.6e-3→1.5e-4`,`3.7e-3→8.4e-4` | [P Fig.16; §6.2] |
| Layered kitchen shelf | full three-network MIS、direct+indirect、equal time | Guo 2018 | spp + crop relMSE | Guo 850 spp vs ours 1024 spp；三个 crop `1.7e-3→9.1e-4`,`1.2e-3→5.3e-4`,`4.5e-4→1.3e-4` | [P Fig.1 bottom; §6.2] |
| Disney coffee cups | environment、512 spp、sampling only；`m∈{.1,.3,.5,.7,.9}`，`α∈{.15,.35,.65,.85}` | Mitsuba Disney sampler | four crop relMSE + time | Disney 2.03 min vs ours 3.10 min；crop pairs `.061→.057`,`.015→.014`,`.064→.044`,`.0095→.0062` | [P Fig.17; §6.3] |
| Arbitrary MIS query values | diffuse ball，16,384 spp | 人工提供的 MIS query values：常数 `-10`、cosine-weighted PDF 的十次方、cosine-weighted PDF | converged image identity | 三图在展示中一致；`-10` 不可能来自 Fig.10 的 `exp` 网络，它只演示 generalized MIS weighting query 可不是 density，不是 sample path 可以谎报 proposal PDF | [P Fig.11; §4.3] |

成功结论按论文证据可收窄为：在这些 family、场景、CPU implementation 和作者选择的 own-converged-GT protocol 下，learned OT map 往往以更少 spp 得到更低的**各方法自身 convergence error**；昂贵 Guo random walk 的时间也可被 MLP 替代。[P §6] 论文没有多 seed、置信区间、per-query latency、相同 GT 的统一误差表或 unbiasedness 数值检验。

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `author-negative` | marginalized inverse transform map | bilinear interpolation 后出现 filament connections | map 对相邻 uniform samples 不连续 | 这是“先固定可压缩 transport target”的直接证据 | [P §3.3, Fig.4, Fig.6] |
| `author-negative` | hierarchical sample warping map | binning 中出现 grid artifacts | hierarchy 的 quadrant choices 破坏连续性 | 失败针对其作为可插值/可压缩 map，不表示该 sampler 本身无效 | [P §3.3, Fig.5–6] |
| `author-negative` | SOT 单独优化 | 即便数万 iterations，低粗糙度 slice 仍有 crevices；网络学到后产生 dark regions | SOT 快但初值差；改为慢 GeomLoss 初始化后再 SOT | solver artifact 会被小网络忠实放大，是 teacher-quality 风险 | [P §5.1] |
| `ablation-inferior` | Xie 2019 RealNVP，hidden 32、相近参数量/相同训练时间 | 256 spp relMSE `.13`；4096 spp 仍 `.082`，且慢 | flow 同时学习 sampling 与 density，结构重且不适合高维条件 | 单一场景/作者复现，不能推广成所有 flow sampler 都失败 | [P Fig.13; §6.1] |
| `ablation-inferior` | fitted Blinn–Phong proposal | anisotropic multiple-bounce slice 的 binning match 较差 | 单 analytic lobe 的 shape prior 不足 | 支持把 analytic proposal 作为 matched control，而不是预设必胜/必败 | [P Fig.12; §4.3] |
| `known-limitation` | highly specular grazing cases | teapot/cup 边缘出现黑边 | learned sampling-weight prediction 引入 bias；white-furnace 会立即暴露 | 这是 transport correctness failure，不只是轻微视觉 artifact | [P §7, Fig.18] |
| `known-limitation` | CPU Eigen inference | 部分场景比 analytic baseline 慢 1.5–2× | integration 未优化 | 未有 GPU/shader 测量，不能声称达到 realtime | [P §6.1, §6.3, §7] |

已获得来源没有报告：不同 OT resolution、network depth/width、encoding、weight head、loss 或训练数据量的正式 ablation；也没有失败 seed 或 training instability 分析。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Architecture | Fig.8/10 可视觉恢复 topology | 2023 无 | 无 | 可恢复层宽/activation，无法恢复初始化、precision、layout、exact disk map |
| Data/query | 正文所称 each material 为 32,768×128² pixel queries、参数/view recipe、GeomLoss initialization+SOT refinement | 2023 无 | database 未找到 | “each material”的 checkpoint 粒度未进一步定义；论文称 public database，但没有可访问 locator |
| Loss/training | 三个 loss、optimizer 名、batch、LR、500 epochs、训练时间 | 2023 无 | 无 | Ranger/default、scheduler end、seed/split/model selection 缺失 |
| Runtime/export | C++/Eigen/Mitsuba BSDF-class replacement | 2023 无 | 无 | 无 binary format、weights、single-query benchmark 或 GPU implementation |
| Assets/evaluation | Fig.1, 9, 11–18 | 2023 无 | scenes/raw images/raw metrics 无 | 无法独立重算 relMSE 或验证 own-GT protocol |
| Later VOR | 2025 CGF title/abstract 已变化 | Wiley 列出 53.6 MB 附件但 403 | 未找到 | VOR 不能由 arXiv v3 静默替代 |

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. **Bias**：sampling MLP 回归 sampling weight，grazing/highly specular 处黑边可见；white-furnace test 会暴露。[P §7, Fig.18]
2. **Runtime overhead**：CPU network inference 没有优化，虽然网络“小”，对 cheap analytic baseline 的整图时间仍约为 1.5–2×；这不是 single-query benchmark。[P §6–7]
3. **RGB scalarization**：一个 scalar PDF 不能完美匹配 RGB 三通道，所谓 perfect 只相对选定 grayscale density。[P footnote 1]
4. **Measured/non-parametric extrapolation**：作者只讨论未来可能推广到 BTF/light field/NeRF，没有实验。[P §7–8]

### 12.2 未报告/材料不可得

- 2025 VOR 与 supplemental 的完整内容；
- 论文所称 importance-map database、训练代码、Mitsuba plugin、weights 与 scenes；
- 三个 material family 的精确参数训练范围与 fixed Disney parameters；
- train/validation/test split、seed、epoch query count、checkpoint selection；
- exact disk mapping、solid-angle pixel weighting、zero-support/boundary handling；
- GeomLoss/SOT 参数、iteration、convergence criterion 与每 slice 预计算时间；
- parameter count、binary size、precision、MAC/FLOP、single-query latency；
- MIS estimator 的完整公式与 Fig.11 中 arbitrary values 如何进入 generalized MIS weights；
- 同一权威 GT 上的跨方法 raw error、统计不确定性和 white-furnace 数值。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

这个方法把容量拆成三部分：reference 先产生 dense slice；OT 为每个 `(ε,ωo)` 解一个高质量 deterministic coupling；MLP 只压缩 coupling 的 point queries。它的低 runtime 网络并不表示方法从少量 query 自行发现了完整 BSDF density。真正昂贵、也最影响 correctness 的部分在 teacher：slice resolution、grayscale reduction、discretization 和 OT solver。

对本项目而言，这更像 `compiler + proposal`，而非 evaluator architecture。它最值得迁移的不是六层 MLP 本身，而是“先构造平滑、可监督的 sampling map，再压缩”这一训练分解。

### 13.2 成功所依赖的假设

1. 固定 family 内，transport map 随参数与 `ωo` 足够平滑；
2. grayscale target PDF 有覆盖完整 BSDF support，且 RGB sampling weight 低频；
3. OT 的离散 coupling 在相邻条件间保持一致，没有 permutation branch switching；
4. 有能力为大量 `(ε,ωo)` slices 支付 reference 与 OT 预计算；
5. runtime 可以接受六层 64/128-wide MLP，或能把它进一步蒸馏/编译。

layered case 已显示 weight 不再接近常数；grazing 黑边证明上述联合近似并不自动满足 correctness，但论文没有做 isolation ablation，无法由黑边单独判定是 target support、coupling continuity、weight capacity 还是数值/训练因素失败。

### 13.3 可迁移机制与不能迁移的部分

可迁移：

- 用 reference density 构造 deterministic transport-map teacher；
- 将 sample direction 与低频 throughput-related quantity联合训练，作为 representation diagnostic；
- 用 map continuity/crevice 可视化在进入 renderer 前发现 proposal artifact；
- 对 current learned sampler 增加 OT teacher matched control，区分“proposal family 容量不足”和“KL optimization 不稳定”。

不能直接迁移：

- `sample()` 只返回 learned weight 而没有可核查 true proposal PDF；
- 把任意 MIS weighting value 当作 proposal `pdf()`。Fig.11 只支持 generalized MIS heuristic 的 query 可采用非 density 值；它不支持在 Monte Carlo throughput `f cos/p_sample` 中谎报采样 density。真实 proposal density、RGB learned weight 与 MIS query score 是三个不同对象；
- 每 family 一套长达六层、宽 64/128 的 CPU MLP，尚未满足本项目 shader budget；
- 持久化 offline corpus/importance-map database；本项目 formal training 必须由 reference GPU-online query 驱动。

### 13.4 与本项目 runtime contract 的关系

本项目要求 `sample()` 与 `pdf()` 属于同一 proposal，连续事件返回 `weight=f|cos|/pdf`，独立 `pdf(wi)` 仍由同一 native proposal 提供并用于 NEE；这比论文的“sample path 用 learned RGB weight、PDF network 只服务 MIS weights”更严格。[N scattering contract]

因此，本论文 sampler 只能直接作为：

- proposal/teacher 研究对象；
- sampling capacity diagnostic；
- compiler 中生成 transport targets 的候选步骤。

要成为 product candidate，至少要让 runtime 同时得到真实 proposal density：例如使用可求 Jacobian 的 bijective map、从同一 map 导出稳定 density，或把 OT map 蒸馏到具有解析 `sample/pdf` 的 proposal family。单独保留 learned sampling-weight head 不能通过 sample↔pdf 与 white-furnace 合同。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

| 主题 | 当前 NVIDIA 实现证据 | 论文机制 | 分类与影响 |
|---|---|---|---|
| Sampler representation | current `nvidia-rta2024-functional-f@2` 的 `11→32→32→32→9` 输出 two-lobe analytic GGX proposal parameters，sample/pdf 由同一 head 计算 | 6×64/128 直接回归 OT direction 与 RGB weight | `not-applicable` 于 evaluator fidelity；可作为 sampler matched control，不替换当前主身份 |
| Sampler objective | 从当前 learned proposal 取样，用 detached evaluator 的 `luminance(f)|cos|` 做 forward-KL score loss | supervised regression 到 offline OT coupling | `interface-adaptation`：可在同一 frozen evaluator 下比较 optimization target |
| PDF/weight | 当前 package 的 sample/pdf 属于同一 analytic proposal，weight 由 `f cos/pdf` 构造 | sample path 直接预测 weight；独立 PDF MLP 只给 MIS weighting query | 直接照搬会违反 runtime contract；必须标为 `intentional-deviation` 或重设计 |
| Data lifecycle | NVIDIA formal route 是 GPU-resident online reference query，不持久化 batch/corpus | 预计算 32,768×128² slices 与 OT maps | `intentional-deviation`：只能在线/临时 teacher 化，不能把 corpus 变成正式产品 |
| Runtime budget | 当前 evaluator/sampler 内联 Slang、regular FP16 path，并有 Torch↔Slang/Falcor parity | C++/Eigen CPU，未给 precision/单 query cost | `author-underspecified`，不能凭 “lightweight” 判断可部署 |

当前 NVIDIA sampler 最大的可检验启发是：**固定 evaluator 与 analytic proposal family，只替换训练监督为 OT transport teacher**。若 OT 监督改善了 variance 与 seed robustness，说明现有 forward-KL estimator/optimization 是瓶颈；若没有改善，则更可能是 two-lobe proposal family 或 evaluator target 本身限制。这个实验不需要改变 evaluator representation。

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| `H-IB-1`：OT teacher 能降低当前 sampler 的 optimization variance | P 中 OT map 比 discontinuous maps/RealNVP 更易回归；Taming 报告已证明 compact-network optimization variance 是独立问题 | online reference 可按小批条件临时求低分辨率 transport target，或离线 target 只用于诊断 | 同一 NVIDIA two-lobe sampler，一组 current forward-KL，一组把 OT samples 拟合到同一 9 参数 family | evaluator checkpoint、query stream、proposal topology、steps、seed set、runtime precision | per-seed validation variance、proposal KL/ESS、PT variance、sample/pdf parity、训练 time | `proposal-training-control` | bootstrap CI 不显示稳定改善，或 teacher cost/偏差抵消收益 |
| `H-IB-2`：直接 map 只在比 two-lobe proposal 明显更复杂的 layered lobes 上有必要 | P Fig.12/16 显示 analytic simple lobe 与 Guo random walk 的困难 | 当前层栈 reference 的 directional density也存在平滑低维 coupling | 预算内 direct-map MLP vs current two-lobe analytic proposal；同 evaluator、同 samples | hidden/MAC cap、reference source、view/material split、sampling measure | proposal variance、support miss rate、grazing error、shader latency/memory | `capacity-diagnostic` | matched budget 下无质量收益，或任何 support hole/white-furnace failure |
| `H-IB-3`：RGB weight head 可作为 auxiliary regularizer，但不应成为 runtime truth | P Fig.7 观察多数 weight slices 低频；Fig.18 又显示 runtime weight bias | 预测 weight 可帮助 latent/view encoding学习能量，但最终仍用 evaluator/pdf重算 | sampler 有/无 auxiliary log-weight loss；runtime 都使用真实 `f cos/pdf` | proposal、evaluator、queries、loss scale sweep、steps | direction loss、PDF KL、energy error、training stability | `training-only` | auxiliary head不改善 proposal或使 evaluator/sampler gradient冲突 |
| `H-IB-4`：共享 `prepare(ε,ωo)` 可回收三网络重复成本 | P 的 sample/eval/pdf 都共享 `(ε,ωo)` 条件，却独立从头编码 | view-conditioned hidden state 可供 evaluator 与 sampler/pdf 复用 | shared-prepare vs three independent networks，保持总参数/MAC matched | source/query、total params/MAC、precision、training schedule | single query与multi-light amortized latency、quality、state bytes | `product-candidate` | shared state降低任一核心质量，或 state/memory 超出预算 |

这些均是迁移假设，不是当前实现结论。尤其 `H-IB-1/2` 在通过 sample→pdf、normalization、support 与 white-furnace 前不得进入 environment/PT 质量排名。

## 16. 证据索引

- `P §1–2`：问题、贡献、related work 与所谓 general/perfect 的作者表述。
- `P §3.1–3.3, Eq.(1)–(3), Fig.2–6, footnote 1`：三个 family、rendering measure、importance-map 分析与 RGB scalarization 边界。
- `P §4.1, Eq.(4)–(5)`：离散 OT 点集与 permutation objective。
- `P §4.2–4.3, Eq.(6)–(8), Fig.7–12`：三个网络、bias/MIS argument 与重要对照。
- `P §5.1, Table 1`：slice/data/OT/input encoding。
- `P §5.2`：loss、optimizer、batch、LR、epoch、hardware、training time。
- `P §5.3`：Mitsuba/Eigen integration 与 sample path 直接返回 weight。
- `P §6, Fig.1, 13–17`：实验 protocol、CPU time 与 relMSE。
- `P §7–8, Fig.18`：bias、performance 与 future work。
- `A-VOR/A-Crossref/S-VOR`：2025 VOR 身份、abstract变化及当前不可得材料；不承担 2023 内部配置证据。
- `N scattering/compiler contracts`：§14 的项目 runtime/data lifecycle 边界。
- `N current NVIDIA formal config/method/model`：§14 current `functional-f@2` sampler topology、objective、precision 与 parity 身份；归档 correspondence 不代表当前实现。

## Evidence review

```text
author_worker: /root
reviewer: /root/nbrdf2021
reviewed_at: 2026-08-29
sources_rechecked:
  - arXiv:2210.13681v3 PDF, SHA-256 6E915EA80DF6B4D42C2B46EA295841B3DCD8C7A5168314EA6CC69A2BA1418A44, all 14 pages visually checked
  - author-hosted UCSB PDF, byte-identical to arXiv v3
  - Eq.(4)-(8), Table 1, footnote 1, Fig.1/8/10/13-18 and their captions
  - Ling-Qi Yan current publication entry, Wiley VOR landing page and Crossref DOI metadata
  - Wiley VOR PDF and cgf70286-sup-0001-SuppMat.pdf direct locators, both returning HTTP 403
  - current NeuralShading scattering/compiler contracts and NVIDIA functional-f@2 config/method/model
findings_closed:
  - rechecked all three network topologies, hidden/output activations and encoded dimensions directly against rendered Fig.8/Fig.10 and Table 1
  - rechecked 32,768x128^2 data statement, GeomLoss-plus-SOT lifecycle, three losses, batch 1,048,576, 500 epochs and training hardware/times
  - rechecked every Fig.1/13-17 time, spp and relMSE value and constrained their interpretation to the paper's own-converged-GT protocol
  - separated true OT target density, learned RGB sampling weight and independent MIS query score; recorded that Fig.11's manual -10 is not an Exp-network output and cannot authorize a false proposal PDF
  - corrected 2025 VOR year boundary and removed unsupported 2026-citation wording; no VOR method detail was inferred through the 403 boundary
  - updated N/I evidence to current nvidia-rta2024-functional-f@2 sources and retained the sample/pdf same-proposal requirement
remaining_evidence_gaps:
  - 2025 CGF VOR PDF 与 53.6 MB supporting information 被 Wiley 403 阻断
  - 2023 promised importance-map database 与 official code/config/weights 未找到
  - exact disk map、OT配置、parameter ranges、split/seed、precision与single-query runtime 未报告
  - “each material”的 family/checkpoint 粒度、Ranger/scheduler 细节、own-converged GT 图像与 raw metrics 仍无法消解
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
