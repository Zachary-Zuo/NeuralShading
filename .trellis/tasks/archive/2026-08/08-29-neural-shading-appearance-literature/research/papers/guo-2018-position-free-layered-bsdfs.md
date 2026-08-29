---
paper_id: "guo-2018-position-free-layered-bsdfs"
title: "Position-Free Monte Carlo Simulation for Arbitrary Layered BSDFs"
authors: "Yu Guo, Miloš Hašan, Shuang Zhao"
year: "2018"
venue: "ACM Transactions on Graphics 37(6), Proceedings of SIGGRAPH Asia 2018"
doi: "10.1145/3272127.3275053"
report_status: "evidence-reviewed"
main_source: "https://projects.shuangz.com/layered-sa18/layered-sa18.pdf"
supplemental_status: "available"
official_code_status: "audited"
official_code_commit: "ca6e9b19fb122c126d605207d9f4790e86b03651"
author_worker: "/root/rta2024"
reviewer: "/root"
last_verified: "2026-08-29"
---

# Position-Free Monte Carlo Simulation for Arbitrary Layered BSDFs

## 1. 研究对象与报告边界

Guo、Hašan 与 Zhao 研究的是薄层状材质的随机 BSDF 求值：在忽略入口与出口的横向位移后，把表面界面和均匀介质中的多次反射、折射、吸收与散射直接写成一个只含深度和方向的 path integral，再对该积分做 Monte Carlo `sample/evaluate/pdf`。该方法属于 `local-material transport` reference，不是 neural appearance、scene-level global illumination 或通用 BSSRDF；场景 path tracer 只把它当作一个随机 BSDF 使用。[P Abstract, §§1,3–5]

本报告覆盖以下正式证据：ACM TOG 37(6) Article 279 的 14 页项目页 PDF、作者发布的两份 3 页 supplemental、第一作者的 Mitsuba 0.6.0 fork、官方 supplemental scenes/config 与作者项目页。代码同时锁定论文发布期 commit 和当前官方 HEAD，以隔离 2019 年的 zero-PDF safety fix。作者另有一个 17 页论文源码编译件，把两份 supplemental 合并为 appendices；它不是新的正式期刊修订版，故事实层仍以 14 页正式 PDF 和两份独立 supplemental 为准。[P; S-MIS; S-BD; C-paper]

本报告重点重建：

1. position-free path state、测度和贡献函数；
2. surface/volume propagation 与多层 boundary handling；
3. forward sampling、单向/双向 evaluation、exact/approximate PDF 和 MIS 独立性条件；
4. 正式实验、成本、负结果、代码默认值与 paper/config gaps；
5. 它作为 Neural Layered BRDFs、MetaLayer 和 BSDF Importance Baking 的 reference/baseline 边界。

不把后续 neural 方法的 dense projection、sampler 或 runtime architecture 反向归因给 Guo 2018，也不把发布代码支持的插件集合等同于论文形式化模型对“arbitrary interface BSDF”的数学边界。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | [official project PDF](https://projects.shuangz.com/layered-sa18/layered-sa18.pdf)，DOI `10.1145/3272127.3275053` | 2026-08-29 | SHA-256 `4A28A33938A4530D6FB17D7DE263238C97397CD0C03369517838E06ECA712375` | 正式 14 页 TOG 版本；方法、主实验、限制和 Appendix A |
| Supplemental MIS proof `S-MIS` | `supp_docs.zip/supp_docs/mis_proof.pdf`，来自 [official project page](https://projects.shuangz.com/layered-sa18/) | 2026-08-29 | PDF SHA-256 `A2F83FF41BA15BDCF1D39439C6D7B011D1D0B2936E24F592C01AEA1B5564E30B`；ZIP SHA-256 `45181A7F83AA2DFA3976B4A1BC65215BE3C9A08AD245C2B718B431C52E246910` | stochastic function/PDF 的 MIS 无偏条件；3 页 |
| Supplemental bidirectional details `S-BD` | `supp_docs.zip/supp_docs/supp-bidir.pdf` | 2026-08-29 | SHA-256 `76BE30ECFF29EC3D67A95F034B77BBCE51214199572279086B106444690CFBAD` | bidirectional estimator、balanced MIS ratio recurrence；3 页 |
| Supplemental webpage/images `A-supp` | [official supplemental page](https://tflsguoyu.github.io/layeredbsdf_suppl/)，`supp_imgs.zip` | 2026-08-29 | ZIP SHA-256 `A1837F8B5E59FE84551FF5B8DC2B59F54C24BC344B011D4DFCE808C2677A2C78` | 等时验证、lobe/PDF/材质参数 sweep 与图像文件名；不是 raw metric dataset |
| Current official code `C` | [tflsguoyu/layeredbsdf](https://github.com/tflsguoyu/layeredbsdf/tree/ca6e9b19fb122c126d605207d9f4790e86b03651) | 2026-08-29 | commit `ca6e9b19fb122c126d605207d9f4790e86b03651`；source ZIP SHA-256 `E28F023F6A212D1159A3A0E4C713F8FB96B89EACE9C611A505E631526EB698C8` | 第一作者官方 Mitsuba 0.6.0 fork；静态审计 runtime、defaults 与后续 safety fix |
| Paper-time code `C-2018` | 同 repo commit `3e414f1507a0a72896b03267c97195991519fbd7`（2018-11-09） | 2026-08-29 | source ZIP SHA-256 `CC1F7E7B12F1783AD342698F46C518AC9FFF59923E048679B94278A0563185B3` | 隔离论文发布期实现；其后唯一 source diff 是 2019 zero-PDF checks |
| Official scenes/config `C-scenes` | [tflsguoyu/layeredbsdf_suppl](https://github.com/tflsguoyu/layeredbsdf_suppl/tree/dd545874aa9e04691f1623ba795b3a6a0216ef50) | 2026-08-29 | commit `dd545874aa9e04691f1623ba795b3a6a0216ef50`；Fig.2/3/8/12/14/15 ZIP hashes见 §16 | 正式 scene XML 与 examples；不含全部 Dropbox 资产、raw metrics 或统一 seed manifest |
| Paper source `C-paper` | [tflsguoyu/layeredbsdf_paper](https://github.com/tflsguoyu/layeredbsdf_paper/tree/c190f80f1d59c18f2c0eee2d31b1737934211fa2) | 2026-08-29 | commit `c190f80f1d59c18f2c0eee2d31b1737934211fa2`；compiled PDF SHA-256 `298E72CCAAD98106DF78E1E45FB49A731B483DDF98C87CE43C29A650DFF4496` | TeX、caption 与 appendix locator；2020 合并 supplemental，不覆盖正式版 |
| Author page/correction `A` | [official project page](https://projects.shuangz.com/layered-sa18/) | 2026-08-29 | 固定 URL | 作者、venue、paper/supp/code入口；未发现作者正式 correction/errata |
| NeuralShading evidence `N` | [runtime contract](../../../../../docs/contracts/scattering_backend.md)、[experiment framework](../../../../../docs/research/experiment_framework.md)、[NVIDIA correspondence](../../../archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md) | 2026-08-29 | repo-local | 仅用于 §§13–15，不回填为 2018 论文事实 |

已检索 official project、三个作者 repo 和 ACM 入口，未发现单独的正式勘误。`layeredbsdf_paper` 在 2020 年把 supplemental 合并到附录，2022 年更新编译件/版权信息；`small fix` commit 只修 bibliography 文件名。不能由“未发现”证明不存在其他未公开说明，本报告只写“第一方公开入口未发现”。

## 3. 原论文的问题、假设与贡献边界

### 3.1 问题与假设

精确 layered BSDF 要积分界面和介质中的所有光路。三维 volumetric random walk 会保留横向位置，代价高且作为局部 BSDF 时引入很多并不需要的状态。作者的核心假设是：材质薄、局部平坦，layer properties 在典型横向 transport footprint 内变化足够慢，因此可忽略入口/出口的水平位移，只保留归一化深度 `z` 和方向 `d`。[P §§1,3.1]

对一个 slab，top/bottom 位于 `z=0/1`，中间是 homogeneous volume。path vertex 的状态为 `z_i∈[0,1]`；`z_i=0/1` 表示界面，严格内部表示 volume event。方向 `d_i∈S²` 按 light-flow convention，固定 `d_0=-ω_i`、`d_k=ω_o`。这个 reduction 不把多个 bounce 压成解析 closure，而是继续随机枚举完整 path。[P §§3.1,4.2, Fig.6]

### 3.2 贡献边界

作者声明的贡献是：

- position-free path integral，可统一任意数量的 surface/volume events；
- 可在界面使用任意可 `sample/eval/pdf` 的 BSDF、在介质使用 phase function，并支持 anisotropy、spatially varying parameters、normal mapping 与 multiple slabs；
- forward random-walk sampler、unidirectional next-event estimator、bidirectional estimator，以及 matching exact/approximate PDF；
- 与 scene path tracing 的 stochastic BSDF/MIS 组合方式；
- 无 precomputation 的逐 query stochastic evaluation。[P §§1,3–5]

“arbitrary”指形式化 vertex scattering model 的可替换性，不代表所有 release plugin 都已支持。官方 README 明确说 exact `conductor`/`dielectric` 不支持，建议用 `roughconductor`/`roughdielectric` 且 `alpha=.001`；这属于 release 实现边界。[C `README.md`]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | ordered surface interfaces + homogeneous volume slabs；每界面 BSDF、每 slab `σ_t,σ_s/phase` | 单 slab `[0,1]`；多 slab 通过 layer/media identity 扩展 | [P §§3.1,4.2,4.6] |
| Runtime query | `f_l(ω_i,ω_o)`、forward `sample(ω_i)`、matching `pdf(ω_o\|ω_i)` | external directions on sphere/hemisphere，按 interface side 判断 reflection/transmission | [P §§4.2,5.1–5.3] |
| Path state | `x̄=(d_0,z_1,d_1,…,z_k,d_k)` | internal directions `S²`；volume depth line measure；interface depth为离散 boundary | [P Eqs.2–8, §4.2] |
| Coordinate convention | `z=0` top，`z=1` bottom；`d`沿 light flow；`d_0=-ω_i,d_k=ω_o` | 局部平坦、无水平坐标 | [P Fig.6, §4.2] |
| Output quantity | layered BSDF `f_l`；sampler返回 stochastic estimate of `f_l |cosω_o|/p`；PDF为 solid-angle directional density | radiometric BSDF；Mitsuba smooth `eval`内部是 cosine-weighted convention，plugin/integrator fused path需按其 ABI解释 | [P Eq.8, §§5.1,5.3; C `include/mitsuba/render/bsdf.h:L344-L449`] |
| Validity/domain restrictions | lateral displacement可忽略；局部 layer parameters慢变；geometric slab boundaries平行 | BSDF，而非空间扩展 BSSRDF | [P §§3.1,6.4] |

论文把 path-space measure 写为所有内部方向的 solid-angle measures 与所有 volumetric depths 的 line measures之积；不存在 surface-area vertex 或 inverse-square geometry term。[P Eqs.7–8, §4.2]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

这篇论文没有 neural network。它的“representation”是运行时临时生成的 position-free path：

```text
ordered layer stack + (ωi, optional ωo)
  → choose a boundary/interface or volume event
  → update (z, direction, layer/media identity, throughput, path PDF)
  → repeat until external exit / termination
  → sample: return exit direction and stochastic weight
  → evaluate: connect/query the fixed external direction, aggregate uni/bidirectional estimators
  → pdf: independently integrate the path-generation probability
```

对 path `x̄`，贡献为

\[
f(\bar x)=v_1s_1v_2\cdots s_{k-1}v_k,
\]

其中 vertex factor `v_i` 在 interface 是 BSDF，在 volume 是 reduced phase function `\hat f_p=σ_s f_p`。均匀介质 transmission 为

\[
\tau(z,z',\omega)=
\exp\!\left(-\sigma_t\frac{|z'-z|}{|\cos\omega|}\right)
\mathbf 1\!\left[\frac{z'-z}{\cos\omega}>0\right].
\]

segment factor 为

\[
s_i=\tau(z_i,z_{i+1},d_i)|\cos d_i|^{\alpha_i},\qquad
\alpha_i=\mathbf1[z_i\text{ interface}]+\mathbf1[z_{i+1}\text{ interface}]-1.
\]

所以 surface–surface segment 含一个 cosine，surface–volume 不含，volume–volume 含 inverse cosine。这个指数并非经验配置：它来自 RTE 从距离到 depth 的变量替换和 boundary rendering equation 的 cosine。[P Eqs.2–8, §4.3, Fig.6, Appendix A]

最终 layered BSDF 是所有 position-free paths 的积分：

\[
f_l(\omega_i,\omega_o)=\int_\Omega f(\bar x)\,d\mu(\bar x).
\]

### 5.2 持久化表示

| 内容 | 持久化/临时 | 正式边界 | locator |
|---|---|---|---|
| Layer parameters | scene/material持久化 | interface BSDFs、slab `σ_t/albedo/phase`、normal/thickness textures | [P §§4.4–4.6,6; C `multilayered.cpp:L1573-L1667`] |
| Path state | per-query临时 | depths、directions、surface/layer IDs、forward/backward throughput、PDF ratios、survival | [S-BD; C `multilayered.cpp:L473-L498`] |
| Learned latent/weights | 不适用 | 无训练、无 neural storage | [P full paper] |
| Precompute/cache | 无 mandatory precomputation | 每个 query现场 random walk；scene renderer可按普通 BSDF共享材质参数 | [P Abstract, §§1,5] |

### 5.3 网络逐层配置（本论文为 layer/path operations）

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Surface vertex | incoming direction + interface BSDF | BSDF `sample/eval/pdf`；可能 reflection/refraction | 不适用 | new direction/vertex factor/local PDF | interface-specific | [P §§4.2,5; C `multilayered.cpp:L380-L470,L499-L738`] |
| Volume propagation | current depth/direction + medium | exponential free-flight in physical distance，映射到 depth；若到 boundary则 surface event | 不适用 | next `z`、transmittance、depth PDF | slab-specific | [P §§4.2–4.3; C `multilayered.cpp:L296-L378,L499-L738`] |
| Volume vertex | direction + phase function | phase `sample/eval/pdf`，factor `σ_s f_p` | 不适用 | new direction/vertex factor/local PDF | medium-specific | [P §4.2; C `multilayered.cpp:L499-L738`] |
| Unidirectional evaluation | fixed `(ω_i,ω_o)` | random walk；每个 surface/medium vertex对 external boundary做 NEE；continuation与 NEE local MIS | power heuristic | stochastic `f_l` estimate | runtime | [P §5.2.1; C `multilayered.cpp:L741-L956`] |
| Bidirectional evaluation | light/camera subpaths | enumerate compatible prefix endpoint pairs；每对分别从两端 importance-sample connecting direction；balanced MIS | balance heuristic/ratio recurrence | stochastic `f_l` estimate | runtime | [P §5.2.2; S-BD; C `multilayered.cpp:L958-L1193`] |
| Directional PDF | `ω_i,ω_o` + proposal rules | 对 path-generation probability `P(x̄)`做另一个 path integral；exact或short-path approximate | 与 stochastic BSDF随机数独立 | `p(ω_o\|ω_i)` estimate | runtime | [P §5.3; S-MIS; C `multilayered.cpp:L1195-L1365`] |

### 5.4 条件化、坐标变换与物理先验

- **Normal mapping。** segment factor 改为 `τ |⟨n(z_i),d_i⟩||⟨n(z_{i+1}),d_i⟩|/|cos d_i|`。由于两端 local normals 可能不同，该 construction一般不 reciprocal；light-side path 需要 Veach adjoint correction。[P §4.4]
- **Refraction。** interface 折射包含方向/measure Jacobian；单个界面满足带 `η²` 的 reciprocity。若 layered material两端都回到 air，相配的折射 `η` factors总体恢复外部 reciprocity；内部 state不能忽略 side/IOR。[P §4.5]
- **Multiple slabs。** 正式通用形式为显式 surface/layer identity，并拒绝跨越错误内部 boundary 的 segment。作者也实现了把双层 BSDF递归嵌套的替代方案，但多层时更慢。[P §4.6]
- **Spatial variation。** thickness、density、albedo、phase或 interface parameters可按 query surface位置取值，但 path内仍使用同一局部参数；这依赖“变化尺度大于横向扩散”的局部性假设。[P §§3.1,5]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset/source assets/scenes | 作者自建 Mitsuba scenes：单 slab、两 slab/三界面、rough dielectric/conductor、HG/vMF、anisotropic microflake、cloth、spatially varying coating/thickness、多层 kettle | [P Figs.1–2,8,11–15; C-scenes] |
| GT/reference renderer | Fig.2 等时图的 reference是 standard path tracing `100K spp`；Fig.4 lobe reference由 forward sampling/binning得到；white furnace用constant illumination | [P Figs.2,4,10] |
| Train/validation/test split | 不适用；无训练 | [P full paper] |
| BSDF sample recipe | 从 `-ω_i` forward random walk；每 vertex按原 interface BSDF/phase proposal继续，直到外部退出 | [P §5.1] |
| Evaluation recipe | unidirectional NEE/local MIS，或 bidirectional两端 subpaths/全部 compatible prefix pairs/two connection proposals/balanced MIS | [P §§5.2.1–5.2.2; S-BD] |
| PDF recipe | exact path integral或 short interface-only approximation；stochastic BSDF与PDF必须用独立 paths/random numbers | [P §5.3; S-MIS] |
| Approximate PDF | 忽略 volume-scattering paths，只保留 short interface paths，并混合 constant Lambertian density；正文常数 `0.1` | [P §5.3.2] |
| Filtering/LOD/footprint | 未报告；方法在单 shading point 使用局部 layer parameters | [P §§3.1,5] |
| Online/offline generation | 全部 runtime online；无预训练或 tabulation | [P Abstract, §§1,5] |

approximate PDF 的 formal path-length bound是：`L` 个 layers时 reflection最多 `2L+1` vertices，transmission最多 `L+1` vertices。它只用于 scene-level MIS weights；不能放进 stochastic BSDF estimator 的分母。实际 `f/p` 已包含在 stochastic BSDF sample weight中。[P §5.3.2]

作者 supplemental 还固定了若干参数 sweep：PDF示例用 roughness pairs `(.005,.005),(.005,.1),(.1,.005),(.1,.1)`、IOR `1.5`、`σ_t=1,g=0`；jade示例固定 albedo `(.4,1,.8)`、IOR `1.65`、phase mixture `0.8 HG(.8)+0.2 HG(-.8)`并 sweep RGB extinction；magnifier对 HG `g=.90/.93/.96/.99` 或 vMF `κ=10/14/25/100` 与 extinction `1/1.41/2/2.83`做组合。[A-supp]

## 7. Loss、optimizer 与训练 lifecycle

本论文不含 neural training，所有训练字段均不适用：

| 项 | 正式配置 | locator |
|---|---|---|
| Target/output transform | 不适用；直接 Monte Carlo估计 physical BSDF | [P §§4–5] |
| Loss terms and weights | 不适用 | [P full paper] |
| Optimizer/hyperparameters | 不适用 | [P full paper] |
| LR schedule | 不适用 | [P full paper] |
| Batch/query count | 无 training batch；render spp见 §§9 | [P §6] |
| Steps/epochs/stages | 不适用 | [P full paper] |
| Initialization/seed/model selection | 无 model；Monte Carlo seed和独立 stream具体值未报告 | [P §5.3; S-MIS] |
| Hardware/training time | 无 training；render hardware见 §9 | [P Table 2] |

不要把 downstream NLB/MetaLayer 用 Guo reference产生 dense targets 的训练 lifecycle算作 Guo 2018 的配置。

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path | 每个 shading query可能调用 stochastic `sample/eval/pdf`；scene path tracer用 fused `evalAndSample`和 global light/BSDF MIS | [P §§5.1–5.3; C `path_layered.cpp:L167-L224`] |
| Parameter count/MAC/FLOP | 不适用；成本由随机 path length、vertex BSDF/phase cost和 estimator strategy决定 | [P §§5–6] |
| State bytes | 未报告；release `PathInfo`含 position/directions/counters/surface/layer IDs/throughputs/PDFs/RR state | [C `multilayered.cpp:L473-L498`] |
| Texture/feature fetches | 未报告；spatial parameters可按 UV读取 | [P §§4.4,6; C `multilayered.cpp:L236-L294`] |
| Precision | paper未报告；README称默认 single precision，遇 warning/crash可尝试 double | [C `README.md`] |
| Hardware | Table 2统一换算到 6-core Intel i7-6800K CPU | [P Table 2 caption] |
| Asymptotic cost | unidirectional随walk length；bidirectional直接 MIS weight evaluation本可为 `O(n_i n_o(n_i+n_o))`，ratio recurrence降到总 `O(n_i n_o)` | [S-BD §§2–3] |
| Static boundedness | 不满足：formal walk依赖随机 bounce count/Russian roulette；双向又枚举subpath endpoint pairs | [P §§5.1–5.2; S-BD] |
| Precompute/amortization | 无 mandatory precompute；approximate PDF以较少路径换运行时 | [P §§5.3–6] |

Table 2 的正式时间如下，括号为 `minutes/megapixel`。它们不是单次 BSDF query latency，也不是 GPU shader timing：

| Figure / resolution / spp | Unidirectional | Bidirectional | Trivial BSDF | locator |
|---|---:|---:|---:|---|
| Fig.1(a), `3000×2000`, 1024 | 2.5 h (25) | 2.2 h (22) | 38 m (6.3) | [P Table 2] |
| Fig.11(b), `1024²`, 256 | 2.2 m (2.1) | 2.6 m (2.5) | 1.3 m (1.2) | [P Table 2] |
| Fig.12 top, `800×1200`, 512 | 15.2 m (7.9) | 24 m (12.5) | 2.4 m (1.3) | [P Table 2] |
| Fig.12 bottom, `512²`, 1024 | 6.4 m (6.1) | 13 m (12.6) | 1.6 m (1.5) | [P Table 2] |
| Fig.13(a/b/c), `876×584`, 256 | 1.1 / 1.1 / 2.5 m (2.2 / 2.2 / 4.9) | 1.4 / 1.4 / 5.4 m (2.7 / 2.7 / 10.5) | .6 / .5 / .5 m (1.1 / .9 / .9) | [P Table 2] |
| Fig.14(b), `640×540`, 256 | 1.5 m (4.3) | 1.9 m (5.5) | .5 m (1.4) | [P Table 2] |
| Fig.15(a/b/c), `1200×1400`, 256 | 6.7 / 7.0 / 67 m (4.0 / 4.2 / 40) | 12 / 13 / 20 m (7.1 / 7.7 / 12) | 3.7 / 3.7 / 4.7 m (2.2 / 2.2 / 2.8) | [P Table 2] |

这些数据明确否定“bidirectional总是更快”：它在很多简单/中等配置更慢；价值在困难高阶 scattering paths上的variance reduction。Fig.15(c) 的 `67→20 min`还同时改变了 nested unidirectional vs explicit bidirectional representation，不能当成纯 estimator ablation。[P §6, Table 2]

## 9. 实验 protocol、baseline、指标与结果

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| 单 slab equal-time | HG slab；所有方法 10 s；reference standard PT 100K spp | PT 98 spp、BDPT 35、MLT 280、ours uni 56、ours bi 26 | 视觉 noise；无 numeric MSE | 作者称两种 ours均明显优于 global methods，uni/bi相近 | [P Fig.2 top/caption] |
| 两 slab/三界面 equal-time | anisotropic microflake；所有方法 10 s；100K-spp reference | PT 60、BDPT 25、MLT 80、uni 15、bi 19 | 视觉 noise | 作者称 bidirectional是clear visual winner | [P Fig.2 bottom/caption] |
| Outgoing lobes | fixed incoming；forward sampling/binning reference；uni/bi同时间 | reference histogram | lobe image，非误差表 | 两种 estimator视觉匹配 reference，包括复杂 multi-lobe | [P Fig.4] |
| PDF/MIS | same scene：exact PDF 64 spp/14 m，approx 64 spp/4.1 m，no MIS 80 spp/4.2 m；supp reference 400 spp/20 m | exact PDF、approx PDF、no MIS | equal-ish time视觉 noise | 该例approx与exact视觉质量相当且大幅更快；no MIS明显更差 | [P Fig.8; A-supp] |
| White furnace | 3 configurations，constant illumination | expected constant response | qualitative invisibility | 通过视觉 energy-conservation sanity check；无数值 energy error | [P Fig.10] |
| Spatially varying/normal/thickness | coating、thin sheet、magnifier、kettle | trivial/author configurations | render image/time | 展示heterogeneous parameters、reflection/transmission和multiple slabs | [P Figs.1,11–15; Table 2] |
| Cloth/micro-CT comparison | anisotropic microflake cloth vs prior volumetric micro-CT simulation | volumetric reference method | stated speedup + visual | 正文称约40× faster；supp文件名为reference 128 spp/36 m、ours 128 spp/2.9 m，而Table 2 paper image为ours 256 spp/1.9 m（换算硬件） | [P Fig.14/Table 2; A-supp filenames] |

cloth 的“40×”不能直接从一组严格 matched 的公开 runtime行读取。`36 min / 2.9 min≈12.4`；只有把 reference 128 spp线性归一到256 spp后再与Table 2的1.9 min比较，才得到约 `72/1.9≈38`。[I] 因此保留作者结论，但不得把它改写为已公开的同硬件、同spp直接测得40×。

supplemental还给出四组等时/较长时验证，均为视觉对照、无MSE或置信区间：

| Configuration | Short time: PT / BDPT / MLT / uni / bi spp | 180 s: PT / BDPT / MLT / uni / bi spp | locator |
|---|---|---|---|
| rough dielectric `α=.05,η=1.5`; `σ_t=1`, albedo `(.5,.7,.95)`, HG `.5`; Cu `α=.05` | 14 s: 32 / 10 / 30 / 16 / 8 | 400 / 120 / 850 / 200 / 108 | [A-supp equal-time 1] |
| `σ_t=2,g=.9`; Cu `α=.01` | 22 s: 40 / 8 / 32 / 20 / 12 | 320 / 64 / 700 / 150 / 100 | [A-supp equal-time 2] |
| `σ_t=5,g=.95`; Cu `α=.01` | 22 s: 28 / 6 / 10 / 10 / 8 | 225 / 48 / 320 / 80 / 64 | [A-supp equal-time 3] |
| two slabs anisotropic | 18 s: 25 / 9 / 30 / 13 / 8 | 250 / 85 / 820 / 125 / 85 | [A-supp equal-time 4] |

不同方法在相同wall time下spp不同，且没有per-pixel error或重复seed统计；只能支持作者给出的视觉/效率结论，不能建立严格跨方法variance ranking。

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `ablation-inferior` | scene MIS关闭 | 80 spp/4.2 m仍明显比approx-PDF 64 spp/4.1 m noisy | 未利用BSDF directional distribution | 对layered BSDF不可默认uniform/light-only sampling | [P Fig.8] |
| `ablation-inferior` / cost tradeoff | unbiased exact PDF | 64 spp/14 m；该例视觉质量与approx 64 spp/4.1 m相近 | exact PDF另做完整 path integral，成本高 | exact不是失败；只说明这一个场景中short-path PDF用于MIS已足够 | [P §5.3, Fig.8] |
| `ablation-inferior` | recursive nested multi-layer evaluation | difficult Fig.15(c)为67 m，对explicit bidirectional为20 m | nested estimator对跨多界面paths效率差 | representation与uni/bi estimator同时改变，不能归因于单一因素 | [P §§4.6,6, Table 2] |
| `author-positive`（baseline结果，不是失败） | standard scene PT/BDPT/MLT直接显式模拟 | Fig.2 equal-time更noisy，尤其anisotropic multi-slab | global strategies没有利用position-free local path space | 它们是baselines，不是作者失败历史 | [P Fig.2] |
| `author-positive`（有边界的成功） | bidirectional vs unidirectional | difficult configurations有优势；多行Table 2反而更慢，单slab视觉接近 | 连接更多path strategies提高难路径概率，但 endpoint-pair成本更高 | 不应把bidirectional写成统一dominance | [P Fig.2, Table 2; S-BD] |
| `paper-code-gap` | 2018 code在near-zero depth/phase/BSDF PDF时没有current guards | 2019 commit加入epsilon check并将throughput置零/提前终止 | commit message仅写“add zero check” | 这是发布后robustness fix，不是paper correction，也未给触发率 | [C-2018→C commit `cc763ca60ba84b4ee821b5620702d28835b390b2`] |
| `paper-code-gap` / `suspected-defect [I]` | release `pdf="TRT"` branch的ID arrays先`resize(n)`再`push_back` | 后续索引前n项仍是零值，和预期ID列表不符 | 作者未报告 | 静态代码审计的高置信疑点；正式scene用`bidirStoch`/`bidirStochTRT`，不能外推为论文结果失效 | [C `multilayered.cpp:L1206-L1220,L1226-L1262`] |

在已获得第一方材料中，没有报告“尝试某个完整算法后失败并放弃”的开发历史。不要从最终 estimator结构反推未披露失败尝试。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Position-free state | depth/direction path、surface/volume measure | S-BD给subpath endpoint和PDF ratios | `PathInfo`保存 `p,wi,wo,surf,layerID,throughputs,PDFs,survival` | 高层对应；代码还需Mitsuba side/medium state |
| Unidirectional | NEE at every event + local MIS | 无额外形式化 | `multilayered.cpp:L741-L956`；multi-layer可nested | 对应；published Fig.2 multi-layer uni用nested |
| Bidirectional | 两侧subpaths、每endpoint pair两种direction proposals | balanced MIS与`O(n_i n_o)` recurrence | `L958-L1193`枚举compatible pairs和ratio weights | 对应；single-vertex paths单独处理 |
| Exact PDF | 对path-generation probability `P(x̄)`积分 | S-MIS证明独立stochastic estimates可进MIS | `pdfTRT`/stochastic modes在`L1195-L1365` | 必须与BSDF estimate使用独立随机数 |
| Approximate PDF media | “remove all volume media”/只留short interface paths | 证明不要求approx PDF无偏，只要求MIS权重条件 | `setParametersPdf`保留 `σ_t/density/orientation`，把albedo设0 | paper-code gap：代码是pure absorption/no scattering，不是字面删除介质 |
| Approx PDF depth | reflection `2L+1`，transmission `L+1` | 无额外mapping | 单个`stochPdfDepth`；default `-1`，examples 2或4 | formal双bound到单config的映射未说明 |
| Approx diffuse term | 正文常数 `0.1` | supplemental正文沿用方法 | Fig.8 XML的`diffusePdf`有配置为`1` | 未解析paper/config冲突；不能把example改写成formal default |
| RR/termination | formal subpaths可由Russian roulette终止 | recurrence含survival probability ratios | `bidirUseAnalog` default false；Fig.2/3/8常设true，Fig.12/14/15未必设置 | formal、plugin default、figure example需分开 |
| `maxDepth` | scene path length由renderer控制；BSDF walk形式上无固定上限 | 未报告 | layered plugin解析`m_maxDepth`但不再使用；scene XML用integrator `maxDepth` | plugin property看似无效；不能当作formal fixed runtime bound |
| Numerical guards | 未报告 | 未报告 | 2019后加入zero-PDF checks | paper-time release缺少，current default已修 |
| Precision/support | 未报告 | 未报告 | README：single默认；exact delta conductor/dielectric不支持 | release工程边界，不改写数学贡献 |
| Evaluation assets | Figs.2/8/12/14/15与Table 2 | 更多equal-time和sweeps | official scene repo有2/3/8/12/14/15，部分资产仍在Dropbox | 无完整all-figure immutable bundle/raw metrics |

Fig.8 的 official no-MIS XML使用 `volpath_simple`，而 exact/approximate PDF variants走layer-aware path integration；这实现了“关闭scene MIS”的实验意图，但不是只翻转layered integrator中的一个布尔量。报告因此保留paper caption的视觉结论，不把三张图解释成除PDF之外所有runtime路径都完全相同。[C-scenes Fig.8 XML; I scope]

current plugin defaults为：`maxDepth=-1`、`mis=true`、`multilayer=false`、`pdf=bidirStochTRT`、`stochPdfDepth=-1`、`pdfRepetitive=1`、`diffusePdf=0`、`bidirUseAnalog=false`、`bidir=true`、`maxSurvivalProb=1`、`nbLayers=2`。[C `multilayered.cpp:L26-L39`] 这些是 release defaults，不应拿来补全每张paper figure；正式结果以对应XML override和paper caption为准。

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

- 方法只适合薄、局部平坦、横向 transport spread可忽略的材质；大尺度geometry/optical变化导致的内部caustic、shadow和不同位置间color bleeding不会被表达。[P §6.4]
- optically thick、high-scattering layers会生成更长path并提高variance/runtime；作者指出这是主要困难区。[P §6.4]
- 它是BSDF approximation，不保留BSSRDF的入口/出口位移。[P §§3.1,6.4]
- 不含wave optics或thin-film interference；界面只通过给定geometric-optics BSDF交互。[P method domain]
- normal mapping construction一般不reciprocal；必须区分camera/light transport和adjoint correction。[P §4.4]
- 双向 estimator成本随两侧subpath lengths乘积增长，虽用recurrence避免了额外线性因子。[S-BD]
- runtime是随机、动态且无硬上界，不满足固定shader execution cost合同。[P §§5.1–5.2]

### 12.2 未报告/材料不可得

- 每张正式图的random seeds、重复run、variance/MSE、confidence interval和reference standard error；
- single-query latency、path-length distribution、tail latency、state bytes、cache/coherence和SIMD/GPU behavior；
- white-furnace数值误差、reciprocity residual或sample/pdf normalization统计；
- Fig.4 lobe bin数、样本数和误差；
- 所有正式scene/assets的一个完整immutable bundle；teaser/Fig.11/Fig.13部分数据依赖Dropbox；
- 论文formal build对应的精确compiler flags、single/double precision与代码revision；
- paper approximate PDF的两种path bound如何映射到release单一`stochPdfDepth`；
- Fig.8 paper `diffusePdf=0.1`与official XML值`1`的原因；
- `pdf="TRT"`疑似ID-array defect的作者确认、影响范围与测试；
- 当前repo 2019 zero-check触发频率和对bias/variance的影响；
- 正式 correction/errata：第一方入口未发现，但无法证明不存在未公开说明。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

Guo 2018不压缩函数容量。所有材质复杂度仍保存在原生ordered interfaces、volumes和每vertex BSDF/phase functions中；query时通过随机path count与多个proposal即时展开。因此它的“表达能力”来自保留原生transport process，而非固定维度latent或解析lobe。对本项目而言，这正适合作为高保真 `reference oracle`，不适合作为部署representation。[I]

相较三维volume path tracing，position-free reduction删除的是横向位置和几何传播自由度；它没有删除bounce序列或多次散射。因此它将“局部薄层”假设换成更低维但仍随机的ground-truth generator。[P §§3.1,4.2; I]

### 13.2 成功所依赖的假设

1. 横向位移相对材质参数变化尺度足够小；
2. 每个interface BSDF与phase都能给出一致的`sample/eval/pdf`；
3. local slab boundaries可由一维depth和side/layer ID描述；
4. stochastic BSDF estimate、directional PDF estimate与MIS所需randomness满足独立性；
5. scene renderer能接受带内部Monte Carlo noise的stochastic BSDF；
6. target quality允许用更多spp平均reference noise；
7. 对困难sharp/anisotropic/high-scattering states，bidirectional更多strategies带来的variance收益足以支付`O(n_i n_o)`成本。[P/S; I]

### 13.3 可迁移机制与不能迁移的部分

可迁移到当前研究的机制：

- 用独立 position-free oracle生成LayerStack的online query GT，并显式登记reference standard error；
- 保留native layer operations和ordered boundaries，而不是先拟合为简单analytic closure；
- 对teacher/reference同时提供`evaluate/sample/pdf`一致性审计；
- exact stochastic PDF与short-path approximate PDF分离：approximation只影响MIS proposal weight，不偷换ground-truth estimator denominator；
- 对困难states使用bidirectional reference作为variance-control候选，并以matched total work而非spp比较；
- 把normal-map non-reciprocity、refraction Jacobian和light-side adjoint correction写入reference contract。

不能直接迁移的部分：

- 无界random walk和quadratic endpoint pairing不能作为本项目实时neural evaluator；
- thin/local-flat assumption只定义当前LayerStack source family reference，不是所有native source materials的公共语义；
- Guo的stochastic `sample/pdf`不会自动被NLB/MetaLayer等learned evaluator继承；后者必须另有matched proposal；
- 2018 CPU render time无法预测Falcor/Slang单query GPU cost；
- approximate PDF的单场景视觉结果不能替代support、normalization、bias和tail-weight测试。

作为下游边界：

- **Neural Layered BRDFs**把Guo random walk当作exact/noisy/expensive layered target和runtime baseline，再把dense directional samples投影到neural latent；NLB没有继承Guo的动态path state或sample/pdf。[N downstream `fan-2022-neural-layered-brdfs.md`]
- **MetaLayer**用Guo bidirectional evaluator生成128-spp query targets与2048-spp render references，并以其作为stochastic baseline；MetaLayer sampler外置，不能把Guo sample/pdf归入learned representation。[N downstream `2023-metalayer.md`]
- **BSDF Importance Baking**在rough dielectric + homogeneous medium + diffuse bottom family上用Guo作reference/runtime baseline和transport-map teacher；其relative MSE对各方法自己的converged GT计算，不能当作共享absolute-GT排名。[N downstream `bai-2023-bsdf-importance-baking.md`]

### 13.4 与本项目 runtime contract 的关系

本项目合同要求 `evaluate()`返回bare linear `f`，`sample()`返回与同一proposal匹配的`f|cos|/pdf`，`pdf()`为solid-angle density，并要求runtime state/read/control flow静态有界。[N `docs/contracts/scattering_backend.md:3-5`] Guo的radiometric对象和sample tuple可作为数学reference；但其path length、memory traffic和branching不静态有界，所以部署类别只能是 `reference oracle`、teacher/query generator或proposal-control，不能直接成为neural material program。

当前experiment framework已经允许stochastic reference对同一query做多次independent `evaluation_samples`并平均bare `f`。[N `docs/research/experiment_framework.md:37-40`] Guo报告提示还需冻结：independent RNG streams、reference SE、sample/pdf support、RR/termination mode和paper-time/current zero-check revision，而不只记录“使用random walk”。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

Guo 2018不是NVIDIA neural materials 2024的架构规范，因此不能用它把当前encoder/evaluator/sampler差异标成NVIDIA论文复现偏差。直接fidelity分类为 `not-applicable`；下面只说明它对验证轨的影响。

| 主题 | Guo 2018 | 当前 NVIDIA functional reproduction | 分类与影响 |
|---|---|---|---|
| Evaluator表示 | dynamic stochastic path integral | z8 + fixed small MLP evaluator | `not-applicable`：Guo可作LayerStack GT，不定义NVIDIA architecture |
| Output measure | physical BSDF；sample weight含`f|cos|/pdf` | correspondence中runtime adapter由cosine-weighted训练路径恢复bare `f` | `interface-adaptation`仅属于当前NVIDIA自身paper/runtime ABI；Guo提供独立measure audit参考，不改变其判定 |
| Source domain | ordered thin interfaces/slabs | 当前LayerStack `1×1` source-domain adaptation | `not-applicable`：两者source假设可重叠，但NVIDIA的1×1 adaptation由其自身correspondence定义 |
| Sampler/PDF | exact random-walk proposal + independent stochastic PDF；可用approx PDF做MIS | learned bounded sampler，需matched `sample/pdf` | `not-applicable`：Guo可作为reference/proposal control；不能要求learned sampler逐path复现Guo |
| Runtime | unbounded CPU stochastic oracle | fixed `prepare/evaluate/sample/pdf` MethodBundle/Slang path | `not-applicable`：只在offline/on-GPU reference轨使用Guo，不进入deployment parity |
| Training lifecycle | 无训练 |当前functional reproduction完成200k而非论文完整300k | `not-applicable`：Guo不能闭合该`budget-adaptation`；只可提高GT/query audit质量 |

现有NVIDIA correspondence明确：200k functional reproduction未达到完整300k，evaluator有cosine→bare-`f` adapter，LayerStack是`1×1` source-domain adaptation，真实PT使用`prepare/sample/pdf/evaluate`路径。[N `archive/.../research/correspondence.md:13,37-40,46`] 本报告不新增NVIDIA `suspected-defect`。

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：独立Guo-style position-free oracle能暴露当前LayerStack reference的measure/boundary错误 | [P Eqs.2–8, §§4.4–4.6] | 两实现对同一restricted source states语义等价但code path独立 | current reference vs isolated Guo oracle；同states、directions和reference budget | interface/medium params、normal convention、RR、precision、seeds、sample count | bare-f差异CI、white-furnace、reciprocity/adjoint、sample/pdf identity、reference SE | reference oracle | 差异均落入双方MC CI且不发现系统性boundary/measure错误 |
| H2：interface-short-path approximate PDF可在不改eval/sample无偏性的前提下降低scene MIS成本 | [P §5.3, Fig.8; S-MIS] | 当前layer states的大部分directional mass由short interface paths解释 | exact stochastic PDF vs short-path+Lambertian PDF；同evaluator/sample、SPP与independent streams | scene/light、RR、support、MIS heuristic、reference、seeds | image bias CI、variance/time、tail weights、PDF normalization、support failures | bounded proposal/control | approximate无time/variance Pareto收益，或出现support hole/可测bias |
| H3：bidirectional position-free teacher在sharp/anisotropic multi-layer states下降低GT target variance | [P Fig.2; S-BD] | 当前hard-state distribution包含uni NEE难采的跨层paths | uni vs bi；matched total vertex/connection work和wall time，不按spp硬配 | source states、queries、precision、RR、random streams、reference | per-query SE、peak/grazing error、time、path-length tail、CI | reference query generator | bi在matched cost下CI不缩小，或`O(n_i n_o)`成本主导并更差 |
| H4：GT生成时独立估计stochastic BSDF与PDF比共享随机流更稳健 | [S-MIS] | 下游online query/MIS或sampler supervision会同时使用两者 | independent streams vs deliberately coupled streams；同total samples和proposal | states/queries、RNG quality、loss、model、training steps、seeds | target bias/variance、MIS bias、training seed variance、final G2 | data/query recipe | 独立分离对bias、variance和训练稳定性均无改善，且coupled estimator经证明仍满足条件 |
| H5：显式layer/media identity比递归nested reference更适合作为深层GT | [P §4.6, Fig.15(c)] | 当前G2s深层stack会重复遇到跨内部boundary paths | explicit multi-slab vs recursive binary nesting；使用同uni或同bi estimator的clean ablation | layer stack、proposal、RR、work budget、seeds、precision | bias CI、variance/time、boundary failures、depth scaling | reference oracle | clean ablation中explicit无质量/成本优势，或实现复杂度引入更多错误 |

H1/H3/H5只属于reference轨；H2是proposal/control；H4是query recipe。它们均不自动成为产品候选，也不放宽本项目static-bounded runtime约束。[N method constraints/runtime contract]

## 16. 证据索引

### `P` Main paper

- Abstract、§§1–2：目标、无预计算、与prior analytic/statistical/global Monte Carlo方法边界。
- §§3.1,4.2、Figs.5–6：small-displacement assumption，以及normalized slab和depth/direction path state。
- §4.2、Eqs.2–8、Fig.6：vertex/segment factors、path contribution、path-space measure和layered BSDF积分；已视觉核对。
- §4.3、Appendix A：RTE到position-free integral的变量变换、volume inverse cosine和exit cosine cancellation；已视觉核对。
- §§4.4–4.6：normal mapping、refraction reciprocity/adjoint、多slab与nested alternative。
- §§5.1–5.3、Figs.7–9：forward sample、uni/bidirectional evaluate、exact/approximate PDF和MIS。
- §5、Figs.1–2,4,8–15、Tables 1–2：实验、应用、timing、视觉结果；所有14页、公式、表、图、caption和脚注已渲染核对。
- §6.4：thin/local、large variations和optically thick/high-scattering限制。

### `S` Supplemental

- `S-MIS`，3/3页：stochastic function/PDF estimates进入MIS的无偏条件、independent randomness与expected partition of unity；公式和脚注已视觉核对。
- `S-BD`，3/3页：两端subpaths、每endpoint pair两种connection proposal、balanced weights、ratio recurrence由cubic-like naive work降到`O(n_i n_o)`；公式与图已视觉核对。
- `A-supp`：四组equal-time images、PDF/jade/magnifier sweeps、cloth文件名和reference/render timing context。

### `C` Official code/config

- current commit `ca6e9b19fb122c126d605207d9f4790e86b03651`；paper-time commit `3e414f1507a0a72896b03267c97195991519fbd7`；2019 source fix commit `cc763ca60ba84b4ee821b5620702d28835b390b2`。
- `src/bsdfs/multilayered.cpp:L26-L39`：release defaults。
- `:L236-L294`：spatial parameters、thread-local media和PDF-copy albedo handling。
- `:L296-L470`：analytic interface-plane traversal、boundary/refraction sampling和Jacobians。
- `:L473-L738`：`PathInfo`与forward path construction；current zero-PDF guards。
- `:L741-L956`：unidirectional NEE/local MIS。
- `:L958-L1193`：bidirectional subpaths、connections和ratio weights。
- `:L1195-L1365`：`pdfTRT`与exact/approx stochastic PDF modes。
- `:L1367-L1555`：`evalAndSample/pdf/sample/eval` entry points。
- `:L1573-L1667`：child interface/volume parameters和plugin state。
- `src/integrators/path/path_layered.cpp:L167-L224,L265-L316`：fused BSDF call、scene emitter/BSDF power MIS、global RR。
- `include/mitsuba/render/bsdf.h:L344-L449`：Mitsuba release sample/eval/pdf measure contracts。
- official scene ZIP SHA-256：Fig.2 `8F89DEFE1967A0FB0525BC5D08927DF53DA6D42268C3D3849DFC26072FCA42B1`；Fig.3 `AA944F1CC45BB6A76C4CBEE5C51B0C7580E2B4E369948AE5373F821C6DA77126`；Fig.8 `D6C82429DE9AC40FD602FAC619E4DF74ED00DEBEF77B18F12C4015D8204699F7`；Fig.12 bottom/top `BE539D9C1563102A06C1EE7C81CBAC45DA374F9D560B947B72646CAF363841FB` / `868B1A932DE2F7B8EF3D4D95E0D646EC75C6805E2F0973509403F5EFAA472D04`；Fig.14 `888931162E8CD9983DBDBF71DEC11D908CCF22904EB59B98B2D2071F6F24F409`；Fig.15 `4871606668B8DEEC858F89382098BE7E845F276A556E29A05170D0769FC6197F`。

### `A` Author material/correction status

- official project page：formal paper、supplement、code入口。
- official supplemental page：additional image comparisons和parameter sweeps。
- official project/repo/ACM入口未发现formal correction/errata；论文源码后续compiled PDF不作为修订版。

### `N` NeuralShading evidence

- `docs/contracts/scattering_backend.md:3-5`：bare linear `f`、solid-angle PDF、matched `sample()/pdf()`和static runtime contract。
- `docs/research/experiment_framework.md:37-40`：stochastic reference的independent repeated evaluation与average recipe。
- `archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md:13,37-40,46`：当前functional reproduction预算、cosine→bare-`f` adapter、LayerStack `1×1` adaptation与真实PT call path。
- downstream reports `fan-2022-neural-layered-brdfs.md`、`2023-metalayer.md`、`bai-2023-bsdf-importance-baking.md`：只用于说明Guo作为各自reference/baseline的边界。

### `I` Derived/transfer notes

- cloth约40×的数值恢复、Table 2“bidirectional不总更快”、runtime静态无界、`pdfTRT` array疑点和各迁移假设均为本报告分析，不是作者原句。
- code locator为locked current commit的静态审计；本author pass未build/run Mitsuba，不能宣称dynamic reproduction。

### 建议提升的 load-bearing 论文

- **Belcour 2018, Efficient Rendering of Layered Materials Using an Atomic Decomposition with Statistical Operators**：`key-baseline` + `failure-explanation`。它是Guo的主要fast approximate layered-material对照，也影响NLB/MetaLayer/importance-baking的sampler/operator边界。
- **Xia et al., Gaussian Product Sampling for Rendering Layered Materials**：`failure-explanation` + `runtime-transfer`。它直接针对position-free Monte Carlo在sharp/high-scattering配置中的proposal variance；需要确认正式年份/版本后锁定。
- **后续position-free variance/improved layered random-walk工作**：trigger为综合报告需要判断“Guo estimator本身的已知variance问题是否已被一手方法修复”；在此trigger前不从二手引用补写标题或结论。

独立复核结论：`evidence-reviewed`；source locked，main/supp 已由作者与 reviewer 分别完整视觉核对，official code/config 已静态审计。未闭合证据缺口继续保留，不把静态疑点写成动态复现事实。

## Evidence review

```text
author_worker: /root/rta2024
reviewer: /root
reviewed_at: 2026-08-29
sources_rechecked:
  - main PDF SHA-256 4A28A33938A4530D6FB17D7DE263238C97397CD0C03369517838E06ECA712375, 14/14 pages independently visually rechecked
  - supplemental MIS proof SHA-256 A2F83FF41BA15BDCF1D39439C6D7B011D1D0B2936E24F592C01AEA1B5564E30B, 3/3 pages independently visually rechecked
  - supplemental bidirectional details SHA-256 76BE30ECFF29EC3D67A95F034B77BBCE51214199572279086B106444690CFBAD, 3/3 pages independently visually rechecked
  - official current repo commit ca6e9b19fb122c126d605207d9f4790e86b03651 and paper-time commit 3e414f1507a0a72896b03267c97195991519fbd7 independently spot-audited at defaults, path/PDF modes and plugin boundaries
  - official Fig.8 configs independently rechecked: formal text diffuse term 0.1 versus mi_trt_middle.xml diffusePdf=1 remains an explicit conflict
findings_closed:
  - position-free path state, measures, vertex/segment factors and layer operations
  - forward sample, unidirectional and bidirectional evaluation, exact and approximate PDF semantics
  - stochastic MIS independence requirement and bidirectional weight complexity
  - formal timings, equal-time result scope and failure classification
  - paper-time/current source delta and post-release zero-PDF guard scope
  - downstream NLB/MetaLayer/BSDF Importance Baking reference boundary
  - corrected formal main-paper locators: §4 formulation, §5 estimators, §6 results/limitations, and Fig.10 white furnace
remaining_evidence_gaps:
  - no formal correction found; absence cannot prove that no unindexed author note exists
  - raw seeds, repeated-run metrics, reference variance and a complete immutable all-figure asset bundle are unavailable
  - formal approximate-PDF reflection/transmission bounds to the single release stochPdfDepth parameter remain underspecified
  - paper diffusePdf 0.1 versus an official Fig.8 XML value 1 remains unresolved
  - pdf=TRT ID-array suspected defect has no author confirmation or dynamic reproduction
  - exact paper build precision/compiler flags and per-figure source revision are unreported
  - code audit was static; Mitsuba was not built or executed in this author pass
review_status: passed-with-explicit-gaps
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
