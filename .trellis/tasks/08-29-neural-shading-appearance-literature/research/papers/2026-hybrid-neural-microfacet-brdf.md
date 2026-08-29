---
paper_id: "2026-hybrid-neural-microfacet-brdf"
title: "A Hybrid Neural-Microfacet BRDF Model for Real-Time Rendering"
authors: "Louis De Oliveira, Anastasia Karpova, Georges Nader, Antoine Houdard, Pierre Mézières, Damien Rioux-Lavoie, Romain Pacanowski"
year: "2026"
venue: "Computer Graphics Forum 45(4), EGSR 2026"
doi: "10.1111/cgf.70540"
report_status: "evidence-reviewed"
main_source: "https://doi.org/10.1111/cgf.70540"
supplemental_status: "available"
official_code_status: "audited"
official_code_commit: "e4d8991612c937faabdba204c31623e097f486ba"
author_worker: "/root"
reviewer: "/root/nbrdf2021"
last_verified: "2026-08-29"
---

# A Hybrid Neural-Microfacet BRDF Model for Real-Time Rendering

## 1. 研究对象与报告边界

论文提出一种 homogeneous measured BRDF 的 hybrid representation：每个材质保存可编辑的 GGX microfacet 参数 `p=(kd,η,α)` 与低维 latent `z`，一个 dataset-shared shallow MLP 输出正值 additive correction `fc` 与 `[0,1]` gate `fg`，最终直接求值：[P Eq.(1)–(3)]

```text
f_t(ω_i,ω_o) = f_c(ω_i,ω_o,z) + f_g(ω_i,ω_o,z) · f_a(ω_i,ω_o,p)
```

它不是把 source BRDF 先改写成 GGX 后再当 GT；measured BRDF 仍是训练 reference，GGX 是目标 representation 内部的 analytic core、fallback 与 sampling proposal。神经部分负责 GGX 无法表达的 residual。[P §3]

本文覆盖 312 个 measured BRDF：MERL 100、RGL 62、UTIA 150；每个 dataset 分别训练一套 shared MLP，不是一套网络跨三个 dataset。主体是 local BRDF evaluation、post-fit editing、analytic importance proposal 与 selective inference；不含空间纹理、filter/LOD、native graph compiler、BTDF 或 layered random-walk reference。[P §4]

这篇论文对本项目属于 `local-material` 的高相关 hybrid-prior candidate：其 `evaluate()` 静态有界，analytic proposal 可给出 `sample/pdf`，但 autodecoder fitting 与 homogeneous measured-BRDF domain 仍未解决“从任意源材质原生参数一次编译出 runtime program”。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---:|---|---|
| Formal metadata `P-M` | [DOI](https://doi.org/10.1111/cgf.70540)，Computer Graphics Forum 45(4)，online 2026-08-12 | 2026-08-29 | Crossref response 未持久化 | venue、DOI、作者、发布日期与 72 references |
| Formal VOR `P` | [Eurographics Digital Library item](https://diglib.eg.org/handle/10.1111/cgf70540)，bitstream `cgf70540.pdf`，13 页 | 2026-08-29 | SHA-256 `D117544DBC4C12976506D826456FC935345967620863B2644ABAC6E4E1422677`；DSpace MD5 `7f5c32641233df2dd9927bf5dc577e2d` | 方法、训练、正式图表、结果与限制的主证据 |
| Formal supplemental `S` | [Digilib bitstream `paper1049_mm_crc4.pdf`](https://diglib.eg.org/bitstreams/b4c86dbf-6488-4306-be63-5984920effca/download)，24 页 | 2026-08-29 | SHA-256 `E1B7D712B9858A479DC96B8FF36CF1293A9F226D846D11B72A797B32F754E453`；DSpace MD5 `733188dac2bb20b0d97e06551c6bc76e` | exact analytic formula、HardGELU、dataset、额外指标/图、sampling 与 scene details |
| Author copies `A-P/A-S` | [author main](https://ubisoft-laforge.github.io/world/hybridrdf/pdf/preprint.pdf)，13 页；[author supplemental](https://ubisoft-laforge.github.io/world/hybridrdf/pdf/preprint-supp.pdf)，24 页；[arXiv v1](https://arxiv.org/abs/2608.09604) | 2026-08-29 | main `D0428E7FC59FC459E8D9BFB408EC14BBD84B377F9C381ED203ED3E3F14F2244D`；supp `AACBBA1C5BD55CA01C05E260A6F27AB602713E2D4CCFCDBB0B5155D8F400C97F`；arXiv PDF `81BEA35A94A13490E00474DC9F18D6A29C8ADF1D323F22F4ED9BAA41BCD7E7FD` | 可用作者版本；三者与正式 bitstream 二进制不同，未把它们静默视为同一 artifact |
| Formal artifact `C-R` | [Digilib ZIP bitstream](https://diglib.eg.org/bitstreams/29fefcc5-dbea-4aab-b355-6f1b30240218/download)，331,702,103 B | 2026-08-29 | SHA-256 `6E2AD284B7EEAC04B8DF8A8C5D5FF88E027D01BFDA4A45D65D68E16AE097C13B`；DSpace MD5 `db1dd3ab1a25af2baccc268788fa95ec` | 3 个 BRDFExplorer executable、README 与 14,976 个 fitted `.brdf`；无训练源码或宿主 renderer 源码 |
| Author project/demo `A/C` | [Hybrid Neural BRDF project](https://ubisoft-laforge.github.io/world/hybridrdf/)；[GitHub Pages source](https://github.com/Ubisoft-LaForge/Ubisoft-LaForge.github.io/tree/e4d8991612c937faabdba204c31623e097f486ba/world/hybridrdf) | 2026-08-29 | path latest commit `e4d8991612c937faabdba204c31623e097f486ba` | WebGL2 progressive accumulation 的 one-hit BRDF viewer、162 个 fitted + 2 个 generic shader assets 与交互编辑；不是 formal training/BRDFExplorer source |
| Formal representative asset `C-RA` | `supplementary_data/shaders/MERL/Ours/32x3/ld4/Ours_ld4_32x3_alum-bronze.brdf` in `C-R` | 2026-08-29 | 29,847 B；SHA-256 `F335AB51ABAEAE9949732266C4D35686933D77BA09E0152837E30D23FCF205D0` | 核对 main `32×3/l4` topology、6-output transform、hybrid combine、GGX-only exported sample/pdf |
| Demo representative asset `C-A` | `shaders/MERL/Ours_ld4_32x2_alum-bronze.brdf` at `C` commit | 2026-08-29 | 21,237 B；SHA-256 `058318C7F62CC9188535E91C0AF7484355A1DCA29447FA6C437177BEB10A668A` | 与 `C-R` 同 hash 的正式 32×2/l4 asset；核对 demo evaluator 与 GGX sample/pdf |
| Video `A-V` | [author presentation video](https://www.youtube.com/watch?v=XaI17HUqSys) | 2026-08-29 | 未持久化 | 已确认存在；本报告没有用其补写正文未披露配置 |
| NeuralShading evidence `N` | [compiler contract](../../../../../docs/realtime_material_compilation.md)、[scattering backend](../../../../../docs/contracts/scattering_backend.md)、[model candidates](../../../../../docs/research/model_candidates.md)、[current NVIDIA config](../../../../../configs/learning/nvidia-rta2024-materialx-formal.json)、[method](../../../../../src/ncls/learning/methods/nvidia.py)、[model](../../../../../src/ncls/learning/models/nvidia_neural_appearance.py) | 2026-08-29 | current correspondence `nvidia-rta2024-functional-f@2` | 只用于 §14–15；不以归档旧 identity 代替当前复现 |

### 2.1 Source availability 结论

- Formal VOR、supplemental、artifact ZIP、project page 与 WebGL runtime source 均可得。`C-R` 的 14,976 个 `.brdf` 恰好覆盖 `312 materials × {Ours, Neural} × {16,32,64} × {2,3 hidden layers} × {l=4,8,12,16}`，因此 `32×3` 与 UTIA 导出物并非缺失；但 archive 只含 export 与 closed binary，不含 fitting/training 或 BRDFExplorer host source。
- GitHub Pages repo 只登记 100 MERL + 62 RGL fitted assets（再加 2 个 generic）；默认身份是 `Ours_ld4_32x2_*`。它能审计浏览器部署路径，但不能替代 formal 32×3 benchmark、UTIA 或 closed BRDFExplorer/CoopVec path。[C `assets.js`; C-A; C-R]
- GitHub demo 路径包含 `LICENSE.txt`（SHA-256 `E9502B582C33A6747415EF0BE569F5D7ACF2E725A107EB39CEE3B2B352CEA449`），为 research-only copyleft 且禁止 commercial use；`C-R` ZIP 本身没有 license file，不能假定两者授权自动相同。

### 2.2 Demo code/config manifest

以下 SHA-256 均锁定到 `C` path commit；它们是 Web demo 的代码/config identity，不是论文训练配置：

| path | SHA-256 |
|---|---|
| `index.html` | `DC07930690840FEFF9090B75BF059577B45D5893FFF0A3FCC902C4B9A4CD0A23` |
| `package.json` | `004C191E17DBDF6CE7B7D7C4CE018FEE95F9B8D90F1B2F62F9754E83804636F0` |
| `package-lock.json` | `D3DD9D4017789470164DFEC68E452EBEC30184ED5D6471C3179676380EB9C702` |
| `vite.config.js` | `363BD4DEA441485753A13C27502C017C7E0B55966DC45106EA9A5F1674F1433C` |
| `js/assets.js` | `AB62D76B284EBE86B666F2723193BA9C615AB691E02D1AF23D7D9611408B3CE6` |
| `js/brdf-parser.js` | `3437D773381C911F4F0186FE877B9584F5087137CDD184579C675DEBD436EE7D` |
| `js/main.js` | `9B912CBB74AA2AF18B5C55607033D1DD003DE7501AF143997E200AB5FCCEDA50` |
| `js/model-loader.js` | `AF0E5FB15257686952482EEDE4493AC7DB86E92ED45B6C131E9D636A4EDAEC41` |
| `js/renderer.js` | `DA492B615B73034B7536DF0D4D7A9DAB806B756F30D24636CD3A39E9925FAD0C` |

## 3. 原论文的问题、假设与贡献边界

作者列出的目标性质是：高 expressiveness、可编辑、低 per-material memory、fast evaluation 与 importance sampling。单一 GGX 快且可编辑，但对 iridescence、layering、diffraction 等复杂外观表达不足；fully neural 模型能拟合，却把主要 lobe 与 residual 一起交给网络，需更大 MLP、缺少物理参数，sampling 也需要额外网络或 proxy fitting。[P §1–2]

核心假设是：大多数 measured BRDF 的主要能量可由单 diffuse+GGX lobe 捕获，剩余误差比完整 BRDF 更容易由小 MLP 建模。为避免 joint optimization 把 GGX 变成任意 basis，训练同时最小化 analytic-only loss `La` 与 full-hybrid loss `Lt`。[P §3.1–3.3, Fig.3]

作者贡献边界：[P §1, §5]

1. positive additive correction + bounded multiplicative gate 的 hybrid formula；
2. joint fit shared MLP、per-BRDF latent 与 per-BRDF analytic parameters；
3. 同内存/MLP size 下与其实现的 Zeltner et al. neural baseline 比较；
4. 直接用 analytic component 的 cosine/GGX mixture 做 proposal；
5. 允许按 bounce 停用 neural correction 的部署策略。

论文没有证明一般 source material 都能被单-lobe GGX 分解；作者明确展示 iridescence 与远离 smooth single-lobe domain 的失败。[P §6, Fig.12]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source | measured RGB BRDF | MERL isotropic、RGL isotropic/anisotropic、UTIA anisotropic | [P §4; S §2] |
| Per-material analytic state | `p={kd,η,α}` | `kd∈[0,1]^3`；colored `η∈[1,10]^3`；isotropic `α∈[0,1]` 或 anisotropic `(αx,αy)∈[0,1]^2` | [P §3.2, Eq.(3)] |
| Per-material learned state | free latent `z` | `z∈R^l`；formal matched comparisons use `l=4` or `8` | [P §3.1, §4.1] |
| Runtime query | `ψ_w(ωi,ωo,z)` | `ωi,ωo` each Cartesian 3D；concatenate with latent, no positional encoding | [P §3.2, Fig.2] |
| Network output | `(fc,fg)` | `fc∈R_+^3` via exp；`fg∈[0,1]^3` via sigmoid | [P Eq.(2), Fig.2] |
| Evaluator output | `ft=fc+fg·fa` | RGB BRDF `f`，不含 cosine | [P Eq.(1)] |
| Analytic sampler | MIS between cosine lobe and fitted GGX lobe | continuous hemisphere PDF；精确 mixture weight/selection recipe 未报告 | [P §5.2] |
| Dataset coordinates | MERL 用 modified Rusinkiewicz 3D table；UTIA spherical；RGL adaptive/model-driven | 只用于 measured reference query；network runtime仍收 Cartesian directions | [S §2] |

论文没有说明 RGB units、negative measured values cleaning、direction validity mask 或 grazing clamp。official asset 的 analytic code对 `NdotL/NdotV` 与 PDF denominator使用小正数 clamp，但这属于 WebGL demo，不可静默回填为 formal training target。[C-A]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

```text
measured BRDF collection of one dataset
  → initialize per-BRDF p_i=(kd,η,α), z_i and shared MLP w
  → each step draw 1024 cosine-weighted (ωi,ωo) pairs
  → every BRDF f_i evaluates the same directional batch
  → La keeps analytic fa(p_i) close to f_i
  → Lt fits fc(z_i,w)+fg(z_i,w)·fa(p_i) to f_i
  → persist one shared MLP + per-BRDF p_i,z_i
  → runtime evaluate analytic GGX and one shallow MLP
```

为了加入未见 measured material，论文 sparse protocol 固定 `w`，只优化该材质的 `p,z`；它不是 encoder 一次前向得到 latent。[P §4.2, Table 3]

### 5.2 Analytic component

```text
fa(ωi,ωo;p) = kd/π + F(ωi,ωo,η) G(ωi,ωo,α) D(ωi,ωo,α)
                         / [4 (ωi·n)(ωo·n)]
```

`D` 是 isotropic/anisotropic GGX，`G=G1(ωi)G1(ωo)`，`F` 使用 exact unpolarized dielectric Fresnel with ambient IOR 1；supplemental 给出 Eq.(1)–(5)。colored `η` 让 specular tint 可编辑，但不表示 conductor complex IOR。[S §1]

作者试过更复杂的 Disney analytic core，观察到额外参数导致 optimization numerical instability，最终使用这套 diffuse+GGX formulation。[S §1]

### 5.3 Network topology

| 模块 | 输入 | 层/运算 | activation | 输出 | shared/per-material | locator |
|---|---|---|---|---|---|---|
| Hybrid MLP | Cartesian `ωi(3),ωo(3),z(l)` | `1–3` hidden layers，width `16–64`（正文早段写常用 16–32，实验含 64）；main configuration `32×3` | hidden HardGELU | final linear 6 | one shared MLP per dataset | [P §3.2, Fig.2, Tables 1/4; S §1] |
| Correction head | final RGB 3 | `exp(raw)` | exponential | positive `fc` | shared weights, conditioned by per-BRDF z | [P Fig.2] |
| Gate head | final RGB 3 | `sigmoid(raw)` | sigmoid | RGB gate `fg` | same | [P Fig.2] |

HardGELU 是分段二次近似：`0` for `x<-1.5`，`x` for `x>1.5`，中间为 `x(x+1.5)/3`。作者称它对 shallow MLP empirically better than ReLU 且便宜，但没有提供独立 activation ablation。[S §1; C-A]

`C-RA` 把 main config具体化为 `10=(ωi3,ωo3,z4) → 32 → 32 → 32 → 6`；前三层 HardGELU，末层 linear，前三维经 `exp`、后三维经 `sigmoid`。export 最后还做 `relu(fc+fg·fa)`，该操作不在 Eq.(1)，对于有限正值分支通常冗余，但仍应登记为 paper↔artifact safeguard，不反推为训练公式。[C-RA]

### 5.4 存储分界

- isotropic per-material analytic state按公式是 7 scalars，anisotropic是 8；再加 `l=4/8` latent，算术分别为 isotropic `11/15`、anisotropic `12/16`。论文图表统一把 matched columns标成 `12/16 params`，且称与 neural latent `12/16`同数，但 `C-RA` 的 MERL isotropic export确实只有一个 `alpha`，未见第 12/16 个 padding state；因此对 isotropic dataset不能把 column label升级为已闭合的 exact scalar count。[P §3.2, §4.1, Fig.4, Table 1; C-RA]
- `32×3` MLP 报告约 14 kB，一份在 dataset 内共享；precision/layout 未报告。[P §3.4]
- formal 与 WebGL `.brdf` 都把网络权重直接展开为 GLSL `mat*vec` 常量，每个材质文件重复内嵌 dataset-shared weights，因此 export layout 不体现论文理想的 shared storage。[C-RA; C-A]
- spatial maps、mips、filtering 与 quantization只是 §6 future discussion，未实现。[P §6]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| MERL | 100 isotropic measured BRDF；`90×90×180` modified Rusinkiewicz table | [P §4; S §2] |
| RGL | 51 isotropic + 11 anisotropic；adaptive resolution，underlying microfacet-model-driven samples | [P §4; S §2] |
| UTIA | 150 anisotropic；supplemental 原文明确印作 elevation resolution `115°`、azimuth `7.5°` | [S §2，p.2 visual] |
| Formal training unit | one separate shared network per dataset | [P §4] |
| Direction batch | 1024 `(ωi,ωo)` pairs per iteration，cosine-weighted hemisphere distribution | [P §3.3] |
| Per-step material coverage | Eq.(4) sum over all `N` BRDFs；正文写每个 BRDF都在该 batch求值 | [P §3.3, Eq.(4)] |
| Split | 主训练用全集；sparse ablation在 MERL 每种 material type 内各取 50% train/50% held-out | [P §4.2, Table 3] |
| BRDF metric queries | `10^6` direction pairs | [P Eq.(6); S §2] |
| Render metric | 512×512 sphere、Uffizi environment、1024 spp；reference为 measured BRDF | [P §4.1; S §2, §4] |

没有 data augmentation、noise model、source normalization、validation set 或 formal checkpoint selection。这里保留两个无法由 artifact 闭合的 supplemental 内部冲突：[S §2]

- MERL 被写成“5 families”，随后只列 `fabric/dielectric/metal/phenolic` 四类；main sparse protocol也只列这四类，不猜第五类。
- UTIA elevation resolution 印作 `115°`，数值与其 dense measured-BRDF语境明显可疑，但正式 PDF、作者 PDF与公开 artifact均未提供勘误/reader；不能擅自补成 `1.15°`。

## 7. Loss、optimizer 与训练 lifecycle

总目标对每个材质、每个 batch direction 等权求和：[P Eq.(4)–(5)]

```text
L = Σ_i Σ_ω [ La(f_i(ω),fa(ω;p_i)) + Lt(f_i(ω),ft(ω;p_i,z_i,w)) ]

La,t(f1,f2) = || log(1 + cosθ_i f1) - log(1 + cosθ_i f2) ||_2
```

log compression降低 dynamic range，incident cosine downweights high-energy grazing samples。`La` 保持 analytic state具有物理/编辑含义，`Lt`监督最终 hybrid output。[P §3.3, Fig.3]

| 项 | 正式配置 | locator |
|---|---|---|
| Optimized state | shared `w` + all per-BRDF `p_i,z_i` joint optimization | [P §3.3] |
| Optimizer | AdamW | [P §3.3] |
| Initial LR | `0.005` | [P §3.3] |
| Schedule | cosine decay，200k steps | [P §3.3] |
| Gradient clipping | clip norm over `0.01` | [P §3.3] |
| Batch | 1024 direction pairs × all BRDFs in selected dataset | [P Eq.(4), §3.3] |
| Hardware/time | RTX 5080；MERL 100 BRDF full train约 10 min | [P §3.3] |
| Initialization/seed/model selection | 未报告 | [P §3.3] |
| AdamW betas/epsilon/weight decay | 未报告 | [P §3.3] |
| Sparse/new-material fitting steps/time | 未报告 | [P §4.2] |

Figure 3 的 `La` ablation显示：带 `La+Lt` 时 analytic MAE `.0154`、full `.0020`；只用 `Lt` 时 analytic MAE `.0366`、full `.0026`。主要损害是 analytic parameters失去对应 final appearance 的能力，同时 final fit也略差。[P Fig.3]

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime evaluator | analytic diffuse+GGX一次 + shared shallow MLP一次 + `fc+fg·fa` | [P §3.4] |
| Formal GLSL benchmark | BRDFExplorer，environment map，100 spp，不用 IS，RTX 5080，Nsight | [P Table 4] |
| GGX-only frame cost | `0.03 ms` | [P Table 4] |
| Hybrid costs | 16×2 `.10`、16×3 `.13`、32×2 `.22`、32×3 `.37`、64×2 `.78`、64×3 `1.40 ms` | [P Table 4] |
| Neural baseline costs | `.16/.17/.30/.45/.98/1.59 ms`，对应同 size顺序 | [P Table 4] |
| Precision | training/formal benchmark precision未报告；release/demo `.brdf` 是文本 float constants，WebGL fragment path声明 `highp` | [P §3.4; C-RA; C-A; C `renderer.js`] |
| Shared bytes | 32×3约 14 kB | [P §3.4] |
| Per-material bytes | 12或16 scalars；实际 precision/packing未报告 | [P §4.1] |

### 8.1 Selective neural evaluation

1080p GPU path tracer、最多七 bounce；把 neural correction只启用到某个 bounce，之后换成 GGX-only。Table 5 未单独写 GPU 型号：[P §5.3, Table 5, Fig.11]

| neural enabled until | inlined | CoopVec | reported speedup |
|---:|---:|---:|---:|
| 0 bounce（GGX only） | 1.10 ms | — | — |
| 1 | 1.90 ms | 1.60 ms | 1.19× |
| 2 | 3.60 ms | 2.35 ms | 1.53× |
| 3 | 5.20 ms | 2.75 ms | 1.89× |
| 4 | 5.40 ms | 2.85 ms | 1.89× |
| 5 | 5.50 ms | 2.90 ms | 1.90× |
| 6 | 5.55 ms | 2.93 ms | 1.89× |
| 7 | 5.57 ms | 2.94 ms | 1.89× |

作者观察 correction在前两 bounce 后视觉影响很小。准确的偏差边界是：若 cutoff 后仍用与 GGX evaluator匹配的 sampler，Monte Carlo 对“cutoff 后改用 GGX”的**截断 transport model**仍可无偏；但该 model 已不再等于所有 bounce 都求值 full hybrid BRDF 的 reference，因此相对 full-hybrid transport存在系统性的 model/reference bias。这不是谎报 PDF 导致的 sampling bias。[P §5.3, Fig.11; I]

Fig.11 同一 scene 的标注 image MAE依次为：GGX-only `.052`、neural enabled through bounce 1 `.010`、through bounce 2 `.007`、through bounce 3 `.0068`；图中的 seven-bounce full evaluation只给视觉 reference，未标一个可并列的 MAE。[P Fig.11]

## 9. 实验 protocol、baseline、指标与结果

### 9.1 Baseline correspondence

“Neural”是作者重实现的 Zeltner et al. 2024 evaluator：shallow MLP + three learned shading frames；移除原 encoder，以 trainable per-BRDF latent替代。作者把 neural latent设为 12/16，并宣称与 hybrid column同 per-material count，MLP size相同。[P §4.1]

这形成作者意图中的 memory/capacity-matched BRDF regression baseline，但有两层边界：第一，它不是 Real-Time Neural Appearance 的 spatial latent texture、encoder、mip/filter、sampler与训练 lifecycle完整复现；第二，isotropic hybrid按已披露 state仅为 `7+l=11/15` scalars，而 Table/Fig统一标 `12/16`，未报告 padding/额外 state。结果不能写成对整个 NVIDIA system 的端到端超越，也不能把 MERL exact byte match视为已证事实。[P §3.2, Fig.4, Table 1; C-RA]

### 9.2 BRDF-space 与 render-space结果

MERL 32×3/l4 对 32×3/l12 neural 时，100 个材质有 88 个 BRDF-space SMAPE更低。[P Fig.5] 主文 Table 1 的全部 render MAE 如下；每个 cell 为 `Hybrid/Neural`，作者的 12-param column对应 hybrid `l=4`/neural `l=12`，16-param column对应 hybrid `l=8`/neural `l=16`，但 isotropic count gap见 §5.4/§9.1。reference是 1024 spp sphere/Uffizi measured-BRDF render：[P Table 1]

| MLP | MERL 12 | MERL 16 | RGL 12 | RGL 16 | UTIA 12 | UTIA 16 |
|---|---:|---:|---:|---:|---:|---:|
| 16×3 | `.0059/.0086` | `.0056/.0084` | `.0267/.0256` | `.0245/.0288` | `.0109/.0123` | `.0106/.0119` |
| 32×3 | `.0051/.0062` | `.0052/.0067` | `.0203/.0238` | `.0203/.0208` | `.0101/.0106` | `.0098/.0113` |
| 64×3 | `.0046/.0049` | `.0044/.0048` | `.0149/.0143` | `.0160/.0153` | `.0094/.0098` | `.0093/.0100` |

Table 1 同时给出 analytic GGX render MAE：MERL `.0099`、RGL `.0333`、UTIA `.0201`。正式例外不只一个 cell：RGL 16×3/12 params 以及 64×3 的 12/16 params 都是 Neural更低；其余表格 cell 是 Hybrid更低。[P Table 1]

Supplemental Table 1 的全部 image PSNR 如下，每个 cell仍为 `Hybrid/Neural`：[S Table 1]

| MLP | MERL 12 | MERL 16 | RGL 12 | RGL 16 | UTIA 12 | UTIA 16 |
|---|---:|---:|---:|---:|---:|---:|
| 16×3 | `39.3681/35.7522` | `39.7280/35.7744` | `35.5394/34.7826` | `36.1161/34.4680` | `38.1736/36.4343` | `38.4437/36.8398` |
| 32×3 | `40.5024/38.7944` | `40.6165/37.9961` | `37.5375/36.0590` | `37.5778/36.4861` | `38.9011/38.3525` | `39.1572/37.4699` |
| 64×3 | `41.1915/40.7123` | `41.5204/40.9319` | `39.1842/39.4303` | `39.0731/38.9614` | `39.6182/39.1176` | `39.7517/38.9291` |

Supplemental Tables 2–4 的全部 BRDF-space SMAPE如下；`12/16`沿用作者 column label，isotropic count gap不在此处暗中修正：[S Tables 2–4]

| MLP | MERL 12 | MERL 16 | RGL 12 | RGL 16 | UTIA 12 | UTIA 16 |
|---|---:|---:|---:|---:|---:|---:|
| 16×2 | `.2815/.6060` | `.2569/.4869` | `.6184/.7552` | `.5977/.7202` | `.0958/.1173` | `.0963/.1088` |
| 16×3 | `.2389/.3890` | `.2331/.3921` | `.6095/.6669` | `.5777/.6697` | `.0913/.1139` | `.0855/.0998` |
| 32×2 | `.2470/.3524` | `.2126/.4055` | `.5899/.6293` | `.5855/.6449` | `.0778/.1046` | `.0733/.1009` |
| 32×3 | `.2003/.2537` | `.1852/.2866` | `.5687/.6549` | `.5491/.6041` | `.0677/.0833` | `.0631/.0868` |
| 64×2 | `.2174/.2681` | `.1782/.2400` | `.5925/.5942` | `.5785/.6082` | `.0634/.0878` | `.0561/.0609` |
| 64×3 | `.1800/.2085` | `.1686/.2072` | `.5100/.5376` | `.5017/.5564` | `.0483/.0657` | `.0442/.0686` |

“300 materials tested”并不在 Table 1 caption，而在 §4.1 紧邻 Table 1 的正文；数据集明细与 §3.2 则给出 `100+62+150=312`。报告保留为 paper-internal count conflict，不把 300 改成正式样本数。[P §3.2, §4.1 prose, Table 1]

### 9.3 Sparse/new-material fitting

MERL 按四个已列 type 分层只用50%材质训练 shared MLP；对 held-out材质固定 `w`，只拟合 `p,z`。Table 3 的全部正式印刷值如下（32×3；BRDF-space两列与 render-space两列分开）：[P §4.2, Table 3]

| BRDFs in shared-network training | Evaluation subset | BRDF MAE | SMAPE | Render MAE | PSNR |
|---:|---|---:|---:|---:|---:|
| 100% | All | `.0031` | `.1759` | `.0087` | `37.594` |
| 50% | All | `.0041` | `.2104` | `.0099` | `36.659` |
| 100% | Train subset | `.0031` | `.2057` | `.0086` | `37.698` |
| 50% | Train subset | `.0032` | `.2049` | `.0088` | `37.530` |
| 100% | Held-out subset | `.0031` | `.1461` | **`.0893`** | `37.491` |
| 50% | Held-out subset | `.0050` | `.2160` | `.0110` | `35.788` |

加粗的 `.0893` 是 VOR 中的原值；它与 `PSNR 37.491`、正文“render-space degradation remains moderate”以及正文全体 MAE `.0087→.0099` 的量级关系冲突。author copy、formal VOR与公开 artifact均未给勘误；`.00893` 只是一个可能解释，不能当作事实替换。[P §4.2, Table 3]

### 9.4 Importance sampling

formal 方法层面在 cosine sampling 与 fitted GGX lobe sampling之间做 MIS；tabulated 对照按 1° angular resolution，每材质 5.62 MB。这里的 sampling proxy只决定方向与 PDF，BRDF evaluator始终是 fitted hybrid model；论文没有给 mixture selection probability、heuristic或完整 host implementation。[P §5.2, Fig.9–10; S Fig.24–26]

red-phenolic 结果：[P Fig.9]

- analytic IS, 1024 spp：MAE `.0042`，1.69 s；
- tabulated IS, 1024 spp：`.0035`，1.21 s；
- cosine, 1024 spp：`.0070`，0.84 s；
- cosine, 2048 spp：`.0050`，1.69 s；
- converged tabulated reference, 16k spp：19.32 s。

Fig.10 对 alum-bronze、blue-metallic-paint2、red-phenolic、two-layer-silver绘制 color-averaged absolute-difference variance；analytic IS明显接近 tabulated、优于 cosine，但没有发布 raw curve/CI。[P Fig.10; S Fig.24–26]

发布物与网页 demo 不能被误写成 formal sampler source：[C-RA; C-A; C `renderer.js`]

- formal 32×3 与 demo 32×2 `.brdf` 的 `sampleBRDF/pdfBRDF` 都只实现 fitted anisotropic GGX NDF proposal；cosine+GGX mixture若由 BRDFExplorer执行，只存在于闭源 executable，无法核对 mixture recipe。
- Web renderer没有加入 formal cosine proposal；它以 `0.5` 选择 `.brdf` 的 GGX proposal或 environment-map CDF，并用 balance heuristic组合两者的 PDF。因此这是 **GGX-vs-environment MIS**，不是论文的 **cosine-vs-GGX BRDF proposal**。
- Web path只做 primary ray/一次 surface hit后的 direct environment或point-light shading，并跨帧 progressive accumulation；它没有递归多-bounce transport，而且对 luminance `>100` 的 contribution做 clamp。故网页视觉结果不能作为 Fig.9–11 或 Table 4–5 的 correctness/timing artifact。

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释 | locator |
|---|---|---|---|---|---|
| `author-negative` | 更复杂 Disney analytic core | joint fitting出现 numerical instabilities | 参数更多、优化更难 | 解析 prior复杂度本身会增加 compiler optimization variance | [S §1] |
| `ablation-inferior` | additive-only neural residual | SMAPE MERL/RGL/UTIA `.4231/.7148/.0842`，显著差 | 没有 multiplicative modulation | 正值 additive alone难同时处理主 lobe比例变化 | [P Table 2] |
| `ablation-inferior` | log-multiplicative only | `.2186/.5907/.0708`，均差于 ours `.1974/.5446/.0644` | 缺少独立 positive correction | 两种 head承担不同残差形态 | [P Table 2] |
| `ablation-inferior` | 把 `log fa` 额外输入同 topology | `.2015/.5493/.0621`；MERL/RGL略差，UTIA略好 | 没有 notable overall difference，故省略 | 不是普遍失败；UTIA单项反而略优 | [P Table 2] |
| `ablation-inferior` | 不用 analytic loss `La` | analytic MAE `.0154→.0366`，full `.0020→.0026`，editability弱 | p不再需贴近目标 | `La`是 representation-role constraint，不只是aux loss | [P Fig.3] |
| `known-limitation` | iridescent / non-smooth / multi-lobe BRDF | current model struggle，Fig.12出现结构误差 | strong single-GGX-lobe bias | 对复杂layer stack可能把容量瓶颈锁死在analytic core | [P §6, Fig.12] |
| `known-limitation` | latent editing | 大范围编辑无保证 | p与z未做 disentanglement | 不能把小范围图示升级成native edit contract | [P §5.1, §6] |
| `known-limitation` | physical validity | 未强制 energy conservation/Helmholtz reciprocity；empirical reciprocal abs error多数 `<1e-4`，energy相近或略低 | 未来加prior/regularization | positive output不等于physical BRDF | [P §6] |
| `known-limitation` | iridescent test | 16×3/l4 MAE `.1350`，32×3/l4 `.1136`；加宽仍保留明显结构误差 | single-GGX-lobe bias | 不是“网络稍大即可消失”的容量问题证据 | [P Fig.12] |
| `paper-code-gap` | formal artifact vs training | `C-R`发布完整 32×3/UTIA 等 exports，但无training/host/CoopVec source、checkpoint或raw metrics | 未说明 | 可核 evaluator export，不足以复现优化与正式宿主路径 | [C-R; C-RA] |
| `paper-code-gap` | formal formula vs exported evaluator | asset用 final `relu(fc+fg·fa)`；η slider为 `[1,5]`（paper state `[1,10]`），latent slider `[-3,3]` | 未说明 | ReLU通常对正项冗余但仍是额外 safeguard；UI range不能回填训练约束 | [P Eq.(1)–(3); C-RA] |
| `paper-code-gap` | isotropic 12/16 parameter label | formal p为7 scalars且 `C-RA` 只有one-alpha；加l4/l8为11/15，但表列写12/16并称matched | 未说明padding/额外state | MERL exact memory match未闭合；RGL/UTIA anisotropic 8+l算术可达12/16 | [P §3.2, Fig.4, Table 1; C-RA] |
| `paper-code-gap` | formal MIS vs public sample/pdf | paper为cosine+GGX MIS；`.brdf`只给GGX，Web host改为GGX+environment MIS | host职责未说明 | proposal边界不能从可运行demo反推formal recipe | [P §5.2; S Fig.25; C-RA; C] |

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Analytic model | diffuse+GGX，p范围 | exact D/G/F Eq.(1)–(5)，Disney instability | representative GLSL实现 anisotropic GGX/exact Fresnel | 基本对应；demo含额外 clamp |
| Architecture | 1–3 hidden，16/32/64，main 32×3 | HardGELU | `C-R`含全部 size/depth/latent export；32×3/l4为10→32→32→32→6、exp/sigmoid heads | topology对应；export多 final ReLU，且权重按材质文件重复 |
| Data | 100+62+150 | database resolution与metrics | `C-R`含312材质全配置；Web只列100+62 | export coverage闭合；GT/raw data与split identity仍不可恢复 |
| Training | Eq.(4)–(5)、AdamW/LR/schedule/clip | 无更多optimizer细节 | 无training code/config/checkpoint | seed/init/weight decay/sparse fit lifecycle不可恢复 |
| Sampling | cosine+GGX MIS | extra variance/renders明确是两种lobe | `.brdf`只实现 GGX `sampleBRDF/pdfBRDF`；closed Explorer可能负责cosine mix；Web renderer另与 env CDF做MIS | release可核GGX proposal，formal mixture仍不可审计；demo不是formal path |
| Runtime | GLSL/CoopVec tables | path scene assets说明 | `C-R`给inlined GLSL与closed Explorer；Web为one-hit viewer；无CoopVec source | 可审计export公式，不能复现Table5 hardware path |

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. latent不可解释、远距离编辑无保证；[P §6]
2. 不强制 reciprocity/energy conservation；[P §6]
3. single-lobe GGX bias难表达 iridescence、multi-lobe与非平滑角向效应；[P §6, Fig.12]
4. 更丰富 analytic core难优化，扩大 MLP又增加 runtime；[P §6; S §1]
5. SVBRDF、quantized spatial maps与LOD只是 future direction。[P §6]

### 12.2 未报告/材料不可得

- formal training code、initialization、seed、AdamW完整参数、model selection；
- sparse/new-material fit steps、时间与停止规则；
- 14 kB 的precision/parameter-count口径、CoopVec topology/precision；
- isotropic hybrid `7+l=11/15` 与作者 Table/Fig `12/16 params` column之间的padding/计数口径；
- Table 5 GPU型号、scene resolution以外的path-tracer配置、统计聚合；
- cosine/GGX mixture weights、heuristic与formal host `sample/pdf`代码；
- raw metric samples、confidence interval、negative/invalid measured-value处理；
- `.0893`、300/312、MERL five/four families 与 UTIA `115°` 的官方勘误；
- artifact ZIP自身的授权条款（GitHub demo license不能自动外推到Digilib archive）。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

表示容量由三个可独立调节的 reservoir 构成：`p`承载 dominant lobe与edit axis，`z`选择dataset-shared residual behavior，`w`存共享 residual basis。与纯 MLP 相比，它把已知的 diffuse+single-GGX structure硬编码进 evaluator，因此 small network不必重复学习主峰；这也解释了小网络优势随width增加而缩小。[P Fig.4]

代价是 capacity placement同时变成 bias placement：当 GT 不接近 single lobe，`La`仍要求 analytic component追随它，且 `fg≤1` 只能 attenuate analytic，`fc>0` 再补回缺失结构。这个正值分解稳健，但可能需要 neural branch以“先压后补”的方式绕过错误 prior。

### 13.2 成功所依赖的假设

- source family多数 query由 diffuse+one GGX lobe解释；
- per-dataset shared residual manifold足以覆盖材质差异；
- cosine-weighted training虽降低grazing权重，仍有足够samples学会高能峰；
- `p,z` joint optimization不会发生严重non-identifiability；
- offline per-material fitting可接受，且runtime只需要homogeneous state。

MERL/RGL上的收益与 UTIA较小收益说明 prior好坏由dataset决定；Fig.12则给出明确反例。

### 13.3 可迁移机制与不能迁移的部分

可迁移：

- optimized analytic core + positive bounded neural correction；
- 用独立 `La`维持 proposal/edit parameters 的可解释性；
- analytic sampler作为无额外网络的低成本 proposal；
- `32×2/32×3` HardGELU作为shader-budget candidate；
- shared network + per-material state，适合多实例 amortization。

不能直接迁移：

- per-material free `p,z` fitting不能替代本项目从原生source参数/graph生成runtime state的 compiler；
- homogeneous measured BRDF没有spatial footprint/filter语义；
- 按bounce关掉correction会改变 transport 中被求值的材料函数；它可对自己的 cutoff model保持 sampler-unbiased，但相对full-hybrid reference有 model bias，必须作为独立runtime identity；
- fitted `p`的editability只在moderate excursions有图示支持，不能保证source-native editable parameters语义。

### 13.4 Runtime contract

`f_t`是bare linear RGB BRDF，形态上与项目 `evaluate()`一致；analytic cosine/GGX mixture可以静态实现一组彼此 matching 的 `sample()/pdf()` proposal。proposal不必等于full hybrid evaluator，但返回 PDF必须是**实际抽样分布**且 support完整，此时 path estimator 对所求值 evaluator仍可无偏，差异只影响variance。formal论文的任意 MIS query能力并不授权把 hybrid evaluator值或另一个 lobe PDF谎报成 sampling PDF。[N scattering contract]

部署上最合理的身份是 `hybrid evaluator + analytic proposal` product candidate；需要在本项目 source family上重新证明：

1. `fa`没有吞掉原生语义；
2. `fc/fg`在grazing、层状多峰、RGB energy上足够；
3. `sample/pdf` parity、normalization、support与white furnace通过；
4. shared network+compiler encoder替代per-material autodecoder后仍保持质量。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

| 轴 | 当前 NVIDIA functional reproduction | 本文 | 分类/影响 |
|---|---|---|---|
| Evaluator | z8 + two learned frames + `3×64` MLP，直接输出response | diffuse+GGX + raw Cartesian/z + 1–3×16/32/64 MLP | 新候选，不是 NVIDIA faithful modification |
| Per-material state | spatial hierarchical z8，由native encoder/materialized texture生成 | homogeneous free `p,z`，12/16 scalars | `interface-adaptation`需要compiler/texture extension |
| Sampler | separate `3×32→9` learned two-lobe analytic proposal | evaluator的 GGX core直接当 proposal，无额外 sampler NN | 可减少runtime sampler成本，但proposal expressiveness更低 |
| Training | GPU online source queries；encoder→latent finetune；evaluator+sampler joint | measured dataset batch；shared autodecoder joint fit；200k | lifecycle不可互换 |
| Formal comparison | 当前实现identity含完整RTA contract | paper baseline只保留3-frame evaluator并用free latent替encoder | 论文结果不能当当前end-to-end implementation ranking |

最有价值的 matched comparison 是在相同 source/query、per-material bytes、MLP MAC、precision 下比较：当前 direct neural evaluator vs `GGX core + residual/gate`。sampler暂时都使用各自最自然且诚实的analytic proposal，并把 evaluator quality 与 PT variance分开报告。

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| `H-HYB-1`：单GGX core能让shader-budget evaluator在层栈主域更省容量 | small MLP在MERL/RGL多数matched cells优于neural baseline | 当前层栈主峰也能被one-GGX近似 | 32×2/3 hybrid vs equal bytes/MAC direct MLP；同compiler encoder | source/query/split/steps/seeds/precision/total bytes/MAC | directional/energy/grazing errors、latency、bootstrap CI | `product-candidate` | matched budget无显著收益，或G2/G2s多峰显著退化 |
| `H-HYB-2`：`La`可保持analytic proposal有用而不损害full fit | Fig.3与Table2 | current native domains允许stable joint fitting | full objective vs remove/anneal `La`，其他完全一致 | init、data、loss transform、steps、seed set | evaluator error、proposal KL/ESS、p stability、edit response | `training-ablation` | `La`不改善proposal或明显压低full-model quality |
| `H-HYB-3`：analytic proposal可替代当前sampler network | Fig.9–10接近tabulated，且无需额外inference | current evaluator energy主要在core lobe，cosine mixture保support | hybrid analytic proposal vs current learned GGX9 sampler，same evaluator | evaluator、queries、spp、integrator、hardware | variance/ESS、sample/pdf parity、latency/state bytes | `sampler-candidate` | variance显著恶化或support/normalization失败 |
| `H-HYB-4`：positive correction/gate比unconstrained residual更稳定 | additive/log-multiplicative均劣于combined | 同结构在online noisy reference上仍可优化 | combined heads vs signed residual，params/MAC matched | dataset、optimizer、loss、init/seeds | failure rate、quality CI、energy/reciprocity | `capacity-diagnostic` | seed robustness或quality无改善，且gate造成prior绕行 |
| `H-HYB-5`：compiler encoder可替代per-material free latent | sparse held-out只拟合p,z仍保持render quality | native parameters足以预测/初始化p,z | autodecoder p,z vs encoder-predicted p,z + equal finetune budget | shared MLP、source split、query budget、state bytes | G1/G2/G2s、compile time、workflow W | `compiler-candidate` | encoder在unseen state明显低于autodecoder且有限finetune无法恢复 |

按bounce停用neural correction不作为上述候选的默认优化；若研究，必须另立 cutoff-transport identity并报告相对full hybrid model的 reference bias，而不是作为无损加速；其 sampler correctness则另按该 cutoff model的真实 proposal/PDF验证。

## 16. 证据索引

- `P`：Digilib VOR SHA-256 `D117...2677`；`P §1–2`为问题、目标性质与 related-work边界。
- `P §3.1–3.4, Eq.(1)–(5), Fig.2–3`：hybrid formula、analytic core、network、joint loss、runtime分界。
- `P §4, Fig.4–6, Tables 1–4`：datasets、baseline correspondence、quality、ablation、sparse与raw cost。
- `P §5, Fig.7–11, Table 5`：editing、analytic sampling、selective inference。
- `P §6, Fig.12`：latent/physical/single-lobe限制。
- `S`：Digilib supplemental SHA-256 `E1B7...E453`；`S §1–2`为 exact GGX/Fresnel、HardGELU、Disney instability、dataset sampling及两处未闭合文本冲突。
- `S Tables 1–4, Fig.6–26`：额外PSNR/SMAPE/render/sampling证据。
- `S §8`：path scene material provenance。
- `C-R`：Digilib release ZIP SHA-256 `6E2A...C13B`，14,976 exports与closed executables；README SHA-256 `5136B0A0...CF96`。
- `C-RA`：formal representative 32×3/l4 SHA-256 `F335...05D0`；main topology、heads、final ReLU与GGX-only exported sample/pdf。
- `C commit e4d899...`：WebGL2 one-hit runtime、asset registry、env MIS与`.brdf` loader；`C-A`为32×2/l4 asset SHA-256 `0583...668A`。
- `N`：§14–15的当前runtime/compiler/sampler边界。

## Evidence review

```text
author_worker: /root
reviewer: /root/nbrdf2021
reviewed_at: 2026-08-29
sources_rechecked:
  - formal Digilib VOR 13/13 pages, SHA-256 D117544D...1422677
  - formal Digilib supplemental 24/24 pages, SHA-256 E1B7D712...754E453
  - author main/supp and arXiv v1 version boundary
  - formal Digilib artifact ZIP manifest and representative 32x3/32x2 exports
  - project page and GitHub Pages path commit e4d8991612c937faabdba204c31623e097f486ba
  - demo code/config/asset hashes and current NeuralShading NVIDIA correspondence
findings_closed:
  - promoted formal VOR/supplemental and artifact bitstreams to primary locators
  - corrected UTIA 1.15-degree invention back to unresolved printed 115 degrees
  - corrected Table 1 locator: 300 appears in section 4.1 prose, while formal dataset total is 312
  - preserved Table 3 printed .0893 without inventing .00893 and transcribed all Table 3 cells
  - preserved supplemental five-family/four-listed-family conflict
  - established formal release coverage: 14976 exports include 32x3 and UTIA; training and host source remain absent
  - separated formal cosine+GGX MIS, exported GGX-only sample/pdf, and Web GGX+environment MIS
  - corrected Web demo scope to progressive one-hit viewer and recorded contribution clamp
  - corrected license claim and separated GitHub demo license from Digilib ZIP license gap
  - exposed isotropic 7+l=11/15 arithmetic versus author 12/16 matched-column label instead of inventing padding
  - classified bounce-selective correction as reference/model bias, not automatically sampling-PDF bias
  - updated current NVIDIA N locators and kept inference/transfer claims after factual evidence
remaining_evidence_gaps:
  - formal training/BRDFExplorer-host/CoopVec source、raw metrics与checkpoints未公开
  - AdamW defaults/init/seed/model selection、sparse fit lifecycle、formal precision未报告
  - isotropic hybrid 11/15 active scalars与12/16 column label之间的padding/计数口径未报告
  - Table 3 held-out MAE .0893、300/312、five/four families、UTIA 115-degree conflicts无官方勘误
  - formal cosine+GGX MIS mixture recipe/host source未公开；exports只提供GGX lobe sampler
  - Digilib artifact ZIP没有单独license file，不能从GitHub demo license外推
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
