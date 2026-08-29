---
paper_id: "2023-metalayer"
title: "MetaLayer: A Meta-Learned BSDF Model for Layered Materials"
authors: "Jie Guo, Zeru Li, Xueyan He, Beibei Wang, Wenbin Li, Yanwen Guo, Ling-Qi Yan"
year: "2023"
venue: "ACM Transactions on Graphics 42(6), Proceedings of SIGGRAPH Asia 2023"
doi: "10.1145/3618365"
report_status: "evidence-reviewed"
main_source: "https://github.com/lingqiyan/lingqiyan.github.io/releases/download/ucsb-archive/publications__paper_siga23metalayer.pdf"
supplemental_status: "available"
official_code_status: "unavailable"
official_code_commit: "not-applicable"
author_worker: "/root/taming2026"
reviewer: "/root/nbrdf2021"
last_verified: "2026-08-29"
---

# MetaLayer: A Meta-Learned BSDF Model for Layered Materials

## 1. 研究对象与报告边界

MetaLayer 研究的是**参数式局部层材质的前向编译**：把一个单层介质、两个界面的物理参数 `Γ` 输入 MetaNet，一次前向生成 BSDFNet 的少量材质专属权重与中间特征；随后 BSDFNet 接收方向对并输出该材质的反射或透射 BSDF 值。它属于 `local-material`，不是 scene-level transport；场景 path tracing 只负责验证编译后的局部散射函数。[P §§3–5, Figs.2–3]

本文“layered material”的正式 domain 比标题中的 arbitrary 更窄：当前实现只有一个均匀介质层、上下两个各向同性 GGX 界面，介质相函数为 isotropic；bottom interface 分 conductive 与 dielectric 两族，后者的 reflection 与 transmission 由两个独立模型处理。多于一层、各向异性、实测材质和能量严格约束均不是已实现能力。[P §§3.3,3.6,5.2,6]

本报告覆盖正式 15 页正文、9 页 supplemental、作者主页入口及其官方附件。作者没有发布可审计的训练、Mitsuba plugin、AVX-512 evaluator、sampler、dataset 或 checkpoint，因此本文对网络图内部不一致之处保留为 source gap，不从常见 MLP 实现反推唯一代码。[A homepage entry; C absence audit]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | 作者主页的 [Paper](https://github.com/lingqiyan/lingqiyan.github.io/releases/download/ucsb-archive/publications__paper_siga23metalayer.pdf)，[legacy author PDF](https://sites.cs.ucsb.edu/~lingqi/publications/paper_siga23metalayer.pdf)，[DOI](https://doi.org/10.1145/3618365) | 2026-08-29 | SHA-256 `FCFEDC1307420C0734683AEADA8E04EFBA67A958E710937A3F9DFF5880D57DE5`；主页 release asset 与 legacy author PDF 同 hash | 正式方法、训练、数据、sampler 与主实验；15 页 |
| Supplemental archive `S` | [author supplementary](https://github.com/lingqiyan/lingqiyan.github.io/releases/download/ucsb-archive/publications__supplementary_siga23metalayer.zip) | 2026-08-29 | ZIP SHA-256 `6B321B270562E365CB0B96B967A8C605B63B285D8059B8521EA992E142091955` | 梯度推导、编辑外推网格、NBRDF 与编码补充图 |
| Extracted supplemental `S` | `MetaLayer_ supplemental.pdf`，9 页 | 2026-08-29 | SHA-256 `8013134A39CD3DBD6342742D1F8F2CA08B9A6055A9D6A3ACEF2EA085D5CAB510` | 对 ZIP 内唯一文件做逐页视觉核对 |
| Author page/video `A` | [Lingqi Yan publication entry](https://lingqiyan.github.io/) 2023 年 MetaLayer 条目；[Video](https://github.com/lingqiyan/lingqiyan.github.io/releases/download/ucsb-archive/publications__video_siga23metalayer.mp4) | 2026-08-29 | 网页明确只列 `Paper / Video / Supplementary`，三者均解析到 `ucsb-archive` release asset | 论文身份和附件真实可用性；视频只核对入口，不用于补写配置 |
| Official code/config/data `C` | 作者主页该条目无 `Code`；GitHub 以正式标题、`MetaLayer BSDF`、`siga23metalayer` 定向检索 | 2026-08-29 | `not-applicable` | 未发现官方 repository、commit、config、dataset、checkpoint 或 renderer plugin；不能做 paper↔code 对照 |
| NeuralShading evidence `N` | [prior art](../../../../../docs/research/prior_art.md)、[model candidates](../../../../../docs/research/model_candidates.md)、[runtime contract](../../../../../docs/realtime_material_compilation.md)、[scattering ABI](../../../../../docs/contracts/scattering_backend.md)、当前 `configs/learning/nvidia-rta2024-materialx-formal.json` 与 `src/ncls/learning/{methods/nvidia.py,models/nvidia_neural_appearance.py}`；[archived NVIDIA correspondence](../../../archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md) | 2026-08-29 | repo-local；当前 `functional-f@2` 与 archived `functional@1` 分开解释 | 只用于 §§13–15，不回填为 2023 论文事实 |

正文 title block、正式引用与每页页眉均一致列作 `ACM TOG 42(6), Article 221, 15 pages`；DOI 为 `10.1145/3618365`。没有用未锁定的二手书目 article number 改写这一一手身份。[P PDF pp.1–15; A DOI]

## 3. 原论文的问题、假设与贡献边界

### 3.1 作者定义的问题

一般 layered BSDF 要处理界面间和介质内的多次散射。adding-doubling/矩阵方法能在运行时快速求值，但每材质预计算和存储昂贵；position-free Monte Carlo 方法更通用、无偏，却在 evaluator 内继续采样，成本和方差随介质散射增加。已有 neural BRDF 则常在容量、运行时成本、未见材质泛化和物理参数编辑之间取舍。[P §§1–2]

### 3.2 方法假设

作者采用 hypernetwork 假设：同一参数族的 layered BSDF 可以共享 BSDFNet 的大部分权重，而材质差异只需要 MetaNet 从 `Γ` 生成一小组逐层条件：

\[
M(\Gamma;\Theta_M)=\Theta_F^*,\qquad
F(\omega_i,\omega_o;\Theta_F)\approx f_s(\omega_i,\omega_o,\Gamma).
\]

这里 `Θ_F*` 只是 BSDFNet 参数的子集；其余 `W+、b+` 在训练后由所有材质共享。MetaNet 还直接生成若干 `v*` 中间特征，因此它不是“生成整个 BSDFNet”，也不只是生成一个输入 latent。[P Eqs.1–5, §§3.1–3.3]

### 3.3 作者列出的贡献

1. 由 BSDFNet 与 MetaNet 构成的 material-parameter→neural-BSDF compiler，可一次前向处理训练参数族内未见材质并直接编辑物理参数；
2. 先 joint training、再按材质交替更新 BSDFNet/MetaNet 的两阶段训练；
3. Rusinkiewicz half/difference coordinates 上的 spherical harmonics encoding；
4. Mitsuba 中 CPU/AVX-512 evaluator 与一个 Belcour-style multi-lobe analytic sampler。[P §1 contributions, §§3.4–4]

“低方差”在本文中来自用确定性 BSDFNet 替换 Guo reference 的**随机 evaluator**，不是证明神经表示无误差，也不是新 neural importance sampler 的贡献。[P §§3.1,4,5.1–5.2]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | top/bottom interface roughness与 dielectric IOR 或 conductor Schlick `R0`；介质 `σt,ρ` | 每个 RGB 通道独立调用时 `Γ∈R^6` | [P §3.3, Fig.3, Table 1] |
| Runtime direction query | normalized incident/outgoing directions `ωi,ωo` | BRDF 为反射半球；BTDF 扩至 whole-sphere elevation | [P §§3.1,3.4,3.6] |
| Coordinate transform | `ωi,ωo→ωh,ωd`，分别做 SH encoding | 两个 84D encoding，合计 168D | [P Eq.6, §3.4] |
| BSDFNet output | 单通道 reflectance/transmittance BSDF value | scalar；RGB 独立求值 | [P §§3.1–3.3; S footnote 1] |
| Model routing | reflective/conductive base 共一个模型；transmissive/dielectric base 的 BRDF、BTDF 分开训练 | 至少三个族级 trained models | [P §3.6] |
| Validity/domain | 单均匀层、两个 isotropic GGX interfaces、isotropic phase | 不含 anisotropy、arbitrary layer count、measured material | [P §§3.6,6] |

论文把 target 写作实际 BSDF value `f_s`，没有把几何余弦写入 Eq.2、loss 或 dataset recipe；但也没有给出 radiometric unit、clamp/positivity rule 或 BTDF 的 eta measure convention。故本报告不把“raw linear `f`、不含 cosine”之外的更强接口假设归给作者。[P Eqs.2–3,7–8]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

```text
physical layer parameters Γ (one scalar-channel at a time)
  → MetaNet M: 6 → 16×256 → {v*, W*, b*}
  → generated state: 4×32 + 5×32 + 5 = 293 scalars

(ωi, ωo)
  → Rusinkiewicz (ωh, ωd)
  → raw direction + SH bases = 168D
  → BSDFNet with shared {W+,b+} and generated {v*,W*,b*}
  → one BRDF/BTDF scalar
  → repeat independently for RGB
```

对中间层，作者把激活拆成三块：`v*` 由 MetaNet 直接产生；`v+` 由 shared affine branch 产生；`v≀` 由 material-specific affine branch 产生：

\[
v_{i+1}^{+}=a\left(W_i^{+}(v_i^*\oplus v_i^+\oplus v_i^{\wr})+b_i^+\right),
\]

\[
v_{i+1}^{\wr}=a\left(W_i^*(v_i^+\oplus v_i^{\wr})+b_i^*\right).
\]

Fig.3 中 composite hidden state 为 `32 v* + 31 v+ + 1 v≀`。每个 predicted `v≀` 只需要一行 32 weights 与一个 bias；这就是 material-specific weight-generation 被压到每层一个 neuron 的关键，而非复制全连接矩阵。[P Eqs.4–5, Fig.3]

### 5.2 持久化表示

| 内容 | shared / per material | 正式大小 | locator |
|---|---|---:|---|
| BSDFNet `W+,b+` | 跨材质共享 | BSDFNet 总体“over 13K parameters”；shared 子集未单列 | [P §3.2] |
| Direct features `v*` | 每材质、每通道 | `32×4=128` | [P §3.2] |
| Predicted affine rows `W*` | 每材质、每通道 | `32×5=160` | [P §3.2] |
| Predicted biases `b*` | 每材质、每通道 | `5` | [P §3.2] |
| Aggregate generated state | 每材质、每通道 | `293` scalars | [P §3.2] |
| RGB aggregate `[P-derived]` | 每材质 | `3×293=879` scalars；论文未报告量化 | [P §§3.2–3.3,5.3; arithmetic] |
| SH lookup table | shared runtime resource | resolution/precision/bytes 未报告 | [P §4 Renderer Integration] |

正文在 §5.3 写“293D vector for each material”，但同页声明 RGB 独立；其 512×1024 texture 约 1.7 GB 的例子只有按 `512×1024×293×3×4 bytes≈1.72 GiB` 才能复现。因此 293 更可靠地解释为**每通道**生成状态，完整 RGB texel 是 879 FP32 数；这是由作者数值交叉核验出的范围，不是代码事实。[P §§3.2,5.3, Fig.16; P-derived arithmetic]

### 5.3 网络逐层配置与未解析的正文冲突

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| MetaNet | 6 physical scalars/channel | 16 hidden layers，每层 256；293-vector 按计数分为 `128 v* +160 W* +5 b*`，论文未声明独立 output heads | **未单独报告** activation、normalization、output constraint | 293 | shared compiler；output per material/channel | [P §§3.2–3.3, Fig.3] |
| BSDFNet（正文 prose） | 168D encoded directions + generated state | “five hidden layers, 64 neurons each except the last 32” | ReLU；normalization 未报告 | scalar | shared weights + generated partial state | [P §§3.2,3.4] |
| BSDFNet（Fig.3） | 图中以 `ωi,ωo` 和紧凑 input bar 起始，未显式画出 §3.4 的 168D SH encoder | 可见 32D 与 `32/31/1` colored bars，但不能由示意图唯一恢复 prose 所述五个 hidden layers 的 exact instantiated order | color legend 定义 `v*/v+/v≀` | scalar | 同上 | [P Fig.3, visually checked] |
| Output head | final hidden | scalar regression | final activation/clamp 未报告 | reflection 或 transmission | model-specific | [P Fig.3; S footnote 1] |

因此可复现的“exact scope”是：MetaNet 生成 4 个 32D direct feature blocks、5 个 32-weight scalar rows 与 5 个 biases；shared branch 留在 BSDFNet。论文没有把这些 block/row 逐一映射到一个无歧义的 layer index list。**不可复现的 exact topology** 是 BSDFNet 的首层 input projection、五个 hidden layers 与 Fig.3 colored bars 的准确对应：prose、示意图与后述 168D encoding 不能共同锁定唯一实例，且没有 official code/config 消解。报告据此拒绝从 `293=4×32+5×32+5` 反推 exact layer order。[P §§3.2,3.4, Fig.3; C unavailable]

### 5.4 条件化、坐标变换与物理先验

- Rusinkiewicz encoding 先把 directions 转为 half/difference vectors，再对每个 vector 拼接 raw 3D vector 和一组 SH bases：`γ(ω)=[ω,Y_0^0,...,Y_l^m,...]^T`。[P Eq.6]
- 作者称“up to ninth order”且把两个 encoded vectors 的总输入固定为 168D。由 Eq.6 的 raw 3D vector 拼接 SH 可得每个方向 `84=3+81`，而 `81=9²` 对应九个完整 SH bands；按常见零基 degree 计数是 `l=0…8`。但有些文献把“ninth order”用作九个 bands，有些用作最高 degree 9（后者会给每方向 `3+100`、总计206D）。因此**固定事实是168D/每方向81个SH值，未锁定的是作者的 order 命名约定**；无代码时不把歧义改写成唯一最高 degree。[P Eq.6, §3.4; arithmetic]
- SH+Rusinkiewicz 是 deterministic directional prior；没有 learned frame、normal map、spatial offset 或 analytic BSDF core。[P §3.4]
- `Γ` 保留可编辑物理参数，但模型只拟合 Guo reference 的 response；reciprocity、positivity 与 energy conservation 没有结构保证。[P §§3.3,6]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| LayeredBRDF corpus | 12,000 reflective/conductive-base BRDFs | [P §3.6] |
| LayeredBTDF corpus | 10,000 transmissive/dielectric-base BSDFs；BRDF/BTDF 模型分开训练 | [P §3.6] |
| Interface model | GGX；top roughness `α1=10^{U(-3,-0.5)}`，bottom `α2=10^{U(-3,0)}` | [P Table 1, visually checked] |
| Dielectric/conductor | `η1,η2~U(1.05,2)`；conductive interface 用 Schlick `R0~U(0,1)` 替换 IOR | [P §3.3, Table 1] |
| Medium | Table 1 原式为 `ρ=1-\mathcal U(0,1)^2`，即采 `u~\mathcal U(0,1)` 后取 `ρ=1-u²`；`σt~\mathcal V({0,1,2,5})`；isotropic phase | [P Table 1, visually checked; §3.4 footnote 1] |
| GT/reference | Guo et al. [2018] bidirectional position-free layered BSDF method | [P §§3.6,5] |
| Training GT spp | 128 spp per directional sample | [P §3.6] |
| BRDF directions | `(θ,φ)` for each of `ωh,ωd`，四个角维各 25 个 stratified samples；`25^4=390,625` pairs/BRDF | [P §3.6, PDF p.8 visually checked] |
| BTDF directions | elevation `θ∈[0,π)` 而非 upper hemisphere；共 `4×25^4=1,562,500` pairs/BSDF | [P §3.6] |
| Train/validation/test | 上述 12k/10k 被称为 training datasets；所有主结果材质未出现在训练集；validation size、test-set size与冻结清单未报告 | [P §§3.6,5 opening] |
| Quantitative test subset | Table 3/4、Fig.20 各用 100 个 randomly selected test BRDFs | [P Tables 3–4, Fig.20] |
| Rendering reference | Guo et al. [2018] 默认 2048 spp | [P §5 opening] |
| Online/offline | 论文描述先生成大型 sampled datasets 再训练；serialization、storage、invalid-coordinate handling未报告 | [P §§3.1,3.6] |

`25^4` 很容易被 PDF text extraction 读成 `254`，但正文 p.8 的上标排版明确是 `25⁴`；BTDF 同页明确写 `4×25⁴`。按 corpus 规模派生，BRDF 约含 46.875 亿 direction pairs，BTDF 约含 156.25 亿；论文未报告这些样本是否全部落盘、压缩格式或生成总时长。[P §3.6; P-derived arithmetic]

supplemental 的编辑网格包含训练凸包内外的参数：例如 `α1=0.5` 超出训练最大约 0.316，`σt=8` 超出训练离散最大 5；作者给出 pairwise reference 图，但没有跨外推区域的 aggregate metric。[S §2, Figs.1–10; P Table 1]

## 7. Loss、optimizer 与训练 lifecycle

### 7.1 正式 loss

作者先对 prediction 与 GT 做 `μ`-law HDR compression：

\[
T(f)=\operatorname{sign}(f)\frac{\log(1+\mu|f|)}{\log(1+\mu)},\qquad \mu=32.
\]

令 `e=|T(f)-T(f_s)|`，使用 reverse-Huber-style loss：`e≤t` 时取 L1，`e>t` 时取 `(e²+t²)/(2t)`，`t=0.1`。作者解释大误差 L2 branch 恢复高亮大值，小误差 L1 branch 保留 BSDF long tail；该 loss 在阈值处一阶连续。[P Eqs.7–8, §3.6]

### 7.2 两阶段训练

| 阶段 | material/query routing | 更新对象 | 目标 | locator |
|---|---|---|---|---|
| Phase 1 | 每次采一个材质及方向集合 `X` | 同时更新 `ΘM,ΘF` | 先让目标不同的两网稳定进入可训练区域 | [P Alg.1 lines 2–8, §3.5] |
| Phase 2a | 对同一材质把 `X→X1∪X2`；在 `X1` 上做 `K1` steps | 冻结 MetaNet，仅更新 BSDFNet `ΘF` | 稳定 shared BSDFNet weights | [P Alg.1 lines 9–17] |
| Phase 2b | 对 `X2` 做 `K2` steps | 冻结 BSDFNet，仅更新 `ΘM` | 让 parameter→generated-state mapping 泛化到新材质 | [P Alg.1 lines 18–24] |

这是一种 train-time alternating schedule；测试/编辑时不做 inner-loop optimization，也没有声明对 `K1` 更新反向传播的 MAML-style higher-order gradient。[P §§3.3,3.5, Alg.1]

### 7.3 正式配置与缺口

| 项 | Reflective/conductive model | Transmissive/dielectric BRDF 或 BTDF model | locator |
|---|---|---|---|
| Framework/hardware | TensorFlow；8×NVIDIA 2080Ti | 同左 | [P §3.6] |
| Optimizer | Adam default parameters | 同左 | [P §3.6] |
| Initial LR | `5e-4` | `5e-4` | [P §3.6] |
| LR schedule | 每 8 epochs `×0.99` | 同左 | [P §3.6] |
| Phase 1 | 100 epochs | 100 epochs | [P §3.6] |
| Phase 2 | 125 epochs | 100 epochs | [P §3.6] |
| Total/time | 225 epochs，约 50 h | 每个 model 200 epochs，约 48 h | [P §3.6] |
| Batch/query count | **未报告** | **未报告** | source gap |
| `K1/K2` 与 `X1/X2` 比例 | **未报告** | **未报告** | source gap |
| Initialization | 仅 Algorithm 1 写随机初始化 `ΘM,ΘF`；具体 initializer 未报告 | 同左 | [P Alg.1 line 1] |
| Seed/repeats/checkpoint selection | **未报告** | **未报告** | source gap |
| Multi-GPU partition/precision | **未报告** | **未报告** | source gap |

## 8. Inference、部署与成本

### 8.1 Runtime call path

对均匀材质，MetaNet 只在参数改变后运行一次，生成 BSDFNet state；hot path 只运行小型 BSDFNet。BSDFNet 被实现为 Mitsuba BSDF plugin，方向 SH encoding 预计算到 LUT，矩阵乘加与 ReLU 用 AVX-512 做 CPU data parallelism。[P §4 Renderer Integration, §5 opening]

对 spatially-varying parameters，作者不是每个 shading point 在线跑 MetaNet，而是对每个 texel 预生成 293D/channel state 并存 LUT；渲染时默认 nearest-neighbor，论文说也支持 linear interpolation。texture state generation 随 texel 数近似线性。[P §§4,5.3]

### 8.2 成本与披露范围

| 项 | 正式配置/数值 | locator |
|---|---|---|
| MetaNet call | 均匀材质 weights “less than 1 second”；intro 同时写“milliseconds”，无同一 benchmark 对齐 | [P §§1,5.1] |
| MetaNet topology | 16×256 trunk +293 output；参数量/MAC 未报告 | [P Fig.3, §3.3] |
| BSDFNet topology | 5 hidden、>13K parameters；因 §5.3 所列图文冲突无法锁 exact count | [P §3.2, Fig.3] |
| Per-material/channel state | 293 scalars；precision 未报告 | [P §3.2] |
| Per RGB material `[P-derived]` | 879 scalars；若 FP32 为 3516 B≈3.43 KiB | [P §§3.2,5.3; arithmetic] |
| SV state example | 512×1024 texture约 1.7 GB；与 RGB×293×FP32 arithmetic 相符 | [P §5.3, Fig.16] |
| Texture prepare | 512×1024：27 s；1024×1024：53 s；800×2000：80 s | [P Figs.16–17] |
| SH LUT | precomputed；分辨率、filter、precision、cache cost 未报告 | [P §4] |
| Runtime precision/quantization | 未报告 | source gap |
| Runtime workstation | Intel Core i9-9900X，64 GB RAM；CPU-only neural BSDF | [P §5 opening] |

正文没有给单次 BSDFNet query latency、MAC/FLOP、SIMD lane utilization、shared-weight bytes、coherent/divergent material batching或 GPU shader实现。scene timing混合了 path tracing、sampler与 evaluator，不可直接转换为 shader-query cost。[P §5]

### 8.3 `sample()/pdf()` 的实际贡献边界

作者另外采用 Belcour-inspired analytic multi-lobe proposal。reflective base 选 `R` 与 `TRT` 两个 lobes，transmissive base 选 `R/TRT/TT` 三个 lobes；按估计能量 `Ei` 随机选择 lobe，再 importance sample visible normals，PDF 由对应 microfacet lobe mixture 计算。[P Eq.9, §4 Importance Sampling]

该 sampler 不是 MetaNet/BSDFNet 输出，也没有 learned proposal。lobe energy/roughness 参数如何从 neural BSDF state 稳定估计只写“refer to Belcour [2018]”；没有 code、单独 sampler quality、PDF normalization、sample↔pdf reciprocity或方差消融。作者在 limitation 中明确它不是 arbitrary layered material 的 optimal sampler。[P §§4,6]

## 9. 实验 protocol、baseline、指标与结果

所有主结果材质均称为未出现在训练集，reference 默认由 Guo et al. [2018] 以 2048 spp 渲染；但作者没有发布 test manifest、seed 或 renderer code。[P §5 opening]

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| Representation capacity | 单个 layered BRDF rendering/error map | NBRDF、同 trainable-parameter 数的 NBRDF+ | image RMSE | `0.020 / 0.011 / 0.010`（NBRDF/NBRDF+/Ours） | [P Fig.4] |
| 100-BRDF representation aggregate | random test BRDF，single point light | NBRDF、NBRDF+ | RMSE mean/STD | NBRDF `0.891/1.960`；NBRDF+ `0.873/2.488`；Ours `0.483/1.043` | [P Table 3] |
| Positional encoding aggregate | 100 random test BRDF，single point light | sinusoidal raw/Rusinkiewicz coords；SH raw coords | RMSE mean/STD | `1.249/2.847`、`1.265/2.241`、`1.230/2.833`；SH+Rusinkiewicz `0.998/1.900` | [P Table 4] |
| Loss | 100 random BRDF，point/environment light | L1、L2 | average image RMSE | point `0.096/0.106/0.092`；environment `0.018/0.018/0.017`（L1/L2/Ours） | [P Table 5] |
| Belcour visual | rough interfaces `α1=0.1,α2=0.3`，`σt=0` | Belcour [2018]、Guo reference | visual only | Ours visually closer；无 numeric aggregate | [P Fig.7, §5.1] |
| Guo equal-spp | Kettle，64 spp | Guo [2018] | time/RMSE | Guo `18.1 s,0.09`；Ours `6.3 s,0.04` | [P Fig.8] |
| Extinction sweep | Dragon，64 spp，`σt=0/0.5/1/2/3` | Guo [2018] | RMSE | Ours `0.079/0.048/0.042/0.040/0.040`；Guo `0.092/0.059/0.048/0.044/0.043` | [P Fig.11] |
| High-extinction runtime | Dragon，64 spp，`σt=5` | Guo [2018] | scene time | Ours约 6× acceleration；柱图未给精确表值 | [P Fig.12, §5.1] |
| Transmissive sweep | Frosted Glass，`α1=α2=0.02` | Guo [2018] | RMSE/spp curves | `σt=0/1/2` Ours `0.092/0.055/0.030`；Guo `0.097/0.098/0.067` | [P Fig.13] |
| NLBRDF roughness | conductive plane，256 spp | NLBRDF、Guo reference | image RMSE | `α1,2=0.01,0.05`: Ours/NLBRDF `0.009/0.018`；`0.1,0.1`: `0.009/0.010`；`0.3,0.3`: `0.002/0.006` | [P Fig.9] |
| Long path | Shoe；depth2 1 spp与depth100 2048 spp | Guo、NLBRDF | heterogeneous CPU/GPU time | depth2 Guo CPU 3.8 s；NLBRDF GPU 0.05 s + CPU 0.01 s且另需55 min data；Ours CPU 0.2 s。depth100 Guo CPU 51 min；NLBRDF unavailable；Ours CPU 5.6 min | [P Fig.10] |
| SV equal time | Globe | Guo [2018] 64 spp 2.4 min | image RMSE | Guo `0.088`；Ours 434 spp/2.4 min `0.040` | [P Fig.15] |
| Editing/extrapolation | `α1×σt` grid及 supplemental 全参数 grids | Guo reference | paired images；部分主图 RMSE | convex-hull 外 appearance 仍平滑，但无 outside-region aggregate | [P Fig.5; S Figs.1–10] |

这些对比回答不同问题：NBRDF+ 接近同参数量 representation；Guo 是 training GT、quality reference 和 stochastic runtime baseline；Belcour 是无介质的 approximate analytic baseline；NLBRDF 使用不同 CPU-GPU architecture 与 per-material data optimization。Fig.10 尤其不是同设备、同 preprocess inclusion 的 latency ranking。[P §§5.1–5.3]

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `ablation-inferior` | 每层把 predicted `v≀` 从 1 neuron增至2 | training loss较高、收敛较差 | 待预测权重变化维数增大，MetaNet更难优化 | 说明 generated-weight budget 同时控制容量与优化条件数，不证明1 neuron普遍最优 | [P §5.4, Fig.18 green] |
| `author-negative` | 像 NBRDF 一样预测 BSDFNet 全部 weights | 作者称大 BSDFNet难以收敛 | predicted parameter space过大 | 没有 matched curve/config，属于作者报告的 rejected direction，不可量化 | [P §§3.2,5.4] |
| `ablation-inferior` | 完全移除 direct `v*` | convergence慢于完整模型 | ReLU可能令经 `W*` 的梯度趋零；`v*` 提供直接 BSDFNet→MetaNet gradient path | supplemental 给出解析推导，但未报告多 seed variance | [P §§3.2,5.4, Fig.18 blue; S §1 Eq.11] |
| `ablation-inferior` | 无 positional encoding | glossy high-frequency恢复失败 | MLP spectral bias | 与 SH+Rusinkiewicz相比部分例子差距很大 | [P Fig.6; S Fig.12] |
| `ablation-inferior` | sinusoidal encoding | glossy lobe出现 ghosting；aggregate mean RMSE 1.249/1.265 | sinusoidal basis与球面/Rusinkiewicz信号不匹配 | 论文没有 iso-dimension/frequency-budget细节 | [P §3.4, Fig.6, Table 4; S Fig.12] |
| `ablation-inferior` | SH on raw `(ωi,ωo)` | mean/STD `1.230/2.833`，差于 SH on `(ωh,ωd)` `0.998/1.900` | half/difference coordinates更适合 specular lobe | 支持“坐标与编码交互”而非单独归功 SH | [P Table 4] |
| `ablation-inferior` | conventional L1 或 L2 | point-light RMSE `0.096/0.106`，高于 ours `0.092` | L1顾 long tail，L2顾 highlight large values | 环境光差仅 0.001，效应依 lighting protocol | [P §5.5, Table 5, Fig.19] |
| `ablation-inferior` | 始终 joint update 的 one-phase training | RMSE曲线较高且不稳 | 两网目标不同；交替更新稳定 shared weights并改善泛化 | 无 seeds/error bars，不能把单曲线称为 variance proof | [P §5.6, Fig.20] |
| `known-limitation` | `α1=0.001, α2=0.01` very smooth | under-estimated highlight、inconsistent shading | `μ`-law compression与 network floating-point accuracy | 失败落在训练 roughness下界，表明边界峰值仍是核心压力 | [P §6, Fig.21] |

supplemental Figure 11 显示 Ours 在8个 NBRDF examples中通常最低 RMSE，但仍有高误差案例（例如一行 `3.981`）；Figure 12 的最佳 encoding 在8例中也有 `4.161`、`3.080`、`2.373` 等大 RMSE。它们说明 aggregate改善没有消除困难高动态峰值。[S Figs.11–12]

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Architecture | 5-hidden BSDFNet、16×256 MetaNet、293 generated state | 推导 `M(Γ,ΘM)={v*,W*,b*}` 与 `v*` gradient shortcut | 无 | 293 scope被两份材料支持；BSDFNet首层/层序存在 paper-internal gap |
| Direction encoding | Rusinkiewicz + SH，168D，“up to ninth order” | 更多 raw/Rusinkiewicz、sin/SH视觉对照 | 无 | 168D锁定每方向81个SH值；“ninth order”是九 bands 还是最高 degree 9 未由代码消解 |
| Data/query | 12k/10k、128 spp、`25^4`/`4×25^4` | 编辑外推 grids | 无 dataset | 无 split manifest、serialization与query validity规则 |
| Loss/training lifecycle | μ-law reverse Huber、Adam、LR、epochs、两阶段 Algorithm 1 | 仅为 gradient path提供推导 | 无 config/log | batch、K1/K2、split ratio、seed与checkpoint选择均无法补齐 |
| Runtime/export | Mitsuba plugin、AVX-512、SH LUT、MetaNet per material/texel | 无 runtime细节 | 无 plugin/kernel | 单 query cost与exact parameter layout不可审计 |
| Sampler | Belcour-style R/TRT/TT mixture | 无 | 无 | 是外接 analytic proposal，不是完整 neural sampler贡献 |
| Evaluation assets | scenes、100-BRDF subsets、Guo 2048-spp reference | NBRDF/encoding额外图 | 无 scene/test manifest | 数值可抄录，不能独立复跑 |

作者主页只提供 Paper、Video、Supplementary 三项下载，没有 Code 链接。定向 repository search 也没有发现官方发布；因此 `official_code_status=unavailable`，而不是把 Mitsuba implementation statement误记为 `audited`。[A homepage entry; C absence audit]

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. **Very smooth surfaces**：`α≈0.001` 时 under-estimate highlight；作者归因于 μ-law 与 network floating-point accuracy。[P §6, Fig.21]
2. **Single layer/two interfaces only**：multiple layers被留作参数集扩展；论文没有验证 layer-count extrapolation。[P §6]
3. **Isotropic only**：interfaces与phase当前均 isotropic；anisotropy会增加参数并让训练更难。[P §3.4 footnote, §6]
4. **Sampler非最优**：R/TRT 或 R/TRT/TT 不能覆盖任意多层/各向异性产生的更多 lobes。[P §6]
5. **Measured materials不直接适用**：MetaNet需要显式物理参数；作者提出未来可先压 measured BRDF latent，但认为这与当前物理编辑目标不一致。[P §6]
6. **Energy conservation无保证**：作者说实际未观察由此导致的 artifact，但没有结构或测试保证 conservation。[P §6]
7. **SV storage高**：293D/channel state 导致 512×1024 RGB texture约1.7 GB。[P §5.3]

### 12.2 未报告/材料不可得

- official training/inference/sampler code、commit、config、checkpoint、dataset、scene manifest；
- BSDFNet exact instantiated topology、SH最高 degree的计数解释、MetaNet activation/normalization；
- batch size、`K1/K2`、`X1/X2`比例、random seed、重复训练、model selection；
- validation/test总体数量和固定 material parameters；
- Guo training target的存储格式、总生成成本、invalid Rusinkiewicz coordinate处理；
- final output positivity、reciprocity、BTDF eta convention与精度；
- MetaNet/BSDFNet单 query MAC、latency、shared bytes、AVX layout与SH LUT规格；
- lobe energy/roughness estimator、sampler PDF exact formula implementation与独立 sampler validation；
- spatial LUT interpolation的训练对应、filter footprint、mipmap/LOD与误差；
- Fig.10 heterogeneous CPU/GPU baseline的严格 preprocess accounting和同设备对照。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

MetaLayer 的核心不是“293D latent + decoder”这么简单。容量分三层：16×256 MetaNet学习 source-parameter manifold；shared BSDFNet保留跨材质 directional basis；293D/channel output把每个材质注入到四个32D intermediate feature blocks和五个逐层 scalar affine rows。相比只在输入 concat 一个短 latent，它把条件化分布到多个深度，接近受强约束的 hypernetwork/low-rank modulation。[P §§3.2–3.3; I]

这种设计把 large compiler移出 hot path，但把 runtime asset放大为879 scalars/RGB material；SV material甚至逐 texel复制。它在“单材质 CPU path tracing”与“海量材质/纹理、shader fixed reads”的 Pareto位置完全不同。[P §§4,5.3; I]

### 13.2 成功所依赖的假设

- source family有短、连续、物理有意义的 `Γ`，并且每个通道可独立处理；
- 单层、isotropic GGX+homogeneous medium 的参数流形足以由固定 shared basis覆盖；
- 128-spp Guo targets和巨量规则方向网格提供足够监督；
- test只要求同一 source family内的新参数组合，不要求新 graph topology、measured BRDF或任意 source family；
- material/texture变更时可以预运行大 MetaNet并持久化高维 generated state；
-外接 analytic sampler即使不是 evaluator-matched optimum，也足以完成本文 scenes。

[P §§3.3–6; I]

### 13.3 可迁移机制与不能迁移的部分

可迁移机制：

1. 把 source compiler 与 runtime evaluator分离，并用 source parameters直接生成部署 state；
2. 在多个 hidden layers注入受限 material-specific state，而非只在输入 concat；
3. 把 half/difference coordinate与球面 basis作为联合轴消融；
4. 对 hypernetwork先 joint bootstrap，再交替稳定 shared evaluator与compiler；
5. 把 sampler视作独立 proposal组件，不能由 evaluator accuracy自动推出 sampling quality。[I]

不能直接迁移：

- `Γ` schema只覆盖本文 layer family，不能作为 MERL、BTF、MaterialX graph或一般原生材质的统一输入；
- 879 floats/texel、未量化、无mip/footprint合同，不满足当前小型 shader program的默认预算；
- 离线 dense corpus与Guo 128-spp target不同于本项目 GPU-resident online reference query；
- 无代码使两阶段 schedule与exact runtime network不能作为 faithful reproduction recipe；
- analytic R/TRT/TT sampler没有证明与 learned evaluator匹配。[N experiment framework; I]

### 13.4 与本项目 runtime contract 的关系

BSDFNet层数固定，理论上可形成静态有界 `evaluate()`；MetaNet自然对应 compiler 或材质编辑时的 `prepare()` 前置阶段。论文也有 sample/pdf proposal。但要进入本项目 ABI，仍需证明：输出是线性 raw `f`；RGB三通道与reflection/transmission event的统一路由；879-state的固定读取和插值；sample/pdf solid-angle measure与同一proposal一致；以及与 `prepare` footprint/LOD语义的结合。[N runtime contract; P §§4–6; I]

因此 MetaLayer当前最适合作为 **M6 typed source compiler baseline、layer-wise conditioning diagnostic 或 teacher/compiler候选**，不是直接的 shader-budget product candidate。[N model candidates §§M6; I]

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

本节所说“当前”专指仓库现行 `correspondence_id=nvidia-rta2024-functional-f@2` 与 formal recipe `nvidia-rta2024-materialx-formal-300k-stage100k@1`。归档 correspondence 仍记录旧 `functional@1`，只能说明迁移前的设计依据，不能充当 `@2` 的运行结果或身份凭证。[N current formal config lines 6–7; N `src/ncls/learning/methods/nvidia.py:398-435`; N archived correspondence lines 7–13]

| 主题 | 当前仓库真实状态 `N` | MetaLayer带来的解释 | classification |
|---|---|---|---|
| Source→state compiler | NVIDIA reproduction 已实现 native parameters `K→64→64→64→64→8` encoder，并在 lifecycle 中 materialize hierarchical z8 | MetaLayer说明“从参数生成更丰富的逐层 state”是可测 M6 baseline；它不证明当前8D encoder错误 | `not-applicable` to fidelity；future `budget-adaptation` experiment |
| Runtime conditioning | NVIDIA 方法包把 hierarchical z8 导出为两张 RGBA latent texture，并按 footprint 做 LOD/filter | MetaLayer的879 floats/RGB material与多层weight injection是不同 representation，不能借名义上的 hypernetwork替换 faithful latent layout | `intentional-deviation` if tested as candidate |
| Training lifecycle | NVIDIA按公开合同先 encoder bootstrap、materialize、再 latent finetune；精确切换点标为 author-underspecified | MetaLayer的joint→alternating two-phase解决的是 hypernetwork/shared-weight稳定性，不是同一 lifecycle | `not-applicable` |
| Sampler | 当前 frozen recipe 为 `learned-sampler-forward-kl-score@1`：从 learned proposal 取样，以当前 learned evaluator 的 `luminance(f)·|cosθi|` 形成目标；完整 estimator 是 later-author-code-informed 选择 | MetaLayer只外接 Belcour multi-lobe proposal；不能用其 R/TRT/TT sampler解释或修改 NVIDIA KL route | `not-applicable` |
| Suspected defects | 当前 config/validator与归档 correspondence 分别约束新旧 identity；公开合同与 author-underspecified 选择不能混写 | 本论文无 official code，且没有证据指向当前 NVIDIA 实现 defect | **无新增 `suspected-defect`** |

[N current config lines 6–81; N `src/ncls/learning/models/nvidia_neural_appearance.py:27-88`; N `src/ncls/learning/methods/nvidia.py:398-435,500-550`; N runtime contract lines 3–7; N archived correspondence lines 19–35; N model candidates M6; I]

MetaLayer对当前复现最有价值的动作是新增一个**隔离候选**：同 source/query/loss/bytes 下比较 generated latent 与 generated layer-wise modulation；不是改写 NVIDIA faithful recipe。[N experiment framework P2/P3; I]

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：逐层 generated modulation在同 state bytes 下优于输入 latent concat | 293D state分布到4个feature blocks与5个affine rows；NBRDF+仍较差 [P §§3.2,5.4] | LayerStack长尾来自条件化深度而非仅方向容量 | M1 shared decoder + equal-byte latent；同decoder width/depth | source/query、train work、loss、seed set、state bytes | test median/p95、G2/G2s、query time、bytes | compiler + bounded evaluator | 三个以上 seeds中 p95/G2s无显著改善，或成本Pareto被control支配 |
| H2：Rusinkiewicz SH的收益来自“坐标×basis”交互 | Table4中SH raw `1.230` vs SH half/diff `0.998` mean RMSE [P Table4] | 当前LayerStack峰也在half/difference域更平稳 | raw/Fourier、half-diff/Fourier、raw/SH、half-diff/SH四格 | basis dimension、MLP、queries、optimizer、work units | dense peak error、grazing bins、median/p95、time | evaluator-only | iso-dimension下SH+half/diff不改善峰值/掠射，或只靠更多features取胜 |
| H3：joint bootstrap→alternating training降低compiler优化不稳定 | one-phase曲线较差；`v*`与小predicted scope改善收敛 [P Figs.18,20; S §1] | 当前M6也有compiler/evaluator目标冲突 | 始终joint、phase2交替、只冻结decoder三组 | initialization pool、total query work、LR search budget | seed success rate、best/median/worst、G2、wall time | training-only | matched work与调参后alternating不降低seed variance或最终error |
| H4：source-only compiler需要target-visible bounded refinement才能覆盖G2s | MetaLayer只展示同族参数泛化，外推仅visual [P Fig.5; S §2] | 本项目source space更广，纯前向compiler可能留系统误差 | optimized-code control、compiler-only、compiler+fixed-budget refinement | runtime decoder/state bytes、train/test queries、refinement budget | G2/G2s、cook time、workflow W | offline compiler | compiler-only已匹配optimized-code control，或refinement收益不足以覆盖cook成本 |
| H5：879-float generated state不适合当前shader budget，但低秩压缩可保留收益 | MetaLayer per-texel 1.7GB例子与固定layer injection [P §5.3] | generated rows存在冗余，可压为更短factor code | full generated-state teacher vs rank/bytes sweep vs M1 latent | evaluator depth、source/query、training work | quality–time–memory Pareto、fixed reads、coherence | teacher→budget candidate | 压到允许bytes后质量退回/差于M1，或random-access读取成本越过硬门槛 |
| H6：analytic R/TRT/TT proposal不能自动匹配neural evaluator | 作者明确sampler非optimal且无sampler ablation [P §§4,6] | evaluator误差会改变目标密度，独立analytic proposal产生mismatch | source-native sampler、MetaLayer analytic mixture、当前learned matched sampler | evaluator、scene、spp、time budget | KL/proposal audit、variance、firefly rate、sample/pdf identity | sampler track | analytic mixture在matched tests中已等价或更优且通过identity，无需learned proposal |

这些假设都要求把 representation quality、compiler generalization、optimization stability 与 sampling variance分开报告；不能用“scene更干净”同时证明四件事。[I]

## 16. 证据索引

### `P` Main paper

- §1：问题、贡献、MetaNet一次前向与“milliseconds”叙述；
- §§3.1–3.3、Eqs.1–5、Figs.2–5：hypernetwork、293 generated scope、BSDFNet/MetaNet结构与编辑；
- §3.4、Eq.6、Fig.6：Rusinkiewicz SH encoding与168D；
- §§3.5–3.6、Algorithm 1、Eqs.7–8、Table1：两阶段训练、loss、数据与参数分布；
- §4、Eq.9、Table2：Mitsuba/AVX-512 integration、SV LUT、analytic sampler；
- §§5.1–5.6、Figs.7–20、Tables3–5：baselines、timings、RMSE、消融；
- §6、Fig.21：smooth failure、multiple layers/anisotropy、sampler、measured material与energy limits。

### `S` Supplemental

- §1、Eqs.1–11：BSDFNet→MetaNet gradient derivation与`v*` shortcut；
- §2、Figs.1–10：参数编辑与训练凸包外视觉验证；
- §3、Fig.11：NBRDF/NBRDF+额外结果；
- §4、Fig.12：坐标×encoding额外结果。

### `C` Official code/config/data

- 未发现 official repository、commit、config、checkpoint、dataset或renderer plugin；该缺失本身只支持“不可审计”，不支持对实现细节的猜测。

### `A` Author material

- Lingqi Yan homepage 2023 MetaLayer entry：只列 Paper、Video、Supplementary；
- GitHub `ucsb-archive` release assets：正文、视频、supplementary下载容器；
- DOI `10.1145/3618365`：论文身份。

### `N` NeuralShading

- `docs/research/prior_art.md:101-111,232-238`：MetaLayer作为LayerStack compiler强基线；
- `docs/research/model_candidates.md:243-294,339`：M5 target encoder与M6 source compiler边界、MetaLayer iso-byte baseline；
- `docs/realtime_material_compilation.md:3-7`、`docs/contracts/scattering_backend.md:3-5`：`prepare/evaluate/sample/pdf`、bare linear `f`与solid-angle sample/pdf合同；
- `configs/learning/nvidia-rta2024-materialx-formal.json:6-81`、`src/ncls/learning/models/nvidia_neural_appearance.py:27-88`、`src/ncls/learning/methods/nvidia.py:398-435,500-550`：当前 `functional-f@2` identity、encoder/materialization 与 evaluator/sampler recipe；
- archived NVIDIA correspondence `lines 7-35`：旧 `functional@1` 身份下的 paper↔implementation 分类与 author-underspecified 选择；只作历史依据，不冒充当前 artifact。

### `I` Inference/transfer

- RGB state bytes、corpus query count等均明确标为 arithmetic；
- §§13–15的compiler定位、迁移风险和实验假设不属于作者结论。

## Evidence review

```text
author_worker: /root/taming2026
reviewer: /root/nbrdf2021
reviewed_at: 2026-08-29
sources_rechecked:
  - main PDF 15 pages; SHA-256 FCFEDC1307420C0734683AEADA8E04EFBA67A958E710937A3F9DFF5880D57DE5; all pages visually checked
  - supplementary ZIP SHA-256 6B321B270562E365CB0B96B967A8C605B63B285D8059B8521EA992E142091955; sole internal PDF 9 pages SHA-256 8013134A39CD3DBD6342742D1F8F2CA08B9A6055A9D6A3ACEF2EA085D5CAB510; all pages visually checked
  - current author homepage MetaLayer entry and its Paper/Video/Supplementary release URLs
  - targeted official repository/config/data search; no author code release found
  - current NeuralShading runtime/compiler docs, NVIDIA formal config and implementation locators; archived identity checked separately
findings_closed:
  - removed unsupported Article 222 secondary-database claim; primary PDF consistently says Article 221
  - removed Fig.3 input-dimension guess and preserved BSDFNet exact-layer-order as unresolved paper-internal/code-availability gap
  - separated MetaNet's evidenced 293-vector partition from an unevidenced multi-head topology
  - recomputed 293D/channel and RGB 879-scalar interpretation against the paper's independent-channel statement and 512x1024 approximately 1.7 GiB example
  - resolved the SH statement to the load-bearing facts: 168D total and 81 SH values per direction; retained ninth-order nomenclature ambiguity
  - rechecked two-phase lifecycle, loss, dataset counts, sampler scope, CPU runtime context, tables/figures and failure classifications
  - updated N/I section to current functional-f@2 identity while retaining archived functional@1 only as historical evidence
remaining_evidence_gaps:
  - official code/config/data 未发现，无法锁 commit 或消解网络、batch、K1/K2 与 runtime layout
  - BSDFNet prose/Fig.3/168D-SH input 仍不能锁定 exact instantiated layer order；报告明确不猜
  - 293D/channel 是由 RGB 独立预测与 1.7GB 例子交叉推导，不是公开 tensor layout；precision/quantization 仍未报告
  - SH 的 81 values/direction 已锁定，但“ninth order”的 band-count/maximum-degree 命名未由代码消解
  - MetaNet activation/normalization/output constraints、batch、K1/K2、X1/X2 比例、seed/checkpoint selection 均不可恢复
  - sampler lobe-parameter estimator、PDF 实现与单独 variance/identity validation 不可审计
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
