---
paper_id: "xu-2025-comprehensive-neural-materials"
title: "Towards Comprehensive Neural Materials: Dynamic Structure-Preserving Synthesis with Accurate Silhouette at Instant Inference Speed"
authors: "Zilin Xu; Xiang Chen; Chen Liu; Beibei Wang; Lu Wang; Zahra Montazeri; Ling-Qi Yan"
year: "2025"
venue: "SIGGRAPH Conference Papers '25, Article 161"
doi: "10.1145/3721238.3730626"
report_status: "evidence-reviewed"
main_source: "https://sites.cs.ucsb.edu/~lingqi/publications/paper_sig25_cnm.pdf"
supplemental_status: "available"
official_code_status: "audited"
official_code_commit: "renderer dd6238576e43ce03780a4dc811b9523b9c277280; training b87fe3565b68f465e7fa621d7f006a50c76b16d8"
author_worker: "/root"
reviewer: "/root/nbrdf2021"
last_verified: "2026-08-29"
---

# Towards Comprehensive Neural Materials：量化、结构保持合成与位移轮廓的组合系统

## 1. 研究对象与报告边界

本文把一个 neural material 拆成三个同时存在但语义不同的子系统：

1. 用 Quantized Triple Plane（QTP）逐材质拟合 6D BTF/SVBRDF，并用 Int8 权重与 activation 降低全屏逐像素 decoder 成本；
2. 只在分解后的空间 `U` feature plane 上执行动态 by-example synthesis，并用 curved autocovariance function（Curved ACF，弯曲后的自协方差函数）控制 patch selection，从而减少规则纹理结构被随机拼接破坏；
3. 对原生带 height field 的材质，把高度图与外观分别做同一类动态合成，再用 shell map 上的 coarse/fine 两级 tracing 生成 parallax 与 silhouette，而不是再训练 offset/alpha network。[P Abstract, §1, §3–4, Eq.1, Fig.1–4]

它是 **per-material spatial appearance representation + synthesis/deployment system**，不是跨材质 compiler：每种材质各自持有 `U/H/D` planes、MLP、量化 scale、height field 与 synthesis 数据。它也没有学习 `sample()/pdf()`；正文明确将 importance sampling 判为正交问题并留作未来工作。[P §4.1, §7][C renderer `NeuralMatRendering.h`, `NBTF.cpp`]

本报告覆盖正式 11 页 SIGGRAPH 2025 论文、2 页 supplemental、作者 project page、公开 renderer/training 两个仓库的固定 commit。论文正式结果与代码当前 default 严格分开记录；公开代码 README 自称 “only for reference purposes”，而静态审计发现多个会阻断 paper-config 直接重建的 correspondence 缺口，因此不能把 code default 当作正式实验配置。[C renderer/training README]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | [作者托管正式 PDF](https://sites.cs.ucsb.edu/~lingqi/publications/paper_sig25_cnm.pdf)；[DOI](https://doi.org/10.1145/3721238.3730626) | 2026-08-29 | SHA-256 `4CAAEAB2E55221ACE52D2F223796F7561CB68C1133BA52464FADB6E8B7A61D7D` | 已完整读取并视觉复核 11 页、Eq.1、Fig.1–12、图注、脚注及正文结果。先前未完整下载的本地缓存未用于证据。 |
| Supplemental `S` | [作者 supplemental](https://sites.cs.ucsb.edu/~lingqi/publications/supp_sig25_cnm.pdf) | 2026-08-29 | SHA-256 `79E8E5626B7DBCE3E7E0AF9B7E8110C6F0E3C5624818A0AC079674056764DFC1` | 已完整读取并视觉复核 2 页、Table 1、Fig.1–2、脚注；提供量化、数据、训练、CUDA、Bezier 与 tracing 细节。 |
| Renderer code `C-R` | [Starry316/ComprehensiveNeuralMaterial](https://github.com/Starry316/ComprehensiveNeuralMaterial)；论文中的旧仓库名 `InstantNeuralMaterial` 当前 301 重定向到该仓库 | 2026-08-29 | commit `dd6238576e43ce03780a4dc811b9523b9c277280` | 审计 CUDA `dp4a` inference、feature/synthesis 载入、硬编码 topology、量化 scales、六个 demo material 与 height-field path；未下载外部 SharePoint asset pack，未构建执行。 |
| Training/export code `C-T` | [Starry316/ComprehensiveNeuralMaterial-Train](https://github.com/Starry316/ComprehensiveNeuralMaterial-Train) | 2026-08-29 | commit `b87fe3565b68f465e7fa621d7f006a50c76b16d8` | 审计 `networks.py`、`dataset.py`、`train_qtp.py`、`export_weights.py` 与 README；code default 与 paper-config 的差异见 §11。 |
| Author page/video/assets `A` | [project page](https://starry316.github.io/sig2025/index.html)；[supplementary video](https://sites.cs.ucsb.edu/~lingqi/publications/video_sig25_cnm.mp4)；renderer README 中 SharePoint asset pack | 2026-08-29 | project page 无版本；video HEAD 为 513,302,114 bytes，未缓存；asset pack 未下载 | 核对标题、作者、DOI、正式 PDF/supp/code/video 入口。视频未用于数值或方法事实；外部资产未审计，不能验证发布 checkpoint 与 paper figure 的身份。 |
| NeuralShading evidence `N` | `configs/learning/nvidia-rta2024-materialx-formal.json`；`src/ncls/learning/methods/nvidia.py`；`src/ncls/learning/models/nvidia_neural_appearance.py`；`shaders/ncls/backends/nvidia_neural_appearance/`；`docs/realtime_material_compilation.md`；`docs/research/experiment_framework.md`；`docs/research/model_candidates.md`；归档 `correspondence.md` 仅作历史 provenance | 2026-08-29 | 当前 workspace；formal correspondence `nvidia-rta2024-functional-f@2` | 只用于 §13–15 的项目映射，不作为本文事实；当前 config/code 优先于归档说明。 |

未发现独立 erratum。Supplemental PDF 的 xref 有解析警告，但两页均能完整渲染，文字、Table 1、Fig.1–2 与脚注已按页面图像核对。

## 3. 原论文的问题、假设与贡献边界

作者把“完整外观”限定为四个彼此耦合的工程轴：quality、performance、dynamic synthesis、parallax & silhouette。论文认为此前方法通常只覆盖其中一部分：高维 measured material 虽能被神经压缩，但单材质 patch 小、全屏推理仍需数毫秒；随机动态 synthesis 会破坏结构；用 offset/alpha network 表达位移既增加 inference，又难与位置 plane 的 synthesis 对齐。[P §1, §3]

具体贡献是：

- **QTP**：保持 Triple Plane 的 `U/H/D` 分解，把 4 层小 MLP 改成 per-tensor symmetric signed Int8 QAT；feature planes 保持 FP32。[P §4.1–4.2][S §1.1]
- **Curved ACF synthesis**：ACF 不是直接生成纹理，而是改变候选 patch 的采样概率；额外曲线 `T` 让用户在“更结构保持”和“更随机”之间调节。[P §4.1, §4.3][S §1.5]
- **two-step height-field tracing**：shell 内只 traverses coarse/fine 两级；coarse 跳过空区，fine 最多 32 步精确定位。[P §4.4][S §1.6, Fig.2]

作者没有声称：一个模型跨材质泛化、从任意原生材质图编译出 QTP、学习完整 indirect transport、学习材质驱动 importance sampler，或在 mobile/通用 shader backend 上运行。[P §4.1, §7][S §1.4]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | 单个 measured BTF，或 Falcor Standard Material 生成的 synthetic BTF；需要 silhouette 时另有显式 2D height field | measured：`400×400×22,801` images；synthetic：`800×800×10,000` images | S §1.2；P §5 |
| Runtime material query | `f_r(u, ω_o, ω_i)`，再重参数化为 `f_r(u,h,d)` | `u∈R²`；`ω_o,ω_i` 位于表面上半球；`h,d∈R²` | P §1, §4.1 Eq.1 |
| Direction coordinates | Rusinkiewicz half/difference；feature plane lookup 再使用 Shirley concentric hemisphere parameterization，而不是 TP 的直接 `θ,φ` | `h,d` 各映射到方形 2D texture coordinate | P §4.1；S §1.2–1.3；C-T `dataset.py` |
| Position coordinate | 原材质局部 2D `u`；直接 lookup 或先经过 dynamic synthesis | U plane periodic/tiled lookup；绝对物理尺寸只对 UBO patch 给出约 `5 cm×5 cm` 背景量级 | P §3.2, §4.1 |
| Output | RGB reflectance/BTF value | 3 scalars；正式 radiometric unit、是否含 cosine、color space 未报告 | P Eq.1, Fig.4；C-T `networks.py` |
| Height-field query | shell entry/exit、动态合成 height field、coarse/fine grid traversal | coarse cell 宽度为 fine 的 `2^5=32` 倍；fine 最多 32 steps | P §4.4；S §1.6 |
| Validity restrictions | 反射半球；神经 offset 方法的 grazing angle 在作者讨论中会 clamp，而本文 height field 使用线性 shell ray | transmission、two-sided、tangent-frame seam、方向边界行为未报告 | P §3.3, §7 |

`f_r` 的输出不应自动等同本项目 canonical `evaluate().f`：论文未说明 cosine、测量归一化、能量单位和 synthetic Falcor export response。迁移前必须以正式 source/reference 对同一 query 做数值 correspondence，而不能只对 RGB 图片。[I]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

论文的核心分解是：

\[
f_r(u,h,d)=\mathcal N\bigl(\operatorname{Syn}(f^{(U)},u), f^{(H)}(h), f^{(D)}(d)\bigr).
\]

`U/H/D` 是三个二维可学习 feature planes。普通 query 对三个坐标分别做 bilinear lookup 并 concat；开启 synthesis 时，只有 `U` lookup 被 `Syn` 替换，方向 planes 不合成。24D feature 进入四个 Quantized Layers（QL）恢复 RGB。[P §4.1–4.2, Eq.1, Fig.4]

Synthesis 的 patch selection 权重来自 decomposed positional function `f^(U)` 的 ACF。论文把 `f^(U)` 写成向量 feature plane，却没有定义 8 channels 如何标量化；release renderer 的 `precomputeFeatureData()` 实际只用 U plane 第一个 `float4` layer 的第一个 channel 计算 ACF，其余 channels 仍分别做 Gaussian transform/inverse LUT。这个 code convention 不能反写成论文正式定义。对空间域 `S`：[P §4.1, Eq.1][C-R `Synthesis.cpp`]

\[
\operatorname{ACF}(u)=\frac{1}{|S|}\int_S
(f^{(U)}(s)-\mu)(f^{(U)}(s+u)-\mu)\,ds.
\]

作者先把 normalized ACF 输入曲线 `T`，再据其分布采样 patch。默认用户控制是从 `(0,0)` 到 `(1,1)` 的 cubic Bézier，两个中间控制点可移动；给定原 ACF 值，通过 Newton method 找曲线上对应点。Fig.9 的展示配置是 `T(x)=x^6`。[P §4.1, §4.3, Fig.3, Fig.9][S §1.5]

位移路径与 appearance decoder 分离：height map 经同类 dynamic synthesis 后，在 implicit shell 中先 traverse coarse level；命中 coarse cell 后，从该 cell 起点进入 fine level，最多 32 步，fine hit 为最终位置。这样避免训练 4D offset network 和额外 5D alpha network。[P §3.3, §4.4][S §1.6, Fig.2]

### 5.2 持久化表示

| 组件 | 正式配置 | shared/per-asset | locator |
|---|---|---|---|
| U plane | measured `400×400×8`；synthetic `800×800×8`；FP32；average mipmap 用于 appearance LoD | per material | S Table 1；P §4.1, Fig.7 |
| H plane | `50×50×8`，FP32 | per material | S Table 1 |
| D plane | `50×50×8`，FP32 | per material | S Table 1 |
| Decoder | `24→32→32→32→3`，四个 QL，无 bias | per material | P Fig.4；S §1.1 |
| Quantization scales | 每层 activation 一个、weight 一个，共 8 个 scalar；renderer 中 hard-coded | per material/config | S §1.1, §1.4；C-R `NeuralMatRendering.h` |
| Synthesis transforms | Gaussian transform plane、inverse LUT、ACF/PDF/sample map；dual square tiling | per U plane | P §4.3；C-R `Synthesis.cpp`, `Inference.cu` |
| Height field | 2D source height map及 coarse/fine max-filtered LoD | per material | P §4.4；S §1.6 |

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | precision | locator |
|---|---|---|---|---|---|---|
| Plane fetch | `u`、`h`、`d` | 各自 bilinear lookup；每 plane 8 channels | 未使用 learned normalization | `8+8+8=24D` | planes FP32；runtime 以两层 `float4` CUDA texture/plane 存储 | P Fig.4；S §1.1, §1.4 |
| QL1 | 24D | `24×32` | input/weight per-tensor signed Int8；Int32 accumulate；dequant + ReLU | 32D | activation 下一层前再量化 Int8 | P §4.2, Fig.4；S §1.1, §1.4 |
| QL2 | 32D | `32×32` | 同上 | 32D | 同上 | P Fig.4；C-R `Inference.cu` offsets 192… |
| QL3 | 32D | `32×32` | 同上 | 32D | 同上 | P Fig.4；C-R `Inference.cu` offsets 448… |
| QL4 | 32D | `32×3` | Int32 accumulate、dequant；Fig.4 的通用 QL 方框画有 ReLU，但 §4.2 的公式只把 `x_(n+1)` 称为 `n=4` 时的 BTF output，未单独写 final activation；release CUDA 明确 final ReLU | RGB | FP16 或 FP32 dequant output | P §4.2, Fig.4；C-R `Inference.cu` |

按 Fig.4 topology 共有 `24×32 + 32×32 + 32×32 + 32×3 = 2,912` 个无 bias scalar weights，即 728 次 `dp4a` 四元素 dot product。这是根据正式 topology 的算术，不是论文报告的 FLOP 统计。[I/C]

### 5.4 条件化、坐标变换与物理先验

- Rusinkiewicz `h/d` 把 BRDF 的角向结构放到两个二维方向 planes；Shirley map 使半球方向进入方形 lookup。[P §4.1][S §1.3]
- QTP 没有跨材质 latent、material encoder、hypernetwork 或 shared weights；“comprehensive”指功能覆盖，不是跨材质泛化。[P §3–4][C-R demo assets]
- Height field 是显式几何先验。作者指出 offset 的 position-direction correlation 会让位置 plane 分解失去清晰语义，所以没有把 offset 强行塞进相同 neural decomposition。[P §3.3, Fig.2]
- Appearance LoD 只对 U plane 做 average mipmapping；height tracing LoD 则做 max filtering，二者用途和 filter 不同。[P §4.1, §4.4]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Measured assets | UBO2014 的 `LEATHER04`（Fig.7）与 `LEATHER11`（Fig.1/8/12），每个 22,801 张、每张 `400×400` | S §1.2；P §5 |
| Measured artifact | supplemental video 的 Leather11 relighting 有 texture-sliding-like issue；作者推测来自使用 UBO2014 的 non-parallax-corrected Leather11 | S p.1 footnote 1 |
| Synthetic assets | 除上述两种 measured BTF 外，论文/video 的其它材质由作者生成；`ω_o`、`ω_i` 分别从 unit square 上均匀 `10×10` 点经 Shirley map 得到，共 `10^4` direction pairs，每张 `800×800` | S §1.2 |
| Synthetic GT | NVIDIA Falcor Standard Material（microfacet-based）记录 `f(u,ω_o,ω_i)`；具体材质参数、lighting-free export path、sample count 与 numeric precision 未报告 | S §1.2 |
| Position queries | 每个方向图覆盖完整 `400×400` 或 `800×800` spatial grid；release loader 给每个 texel 加一个 cell-size 内 random offset | S §1.2；C-T `DatasetUBO.__getitem__` |
| Direction queries | measured 使用 UBO2014 全部 angle dictionary；release 预计算 `h/d` 后映射到 Shirley square | C-T `dataset.py` |
| Formal split | 未报告 train/validation/test asset 或 angle split，也未报告 figure query manifest | P/S 未报告 |
| Code split | `train_qtp.py` 的 train 与 validation 都构造同一完整 UBO dataset；`testing=True` 只把长度改为 1，因此 test 输出第一 angular image，不是 held-out test | C-T `train_qtp.py`, `dataset.py` |
| Filtering/LoD | appearance：U plane average mipmap；height hierarchy：max-filtered coarse/fine；direction planes无 LoD protocol | P §4.1, §4.4, Fig.7 |
| Online/offline | BTF/feature planes/ACF/transform/LUT 均离线训练或 precompute；runtime 是随机访问 lookup + decoder + 可选 synthesis/tracing | P §4；S §1.4 |

Fig.6/8 的 GT/reference 是 measured 6D table 的 slice 或 offline interpolation/render；Fig.8 明确说明 tensor decomposition 和 reference 都从 6D data table 离线插值并渲染。论文没有提供相机、光照、tone mapping、PSNR/FLIP 计算域与 mask 的完整 manifest。[P Fig.6, Fig.8]

## 7. Loss、optimizer 与训练 lifecycle

### 7.1 正式论文/supplemental 配置

| 项 | 正式配置 | locator |
|---|---|---|
| Target transform | 一般材质直接 RGB；极高动态范围 glossy material 在训练前用 `log(x+1)`；完整 inverse transform/runtime correspondence 未报告 | P §4.2, Fig.11 |
| Loss | output MAE；feature planes 与 QTP joint training | P §4.2 |
| Quantization | PyTorch + Brevitas；weight 和每层 input activation 都做 symmetric signed Int8 per-tensor QAT；no bias | S §1.1 |
| Optimizer | AdamW，并称 follows TP setting；betas、weight decay、epsilon 未报告 | S §1.3 |
| LR | network 与 neural textures 同为 initial `5e-4`、minimum `5e-5` | S §1.3 |
| Schedule | cosine annealing with warm restarts；first restart epoch 20，间隔每次 ×2，因此 epochs 60、140，下一次在 300 边界 | S §1.3, Fig.1 |
| Epochs/time | 最多 300 epochs；单 RTX 4090 约 18 h；作者允许依 loss 早停或继续 | S §1.3 |
| Batch/seed/model selection | 未报告；没有正式 checkpoint identity 或 repeated-seed statistics | P/S 未报告 |

Supplemental Fig.1 是一个明确的训练负结果：TP 的 exponential LR decay 在约 50 epochs 后 loss 停在较高平台；warm restarts 虽有周期性跳升，之后继续下降到更低 loss。图没有给原始数值、seed、材质名或方差，因此只能证明该展示 run 的优化轨迹，不能外推成普遍胜率。[S Fig.1]

### 7.2 官方 training code default

`train_qtp.py` 当前 default 是 U `400×400×16`、H/D `50×50×8`、hidden 32、batch size 1（一个 batch 是完整 `400×400` direction image）、50 epochs、planes LR `5e-4`、MLP LR `3e-4`、seed `20260410`。它使用 `CosineAnnealingLR(T_max=epochs, eta_min=1e-5)`，不是 warm restarts；而且同一次 epoch loop 在保存后和记录 LR 后各调用一次 `scheduler.step()`，即每 epoch 两次。[C-T `train_qtp.py`]

因此这些 default 是 release example，不是正式配置。更严重的 export/runtime correspondence 见 §11。

## 8. Inference、部署与成本

### 8.1 正式 runtime path

论文因 shader language 对 low-bit arithmetic 支持不足，在 Falcor 中用 custom CUDA kernel 做 full-screen large-batch、thread-level inference；作者认为 CUTLASS 等 GEMM library 不适合该工作负载，因为没有相应 thread-level API 或 Int8 path。每个 block 先把 weights 放进 shared memory；方向/位置输入压为 FP16 以减带宽；每四个 Int8 元素 pack 到 Int32，用 CUDA `dp4a`；层输出 Int32，再 dequant 到 FP16、ReLU、requant。Feature planes 是 two-layer `float4` CUDA textures，scale 是 constant/hard-coded value。[S §1.4]

没有 synthesis 时，kernel 对 H/U/D 各做两次 `float4` fetch，共恢复 24D。开启 synthesis 时，U feature 由 dual patch、Gaussian-domain blend 与 inverse LUT 恢复；论文“3→2 texture fetch”只指 dual square tiling 相对 hex tiling 的 patch fetch，不是整个 decoder 只有两次 texture access。[P §4.3][C-R `Inference.cu`]

### 8.2 Storage

Supplemental Table 1 报告：

| 组件 | measured BTF | synthetic material | precision |
|---|---:|---:|---|
| U plane | 4.88 MB (`400×400×8`) | 19.53 MB (`800×800×8`) | FP32 |
| H plane | 78.12 KB | 78.12 KB | FP32 |
| D plane | 78.12 KB | 78.12 KB | FP32 |
| network parameters | 5.68 KB | 5.68 KB | 表中未解释为何高于 2,912 Int8 scalar 的 2.84 KiB |

按表内数字直接相加，每材质约为 5.04 MB（measured）或 19.69 MB（synthetic）；U plane 分别约占 96.8% 与 99.2%。这说明论文的速度贡献来自 decoder arithmetic 量化，而总表示容量几乎全部仍在 FP32 spatial plane；这两个比例是本报告算术 `[I]`，不是作者结论。

### 8.3 Fig.5 standalone full-screen inference

RTX 4090、1 SPP、standalone program；TP time 以启用/禁用 inference 的差值测量，QTP 包含 I/O 和 computation。单位 ms：[P §5.1, Fig.5]

| Resolution | TP + histogram synthesis | TP no synthesis | QTP + ACF synthesis | QTP no synthesis |
|---|---:|---:|---:|---:|
| `3840×2160` | 11.01 | 9.03 | 0.988 | 0.754 |
| `2560×1440` | 4.87 | 3.94 | 0.465 | 0.335 |
| `1920×1080` | 2.75 | 2.20 | 0.263 | 0.183 |

Fig.5 还把 1K、no-synthesis QTP 的 0.183 ms 分成 0.146 ms computation 与 0.037 ms I/O。该范围不是本项目的 isolated single-query latency：它受 full-screen coherence、CUDA launch、valid-pixel mask 与一次 material batching 影响。[P Fig.5][C-R `Inference.cu`]

### 8.4 Fig.12 renderer end-to-end component timings

Fig.12 在 RTX 4090、`1920×1080`、1 SPP/frame 的 renderer 内逐材质报告。标签中的 `Inference` 与 `Tracing` 是分项；`Tracing w/ HF` 同时含 height-field synthesis/tracing，但论文未给出 frame 其它成本。[P §5.1, Fig.12]

| Material | TP inference syn/no-syn | QTP inference syn/no-syn | no-displacement tracing | QTP tracing w/HF syn/no-syn |
|---|---:|---:|---:|---:|
| Tile | 2.49 / 1.89 | 0.23 / 0.16 (`10.8×/11.8×`) | 0.30 | 1.65 / 1.21 |
| Ceramic Tile | 2.19 / 1.83 | 0.25 / 0.17 (`7.5×/10.7×`) | 0.29 | 1.30 / 1.03 |
| Weave | 1.13 / 1.03 | 0.11 / 0.09 (`10.2×/11.4×`) | 0.30 | 2.95 / 1.65 |
| Leather11 | 1.63 / 1.32 | 0.14 / 0.11 (`11.6×/12×`) | 0.31 | 1.29 / 1.05 |

Fig.12 在 Ceramic Tile 的 synthesis case 上印有 `7.5×`，但同一行显示的 `2.19/0.25 ms` 相除约为 `8.76×`；no-synthesis 的 `1.83/0.17≈10.76×` 与 `10.7×` 标签相符。本报告保留原始时间和作者标签，并把前一项记为 **figure-internal arithmetic gap**，不自行用推算值改写图中 headline。[P Fig.12][I]

作者没有与 Zeltner et al. 2024 做 measured performance comparison，理由是后者用了 modified LLVM-based DirectX compiler 与 custom Tensor Core intrinsics，pipeline/backend 不同。正文随后关于“unoptimized theoretical complexity”的数量级说法没有 matched implementation 数据，不能拿来排名。[P §6]

## 9. 实验 protocol、baseline、指标与结果

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| Full-screen performance | RTX 4090，1 SPP，standalone，4K/2K/1K；TP 用 enable-disable difference | Triple Plane FP32，有/无 synthesis | average runtime | QTP 0.988/0.465/0.263 ms（有 synthesis），0.754/0.335/0.183 ms（无）；TP 对应 11.01/4.87/2.75 与 9.03/3.94/2.20 ms | P Fig.5 |
| Material slice recovery | 直接显示同一 2D slice；具体 split 未报告 | NeuMIP、TP、GT | PSNR/FLIP | NeuMIP 35.06/0.089，TP 35.20/0.064，QTP 35.79/0.066；QTP/TP 都保留 pebble edge highlight，NeuMIP 丢失较多 | P Fig.6 |
| U-plane LoD | Leather04；U plane 从 400² 降至 200²/100²/50²/25²，QTP 与对应 GT mip 对比 | origin/GT | PSNR/FLIP | 34.82/.079，37.09/.073，37.11/.078，34.14/.101，30.83/.134；中间 LoD 反而有更高 PSNR，因此不能把每级当单调误差曲线 | P Fig.7 |
| Rendered BTF recovery | Leather11，无 synthesis（UV tiling）；reference 与 tensor decomposition offline interpolate/render | tensor decomposition、NeuMIP、TP、reference | PSNR/FLIP | tensor 34.57/.070，NeuMIP 32.62/.192，TP 35.39/.075，QTP 36.11/.063；QTP 在一侧 grazing 有误差，但另一侧小于 TP | P Fig.8 |
| Structured synthesis | Ceramic Tile；同一 material，Fig.9 的 curve `T(x)=x^6` | UV tiling、histogram synthesis、original ACF | visual structure/repetition | UV 完全重复；hist 完全随机；original ACF 保留一些结构但仍破坏；Curved ACF 在展示例中最好地保留 grid pattern 且无显式重复 | P Fig.9 |
| Height tracing | Tile/Fabric；two-step 与 small-step ray marching，含/不含 synthesis 分报 | no displacement、ray marching | ms + visual | Tile：no disp .39；two-step 1.34/.99；ray 3.88/1.70。Fabric：.39；two-step 3.40/1.71；ray 6.29/2.47。two-step 比 ray marching 快，但相对 no displacement 仍增加约 0.60–3.01 ms | P Fig.10 |
| Accumulation precision | glossy Metal，1K，no synthesis | FP16 vs FP32 dequant/activation accumulation | visual + ms | FP16 0.10 ms 但 highlight 有明显量化损失；FP32 0.17 ms 修复该展示 artifact | P Fig.11 |
| Full combined system | 4 materials、1K、1 SPP renderer | TP + hist synthesis；QTP + Curved ACF；再加 HF | 分项 ms + visual | QTP inference 为 TP 的展示加速 7.5×–12×；HF 路径的耗时和材质 height distribution 明显相关，见 §8.4 | P Fig.12 |
| LR schedule | 单个未具名 run；未报告 seed/variance | TP exponential decay vs warm restarts | training MAE curve | exponential curve在较高 loss 停滞，warm restarts 最终继续下降 | S Fig.1 |

“comparable quality”只在这些材料/视图与 PSNR/FLIP 图例上成立。没有完整 test manifest、跨材质统计、置信区间、相同 storage budget 或多 seed；因此不能把 Fig.6/8 的单例差异当成 QTP 普遍优于 TP/NeuMIP。

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释 | locator |
|---|---|---|---|---|---|
| `author-negative` | 对 offset function 沿 position/direction 做与 color 相同的 decomposition/synthesis | decomposed positional offset feature 模糊、无有用语义；合成失败 | 同一 reference position 随 view direction 覆盖不同 meso-geometry，造成强 position-direction correlation；offset blending 也非平凡 | 这是对“所有 6D 分量都可拆成 U/H/D”的直接反例；不能把位移 proxy 当普通可滤 latent | P §3.3, Fig.2 |
| `author-negative` | TP exponential LR decay | 展示 curve 较早停在高 loss | schedule 收敛能力不足；warm restarts 改善 | 只有单 run 证据，仍是有价值的 optimizer protocol 候选，不是普遍定理 | S Fig.1 |
| `ablation-inferior` | histogram-preserving dynamic synthesis 的随机 patch selection | structured tile 的 grid 被完全打乱 | 随机 selection 不考虑原结构 | blending 本身未被判失败；问题位于 proposal/selection distribution | P §3.2, Fig.9 |
| `ablation-inferior` | original ACF sampling | 相对 histogram 更保结构，但 Fig.9 仍有 grid disruption | 对 highly structured pattern，原 ACF 概率不够集中 | curved ACF 是对 selection sharpness 的手工可调校准，不是 learned universal prior | P §4.1, Fig.9 |
| `known-limitation` | FP16 dequant/activation on glossy Metal | highlight dynamic range 发生可见 loss | 即使 target 用 `log(x+1)`，FP16 中间值仍不足；FP32 修复但更慢 | 量化部署必须按材质 tail 分层，不能只看 average PSNR | P §6, Fig.11 |
| `known-limitation` | QTP at grazing angles | Fig.8 一侧有局部 error，另一侧又小于 TP | 作者称不是一致性 error | 缺少方向分层统计；当前证据不能定位是 coordinate、QAT、reference interpolation 还是训练覆盖 | P Fig.8 |
| `known-limitation` | U-plane high LoD | LoD3/4 出现轻微差异，LoD4 PSNR/FLIP 明显变差 | 远处使用，作者认为 blur 不显著 | 需要 footprint-conditioned query 评测，不能仅用远景主观不可见作为滤波正确性 | P Fig.7 |
| `known-limitation` | linear ray in shell + only two LoD | 更快但不处理 nonlinear shell path；动态 tracing hierarchy 仍可改进 | 作者列为 future work | 这条机制只适合显式 height field material，不能覆盖任意原生 source graph | P §7 |
| `known-limitation` | Leather11 source | supplemental relighting 有 texture sliding | 作者推测使用了 non-parallax-corrected UBO data | source identity/校正版本必须冻结，不能把 acquisition artifact 归咎于 network | S footnote 1 |
| `paper-code-gap` | public training/export/runtime | formal config 无法由当前 defaults 直接连通 | README 只称 reference code，未给 migration | 具体静态缺口见 §11；不据此否定论文结果，但 release 不能视为 formal reproduction package | C-T/C-R |

正文 §3.3 称 offset+alpha 两个 network 在 shader 约增加 2.43 ms，§5.1 又称两个 same-complexity dummy networks 约 3.4 ms；两处都是 1 SPP、1K，但没有说明 topology、precision、是否含 I/O 或为何不同。本报告保留为 **paper-internal runtime protocol gap**，不择一合并。[P §3.3, §5.1]

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Plane channels | Fig.4 明示 U/H/D 各 8，concat 24 | Table 1 三者 channels=8 | `train_qtp.py`、`export_weights.py`、README default 为 U=16、H/D=8；renderer `IN_NUM=24` 且每 plane 两层 float4 | **paper-code-gap**：training/export default 是 32D input，renderer 固定 24D；不能直接互换 |
| Topology/activation | `24→32→32→32→3`；Fig.4 通用 QL 方框含 dequant+ReLU，但 §4.2 公式没有为 final output 单独写 activation | signed Int8、no bias | training final `LeakyReLU`；CUDA final `dequantize..._relu`；hidden 为 ReLU | **paper-internal ambiguity + paper-code-gap**：runtime 支持 final ReLU，training final response 却不同；不能只凭示意图消除差异 |
| Network storage | S 不在正文 | Table 1 报 5.68 KB | 正式 2,912 Int8 scalar 理论为 2.84 KiB；code 未解释额外一倍 | 未解析；不能用理论算术覆盖表值 |
| Training schedule | MAE joint train | AdamW、`5e-4→5e-5` warm restarts、最多 300 epochs | default planes `5e-4`、MLP `3e-4`，plain cosine `eta_min=1e-5`、50 epochs，且每 epoch 两次 `scheduler.step()` | 正式值与 default 是 **paper-code-gap**；重复 step 是未解释的 **release-code anomaly / suspected-defect**，只适用于该示例 loop，不外推到论文内部训练 |
| Glossy log target | `log(x+1)` for highly glossy | 未补充 inverse path | `train_qtp.py` 没有 log target，CUDA output 没有 exp/inverse transform | **paper-code-gap**：Metal formal checkpoint lifecycle 不可由公开 path 确定 |
| Split | 未报告 | 未报告 | train/validation 同一 dataset；test 是第一 angular image | code 不能提供 held-out evidence；paper figures 的 selection 未解 |
| Direction mapping | Rusinkiewicz + Shirley | 明示训练 lookup 用 Shirley | `dataset.py` 实现 half/difference 与 Shirley square | 语义对应；边界/seam tests 未提供 |
| Feature storage | three FP32 planes | two-layer float4 CUDA texture | training exporter 可导 FP32 planes | 大方向对应；但 exporter 输出 `PlaneInfo_*`，renderer只读取 `PlaneMeta_*`，仓库中无 rename/migration step | **release-helper contract mismatch**；文件 payload 语义可对应，文件名不能直接连通 |
| Int8 weight export | signed Int8, pack 4→Int32 | `dp4a` | exporter 对 quant weight 调 `.int()` 后以 NumPy Int32 raw bytes 写出；renderer通用 `readBinaryFile` 把同一 raw bytes解释为 float，再 numeric cast 为 int并重新 pack | **release-helper dtype contract mismatch**；若直接连接两个公开 helper，数值语义不相容；正式 asset 可能由未公开工具或转换步骤生成，故不推断正式 checkpoint 错误 |
| Paper 24D export | 24D first layer | — | `exportModelWeightsInt8` 将各 output row `extend` 进 list 后直接 `np.array`；paper-config 下第一层 row width=24、后层=32，当前代码没有 flatten/padding/layout step | **release-helper static blocker**：paper topology 的公开 export path 未定义 rectangular array；不据此判定论文内部 exporter |
| Custom data | 允许 synthetic/custom | synthetic recipe | `DatasetCustomized` 在读取首图并设置 `height/width` 之前访问两者来创建 UV grid | **release-example defect**：该 loader 按当前初始化顺序不可用；它不是论文 formal synthetic-data 参数的证据 |
| Runtime kernel | custom CUDA full-screen | weights shared-memory、`dp4a`、FP16/FP32 dequant | renderer kernel硬编码 24/32 topology、16×16 blocks、六材质量化 scales | 正式 topology 大体对应；asset pack/checkpoint 未审计，无法做 binary parity |
| Synthesis | Curved ACF + dual square tiling + histogram-preserving blend | Bézier/Newton | renderer计算 ACF、维护 curve UI、sample map、Gaussian transform/inverse LUT；ACF 只取 U plane 首个 scalar channel | 核心机制可追踪；首 channel 是未在论文中定义的 release convention，paper figure 的 control points/seeds 除 `x^6` 例外未报告 |

以上差异来自固定 commit 的静态 correspondence 审计，尚未在外部 asset pack 上执行。其中只有重复 scheduler step 与 custom-loader 初始化顺序标为 release-code anomaly/defect；U16 default、formal schedule、final activation、文件名、dtype 与 24D layout 属于 paper↔release correspondence 或 helper contract 缺口。它们共同证明“公开 helper 不能直接作为 paper-config build recipe”，不证明作者内部用于论文的训练/导出资产错误。

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

- Int8 并非无损：glossy/high-dynamic-range material 的 FP16 dequant 会损失 highlight；FP32 更好但更贵。[P §6, Fig.11]
- QTP 的目标是综合功能与速度，不追求 Zeltner et al. 所称 film-quality；作者承认 low-bit 具有 quality trade-off。[P §5]
- Curved ACF 依靠用户选曲线；论文没有自动选择、学习或跨材质泛化曲线。[P §4.3]
- two-step tracing 只追 linear ray in shell，未采用 nonlinear shell tracing；作者希望未来有更高效 dynamic hierarchy。[P §7]
- importance sampling 未研究。[P §4.1, §7]
- 动态 tracing 开销依 material height distribution 变化，不能用单个 aggregate 代表。[P §5.1]
- Leather11 video 存在疑似 source parallax-correction artifact。[S footnote 1]

### 12.2 未报告/材料不可得

- formal checkpoint、paper figure 的 config/seed、asset manifest、train/test split 与 raw metric files；
- AdamW betas/weight decay/epsilon、batch/query 数、early-stop/model-selection rule；
- synthetic Standard Material 的完整参数、GT extraction path、sample count、precision；
- ACF 从 8-channel U plane 汇聚到 scalar proposal 的完整论文定义、curve presets/seeds；
- PSNR/FLIP 的 color space、tone mapping、crop/mask、view/light manifest；
- Table 1 network 5.68 KB 与 2,912 Int8 weights 算术的对应；
- 2.43 ms 与 3.4 ms two-network overhead 的 protocol 差异；
- single-query latency、warp occupancy、register/shared-memory pressure、不同 GPU、mobile/shader backend；
- energy conservation、reciprocity、transmission、`sample()/pdf()`；
- public SharePoint asset pack 的 immutable hash/version，以及它与论文正式 checkpoint 的对应。
- Fig.12 Ceramic Tile synthesis speedup 标签 `7.5×` 与显示时间 `2.19/0.25 ms` 的算术差异。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

QTP 的“低比特”只覆盖 decoder weights/activations。按 Supplemental Table 1，measured material 约 96.8%、synthetic material 约 99.2% bytes 位于 FP32 U plane；H/D 也未量化。换言之，它把 high-frequency spatial identity 显式存进 dense plane，用极小 MLP 做跨 plane mixing。对于当前项目，这更接近 **per-asset spatial neural texture + tiny decoder**，而不是“用 Int8 MLP 编译任意原生材质”。

这也解释了为什么论文能在单材质 full-screen coherent workload 上极快：容量和插值已前移到 texture hardware，MLP 只有 2,912 scalar weights。但当前项目的第一源材质族是可编辑层栈状态，formal evaluator 要在未见 material/state 上泛化；若为每个状态训练 5–20 MB plane，就改变了 compiler 身份和 G1/G2/G2s 问题。

### 13.2 成功所依赖的假设

1. `f_r(u,h,d)` 可被三个 2D planes + tiny decoder 分解；
2. position plane 的 semantic content 足够稳定，允许 mipmap、Gaussian transform、ACF selection 与 patch blending；
3. 同一材质在 full-screen 中形成足够大的 coherent batch，custom CUDA kernel 能摊薄 launch/weight load；
4. source 有显式 height field，且 appearance 与 height 可分别合成后重新对齐；
5. Int8 per-tensor scale 足以覆盖大多数材质，例外可切换 FP32 accumulation。

论文自己的 offset negative result 正好标出假设 2 的边界：只要某个量把 position 与 direction 强相关地纠缠，空间 plane 就不再是可独立合成/过滤的 latent。

### 13.3 可迁移机制与不能迁移的部分

可迁移：

- 对 **固定小 decoder** 做 matched QAT，而不是继续压 width；
- 把 decoder compute、I/O、feature bytes 分开计量；
- `prepare()` 对可复用 spatial/material latent 做 footprint filtering，并把 appearance average mip 与 geometry max hierarchy 分开；
- 对极端动态范围按 state/material 做 precision fallback，而不是全局假设 FP16/Int8 足够；
- 把 optimizer schedule 当独立轴，特别是 compact network 的 plateau/restart 行为。

不能直接迁移：

- U/H/D per-material planes 不能替代当前跨 state compiler；
- Curved ACF 解决的是 texture synthesis patch proposal，不是 BSDF importance sampling；
- height-field shell tracing 只适合原生提供 height field 的 source family；不得要求层栈或任意 MDL 先反演成 height field；
- full-screen CUDA timing不能当作 Slang `evaluate()` 的随机单 query latency。

### 13.4 与本项目 runtime contract 的关系

QTP decoder 的 topology、plane fetch 数与 per-query work 静态有界，理论上能实现随机访问 `evaluate(wo,wi)`。但只有 U lookup与 spatial LoD 可放进 `prepare()`；`h/d` 同时依赖 `wo/wi`，多 incident-direction query 时必须在 `evaluate()` 重新构造和 fetch。Synthesis sample map 与 height tracing 又属于 texture/geometry workflow，不应塞进每次 BSDF evaluator。

论文没有 `sample()/pdf()`，因此最多贡献 evaluator/feature-filter/quantization 候选；matched sampler 仍须按本项目合同另行训练和验证。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

当前 NVIDIA formal identity 是 `nvidia-rta2024-functional-f@2`：每个已编译 source asset 持有两张 RGBA16F latent mip chains，shared evaluator 为 `20→64→64→64→3` 并固定 `exp(raw-3)` response，另有独立 `11→32→32→32→9` proposal；formal config 通过 GPU online reference query 训练，而不是离线持久化 BTF corpus。[N `configs/learning/nvidia-rta2024-materialx-formal.json`; `src/ncls/learning/methods/nvidia.py`; `src/ncls/learning/models/nvidia_neural_appearance.py`; `shaders/ncls/backends/nvidia_neural_appearance/`]

| QTP 证据 | 当前 NVIDIA 状态 | correspondence 分类 | 影响 |
|---|---|---|---|
| Int8 QAT 可在 full-screen BTF workload 把 tiny decoder 降到 0.09–0.25 ms/1K | 当前 formal identity 不是 Int8，且有固定 FP response | `intentional-deviation` 候选 | 只能建立新的 quantized identity 做 matched control，不能回写成 faithful 2024 reproduction |
| QTP 不继续缩网络，而改变 arithmetic precision | 当前 evaluator 64-wide，Taming 报告又显示 compact optimization 本身有 variance | `budget-adaptation` 候选 | 先稳定训练/quality，再测 FP16/Int8；否则会把优化失败误判量化失败 |
| FP16 glossy highlight 失败、FP32 修复 | 当前输出有指数 response，峰值同样可能放大量化误差 | `runtime-risk` | 必须按 high-tail state/方向分层报告，不只看 aggregate median |
| U-plane average mip 支持 LoD | 当前两条 latent mip chain 已有 footprint LOD 与 stochastic adjacent-level selection | `faithful-adjacent` | 可测试 latent mip 的 filter/quantization，但 QTP 没有证明当前 native feature pyramid 的语义等价 |
| QTP 每个 BTF/material 独立拟合、FP32 planes 占主要 bytes | 当前 formal package 虽也有 per-asset latent，但它由 source adaptation/compiler lifecycle 生成，并受 shared network、online query、unseen source/state protocol 与 `B_asset` 合同约束 | `not-applicable` 于主 compiler identity | 不能因两者都有 texture latent 就视为同一方法；若引入 QTP，只能作为 per-material BTF control 或显式 per-asset cache/capacity diagnostic，并单报 `B_asset` |
| 无 sampler | 当前 NVIDIA 有 `11→32→32→32→9` proposal | `not-applicable` | QTP 不改变 sampler correspondence，也没有证明 quantized evaluator 与 proposal 匹配 |

最直接的改进方向不是复制 Triple Plane，而是：在当前 formal evaluator 和 query protocol 不变的条件下，把 decoder precision 作为部署轴；同时保留 latent bytes、fetch 数、prepare/evaluate 分工与 output response 的完整 parity。

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H-CNM-1：对已稳定收敛的当前 evaluator 做 per-tensor Int8 QAT，可在不改 topology/query 的情况下改善单查询成本 | QTP Fig.5/12 的 7.5×–12× full-screen speedup | Slang/CUDA target 有可部署低比特 dot path，且当前 response tail 可校准 | FP baseline vs QAT；同 checkpoint selection、同 source/query/seed budget；另设 PTQ 诊断 | architecture、latent、loss、train queries、sampler冻结 | 四层 quality、peak-tail slices、`C_eval`、package bytes、真实 shader time | evaluator deployment | 若真实 backend 无 speedup，或任一预冻结 quality/finite/parity contract 失败，则否定该量化配置 |
| H-CNM-2：quantization loss 主要集中在 high-dynamic-range direction/state，可用受限 precision fallback 获得更好 Pareto | Fig.11 FP16 vs FP32 Metal | 当前层栈也存在稀疏高峰，且可由 `prepare` state statistic 预测 | global FP16、global FP32、固定规则 mixed precision 三者；禁止 learned oracle routing | model/queries/budget/response相同 | peak-weighted error、p95、tail recall、branch rate、time | prepare-conditioned evaluator | fallback 不能在 matched time/quality Pareto 上胜过 global precision，或 routing 对 unseen state 不稳 |
| H-CNM-3：warm-restart schedule 能降低 compact evaluator 的 plateau/seed variance | S Fig.1 单 run；Taming 另有 compact optimization variance 证据 | 当前 optimizer plateau 与该 schedule 机制同类 | 原 schedule vs frozen cosine warm restart；相同多 seed、step count、batch/query recipe | model、loss、data、seed set、total steps | seed success rate、bootstrap CI、curve AUC、final quality | training-only | 多 seed CI 不改善或相同预算下 median/tail 更差 |
| H-CNM-4：只过滤可复用 latent、方向 query 保持原分辨率，可改善 footprint LoD 而不破坏窄峰 | Fig.7 U-plane average mip；论文把 appearance LoD 与 height max hierarchy分开 | 当前 `prepare()` latent mip 是可滤 material state，而不是 view-light-correlated offset | 当前 latent mip vs matched learned/analytic prefilter；同 evaluator和 footprint distribution | source、query、network、fetch budget | footprint reference error、远近分层、峰能量、texture reads | `prepare()` / fixed-fetch | 相对无滤波在 matched footprint GT 上不改善，或窄峰/边界系统性变差 |
| H-CNM-5：若未来 spatial material family 的 U latent 有稳定 ACF，Curved ACF 可作为 runtime synthesis proposal | Fig.9；offset decomposition 失败给出反例边界 | native source 的空间 latent与方向足够解耦，且合成目标允许 stationarity | random histogram、original ACF、fixed curve ACF；同 patch count/blending | source patch、decoder、curve set、seed、fetch | structure spectrum、seam、direction consistency、runtime | optional per-asset synthesis | 不同方向下结构漂移、source-native edit语义破坏，或无 matched quality gain |

这些假设都不是当前任务的 hard gate；只有进入正式 candidate experiment 时，才按 `experiment_framework.md` 冻结配置并报告 bootstrap CI。

## 16. 证据索引

- `P §1–3 / Fig.1–2`：四个目标轴、prior boundary、offset decomposition failure 与 2.43 ms 陈述。
- `P §4.1 / Eq.1 / Fig.3–4`：U/H/D 分解、ACF、QTP topology/quantization 与 LoD 假设。
- `P §4.2–4.4`：MAE、log transform、Curved ACF、dual tiling、histogram blending、two-step tracing。
- `P §5 / Fig.5–12`：性能、质量、LoD、synthesis、tracing、precision limitation 与 full system。
- `P §6–7`：quantization/network/backend comparison 边界、importance sampling 与 nonlinear tracing future work。
- `S §1.1 / Table 1`：Brevitas QAT、precision、scale/no-bias、plane 与 network storage。
- `S §1.2–1.3 / Fig.1 / footnote 1`：measured/synthetic data、Shirley queries、warm restarts、300 epochs/18 h、Leather11 artifact。
- `S §1.4–1.6 / Fig.2`：CUDA `dp4a` path、Bezier/Newton、coarse/fine 32-step tracing。
- `C-T networks.py/train_qtp.py/dataset.py/export_weights.py`：release topology/defaults、split、scheduler、export correspondence 与 static defects。
- `C-R Inference.cu/NBTF.cpp/MLPCuda.cpp/Synthesis.cpp/NeuralMatRendering.h`：hard-coded 24/32 runtime、feature/weight I/O、ACF synthesis、scales 与 material asset wiring。
- `A project page`：正式 DOI/PDF/supp/code/video links；旧仓库名重定向。
- `N configs/learning/nvidia-rta2024-materialx-formal.json; src/ncls/learning/methods/nvidia.py; src/ncls/learning/models/nvidia_neural_appearance.py; shaders/ncls/backends/nvidia_neural_appearance/`：当前 `functional-f@2` identity、formal online-query recipe、runtime package 与 response；归档 `correspondence.md` 只作历史 provenance。
- `N docs/realtime_material_compilation.md; docs/research/experiment_framework.md; docs/research/model_candidates.md`：当前 runtime contract、候选身份与评测边界。
- `I`：§4 输出语义边界、§5 parameter arithmetic、§8 capacity arithmetic、§13–15 迁移分析；均不反写成作者结论。

## Evidence review

```text
author_worker: /root
reviewer: /root/nbrdf2021
reviewed_at: 2026-08-29
sources_rechecked:
  - main PDF SHA-256 4CAAEAB2E55221ACE52D2F223796F7561CB68C1133BA52464FADB6E8B7A61D7D, all 11 pages, Eq.1, Fig.1-12 and captions
  - supplemental SHA-256 79E8E5626B7DBCE3E7E0AF9B7E8110C6F0E3C5624818A0AC079674056764DFC1, both pages, Table 1, Fig.1-2 and footnote
  - renderer commit dd6238576e43ce03780a4dc811b9523b9c277280
  - training commit b87fe3565b68f465e7fa621d7f006a50c76b16d8
findings_closed:
  - independently rechecked formal U/H/D=8 and 24->32->32->32->3 topology, QAT precision, storage arithmetic, Curved ACF/Bezier/dual-tiling and two-step height tracing
  - corrected Fig.5 4K QTP+ACF timing from 0.980 to 0.988 ms and rechecked all Fig.5-12 values against rendered pages
  - recorded the Fig.12 Ceramic Tile 7.5x label versus 2.19/0.25 ms arithmetic conflict without silently rewriting either value
  - separated paper final-activation ambiguity, training LeakyReLU and renderer final ReLU
  - reclassified U16 defaults, schedule/double-step, log response, PlaneInfo/PlaneMeta, Int8 dtype, 24D export and custom loader at their actual paper-code or release-helper scope
  - rechecked the Leather11 non-parallax-corrected-source footnote and kept it separate from network failure
  - updated NVIDIA impact to current functional-f@2 config/code evidence and preserved the per-material BTF versus compiler-identity boundary
remaining_evidence_gaps:
  - formal checkpoints/assets/configs and paper figure manifests not audited
  - public SharePoint asset pack has no frozen hash in this review and was not downloaded, so paper-checkpoint identity and binary parity remain unverified
  - Table 1 network storage arithmetic, Fig.12 Ceramic Tile speedup label arithmetic, and 2.43 ms versus 3.4 ms overhead conflict remain unresolved
  - public helper export/runtime path was not built or executed; static contract mismatches have no formal-asset end-to-end parity evidence
  - paper does not define how vector-valued U features are reduced for ACF; release code's first-channel convention is not formal evidence
  - formal output units, splits, optimizer details, seeds and metric definitions not reported
review_status: evidence-reviewed
```

### 完成检查

- [x] main paper 已完整阅读，关键公式/图/表/图注/脚注已视觉核对；
- [x] supplemental/appendix/勘误的可用性已检查；supplemental 两页已完整纳入，未发现 erratum；
- [x] official code/config/data 的可用性与 commit 已检查；renderer/training 已静态审计，外部 asset pack 未审计；
- [x] architecture、training、runtime 和主要结果均有 locator；
- [x] 失败尝试与较差消融正确分类；
- [x] paper/code gap 和“未报告”保留；
- [x] `I` 分析晚于事实层，没有改写作者结论；
- [x] NVIDIA 影响引用真实 `N` 证据；
- [x] 假设包含 matched control、部署类别和证伪条件；
- [x] 独立 evidence review 已完成。
