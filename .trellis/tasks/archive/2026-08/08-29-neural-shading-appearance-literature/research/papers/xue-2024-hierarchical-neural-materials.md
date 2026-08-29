---
paper_id: "xue-2024-hierarchical-neural-materials"
title: "A Hierarchical Architecture for Neural Materials"
authors: "Bowen Xue; Shuang Zhao; Henrik Wann Jensen; Zahra Montazeri"
year: "2024"
venue: "Computer Graphics Forum 43(6), e15116"
doi: "10.1111/cgf.15116"
report_status: "evidence-reviewed"
main_source: "https://diglib.eg.org/server/api/core/bitstreams/160e2378-c002-4c11-af34-57e5ad248d8d/content"
supplemental_status: "unavailable"
official_code_status: "audited"
official_code_commit: "a8978bc71034984121ebf7326c1a527e25238ca5"
author_worker: "/root/belcour2018_review"
reviewer: "/root/dualband2025_review"
last_verified: "2026-08-29"
---

# A Hierarchical Architecture for Neural Materials

## 1. 研究对象与报告边界

本文是在 NeuMIP 的**逐材质、空间变化、多分辨率 neural appearance** 框架上做的一次结构与训练目标改造。它保留 NeuMIP 的 view-conditioned neural offset 与 neural texture pyramid，把 pointwise MLP decoder 换成在完整二维 query buffer 上运行的 Inception convolution decoder，并为 `u, ω_i, ω_o` 增加 Fourier encoding；训练时再加入 Sobel gradient loss 与 HDR output remapping。[P §3–4, Figs.2–3]

正式输入是一条 7D query：二维 UV `u`、各用二维切平面坐标表达的 `ω_i/ω_o`，以及 footprint/prefilter kernel size `σ`；输出被作者称为对应的线性 RGB reflectance。[P §3.1, §4.1] 论文没有定义这个 RGB 是否为 bare BRDF、cosine-weighted BRDF、直接光照后的局部 radiance，或包含其他 normalization；代码还提供可选 `cosine_mult`。因此 exact scattering measure 保持未报告，不由 “reflectance” 一词补全。[P §4.1; C `dataset/dataset_reader.py:217–225`]

实验覆盖 KeyShot 合成的 cloth、displacement/height-map 材质、NeuMIP 数据与 UBO 2014 measured BTF。renderer 只验证 Mitsuba 2 direct illumination，所有正式比较图为 1 SPP。[P §4.1–4.3, §5]

它不提供：

- 任意源材质 family 的 compiler 或跨材质泛化；
- 独立 `sample()/pdf()`，importance sampling 或 global illumination；
- 与 query 顺序无关的 single-query evaluator；
- source-native editable parameters、material holdout 或 unseen-material test。

因此本报告把它分类为 `local-material / spatial-buffer appearance`；迁移判断统一留到 §13–15。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | hash/commit | 用途与边界 |
|---|---|---:|---|---|
| Formal main `P` | [Eurographics Digital Library PDF](https://diglib.eg.org/server/api/core/bitstreams/160e2378-c002-4c11-af34-57e5ad248d8d/content)，CGF 43(6), e15116，10 页 | 2026-08-29 | SHA-256 `563A86CDC171B9D4CA452E745EEBF684A75076329ED491F19B1B3A216D0E2F46` | 正式方法与结果的最高优先级来源；10 页均渲染复核 |
| arXiv current `P-v3` | [arXiv:2307.10135v3](https://arxiv.org/abs/2307.10135)，2024-04-24，10 页 | 2026-08-29 | SHA-256 `811AD32A225D6BF11DF606EA06D6270332319763A369E4350E498D2E49AA2A11` | 当前作者稿；文本/图表与正式方法对应，但文件并非逐字节相同 |
| arXiv source `P-src` | v3 source tar | 2026-08-29 | SHA-256 `CD0967495DCFE240B8A832BBB1C221485A57081BA619DD17ED7ACF92A0B539BA` | 核对 Eq.(2) 的原始 TeX，证明 reciprocal fourth-power 不是 PDF 提取错误 |
| arXiv old `P-v2` | arXiv v2，2023-12-01，9 页 | 2026-08-29 | SHA-256 `AA92A2FEC8CA6445BD2DCE58B5C3A6E1B7DB20FB0AAF7873402877822D4EC4AB` | 已改为当前标题与 25-channel Inception 主架构；用于版本审计，不替代正式版 |
| arXiv old `P-v1` | arXiv v1，2023-07-19，9 页，题为 *An Improved NeuMIP with Better Accuracy* | 2026-08-29 | SHA-256 `4359215A607DABBB10C7B31AAB65B50E5B814F5AF8944D36B2BE39068D2C84C8` | Inception 仍是 optional 256-channel 扩展、30k/80k iterations 与 16/64 SPP；仅登记实质演化 |
| Publisher supplemental `S` | Wiley 列出的 `cgf15116-sup-0001-SuppMat.zip`，页面标称 183.9 MB | 2026-08-29 | 未取得 | direct download 被 Cloudflare HTTP 403 阻断；in-app browser runtime 也不可用；不猜附件内容 |
| Official repository `A/C` | [bowenxueai/A-Hierarchical-Architecture-for-Neural-Materials](https://github.com/bowenxueai/A-Hierarchical-Architecture-for-Neural-Materials)，tag `v1.0.0` | 2026-08-29 | commit `a8978bc71034984121ebf7326c1a527e25238ca5` | 2025 年后发布的第一方 implementation snapshot；不是与论文同时冻结的 artifact |
| Repo metadata/tree `A/C-meta` | GitHub public API，commit/release/recursive tree | 2026-08-29 | tree response SHA-256 `AF5C262AF7A1A29C313F2DC3D51C3B348B7177DFC96192B44A757F43F5DD1A38` | 检查文件全集、binary/data/checkpoint/config/license 边界 |
| Released example data `C-data` | `datasets-deferred-buffer/rd_plane_s2.h5`, `rd_plane_s4.h5` | 2026-08-29 | SHA-256 `4AB202A5FBF5A61E61E7D010EB379642AD0EB2572A7C1123DB27DBE6BEEACEB5`, `84C073446E28C1339997FEF46549E55B6857F423734BB781BFED9E799AA13ED8` | 两个单帧 512² float16 deferred buffers；不是正式 500-pair material datasets |
| Related evidence `N-Taming` | [Taming evidence-reviewed report](./bitterli-2026-taming-optimization-variance.md) | 2026-08-29 | repo-local，`evidence-reviewed` | 仅用于 §13.3 核对 `M_pow` 公式、`n=3` 与作者对 Xue fourth-root precedent 的表述 |
| Project evidence `N` | `docs/contracts/scattering_backend.md`、`docs/realtime_material_compilation.md` 与 current NVIDIA formal config | 2026-08-29 | repo-local | 只用于 §13–15，不回填论文事实 |

### 2.1 版本演化是 load-bearing

`P-v1` 不是当前论文的轻微排版版本：其标题不同，主 decoder 仍可保持 NeuMIP MLP，Inception 是 optional 256-channel、six-layer 变体；它写普通模型 30,000 iterations、Inception 80,000 iterations，Figure 7 为 16 SPP、其余 64 SPP，并给出约 5 ms 的旧口径。[P-v1 §3–5]

`P-v2/P-v3/P` 把 Inception 收为核心 25-channel decoder，正式结果改成 1 SPP，runtime 改为 `0.035 s` 对 `0.028 s`。本报告只把旧稿用于说明配置如何变化，所有主结论以 `P` 为准。

### 2.2 supplemental 获取边界

正式论文 Figure 6/8 明确让读者查看 accompanying video，Figure 8 说明视频包含 light rotation 与逐渐 zoom-in 的 LoD 展示。[P Figs.6,8] Wiley 确实列出 ZIP，但当前公开下载被防自动化挑战拦截；没有登录、token 或凭据操作。因而 `S` 只能支持“附件存在”这一元数据事实，不能支持视频质量、额外配置或失败案例。

## 3. 原论文的问题、假设与贡献边界

NeuMIP 已能用 neural offset、texture pyramid 与小 decoder 重建多尺度材质，却会平滑掉强 self-shadow、锐利 highlight 与其他高频结构。作者把问题分为两类：[P §1, §3]

1. pointwise MLP decoder 对局部多尺度图像结构的表达不足；
2. 标准 L1/L2 对 HDR 明暗区域和边缘的训练权重不足。

正式贡献是：[P §1]

- 以两个 Inception modules 组成的 hierarchical decoder，同时利用多种 spatial kernel；
- 对 `u,ω_i,ω_o` 做固定 Fourier feature mapping；
- Sobel gradient loss 与 output remapping，abstract 还称 loss 会随 learning progress 自适应；
- synthetic、measured BTF、LoD 与 non-flat surface 展示。

贡献不包括重设计 neural texture pyramid 或 offset：正文明确说 pyramid 保持 NeuMIP 原样，差异集中在 decoder/input/loss。[P §3, Fig.2]

论文所说 “88% lower MSE” 是 Figure 1 teaser 的特定比较，不是跨所有场景、LoD、metric 与运行的统一比例。[P Fig.1] 正式表无随机种子、置信区间或统计显著性。

## 4. 输入、输出、坐标与 query domain

| 项 | 正式定义 | shape/domain | locator |
|---|---|---|---|
| Spatial input | texture position `u` | 2D，作者称所有 encoding 输入均 normalized | [P §3.1] |
| Angular input | incoming/light `ω_i` 与 outgoing/camera `ω_o` | 各用 2D query coordinates，总计 4 scalars；精确 hemisphere map/frame 未报告 | [P §3.1, §4.1] |
| Footprint | prefilter kernel size `σ` | 1 scalar；用于 texture pyramid/LoD，不进入论文所写 Fourier set | [P §3.1, §4.1] |
| Total query | `(u,ω_i,ω_o,σ)` | 7D | [P §3.1, §4.1] |
| Output | RGB reflectance | 3D linear RGB，exact scattering measure 未报告 | [P §4.1] |
| Spatial support | 一幅 query buffer | renderer 将整个 buffer 送 GPU batch evaluation | [P §4.2; C `neural_rendering.py:1190`] |
| LoD | camera-distance-derived footprint | per query；level 0 最近，数字越大越 coarse | [P §4.2, §5.2, Table 2] |

代码把 H5 的 `ground_camera_dir/ground_light` 各截为前两通道，和 `ground_camera_target_loc/query_radius` 一起构造二维图像张量。[C `dataset/dataset_reader.py:181–225`] 它没有在数据层保存 normal/tangent frame、wavelength、material parameter 或 event type。

`--cosine_mult` 默认关闭；启用时 target 乘 `sqrt(1-wi_x²-wi_y²)`，即在该坐标假设下的 `cos θ_i`。[C `neural_rendering.py:1905`; `dataset/dataset_reader.py:217–225`] 因此 C 同时存在 raw-target 与 cosine-weighted target 路线，但没有 metadata 把正式 checkpoint 映射到其中之一。

## 5. Representation、逐层网络与数据流

### 5.1 正式三阶段主干

```text
(u, ωo) → neural offset → u'
u' + σ → neural texture pyramid → latent feature
γ(u), γ(ωi), γ(ωo) + latent feature → 2×Inception decoder → RGB
```

第一阶段用 view-conditioned neural offset 补偿 micro-geometry；第二阶段以更新后的 `u'` 查询 neural texture pyramid并按 `σ` 过滤；第三阶段把 texture feature 与 encoded query 解码为 RGB。[P §3.1, Fig.2]

论文没有重新给出 offset network、latent channels、pyramid resolution、mip storage、texture interpolation、初始化或 precision，统一以 “identical to NeuMIP”/“pyramid remains the same” 带过。[P §3.1] 这些项目不能从 NeuMIP 常见配置反推成 Xue formal config。

### 5.2 Inception decoder 的精确 topology

正式 decoder 由四个串联层级构成：[P §3.1, Fig.3]

1. outer `1×1 convolution: input → 25 channels`；
2. Inception module A，输入/输出 25 channels；
3. Inception module B，输入/输出 25 channels；
4. outer `1×1 convolution: 25 → RGB`。

每个 Inception module 有四个并行 branch：

| Branch | 运算 | 输出 channels |
|---|---|---:|
| 1 | `1×1 conv` | 7 |
| 2 | `1×1 reduction → 3×3 conv` | 12 |
| 3 | `1×1 reduction → 5×5 conv` | 3 |
| 4 | `3×3 max-pool → 1×1 conv` | 3 |

四支 concat 为 `7+12+3+3=25`。P 进一步把 `7:12:3:3` 写成 `2:4:1:1`，但这不是严格相等（第一项 `7/3≠2`），只能理解为近似比例；code 使用的 exact channels 仍是 `7:12:3:3`。padding 保持二维尺寸。[P §3.1, Fig.3] 正文没有列 activation、bias、normalization 或 branch 内 reduction channels；official code 将 branch 2/3 的 reduction 与 output channels 都分别设为 12/3，每次 convolution 后 ReLU，max-pool stride 1。[C `network1.py:44–66`; `neural_rendering.py:379–419`]

代码 `FullyConnected3` 与正式图一致地使用两个 `Inception(25,7,(12,12),(3,3),3)`，但两个 outer conv 后没有额外 activation。[C `neural_rendering.py:395–400`] 代码类名虽叫 `FullyConnected3`，实际为 `Conv2d` 并在 `H×W` query buffer 上混合空间邻域。

### 5.3 Fourier encoding

正式公式对归一化 scalar `p∈{u,ω_i,ω_o}` 定义：

```text
γ(p) = (sin(2^0 πp), cos(2^0 πp), …,
        sin(2^(L-1) πp), cos(2^(L-1) πp)).
```

`u` 用 `L=10`，方向用 `L=4`。[P §3.1]

official code 的 `--pe 1` 也使用 position 10 frequencies、camera/light 各 4 frequencies，但实现包含 raw input `x`，并计算 `sin(2^k x), cos(2^k x)`，不乘论文公式的 `π`。[C `positionembeding.py:3–42`; `neural_rendering.py:1009–1039`] 这构成明确的 basis gap，不能称公式逐项 code-faithful。

在 `--pe 1` 下，code 的 channels 是：`u: 2×(2×10+1)=42`，每个 3D direction `3×(2×4+1)=27`；两个方向与 position 共 96 channels，再加 neural texture等特征。[C same] 论文只说方向 query 是 2D，但 code 先用正半球第三轴恢复到 3D 再 encoding；这也是 paper/code representation 差异。

### 5.4 spatial-buffer dependency

正式 renderer 不是对互不相关的 query 独立调用 decoder，而是把完整 material query buffer 交给 GPU。[P §4.2] official code 的 `3×3/5×5/max-pool` 确实跨相邻 buffer pixels 读取。[C `network1.py:49–64`] 因而输出除自身 `(u,ω_i,ω_o,σ)` 外还依赖 buffer 中邻居 query 的排列与边界 padding。论文没有给 arbitrary query permutation、sparse shading、tile seam、divergent LoD 或 single-query protocol。

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 正式配置 | locator |
|---|---|---|
| Synthetic sources | Basket cloth、Twill cloth、Metal ring、Bump | [P §4.1] |
| Synthetic renderer | KeyShot path tracer | [P §4.1] |
| Geometry | Metal ring/Bump 用 height-map displaced geometry；cloth 用作者既有 ply-based cloth models | [P §4.1] |
| NeuMIP assets | Victorian cloth、Turtle shell；作者重新训练 NeuMIP 并“tweaked for best results” | [P §4.1] |
| Real measured input | UBO 2014 Leather BTF | [P §5.2, Fig.6] |
| Generated data amount | “500 input-output value pairs” | [P §4.1] |
| Training batch | mini-batch size 30,000 | [P §4.1] |
| Split | 每材质使用全部 available training data；无 train/validation/test split | [P §4.1] |
| Model granularity | per material | [P §4.1] |
| Query distribution | 未报告 `u/ω_i/ω_o/σ` 的采样分布、每 pair 的图像尺寸/单位、filter kernel 形状 | [P §4.1] |
| Persistent training data | 论文依赖离线生成 dataset；不是 GPU-online reference query | [P §4.1] |

“500 pairs” 少于 batch size 30,000，说明 pair 与 minibatch sample 的单位至少没有被正文充分定义；本报告不把 500 猜成帧数或材质数。released loader 把 H5 第一维当 dataset item，而每个 item 是完整 `H×W` buffer；默认 CLI batch `4` 指四幅 buffer，不是 30,000 scalar queries。[C `dataset/dataset_reader.py:79–90,181–225`; `neural_rendering.py:1303–1312,1883`]

release 中两个 H5 都只有一幅 `512×512` float16 buffer，字段为 camera/light 2D、UV、query radius、RGB、valid 与 light multiplier；它们显然不足以构成正式 500-pair 数据，也没有材质 source locator。[C-data]

official tree另含 11 个 reference EXR、4 个 OBJ、23 个约 51–52 MB `.pth` checkpoints，但文件名主要是 `metal/bump` ablation 简写；没有 manifest 把它们对应到 Figure/Table、paper config、source hash 或 seed。[C-meta]

## 7. Loss、optimizer 与训练 lifecycle

### 7.1 正式 gradient loss

论文用 Sobel filters 定义：

```text
L_G(I, Î) = (Gx(I)-Gx(Î))² + (Gy(I)-Gy(Î))²,
Gx(I)=kx*I, Gy(I)=ky*I.
```

`kx=[[1,0,-1],[2,0,-2],[1,0,-1]]`，`ky=[[1,2,1],[0,0,0],[-1,-2,-1]]`。[P Eq.(1)] 论文没有说明 RGB 是逐通道还是先转 luminance/grayscale，也没有说明 pixel/channel reduction、boundary、padding 或 gradient term weight。

release `fgradloss` 先把 RGB 转 grayscale，再使用 `CannyFilter` 的 Sobel outputs，最后取 `|ΔGx|` 与 `|ΔGy|` 的 mean 和；传入的 pixel `weight` 参数没有实际参与该函数。[C `utils/tensor.py:694–708`] 因而 C 是 **gradient L1**，不是 P Eq.(1) 的 squared gradient。CannyFilter 还包含其自身 kernel/module construction，论文未逐项冻结。

### 7.2 fourth-root prose 与 printed Eq.(2) 的冲突

正文连续文字明确说：对 prediction 与 reference 应用 **fourth root** 后，再计算 L1 与 gradient loss。[P §3.2]

但正式 Eq.(2) 与 v3 TeX 实际写的是：

```text
L = (1/n) Σ_i [L1(1/I_i^4) + L_G(1/I_i^4)]
```

下一句又称 `I^-4` 与 `Î^-4` 是 per-pixel exponents。[P Eq.(2); P-src `sec_method.tex:106–113`] 这不是 fourth root，而是 reciprocal fourth power；在零值附近奇异，目标与正文意图完全不同。并且 Eq.(2)/TeX 的两个 loss term 字面上都只写 `I`，prediction/reference 双参数只由前文定义和下一句暗示，literal objective 本身也不完整。该 gap 已从 PDF render 与原始 TeX 双重确认，不是 extraction artifact。

release code 的 `kaifang(x,n)=pow(clamp(x,1e-7),n)`，`4maploss` 明确对 prediction/GT 使用 exponent `0.25`，然后相加 gradient L1 与 pixel L1。[C `neural_rendering.py:1405–1410,1605–1607`] 因而：

- P prose 与 C 支持 `x^(1/4)`；
- P Eq.(2)/TeX 支持 `x^(-4)`；
- 没有作者勘误，不能由 C 静默改写正式公式。

### 7.3 “adaptive to learning progress” 的未闭合 lifecycle

abstract 声称 gradient loss 会随 learning progress 自适应，但正式 §3.2 只给静态 Eq.(2)，§4.1 也没有 stage boundaries。[P Abstract, §3.2–4.1]

代码提供多个互不等价的 schedule：[C `neural_rendering.py:1579–1616`]

| code loss | phase 1 | phase 2 | phase 3 |
|---|---|---|---|
| `stagegl1` | 前 1/4：linear L1 | 后 3/4：linear gradient+L1 | — |
| `stage1` | 前 1/4：fourth-root L1 | 中 1/2：linear gradient+L1 | 后 1/4：square-root gradient+L1 |
| `stage2` | 前 1/2：fourth-root L1 | 后 1/2：linear gradient+L1 | — |
| `stage3` | 前 1/4：fourth-root L1 | 后 3/4：fourth-root gradient+L1 | — |
| `4maploss` | 全程 fourth-root gradient+L1 | — | — |

没有 config、command、checkpoint metadata 或 paper statement 能确定哪一个生成正式结果。代码存在这些分支只证明作者探索过这些 recipe，不能把某一个回填为 formal adaptive schedule。

### 7.4 optimizer 与生命周期

| 项 | Paper | released code | correspondence |
|---|---|---|---|
| Optimizer | Adam | Adam | 名称一致；β/ε/weight decay 未在 P 冻结 |
| LR | 未报告 | base experiment `5e-4`；release default experiment `1e-3`；启用 PE/RPE 后再除 2 | C 不能证明 P exact |
| Steps/epochs | 未报告 | `max_iter=30,000` default | P-v1 曾写 30k/80k，但不可回填正式版 |
| Batch | 30,000，单位未定义 | 4 full buffers | 实质单位/数值 gap |
| Seed | 未报告 | 未设置全局 deterministic seed | 不可恢复 |
| Selection | 未报告；使用全部 training data | 每 100 iter 保存并周期算 loss；final 调 `calculateloss1` | 无 formal best/last identity |
| Time | 约 90 min / material | 未提供复现 timing | P-only observation |
| Hardware | training hardware 未报告；只有512² performance timing明确使用 NVIDIA V100 | CLI 可选 GPU count，默认 1 | 不能把 inference GPU 静默回填为 training GPU；C 没有 exact GPU/precision config |

代码还有未在 P 披露的 mip/filter lifecycle：每 step 从 `[5,5,5,7,num_mips]` 随机选可用最高 level并加 pixel weight；前 3000 iterations 每 100 steps fuse blur；`σ` 指数衰减，超过 `sigma_1_time` 冻结 offset。[C `neural_rendering.py:846–859,1043–1067,1386–1399`] release default experiment 的 `sigma_1_time=100000`，却默认只跑 30,000 steps，因而 freeze boundary 永远不触发。[C `experiments/simple.py:126–141`; `neural_rendering.py:1881`]

### 7.5 README quickstart 不复现论文

README 只给 `python neural_rendering.py --dataset ...`。其 argparse defaults 是 `--net 2 --pe 0 --loss comb1 --max_iter 30000 --batch 4`。[C `README.md`; `neural_rendering.py:1881–1911`] `net=2` 是六层 125-wide `1×1 Conv` MLP，不是 Inception `net=3`；`pe=0` 关闭 encoding；`comb1` 是 `log1p` MSE 加 `0.01×linear L1`，不是 gradient/fourth-root loss。[C `neural_rendering.py:92–124,741–768,1420–1425`] 因此 README 是可启动示例，不是 paper reproduction command。

此外，两个 released H5 各只有一个 dataset item，而 default DataLoader 使用 `batch=4, drop_last=True`；即便同时传入这两个文件也凑不出一个 training batch。[C-data; C `dataset/dataset_reader.py:79–90`; `neural_rendering.py:1303–1312`] 这进一步说明 released sample data + README defaults 不是自足的 formal reproduction package。

## 8. Inference、部署与成本

| 项 | 正式配置/结果 | locator |
|---|---|---|
| Renderer | Mitsuba 2，direct illumination only | [P §4.2, §6] |
| Invocation | whole material query buffer batched to GPU | [P §4.2] |
| LoD | camera distance per query | [P §4.2] |
| Comparison spp | 1 SPP | [P §4.2] |
| Hardware | NVIDIA V100 GPU | [P §4.3] |
| Resolution | `512×512` query texture | [P §4.3] |
| Ours | `0.035 s` | [P §4.3] |
| NeuMIP | `0.028 s` | [P §4.3] |
| Author summary | about 25% longer evaluation | [P §4.3] |

论文没有给 warm-up、batch count、timer boundary、precision、replicates、median/p90、memory bandwidth、parameter bytes、FLOP/MAC 或 single-query latency。`0.035/0.028=1.25` 正好对应作者写的 25% overhead；正式测量对象仅是整幅 512² buffer，single-query cost 未测。

P §4.3 说 convolution 增加 complexity/parameter count；结论又说与 NeuMIP “identical sizes”。没有参数表解释 “size” 指深度、神经元、checkpoint bytes 还是 comparison setting，因此保留为内部措辞冲突。[P §4.3, §6]

release network 先输出 YUV-like 3 channels，再 `yuv_to_rgb` 与 `exp(result)-1`；inference 才 clamp 到 nonnegative。[C `neural_rendering.py:1190–1216`] P 只写 linear RGB prediction 与 training remap，未披露这一 log/output transform。official code也没有 Slang/CUDA shader、TensorRT、raster integration或正式 Mitsuba plugin reproduction command。

## 9. 实验 protocol、baseline、指标与结果

### 9.1 baseline 与 protocol

主 baseline 是作者重新训练的 NeuMIP；Victorian/Turtle shell 的 NeuMIP parameters 被 “tweaked for best results”。[P §4.1] 没有公开 baseline config、seed、training time matching 或 checkpoint selection。作者另把 NeuMIP MLP 放大 10×/100×/300×，用于反驳“只加容量即可恢复细节”。[P Fig.5, §5.2]

指标为 MSE、LPIPS、PSNR，但 data range、颜色空间、tone map、crop、MSE scaling 与 LPIPS implementation 未报告。所有结果据称是在 whole dataset 上 average，未给 per-frame values、variance 或 raw outputs。[P §5.2]

### 9.2 Table 1 正式数值

| Scene | MSE ours / NeuMIP | LPIPS ours / NeuMIP | PSNR ours / NeuMIP |
|---|---:|---:|---:|
| Ring | `0.342 / 3.474` | `0.139 / 0.215` | `38.472 / 29.887` |
| Leather | `0.029 / 0.540` | `0.012 / 0.134` | `32.792 / 28.601` |
| Metal grid | `0.076 / 0.401` | `0.156 / 0.314` | `30.355 / 28.066` |
| Turtle shell | `0.056 / 0.703` | `0.070 / 0.163` | `32.181 / 30.785` |
| Victorian fabric | `1.335 / 8.071` | `0.104 / 0.141` | `35.252 / 30.849` |
| Twill cloth | `0.005 / 0.018` | `0.120 / 0.262` | `29.507 / 28.157` |

[P Table 1] 表中六场景三个 metric 均由 ours 数值优于 NeuMIP，但未报告 uncertainty，也没有共享 source-renderer re-evaluation protocol足以独立复算。

### 9.3 LoD Table 2

| Scene/method | LoD 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Basket ours | .792 | .637 | .457 | .264 | .119 | .043 | .012 | .003 |
| Basket NeuMIP | 6.367 | 6.018 | 5.437 | 4.595 | 3.568 | 2.585 | 1.960 | 1.581 |
| Metal ring ours | .062 | .046 | .035 | .025 | .019 | .016 | .014 | .013 |
| Metal ring NeuMIP | 3.093 | 2.989 | 2.767 | 2.368 | 1.867 | 1.305 | .802 | .448 |

[P Table 2] 作者解释 coarse LoD 自然下采样、high-frequency逐渐消失，所以误差变小。它不等于模型在更远处“泛化更好”。

### 9.4 qualitative/large-MLP 结果

- Figure 4 是 component removal qualitative ablation，使用 Metal ring 与 Basket cloth。[P §5.1]
- Figure 5 的 enlarged NeuMIP MSE 是 original `1.498`、10× `.839`、100× `.379`、300× `.274`；更大 MLP 仍漏 self-shadow/highlight，且增加 training/query time，但没有时间/参数量数表。[P Fig.5, §5.2]
- Figure 6 展示 synthetic、NeuMIP 与 UBO measured BTF 输入；作者称本方法在 leather 的 fine-grain/highlight 更接近 reference。[P Fig.6, §5.2]
- Figure 8 展示 non-flat surface，但正文没有 curvature-conditioned input；这是 renderer integration visual，不是 curved-surface method 的独立 quantitative validation。[P Fig.8]

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | locator |
|---|---|---|---|---|
| `author-negative` | 原 NeuMIP/等深等 neuron count fully connected decoder | 明显低于 Inception；加深/加宽 FC 也未显著改善复杂细节 | pointwise FC 不会利用多尺度 spatial kernels | [P §3.1] |
| `ablation-inferior` | w/o Inception | overall expressiveness/color 较差 | hierarchical architecture捕获多尺度 feature | [P Fig.4, §5.1] |
| `ablation-inferior` | w/o input encoding | edge/high-frequency恢复较差 | spectral bias | [P Fig.4, §5.1] |
| `ablation-inferior` | w/o gradient loss | edge/high-frequency恢复较差 | gradient supervision | [P Fig.4, §5.1] |
| `ablation-inferior` | w/o remapping | 红框区域的 back yarn 缺失 | MSE均匀分配全图 error，低 luminance区域相对更重要 | [P Fig.4, §5.1] |
| `author-negative` | NeuMIP 10×/100×/300× | 误差下降但仍漏 sharp highlight/self-shadow；cost 增加 | 容量大小不能替代 hierarchical structure | [P Fig.5, §5.2] |
| `known-limitation` | convolution footprint | 比 NeuMIP 慢，参数/邻域更大 | 可继续优化 | [P §4.3, §6] |
| `known-limitation` | direct illumination, single bounce | 没有 indirect transport/importance sampling | importance sampling 为 future work | [P §4.2, §6] |

Figure 4 是从 full model 每次关闭一个 component 的 qualitative leave-one-component-out 展示，不是逐列累积 recipe，也不提供 numeric factorial ablation；它没有把 network、encoding、gradient 与 remap 的 interaction 分离。尤其 gradient 与 remap 在 final loss 里耦合，不能由 Figure 4 单独推断两者各自的完整 causal gain。

代码中的大量 `loss` branches、`net=1…7`、PE modes 与 23 个 checkpoint 名证明开发中存在更多尝试，但没有 author text把它们标为成功/失败或对应正式图。它们保持 `unclassified implementation variants`，不能被本报告重新解释成 author-negative。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper `P` | Supplemental `S` | Code `C` | 结论/冲突 |
|---|---|---|---|---|
| Core decoder | two Inception, 25 ch, exact `7/12/3/3`；另写近似 `2/4/1/1` | 未取得 | `net=3` 完整对应 exact topology | topology 高置信；P 的 ratio 等式并不严格，activation 等由 C 补充但不是 P formal disclosure |
| Default command | 未给 | 未取得 | README 默认 `net=2,pe=0,loss=comb1,batch=4` | quickstart 不复现正式方法；两个released H5合计仅2 items且`drop_last=True`，无法形成default batch |
| Fourier features | `π`, 不含 raw `p`；u L10，dir L4 | 未取得 | 不乘 `π`，包含 raw input；先恢复 3D direction | 实质 basis/input gap |
| Gradient | squared Sobel differences | 未取得 | grayscale absolute Sobel differences | L2-vs-L1 与 channel reduction gap |
| Remap | prose fourth root；Eq.(2) reciprocal fourth power | 未取得 | `pow(clamp(x),.25)` | P 内部冲突；C 与 prose一致但不是勘误 |
| Adaptive schedule | abstract 声称 adaptive，正文无 schedule | 未取得 | 多个互斥 `stage*` | formal run identity 不可确定 |
| Batch/data | 500 pairs，batch 30,000，单位未定义 | 未取得 | batch 4 full buffers；release data each 1×512² | 不可对应 |
| Texture/filter | “same as NeuMIP” | 未取得 | 7-ch textures、mip随机化、blur/σ schedule与多个 experiment switches | released variant 不能自动填 formal config |
| Output/measure | linear RGB reflectance | 未取得 | YUV→RGB，`exp-1`；optional cosine multiplier | exact `f` correspondence unresolved |
| Runtime | Mitsuba 2 direct, V100 512² timing | video未取得 | Python/PyTorch renderer utilities，无 formal command | 正式 timing不可复跑 |
| Assets/results | Figures/Tables | video未取得 | 23 checkpoints、11 EXR、2 example H5、无 manifest | 无 Figure/Table checkpoint/source mapping |

### 11.1 official repository 的可复现性边界

仓库在论文发表后约一年才公开；当前 `v1.0.0` release 无 attached assets，repository metadata `license=null`，tree 中也无 LICENSE。没有 `requirements.txt`、`environment.yml`、locked package versions、formal shell script、random seed、material manifest 或 results manifest。[A/C-meta]

递归 tree 有 179 items/168 blobs，包含约 1.16 GiB checkpoint payload、61 `.py`、43 `.pyc`、23 `.pth`、11 `.exr` 与两个 `.h5`。本次固定 commit 后审计了所有可读 source/config-like files并取得两个 H5；未把 23 个大 checkpoint内容全部下载。checkpoint 文件名可证明 ablation artifacts 存在，不能证明哪一个是 Table 1/2 final model。

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. **没有 importance sampling**：若要放入 physics-based Monte Carlo，作者说仍需 efficient importance sampling。[P §6]
2. **只捕获 direct single light bounce**：global illumination 留待 future work。[P §6]
3. **convolution footprint/cost**：比 original NeuMIP 稍慢；作者希望优化。[P §4.3, §6]
4. **曲面方法边界**：作者只展示 non-flat renderer result，并提出未来结合 curved-surface neural model；没有证明当前 representation 在曲率变化下严格成立。[P §5.2, §6]
5. **BSSRDF**：subsurface scattering 不在当前方法内。[P §6]

### 12.2 未报告/材料不可得

- formal dataset/source assets、500-pair 中 pair 的单位与 query distribution；
- UV/angle normalization、coordinate frame、angular disk map、filter kernel 定义；
- output 是 `f`、`f cos` 还是其他 reflectance/radiance convention；
- neural offset、texture pyramid 的 formal channels/resolution/bytes/precision；
- Eq.(2) fourth-root/reciprocal-power及其缺失prediction/reference双参数的勘误；
- adaptive loss 的 exact phase schedule；
- LR、Adam hyperparameters、steps、training GPU、seed、validation、checkpoint selection；
- Table/Figure 的 raw predictions、metric code、data range、tone map 与统计 uncertainty；
- 512² timing 的 protocol、precision、memory、single-query latency；
- supplemental video/ZIP 内容；
- official code 的依赖版本、license、formal command与checkpoint manifest。

## 13. 本项目分析 `[I]`

### 13.1 真正的新容量来自 query-buffer convolution

两层 Inception 的最大 spatial path 连续经过两个 `5×5` kernels，因此一个输出在 code topology 下最多可受约 `9×9` buffer footprint影响；短路径仍保留 `1×1/3×3/pool` feature。这是结构能恢复局部 edge/shadow 的直接机制，而不只是“参数更多”。但它利用的是**query buffer adjacency**，不保证等价于 material-space 邻域：screen 上相邻 pixels可能跨 UV seam、几何边界、不同 footprint 或不同材质，随机 query batches更没有稳定二维邻接。

因此其成功条件包含一个 NeuMIP point-query 模型没有的隐式假设：训练与推理 query 必须按有意义的二维 image layout 排列。该假设解释了为何 paper 强调整幅 batch GPU evaluation，也解释了它与本项目随机访问 `evaluate(wo,wi)` 的根本接口差异。

### 13.2 gradient/remap 是训练域重加权，不是物理分解

Sobel loss使 optimizer显式重视 screen-space边缘；fourth root把低 luminance 差异放大。二者可以改善阴影/highlight图像，却没有增加 reciprocity、energy conservation、BRDF positivity、filter consistency 或 transport factorization。它们的收益必须在同一 query/layout 与 image loss protocol下理解，不能自动迁移为 local scattering accuracy。

代码的 `fgradloss` 完全忽略传入 pixel weights，还先转 grayscale；这意味着 gradient branch与 color L1 branch的 weighting不一致。若要复现，应把 P squared-RGB/Sobel 与 C grayscale-L1/Sobel 分成两个明确 identity，不能混成一个“Xue loss”。

### 13.3 与 Taming power mapping 的关系

Taming 2026 将 Xue 称为 related fourth-root precedent，并定义 `M_pow(x)=n(x^(1/n)-1)`，正式取 `n=3`。[N-Taming §7.3, §12] Xue prose/C 的 `x^(1/4)` 与 Taming family在 `n=4` 时具有相同 root exponent，但并非字面同式：Taming还乘 `4`、减常数 `1`；对纯 L1，常数在差分中消去、乘数只改变整体 scale，但一旦与其他 loss、optimizer/loss scaling组合，gradient magnitude仍不同。更重要的是，Xue printed Eq.(2) 的 `x^-4` 完全不属于这一 root family。

可迁移的表述应是“以 `n=4` root exponent为 precedent”，不是“Xue已经验证 Taming 的 normalized power mapping”。

### 13.4 结果能支持什么

Table 1/2 与 Figure 4/5 支持：在作者的 per-material full-buffer training、这些材质、1-SPP direct renderer与未公开 metric pipeline下，hierarchical/full method consistently优于其 NeuMIP reproduction。它不支持：

- unseen material/source family泛化；
- random-access local BRDF query优越性；
- GI/path tracing variance降低；
- 相同参数/MAC/latency下必然支配 pointwise MLP；
- root、gradient、encoding、Inception各自独立的精确增益。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

| 主题 | current project contract/config | Xue mechanism | 分类与影响 |
|---|---|---|---|
| Query identity | `prepare()` 可复用 state；`evaluate(wo,wi)` 随机访问、输出 bare linear `f` | complete 2D buffer convolution，RGB measure未定义 | `interface-adaptation`；原 decoder不能直接成为 evaluator candidate |
| Input encoding | current NVIDIA有自己的方向/latent feature identity | raw+Fourier `u/ω` | 可做 matched conditioning axis，但必须固定 output/loss/params |
| Loss | current formal evaluator `log1p-l1@1` | P/C fourth-root + gradient variants | `training-only candidate`；先做 root-only，再单独 gradient，不能一次换四项 |
| Filtering | current online query明确 latent mip/filter recipe | P 继承 NeuMIP，C有独立 blur/σ/mip sampling | `author-underspecified`；不能以 release default替换 current frozen filter |
| Data | GPU-resident online reference，不持久化 corpus | 500-pair offline buffers | `intentional-deviation`；迁移 loss不要求恢复其 corpus |
| Sampler | current `sample/pdf` 同一 proposal | 无 sampler/pdf | `not-applicable`；不影响 matched sampler contract |
| Deployment | statically bounded single-query shader | V100 full-buffer PyTorch conv | 只能将 root/encoding或局部 stencil变体单独评估；原 runtime class不匹配 |

最小风险迁移顺序不是复刻 entire released repo，而是保留 current NVIDIA `functional-f@2` source/query/model identity，分别引入：

1. fourth-root mapped L1（先不加 gradient）；
2. query-local Fourier features的 iso-parameter control；
3. 若需要 spatial context，再设计 `prepare()` 可缓存、材质空间定义明确的固定 footprint，而不是依赖 screen buffer邻接。

这只是研究候选拆分，不是已授权执行或 hard gate。

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| `H-XUE-1`：fourth-root L1 改善当前 HDR evaluator 的低值细节 | P prose/Fig.4；C `4maploss`；Taming称其为 root precedent | current `f` dynamic range也存在 low-value gradient starvation | current `log1p-l1@1` vs `pow-clamp-l1@n4`；同 seeds/query/model/steps | output transform、optimizer、directions、filter、budget | linear-f error按能量分层、HDR FLIP、energy、median/p90、CI | training-only | 低值 strata无改善，或 peak/energy/Pareto显著恶化 |
| `H-XUE-2`：Sobel auxiliary能恢复 spatial high-frequency而不破坏 pointwise correctness | P Eq.(1)/Fig.4 | online queries可组成 material-space邻域而非 screen-neighbor伪邻域 | no-gradient vs P squared Sobel vs C grayscale-L1 Sobel 三项 | root mapping、query tuples、neighborhood construction、model、steps | local f、spatial derivative、seam/edge、energy、training cost | training-only，需邻域 query | 只改善image metric而local `f`/seam变差，或 permutation/layout改变输出 |
| `H-XUE-3`：Fourier input的收益超过第一层容量增加 | P §3.1/Fig.4；C有明确basis gap | current compact decoder仍 spectral-bias limited | raw、P `π/no-raw`、C `no-π/with-raw`；另加 iso-parameter projection control | loss、seed/query、total params/MAC、output/filter | angular/spatial frequency strata、grazing、latency、params | runtime evaluator改变但静态有界 | iso-parameter下收益消失，或高频改善伴随alias/LoD不一致 |
| `H-XUE-4`：固定 material-space stencil可保留Inception优势 | P Fig.3–5；buffer conv恢复细节 | 有稳定UV/footprint邻域且 `prepare` 可复用读取 | pointwise MLP vs fixed 3×3/5×5 material-space gather，同参数/MAC control | source/query/filter、loss/encoding、seeds、fetch count cap | local/spatial质量、UV seam、tile边界、single/multi-light latency、state bytes | product-candidate | seam/permutation错误、固定读取超预算，或 quality–cost不支配MLP |
| `H-XUE-5`：paper Eq.(2) 是排版缺陷而非可用 objective | P prose与C root一致、P TeX却为 reciprocal power | root branch代表作者运行意图 | 仅 diagnostic：在正值clamp下比较 root与 reciprocal-power的 finite/gradient统计；不进入正式排名 | input batches、scale、clamp、model init | nonfinite rate、gradient quantiles、early loss | diagnostic only | reciprocal-power稳定且作者提供勘误/正式证据；在此之前不能提升为formal config |

这些 hypothesis 都需要独立 planning/freeze 才能运行；论文数值不是本项目 hard gate，旧 NV artifact 也不能被 Xue checkpoint替换后继续沿用原 correspondence identity。

## 16. 证据索引

- `P pp.1–2, Figs.1–2`：问题、贡献、88% teaser、NeuMIP inherited pipeline。
- `P pp.3–4, §3.1, Figs.3–5`：Inception topology、25 channels、Fourier设计与 component/large-MLP消融。
- `P pp.5–6, Figs.6–8, Table 1`：synthetic/measured BTF、LoD、non-flat展示与主指标。
- `P p.7, Eq.(1)–(2), Table 2`：Sobel loss、root prose、reciprocal-power printed equation、LoD数表与 dataset/training。
- `P p.8, §4.2–5.2`：Mitsuba direct-only、whole-buffer evaluation、1 SPP、V100 timing与结果解释。
- `P p.9, §6`：importance sampling/GI/convolution/curved model/BSSRDF限制。
- `P-src sec_method.tex:106–113`：Eq.(2) 原始 `1/I^4`，排除 PDF OCR误差。
- `P-v1 pp.1–8`：旧标题、optional 256-channel Inception、30k/80k、16/64 SPP与5ms旧口径。
- `P-v2`：current-title 25-channel架构的过渡版本；不承担正式结论。
- `C network1.py:44–66; neural_rendering.py:379–419`：Inception exact operations。
- `C positionembeding.py:3–42; neural_rendering.py:1009–1039`：with-raw/no-π encoding与L counts。
- `C neural_rendering.py:1379–1425,1540–1616`：target transforms、loss branch与stage schedule。
- `C utils/tensor.py:694–708`：grayscale gradient L1 implementation。
- `C dataset/dataset_reader.py:53–90,181–225`：H5 item/buffer unit、fields、optional cosine。
- `C experiments/simple.py:7–40,126–141; neural_rendering.py:650–830,1043–1067,1386–1399,1861–1911`：release lifecycle/defaults与formal-gap。
- `A/C-meta`：2025 release identity、no license/environment/config manifest、tree/checkpoint/data inventory。
- `S`：Wiley advertised ZIP；内容不可得，不承担配置证据。
- `N scattering/compiler contracts + current NVIDIA formal config`：§14–15 current runtime/data/loss边界。

## Evidence review

```text
author_worker: /root/belcour2018_review
reviewer: /root/dualband2025_review
reviewed_at: 2026-08-29
sources_rechecked:
  - formal EG Digital Library PDF，10页，SHA-256 563A86CDC171B9D4CA452E745EEBF684A75076329ED491F19B1B3A216D0E2F46；逐页渲染核对Figs.1–8、Tables1–2、Eqs.1–2、图注与限制
  - arXiv v3 PDF/source、v2 PDF、v1 PDF；固定四份hash并审计实质版本演化
  - Wiley supplemental metadata/direct locator；HTTP 403且in-app browser unavailable
  - official repo v1.0.0/current commit a8978bc71034984121ebf7326c1a527e25238ca5；递归tree、全部source-like files、README、loss/network/dataset/filter/CLI代码
  - released rd_plane_s2/s4 H5 schemas与SHA-256；checkpoint/reference/data inventory
findings_closed:
  - 独立视觉复核25-channel two-Inception exact topology与7:12:3:3四支配置，并指出P写成2:4:1:1只可视为近似比例
  - 分离formal Fourier公式和code with-raw/no-pi/3D-direction实现
  - 双重确认P Eq.2的reciprocal fourth-power与正文/code fourth-root冲突，并保留公式未显式写出prediction/reference双参数的问题
  - 确认P Eq.1是squared Sobel difference，而C先转grayscale后计算gradient L1且忽略传入pixel weight
  - 恢复C全部stage/adaptive-loss branches，并保留formal run无法映射
  - 确认README defaults不启用paper Inception/PE/loss，且released H5 items不足以组成default drop-last batch
  - 复核Table1/2、Figure5大MLP数值和512x512 V100 timing
  - 将V100限定为formal inference timing硬件；training GPU保持未报告
  - 将Figure4 remapping负结果修正为红框back yarn缺失，并明确其为one-at-a-time qualitative removal而非累积recipe
  - 分离whole-buffer convolution与本项目random-access evaluator语义
remaining_evidence_gaps:
  - publisher 183.9 MB supplemental ZIP/video被Cloudflare 403阻断
  - paper Eq.2无作者勘误；adaptive loss exact schedule不明
  - formal source/query/filter、pair/batch单位、LR/steps/training GPU/seeds/checkpoint selection未报告
  - output/cosine measure、metric normalization、raw results与single-query cost未报告
  - official repo无license/dependency lock/formal command/checkpoint manifest；23个大checkpoint未逐个内容审计
review_status: evidence-reviewed
```

### 完成检查

- [x] main paper 10页已完整阅读并逐页视觉核对；
- [x] arXiv v1/v2/v3与source的实质差异已检查；
- [x] supplemental/视频/勘误可得性已检查，阻断边界明确保留；
- [x] official code/config/data已固定commit并审计；
- [x] architecture、input、loss、training、runtime与主要结果均有locator；
- [x] fourth-root、gradient norm、README defaults与paper/code冲突已保留；
- [x] author-negative、ablation-inferior、known-limitation和unclassified variants已分开；
- [x] `P/S/C/A` 事实先于 `N/I` 分析；
- [x] 迁移假设包含matched control、frozen axes、metrics、runtime class与falsification；
- [x] 独立 evidence review 已完成。
