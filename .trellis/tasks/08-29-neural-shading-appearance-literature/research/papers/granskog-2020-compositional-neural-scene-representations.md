---
paper_id: "granskog-2020-compositional-neural-scene-representations"
title: "Compositional Neural Scene Representations for Shading Inference"
authors: "Jonathan Granskog, Fabrice Rousselle, Marios Papas, Jan Novák"
year: "2020"
venue: "ACM Transactions on Graphics 39(4), Proceedings of SIGGRAPH 2020, Article 135"
doi: "10.1145/3386569.3392475"
report_status: "evidence-reviewed"
main_source: "https://jannovak.info/publications/CNSR/CNSR.pdf"
supplemental_status: "available"
official_code_status: "audited"
official_code_commit: "9c9033a1ca05095c7e2ccfeb4da3046b687bef3d"
author_worker: "/root"
reviewer: "/root/belcour2018_review"
last_verified: "2026-08-29"
---

# Compositional Neural Scene Representations for Shading Inference

## 1. 研究对象与报告边界

Granskog、Rousselle、Papas 与 Novák研究的是 scene-level image synthesis：先从同一个 3D scene 的三张 path-traced observations、对应 G-buffers 和 camera matrices提取一个 view-independent global latent，再把该 latent 与 novel-view camera/G-buffer 输入 image generator，生成该视角的 HDR shaded image。论文的主要创新不是 encoder/generator backbone，而是把 latent 显式分为 lighting、geometry、material 三个连续 partitions，学习各 partition 的大小，并可增加 null partition 作为未使用容量的 reservoir。[P §§1,3–4, Figs.1–2,5–6]

本文是 LightFormer 与 Active Exploration for Neural GI 的重要 scene baseline，但它不是 local neural material，也不提供 `evaluate(wo,wi)`、material-direction `sample()/pdf()` 或跨任意 native material family 的编译器。它的 runtime query 是“已编码场景 + novel-view raster G-buffer → image/residual transport”，主要容量是 per-scene global representation、query G-buffer 和大 image generator。[P §§3,6]

本报告重建：

1. observation encoder、three-view aggregation 与三种 generator 的逐层配置；
2. static/adaptive/null partition 的 forward、gradient 与 compression机制；
3. PrimitiveRoom/ArchViz procedural datasets、训练 lifecycle 与 HDR loss；
4. generator/G-buffer/partition消融、attribution、auxiliary shadow generator、indirect-GI和relighting；
5. paper↔supplemental↔official code/config 的正式冲突；
6. 它对本项目 local material compiler、scene transport第二波和当前 NVIDIA复现的可迁移/不可迁移边界。

不把“lighting partition可交换”写成对任意 light parameters 的解析控制：partition来自其他已编码 scene observations，且正式模型的 compositing仍受训练分布和scene geometry约束。也不把 400 ms image prediction写成完整 frame cost；three observation acquisition/encoding、G-buffer/direct-light generation与其他前后处理是否计入该数值并未完整披露。[P §§3.4,6]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | [作者正式项目页 PDF](https://jannovak.info/publications/CNSR/CNSR.pdf)，13 页；DOI `10.1145/3386569.3392475` | 2026-08-29 | SHA-256 `1EAA6AED7FCB7560814ECE83A744AD1DA1ED70EEFA7C91D5E42304E2828AEBBE` | 作者版 TOG/SIGGRAPH 2020 正文；13/13页已读取并render核对 |
| Supplemental `S` | [作者项目页 supplemental](https://jannovak.info/publications/CNSR/CNSR_supplementary.pdf)，7 页 | 2026-08-29 | SHA-256 `C14948C07DB1E433BCA75895F179FDC93AA14DDDCFA52FEF6670524ADCCC2363` | generator topology/parameters/training time、dataset细节、resolution/G-buffer/reflection消融；7/7页已读取并render核对 |
| Official project `A` | [Jan Novák project page](https://jannovak.info/publications/CNSR/index.html) | 2026-08-29 | 固定 URL | 正式 paper/supp/code/video/talk入口、摘要、卷期和作者 |
| Official code `C` | [jonathangranskog/shading-scene-representations](https://github.com/jonathangranskog/shading-scene-representations/tree/9c9033a1ca05095c7e2ccfeb4da3046b687bef3d)，MIT；公开 repo 仅一个 commit | 2026-08-29 | commit `9c9033a1ca05095c7e2ccfeb4da3046b687bef3d`；source ZIP SHA-256 `452297EF03CB8E65525B0CF48E3FA485DD0A1E16FBA1143027741DE6AC2A1A3F` | 静态审计 encoder/generators、partition、loss、dataset和configs；未执行旧Torch/OptiX pipeline |
| Official datasets/checkpoints `C-assets` | official README 指向 Google Drive 的 PrimitiveRoom/ArchViz pre-rendered datasets 与两个 1M checkpoints | 2026-08-29 | 未下载/未hash；source archive 中仅 placeholder README | 核对公开可用性和README identity；未审计大文件manifest、checkpoint tensors或正式 dataset SPP |
| Conference video/talk `A-video` | project page提供 MP4/video/talk入口 | 2026-08-29 | 未下载 | 当前 P/S/C 已能闭合主要机制；不从视频补写未在formal source出现的配置 |
| NeuralShading evidence `N` | [runtime contract](../../../../../docs/contracts/scattering_backend.md)、[experiment framework](../../../../../docs/research/experiment_framework.md)、[method constraints](../../../../../.trellis/spec/project/method-constraints.md)、LightFormer/Active Exploration个体报告及[NVIDIA correspondence](../../../archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md) | 2026-08-29 | repo-local | 只用于 §§13–15；不回填成2020论文事实 |

Supplemental 标题页把题名写成 “... for **Shadow** Inference”，而正式论文、DOI与项目页是 “... for **Shading** Inference”。其 DOI、authors、article 135 与内容一致，本报告将它视为正式 supplemental 的标题 typo，不建立第二篇论文身份。[S p.1; A]

## 3. 原论文的问题、假设与贡献边界

### 3.1 问题和运行假设

给定一个可由 classical renderer访问的 3D scene，模型先观察 `n` 个 views：

\[
\{(c_i,i_i,g_i)\}_{i=1}^{n},
\]

其中 `c_i` 是 camera matrix，`i_i` 是包含目标 light transport 的 HDR color image，`g_i` 是 visible surfaces 的 G-buffer。encoder `R` 得到 partial representation `r_i=R(c_i,i_i,g_i)`，再用 permutation-invariant componentwise average：

\[
r=\operatorname{avg}\{r_i\}_{i=1}^{n}.
\]

novel view `v` 时，generator接收 `(c_v,g_v,r)` 并输出 shaded image。正式大多数实验用 `n=3`、observation/query training resolution `64×64`，但 test时可提高 query G-buffer resolution。[P Fig.2, §§3.1–3.4; S §1]

作者有意只把 world-space position、normal和object ID放入 attribution实验的 G-buffer；material、lighting以及offscreen/occluded geometry必须经global representation传递。Indirect-GI application则主动偏离这项分析性约束，额外提供 albedo、roughness与direct-illumination buffer。[P §§3.3,6.1]

### 3.2 正式贡献

- 把 latent 显式约束为 lighting `R_L`、geometry `R_G`、material `R_M` partitions；batch内只变化一种scene factor，通过forward averaging与gradient correction促使不变因素的partition保持一致；[P §4.1]
- 用softmaxed trainable size parameters与逐渐变尖的sigmoid boundaries学习每个partition的relative size，使bit allocation适配各因素entropy；[P §4.2, Eqs.1–3]
- 增加不会送入generator的null partition `R_∅`，并用active-dimension penalty压缩representation；[P §4.3]
- 用 gradient×input 把输出pixel/patch回溯到latent partitions、observation pixels/channels和query G-buffer，定位U-net对visible G-buffer geometry过度依赖、对offscreen casters失败的原因；[P §5]
- 添加并行训练的auxiliary shadow pixel generator，使shared encoder提取更多geometry；[P §5.4]
- 展示hybrid indirect illumination与lighting-partition swap/interpolation两个应用。[P §6]

作者明确不声称 Pool encoder、GQN/U-net/pixel generators本身新颖；贡献在partition constraints、分析与组合。[P §3]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Scene source | PrimitiveRoom或ArchViz procedural 3D scene instance，含geometry/material/light/camera可供path tracer和G-buffer renderer访问 | 固定recipe内的variable scenes；不是任意unseen scene family | [P §3.4; S §2] |
| Observation | HDR beauty `i_i` + geometry G-buffer `g_i` + camera `c_i` | 正式通常3 views，每view `64×64`；base G-buffer为position3+normal3+ID1，beauty3，共10 image channels；camera matrix 16 | [P §§3.1,3.3–3.4; C configs/code] |
| Persistent/query-time scene state | view-independent global vector `r` | 128–512 scalars；多数released configs为256；按lighting/geometry/material/(optional null)连续切分 | [P §§3.4,4; C configs] |
| Novel-view query | global `r` + novel camera `c_v` + novel G-buffer `g_v` | base query G-buffer 7 channels；可在GI/reflection实验加入albedo、roughness、direct、mirror direction/hit/normal | [P §§3.2–3.3,6.1; S Figs.7–8] |
| Output | full HDR beauty，或application-specific indirect/shadow buffer | image-space RGB；paper base train 64²，pixel generator测试到128²/256²，GI timing at1k² | [P §§3.4,5.4,6.1; S §1/Figs.3–4] |
| Spatial coordinates | query G-buffer world positions/normals/object ID；camera matrix | screen-space pixels + world-space surface attributes；不显式query arbitrary `wi` | [P Fig.2, §3.3] |
| Temporal domain | static scene/view synthesis；novel views逐帧可query同一个`r` | 无history/reprojection/temporal loss；temporal stability为per-pixel generator结构的qualitative性质 | [P Abstract, §3.2; S §1] |
| Validity restrictions | scene需落在trained procedural recipe；novel scene需先获得三张observations并重新encode | constrained cross-instance generalization，非zero-shot arbitrary scene | [P §§3.4,7, Fig.21] |

base pixel generator的每个pixel独立处理，但不是local material evaluation：同一pixel仍读取global scene latent与camera matrix，并输出包含visibility、indirect transport、reflection等scene-dependent radiance。[P §§3.2–3.3; S §1]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

```text
offline/procedural dataset
  batches where exactly one factor varies: lighting OR geometry OR material
    → for each scene: 3 observation views + 1 query view
    → Pool encoder independently maps each observation to ri
    → componentwise mean across observations gives r
    → training-only static/adaptive masks average invariant partitions across batch
    → image generator consumes r + query camera + query G-buffer
    → HDR image/residual loss; optional null penalty/auxiliary head

runtime on a new in-distribution scene
  render 3 high-quality 64² observations + G-buffers
    → encode and average once into global r
    → for each novel view: render query G-buffer
    → Pixel/U-net/GQN generator → beauty/indirect/shadow image
```

当一个训练batch只变化 factor `p` 时，`p` partition保留每个scene自己的observation-mean activation；其他partitions在整个batch与observations上求均值，使它们对该变化不敏感。backprop时，不变partitions的gradient被替换为它相对per-dimension batch mean gradient的差。[P §4.1]

### 5.2 Static、adaptive 与 null partition

Static partition把 `R` 近似等分为三个不重叠段。Adaptive partition为 `k` 个partitions学习 `s=(s_1,…,s_k)`，softmax `σ(s)`给出partition-of-unity lengths：

\[
b^-(i)=\lVert R\rVert\sum_{j<i}\sigma(s)_j,\qquad
b^+(i)=b^-(i)+\lVert R\rVert\sigma(s)_i.
\]

对dimension `d`，中心/左右neighbor weights为：

\[
w_p(d)=1-w_{p^-}(d)-w_{p^+}(d),
\]
\[
w_{p^-}(d)=1-S(\alpha(d-b^-(p))),\qquad
w_{p^+}(d)=S(\alpha(d-b^+(p))).
\]

`α`随训练增加，使fuzzy boundaries收敛到step；representation视为circular domain，所以首尾partition相邻。[P §4.2, Eqs.1–3, Fig.6]

正文同段写“all partitions initial size `||R||/(k+1)`”，但上述 `k` 个softmax lengths求和为1，若没有额外null partition则均匀初始化应为`||R||/k`。official code在无null时建立3个全1 `deltas`，有null时建立4个，因此实际分别初始化为1/3和1/4。该formal文字/定义冲突未发现勘误。[P §4.2; C `representation.py:19-25`]

Null partition不送入generator。论文loss增加：

\[
\beta(\lVert R\rVert-\lVert R_\emptyset\rVert),
\]

促使unused dimensions进入reservoir。[P §4.3] code则实现为

```text
partition_loss * r_dim / 256 * (1 - empty_fraction)
```

即相对256维做额外normalize。若把code系数记作 `λ=partition_loss`，它与正文逐active-dimension系数的对应关系是 `β=λ/256`；因此formal figure的 `β=4×10^-4` 需要 `λ=0.1024`，并不等于release default `0.01`。公开JSON没有null配置或formal `β/λ` override，故只能确定归一化关系，不能恢复正式run的exact config。[C `representation.py:119-121`; `settings.py:24-25`; configs]

### 5.3 Pool encoder逐层配置

正式main experiments使用Eslami-style Pool/Tower encoder。以base 10-channel observation、`r_dim=256`、64²为例：[P §3.1; S Table 1; C]

| 顺序 | 输入 | 层/运算 | activation | 输出 | locator |
|---|---|---|---|---|---|
| 1 | beauty3+position3+normal3+ID1 | Conv `2×2`, stride2, `10→256` | ReLU | `256×32×32` | [C `representation.py:243,265`] |
| 2 | previous | Conv `3×3`, `256→128` | ReLU | `128×32×32` | [C `:245,268`] |
| 3 | previous | Conv `2×2`, stride2, `128→256`；加skip Conv `2×2`, stride2, `256→256` | ReLU before/add | `256×16×16` | [C `:247-250,266-270`] |
| 4 | previous + camera16 spatial repeat | Conv `3×3`, `272→128` | ReLU | `128×16×16` | [C `:250-253,271-275`] |
| 5 | previous | Conv `3×3`, `128→256`；加camera-conditioned skip Conv `3×3`, `272→256` | ReLU before/add | `256×16×16` | [C `:250,255,271-275`] |
| 6 | previous | Conv `1×1`, `256→256` | ReLU | `256×16×16` | [C `:257,276`] |
| 7 | previous | average pool over `16×16` | none | `256×1×1 = r_i` | [C `:260,277-279`] |
| aggregate | three `r_i` | componentwise mean | none | global `r` | [P §3.1; C `representation.py:137-142`] |

Supplemental报告 Pool encoder `2,000,644` trainable parameters。按release topology，该数包含256-dim backbone以及4个adaptive size scalars，和带null的实验一致；无null时只需3个size scalars。[S Table1; C]

### 5.4 Image generators逐层配置

| 模块 | 正式配置 | 参数/训练 | 关键行为 | locator |
|---|---|---|---|---|
| Pixel generator | `1×1` input projection to512；10 hidden `1×1` layers、每层512；representation、camera和G-buffer在输入及每个hidden layer重复concat；final `1×1→RGB` | `4,199,939` generator params；encoder+generator约8.5天 | code用LeakyReLU（PyTorch default negative slope 0.01）和default final ReLU；每pixel独立，resolution independent但texture detail会模糊 | [S §1.1/Table1/Figs.1,3–4; C `generator.py:450-526`; settings] |
| U-net | 7 scales；每scale两个`3×3` Conv+ReLU；encoder channels128,256,之后5层512；每level末`2×2` max-pool；decoder `4×4` transpose Conv stride2并skip concat，channels镜像 | `80,596,099`；encoder+generator约8.5天 | `r,c_v`注入每个encoder scale；大receptive field，但高resolution出现shadow shrink/other artifacts并过度依赖query G-buffer | [S §1.1/Table1/Figs.1,4; C `generator.py:270-411`] |
| GQN generator | 12 unshared convolutional LSTM cores；state `16×16×128`；per-core latent `16×16×3`；query G-buffer downsample至16² | `147,735,199`；encoder+generator约10天 | probabilistic ELBO；std从2.0 anneal到0.7（code 200k）；splotchy/blurry且未充分利用G-buffer | [S §1.1/Table1/Figs.1–3; C `generator.py:15-214`] |
| Auxiliary shadow pixel generator | 与main generator共享encoder/global `r`，单独预测grayscale partial light visibility；最初的U-net案例把shadow output再输入U-net；loss与main相同但对shadow reference求值，两loss相加 | exact params/formal config未报告 | U-net案例同时得到shadow image和encoder steering；后续Pixel main消融也共享encoder，但正文未说明auxiliary shadow image是否再作为Pixel输入 | [P §5.4, Figs.14–16] |

Release `PixelCNNSettings`默认8 hidden layers，但paper configs显式写10；因此配置生效时与supplemental一致。`propagate_representation`和`propagate_viewpoint`默认true，`archviz/room *_beauty.json`显式或默认再使`propagate_buffers=true`，才与supplemental“全部输入送入每层”一致。[C settings/configs]

### 5.5 容量和语义分界

- per-scene可变信息在128–512 scalar global `r`；[P §3.4]
- visible high-resolution geometry由query G-buffer按pixel提供；
- dataset固定规律、常见material/lighting biases可被generator weights吸收，论文明确指出固定glossy floor material可能被weights记住；[P §5.2]
- offscreen geometry、lighting/material appearance与观察中全局transport clues必须经`r`；
- image-space synthesis capacity主要在4.2M–147.7M parameter generator，而不是256维latent本身。[S Table1]

这解释了为什么partition attribution可分析“信息流向”，却不能把partition size直接解释为scene物理参数的bit-exact编码。

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Datasets | PrimitiveRoom与ArchViz，各144k procedurally generated training instances | [P §3.4] |
| Batch-file construction | official script默认9000 training batches +1000 testing batches；每batch 16 scenes；每scene 4 views（3 observations+1 query），所以9000×16=144k training scene instances | [C `create-dataset-json-files.py:20-24,77-80`; configs] |
| PrimitiveRoom | rectangular room；random primitive presence/XZ position/Y rotation；mirror、GGX roughness0.5 glossy、random-hue Lambertian；random-hue diffuse walls；一个position-random spherical emitter，正式supp说intensity固定 | [S §2] |
| ArchViz | living/dining regions；random furniture presence/placement；sofa/armchair/carpet diffuse colors；table/chairs material；teapot location/material；4 wallpaper patterns+random hue；ceiling luminaire位置与window-like emissive quad；两灯intensity/tint变化；back-wall mirror位置变化 | [S §2] |
| GT/reference | beauty/feature buffers由path tracing生成，含multi-bounce color bleeding和mirror reflections | [P §3.4] |
| Factor batches | 每training batch随机选择lighting、geometry或material之一变化，其他因素固定；factor ID写入batch file | [P §4.1; C create/generate scripts] |
| Views | 每scene 3 random observations +1 query；训练64²；test query resolution可提高 | [P §3.4; S §1] |
| Base observation buffers | beauty, position, normal, ID；camera16 | [P §§3.1,3.3; C configs] |
| Base query buffers | position, normal, ID；camera16 | [P §3.3; C configs] |
| Stored precision | release dataset generator把poses和all passes写为float16 compressed NPZ | [C `generate-dataset.py:101-123`] |
| Train/test | main Fig.9在4000 held-out procedurally generated test scenes求平均；supp generator comparison在16个hand-picked difficult scenes | [P §4.2/Fig.9; S §1.2/Fig.2] |

正式 P/S 未报告 training/reference SPP。release 两份 `save_*` configs写1 spp；cached Pixel/GQN/U-net configs分别出现4/1/512 spp，但cached loader读取已渲染NPZ，不会按该字段重新render。README又明确警告example defaults会产生noisy low-spp images。因而这些字段都不能升级为formal 144k dataset的render SPP。[C README/configs/dataset loader]

official dataset generation默认9000/1000 batches，但README示例命令只生成90/10并明确称不足以训练。报告必须区分script default、README smoke/example与formal dataset，不能把90+10写成论文split。[C README; create script]

## 7. Loss、optimizer 与训练 lifecycle

| 项 | 正式配置 | locator |
|---|---|---|
| HDR transform | `log(i+1)`，input、prediction与loss都在log space；最终`exp(y)-1`恢复HDR | [S §2 HDR images; C generator] |
| Deterministic generator loss | pixelwise L1 + DSSIM，经验scale到近似同量级；release实现为`2*abs(pred-log1p(target)) + (1-SSIM(log1p(target),pred))` | [P §3.4; C `generator.py:539-551`] |
| GQN loss | 原GQN ELBO；normal output std anneal，learning rate不anneal | [S §1.1; C generator] |
| Partition loss | static/adaptive disentanglement通过batch averaging/gradient correction；optional null penalty `β(active dims)` | [P §4] |
| Auxiliary loss | main image loss + auxiliary shadow image loss | [P §5.4] |
| Optimizer | Adam，LR `1e-4`；release用PyTorch default betas/epsilon，无weight decay | [P §3.4; C `training.py:24,37`] |
| LR schedule | formal fixed LR；GQN只anneal likelihood std | [S §1.1; C] |
| Batch | 16 scenes；每scene三observations+一query | [P §3.4] |
| Steps/epochs | 1,000,000 batches =111 epochs | [P §3.4] |
| Hardware/time | single NVIDIA Tesla V100；pixel/U-net complete model约8.5天，GQN约10天 | [P §3.4; S Table1] |
| Seed/model selection | formal seed、run count、checkpoint selection未报告；release configs均`seed=-1`（不固定） | [C configs/settings; gap] |
| Partition sharpening | paper只说progressive；release defaults start5→end100 over500k，final100 over100k | [P §4.2; C `settings.py:50-55`, `representation.py:35-41`] |

Paper要求对batch-invariant partitions做gradient correction，但没有给出correction strength或逐元素方程。Code实现为 `g' = λm(ḡ-g) + (1-λm)g`，其中 `λ=gradient_reg`，`m`是由当前varying factor决定的batch-invariant-dimension mask；`λ=0`时直接return。八份公开JSON并不一致：两份Pixel beauty配置和两份`save_*`配置未override，继承default `0`；两份GQN与两份U-net配置显式设为`1.0`。因此released Pixel/pretrained-compatible config若用于训练只做forward averaging，不执行correction，而GQN/U-net configs会执行；formal main run到底采用哪个strength仍不可恢复。这是paper↔release identity/config gap，不是论文方法失败。[P §4.1; C `settings.py:55`, `representation.py:154-181`; configs]

另一个身份冲突是：paper与README pretrained checkpoints均称1M，公开两份Pixel beauty JSON却写2M；其余configs通常也是2M，但`room_beauty_gqn_buffers.json`甚至写20M。README明确说两个1M Pixel checkpoints使用对应beauty JSON加载，故JSON可作为兼容topology/inference入口，不能当作formal training-length manifest，也不能把20M反推为supplemental GQN正式训练长度。[P §3.4; S Table1; C README/configs]

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Per-scene prepare | render three random 64² observations+G-buffers，Pool encode each andmean aggregate为`r` | [P §3.4] |
| Per-view query | render novel-view G-buffer；broadcast `r,c_v`；generator输出image | [P Fig.2, §3] |
| Parameters | Pool encoder2,000,644；GQN147,735,199；U-net80,596,099；pixel4,199,939 | [S Table1] |
| Per-scene state | `r` 128–512 scalars；formal precision/serialization未报告 | [P §3.4] |
| G-buffer reads | base7 channels/pixel；GI/reflection variants更多；exact bandwidth/fetches未报告 | [P §§3.3,6.1; S Fig.8] |
| Precision | release training datasetfloat16；model/inference precision未报告，PyTorch通常由runtime决定 | [C generate script; gap] |
| Hardware/backend | unoptimized PyTorch inference；OptiX ray tracing；RTX6000 for GI timing | [P §6.1] |
| Indirect prediction | 1k×1k，400 ms | [P §6.1] |
| Direct buffer | 1k×1k, 8k spp，7 min | [P §6.1] |
| Indirect path-traced reference | 1k×1k, 8k spp，25 min | [P §6.1] |
| Prepare/amortization scope | 400 ms明确是predict indirect；three observation render/encode、query G-buffer/direct trace与data movement未证实包含 | [P §§3.4,6.1] |

Pixel generator每pixel `1×1` MLP，能在test时提高resolution，但成本随pixel count线性增长；它不是“单次query的小MLP material evaluator”，因为输入含global scene state、camera与scene G-buffer，输出是integrated radiance。U-net/GQN还包含大screen-space/recurrent state。完整方案不符合本项目material program的fixed small reads/state合同，但可作为scene-level teacher/baseline。[N/I]

公开 `archviz_beauty.json`把Pixel generator `render_size`写128，而paper/supplemental说明formal generator comparisons和training query G-buffer是64²；README又说1M ArchViz checkpoint配该JSON。该release config可能用于checkpoint inference或后来resolution setting，无法证明formal paper model在128²训练。[P §3.4; S §§1.1–1.2; C config/README]

## 9. 实验 protocol、baseline、指标与结果

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| Partition quality penalty | Pixel generator；4000 held-out scenes；HDR先gamma2.2、clip[0,1]、8-bit后算PieAPP | monolithic/static/adaptive | MAPE/PieAPP/RMSE | Primitive：mono `.09/.59/.07`，static `.14/.66/.08`，adaptive `.13/.66/.09`；ArchViz：mono `.05/.35/.12`，static `.05/.37/.16`，adaptive `.05/.36/.14`。partitioning有质量penalty，adaptive只部分缓解 | [P Fig.9 + footnote] |
| Entropy-sensitive sizes | PrimitiveRoom many-lights vs many-materials；adaptive+null | learned partition percentages | size % | many lights L/G/M/null=`39/20/37/4%`；many materials=`30/23/43/4%` | [P Table1] |
| Null compression | PrimitiveRoom，`β=4e-4`，400k iterations | initial R=128/256/512 | active dims/visual | converges `49/51/57` active dims；相互质量接近但低于uncompressed，teapot color丢失 | [P Fig.10] |
| Compression tradeoff | R=256，400k | β=`4e-4,4e-5,4e-6` | active dims/visual | `51,180,228` active dims；teapot/shadow差异明显 | [P Fig.11] |
| Generator comparison | each generator own encoder，ArchViz；3 obs；train64²；16 hand-picked difficult scenes | GQN/U-net/pixel | MAPE/PieAPP/LPIPS | Primitive GQN/U-net/pixel=`.66/.43/.52`, `1.77/1.25/.90`, `.22/.18/.16`；ArchViz=`.28/.11/.09`, `1.45/.63/.50`, `.14/.06/.04` | [S Fig.2] |
| Generator resolution | train64²，test64/128/256 | U-net vs pixel | qualitative | U-net shadow/details随resolution缩小/错位；pixel更稳定但textures模糊 | [S Figs.3–4] |
| G-buffer dependency | Pixel generator with coordinate map only vs geometry G-buffer | no/with G-buffer | qualitative/training observation | no G-buffer训练更慢、结果显著模糊 | [S Fig.5] |
| Out-of-view caster attribution | in-view/out-of-view cylinder，pixel vs U-net；red patch gradient×input | representation vs G-buffer | relative attribution/visual | out-of-view U-net geometry partition仅10%且漏shadow；aux后17%并恢复shadow；pixel依赖global geometry更强 | [P Fig.14] |
| Auxiliary generator | pixel main with/without auxiliary shadow head | same dataset/model family | visual, partition %, aggregate metrics | partitions L/G/M/null从`25/17/53/5%`变`23/28/43/6%`；shadow略好、material变差、quantitative metrics unchanged | [P Figs.15–16, Table2] |
| Indirect GI hybrid | Pixel generator predictsindirect；direct ray trace+sum | path-traced indirect | qualitative/runtime | local/distant interreflection较好，mirror reflection过模糊；1k² prediction400ms；8k-spp direct7min、indirect reference25min | [P Fig.17, §6.1; S Fig.7] |
| Reflection features | add roughness→reflection direction→reflection hit position+normal | nested G-buffer variants | qualitative | progressively改善teapot/mirror reflections；最后variant需额外trace reflection ray | [S Fig.8] |
| Latent editing | swap/interpolate lighting partitions | encoded scenes/reference | qualitative | shadows/highlights/color bleeding随partition变化；material transfer只在identical geometry mapping下直接成立 | [P Figs.18–19, §6.2] |
| OOD material | wall使用train未出现的gray | reference | qualitative failure | prediction把gray walls映成green/pink shades | [P Fig.21] |

Supplemental正文称pixel generator在“all metrics”最好，但PrimitiveRoom MAPE图值是U-net `0.43`优于pixel `0.52`。图中legend和数字清晰，故保留formal prose↔figure conflict，不把Pixel写成所有条件无例外第一。[S §1.2/Fig.2]

不同指标集不可直接合并：main Fig.9用4000 procedurally held-out scenes与MAPE/PieAPP/RMSE；supp Fig.2用16个hand-picked scenes与MAPE/PieAPP/LPIPS。二者也没有seed/CI，不能把微小差异写成稳定ranking。

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `ablation-inferior` | static partition | 通常质量最差 | equal capacity不匹配各因素entropy | partition语义约束有真实quality cost，不是免费可解释性 | [P Fig.9, §4.2] |
| `ablation-inferior` | adaptive partition vs monolithic | 比static缓解但仍低于monolithic | disentanglement本身保留negative impact | 不能只报可编辑性而省略capacity/quality penalty | [P Fig.9, §7] |
| `author-negative` | high β/null compression | active dims降至约51，但teapot color/shadow信息丢失 | compression ratio与information loss trade off | null size不是无损“自动选最优latent” | [P Figs.10–11] |
| `author-negative` | GQN generator | 最大模型却出现splotchy/blurry、G-buffer利用差 | architecture不是为image-to-image translation设计 | parameter count不等于task-aligned capacity | [S §§1.1–1.2, Figs.2–3] |
| `author-negative` | U-net extrapolate resolution | shadows/structures随resolution改变 | convolutional neighborhood/receptive-field绑定training pixel scale | screen-space network的filter semantics不能由resize自动获得 | [S Figs.3–4] |
| `author-negative` | Pixel generator without G-buffer | training慢、结果模糊 | global latent/coordinate不足以恢复visible per-pixel geometry | scene representation不替代cheap raster visibility | [S Fig.5] |
| `author-negative` | U-net out-of-view shadow caster | 漏掉shadow | U-net过度利用visible query G-buffer，未从geometry partition取足信息 | strong local input可造成global latent shortcut/underuse | [P Fig.14] |
| `mixed-result` | auxiliary shadow generator + pixel main | shadow略好、geometry partition17→28%，但materials变差、总体metrics unchanged | auxiliary head重平衡encoder attention，不保证整体更好 | 是training-time information allocation工具，不是无条件质量模块 | [P Figs.15–16, Table2] |
| `author-negative` | base indirect-GI reflections | mirror/teapot反射过模糊 | base G-buffer缺reflection-ray information | learned global latent不能替代明确的secondary-hit locator | [P Fig.17; S Fig.8] |
| `ablation-improved-with-cost` | reflection direction/hitpoint G-buffer |质量逐步改善 | 给generator更多物理路径线索 | hitpoint variant增加一条reflection ray，成本域已改变 | [S Fig.8] |
| `author-negative` | unknown gray wall material | 输出green/pink | train recipe未出现该wall state，generalization差 | constrained procedural interpolation不可称native-source OOD泛化 | [P Fig.21, §7] |
| `known-limitation` | fine-grained material transfer | geometry不同时不能直接swap material partition | 缺surface semantic mapping | coarse L/G/M disentanglement不等于per-object compositionality | [P §6.2, §7] |
| `reported-conflict` | “Pixel all metrics best” | PrimitiveRoom MAPE图为U-net更低 | 未解释 | 保留，不代作者改值 | [S §1.2/Fig.2] |

论文没有报告partition order、number of partitions、sigmoid schedule、generator width等系统性failed sweep；不能从最终选择反推作者试过哪些组合。auxiliary generator是明确的mixed result，不应只记成功shadow图而删除material退化和unchanged aggregate metrics。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Title | Shading Inference | Shadow Inference | repo为Shading | supplemental标题typo；身份仍为同一DOI/article |
| Pool encoder | 引用Eslami pool，输出1×1×k | 参数2,000,644 | topology可完整恢复 | correspondence基本闭合；参数数与256dim+4size scalars一致 |
| Pixel generator | per-pixel MLP | 10 hidden×512，inputs每层传播，4,199,939 params | code10由JSON override；LeakyReLU/final ReLU；propagate flags | formal层数闭合；activation/final与默认flags主要来自code |
| Partition gradient | forward averaging + gradient replacement；未给strength/方程 | 未补配置 | default `gradient_reg=0`；Pixel beauty与save configs继承0，GQN/U-net四份configs设1.0；code方程见§7 | released configs分裂：Pixel路径不执行、GQN/U-net路径执行；formal main strength不可恢复 |
| Adaptive initial size | `k` softmax partitions，却写`R/(k+1)` | 未解释 | 无null3个uniform、有null4个uniform | formal内部定义冲突；code选择`/k` |
| Null loss | `β(active dimension count)`；Fig β=`4e-4` | 未补 | code为`λ/256×active dims`，default `λ=.01`；formal β对应`λ=.1024`；无null config | normalization闭合，exact formal config不可恢复 |
| Training length | 1M batches/111 epochs | complete models8.5/10天 | README checkpoints1M；多数JSON 2M，Room GQN JSON 20M | JSON不是formal run manifest；20M也不能回填正式GQN run |
| Training resolution | observation/query G-buffer64² | generator comparison全train64² | `archviz_beauty.json` generator128 | release config与formal ArchViz training resolution冲突 |
| Dataset SPP | path-traced/high-quality，未给SPP | 未给SPP | save configs1 spp；cached Pixel/GQN/U-net字段为4/1/512但不rerender；README警告default noisy | 不可用release字段补正式SPP |
| Seeds | 未报告 | 未报告 | JSON `seed=-1` | release默认不固定，正式run不可重复性未知 |
| Formal application configs | indirect/aux/null/reflection variants | only architecture/figures | release无对应JSON | 不能端到端复现正式application/ablations |
| Assets | 144k recipe、official links | dataset details | Drive datasets/checkpoints未在archive | 可用性存在，当前未审计manifest/checkpoint |

official repo只有一个commit并提供MIT license、PyTorch1.5/torchvision0.6 requirements和OptiX5.1.1 build说明。它是有价值的static correspondence source，但缺锁定CUDA/driver、formal command/manifest与全部figure configs；本author pass没有在当前PyTorch2.11/CUDA12.8环境强行运行旧pipeline。[C]

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

- 大多数models仍有明显visual artifacts；quality/scaling不是靠增大network/dataset必然消失。[P §8]
- explicit disentanglement造成prediction-quality penalty；adaptive size不能消除该penalty。[P §§4.2,7]
- coarse lighting/geometry/material partition不是per-object/fine-grained compositionality。[P §7]
- 对训练recipe外材料/配置generalization差。[P Fig.21, §7]
- base model对high-frequency features、shadows和reflections处理差；U-net和pixel有不同failure modes。[P §§5–7; S §1]
- material partition只可在geometry/material-to-object mapping一致时直接swap。[P §6.2]
- indirect-GI是未来hybrid方向演示；作者明确不与state-of-the-art GI renderer比较。[P §6.1]
- model依赖classical renderer提供3D scene/G-buffer，非image-only novel-scene reconstruction。[P Scope/§3]
- observation-set形式未必是最佳scene input；作者列出omnimap、voxel、radiance samples、light fields/text等替代方向。[P §7]

### 12.2 未报告/材料不可得

- formal dataset/render SPP、path tracer convergence/noise、exact train/test seed和camera sampling distribution；
- 1M formal run config、gradient correction weight、sigmoid schedule、nullβ到code normalized loss的mapping；
- optimizer betas/epsilon是否严格使用release defaults、mixed precision、checkpoint selection和multi-run variance；
- official Drive dataset/checkpoint hashes、tensor shapes、dataset manifest和每figure checkpoint mapping；
- null/auxiliary/indirect/reflection实验的official JSON和完整topology/data recipe；
- MAPE denominator/epsilon、DSSIM exact definition/weight，除code release可见实现外的formal version identity；
- 400 ms是否含encoder、three observation render、G-buffer、transfer/synchronization；single-scene prepare cost；
- per-frame temporal protocol、moving scene/light、history/reprojection和temporal metric；
- model state bytes、activation memory、FLOP/MAC、precision、tile scheduling和1k² generator的exact configuration；
- result error bars、seeds、CI；4000 test与16 hand-picked metrics不可互相代替；
- supplemental “Pixel all metrics”与Primitive MAPE图值冲突的correction；
- paper `R/(k+1)`与softmax partition count冲突的correction。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

CNSR的256维global latent并不独自表示scene appearance。容量分布为：

1. **observations**：三张HDR path-traced images已包含多bounce transport；encoder学习压缩observed evidence；
2. **query G-buffer**：直接visible geometry以per-pixel world position/normal/ID输入，避免latent承担完整3D visibility；
3. **global latent**：主要传递lighting、materials与offscreen/occluded geometry clues；
4. **generator weights**：4.2M–147.7M shared parameters存dataset prior，甚至可吸收不变化的floor material；
5. **classical renderer**：负责primary visibility，hybrid GI还负责direct lighting，reflection改进甚至加secondary ray。

所以“scene representation只有约51 active dims”不能解释为51个scalar足以普遍表示GI；这是特定recipe、共享generator、三张transport-rich observations和G-buffer共同条件下的best observed compressed latent。[P/S; I]

### 13.2 成功所依赖的假设

1. 每个新scene可支付三张高质量observations及G-buffer，并可对novel view产生query G-buffer；
2. procedural family的variation受限，shared generator能学习强scene prior；
3. lighting/material/geometry可通过“每batch只变一个因素”的生成器获得近似正交supervision；
4. three views足以暴露决定novel-view appearance的clues，未观察部分可由dataset prior推断；
5. output允许image-space approximation和一定blur/artifacts；
6. scene静态或重编码成本可接受；
7. hybrid renderer可把高频/visible/secondary-ray线索显式加入G-buffer。

这些假设与本项目local material compiler不同：native source material的GT来自reference query，不应依赖scene observations或把lighting/visibility烘进material latent。[I]

### 13.3 可迁移机制

- **受控因子batch**：对可编辑native source axes，一批只改变一个语义因素，可诊断latent是否把不变因素错误耦合；前提是source本身具有这些axes，不强造lighting/material/layer参数。
- **adaptive/null capacity diagnostic**：先给较大latent，再观察active semantic partitions是否收缩，可用来判断容量需求；结果只作为soft-line，不称upper bound或自动产品size。
- **auxiliary-head steering**：用training-only任务迫使shared encoder保留主loss不敏感但部署重要的信息；例如对局部散射可考虑grazing/high-frequency或sampler-support诊断，但必须证明主目标不退化。
- **attribution对输入捷径**：比较模型对source latent、query coordinates、analytic core和G-buffer-like features的gradient×input，定位某个强输入让latent失效；归因是诊断，不是因果证明。
- **hybrid分工**：把便宜、确定、high-frequency的物理量显式计算，让network只预测昂贵residual。本文indirect GI与reflection-ray ablation直接支持这一路径，但新增输入成本必须计入matched control。
- **明确representation reuse**：three observations→global `r`类似scene-level `prepare()`，novel views复用；启发本项目严格量化`prepare`和per-query成本边界。

### 13.4 不能迁移的部分

- scene latent含lighting、visibility、offscreen geometry，不能成为material program；
- image generator输出integrated radiance，不是bare local scattering `f`，也没有matched `sample()/pdf()`；
- coarse L/G/M partition依赖procedural factor oracle，不能要求任意MaterialX/MDL/measured source提供同类分解；
- swap encoded lighting不等于接收任意light parameter并跨scene泛化；
- U-net/pixel per-pixel resolution行为不能替代材质footprint filtering/LOD；
- 400 ms/1k² RTX6000是scene image inference，不与tiny shader MLP single-query cost可比；
- 三张observations的online acquisition与current GPU-resident material reference query语义不同。

### 13.5 与scene transport第二波的关系

CNSR是完整scene-level transport基线，第二波必须研究而不能降为“与local无关”。它回答了三个关键问题：

1. 只给visible G-buffer会形成visibility shortcut，offscreen effects需要global representation或额外rays；
2. 低频indirect transport较易image-space预测，high-frequency reflection/shadow需要明确path clues；
3. scene representation的“可编辑”取决于训练factorization和scene prior，而不是latent命名。

相对LightFormer，CNSR用三张transport-rich observations编码整个scene，而不是围绕light构造显式light-oriented representation；相对Active Exploration，它依赖离线固定144k data，而不是在training中按动态`loss×update-influence` target选择scene/patch state并render/reuse samples。具体数值不直接排名，机制差异会进入scene/local cross-synthesis。[N downstream reports; I]

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

CNSR不是Real-Time Neural Appearance Models的架构precedent，直接fidelity分类为`not-applicable`。影响集中在“不要把scene global latent机制误迁到material compiler”和“可借用training diagnostic”：

| 主题 | CNSR | 当前 NVIDIA functional reproduction | 分类与影响 |
|---|---|---|---|
| Query semantics | scene observations+novel G-buffer→image radiance | source material state+directions→bare `f` | `not-applicable`：不能以CNSR image quality为NVIDIA fidelity证据 |
| Latent | per-scene L/G/M global vector，含lighting/visibility clues | per-material/spatial latent，不应含scene lighting | `intentional-deviation`若有人拟议加入scene latent：会改变产品语义，不是忠实修复 |
| Prepare | 三张path-traced observations编码 | material latent/filter/view-conditioned prepare | `not-applicable`；只能启发单独计费amortization，不能复制input |
| Evaluator | 4.2M pixel generator或更大image net | fixed small MLP | `not-applicable`：成本和output domain完全不同 |
| Auxiliary task | shadow head重分配geometry capacity，mixed result | 当前evaluator training可有diagnostic ablation | `not-applicable`于fidelity；若采用必须作为新candidate并做matched主质量检查 |
| Partition/null | semantic capacity allocation/compression | compact latent/network候选 | `not-applicable`于复现；可做capacity diagnostic，不能改名为NVIDIA方法 |
| G-buffer/analytic residual |显式high-frequency/visible information | current analytic core/coordinates可能承担类似cheap cues | `not-applicable`于fidelity；可提出新方法假设，但必须计入fetch/prepare/runtime |

当前correspondence中的functional reproduction预算、source adaptation、cosine-to-bare-`f` adapter和MethodBundle/Slang parity均由RTA自身证据决定；CNSR不新增`suspected-defect`。[N correspondence]

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：training-only auxiliary head可迫使local evaluator保留主loss忽略的sharp/grazing信息 | [P §§5.3–5.4, Figs.14–16] | current encoder也存在query-feature shortcut，且aux target可由同一native reference定义 | base candidate vs same candidate+aux head；deployment导出时移除head；same main loss/work或明确matched training cost | source/query split、backbone/latent、optimizer、steps、seeds、main samples、runtime export | G1/G2/G2s、grazing/peak bins、seed variance、main-vs-aux tradeoff、export parity | training-only diagnostic | mainquality/seed stability无改善，或aux改善局部却显著损害整体/改变runtime |
| H2：adaptive+null latent可作为当前source family的capacity diagnostic | [P §§4.2–4.3, Figs.9–11] | source可定义不造假的semantic factors，fuzzy partition能稳定离散 | fixed latent widths vs overallocated adaptive/null；same backbone/query/training work | source semantics、split、loss、optimizer、seeds、runtime evaluator shape、selection rule | active dims、quality CI、seed stability、parameter bytes、hard-budget projection | capacity diagnostic, not product | active size跨seed/state不稳定、质量penalty无Pareto价值，或semantic batches不可合法构造 |
| H3：显式cheap physical cues + neural residual优于让latent吸收全部高频信息 | [P §§6–7; S Figs.4,7–8] | currentanalytic core/coordinates能提供类似reflection/path clue且成本有界 | neural-only vs cue+residual；matched total reads/MAC/time和same target | backbone capacity、source/query split、teacher、loss、steps、seeds、precision | local f error bins、G2/G2s、single-query time、reads/state bytes、CI | product candidate if bounded | cue variant在matched cost下无Pareto优势，或只因额外oracle work改善 |
| H4：按semantic factor构造batches可降低可编辑参数的latent cross-talk | [P §4.1, Figs.18–19] | native source exposes真正独立的editable axes，且batch averaging不破坏joint effects | ordinary iid batches vs one-factor batches；same queries/work/backbone | source family、parameter marginals、model/optimizer/steps、seeds、loss | parameter-swap consistency、joint-state G2、reciprocity/energy、quality CI | query recipe | factor batches降低joint generalization或不能减少cross-talk；source axes非正交使约束错误 |
| H5：scene-level method必须把offscreen visibility information source单独计费 | [P Fig.14, §§5.3,6.1; S Fig.8] |后续scene methods也可能用G-buffer shortcut并隐去extra ray/observation cost | visible-only G-buffer vs global/extra-ray variants；matched full pipeline time | scene/camera/light、network、resolution、SPP、rays、precision、hardware、seeds | image error/temporal error、prepare+render+inference breakdown、memory、visibility cases | scene transport evaluation contract | full-cost核算后额外information没有Pareto收益，或visible-only已覆盖目标domain |

H1–H4只能形成新candidate/diagnostic，不回写当前NVIDIA方法身份；H5属于scene transport评测合同。所有observed quality/time只作report-only结果，不成为研究任务hard gate。[N research execution]

## 16. 证据索引

### `P` Main paper

- Abstract、§§1,3、Figs.1–4：hybrid scene-image query、three observations、Pool encoder、G-buffer和formal training/data。
- §4、Eqs.1–3、Figs.5–11、Table1：static/adaptive/null partitions、soft boundaries、compression和quality penalty。
- §5、Figs.12–16、Table2：gradient×input attribution、U-net shortcut、auxiliary shadow generator和mixed result。
- §6、Figs.17–19：indirect-GI hybrid、400ms/7min/25min scope、lighting swap/interpolation。
- §§7–8、Figs.20–21：generalization、coarse compositionality、unknown materials、reflection/G-buffer/future bounds。
- 13/13页已按SHA source渲染并视觉核对公式、图、表、caption与footnote。

### `S` Supplemental

- §1/Table1/Figs.1–4：Pool+GQN/U-net/pixel parameter counts、逐层结构、训练时间、generator results和resolution failure。
- §2/Figs.5–6：G-buffer ablation、PrimitiveRoom/ArchViz recipes、HDR `log1p`。
- §3/Figs.7–8：indirect examples和roughness/reflection direction/hitpoint输入阶梯。
- 7/7页已按SHA source渲染核对；supplemental title `Shadow` typo已保留。

### `C` Official code/config

- commit `9c9033a1ca05095c7e2ccfeb4da3046b687bef3d`，source ZIP SHA-256 `452297EF03CB8E65525B0CF48E3FA485DD0A1E16FBA1143027741DE6AC2A1A3F`，MIT。
- `code/GQN/representation.py:8-181`：partition state、fuzzy/circular masks、forward averaging、null loss、gradient correction及`gradient_reg=0` early return。
- `:230-282`：Pool/Tower encoder exact layers、camera injection、residual paths和average pool。
- `code/GQN/generator.py:15-214`：GQN；`:270-444`：U-net；`:450-575`：Pixel generator与log losses。
- `code/GQN/training.py:13-65`：Adam1e-4、backprop/correction/step。
- `code/util/settings.py`：default partition/sigmoid/generator flags；`code/util/config.py`：JSON override与seed behavior。
- `code/util/datasets.py`、`create-dataset-json-files.py`、`generate-dataset.py`：factor batching、three observations/query split、9000/1000 batch defaults与float16 NPZ。
- `configs/*.json`：多数2M且Room GQN 20M vs formal1M、ArchViz128 vs formal64、seed=-1、Pixel/save `gradient_reg=0` vs GQN/U-net `1.0`、缺null/aux/GI configs与cached SPP语义边界。
- README：OptiX5.1.1、public data/checkpoints、1M checkpoint identity、example90/10不足与default low-spp warning。

### `A` Author material

- Jan Novák project page：formal P/S/C/video/talk入口、TOG39(4)/SIGGRAPH 2020身份。
- official GitHub project page：public README、license、data/checkpoint links。
- 未发现formal errata/correction；不能由未发现证明不存在未索引说明。

### `N` NeuralShading evidence

- `docs/contracts/scattering_backend.md`：local bare `f`、matched sample/pdf和static-bounded runtime；用于证明CNSR不是material program。
- `docs/research/experiment_framework.md`：matched controls、source/query冻结、CI和cost breakdown。
- `.trellis/spec/project/method-constraints.md`与research-execution规则：capacity diagnostic、product candidate和observed result边界。
- `ren-2024-lightformer.md`、`diolatzis-2022-active-exploration-neural-gi.md`：CNSR作为scene baseline的下游位置；不替代本报告一方事实。
- NVIDIA correspondence：当前functional reproduction identity，只用于§14。

### `I` Derived/transfer notes

- “capacity在observation/G-buffer/global latent/generator/classical renderer五处”是本报告分析。
- Pixel per-pixel network不是local material evaluator、400ms不是完整pipeline、lighting swap不是arbitrary light control、51 active dims不是通用GI容量结论，均由P/S/C与项目合同对照得出。
- formal/code conflicts均保持source status，不把static audit写成dynamic reproduction。

### Load-bearing 结论

CNSR是由LightFormer正式baseline与Active Exploration直接comparison触发的完整报告。其P/S/C已经足以解释后续方法为何转向active data、light-oriented features与更显式scene inputs；目前没有再触发必须提升的一般GQN、U-net、Deep Shading或attribution原始论文。若后续要把其中某一backbone/attribution算法实现为本项目candidate，再按`direct-inheritance`单独提升。

## Evidence review

```text
author_worker: /root
reviewer: /root/belcour2018_review
reviewed_at: 2026-08-29
sources_rechecked:
  - main PDF SHA-256 1EAA6AED7FCB7560814ECE83A744AD1DA1ED70EEFA7C91D5E42304E2828AEBBE, 13/13 pages independently rendered and visually checked
  - supplemental PDF SHA-256 C14948C07DB1E433BCA75895F179FDC93AA14DDDCFA52FEF6670524ADCCC2363, 7/7 pages independently rendered and visually checked
  - official code commit 9c9033a1ca05095c7e2ccfeb4da3046b687bef3d and all eight released configs independently statically audited
findings_closed:
  - scene query domain, three-observation aggregation and G-buffer contents
  - exact Pool/Pixel/U-net/GQN topology and parameter/training-time scope
  - static/adaptive/null partition mathematics and attribution/auxiliary lifecycle
  - datasets, factor batches, HDR loss, formal metrics and hybrid GI applications
  - formal negative results and scene/local transfer boundary
  - corrected the released gradient-reg split: Pixel/save inherit zero while GQN/U-net explicitly use 1.0
  - recorded the Room GQN 20M exception instead of treating every released config as 2M
  - separated cached 1/4/512-spp fields from the unrecoverable formal dataset render SPP
  - verified the supplemental Shadow title, R/(k+1), null-loss normalization and Primitive MAPE prose/figure conflicts
remaining_evidence_gaps:
  - official Drive datasets/checkpoints not downloaded or tensor/manifest audited
  - formal SPP, seed, full configs, application configs and runtime breakdown unreported
  - paper gradient correction strength versus the split released configs remains unresolved for the formal main run
  - paper 1M/64-square versus released 2M/20M/ArchViz128 config conflicts unresolved
  - formal null-loss run coefficient/config, R/(k+1) definition and supplemental all-metrics claim conflicts unresolved
  - code audit is static; legacy PyTorch/OptiX pipeline was not executed
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
