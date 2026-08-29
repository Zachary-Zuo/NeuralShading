---
paper_id: "zheng-2024-superposed-deformable-feature-fields"
title: "Neural Global Illumination via Superposed Deformable Feature Fields"
authors: "Chuankun Zheng; Yuchi Huo; Hongxiang Huang; Hongtao Sheng; Junrong Huang; Rui Tang; Hao Zhu; Rui Wang; Hujun Bao"
year: "2024"
venue: "SIGGRAPH Asia 2024 Conference Papers"
doi: "10.1145/3680528.3687680"
report_status: "evidence-reviewed"
main_source: "project-root/3680528.3687680.pdf"
supplemental_status: "unavailable"
official_code_status: "unavailable"
official_code_commit: "not-applicable"
author_worker: "/root/belcour2018_review"
reviewer: "/root/nelif_full_report"
last_verified: "2026-08-29"
---

# Neural Global Illumination via Superposed Deformable Feature Fields

> 本报告已完成 formal main 的完整 author pass 与独立 evidence review。主 PDF 共 11 页，已逐页读取并渲染核对；正文多次指向 supplemental 与 supplemental video，但本轮没有找到可匿名访问的一方 locator，因而 supplemental 中的网络尺寸、offset regularization、逐 scene field 配置和额外消融仍保留为证据缺口。下文先记论文事实 `[P]`，再写项目分析 `[N/I]`，不以题名或后续论文反填本文实现。

## 1. 研究对象与报告边界

这篇论文处理的是**每个 scene 单独训练的动态场景全局光照**：在 camera、lighting、material 与 object transformation 变化时，从当前 scene state 和首交点 G-buffer 预测整幅图像中的 global illumination，包括 caustics、indirect shadows、indirect highlights 和 soft shadows。[P: pp.1,3–4,6] 它不是局部材质函数，也不提供随机访问的 `evaluate(wo, wi)`、`sample()` 或 `pdf()`。

论文的核心分解不是按固定顺序串联对象 transfer function，而是两层可交换求和：[P: Eq.(3), Eq.(6), Fig.2]

1. 对目标对象 `i`，把各 source object/light `j` 指向 `i` 的 object-object latent `r_{ji}` 相加，得到 object-oriented scene representation `r_i`；
2. 每个对象的 deformable feature field 产生一个 feature output `F_i`，全部 `F_i` 再逐元素相加，由 final decoder 与 G-buffer 一起恢复 radiance。

这两个“加法”发生在不同语义层，不能合并理解成同一 tensor，也不能把 Fig.7–9 的 learned feature visualization 解释成物理上唯一的逐对象 radiance 分解。[P: pp.4–7; I]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | hash/commit | 本报告用途与边界 |
|---|---|---:|---|---|
| Formal main `P` | 项目根目录 `3680528.3687680.pdf`；DOI [10.1145/3680528.3687680](https://doi.org/10.1145/3680528.3687680)；11 页 | 2026-08-29 | SHA-256 `4D5B6E0BB79D274735A8DD5BB4980665396B135241ED2D8476C2BAD8D31F8E0F` | 主要方法、Eq.(1)–(9)、Fig.1–12、Table 1–2、正文结果与限制；11/11 页均完成文本读取和逐页渲染核对 |
| Supplemental `S` | 正文 pp.5–7 指向 supplementary material/video；DOI metadata、作者/团队入口和公开检索未得到匿名可访问的一方文件 | 2026-08-29 | unavailable | 不据此声称 supplemental 从未存在；其应承载但本轮不可核验的配置列为 explicit gaps |
| Official code/config/data `C` | 正式 PDF、Crossref/DBLP 与可得作者/团队入口未给 official repository 或 archive locator | 2026-08-29 | unavailable | 不猜代码 identity、默认配置、license 或 commit |
| Bibliographic metadata | [DBLP](https://dblp.org/rec/conf/siggrapha/ZhengHHSHTZ0B24)、Crossref DOI metadata | 2026-08-29 | not-applicable | 只核对作者、会议、DOI 与 `1–11` / article `24:1–24:11` 两种页码口径 |
| 本项目既有证据 `N` | [AE](./diolatzis-2022-active-exploration-neural-gi.md)、[NeLT](./zheng-2023-nelt.md)、[Dual-Band](./mo-2025-dual-band-neural-gi.md) evidence-reviewed 报告；[current NVIDIA correspondence](../implications/current-nvidia-correspondence.md) | 2026-08-29 | 当前共享工作树 | 只用于跨论文关系和项目 ABI 分析，不替代本文 P/S/C |

PDF 未加密、无 embedded attachment。公开来源审计没有使用账号登录、SSH/token、Git clone 或 Git credential；未找到的 supplemental/code 只记为本轮 `unavailable`，不外推为“不存在”。

## 3. 原论文的问题、假设与贡献边界

### 3.1 问题形式化

论文先把 rendering process 写成：[P: p.3, Eq.(1)]

\[
L(\mathbf{o},\boldsymbol\omega)=G(\mathbf{o},\boldsymbol\omega;\mathcal S,\mathcal L),
\]

其中 `S` 是 scene objects 集合，`L` 是 light sources 集合，`o, ω` 是 camera origin/direction。正文随后转到首交点表达 `L(x,ω_o)`，并在每个对象的 local coordinates 中使用 `x_i,ω_i`。[P: p.4, Eq.(2)–(3)] 论文目标是学习动态 scene 的 image-space radiance，而不是求一个材质散射量或单条 transport path 的 unbiased estimator。

### 3.2 对 NeLT 分解的论文内批评

正文用 Eq.(2) 概括 NeLT 为按人工顺序复合的 object transfer functions `T_0 ∘ T_1 ∘ … ∘ T_n`，并列出三个问题：[P: p.4]

\[
\begin{aligned}
L(\mathbf x,\boldsymbol\omega_o)={}&T_0(\mathbf x_0,\boldsymbol\omega_0;\varnothing,\mathcal L)\circ
T_1(\mathbf x_1,\boldsymbol\omega_1;\{s_0\},\mathcal L)\circ\cdots\\
&\circ T_n(\mathbf x_n,\boldsymbol\omega_n;\{s_0,\ldots,s_{n-1}\},\mathcal L).
\end{aligned}
\]

- **Unbalance**：越晚插入的对象条件集合越大、要承担的效果越多；且原 transfer function 只为 non-emissive objects 设计；
- **Isolation**：每个效果按指定顺序只分给一个对象，其他对象即使有贡献也被排除；
- **Simplicity**：hypernetwork 生成的 lightweight MLP 难以表达高频细节。

这是本文作者对 NeLT 的 later-paper 陈述，不等于独立 NeLT artifact 已逐项支持这三项批评；本任务已有 evidence-reviewed 的 [NeLT formal-main 报告](./zheng-2023-nelt.md)，这里只用它核对 method identity，并保留两篇论文各自的证据口径。[P↔N boundary]

### 3.3 论文声称的贡献

本文以 Eq.(3) 把自己的替代形式写为：[P: p.4, Eq.(3)]

\[
L(\mathbf x,\boldsymbol\omega_o)=
G\!\left(A\!\left(\{F_i(\mathbf x_i,\boldsymbol\omega_i;\mathbf r_i)\}\right)\right),
\]

其中实践中的 `A` 是 element-wise addition，式中为简洁省略了送入 final decoder 的 G-buffers。作者同时明确限定：主要贡献是经分析与组合技术形成的 robust/effective rendering framework，**不是**提出新的单个 technical module（例如 deformable feature field）。[P: p.2, p.4]

- 用 object-object representations 隐式编码 object/light 间交互，并把到达同一对象的 representations 相加为 object-oriented scene state；[P: §3.2, Eq.(4)–(6)]
- 每个 dynamic object/light 以及剩余 static objects 各有一个 deformable multi-resolution triplane feature field；field 由对应 `r_i` 条件化的 hypernetworks 生成 decoder 参数；[P: §3.3, §4.2, Fig.4]
- 所有 field outputs 以 element-wise addition 做 order-invariant aggregation，再由 final decoder 预测全局光照；emissive 与 non-emissive object 采用同一建模方式；[P: p.4, Eq.(3)]
- 在三个 512×512 动态 scenes 上相对 AE、NeLT 与 matched-time OIDN 报告质量/时间，并消融 feature field、coarse-to-fine 与 deformation。[P: §4–5, Table 1–2, Fig.5,10–12]

## 4. 输入、输出、坐标与 query domain

| 项 | 正式定义 | shape/domain 与未报告项 | locator |
|---|---|---|---|
| Scene state | scene objects `S`、lights `L`；每个 object 的 local-to-world matrix `M_i`；object `i` 的 variable material parameters `v_i` | `v_i` 的具体参数、尺度、编码和各 scene range 在 main 未给 | P: Eq.(1), Eq.(4)–(5) |
| Object-pair condition | `M_{ij}=M_i × M_j^{-1}`，送入 source object `i` 的 `R_i` | 正文称其为从 `i` local 到 `j` local；matrix convention、向量乘法侧与 Eq.(4) 的实际实现未报告，故仅保留印刷公式 | P: p.5, Eq.(4) |
| Pixel query | 首交点在 object `i` local space 的 `x_i`、view direction `ω_i`，加上 local G-buffer `g_i` | G-buffer 明确包含 position、normal、view direction；完整 channel list、background/miss path、normal transform 规则在 main 未给 | P: Eq.(7), pp.5–6 |
| Field condition | object-oriented scene representation `r_i=Σ_j r_{ji}` | latent width与normalization未报告；prose 写“other objects”，Fig.2 又画出 `r_AA,r_BB,r_CC` self terms，见 §11 | P: Eq.(5)–(6), Fig.2 |
| Field coordinate | `x_i + T_i(x_i;r_i)_p` 采样 object-local triplanes | `_p` 是 offset decoder 前 3 channels；offset scale/bounds/regularizer 在 main 未给 | P: Eq.(7), Fig.4 |
| Auxiliary condition | `T_i(x_i;r_i)_h` | `_h` 是其余 channels；维数未给；windowed positional encoding 也用于该 auxiliary feature | P: Eq.(7)–(8), pp.5–6 |
| Per-object output | `F_i(x_i,ω_i;r_i)`，即 object decoder 的 learned feature | feature width/物理单位未报告；不是被证明的独立 radiance contribution | P: Eq.(3), Eq.(7), §5.3 |
| Final output | `L(x,ω_o)`，由 summed features 与 G-buffer 经 final decoder `G` 得到 | colour space、是否只预测间接项、log-domain inverse 和 output activation未报告；Eq.(3) 为简洁省略了 G-buffer | P: Eq.(3)及其后 prose, Fig.2 |

query 依赖一份完整 scene rasterization/G-buffer 与 scene state，因而属于 whole-scene renderer 的 pixel query。论文没有定义 hemisphere measure、BSDF cosine convention、direction sampler 或 temporal state。[P]

## 5. Representation、逐层网络与数据流

### 5.1 端到端数据流

按 Fig.2、Fig.4 和 Eq.(3)–(8)，可以恢复的正式 pipeline 是：[P: pp.4–6]

1. 对 scene 中 object/light pairs 计算相对 transform `M_{ij}`；source object `i` 的 object-to-all encoder `R_i` 接收 `M_{ij},v_i`，产生 `r_{ij}`；
2. 对目标 object `i`，将所有指向它的 `r_{ji}` 相加为 `r_i`；
3. 把当前 pixel 的 G-buffer position/normal/view direction 变换到 object `i` local space；
4. `r_i` 条件化 offset hypernetwork 和 object hypernetwork，分别生成 offset decoder `T_i` 与 object decoder `D_i`；
5. `T_i(x_i;r_i)` 的前 3 channels 给出位置 offset，剩余 channels 给 auxiliary feature；在 shifted local position 上对 multi-resolution triplanes 双线性采样；
6. `D_i` 接收 sampled triplane features、auxiliary feature、local G-buffer，并受 `r_i` 条件化，输出 `F_i`；
7. 对所有 object/light fields 的 `F_i` 做 element-wise sum；summed feature 与 G-buffer 进入 final decoder，得到 image radiance。

Fig.4 把 hypernetwork、offset hypernetwork、triplane、object decoder、final decoder 分开画出。main 只给模块关系，明确把“all neural modules”的细节放到 supplemental；因此不能从示意图推导 MLP 层数、hidden width 或 feature channels。[P: p.5]

### 5.2 Object-object 与 object-oriented scene representations

正式式子为：[P: p.5, Eq.(4)–(6)]

\[
M_{ij}=M_i M_j^{-1},\qquad
\mathbf r_{ij}=R_i(M_{ij},\mathbf v_i),\qquad
\mathbf r_i=\sum_j \mathbf r_{ji}.
\]

`R_i` 是每个 object 的 object-to-all encoder：同一个 source `i` 针对不同 target `j` 使用 `M_{ij}`，输出其对 `j` 的 latent impact。之后以 target 为中心求和。该设计令 aggregation 对输入 object 顺序不敏感，但不意味着不同 scene 可共享 encoder，也不证明 latent 有物理可加性；论文只用端到端训练学习这套加法表征。[P/I]

### 5.3 Deformable feature field

正文 Eq.(7) 为：[P: p.5]

\[
F_i(\mathbf x_i,\boldsymbol\omega_i;\mathbf r_i)=
D_i\!\left(
\mathcal F_i(\mathbf x_i+T_i(\mathbf x_i;\mathbf r_i)_p),
T_i(\mathbf x_i;\mathbf r_i)_h,
\mathbf g_i;\mathbf r_i
\right).
\]

`T_i` 同时输出 3D positional offset 与 auxiliary feature。作者的动机是：动态 light/object 令 shadow 等效果发生位移和形变，static grid 必须在许多位置分别记住这种变化；deformable grid 可在 canonical space 学 template，再通过 offset 变形。[P: Fig.3 caption] 这是设计动机和实验支持，不是对任意 transport 的普遍定理。

### 5.4 Multi-resolution triplane 与 coarse-to-fine

每个 level `l∈[0,m−1]` 有三张可优化 planes `P^{XY}_{i,l},P^{XZ}_{i,l},P^{YZ}_{i,l}`，分别在 `(x_0,x_1)`、`(x_0,x_2)`、`(x_1,x_2)` 双线性采样；所有 levels/planes 的加权结果拼接为 `𝓕_i(x)`。[P: Eq.(8), Fig.4]

\[
\mathcal F_i(\mathbf x)=\big[
w_0(\alpha)P^{XY}_{i,0}(x_0,x_1),
w_0(\alpha)P^{XZ}_{i,0}(x_0,x_2),
w_0(\alpha)P^{YZ}_{i,0}(x_1,x_2),\ldots,
w_{m-1}(\alpha)P^{XY}_{i,m-1}(x_0,x_1),
w_{m-1}(\alpha)P^{XZ}_{i,m-1}(x_0,x_2),
w_{m-1}(\alpha)P^{YZ}_{i,m-1}(x_1,x_2)
\big],
\]

\[
w_j(\alpha)=\frac{1-\cos\left(\pi\,\mathrm{clamp}(\alpha-j,0,1)\right)}{2}.
\]

训练早期只启用低 resolution planes；`α` 随训练线性增加，依次解锁更高 resolution planes。相同的 windowed positional encoding 也应用于 offset decoder 的 auxiliary features。[P: pp.5–6] main 没有报告 `m`、各 level resolution/channel、`α` 起止值、增长时长或 bilinear boundary behavior。

### 5.5 有证据的网络/张量配置

| 模块 | 输入 | 正式运算 | 输出 | shared/per-object | main 中未报告 |
|---|---|---|---|---|---|
| `R_i` object-to-all encoder | `M_{ij},v_i` | neural encoder | `r_{ij}` | per source object `i` | matrix flattening、layers、width、activation、latent dim |
| Object aggregation | `{r_{ji}}_j` | element-wise sum | `r_i` | per target object | self term/default/null representation、normalization |
| Offset hypernetwork | `r_i` | 生成 `T_i` 的参数 | offset decoder parameters | per field, scene-state conditioned | hypernetwork topology、generated parameter subset |
| `T_i` offset decoder | `x_i`，由 `r_i` 条件化 | MLP/decoder（main 不给层数） | 前 3 channels `_p`；其余 `_h` | per field | hidden width、activation、aux dim、offset regularization |
| `𝓕_i` triplane | `x_i+offset` | `m` levels × 3 planes，bilinear sampling，window weights | concatenated sampled features | per field | levels、resolutions、channels、storage dtype |
| Object hypernetwork + `D_i` | `r_i` 生成 decoder；decoder 接收 triplane feature、auxiliary feature、`g_i` | neural decoder | `F_i` | per field | topology、output dim、activation |
| Field aggregation | `{F_i}` | element-wise addition | superposed latent | scene-level | scaling/normalization |
| Final decoder `G` | superposed latent + G-buffer | neural decoder | radiance prediction | per scene | topology、input/output transform、output activation |

所有更具体的 shape 都没有 main-paper 证据；正文明确将 module details、offset regularization、local G-buffer computation 放到 supplemental。[P: pp.5–6]

### 5.6 Field 数量不是固定 architecture 常数

训练规则是：每个 dynamic object/light 一个 field，再为 remaining static objects 建一个额外 field。[P: §4.2] 论文实际 visualizations 的 grouping 为：[P: §5.3, Fig.7–9]

- Ajar：metallic sphere field + remaining/static field，共 2 个；
- Watercolor：moving shelf field + remaining/static field，共 2 个；
- Hall：两个 top spotlights 分别一个 field、围绕 statue 的 light set 一个 field、remaining static objects 一个 field，共 4 个。

因此 “object” 可是单个几何对象、单盏 light、成组 lights 或剩余 static composite，partition 是 scene-specific；main 没给自动 partition 算法。

## 6. 数据、GT/reference 与 query/sampling recipe

### 6.1 三个 per-scene datasets

| Scene | 动态内容 | 正文给出的 field grouping | locator |
|---|---|---|---|
| Ajar | 基于 Bitterli Ajar；桌面装饰，irregular metallic sphere 由两盏小灯照亮并产生动态 caustics；camera、sphere rotation/roughness、main light intensity 可变 | sphere + remaining，2 fields | P: §4.1, §5.3, Fig.6–7 |
| Watercolor | painting studio；environment lighting 可旋转、shelf 移动、camera 动态 | shelf + remaining，2 fields | P: §4.1, §5.3, Fig.6,8 |
| Hall | 中央 statue；可旋转彩色 light set；两盏旋转 spotlights 在地面投射 SIGGRAPH logo；camera 和 outside-light intensity 可变 | 两盏 top spotlights各自 + surrounding light set + remaining static，4 fields | P: §4.1, §5.3, Fig.6,9 |

作者对每个 scene 的全部 dynamic components 做 uniform randomization，生成 6,000–8,000 个随机 scene states 训练、100 个随机 states 测试。[P: §4.1] main 没给 validation split、每个参数的 range、random seed、不同 scene 的确切训练数、camera sampling distribution 或 train/test state 去重机制。

### 6.2 GT 与 G-buffer

- Falcor path tracing，512×512，4,096 spp，生成 ground-truth images 与对应 G-buffers；[P: §4.1]
- 训练图像再经 Yu et al. 2021b 的 offline denoiser 抑制 residual noise；[P: §4.1]
- 正文没有说明 test GT 是否也经过该 denoiser、path length/RR/light sampling、颜色空间、tone mapping、G-buffer 完整 channel list、每张图如何构成 minibatch item，也没有报告 Falcor version/commit。[P gap]

这里的 query unit 从训练描述看是 scene observation/image；minibatch 16 的含义更可能是图像/观察 batch，但 main 没有明确 pixel crop、全帧或像素采样细节，故不进一步补写。[P gap]

## 7. Loss、optimizer 与训练 lifecycle

正文 Eq.(9) 写为：[P: §4.2]

\[
\mathcal L=\mathcal L_1(L,\hat L)+\mathcal L_{SSIM}(L,\hat L),
\]

并把第二项称作 structural dissimilarity loss。HDR 值采用 `log1p` 变换 `\tilde x=log(1+x)` 以稳定训练。[P: §4.2]

| 项 | 正式配置 | 证据边界 |
|---|---|---|
| Target/prediction transform | HDR 值用 `log1p` | main 未说明变换在两项 loss 前后怎样放置、推理 inverse、负值处理 |
| Loss | `L1 + L_SSIM`，式中系数均印为 1 | reduction、SSIM window/implementation、channel aggregation未给 |
| Optimizer | Adam | `β1/β2/ε`、weight decay 未给 |
| Learning rate | `10^-4` | schedule/warmup 未给 |
| Minibatch | 16 | item/query composition 未给 |
| Training duration | 每个 scene 约 15 h，4× NVIDIA RTX A6000 | steps、epochs、images/step、DDP策略未给 |
| Lifecycle | end-to-end per-scene training；coarse-to-fine 中 `α` 线性增加 | `α` schedule、是否分 stage、冻结/解冻顺序未给 |
| Initialization/reproducibility | 未报告 | seed、init、checkpoint selection、repeat count/variance均未给 |

feature planes、object encoders、hypernetworks、decoders 与 final decoder 是否使用同一 optimizer group、各自 LR 或 regularization，main 没有说明。offset regularization 被明确指向 unavailable supplemental，不能猜其形式或权重。[P↔S gap]

## 8. Inference、部署与成本

### 8.1 正式运行时路径

论文的实时对象是固定 scene 的 renderer。scene state 变化后，object-pair encoders 和 hypernetworks产生当前 object-oriented conditions/decoder；每个 pixel 使用 raster G-buffer 变换到各 object local frames，query各 field，再 sum-decode整图。[P: Fig.2,4] 正文没有把 stage timing 拆开，也没有说明 object encoders/hypernetwork 输出是逐 scene state、逐 frame还是逐 pixel重算。

### 8.2 报告成本口径

Table 1 的时间均在 NVIDIA RTX 4090 测量；AE、NeLT 与本文均是 PyTorch FP32。[P: Table 1 caption] 本文三个 scenes 分别为 18.9 ms、19.0 ms、26.6 ms，但 main 未说明是否包括 G-buffer generation、scene-state encoding、hypernetwork、全部 fields、final decoder、framework synchronization或 display。故这些数值只能视作论文的 end-to-end-ish renderer timing row，不能拆成单次 field query 成本。[P gap]

OIDN 行写成 `- / total (spp)`：caption 说明 total 包含 path tracing 与约 1–2 ms denoising；作者通过增加 spp 把 OIDN 完整时间对齐本文。表中没有把 path-tracing 时间单列，因此 dash 不能解释为零成本。[P: Table 1]

### 8.3 未报告的部署信息

parameter count、hypernetwork/generated-weight bytes、triplane resolutions/channels、per-scene/per-object memory、MAC/FLOP、texture fetch count、FP16/INT8、export format、kernel fusion、temporal reuse 和 multi-GPU inference 都未报告。方法对固定 scene/object partition 的执行图是有限的，但论文没有给跨 scenes 通用 `N_max` 或静态 read cap；scene object 数增加时总成本显著增加正是作者限制。[P: §6]

## 9. 实验 protocol、baseline、指标与结果

### 9.1 Baseline identity 与训练口径

- **AE / Active Exploration**：使用同一 dataset，equal or longer training；为了与本文对齐和训练效率，作者使用 **uniform data sampling**，没有启用 AE 的 adaptive exploration policy。作者明确把 adaptive training 视作可能收益与未来工作。[P: §5.1]
- **NeLT**：使用同一 dataset，另生成计算其 target functions 所需的数据；所有 static objects 合为一个 composite，每个 non-emissive object 分开训练，且每个对象的训练时间不短于本文完整模型训练时间。[P: §5.1] 因此它不是 total data/compute matched control。
- **OIDN**：使用“latest official implementation”，但无 version/commit；通过调 spp 使 path tracing + denoising total time 接近本文。[P: §5.1, Table 1 caption]

三个 learned baselines 的实际 paper/code implementations 是否逐项相同、AE/NeLT 网络尺寸是否调参，main 未报告。结果没有置信区间、重复训练或 per-frame distribution。[P gap]

### 9.2 Table 1 数值

下表逐项抄录正式 Table 1；L1/SSIM/LPIPS 为 100 个 test states 上怎样聚合，正文没有进一步说明。[P: p.8, Table 1]

| Scene | Method | Time (ms) ↓ | L1 ↓ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|
| Ajar | AE | 66.3 | 0.0042 | 0.9856 | 0.0151 |
| Ajar | NeLT | 56.7 | 0.0050 | 0.9790 | 0.0211 |
| Ajar | OIDN | `- / 19.0 (50 spp)` | 0.0026 | 0.9929 | 0.0091 |
| Ajar | Ours | 18.9 | 0.0023 | 0.9941 | 0.0058 |
| Watercolor | AE | 67.1 | 0.0090 | 0.9797 | 0.0159 |
| Watercolor | NeLT | 56.9 | 0.0136 | 0.9573 | 0.0363 |
| Watercolor | OIDN | `- / 20.1 (17 spp)` | 0.0112 | 0.9652 | 0.0280 |
| Watercolor | Ours | 19.0 | 0.0079 | 0.9812 | 0.0176 |
| Hall | AE | 66.5 | 0.0154 | 0.9385 | 0.0807 |
| Hall | NeLT | 28.1 | 0.0168 | 0.9201 | 0.0926 |
| Hall | OIDN | `- / 26.7 (31 spp)` | 0.0244 | 0.8981 | 0.0925 |
| Hall | Ours | 26.6 | 0.0134 | 0.9449 | 0.0773 |

作者据此称本文在大多数 cases 优于 baselines。严格看表，本文在三个 scenes 的三项质量指标中均是最佳值，同时 learned runtime 更低；但 baseline 的训练/target-data工作量不完全 matched，OIDN 又是 stochastic rendering + denoising 的不同方法类，因此不能把该表读成 architecture-only 因果比较。[P/I]

### 9.3 Qualitative 结果

Fig.5 的作者解读是：AE 难以预测 Ajar caustics 与 Watercolor shelf shadows；NeLT 能产生 plausible dynamic caustics/shadows，但对由细几何和多材质组成的 Watercolor shelf 存在 shading bias，Hall 中混合 light representation 也限制复杂 lighting；OIDN 倾向过度平滑。[P: §5.2, Fig.5 caption] 正文另声称 OIDN 有 temporal flickering，并要求观看 supplemental video；video 不可得，所以本轮只能记录 author claim，不能视觉独立核验。[P↔S gap]

Fig.7–9 显示 learned `F_i` 的可视化：sphere field 对 Ajar caustics/部分 indirect highlights 响应，shelf field 对 Watercolor shelf shadow 提供强线索，Hall spotlights fields 捕获 projection patterns。[P: §5.3] 同时其他 fields 也会在相同对象上提供 auxiliary information；这恰好说明 attribution 不是互斥物理分解。

## 10. 消融、失败尝试与负结果

### 10.1 Feature field 与 coarse-to-fine

Watercolor 上保持“same settings”，比较 full、去掉 feature fields、去掉 coarse-to-fine。[P: §5.4] Table 2 是 dataset-level table，Fig.10 caption 是展示帧的单图数值，两种口径不可混用：

| Variant | Table 2 L1 ↓ | SSIM ↑ | LPIPS ↓ | Fig.10 shown-frame L1 ↓ |
|---|---:|---:|---:|---:|
| Ours | 0.0079 | 0.9812 | 0.0176 | 0.0101 |
| w/o Feature Fields | 0.0125 | 0.9635 | 0.0311 | 0.0144 |
| w/o Coarse-to-Fine | 0.0099 | 0.9755 | 0.0226 | 0.0127 |

这是 `ablation-inferior`：两项都优于相应删除版，但 main 没给训练 curves/repeats，不能据此分离 optimization speed、final capacity 与 regularization 贡献。[P/I]

### 10.2 Deformable vs static field

作者构造一个 square room、若干 static objects、dynamic area light 的简单场景，并在同设置下训练 deformable 与 static field；Fig.11 shown-frame L1 为 deformable `0.0073`、static `0.0091`。正文称 static variant 即使显著延长训练仍不能恢复动态 shadows。[P: §5.4, Fig.11] 但“significantly longer”的时长、停止准则与完整 metric 未给，故属于有方向的 author-negative，而非可精确重现实验。

### 10.3 作者报告的失败案例

Fig.12 显示 training data 中罕见出现的 high-frequency effects 会漏失；作者把原因归于 coverage不足，并建议 adaptive training 与更多 shading clues。[P: §6, Fig.12] 这是明确的 `known-limitation`，不是本报告推测。

### 10.4 不可得的额外消融

正文称 supplemental 还包含：不同 feature-field 数量，以及不同 final-decoder sizes 的 pros/cons。[P: §5.4] 由于 supplemental 无可得 locator，本报告不补写 variant 配置、结果或结论。

## 11. Paper ↔ supplemental ↔ code correspondence与相关方法边界

### 11.1 跨工件对应

| 主题 | Formal main `P` | Supplemental `S` | Official code/config `C` | 当前结论 |
|---|---|---|---|---|
| Neural topology | 模块图与 Eq.(3)–(8)；无层数/width | 正文称“all neural modules”在 S | 无 locator | pipeline闭合，逐层配置未闭合 |
| Offset/local G-buffer | offset前3 channels、aux其余channels；world-to-local position/normal/view | 正文称 regularization 和 local G-buffer detail 在 S | 无 locator | 正则、transform实现未闭合 |
| Scene/field config | main 给三 scene 语义与可视化中的 2/2/4 fields | 正文称 specific settings/dynamics 在 S | 无 locator | field grouping可恢复，尺寸/range不可恢复 |
| Field count/final decoder ablation | 只说明做过 | 结果指向 S | 无 locator | 不写数值/排序 |
| Temporal OIDN comparison | 正文声称 flicker | video指向 S | 无 locator | author claim，未视觉核验 |
| Runtime | Table 1: RTX4090、PyTorch FP32 | 未取得 | 无 locator | 无参数/显存/stage timing/code identity |

### 11.2 正文内部语义 gap

- Eq.(6) 前 prose 写“sum … from other objects to object `i`”，但 Fig.2 明确画出 `r_AA,r_BB,r_CC` 并加入各自 `r_i`。main 没有解释 self term 是否恒等、material-only 或实际省略；本报告保留冲突，不自行删除 self terms。[P↔P]
- Eq.(4) 印刷为 `M_{ij}=M_i×M_j^{-1}` 并称从 `i` local 到 `j` local。matrix/vector convention 未给，不能擅自把顺序改成通常写法。[P gap]
- Eq.(3) 只写 `G(A({F_i}))`，正文明确说为了简洁省略 G-buffers；Fig.2 则把 G-buffers 送到 final decoder。因此复现不能把 Eq.(3) 当作 final decoder 只吃 summed fields。[P↔P clarification]

### 11.3 与 AE 的精确关系

本文把 NeLT-style object-oriented representation 施加到 AE-style explicit scene representation，并将 AE network 作为 baseline；但 baseline 使用 uniform sampling，主动禁用了 AE 的 adaptive selector。[P: §3.2, §5.1；N: [AE §5–7](./diolatzis-2022-active-exploration-neural-gi.md)] 因而 Table 1 不回答“本文 representation vs AE representation 且双方各自最佳 sampling policy”的问题，也不能把本文训练 recipe称为 Active Exploration。

### 11.4 与 NeLT 的精确关系

NeLT 是正文直接 predecessor：两者都以 object-local coordinates、object-oriented representation 和 hypernetwork 分摊 scene-related 与 pixel-related computation；本文以 pairwise latent sum 消除 ordered composition，把 lightweight transfer MLP 替换为 deformable triplane-backed fields，并进一步统一建模 emissive/non-emissive objects/lights。[P: §3.1–3.3；N: [NeLT §5.1–5.3](./zheng-2023-nelt.md)] 本报告中对 NeLT 三项缺陷的措辞仍只归属于本文作者的 later-paper characterization；独立 NeLT 报告用于核对 method identity，不把另一篇论文的配置或判断反填进本文事实层。

### 11.5 与 Dual-Band 的时间方向

Dual-Band 是 2025 年后续 scene-GI 工作；其报告把本文 backbone 称为 simplified FieldGI，并明确移除 deformation，再加入 principal/secondary/fusion 结构。[N: [Dual-Band §11](./mo-2025-dual-band-neural-gi.md)] 该关系说明本文是后续选择的起点之一，但**不能**把 Dual-Band 的 8-level triplane、head topology、GT/query、two-stage lifecycle 或 runtime配置反填到 2024 本文。[N/I]

### 11.6 与其他正文 baseline

OIDN 是 matched-total-time path tracing + denoising control，不是 learned scene representation；AE/NeLT 是 learned scene baselines。论文 related work 还讨论 CNSR/Neural Scene Graph/PRT/feature grids，但它们没有进入 Table 1，不能称为正式 quantitative baselines。[P: §2, §5]

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

- 动态对象越多，总成本仍显著增加，限制 many-dynamic-object scenes；[P: §6]
- large-scale complex scene 若要保持细节，需要极高 resolution triplanes，memory dramatic growth；作者提出 hash table 或 virtual texture 为可能方向，不是已验证方案；[P: §6]
- 依赖足够 training coverage，罕见 high-frequency effects 会漏失；adaptive training/更多 shading clues 是未来方向；[P: §6, Fig.12]
- element-wise addition 的更强替代（如 attention）被留作 future work。[P: p.4]

### 12.2 关键未报告项

- S：网络逐层结构、width/channel、offset regularization、local G-buffer计算、每 scene具体 dynamics/field settings、field-count/final-decoder-size ablations、video；
- C：official repository/archive、code commit、license、configs、assets、Falcor scene/data generator；
- Training：steps/epochs、LR schedule、`α` schedule、seed/init/model selection、loss implementation和log1p lifecycle；
- Runtime：parameters/MAC/bytes/fetch、triplane memory、stage timing、precompute/amortization、G-buffer成本、FP16/export；
- Evaluation：test aggregation、repeat/variance/CI、test denoising状态、temporal metric、OIDN version。

这些缺口不会阻止恢复 formal main 的方法逻辑，但会阻止 exact reproduction 与 shader-budget判断。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

本文把容量分在三处：per-field multi-resolution triplane 承担高频 spatial detail；hypernetwork-generated offset/object decoders把 scene state 转成 deformation和pixel-conditioned解码；final decoder整合跨字段 summed latent 与G-buffer。object-pair encoders承担scene交互条件，但其latent本身没有被证明能独立还原transport。[P→I]

这与单个 monolithic whole-buffer network 的关键差别，是显式把“随 object motion 移动的空间细节”绑定到 object-local persistent field。代价则是 field数量和triplane空间分辨率直接进入runtime/memory，正好对应作者两个主要限制。[P→I]

### 13.2 成功依赖的假设

- scene可预先分成少量、语义稳定的 dynamic object/light groups 与一个 static remainder；
- object-local deformation能把足够多的动态shadow/caustic结构对齐到可复用template；
- pairwise latent的简单求和在训练分布中足以保留交互信息；
- per-scene uniform random states能覆盖测试时罕见高频事件。

前三项没有被论文完整独立消融；第四项被 Fig.12 明确反例削弱。因此“superposition”应视作经验有效的learned aggregator，而不是transport的精确线性分解。[P/I]

### 13.3 对 local neural material 的可迁移与不可迁移部分

可能可迁移的是**训练组织机制**：coarse-to-fine逐步开放spatial capacity、将可复用状态放入`prepare()`、用matched static/deformable control检查是否真的需要coordinate warp。[I] 不可直接迁移的是scene object graph、world-to-local多对象G-buffer、逐对象triplanes和whole-image final decoder；它们改变了本项目local `evaluate(wo,wi)` 的query identity与memory/read contract。

### 13.4 结果解释边界

Table 1 的成功同时改变 representation、per-object partition、training work和baseline policy；Fig.10/11 支持本文内部组件有益，但没有给variance或memory-normalized control。更稳妥的结论是：在三项固定scene实验与作者配置下，deformable fields和C2F variants均优于对应删除版；不能据此给任意材质/scene representation排quality名次。[P/I]

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

当前 functional-f@2 backend 是local、随机访问、运行成本静态有界的material evaluator；本文是per-scene、whole-buffer GI network。它不能直接进入当前 `ScatteringState` 或替代 `evaluate(wo,wi)`，也没有`sample/pdf` identity。[N: [current NVIDIA correspondence §1, §5](../implications/current-nvidia-correspondence.md)]

可记录为candidate的只有两个机制层启发：[I]

1. 在未来存在spatial/material footprint状态时，比较static persistent feature与conditioned coordinate deformation；
2. 把对同一shading point可复用的condition-to-weight/latent工作放到`prepare()`，明确量化amortization，而不是把scene-level hypernetwork成本藏进单次query。

这两项都是迁移假设，不是当前NV实现的已知defect，也未授权成为当前kill test。本文没有参数/bytes/fetch数据，故尚不能声称满足shader budget。

## 15. 可证伪的迁移假设 `[I]`

以下只登记未来matched experiments，非已授权run或hard gate。

### H-SDF1：object/local state superposition 是否优于同预算 monolithic scene latent

- **适用域**：后续 scene-level transport track，不进入local material evaluator；
- **Matched control**：相同GT/query states、training states、loss、optimizer、steps、total parameters、persistent bytes和measured runtime class；A为一个monolithic scene latent/decoder，B为固定partition的per-object latent sum；
- **Frozen axes**：G-buffer、field/decoder最大宽度、precision、resolution、scene split、uniform sampling；不同时启用adaptive exploration；
- **Metrics**：linear-HDR L1/relative error、SSIM/LPIPS、rare-event分层error、per-frame latency、peak/persistent bytes；bootstrap CI；
- **Falsification**：B在matched cost下未改善held-out object-transform states，或收益只来自更高bytes/runtime，则否定“superposition本身带来收益”。

### H-SDF2：coarse-to-fine spatial capacity 是否改善优化稳健性

- **Matched control**：同一feature planes、decoder、training queries与总steps；A从step 0全开所有levels，B仅用Eq.(8) window逐级开放；
- **Frozen axes**：initialization seeds、optimizer/LR、loss、query order、final active capacity与inference runtime；
- **Metrics**：多seed final/anytime validation error、rare high-frequency subset、gradient/optimization variance、最终latency/bytes相等性；
- **Falsification**：B的多seed CI不优于A，或最终收益来自减少有效训练而非稳定性，则不采用该schedule。

### H-SDF3：conditioned deformation是否值得`prepare()`与额外state成本

- **适用域**：仅在未来有spatial coordinate/footprint的material family；本文结果不能支持纯angular BRDF直接套用；
- **Matched control**：A static grid，B同grid加3D bounded offset；等总parameter/persistent bytes，并分别报告`prepare()`、单次`evaluate()`和多query amortized cost；
- **Frozen axes**：source/reference、material splits、query recipe、loss、sampling、decoder、precision、fetch cap；
- **Metrics**：G1/G2/G2s、filtered/spatial detail error、W稳健性、single-query及packet latency、bytes/fetch；
- **Falsification**：在matched runtime/bytes下B不能改善未见parameter states，或offset破坏过滤/随机访问稳定性，则deformation不进入candidate。

## 16. 证据索引

- `P-main-method`：formal PDF pp.3–6，§3，Eq.(1)–(8)，Fig.2–4；问题、pair representations、两层sum、hypernetwork/deformable triplane/C2F。
- `P-main-data-training`：pp.6–7，§4，Eq.(9)，Fig.6；三 scenes、uniform dynamic-state generation、6k–8k/100、512²/4096 spp/Falcor、offline denoiser、Adam配置与15 h。
- `P-main-results`：pp.7–8与pp.10–11，§5、Table 1–2、Fig.5,7–12；baseline口径、数值、feature visualizations、消融与failure。
- `P-main-limitations`：pp.7–8，§6；object-count cost、large-scene triplane memory、rare-event coverage、未来方向。
- `P-meta`：[DOI](https://doi.org/10.1145/3680528.3687680) 与 [DBLP](https://dblp.org/rec/conf/siggrapha/ZhengHHSHTZ0B24)；仅书目与页码。
- `S-gap`：formal main 对 supplementary material/video 的显式引用；本轮未取得可匿名访问的一方artifact。
- `C-gap`：formal main/metadata/author-team入口未给official code/config/data locator。
- `N`：本任务 evidence-reviewed [AE](./diolatzis-2022-active-exploration-neural-gi.md)、[NeLT](./zheng-2023-nelt.md)、[Dual-Band](./mo-2025-dual-band-neural-gi.md) 报告，以及 [current NVIDIA correspondence](../implications/current-nvidia-correspondence.md)；仅用于§11、§14。
- `I`：§13–§15，均晚于事实层。

## Evidence review

```text
author_worker: /root/belcour2018_review
reviewer: /root/nelif_full_report
reviewed_at: 2026-08-29
sources_rechecked: [formal main PDF SHA-256 4D5B6E0BB79D274735A8DD5BB4980665396B135241ED2D8476C2BAD8D31F8E0F, PDF text extraction, 11/11 full-page visual renders including equations/footnote/figures/tables/captions, DOI/ACM availability, DBLP record, public first-party supplemental/code locator search, evidence-reviewed AE/NeLT/Dual-Band reports, current NVIDIA correspondence]
findings_closed: [F1 restored explicit Eq.2/Eq.3/Eq.8 forms and confirmed Eq.1-9, F2 recorded the p.2 author boundary that novelty is the rendering framework rather than a new standalone module, F3 updated stale NeLT blocked-source wording after its independent formal-main review and bounded later-paper characterization, F4 added reproducible N locators for AE/NeLT/Dual-Band/current-NVIDIA claims, F5 confirmed Table1-2 and Fig.10-12 values plus aggregate-vs-shown-frame distinction, F6 confirmed two distinct superposition levels and preserved Eq.4 convention/Eq.6 self-term gaps, F7 confirmed unreported topology/training/runtime fields remain unknown]
remaining_evidence_gaps: [supplemental and video unavailable, official code/config/data unavailable, module layer widths and tensor dimensions, offset regularization, exact local-G-buffer transform, per-scene field resolutions/configs, feature-count/final-decoder ablation results, alpha schedule, steps/seeds/model selection, parameter-memory-MAC-fetch-stage timing, test aggregation and temporal metrics]
review_status: evidence-reviewed
```

### 完成检查

- [x] main paper 11/11 页已完整阅读，关键公式/图/表/图注已逐页渲染核对；
- [x] supplemental/appendix/video 的可用性已检查；不可得项保留 explicit gaps；
- [x] official code/config/data入口已检查；未发现可审计locator，未猜commit/license；
- [x] architecture、training、runtime和主要结果均给出page/section/equation/figure/table locator；
- [x] 失败尝试、较差消融、single-frame与dataset-level数值已分类且不混口径；
- [x] paper↔supplemental↔code gaps与正文内部符号/语义gap已保留；
- [x] `P/S/C/A`事实与`N/I`分析已分离，分析晚于事实层；
- [x] AE、NeLT、Dual-Band关系没有反填对方配置；
- [x] NVIDIA影响只写candidate/identity边界，不把scene renderer放入当前ABI；
- [x] 迁移假设具有matched control、frozen axes、metrics、runtime class和falsification，且未写成已授权run；
- [x] 独立 evidence review 已由 `/root/nelif_full_report` 完成；所有 finding 已关闭，报告状态为 `evidence-reviewed`。
