---
paper_id: "zheng-2023-nelt"
title: "NeLT: Object-Oriented Neural Light Transfer"
authors: "Chuankun Zheng; Yuchi Huo; Shaohua Mo; Zhihua Zhong; Zhizhen Wu; Wei Hua; Rui Wang; Hujun Bao"
year: "2023"
venue: "ACM Transactions on Graphics 42(5), Article 163"
doi: "10.1145/3596491"
report_status: "evidence-reviewed"
main_source: "project-root/3596491.pdf"
supplemental_status: "unavailable"
official_code_status: "unavailable"
official_code_commit: "not-applicable"
author_worker: "/root/nelt_full_report"
reviewer: "/root/belcour2018_review"
last_verified: "2026-08-29"
---

# NeLT: Object-Oriented Neural Light Transfer

> 本报告已经从原先的摘要级来源审计升级为完整 author pass。用户提供的 16 页正文已逐页渲染并视觉核对；本文把公式、图表、图注、脚注、训练和比较配置先按原论文恢复，再进行项目分析。正文明确把逐层网络结构和若干额外实验放在 supplementary，但本次允许访问的一手入口没有取得这些文件，因此精确层数、hidden width、neural texture shape 和 insertion-order 数值仍保持“未报告/未取得”，不从示意图猜实现。

## 1. 研究对象与报告边界

NeLT 研究的是**场景级、对象导向的动态光传输表示**。给定背景对象集合 `S`、光源集合 `L` 和将要插入的对象 `m`，作者不让一个 monolithic network 直接表示整个场景，而是学习“插入 `m` 会怎样改变已有 radiance field”的 per-object neural light transfer function，再依次组合各对象的 direct/indirect transfer，生成最终 GI 图像。[P §1、§3，Fig.1–4]

它支持论文训练域内的动态 camera、rigid object transform、lighting 和 material variation，并输出 diffuse direct lighting、specular direct lighting、shadow 和 multibounce indirect lighting。作者把它定位为一种 flexible scene representation，而不是成熟的通用 GI renderer；正文明确限制为 rigid opaque objects，且不处理 highly specular surface materials。[P §1、§3、§7]

对本项目而言，NeLT 属于已批准的 scene-transport 第二波，不是局部 `evaluate(wo, wi)` neural material：

- query 是 camera-visible point、scene/object/light representations 和 per-pixel G-buffer；
- output 是已经积分了 lighting、visibility 和 interreflection 的 image-space radiance/transfer component；
- 持久化容量主要是 object-specific networks，运行成本随 NeLT object 数线性增长；
- 没有与 evaluator 匹配的 `sample()/pdf()`，也不以 bare scattering `f` 为输出。[I]

事实标签沿用任务约定：`P` 为 main paper，`S` 为正文提及但本次未取得的 supplementary/video，`C` 为 official code/config/data，`A` 为作者入口，`N` 为本项目已有证据，`I` 为本报告分析。`P/S/C/A` 事实先于 `N/I`。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | 项目根目录 `3596491.pdf`；正式 DOI [10.1145/3596491](https://doi.org/10.1145/3596491)；16 页 | 2026-08-29 | SHA-256 `56566C3D92F29EF6DF6D4B424DECB18CE1BFB9CF2582EC275D4213E028CA5ED5` | 正文全部方法、公式、训练、Table 1–7、Fig.1–13、限制；16/16 页已渲染并视觉核对 |
| Supplemental `S` | 正文多次指向 “supplementary files/materials/video”，尤其用于 exact network structures、insertion order、AE/noisy comparisons 和更多动态结果 | 2026-08-29 | 未取得 | 可确认作者声称存在，但本地未提供；ACM DOI 页面被公开访问网关返回 403，作者公开页面未给 supplement locator；没有登录或请求凭据 |
| Official code/config/data `C` | 本次核对的 DOI/Crossref、Zhizhen Wu 与 Zhihua Zhong publication entry 只给 paper link，未给 NeLT code/config/data | 2026-08-29 | `not-applicable` | 没有可审计 official repository/commit；“入口未发现”不等于证明作者从未发布 |
| Author material `A` | [Zhizhen Wu publication entry](https://wzz.ink/)、[Zhihua Zhong publication entry](https://isaac-paradox.github.io/publications/) | 2026-08-29 | 页面无版本 | 核对作者身份和 paper link；Zhizhen Wu 页面把书目信息显示为 `Vol.42, No.162`，与正式 PDF 的 `Article 163` 冲突，不能用于 article number 裁决 |
| Bibliographic metadata `P-meta` | Crossref REST、DOI、DBLP `journals/tog/ZhengHMZWHWB23` | 2026-08-29 | 不适用 | 核对题名、TOG 42(5)、Article 163、16 页和 DOI；Crossref relation 未列 supplementary |
| Related primary evidence `N-related` | 本任务 `diolatzis-2022-active-exploration-neural-gi.md`、`granskog-2020-compositional-neural-scene-representations.md` | 2026-08-29 | 当前工作树 | 只用于判定 NeLT 对 baseline 的实际改动和跨论文关系，不覆盖 NeLT 自身正文 |
| NeuralShading project evidence `N` | `docs/contracts/scattering_backend.md`、`docs/research/experiment_framework.md`、本任务 NVIDIA correspondence | 2026-08-29 | 当前工作树 | 只用于 §13–§15 的迁移、成本和复现边界 |

本 author pass 没有执行 Git clone/fetch、SSH/token、账号登录或 credential helper，也没有把 Scribd/ResearchGate 等第三方转载当作一手证据。

## 3. 原论文的问题、假设与贡献边界

### 3.1 问题设定

作者把带 GI 的 scene radiance field 写为：[P Eq.(1)]

\[
L_{in}(o,\omega)=G(o,\omega;S,L),
\]

其中 `o` 是 camera origin，`ω` 是 camera ray direction，`S` 是 objects，`L` 是 light sources。论文要解决的是：在 object、material、light 和 viewpoint 改变时，不在每帧 path trace，而是把 `G` 对 object insertion 的变化预先学习为可组合 neural function。[P §3]

### 3.2 作者的三个主要贡献

1. **对象导向的 light-transfer 定义**：一个 NeLT 同时表达新对象自身的 radiance、它对背景投下的 shadow，以及它引起的 indirect-light change。[P Eq.(2–3)，Fig.2]
2. **显式 scene composition**：direct transfer 用前景 radiance/背景 multiplicative shadow ratio 组合，indirect transfer 用前景 radiance/背景 additive residual 组合；按对象逐次插入即可恢复场景图像。[P Eq.(4–5)，Fig.3–4]
3. **hypernetwork + neural texture 架构**：global object/light/scene representations 先生成 neural texture 和小型 per-pixel networks；direct 与 indirect 两条路径各自执行相应的 UV encoder、一次 bilinear fetch 和 decoder，其中 direct 的 local feature 由 diffuse/specular decoders 共用，从而避免在每像素反复解码高维 global representation。[P §4.2，Fig.5–7]

### 3.3 成立范围

- geometry：rigid、opaque；deformation 未支持；
- material：Lambertian diffuse + GGX specular 数据模型，正文排除 highly specular surfaces；
- lighting：diffuse area lights 与 environment lights；point/directional 被作者视为两者特例；spotlight/VPL 属 future work；
- scene：per-object/per-composite-object networks，跨 novel scene/general light pattern 不是保证；
- inference：需要 rasterized G-buffer、对象/背景/光的 representations 和已经组合到当前阶段的 direct/indirect radiance；不需要 path-traced/noisy shading image作为每帧输入；
- training：依然依赖 customized OptiX path tracer 生成高 spp reference components。[P §§3–7]

## 4. 输入、输出、坐标与 query domain

| 项 | 正文定义 | shape/domain | locator |
|---|---|---|---|
| Camera query | `o, ω`，visible point `x` 是从 `o` 沿 `ω` 看到的 `S+m` 首交点 | 每像素一条 camera ray；网络实际取 `g(x)` 和部分 decoder 的 view direction | P Table 2、Eq.(1)、§4.2 |
| Scene state | foreground object `m`，已有 background objects `S`，lights `L` | object set/light set；不是一个固定长度 scene parameter vector | P Eq.(1–6) |
| Foreground representation `z_f` | 按 surface area uniform sample `m`，逐样本编码后 average | 每场景/对象 global representation；sample 数正式配置为 1000 | P §4.1.1、§5.1 |
| Background representation `z_b` | 从 foreground center rasterize 六面 panoramic G-buffers，再由 background encoder 编码 | six `64×64` cubemap faces、64 spp；global representation | P §4.1.1、§5.1.2 |
| Occlusion representation `z_g` | 从 background observation 的 position/normal geometry features 编码 | 用于 direct-light sample weights，不含 background material | P §4.1.2、Fig.5 |
| Direct/indirect light reps `z_l^1,z_l^*` | area/environment light samples 编码并按 scene-conditioned weights 聚合 | 每个 area light 500 points；environment 2000 directions | P §4.1.2、§5.1.1 |
| Per-pixel G-buffer `g(x)` | position `P`、normal `N`、roughness `R`、diffuse color `D`、specular color `S`、view direction `V`、foreground mask `M` 的不同子集 | feature set 因 UV encoder/decoder 而异；position 另有 positional encoding `P_e` | P Table 3 |
| Coordinate frame | background observations、light samples 和 output-view G-buffers 全部变换到 foreground-object local frame | origin = object center；axes = object orientation | P §4.1.3 |
| Output | foreground direct diffuse `T_m^{1DF}`、foreground direct specular `T_m^{1S}`、background diffuse-shadow factor `T_m^{1DB}`、indirect transfer `T_m^*` | 训练/inference 每个 RGB channel 独立；组合后是 image-space radiance | P §4.2、Eq.(7–9)、Fig.5–7 |

### 4.1 Object-oriented transform 的含义

这一步不是把 surface point 变成一般 local tangent frame，而是把**对象外部世界**表达在新插入对象 `m` 的 rigid local frame 中。它用对象中心和 orientation 消除 `m` 的全局平移/旋转，使同一个 object-specific NeLT 能在训练覆盖的场景位置和方向变化下复用。[P §4.1.3]

### 4.2 Runtime input 不等于 path-traced shading input

正文 abstract 所说“不需要 path tracing 或 shading results 作为 input”限定的是 **NeLT inference**：它仍需要清晰 G-buffers，并且需要 foreground/background/light 的采样/观测 representation。训练 GT 来自 256 spp path tracing；CNSR baseline 还需要额外 path-traced observations。不能把 abstract 扩写成“整个 lifecycle 不使用 path tracing”。[P Abstract、§5.1.2、Table 4 caption]

## 5. Representation、逐层网络与数据流

### 5.1 Neural light-transfer 与 composition algebra

原始 object transfer 定义为：[P Eq.(2)]

\[
\begin{aligned}
T_m(o,\omega;S,L)=&\ G(o,\omega;S+m,L)M_m(o,\omega)\\
&+(1-M_m(o,\omega))\,[G(o,\omega;S+m,L)-G(o,\omega;S,L)].
\end{aligned}
\]

`M_m=1` 表示 camera ray 直接看到 `m`，此时 `T_m` 存 `m` 的 radiance；否则存插入 `m` 给背景 radiance 带来的 change。作者随后把 `G=G^1+G^*` 分成 first-bounce direct 与 higher-bounce indirect，并定义：[P Eq.(3)]

\[
\begin{aligned}
T_m^1=&\ G^1(S+m)M_m+(1-M_m)\frac{G^1(S+m)}{G^1(S)},\\
T_m^*=&\ G^*(S+m)M_m+(1-M_m)[G^*(S+m)-G^*(S)].
\end{aligned}
\]

这里省略了 `o,ω,L`。同一 `T_m^1` 在 mutually exclusive mask 两侧存不同 measure：前景是 direct radiance，背景是插入前后 direct radiance ratio，用 multiplicative ratio 表达 shadow；`T_m^*` 的背景则是 additive indirect change。[P §3]

正文没有说明当分母 `G^1(S)` 为零或很小时，direct background ratio 如何稳定化、裁剪或标记无效，也没有交代该 ratio target 的数值生成策略；这是复现 Eq.(3) 时必须保留的 main-paper gap，不能默认加 epsilon。[P Eq.(3) 未报告]

对应 composition 为：[P Eq.(4–5)]

\[
G^1(S+m)=M_mT_m^1(S)+(1-M_m)T_m^1(S)G^1(S),
\]

\[
G^*(S+m)=M_mT_m^*(S)+(1-M_m)[T_m^*(S)+G^*(S)].
\]

因此 scene 从 empty space 的 direct/indirect fields 开始，逐次插入 object。理想 transfer 对所有 `S,L` 完全学习时 order-independent；实际 approximation 会让 insertion order 产生轻微差异，作者要求 inference 固定 order 以保持一致并避免 flicker，具体数值只在未取得的 supplementary。[P §3，Fig.3]

所有 mutually stationary static objects 可以合成一个 composite object；empty-space transfer 退化为 Eq.(1) 的 radiance field，从而可由另一 scene representation method 初始化。[P §3]

### 5.2 Representation extraction

1. **Background `z_b`**：从 foreground center 向六个 cubemap 方向 rasterize background panoramic G-buffers，编码几何与材质。作者承认完整 multiscattering 理论上需要所有 foreground/background information；实际假设 foreground center 不可见区域对 indirect transfer 影响可忽略。[P §4.1.1]
2. **Foreground `z_f`**：按 surface area 均匀采 object surface，记录 `P,N,R,D,S`，per-sample encoder 后 average。object-specific network 本身可隐式记 geometry，但动态 foreground material 仍须由这些 samples 提供。[P §4.1.1、Table 3]
3. **Light samples**：area light 按 emitted radiance importance sample luminous surface，记录 position、normal、sampling probability 与 emitted radiance；environment 按 emitted radiance importance sample direction，记录 direction、probability 与 radiance。[P §4.1.2]
4. **Expected power normalization**：为每个 light sample 算 expected power，并除以所有 samples 的 mean `ē`；sample feature 是 `P,N,E`。environment 没有 position/normal，作者用 dummy position 兼作 class label，并把 sampled direction 对齐到 normal slot，让两种 light 共享一个 encoder。[P §4.1.2、Table 3]
5. **Sample aggregator**：先用 MLP 根据 sample `P,N` 与 `z_b` 产生权重，对 encoded light samples weighted sum 得 indirect rep `z_l^*`；direct rep `z_l^1` 用不含 material 的 `z_g` 代替 `z_b`，使 direct transfer 对周围 object material invariant。[P §4.1.2]

### 5.3 Hypernetwork、neural texture 与 pointwise decoder

正文把 direct/indirect transfer 改写为：[P Eq.(6)]

\[
T_m^1(o,\omega;S,L)\rightarrow T_m^1(g(x);z_l^1),\qquad
T_m^*(o,\omega;S,L)\rightarrow T_m^*(g(x);z_s,z_l^*),
\]

其中 `z_s=[z_b,z_f]`。Indirect data flow 是：[P Eq.(7)，Fig.6]

1. neural texture generator `N_m^*` 由 `(z_s,z_l^*)` 产生 object-conditioned 2D neural texture；
2. UV hypernetwork `H_m^{*U}` 由同一 global reps 产生 UV encoder `U_m^*` 的 weights `w_m^{*U}`；
3. indirect hypernetwork `H_m^*` 产生 decoder `F_m^*` 的 weights `w_m^*`；
4. `U_m^*(g(x);w_m^{*U})` 给出 texture coordinates；
5. 使用 PyTorch `torch.grid_sample` 对 neural texture bilinear fetch；
6. `F_m^*` 读取 per-pixel G-buffer、view-direction map 与 local fetched feature，输出 `T_m^*`。

`U_m^*` 和 `F_m^*` 都是逐 pixel 独立执行的 MLP，因此没有 CNN 邻域依赖；这是作者主张 multiview consistency 的结构来源。主文没有 exact layer count/width/activation、UV range、texture resolution/channels 或 hypernetwork output dimension，明确把这些放在 supplementary。[P §4.2.1、footnote 1]

Direct network 采用同一思想，但由 `z_l^1` 产生一张 direct neural texture、一个 UV encoder 和 diffuse/specular 两套 decoder weights。它显式拆成：[P Eq.(8)，Fig.7]

\[
T_m^1=T_m^{1D}+T_m^{1S},
\]

并让 diffuse decoder 以不同 output channels 同时输出 foreground direct diffuse `T_m^{1DF}` 与 background shadow `T_m^{1DB}`：

\[
T_m^1=(T_m^{1DF}+T_m^{1S})M_m+(1-M_m)T_m^{1DB}.
\]

为了简化 shadow integral，作者只对 diffuse component 预测 shadow；specular shadow 没有被单独分解，后来成为 Fig.13(a) 的明确失败。[P §4.2.2、Fig.5/7/13]

### 5.4 正文能够恢复的模块级配置

| 模块 | 输入 | 已知运算 | 输出 | shared/per-asset | 未闭合配置 | locator |
|---|---|---|---|---|---|---|
| Background encoder | six panoramic G-buffers，`PNRDS` | encoder；exact topology 未给 | `z_b` | 每个 NeLT 的训练体系；是否跨 object share 未报告 | layers/width/latent dim/activation | P §4.1.1、Table 3 |
| Occlusion encoder | background `PN` | encoder | `z_g` | 未报告 | 同上 | P §4.1.2、Table 3 |
| Foreground encoder | 1000 samples，`PNRDS` | per-sample encode + average | `z_f` | 未报告 | 同上 | P §4.1.1、Table 3 |
| Light encoder | light sample `PNE` | shared area/environment encoder | encoded samples | shared across light types | layers/feature dim | P §4.1.2、Table 3 |
| Direct sample aggregator | encoded light samples + weights from `PN,z_g` | MLP weights + weighted sum | `z_l^1` | per NeLT | normalization/MLP topology | P §4.1.2、Fig.5 |
| Indirect sample aggregator | encoded light samples + weights from `PN,z_b` | MLP weights + weighted sum | `z_l^*` | per NeLT | 同上 | P §4.1.2、Fig.5 |
| Direct texture generator `N_m^1` | `z_l^1` | neural texture generation | direct texture | object-specific | texture H×W×C/topology | P Eq.(8)、Fig.7 |
| Indirect texture generator `N_m^*` | `z_s,z_l^*` | neural texture generation | indirect texture | object-specific | 同上 | P Eq.(7)、Fig.6 |
| Direct UV encoder `U_m^1` | `PNRM` | MLP；weights by `H_m^{1U}` | UV | generated per state | layers/width/output mapping | P Eq.(8)、Table 3 |
| Indirect UV encoder `U_m^*` | `PNRDSVM` | MLP；weights by `H_m^{*U}` | UV | generated per state | 同上 | P Eq.(7)、Table 3 |
| Diffuse direct decoder `F_m^{1D}` | `P_eNRDM` + local texture feature | pointwise MLP；weights by `H_m^{1D}` | `T_m^{1DF}`, `T_m^{1DB}` | generated per state | layers/width/channels | P Eq.(8)、Fig.7、Table 3 |
| Specular direct decoder `F_m^{1S}` | `P_eNRSVM` + local feature | pointwise MLP；weights by `H_m^{1S}` | `T_m^{1S}` | generated per state | 同上 | P Eq.(8)、Table 3 |
| Indirect decoder `F_m^*` | `P_eNRDSVM` + local feature | pointwise MLP；weights by `H_m^*` | `T_m^*` | generated per state | 同上 | P Eq.(7)、Table 3 |

`P_e` 只表示 position 经过 NeRF-style positional encoding；正文没有 frequency 数、是否含 raw position 或 frequency scale。任何精确 topology 都必须等 supplementary 或 official code 解锁后恢复。[P Table 3 note]

## 6. 数据、GT/reference 与 query/sampling recipe

### 6.1 四个数据集

| Dataset | 原论文变化轴 | object organization | locator |
|---|---|---|---|
| Box | box 内 movable figure；static objects 有不同 material；四种 shape 的 lights | static objects 可做 composite；一个 dynamic object | P §5.1.1、footnote 2 |
| Outdoor | environment lighting + 3 local lights；environment map 可旋转/更换；statue material 可编辑 | 一个 dynamic object | P §5.1.1、§6.2 |
| Indoor | bed、sofa、table 的 position/material 可变；window area lights 和 top lights 的 size/position 可变 | 三个 dynamic objects，各自独立 NeLT | P §5.1.1、footnote 2 |
| Room | 从 room-model library 随机选 room，随机放 furniture/foreground/area lights，再 randomize whole-scene materials | 用于跨 scene generalization：训练 sofa NeLT，再插入未见 Indoor | P §5.1.1、§6.3 |

每个 dataset 约 6000 random training scenes、100 random test scenes，每 scene 八个 random views。若按正文直接展开，是约 48,000 training views 和 800 test views/dataset；论文没有给 exact count、random seed、scene split manifest 或 test-view selection rule。[P §5.1.1]

### 6.2 Reference generation

| 项 | 正文配置 | locator |
|---|---|---|
| Renderer | customized OptiX path tracer | P §5.1.2 |
| Material model | Lambertian diffuse lobe + GGX specular lobe | P §5.1.2 |
| Training image | `256×256`, `256 spp` per view | P §5.1.2 |
| Background observation | six `64×64` cubemap images, `64 spp` | P §5.1.2 |
| Recorded targets | reference image、G-buffers、direct lighting、indirect lighting、direct diffuse component | P §5.1.2 |
| Object insertion | generation 时随机插入 dynamic objects，并在每次 insertion 后记录数据；对应 foreground object 用这些数据训练 | P §5.1.1 |
| Foreground samples | 1000 surface-area-uniform points/object | P §5.1.1 |
| Area-light samples | 500 points **for each area light** | P §5.1.1 |
| Environment samples | 2000 points/directions | P §5.1.1 |

GT 是 scene/image transport reference，不是局部 BSDF query corpus。正式训练数据是 offline precomputed images/components；论文没有报告 storage bytes、data format、component precision、per-dataset generation time 或 path-tracer convergence check。

## 7. Loss、optimizer 与训练 lifecycle

### 7.1 HDR 预处理

训练 radiance 和 light-sample emitted radiance 逐 RGB channel 使用：[P §5.2.1]

\[
\tilde{x}=\log(1+x).
\]

在 `log1p` 前，light samples 的 expected power 以及所有 shading results 都除以 mean expected power `ē`。作者说此举缓解高 dynamic range 造成的不稳定；论文没有给 clamp、negative-value policy 或 inverse-transform evaluation pipeline。[P §5.2.1]

### 7.2 Joint component loss

令 hat 表示 target，正文 Eq.(9) 为：[P §5.2.2]

\[
\begin{aligned}
\mathcal L=&\ \bar e\,[
\ell_1(\hat T_m^{1DF},T_m^{1DF})+
\ell_1(\hat T_m^{1S},T_m^{1S})+
\ell_1(\hat T_m^*,T_m^*)]\\
&+\ell_1(\hat T_m^{1DB},T_m^{1DB}).
\end{aligned}
\]

因此 foreground diffuse、specular 和 indirect loss 乘回 `ē`，shadow ratio loss 不乘。四项没有额外论文权重。RGB channels 被视为在该 light-transport formulation 中独立，dataset 通过 channel splitting 扩展，training 和 inference 每 channel 独立执行；正文没有说明是三套参数、共享 network 的三次调用还是 channel-conditioned samples，精确实现需 code/supplementary 才能裁决。[P §5.2.2]

### 7.3 Optimizer 与生命周期

| 项 | 正式配置 | locator |
|---|---|---|
| Optimizer | Adam | P §5.2.2 |
| Learning rate | `1e-4` | P §5.2.2 |
| Minibatch | 64 | P §5.2.2 |
| Iterations | 200,000 | P §5.2.2 |
| Hardware | two Nvidia RTX A6000 GPUs | P §5.2.2 |
| Training time | approximately 60 hours | P §5.2.2 |
| Train granularity | object-specific NeLT；static mutually stationary objects可合并为 composite transfer | P §3、§6.1 |
| Model selection | 未报告 validation cadence、best/last checkpoint、early stopping | P 未报告 |
| Reproducibility | optimizer betas/epsilon、weight decay、initialization、seed、precision、distributed strategy 未报告 | P 未报告；S/C 未取得 |

NeLT 没有使用 AE 的 active-exploration training。正文明确说 baseline 中只采用 AE network 与 scene representation，因为作者认为 AE 的 training strategy 与 NeLT orthogonal；把两者组合留作 future work。[P §6.1.1、§7]

## 8. Inference、部署与成本

### 8.1 Runtime path

1. rasterizer 为当前 view 生成 per-pixel G-buffers；
2. 对当前 object insertion stage，sample/observe foreground、background 和 lights，构建 `z_f,z_b,z_g,z_l^1,z_l^*`；正文没有说明这些 representations 是每帧、仅 scene change、light change 还是 object change 时重算；
3. hypernetworks 根据 global reps 生成 neural texture 和 UV/decoder weights；
4. 全分辨率 pixels 并行执行 pointwise UV MLP、bilinear fetch 和 small decoder；
5. 用 Eq.(4–5) 更新 direct/indirect image；
6. 对所有单独 modeled dynamic objects 重复，最后合成 GI。[P Fig.3–7、§6.1]

### 8.2 正文测得的 frame/inference time

| Resolution/scene | Box | Outdoor | Indoor | 测量范围 |
|---|---:|---:|---:|---|
| `256×256` NeLT | `26.66 ms` | `28.43 ms` | `60.16 ms` | PyTorch，single RTX 3090；G-buffer time excluded |
| `1024×1024` NeLT | `300.62 ms` | `326.07 ms` | `656.53 ms` | 同上；Indoor 含三个单独 dynamic-object NeLT |
| `1024×1024` all-dynamic composite transfer | `152.03 ms` | `151.20 ms` | `152.53 ms` | 把 whole scene（lights除外）视为一个 composite，quality/dynamics 不同 |

[P Table 4、Table 7]

正文还声称 small decoders 用 tiny-cuda-nn 可在 single RTX 3090、`512×512` 达到约 20–50 fps；具体 backend、precision、各 scene 对应 fps 和包含哪些 representation/composition cost 只指向未取得的 supplementary video，不能把它与 Table 4 PyTorch timing 混成同一正式配置。[P §6.1.2]

### 8.3 Cost boundary 与遗漏

- Table 4 的 AE/CNSR/NeLT 都在 PyTorch 测；denoisers 的 time = PBRT4 GPU noisy render + PyTorch denoise。[P Table 4 caption]
- CNSR 需要 additional path-traced observation images，但其 acquisition time **没有**计入 timing；不能用表中 time 做完整 pipeline ranking。[P Table 4 caption]
- comparison 的 G-buffers 是 16-spp noisy-image byproducts；caption 写 “16 samples per sample”，上下文指 16-spp。作者认为 rasterizer G-buffer cost 相对 inference 可忽略，因此所有方法均不计 G-buffer time。[P Table 4 caption]
- Indoor 比其他 scene 慢，是三个 dynamic-object NeLT 分别 inference；作者明确指出 per-frame time 随 NeLT object count 线性增长。[P §6.1.2、§7]
- per-object parameter count、hypernetwork-generated weight bytes、neural texture bytes、activation memory、MAC/FLOP、texture reads、precision、representation build time、composition pass bandwidth均未报告。
- 60-hour training 与 offline dataset generation 不含在 frame time；没有 serialized deployment bundle 或 cross-backend parity evidence。
- 论文结论称 inference 不需 customized hardware such as RT core；这不覆盖 training reference generation，后者明确基于 OptiX path tracer。[P §5.1.2、§8]

## 9. 实验 protocol、baseline、指标与结果

### 9.1 论文自己的方法分类表

Table 1 是作者用来解释 task difference 的 qualitative taxonomy，不是统一成本/质量 benchmark：

| Method | View | Scene | GI input space | Primitive | Rendering（原表） |
|---|---|---|---|---|---|
| CNSR | Consistent | Preset | Parameter Space | Scene | Path Tracing |
| AE | Consistent | Preset | Parameter Space | Scene | Rasterization |
| Deep Shading | Dependent | Scalable | Screen Space | Pixel | Rasterization |
| Denoising | Dependent | Scalable | Path Space | Pixel | Path Tracing |
| NeLT | Consistent | Scalable | Object Space | Object | Rasterization |

[P Table 1]

脚注式解释澄清：Deep Shading/denoising 用 CNN surrounding pixels，故作者归为 view-dependent；AE/CNSR 是 preset scene，不跨 unseen scene；NeLT按 object逐步组合；CNSR novel-view rendering 使用 rasterization，但 representation 需要从 path-traced scene observations 提取。由此原表 CNSR 的 “Path Tracing” 指完整 evidence source，不应简写成 runtime 全程 path tracing。[P Table 1 explanation]

### 9.2 Baseline correspondence

| Baseline | NeLT paper 实际配置 | 与原方法身份的边界 | locator |
|---|---|---|---|
| CNSR | same NeLT dataset；equal or longer training；在 original inputs 外增加 material-related G-buffers | 这是为可变 material 做的 baseline adaptation；path-traced observations 的取得成本未计时 | P §6.1.1、Table 4 caption |
| AE | **只使用其 network 和 scene representations；不使用 active-exploration training strategy**；same data、equal or longer training | 表中明确叫 `AE (uniform)`；因此只检验 AE architecture under uniform training，不是完整 AE method | P §6.1.1、Table 4 |
| WSKP | 重新渲染 random 2–32 spp noisy input/clean GT；用作者 original implementation 在每个 NeLT dataset fully retrain | real-time path-traced denoiser；Table 给 4/16 spp | P §6.1.1 |
| AFGSA | 用作者 released pretrained model，对每个 NeLT dataset fine-tune | offline denoiser；Table 给 4/16 spp | P §6.1.1 |
| Deep Shading | 仅 related-work/Table 1 taxonomy，无 Table 4 数值 | 不能把它写成 quantitative baseline | P §2.2、Table 1 |

用户提到的 “AE” 已由正文直接识别：**Active Exploration for Neural Global Illumination of Variable Scenes**，Stavros Diolatzis、Julien Philip、George Drettakis，TOG 41(5), Article 171, DOI `10.1145/3522735`。[P References] 本任务已经有该文的独立 `evidence-reviewed` 报告；NeLT 自己的 Table 4 只测了去掉 active training 后的 `AE (uniform)`。

### 9.3 256×256 quantitative comparison

| Method | Box time/L1/SSIM | Outdoor time/L1/SSIM | Indoor time/L1/SSIM |
|---|---|---|---|
| AE (uniform) | `29.55 / 0.0091 / 0.9663` | 不支持/未测 | 不支持/未测 |
| CNSR | `50.14 / 0.0100 / 0.9653` | `51.21 / 0.0126 / 0.9491` | `50.98 / 0.0198 / 0.9188` |
| WSKP 4 spp | `42.00 / 0.0105 / 0.9443` | `37.30 / 0.0228 / 0.8734` | `37.01 / 0.0238 / 0.8567` |
| WSKP 16 spp | `81.81 / 0.0076 / 0.9594` | `60.93 / 0.0156 / 0.9239` | `60.34 / 0.0167 / 0.9069` |
| NeLT | `26.66 / 0.0081 / 0.9656` | `28.43 / 0.0127 / 0.9465` | `60.16 / 0.0177 / 0.9221` |
| AFGSA 4 spp | `192.17 / 0.0077 / 0.9607` | `187.52 / 0.0160 / 0.9260` | `188.22 / 0.0163 / 0.9197` |
| AFGSA 16 spp | `231.98 / 0.0062 / 0.9678` | `211.28 / 0.0118 / 0.9506` | `211.69 / 0.0121 / 0.9410` |

[P Table 4(a)]

### 9.4 1024×1024 quantitative comparison

| Method | Box time/L1/SSIM | Outdoor time/L1/SSIM | Indoor time/L1/SSIM |
|---|---|---|---|
| AE (uniform) | `466.65 / 0.0090 / 0.9633` | 不支持/未测 | 不支持/未测 |
| CNSR | `655.24 / 0.0099 / 0.9641` | `658.32 / 0.0112 / 0.9416` | `657.28 / 0.0196 / 0.9087` |
| WSKP 4 spp | `147.69 / 0.0082 / 0.9517` | `105.40 / 0.0151 / 0.9018` | `107.41 / 0.0169 / 0.8801` |
| WSKP 16 spp | `473.86 / 0.0061 / 0.9617` | `303.61 / 0.0104 / 0.9339` | `298.52 / 0.0119 / 0.9147` |
| NeLT | `300.62 / 0.0079 / 0.9630` | `326.07 / 0.0112 / 0.9383` | `656.53 / 0.0175 / 0.9129` |
| AFGSA 4 spp | `2478.51 / 0.0059 / 0.9624` | `2436.22 / 0.0113 / 0.9256` | `2438.23 / 0.0116 / 0.9200` |
| AFGSA 16 spp | `2802.07 / 0.0050 / 0.9661` | `2631.82 / 0.0086 / 0.9454` | `2626.73 / 0.0093 / 0.9319` |

[P Table 4(b)]

作者自己的解释是：AFGSA 16 spp 在所有 scene 的 L1/SSIM 最好但最慢；多数 case WSKP 16 spp 得 best L1，CNSR/AE 得较高 SSIM，NeLT 通常比 CNSR 有更低 L1、比 WSKP 有更高 SSIM，因而作者称“quantitatively comparable”。这不是 NeLT 全指标领先。[P §6.1.2]

Figure 9 显示 NeLT 相对 CNSR 有更清晰 shadow，相对低-spp denoiser 少 blur/dirty mottled blocks。作者把 denoiser flicker 归因于 stochastic path tracing，并称 NeLT 的 per-pixel pointwise network 理论上 multiview consistent、temporally stable；但正文没有 temporal metric、sequence protocol 或 repeated-frame variance，主要 evidence 是图和 supplementary video。[P §6.1.2、Fig.9]

### 9.5 为什么 AE 只有 Box

Table 4 caption 明确说 AE 结果在 Outdoor/Indoor 不可得，因为它不支持 texture modification 这类 nonparametric variation。正文还说 AE 和 CNSR 的 global scene-oriented representation 能产生 smooth appearance，却难学 movable-object shadows 和 variable-material reflections；作者把原因解释为 global representation 没有分离 local object-oriented dynamicity。该解释没有用 matched representation-only ablation证明，必须保留为作者分析。[P §6.1.2、Table 4 caption]

### 9.6 Dynamics 与 material editing

Box 用不同形状/位置/radiance 的 area lights、moving object/view；Outdoor 用 changeable/rotatable environment map、local lights 和 editable statue material；Indoor 用 movable sofa/bed/table 和 editable bed material。Fig.10 展示 roughness `0.3/0.65/1.0` 与两组颜色时，diffuse/specular/indirect components 随设置变化；论文没有报告 material parameter test distribution、extrapolation range 或 component-wise metric。[P §6.2、Fig.10]

### 9.7 Generalization experiments

| Experiment/method | Time (ms) | L1 | SSIM | protocol |
|---|---:|---:|---:|---|
| OutdoorGeneral WSKP 4 spp | `106.51` | `0.0150` | `0.8997` | same data retrain |
| OutdoorGeneral NeLT | `323.21` | `0.0159` | `0.9329` | 196 train environment maps random rotation，9 unseen test maps |
| OutdoorGeneral NeLT fine-tuned | `325.73` | `0.0130` | `0.9357` | test maps fine-tune about 30 min |
| OutdoorGeneral WSKP 16 spp | `306.46` | `0.0102` | `0.9337` | same data retrain |
| OutdoorGeneral AFGSA 4 spp | `2429.92` | `0.0122` | `0.9228` | same data fine-tune |
| OutdoorGeneral AFGSA 16 spp | `2633.58` | `0.0095` | `0.9428` | same data fine-tune |
| Room→Indoor WSKP 4 spp | `108.55` | `0.0151` | `0.8902` | same data retrain |
| Room→Indoor NeLT | `653.27` | `0.0203` | `0.9206` | sofa NeLT trained on randomized Room，insert into unseen Indoor context |
| Room→Indoor NeLT fine-tuned | `653.41` | `0.0161` | `0.9235` | Indoor fine-tune about 30 min |
| Room→Indoor WSKP 16 spp | `300.12` | `0.0105` | `0.9240` | same data retrain |
| Room→Indoor AFGSA 4 spp | `2442.12` | `0.0103` | `0.9293` | same data fine-tune |
| Room→Indoor AFGSA 16 spp | `2631.62` | `0.0083` | `0.9389` | same data fine-tune |

[P Table 5、§6.3]

OutdoorGeneral 支持作者“unseen environment map 有一定 generalization”的主张，但 NeLT 的 L1/SSIM 没有超过 AFGSA 16 spp，fine-tune 只部分改善。Room→Indoor 未 fine-tune 时会 highlights/shadows 不准且整体偏亮，约 30 min fine-tune 后改善；作者承认 totally untrained object-space local-light pattern 会产生不可预知结果，local-light generalization 无法做全面结论。[P §6.3、Fig.11]

## 10. 消融、失败尝试与负结果

### 10.1 正式 ablation

| Variant | 具体改动 | Box 1024 L1/SSIM | 作者结论与边界 | locator |
|---|---|---|---|---|
| Full NeLT | neural texture + direct/indirect GI decomposition | `0.0079 / 0.9630` | best observed | P Table 6 |
| w/o Neural Texture | 移除 `N/U/grid_sample` local feature path；`F_m^*,F_m^{1D},F_m^{1S}` 直接从 G-buffer + hyper-generated weights 输出 | `0.0085 / 0.9602` | 颜色更不准、shadow 明显更软；支持 local neural texture | P Eq.(10)、Fig.12、Table 6 |
| w/o GI Decomposition | 直接学未拆分 `T_m`；decoder size tripled；L1 training | `0.0084 / 0.9620` | full component decomposition 更好；但正文没有 parameter/MAC bytes，不能确认严格 parameter-matched | P §6.4、Fig.12、Table 6 |

### 10.2 Composite all-dynamic-object tradeoff

作者尝试把 whole scene（lights 除外）作为一个 composite object：分别 sample 每个 variable object 后 concatenate foreground samples，用一次 network 直接预测 scene direct/indirect fields。1024² 结果是：[P §7、Table 7]

| Scene | Time | L1 | SSIM | 相对 per-object NeLT |
|---|---:|---:|---:|---|
| Box | `152.03 ms` | `0.0077` | `0.9632` | 比 per-object `300.62 ms` 更快，aggregate metric 略好 |
| Outdoor | `151.20 ms` | `0.0113` | `0.9373` | 比 `326.07 ms` 快，quality接近 |
| Indoor | `152.53 ms` | `0.0229` | `0.8981` | 比 `656.53 ms` 快，但明显退化 |

尽管 aggregate metrics 在 Box 可接受，作者明确说这种 composite function 的 highly dynamic GI effects 变差，类似 CNSR/AE；实际建议只给重要 dynamic objects（如 avatar）独立 NeLT，其余用 CNSR、AE 或 composite transfer。这是“更短固定 pass vs object-local dynamic quality”的真实失败/折中，不应只记录为加速成功。[P §7]

### 10.3 作者展示的 failure cases

| Failure | 原因/观察 | 分类 | locator |
|---|---|---|---|
| Specular-component shadow 缺失 | direct network只分解 diffuse shadow，没有显式 specular shadow direction | design limitation / author-negative | P §4.2.2、Fig.13(a) |
| High-frequency detail 丢失 | highly specular reflection、complex occlusion 仍困难 | known limitation | P §7、Fig.13(b) |
| Novel-scene highlight 错误 | Room→Indoor context 偏离训练；会产生 incorrect highlights | generalization failure | P §6.3、Fig.13(c) |
| Novel-scene shadow 过暗 | 未见 geometry/lighting context；fine-tune 能缓解但非零样本解决 | generalization failure | P §6.3、Fig.13(d) |
| Insertion order approximation | 理论 order-independent，学习误差使结果略变；需固定 order 防 flicker | approximation limitation；数值在缺失 S | P §3 |
| 多 dynamic objects | inference 随独立 NeLT 数线性增长；Indoor 约翻倍/更慢 | scaling limitation | P §6.1.2、§7 |
| Object deformation | 没有 deformation representation | missing capability | P §7 |
| Material-entangled NeLT | 当前 network output 含 material；任意 appearance texture switching 尚未实现 | future-work boundary | P §7 |
| General lights/multibounce clues | sample representation尚未覆盖 spotlight；VPL 仅 future idea | future-work boundary | P §7 |

作者建议 neural shadow mapping 式 auxiliary features 和 AE 的 MCMC active training 可能改善 high-frequency effect/generalization，但正文没有执行这些组合，必须记作 future hypothesis 而非 NeLT 的已验证配置。[P §7]

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Main paper `P` | Supplemental `S` | Official code/config `C` | 结论/冲突 |
|---|---|---|---|---|
| Formal identity | TOG 42(5), Article 163, 16 pages | 未取得 | 不适用 | Zhizhen Wu author page 写 `No.162`，以正式 PDF/DOI 的 163 为准 |
| Architecture | 完整 module graph、Eq.(7–8)、feature table | 正文说 exact network structures 在 S | 未发现 locator | layer/width/texture/latent/activation 不可闭合；不能从 Fig.6–7方框猜 topology |
| Insertion order | 明确 approximation causes order dependence、需固定 inference order | 具体 comparison 在 S | 不可得 | direction known，effect size unknown |
| AE comparison | 明确 only network/scene representation，uniform training | AE qualitative/noisy inputs 因篇幅放 S | 不可得 | Table 4 是 `AE (uniform)`，不是完整 Active Exploration method |
| CNSR pipeline | Table 1 “Rendering=Path Tracing” | 未取得 | 不可得 | 同表解释称 runtime rasterization but path-traced observations；应登记为 hybrid evidence path，不是简单矛盾 |
| RGB execution | channel splitting，training/inference each channel independently | 可能有实现细节但未取得 | 不可得 | parameter sharing/三次调用无法判定 |
| Training | 200k、batch64、Adam1e-4、60h/2×A6000 | exact topology/config 未取得 | 不可得 | betas/seed/precision/checkpoint selection gap |
| Runtime | PyTorch RTX3090 Table4；tiny-cuda-nn 20–50fps statement | tcnn demo 在 supplementary video | 不可得 | 两个 backend/cost scopes不能混合；representation/G-buffer breakdown未报告 |
| G-buffer timing | Table4 excludes it；caption 原文写 “16 samples per sample”，紧接着说 16-spp noisy image byproduct | 未取得 | 不可得 | 报告保留原文 typo/ambiguity，不静默改表述 |
| Path tracing claim | inference 不需 path-traced/shading input | 未取得 | training uses OptiX PT | 这是 lifecycle scope distinction，不是 paper contradiction |

本报告没有 official code correspondence，因此不能声称“复现过 NeLT”。它只是一份 main-paper-complete、supplement/code-gapped 的方法报告，下一步独立 reviewer 应重点复核公式的 foreground/background measure、Table 4 cost exclusions 和 AE identity。

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

- 只处理 rigid opaque objects，不含 deformation；不面向 highly specular surface materials。[P §3、§7]
- per-object NeLT inference time 随 modeled-object count 线性增长；many dynamic objects 成本高。[P §7]
- composite many-object transfer 虽快，但 object-local highly dynamic GI 退化，类似 monolithic scene representation。[P §7]
- 背景只从 foreground center 可见的 panoramic G-buffer 表示；不可见区域被假设对 indirect transfer 影响小。[P §4.1.1]
- direct shadow 只在 diffuse component 预测，specular shadow/complex visibility 失败。[P §4.2.2、Fig.13]
- high-frequency reflection/occlusion 会丢细节。[P §7、Fig.13]
- generalization 取决于 test state 与 training state 的偏离；unseen local-light pattern、scene geometry/material 会导致错误 highlight/shadow，需要额外 fine-tune。[P §6.3、§7]
- 当前 sample-based lighting 表示主要覆盖 area/environment lights；material/light/deformation更一般的 factorization 是 future work。[P §7]
- 作者明确说目标是探索 flexible scene representation，不是证明优于 path tracing/denoising 的 mature GI solution。[P §1、§6.1.2]

### 12.2 未报告/未取得

- exact encoder、aggregator、hypernetwork、texture generator、UV encoder、decoder topology；
- latent dimensions、neural texture resolution/channels、generated weight count、activation、normalization、positional-encoding frequencies；
- per-object/shared parameter count、FP precision、serialized bytes、activation/state memory、MAC/FLOP/read count；
- representation extraction/build cadence 与时间，light/object/background sampling在 runtime 是离线 cache、每 scene change 还是每 frame；
- train/test scene manifest、camera/material/light distributions、seed、重复训练、variance/CI；
- GT storage/layout/precision、OptiX integrator、bounce count、sampling strategy、generation time和convergence；
- Adam betas/epsilon/weight decay、initialization、LR schedule、mixed precision、checkpoint selection；
- Eq.(9) component loss 的 batch aggregation/reduction exact implementation；RGB independent究竟对应独立weights还是独立samples；
- Eq.(3) direct background ratio 在 `G^1(S)=0` 或近零处的 target construction、稳定化、clamp/invalid policy；
- L1/SSIM 是 linear/log/tonemapped domain，SSIM window/normalization、per-image/aggregate protocol；
- temporal/multiview consistency 的定量 protocol；
- insertion-order comparison、AE/noisy images、tcnn setup 和额外 dynamics（均指向未取得 S/video）；
- official code/config/data/asset license 与可执行命令。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

NeLT 的 object-oriented representation 不是一个单独 latent。完整容量由六部分共同承担：

1. **object-specific learned weights**：每个重要 dynamic object 一套 NeLT；static group 可有 composite NeLT；
2. **global sampled/observed representations**：foreground surface samples、background cubemap G-buffer、light samples显式携带 geometry/material/light；
3. **hypernetwork-generated program state**：UV/decoder weights 随当前 global representations 变化；
4. **neural textures**：把高维 global context 空间化，供 pointwise query 一次 bilinear fetch；
5. **query G-buffer/rasterizer**：primary visibility、surface attributes 和 mask不由 network latent重建；
6. **composition state**：上一 insertion stage 的 direct/indirect radiance fields参与下一 object composition。

所以“per-object function”不表示一个小型独立 shader closure。它更像 object-conditioned scene transport program，依赖 external renderer、scene representations 和 image buffers。[P/I]

### 13.2 成功所依赖的假设

- object boundaries 与 rigid canonical frame 可知，scene changes 可表达成按对象 insertion；
- object count有限，重要 dynamic objects可付出独立 network inference；
- foreground center 的 six-face observation足以概括关键 background interaction；
- lighting 能被固定数量 importance samples 和 permutation-style aggregation概括；
- Lambertian+GGX、非 highly-specular domain 足以覆盖目标内容；
- transfer approximation误差不会让 fixed insertion order 下的逐步 composition失稳；
- runtime能提供全分辨率 G-buffer 和多张 direct/indirect composition buffers；
- 60h/object-family training 与 offline path-traced data可接受；
- image-space L1/SSIM允许丢失部分 sharp/high-frequency effects。

这些是 NeLT quality/flexibility 的共同前提；不能只把成功归因于 hypernetwork。

### 13.3 相对 AE、CNSR、denoiser 与 Deep Shading 的真实关系

- **AE**：Active Exploration 的核心贡献是训练 query selection/reuse/adaptive resolution；NeLT baseline把这部分拿掉，只用其 scene network/representation。因此 NeLT 证明的是 object-oriented representation 相对一个 `AE architecture + uniform data` 的差异，不能据此裁决完整 AE training 是否弱于 NeLT。作者反而把 `AE MCMC + NeLT`列为 future work。[P; N-related]
- **CNSR**：CNSR用几张 path-traced observations编码 entire scene，再以 G-buffer query；NeLT用 foreground/background/light 的 object-local samples/observations，并显式递推 composition。二者都不是 bare material evaluator，且CNSR被改加 material G-buffers后才进入本表。[P; N-related]
- **Denoisers**：WSKP/AFGSA每帧消费 noisy path-traced image，保留 path tracer 对复杂 visibility/material/deformation 的广泛支持；NeLT把训练域 transport预存在 networks 中。作者没有声称 NeLT 是更成熟/更通用的 GI solution。[P]
- **Deep Shading**：只作为 screen-space CNN taxonomy参照，没有 matched quantitative test。其 surrounding-pixel CNN带来view dependence的说法是 NeLT作者的分类，不是本文重新复现 Deep Shading 得出的结果。[P]

### 13.4 可迁移机制

1. **`prepare` 生成小 program state**：global condition先经 hypernetwork/neural-texture generator，per-query只执行轻量 UV/fetch/decoder。这与本项目“prepare复用、evaluate固定有界”在系统结构上相似，可作为新 candidate；但必须静态限制 generated weights/texture bytes和prepare cost。
2. **按物理角色选择 composition algebra**：direct background shadow 用 multiplicative ratio，indirect background change 用 additive residual；说明 output transform/组合运算应由量的物理角色决定，而非所有 component统一回归同一 residual。
3. **canonical frame**：把外部条件变到 object local frame，减少 rigid transform variation；对 native source material可测试是否用 material-local/tangent canonicalization降低G2 state variance，但不能引入scene object语义。
4. **set/sample encoder**：light/foreground samples经shared encoder + aggregation获得固定尺寸 state，启发对 variable-size native resource graph做 bounded compile-time aggregation；不得把方向采样的runtime数量变成不定长。
5. **typed loss heads**：full model优于不分GI component版本，提示用语义正确、同一 reference可观测的分量监督可能降低干扰；对本项目必须保持最终 output仍是bare `f`，auxiliary heads只在training存在。
6. **负面机制同样可迁移**：object数线性 scaling、中心观测visibility gap、specular shadow未分解和high-frequency loss都提醒我们必须把状态/read/query成本和hard strata写进matched evaluation，而不是只报aggregate image metric。

### 13.5 不能迁移的部分

- `T_m` 编译的是 lighting、visibility、interreflection 和 environment，不是 source material 的 intrinsic scattering；
- direct/indirect image decomposition 不能替代项目 `evaluate/sample/pdf` contract，也不能强迫 native material先改写成scene component；
- foreground/background mask、cubemap、light samples和object insertion不属于单一着色点random-access evaluator input；
- per-object model以及随object count线性调用不满足小型shader evaluator的单次固定预算；
- neural texture是image/object-local 2D cache，未证明能在 arbitrary surface filtering/LOD 下工作；
- RGB independent training可能丢失跨channel correlation，不能直接用于共享RGB evaluator或更一般光谱传输；
- 论文的 20–50fps/512²不能换算成单次 `evaluate(wo,wi)` latency，也未计齐prepare/G-buffer/composition；
- generalization主要在同一 object/task family内，不是未见 native material/source graph 的 compiler generalization。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

状态：`not-applicable` 于 fidelity，`candidate-inspiration-only` 于后续改进。

| 主题 | NeLT | 当前 NVIDIA appearance reproduction | 分类/影响 |
|---|---|---|---|
| Target | scene/object insertion 引起的 image radiance transfer | local material directional response/bare `f` | `not-applicable`；不能作为 target/adapter defect evidence |
| Conditioning | object/background/light samples + G-buffer + previous composition | source material state/latent + directions | `not-applicable`；不能把scene lighting藏入material state |
| Architecture | hypernetwork generates UV/decoder weights + neural texture | compact fixed evaluator MLP | 只有“prepare生成bounded state”可做独立candidate；不是忠实NVIDIA结构 |
| Output split | diffuse/specular/shadow/indirect scene components | local scattering evaluator | 只能做training auxiliary/analytic residual候选，不能更改复现identity |
| Runtime | per-object full-image passes，object数线性；数百ms/1024² PyTorch | fixed reads/small MLP/single query | cost domain不同；不能直接比较time/Pareto |
| Sampling | scene data中的light/object importance samples；无BSDF sampler | evaluator匹配 `sample()/pdf()` | `not-applicable`；不能视作matched sampler prior art |
| Generalization |同一object在lighting/material/context变化，novel scene会退化 | unseen material/source states | 不可互相替代 |

因此 NeLT 不新增当前 NVIDIA reproduction 的 `suspected-defect`。若引入 hypernetwork、component heads、object/sample set encoder或canonical coordinates，必须注册为新 candidate，用 current frozen source/query split、matched budget和bootstrap CI单独比较。[N/I]

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：prepare-time hypernetwork生成固定小 evaluator state，可在不增加per-query cost时提高state-conditioned material quality | NeLT将global reps编成UV/decoder weights和texture，并相对pixelwise global decode声称更快；exact成本未闭合[P §4.2] | native material state在同一shading point可复用，且generated state可静态有界 | fixed MLP concat-latent vs same per-query MAC/read budget的generated-weight MLP；prepare/state bytes单独计 | source/query recipe、backbone per-query depth/width、precision、training steps/seeds、state-update frequency | G1/G2/G2s、single-query median/p90、prepare latency、state bytes/read count、CI | product candidate only if hard-bounded | matched total cost无Pareto收益，prepare/bytes破预算，或generated weights跨state不稳定 |
| H2：按物理角色用typed target/transform，比统一RGB residual更易保留sharp/low-energy结构 | full NeLT优于 w/o GI decomposition，且direct shadow ratio/indirect residual使用不同 algebra[P Eq.3、Table6] | current reference能无歧义地产生不改变native semantics的分量或analytic core residual | unified-output baseline vs same final evaluator、same params/MAC的training-only typed heads/targets；部署仍输出bare `f` | source semantics、query split、loss total scale、optimizer、steps/seeds、final export | local f error、grazing/peak/low-energy bins、reciprocity/energy、seed variance、runtime parity | training-only auxiliary or bounded residual candidate | aggregate或hard-strata无改善，final semantics变化，或component label不能由reference权威定义 |
| H3：canonical local coordinates降低rigid/source-frame variation，而不损害anisotropic orientation semantics | NeLT把background/light/query变到object frame支持rigid transforms[P §4.1.3] | source family有权威local/tangent frame，变换不会抹掉原生orientation参数 | raw/global coordinates vs canonical coordinates；same model/query/training work | native parameterization、frame convention、hemisphere mapping、seeds、budget | G2 transform strata、anisotropy orientation error、quality/time CI | query representation candidate | canonicalization无G2收益、引入frame discontinuity，或破坏anisotropic/editable semantics |
| H4：AE式error-guided online query选择可改善NeLT型hard high-frequency state，但需与representation改动分离 | NeLT明确没有用AE active training，并把MCMC combination列future work；现有AE报告给完整训练机制[P §6.1.1、§7; N-related] | material online reference query也存在稀有high-error states且可不改变test distribution | uniform、reuse-only、error-guided active三组；同reference query count/renderer work/model/steps | proposal、history/replay、selection checkpoint、state/query bounds、test set、seeds | held-out G1/G2/G2s、coverage、training seed variance、reference walltime、bias/reweight诊断 | training query recipe | improvement只来自更多reference work，coverage下降/held-out bias，或matched budget无CI优势 |
| H5：compile-time set encoder可把variable-size native resources变成fixed bounded state | NeLT把light/foreground samples共享编码并聚合成fixed representations[P §4.1] | native source graph/resources可按原生语义采样/编码，且聚合不丢关键ordering/topology | direct fixed descriptor vs set encoder；same state bytes/evaluator/read budget；保留原source reference | source family、resource sampling cap、ordering labels、query recipe、training work/seeds | unseen material/state G2、resource-size strata、bytes/prepare/query time、CI | compiler/prepare candidate | 对layer/order/topology敏感source明显退化，或需要unbounded samples/state才有效 |

H1–H5 都是新候选或training diagnostic，不是 NeLT reproduction，也不修改当前 NVIDIA method identity。所有论文 observed time/quality 只作为研究证据，不成为项目 hard gate。[N research-execution]

## 16. 证据索引

### `P` Main paper

- Page 1、Abstract、Fig.1：object-oriented NeLT、动态编辑轴、输入/输出 broad claim。
- Page 2、Table 1、§1：CNSR/AE/Deep Shading/denoising taxonomy、贡献与“非成熟GI solution”边界。
- Page 3、Table 2、§2：符号表、related methods。
- Page 4–5、Eq.(1–5)、Fig.2–3：transfer definition、direct ratio、indirect residual、scene composition、order dependence、composite object。
- Page 6、Fig.4、Eq.(6)、§4.1：background/foreground/light representations、center-visible assumption、sample aggregation。
- Page 7–8、Fig.5–7、Eq.(7–8)、Table 3、footnote 1：object transform、hypernetwork/neural texture/UV/fetch/decoder pipeline、associated features、diffuse/specular/shadow split。
- Page 8–9、Fig.8、§5、Eq.(9)、footnote 2：四数据集、6000/100 scenes、8 views、sampling/SPP、HDR transform、loss、Adam/200k/60h。
- Page 9–10、Table 4、§6.1：baseline adaptation、AE uniform、timing exclusions、quantitative comparison、author analysis。
- Page 11、Table 5、Eq.(10)、§6.2–6.4：dynamics、unseen environment/Room generalization、two ablations。
- Page 12–14、Fig.9–12、Table 6–7、§7：visual comparison、material components、composite-object tradeoff、future work。
- Page 15、Fig.13、§8：specular shadow/high-frequency/novel-scene failure和结论。
- Page 15–16、References：AE完整题名/DOI身份、收稿修订日期。
- 16/16 页已从 hash-locked PDF 以 150 dpi PNG 渲染，并逐页视觉核对公式、表格、图注、脚注和页码；文本提取仅辅助检索。

### `S` Supplemental/video

- Main paper明确把 exact network structures、insertion-order comparison、更多 dynamics、AE/noisy images 和 tiny-cuda-nn video evidence 指向 supplementary。
- 本次没有取得文件或 hash，故没有引用任何 S 中的具体数字/配置。

### `C` Official code/config/data

- DOI/Crossref与两位作者 publication entries 没有给 official code/config/data locator。
- 没有 Git 网络探测、clone、SSH/token或登录；没有用非官方代码回填结构。

### `A` Author material

- Zhizhen Wu entry确认 paper/作者与摘要；其 `No.162` 与正式 Article 163冲突。
- Zhihua Zhong entry只给 ACM paper link，没有 project/code/supplement link。

### `N` NeuralShading evidence

- `diolatzis-2022-active-exploration-neural-gi.md`：完整 AE 方法、supplement/code、MCMC/reuse/adaptive-resolution evidence；用于证明 NeLT Table4只测 `AE (uniform)`。
- `granskog-2020-compositional-neural-scene-representations.md`：CNSR three-observation/global-latent/G-buffer pipeline；用于解释 NeLT baseline identity，未替代 NeLT正文事实。
- `docs/contracts/scattering_backend.md`：bare `f`、fixed reads、matched `sample()/pdf()`；用于 scene/local boundary。
- `docs/research/experiment_framework.md`、research-execution spec：matched controls、source/query freeze、CI和cost scope。
- NVIDIA correspondence：当前 reproduction identity，只用于 §14。

### `I` Derived/transfer notes

- “容量分布在object weights、sample/observation reps、generated state、neural texture、G-buffer和composition state”是本报告分析。
- “AE comparison不能裁决完整 Active Exploration”“20–50fps不能换算single material query”“hypernetwork只能作为prepare-bounded candidate”均由 P/N 对照得出。
- 所有未取得的 layer/config/runtime detail 均保持 evidence gap，没有用后续 LightFormer/Dual-Band/SDF 反推 NeLT。

## Evidence review

```text
author_worker: /root/nelt_full_report
reviewer: /root/belcour2018_review
reviewed_at: 2026-08-29
sources_rechecked:
  - hash-locked formal main PDF 56566C3D92F29EF6DF6D4B424DECB18CE1BFB9CF2582EC275D4213E028CA5ED5; independently rendered and visually checked 16/16 pages
  - Eq.1-10, Fig.4-7 and Fig.13, Table1-7, captions and footnotes
  - DOI/Crossref/DBLP identity plus Zhizhen Wu and Zhihua Zhong first-party publication entries
  - evidence-reviewed Active Exploration and CNSR reports for cross-paper identity only
findings_closed:
  - normalized supplemental/code front-matter states to the report schema and advanced the independently reviewed report to evidence-reviewed
  - clarified that direct and indirect networks each perform their own texture fetch while direct diffuse/specular decoders share one local feature
  - recorded the unreported zero/near-zero denominator policy for the direct background ratio instead of assuming an epsilon
  - restored the four missing 4-spp WSKP/AFGSA rows from Table 5
  - rechecked all Table4-7 values, cost exclusions, AE(uniform) identity, foreground/background measures and Fig.13 failure classification
remaining_evidence_gaps:
  - supplementary PDF/video not obtained despite explicit main-paper references
  - no official code/config/data locator or commit at audited first-party entries
  - exact network/latent/neural-texture topology, ratio stabilization, state bytes, representation-build cost and reproducible training config remain unclosed
review_status: evidence-reviewed
```

### 完成检查

- [x] main paper 已完整阅读，16/16页关键公式/图/表/图注/脚注已视觉核对；
- [x] supplemental/appendix/勘误的可用性已检查，并将“正文明确提及但未取得”与“不存在”分开；
- [x] official code/config/data 的一手入口可用性已检查；未发现可审计 locator，不以猜测补结构；
- [x] architecture、training、runtime 和主要结果均按正文恢复；正文明确转交 supplementary 的逐层细节保留缺口；
- [x] AE、CNSR、denoiser、Deep Shading 的任务/输入/修改与 timing scope 已分开；
- [x] 正式消融、composite tradeoff、generalization failure和Fig.13失败均已分类；
- [x] paper/source gap、author-page冲突和原文 typo/ambiguity 已保留；
- [x] `I` 分析晚于 `P/S/C/A` 事实，没有把 future work写成成功尝试；
- [x] NVIDIA影响只作scene/local boundary与candidate inspiration，不新增伪fidelity defect；
- [x] 迁移假设包含matched control、冻结轴、成本、runtime class与证伪条件；
- [x] 独立 reviewer 已重读16页PDF并复核公式、baseline identity、Table 1–7数值、cost scope和来源缺口。
