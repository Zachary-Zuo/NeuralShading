---
paper_id: "guo-2022-neural-light-probes"
title: "Efficient Light Probes for Real-time Global Illumination"
authors: "Jie Guo, Zijing Zong, Yadong Song, Xihao Fu, Chengzhi Tao, Yanwen Guo, Ling-Qi Yan"
year: "2022"
venue: "ACM Transactions on Graphics 41(4), Article 202"
doi: "10.1145/3550454.3555452"
report_status: "evidence-reviewed"
main_source: "https://sites.cs.ucsb.edu/~lingqi/publications/paper_nlp.pdf"
supplemental_status: "unavailable"
official_code_status: "unavailable"
official_code_commit: "not-applicable"
author_worker: "/root/lightprobes2022"
reviewer: "/root/lightprobes2022_review"
last_verified: "2026-08-29"
---

# Efficient Light Probes for Real-time Global Illumination

## 1. 研究对象与报告边界

这篇论文研究的是**静态场景、整帧图像级的 probe-based global illumination**，不是局部 neural material evaluator。方法把 diffuse transport 烘焙进 lightmap，把包含多次 glossy bounce 的 incident radiance 烘焙进规则三维网格上的 light probes；运行时先把相邻 probes 的内容以有界的 screen-space reflection search 重投影到当前视点，再用每场景独立训练的 encoder-decoder 从低采样率 lightmap/probe 结果和 G-buffers 重建最终 1080p 图像。[P Abstract, §1, §3–4]

报告覆盖 TOG 41(4) Article 202 的 14 页作者公开版。正式标题是 *Efficient Light Probes for Real-time Global Illumination*；本报告不以任务检索时使用过的“Neural Light Probes”替代正式标题。[P p.1]

本文的 neural component 处于 deferred image reconstruction 阶段：输入含当前视点的 depth、normal、diffuse albedo、reflected albedo 和已经重投影/混合的 lighting image，输出是整张最终图像。它不输出 `f(wo,wi)`、局部 transport kernel、重要性采样 proposal 或 `pdf`，也没有跨场景通用模型。故它是本任务的 `scene-transport` 论文；与 local neural material 的联系只在“如何把昂贵 reference 工作前移到 bake、怎样以物理 buffer 条件化小网络、怎样做时间/空间摊销”这些机制层面。[P §4.4, §6][I]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | [作者 PDF](https://sites.cs.ucsb.edu/~lingqi/publications/paper_nlp.pdf)，TOG 41(4), Article 202，14 pages；DOI `10.1145/3550454.3555452` | 2026-08-29 | SHA-256 `F189B422F6B4ED17CB29436414FB0AD8B149075CCD56EC9BD3B8B8652B275DD3`；67,623,680 bytes | 本报告全部方法、公式、配置、表格、消融和限制事实。14 页均已渲染；重点视觉核对 Fig. 3–16、Eq. 1–9、Table 1–3。 |
| Supplemental `S` | 正文称四场景 animation sequences 与小高光闪烁出现在 supplemental video；未找到独立 supplemental 文档 | 2026-08-29 | unavailable | 现有本地 `video.mp4` 是未完成且容器未验证的部分下载，不作为证据；正文对视频内容的描述仍只标 `P`。 |
| Official code/config/data `C` | 作者 publication entry 只暴露 paper/video 入口，未提供 code、config、checkpoint、dataset manifest | 2026-08-29 | unavailable | 无法审计 gradient update、网络实现、训练超参、TensorRT export 或 baseline 配置。 |
| Author page/video `A` | [Ling-Qi Yan publication list](https://lingqiyan.github.io/) 中的论文与视频入口 | 2026-08-29 | not-applicable | 只用于确认作者公开入口与 video availability；未用未验证视频补写方法或数值。 |
| NeuralShading evidence `N` | `docs/realtime_material_compilation.md`；`docs/research/experiment_framework.md`；`docs/research/model_candidates.md` | 2026-08-29 | 当前 workspace | 只用于第 13–15 节的接口边界、NVIDIA correspondence 与可证伪假设。 |

PDF 的双栏文本提取会把图中数字与相邻正文交错；本报告的 Fig. 9 网络拓扑、Table 1–3 和 Eq. 3–4 均回到渲染页视觉核对。没有尝试通过登录、SSH、Git 凭据或非公开来源补全材料。

## 3. 原论文的问题、假设与贡献边界

作者以实时复现 physically based GI 为目标，尤其关注传统 irradiance/environment probes 难以覆盖的 long glossy paths。GPR 可以处理这类路径，但依赖 2048 spp 的高质量 lightmaps/probes，bake 很慢；其局部 two-level search 与 filter footprint 估计还可能产生 highlight glossiness 偏差、parallax error 和物体边缘的 specular geometric aliasing。[P §1, §4.3, §5.1]

论文的方法假设是：

1. 场景几何、材质和照明可以在 precomputation 阶段固定，并为每个场景烘焙 lightmap、规则网格 probes 与训练图像；[P §3–4, §6]
2. probe radiance 即使只有 128 spp、lightmap 即使只有 256 spp，仍保留了足够的 transport structure；scene-specific CNN 可以在 clean G-buffers 引导下去噪、修复 aliasing 和融合 probe；[P §1, §4.1–4.4]
3. glossy shading point 对应的 parallax-correct probe lookup 可以通过当前视点 screen space 中的迭代几何搜索得到，而不必把搜索限制在 GPR 的局部 footprint；[P §4.3]
4. 历史帧重投影与 temporal loss 能抑制 search/network 带来的时序闪烁。[P §4.4, Fig. 15]

作者列出的贡献包括：低质量 probe/lightmap 驱动的实时完整 pipeline；gradient-based reflection search；带 G-buffer modulation 的轻量网络；以及在四个复杂 glossy 场景上 1080p、超过 30 FPS 的展示。[P p.2 contributions, §5]

“solve the rendering equation”“every physically-based shading effect”与“full global illumination”是作者的目标性表述；正式实验覆盖的是四个静态室内场景、预烘焙 diffuse 与 glossy transport，并没有验证动态几何、动态照明、参与介质或局部可组合 transport operator。[P §3, §5–6]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Scene/source input | 固定场景；低 spp diffuse lightmap；规则 3D grid 上的 glossy probes；每场景训练数据 | scene-specific | [P §3–4, §6] |
| Per-probe payload | glossy-only path-traced incident radiance；把所有表面当 perfect mirrors 时的一次反射 reflected world position；从 probe 直接可见 shading point 的 material ID | 每个 probe 一组 `360°` panorama；正式 probe resolution `1024×512` | [P §3, §4.2, Fig. 3] |
| Runtime view query | 当前相机/视点 `o`，当前帧每 pixel 的 shading point 与 G-buffers；取包围视点的 `K=8` 个最近 probes | 1920×1080 screen-space whole-viewport search | [P Eq. 1–2, §4.3, §5] |
| 坐标 | world-space probe position `p_k`、shading/reflected positions `x,z`；probe panorama coordinates；当前视点 screen coordinates；tangent-plane virtual image `z'` | world + panorama + screen space | [P Fig. 4–5, §4.3] |
| Neural input image | `I = I_d + I_g`，其中 `I_d` 来自 diffuse lightmap，`I_g=Σ_k W_k I_{g,k}` 来自 8 个 reprojected glossy images | per-frame image | [P Eq. 2, Eq. 4, Fig. 9] |
| Neural conditioning | depth `D`、normal `N`、diffuse albedo `A`、glossy 区域一次反射后几何的 reflected albedo `B`；另有 motion vectors 用于 temporal warp | per-frame G-buffers/history | [P §4.4, Eq. 5, Eq. 8] |
| Output | network reconstruction `I_o`，以 path-traced ground-truth image `Î` 监督 | 1920×1080 image；颜色空间、线性/显示域和 exposure 未报告 | [P Eq. 2, §4.4–5] |
| 有效域 | 预计算静态场景；screen-space 候选必须留在 viewport、material ID 连续且通过 probe visibility test | novel camera views within trained/baked scene coverage | [P §4.3, §6] |

运行时可编辑轴实质上是 camera/viewpoint；radiance 已烘焙到 lightmap/probes，正文没有提供在不重烘焙且不重训时改变几何、材质或照明的接口。[P §3–4][I]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

预计算阶段：

1. 用 modified Mitsuba 生成 `2048×2048`、256 spp 的 diffuse lightmap；[P §5]
2. 在规则三维网格的 probe locations 生成 `1024×512`、128 spp 的 360° probe panoramas；每个 probe 存 glossy incident radiance、one-bounce mirror reflected position 与 material ID；[P §4.2, §5, Fig. 3]
3. 对一个场景渲染覆盖场景视角的 20 张 ground-truth images，并为该场景单独训练网络；这些训练 views 不再用于 inference；[P §4.4]
4. 训练时间与 GT generation 都算进 precomputation。[P §4.4, §5.4, Table 2]

运行时：

1. lightmap query 生成 `I_d`；[P Fig. 2, §4.1]
2. 对 `K=8` 个 nearest probes 分别执行 `I_{g,k}=P{o,p_k,M_k}`，得到 parallax-corrected glossy images；[P Eq. 1, §4.3]
3. 以 Eq. 4 的权重得到 `I_g=Σ_k W_k I_{g,k}`；[P Eq. 4]
4. rasterize 当前帧 `D,N,A,B`，组合低质量输入 `I=I_d+I_g`，并结合 temporal reprojection；[P Eq. 2, §4.4]
5. TensorRT 执行 `I_o=Φ_ζ(I,D,N,A,B)`，输出 final frame。[P Eq. 2, §4.4–5]

### 5.2 Probe representation 与 reflection search

对某个 glossy pixel，当前相机 `o` 看到的 primary ray 在 glossy point `x` 反射后命中 `z`。算法先以 `x` 的 tangent plane 构造 `z` 的局部 mirror virtual image `z'`，再把 probe position `p` 到 `z'` 的直线与 tangent plane 相交，得到第一候选 `x_1`。平面时作者认为第一候选就是最优；曲面时从 `x_1` 沿其 gradient 继续迭代。[P Fig. 4–5, §4.3]

第 `k` 个候选的 correctness/confidence 写成：

```text
C(x_k) = ((→px_k + →zx_k) / ||→px_k + →zx_k||) · N(x_k) .
```

当它达到 `T_max`，或迭代达到 `N_max` 时停止。正式参数为 `T_max=0.999`、`N_max=20`。[P Eq. 3, §4.3, §5]

两个 special cases：

- 连续候选落在不同 material IDs 时，不跨物体接受候选，而是沿原 gradient 连续缩短 step，直到 material ID 一致；
- candidate projection 离开 viewport 时采用相同的 shortened search。[P Fig. 5, §4.3]

visibility check 使用 probe 中的 reflected-position panorama：按找到的 probe coordinate 查询位置，与目标 `z` 比较；距离大于 `D_max=0.1` 时丢弃该候选，否则会产生 Fig. 7 的 ghosting。`D_max` 的 scene-unit 归一化没有说明。[P §4.3, Fig. 7, §5]

八张图的 published blend 为：

```text
I_g = Σ_{k=1..8} W_k I_{g,k},
W_k = exp(-τ C_k) / Σ_{j=1..8} exp(-τ C_j),   τ=100.
```

[P Eq. 4, §5] Eq. 3/正文把较接近 `T_max=0.999` 的 `C(x_k)` 称为更优 correctness/confidence，但 Eq. 4 在 `τ>0` 时给更大的 `C_k` 更小的权重。正文没有说明 blend 中的 `C_k` 是否先被变换为 error（例如 `1-C`），也没有给被 visibility rejection 的 probe 和 all-invalid pixel 如何重新归一化；该记号缺口不能自行修正。[P Eq. 3–4, §4.3]

### 5.3 持久化表示

| 资产 | 论文配置 | shared/per-scene | locator |
|---|---|---|---|
| Diffuse lightmap | `2048×2048`，256 spp | per-scene | [P §5] |
| Probe grid | 通常 256 probes，规则 3D grid | per-scene | [P §4.2, §5, Fig. 13] |
| Probe radiance | glossy-only path traced panorama，128 spp、`1024×512` | per-probe | [P §4.2, §5] |
| Reflected position | perfect-mirror one-bounce world position panorama | per-probe | [P §4.2, Fig. 3] |
| Material ID | probe 中直接可见 shading point 的 ID panorama | per-probe | [P §3, §4.2] |
| Network weights | 每场景独立训练的 `Φ_ζ` | per-scene | [P §4.4, §6] |
| History | temporal reprojection 累积 two consecutive frames | per-view/runtime | [P §4.4] |

probe texture format、radiance/position/ID precision、compression、mip、GPU bytes、network parameter bytes 均未报告；不能只凭 resolution 推算正式 storage。

### 5.4 网络逐层配置

Fig. 9 是唯一逐层图。它显示两个 encoder branches、三个 channel scales `24→32→64`、三处 `GM(n)`、对称 decoder 与 skip connections：

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Image encoder，scale 24 | `I=I_d+I_g` | `3×3,s1` convolution → `3×3,s1` convolution → `GM(24)` | 第一条为图例所示 convolution with LReLU；第二条普通 convolution；normalization 未报告 | 24 channels | per-scene network | [P Fig. 9, §4.4] |
| Image encoder，scale 32 | 24-channel features | `3×3,s2` convolution → `3×3,s1` convolution → `GM(32)` | downsampling convolution with LReLU；第二条普通 convolution | 32 channels | per-scene network | [P Fig. 9, §4.4] |
| Image encoder，scale 64 | 32-channel features | `3×3,s2` convolution → `3×3,s1` convolution → `GM(64)` | downsampling convolution with LReLU；第二条普通 convolution | 64 channels | per-scene network | [P Fig. 9, §4.4] |
| G-buffer encoder | packed `D,N,A,B` | `3×3,s1` → `3×3,s2` → `3×3,s2` | Fig. 9 均标为 convolution with LReLU | 24、32、64 channels | per-scene network | [P Fig. 9, §4.4] |
| `GM(n)` | image feature `F_i`、同尺度 G-buffer feature `F_G` | `F_o=Conv1(F_G)⊗F_i⊕Conv2(F_G)`；两条 `1×1,s1` convolutions | GM 内 activation/normalization 未报告 | `n∈{24,32,64}` | per-scene network | [P Eq. 5, Fig. 9] |
| Decoder | deepest 64-channel feature + mirrored skips | 三个 scale groups；Fig. 9 显示 `64/32/24` 各三个 feature maps，scale 间 `×2` upsampling，组内 `3×3,s1` transposed convolutions，最后按图例为 `1×1,s1` transposed convolution | 每个 transposed convolution 后 LeakyReLU；output activation、normalization 未报告 | final image；明确 output channel count 未标 | per-scene network | [P Fig. 9, §4.4] |

正文说 all convolution operations in both encoder branches 使用 `3×3` kernel，stride-2 convolution 降采样；GM 是明确例外，使用两条 `1×1` convolution。decoder 的 skip connection 连接 mirrored layers。图和正文都没有给 input channel packing、bias、padding、negative slope、parameter count、normalization、weight precision 或 TensorRT layer fusion。[P §4.4, Fig. 9]

### 5.5 条件化与 temporal reprojection

GM 受 SPADE 启发，不直接 concatenate 两分支，而是让 clean G-buffer feature 产生 pixel-wise scale 与 bias，调制 noisy image feature。它是 screen-space conditional affine modulation，不是 material latent compiler 或物理 BRDF prior。[P Eq. 5, §4.4]

temporal reprojection 用 rendered motion vectors 把 previous frames 累积到 current input。正文原句为“accumulate two consecutive frames in the input”，两个 decay factors 分别为 `0.3` 和 `0.1`；它没有明确这两个数对应 current+previous，还是 two history frames，也没有披露 disocclusion rejection、clamping、history validity 或 tensor packing。[P §4.4]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Scenes | Bathroom、Kitchen、Livingroom、Staircase，来自 GPR/Rodriguez et al. 2020；均含复杂 glossy surfaces | [P §5] |
| Per-scene training views | 20 张 path-traced ground-truth images，覆盖 whole scene 的 wide range of views；这些 views 不在 inference 使用 | [P §4.4] |
| Runtime/test views | novel camera views/animation sequences；Table 1 每个 scene 对 20 frames 取平均 | [P §5, Table 1] |
| Diffuse input | low-quality lightmap，256 spp、`2048×2048` | [P §5] |
| Glossy input | 256 probes（通常配置），128 spp、`1024×512`，glossy paths only | [P §4.2, §5] |
| Ground truth | path-traced final images；exact renderer settings、spp、bounce/termination、BRDF 和 color pipeline 未报告 | [P §4.4–5] |
| G-buffers | current-view depth、normal、diffuse albedo、one-bounce reflected albedo；motion vectors用于 temporal warp | [P §4.4] |
| Search recipe | `K=8` nearest probes；whole viewport；`T_max=.999, N_max=20, D_max=.1, τ=100` | [P §4.3, §5] |
| Filtering/LOD | lightmap/probe texture filtering、mip/footprint、anti-aliasing recipe 未报告；network 负责修复噪声/aliasing | 未报告 |
| Split | 20 training views 与 inference views 隔离；validation set、camera sampling、20-frame evaluation 与训练场景内容的精确隔离规则未报告 | [P §4.4, §5.2] |
| Generation mode | lightmaps/probes/GT offline；current-view G-buffers/search/network online | [P §4–5] |

modified Mitsuba 用于 precomputation 的 lightmap/probe data；real-time stage 建在 SIBR、OpenGL 与 TensorRT 上。正文没有明确 GT 与所有 baseline 是否共享完全相同的 integrator、tone mapping 和 exposure。[P §5]

temporal loss 要求 consecutive frames 与 motion vectors，但训练数据章节只说 20 ground-truth images；sequence formation、每个 epoch 的 frame pairs、camera trajectory 和 history bootstrap 均未报告。[P §4.4]

## 7. Loss、optimizer 与训练 lifecycle

网络使用三项 loss：

```text
L_L1    = ||Î - I_o||_1
L_LPIPS = Σ_l ||ω_l ⊙ (F_l(Î) - F_l(I_o))||_2^2
L_T     = ||I_o - W(I'_o, V)||_1
L       = 0.8 L_L1 + 0.03 L_LPIPS + 0.15 L_T.
```

LPIPS 使用 pretrained VGG-16 feature network 和 LPIPS 预学习的 channel weights；temporal term 把另一连续帧 `I'_o` 以 motion vector `V` warp 后比较。[P Eq. 6–9, §4.4]

| 项 | 正式配置 | locator |
|---|---|---|
| Target/output transform | §6 说明 HDR inputs 使用 logarithm compression，但正文没有给公式、归一化、inverse transform 或 loss 所在域；output activation 未报告 | [P §6] |
| Loss | `λ_L1=.8`、`λ_LPIPS=.03`、`λ_T=.15` | [P Eq. 6–9] |
| Perceptual network | pretrained VGG-16；layer set、input normalization 未报告 | [P Eq. 7] |
| Optimizer | Adam；`lr, β_1, β_2, ε, weight_decay` 未报告 | [P §4.4] |
| LR schedule | 未报告 | 未报告 |
| Epochs | 1000 epochs，作者称通常收敛 | [P §4.4] |
| Batch/patch/query count | 未报告；20 GT views 如何 crop/sample、每 epoch samples 数未报告 | 未报告 |
| Initialization/seed/model selection | 未报告；没有 validation/checkpoint selection/repeated seeds | 未报告 |
| Training identity | 每个 scene 独立训练一个 model | [P §4.4, §6] |
| Training time | 约 4 h/scene；Table 2 为 3.9 h/scene | [P §4.4, Table 2] |
| Training hardware | §5.4 称 GT generation + training 的约 12 h overhead 使用 RTX 3090Ti；CPU/GPU work 的精确拆分未报告 | [P §5.4] |

`0.8+0.03+0.15=0.98` 是论文给出的原始权重组合；正文没有说明是否再归一化，不能自行把它改为和为 1。

## 8. Inference、部署与成本

### 8.1 Runtime path

gradient-based reflection search、`D/N/A/B` 先写入 OpenGL buffers，再映射成 CUDA tensors；作者称 mapping cost 相对 whole pipeline negligible。network 由 TensorRT inference。runtime backend 是 SIBR + OpenGL + CUDA/TensorRT。[P §4.4–5]

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Frame resolution | `1920×1080` | [P §5] |
| Search bounds | `K=8`；基本候选迭代以 `N_max=20` 停止；material-ID/viewport special cases 会连续缩短 step，但其子搜索次数是否计入 `N_max`、是否另有 cap 均未报告；另有 visibility test | [P §4.3, §5] |
| Raster/raycast | Table 3：raster `0.4–0.6 ms`，raycast `4 ms` | [P Table 3] |
| Ours search | `4.4–6 ms`/frame | [P Table 3] |
| Network inference | `14.6 ms`/1080p frame，四场景相同表值 | [P Table 3] |
| Total | Bathroom `25.1 ms`、Kitchen `25.2 ms`、Livingroom `25.2 ms`、Staircase `23.6 ms` | [P Table 3] |
| Hardware | 方法总览：Intel i7-8700K + RTX 3090Ti；Table 3：Intel i7-6900K + RTX 3090Ti | [P §5, Table 3 caption] |
| Precision/quantization | 未报告 | 未报告 |
| Parameters/MAC/FLOP/model bytes | 未报告 | 未报告 |
| Probe/lightmap/state bytes | 未报告；texture formats/precision 未披露 | 未报告 |
| Texture/feature reads | 最多 8 probes、20 search iterations，但每 iteration 的 exact buffer reads、cache/coherence 未报告 | [P §4.3][I] |
| History | two consecutive frames，decays `.3/.1`；history bytes/rejection 未报告 | [P §4.4] |

作者报告 whole pipeline `>30 FPS`。Table 3 的 `23.6–25.2 ms` 与此一致，但这是 RTX 3090Ti、固定 1080p、四个场景上的整帧 timing，不是 material-query latency，也没有 warmup、重复次数、median/p90 或 TensorRT precision。[P Abstract, §5.4, Table 3]

### 8.2 Precomputation cost

Table 2 的机器是 four Intel Xeon Gold 5118 CPUs、128 GB RAM；§5.4 另说明 GT generation + network training 使用 RTX 3090Ti。论文没有把各列对应的 CPU/GPU device 写成完整 job graph。[P Table 2 caption, §5.4]

| Scene | Method | Lightmap | Probes | GT | Train | Total |
|---|---|---:|---:|---:|---:|---:|
| Bathroom | GPR | 2.9 h | 18.3 h | – | – | 21.2 h |
| Bathroom | Ours | 0.08 h | 0.8 h | 8.2 h | 3.9 h | 13.0 h |
| Kitchen | GPR | 1.5 h | 16.5 h | – | – | 18.0 h |
| Kitchen | Ours | 0.04 h | 0.6 h | 8.1 h | 3.9 h | 12.6 h |
| Livingroom | GPR | 3.1 h | 17.1 h | – | – | 20.2 h |
| Livingroom | Ours | 0.09 h | 0.9 h | 7.3 h | 3.9 h | 12.2 h |
| Staircase | GPR | 2.3 h | 19.2 h | – | – | 21.5 h |
| Staircase | Ours | 0.06 h | 0.8 h | 6.7 h | 3.9 h | 11.5 h |

[P Table 2] Low-spp lightmap+probe 本身均低于 1 h/scene，但要得到每场景 neural model，还需 6.7–8.2 h GT generation 与 3.9 h training。因而论文展示的优势是 total precomputation 从 GPR 的 `18.0–21.5 h` 降为 `11.5–13.0 h`，不是把可交付 scene bake 整体降到 1 h 以内。[P §5.4, Table 2]

## 9. 实验 protocol、baseline、指标与完整结果

### 9.1 Baseline identities 与公平性边界

| Baseline | 论文中的配置 | 比较边界 | locator |
|---|---|---|---|
| RTRT/Falcor | precomputed diffuse lightmap 2048 spp；实时预算只允许 glossy interreflection 到 3 bounces | 理论上可更长路径，但本文配置受 frame budget 限制 | [P §5.1, Fig. 10] |
| GPR | lightmaps/probes 2048 spp；two-level local gathering + filtering | 是主要 quality/time/precompute baseline | [P §4.3, §5.1, Table 2–3] |
| LFP | 同 probe count；probe 2048 spp、`1024×1024` | Fig. 11 qualitative/single-view RMSE；高光缺失 | [P §5.1, Fig. 11] |
| ISG | 直接应用于 2048 spp probe data | traditional search/query baseline | [P §5.1, Fig. 12, Table 1] |
| ULR | 直接应用于 2048 spp probe data | image-based rendering baseline | [P §5.1, Fig. 12, Table 1] |
| Ours (Search) | 256 spp lightmap、128 spp probes；只到 gradient search/blend | 隔离 search+低质量缓存，未经过 network | [P Fig. 10, Table 1] |
| Ours (Final) | Ours (Search)+G-buffers+network+temporal components | 完整方法 | [P §4.4–5, Table 1] |

baseline 的 source revisions、Falcor commit、GPR/LFP/ISG/ULR implementation commits、parameter tuning 和精确 renderer parity 均未报告。Table 1 对四场景各取 20 frames 的 RMSE、DSSIM、LPIPS 平均；metric 的颜色域、mask、exposure、DSSIM 定义和 frame selection 未披露。[P §5.2, Table 1]

### 9.2 Table 1 完整数值

| Scene | Method | RMSE↓ | DSSIM↓ | LPIPS↓ |
|---|---|---:|---:|---:|
| Bathroom | ISG | 0.087 | 0.131 | 0.093 |
| Bathroom | ULR | 0.110 | 0.150 | 0.134 |
| Bathroom | RTRT | 0.057 | 0.068 | 0.055 |
| Bathroom | GPR | 0.033 | 0.041 | 0.044 |
| Bathroom | Ours (Search) | 0.043 | 0.093 | 0.139 |
| Bathroom | **Ours (Final)** | **0.022** | **0.029** | **0.030** |
| Kitchen | ISG | 0.078 | 0.132 | 0.092 |
| Kitchen | ULR | 0.060 | 0.117 | 0.103 |
| Kitchen | RTRT | 0.041 | 0.049 | 0.048 |
| Kitchen | GPR | 0.028 | 0.035 | 0.049 |
| Kitchen | Ours (Search) | 0.043 | 0.108 | 0.139 |
| Kitchen | **Ours (Final)** | **0.024** | **0.031** | **0.032** |
| Livingroom | ISG | 0.126 | 0.190 | 0.110 |
| Livingroom | ULR | 0.099 | 0.195 | 0.107 |
| Livingroom | RTRT | 0.055 | 0.081 | 0.062 |
| Livingroom | GPR | 0.034 | 0.044 | 0.047 |
| Livingroom | Ours (Search) | 0.044 | 0.078 | 0.110 |
| Livingroom | **Ours (Final)** | **0.022** | **0.022** | **0.020** |
| Staircase | ISG | 0.069 | 0.093 | 0.064 |
| Staircase | ULR | 0.054 | 0.083 | 0.061 |
| Staircase | RTRT | 0.053 | 0.074 | 0.051 |
| Staircase | GPR | 0.020 | 0.021 | 0.030 |
| Staircase | Ours (Search) | 0.030 | 0.060 | 0.100 |
| Staircase | **Ours (Final)** | **0.014** | **0.013** | **0.008** |

[P Table 1] `author-positive`：Ours (Final) 在这张表的四个场景、三个指标上均为最低值。该结论只适用于本文 20-frame/scene、固定场景与未公开 color/metric pipeline；没有 multi-seed confidence interval。[P §5.2, Table 1]

`Ours (Search)` 的 RMSE/DSSIM 通常优于 ISG/ULR，但 LPIPS 并不一致：Bathroom/Kitchen/Staircase 的 `0.139/0.139/0.100` 都高于相应 ISG/ULR，Livingroom 的 `0.110` 只与 ISG 持平且略高于 ULR `0.107`。所以作者“search beat some traditional methods”的表述可由 RMSE/DSSIM 与 Fig. 12 支持，不能扩写成所有指标均胜。[P §5.2, Table 1]

### 9.3 Search 与 runtime 对照

Fig. 8 在 Kitchen 的一个 view、相同高采样率 probe input 上报告：Ours search `6 ms / RMSE 0.026`；GPR gathering `14 ms / 0.028`；GPR filtering `7 ms / 0.027`，硬件 RTX 3090Ti。它支持 search 在该案例更快且 RMSE 略低，不提供跨场景统计。[P Fig. 8]

Table 3：

| Scene | Method | Raster | Raycast | Search | Infer | Total |
|---|---|---:|---:|---:|---:|---:|
| Bathroom | GPR | 0.5 ms | 4 ms | 20.5 ms | – | 25.0 ms |
| Bathroom | Ours | 0.5 ms | 4 ms | 6 ms | 14.6 ms | 25.1 ms |
| Kitchen | GPR | 0.6 ms | 4 ms | 21.6 ms | – | 26.2 ms |
| Kitchen | Ours | 0.6 ms | 4 ms | 6 ms | 14.6 ms | 25.2 ms |
| Livingroom | GPR | 0.5 ms | 4 ms | 21.6 ms | – | 26.1 ms |
| Livingroom | Ours | 0.6 ms | 4 ms | 6 ms | 14.6 ms | 25.2 ms |
| Staircase | GPR | 0.4 ms | 4 ms | 19.5 ms | – | 23.9 ms |
| Staircase | Ours | 0.6 ms | 4 ms | 4.4 ms | 14.6 ms | 23.6 ms |

[P Table 3] 完整方法的 total 与 GPR 接近；主要变化是用 `4.4–6 ms` gradient search + `14.6 ms` neural inference 取代 GPR 的 `19.5–21.6 ms` search/filter。论文证明的是在近似相同 frame budget 下质量提高，而不是 whole pipeline latency 显著低于 GPR。[P Table 1, Table 3]

### 9.4 Probe count 与 qualitative evidence

Fig. 13 把 256 probes 与 489 probes 比较；489 probes 保留更多细节，但增加 time/storage。没有给数值质量、额外 bake time、bytes 或 runtime delta。[P Fig. 13, §5.3]

Fig. 10 的四个 single-view RMSE 依次为：

| Scene | RTRT | GPR | Ours (Search) | Ours (Final) |
|---|---:|---:|---:|---:|
| Bathroom | 0.036 | 0.034 | 0.044 | 0.020 |
| Kitchen | 0.034 | 0.027 | 0.040 | 0.026 |
| Livingroom | 0.049 | 0.032 | 0.040 | 0.017 |
| Staircase | 0.047 | 0.021 | 0.029 | 0.012 |

[P Fig. 10] Fig. 11 的两个 LFP examples 为 `0.110/0.129`，Ours 为 `0.019/0.017`；Fig. 12 的两个 examples 分别为 ISG `0.113/0.088`、ULR `0.093/0.061`、Ours (Search) `0.053/0.046`、Ours (Final) `0.020/0.021`。[P Fig. 11–12] 这些都是所示单帧/裁图的 RMSE；Table 1 的 20-frame averages 是更完整的正式数值证据，本报告不以它们替代 Table 1 聚合。

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `author-positive` | 完整 Ours (Final) | Table 1 四场景三指标最低；1080p `23.6–25.2 ms` | accurate search + robust neural reconstruction | 结果依赖 per-scene training、固定 bake 和 image-space context，不能直接归因于 compact local network | [P Table 1, Table 3][I] |
| `author-positive` | gradient search vs GPR gathering/filtering | Fig. 8 为 `6 ms/0.026`，对比 `14 ms/0.028` 与 `7 ms/0.027` | whole-viewport gradient search 快速收敛，避免局部 footprint 偏差 | 单 view、无分布统计 | [P §4.3, Fig. 8][I] |
| `ablation-inferior` | 去掉 visibility check | 出现 ghosting | occluded optimal position 不应从对应 probe 使用 | 只有 qualitative case | [P Fig. 7, §4.3] |
| `ablation-inferior` | 256 probes vs 489 probes | 256 probes 细节较少；489 更好但更贵 | probe density 决定 spatial coverage | 未量化 quality–bytes–time 曲线 | [P Fig. 13, §5.3][I] |
| `ablation-inferior` | GM 改为 simple concatenation | Fig. 14 出现明显 fusion artifacts | GM 更好利用 clean G-buffers、抑制 noisy input 影响 | 没有参数量 matched 或数值指标 | [P Fig. 14, §5.3][I] |
| `ablation-inferior` | 去掉 LPIPS loss | Fig. 14 perceptual detail 较差 | LPIPS 使结果 perceptually better | 没有单独数值或重复实验 | [P Fig. 14, §5.3] |
| `ablation-inferior` | 同时去掉 temporal reprojection 与 temporal loss | 五连续帧的 bin/bottle edges 出现 flickering highlights | 两个 temporal components 提升稳定性 | 联合消融不能区分 inference history 与 training loss 的贡献 | [P Fig. 15, §5.3][I] |
| `author-negative` | network reconstruction of HDR highlights | Bathroom bowl 在 Search 保留的 highlights，经 Final 后亮度降低或消失 | HDR logarithm compression + insufficient G-buffer information | 这是明确的 network failure，不是 probe/search failure | [P §6, Fig. 16][I] |
| `author-negative` | small highlights in animation | 偶发 temporal flickering | 与高频高光重建问题相关；作者建议更复杂 network 作为未来方向 | 已有 temporal 组件仍未完全解决 | [P §6] |
| `known-limitation` | per-scene training | GT+training 增加约 12 h precompute | general cross-scene model 可望把 precompute 降至 <1 h | 这是作者 future work，不是已证明结果 | [P §5.4, §6][I] |
| `known-limitation` | dynamic scene/probe update | 即使 128 spp，glossy probes 仍无法负担动态场景更新 | glossy dynamic probe update 仍是挑战 | runtime 只支持 baked scene state | [P §6][I] |
| `known-limitation` | regular-grid placement | 可能 spatial coverage 不足 | adaptive/learned placement 值得研究 | 正文未做 placement ablation | [P §6] |

在已获得第一方材料中，没有 optimization divergence、多 seed 失败、不同 network depth/width、不同 loss weights、不同 history length、不同 `K/N_max/τ/D_max`、pure network without reflection search 或 general cross-scene model 的正式失败结果。作者没有采用某个变体不等于尝试后失败。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Architecture | Fig. 9 给 24/32/64 scales、kernel/stride、GM、decoder/skip | 无独立文档；video 未用 | 不可得 | 可以重建 stage topology，不能恢复完整 tensor schema、padding、activation slope、parameter/precision/export |
| Gradient search | Fig. 4–5 与 Eq. 3 给几何构造/停止，描述 shortened-step/visibility | 不可得 | 不可得 | 曲面后续 gradient/update equation、step schedule、invalid handling 不足以 faithful implementation |
| Confidence blend | Eq. 3 的 correctness/confidence 以 `.999` 为最优阈值；Eq. 4 使用 `exp(-τC)` | 不可得 | 不可得 | 若 `C` 未变换，较大 confidence 得到较小 weight；正文未解释 `C_k` 是否是 error map，保留为 internal notation gap |
| Data/GT | 4 scenes、20 GT views、20 evaluation frames/scene、low-spp inputs | animation video 未验证 | 不可得 | GT spp、view/split、metric/color pipeline、assets manifest 未报告 |
| Loss/training | Eq. 6–9、Adam、1000 epochs、约 4 h、per-scene | 不可得 | 不可得 | LR/betas/batch/crop/schedule/seed/checkpoint/log transform 未报告 |
| Temporal | two consecutive frames、`.3/.1`，motion warp，joint ablation | 正文称 video 展示 animation | 不可得 | history slot semantics、rejection/clamp/sequence recipe 未报告 |
| Runtime | OpenGL→CUDA→TensorRT、Table 3 breakdown | video 不承载正式 timing | 不可得 | §5 general machine 为 i7-8700K；Table 3 measurement machine 为 i7-6900K，均 RTX 3090Ti；不可混为同一 CPU 配置 |
| Precompute | Table 2 逐场景 CPU-server cost；§5.4 称 GT/train 用 RTX 3090Ti | 不可得 | 不可得 | device/job breakdown 不完整，但 totals 与约 12 h neural overhead 一致 |

没有 official code 不自动构成 `paper-code-gap`；这里的结论是 `code-unavailable`。真正可定位的 gap 是 paper 内部未定义/未对应的配置与 Eq. 3–4 记号关系，而不是猜测代码会怎样实现。

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

作者明确声明：[P §6]

1. HDR logarithm compression 与不足的 G-buffer information 会降低或抹去高频 highlights，并偶发小高光时序闪烁；
2. 每场景单独训练增加 precomputation；general model 是尚未实现的未来方向；
3. 128 spp glossy probe 仍无法支持 dynamic-scene rapid update；DDGI 的动态更新只覆盖 diffuse，不解决这里的 glossy case；
4. regular-grid probe placement 可能 spatial coverage 不足；adaptive 或 joint learned placement 尚未验证。

由方法 domain 直接决定、但应与作者主张分开的边界：[P §3–4][I]

- baked lightmap/probe radiance 与 per-scene model 绑定固定 geometry/material/lighting；
- screen-space search 看不到 viewport 外的信息，虽有 shortened-step fallback，不能等价于 unrestricted ray/path query；
- 输出是 final image，不可按 bounce、方向或局部 surface query 组合，也不提供 matched sampler/pdf；
- temporal history 引入 camera trajectory/history-state dependence。

### 12.2 未报告/材料不可得

- official code、formal config、checkpoint、dataset/camera trajectories 与 verified supplemental；
- ground-truth spp、bounce limits、Russian roulette、BRDF/material parameters、light settings、tone map/exposure/color space；
- gradient 的精确定义、曲面 update equation、shortened-step 初值/衰减/迭代 cap 及其是否计入 `N_max`、all-invalid handling；
- Eq. 4 中 `C_k` 与 Eq. 3 confidence 的变换关系；
- regular grid 的 dimensions/spacing、scene scale 与 `D_max=.1` 单位；
- probe radiance/position/material-ID texture formats、precision、compression、mips、bytes；
- `D/N/A/B` channel packing、normal/depth encoding、motion-vector convention；
- HDR logarithm transform/inverse、network input/output range 和 output activation；
- padding、bias、LeakyReLU slope、initialization、parameter count、MAC/FLOP、TensorRT precision/fusions；
- Adam 超参、learning-rate schedule、batch/patch、temporal pair sampling、seed、validation、model selection；
- timing 的 warmup/repeats/aggregation、GPU clocks/precision、OpenGL-CUDA mapping exact cost；
- baseline code revisions、renderer parity 与 tuning protocol；
- cross-scene、relighting、material editing、dynamic geometry 的结果。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

方法的主要容量分散在四处，而不是一个 compact MLP：

1. per-scene `2048²` lightmap；
2. 256 个 probes 的三类 `1024×512` panorama payload；
3. 当前视点整屏 G-buffers 与 bounded iterative search；
4. 每场景独立的 multi-scale CNN 和 temporal history。

network 本身通道数只有 24/32/64，作者称 light-weight，但没有参数量/bytes；“small network”不能代表 total representation compact。它能以较低 input spp 换取质量，是因为 scene radiance 已在 probes、GT images 和训练 weights 中多重摊销。

### 13.2 成功所依赖的假设

- geometry/material/lighting 固定到 bake identity；
- 20 views 足以覆盖这个 scene 的 screen-space appearance；
- nearby 8 probes 的 panorama 中存在当前 view 所需 transport，search 主要修 parallax，而不是生成缺失路径；
- reflected position/material ID/clean G-buffers 对错误模式提供强条件；
- downstream CNN 可以把 low-spp noise/aliasing 与真实高频 highlight 区分开；Fig. 16 正是这个假设失败的反例；
- 时间连续、motion vector 可靠，history 能复用而不会产生新 ghosting。

因此 Table 1 的成功不是跨场景 transport generalization 证据，而是四个 per-scene bake-and-fit workflows 的结果。

### 13.3 可迁移机制与不能迁移的部分

可迁移机制：

- **先显式解决坐标/visibility，再学习 residual reconstruction**：gradient search 承担 probe↔view parallax correspondence，CNN 只修复低 spp、融合和缺失结构；这比让网络同时猜 correspondence 和 radiance 更容易审计；
- **clean physical buffers 做 affine modulation**：GM 比直接 concat 更接近“由可靠状态调制 noisy signal”的条件化；可以作为 scene-level auxiliary renderer 或未来 environment-integration 模块的候选；
- **把 reference cost 计入完整 bake lifecycle**：low-spp cache 本身 <1 h，但 GT+training 把 total 拉到 11.5–13 h；这提醒 compiler 研究必须报告 source query、fit 与 deployment asset 的总成本；
- **分开 search-only 与 final**：Table 1 的两个 identity 能区分 transport query 与 learned reconstruction 的贡献，这种中间结果审计值得保留；
- **空间/时间摊销**：整帧 CNN 和 history 适合 coherent image workload，说明“实时”可以来自 query batching 与 temporal reuse，而不只来自单次 evaluator 极小。

不能直接迁移：

- image/G-buffer/network 输出不是线性 BSDF `f`；
- probe radiance 已含 scene lighting、visibility 和多 bounce transport，不能当 material latent；
- network 不接受独立 `wo,wi`，不能随机访问方向；
- 没有 `sample()/pdf()`，不能替换 path tracer 的 local scattering program；
- per-scene weights 与 current-view/history state 不满足未见材质/参数状态的 G2/G2s；
- 1080p CNN latency 不能与 shader 中单次 MLP query latency横向排名。

### 13.4 与本项目 runtime contract 的关系

对固定 1080p 和固定 assets，这条 pipeline 的主要 nominal work 有明确边界：`K=8`、基本候选迭代 `N_max=20`、固定-depth CNN、固定 history count；但论文没有说明 special-case shortened-step 子搜索是否计入 `N_max` 或另有 cap，因此现有 `P` 证据不足以证明整条 search 的严格静态上界。它至多是一个**接近 bounded、但实现 correspondence 尚有缺口的 scene-level image program**。本项目目标合同则要求运行成本、状态和读取数可由实现静态确认，并要求 `prepare()` 生成着色点可复用状态，`evaluate(wo,wi)` 随机访问并直接输出线性 `f`，必要时提供 matched `sample/pdf`。[P §4.3][N `.trellis/spec/project/method-constraints.md`；`docs/realtime_material_compilation.md` §开头；`docs/research/experiment_framework.md` §0, §5][I]

因此该方法不应注册为 local neural material candidate。更合适的角色是：未来 scene transport / environment integration 阶段的 load-bearing reference、screen-space postprocess/control，或用于研究“低成本辅助 buffer + learned reconstruction”的 capacity diagnostic。若把它用于本项目 viewer，也必须是独立部署轨道，不能把 final-image error 回写成 local evaluator quality。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

| Correspondence | 状态 | 影响 |
|---|---|---|
| NVIDIA evaluator 直接拟合线性 RGB `f` | `not-applicable` | 本文输出 scene/view/history-dependent final image，不能替换 evaluator target、loss 或 query recipe。[N `docs/research/experiment_framework.md` §0, §2, §5] |
| NVIDIA learned sampler / `sample/pdf` | `not-applicable` | 本文 reflection search 是 probe reprojection，不是 hemisphere proposal，也没有 probability density。 |
| `prepare()` 可复用 view-conditioned state | `interface-adaptation` | “nearest probes + current-view G-buffers + history”可看作 scene renderer 的 prepare，但不能跨多个 `wi` 复用成 local scattering state。[N `docs/realtime_material_compilation.md` §开头] |
| 物理条件 buffer 调制 learned decoder | `interface-adaptation` | GM 提示在未来整帧 environment integration/scene track 中，把 visibility/depth/albedo 等可靠特征用于 modulation；不应无证据地加进当前 compact MLP。 |
| log-domain training | `author-underspecified` | 本文只在 limitation 中提 HDR logarithm compression，未给 transform；不能拿它作为当前 NVIDIA log-L1 的 faithful prior。当前 NVIDIA recipe 明确保持 runtime/target 为线性 `f`。[N `docs/research/experiment_framework.md` §2, §5] |
| 训练/部署成本 | `not-applicable` 于单-query对比 | 本文 `14.6 ms/1080p` 是 coherent TensorRT image inference；当前候选登记 `C_eval/C_prepare/B_asset/state bytes`。两种 cost domain 必须分表。[N `docs/research/experiment_framework.md` §0] |

对当前复现最直接的提醒不是“采用这张 CNN”，而是**保留语义分层**：local `f` 误差、environment integration、screen-space correspondence、temporal reconstruction 是四种不同对象。本文只为后三者提供 scene-specific evidence，不能用 final-image improvement 掩盖 local evaluator 或 sampler 的错误。

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：在未来 scene-probe renderer 中，先做 bounded gradient correspondence 再学习 reconstruction，比局部 grid gathering/filtering 更稳 | Fig. 8 的单例与 Table 3 的 search time | 同类 probe payload/scene 中，whole-viewport iterative search 能稳定找到更准 reflection coordinate | 同一 probes、同一 current-view G-buffers、无 neural network；本文 search vs GPR-style gathering/filtering | scene/camera frames、probe count/resolution/spp、visibility rule、hardware、`K/N_max` | search-only RMSE/LPIPS、invalid/ghosting rate、ms median/p90 | scene-level bounded search，8×20 max | matched 多场景结果不改善 error，或同质量下 search time/invalid rate更差 |
| H2：在 image-space transport reconstruction 中，G-buffer affine modulation 比 concat 更有效 | Fig. 14 qualitative GM ablation | clean physical buffers 与 noisy radiance 的关系适合 scale+bias modulation | 参数量/MAC matched 的 concat vs GM；相同 encoder/decoder/data/loss/seed budget | scene split、input spp、history、optimizer、parameter/MAC、TensorRT precision | linear/HDR image error、LPIPS、highlight recall、temporal instability、latency | coherent full-frame network | bootstrap CI 不支持质量改善，或改善只来自额外参数/MAC，或 highlight recall恶化 |
| H3：低 spp baked transport + learned correction 可以在**计入 GT 与 training 后**改善 quality–precompute–runtime Pareto | Table 1–3；低 spp cache 与完整 totals | scene/reference 类型与训练覆盖足够时，减少 probe spp 的节省大于监督/fit 成本 | 高 spp probes+analytic query vs 多个冻结 spp 的 probes+同一 network；必须含 GT generation、fit、asset bytes 和 runtime | scene/lighting/material、probe placement/count/resolution、training views、hardware、quality protocol | total bake hours、bytes、frame ms、RMSE/LPIPS、highlight/temporal score | per-scene asset compiler + image decoder | 计入 GT/fit 后无 Pareto 点，或低 spp 在高光/未见 view 上系统失败 |
| H4：把 search-only 与 final 分开评测能定位 scene transport 错误来自 correspondence 还是 reconstruction | Table 1 同时报告 Ours (Search)/(Final)，Fig. 16 显示 Search 有而 Final 丢失的高光 | 本项目未来 environment integration track 也能保存同样的中间语义 | 同一 scene run 同时报 raw transport query、reconstruction 与 GT，不改任何预算 | source snapshot、camera/lighting、buffers、exposure、model checkpoint | raw/final error、peak recall、edge ghosting、temporal stability | evaluation protocol，不进入 local shader | 中间输出无法定义可比较 measure，或 failure attribution 在重复帧/场景中不稳定 |

这些都是后续 scene-level 任务假设，不是本研究报告的质量 hard gate，也不授权修改当前 NVIDIA evaluator formal config。

## 16. 证据索引

### `P` Main paper

- p.1 Abstract/Fig. 1：问题、1080p/>30 FPS、low-resolution/low-spp probes、temporal strategy；
- §1/p.2：GPR 局限、pipeline、贡献与 static-scene long glossy paths；
- §3, Eq. 1–2, Fig. 2–3：lightmap/probe payload、`K` probe reprojection、neural input/output；
- §4.1–4.2：low-spp lightmap、probe generation 与三类 panorama payload；
- §4.3, Eq. 3–4, Fig. 4–8：gradient search、special cases、visibility、blend、GPR search 对照；
- §4.4, Eq. 5–9, Fig. 9：24/32/64 network、GM、temporal reprojection、loss、Adam/1000 epochs/20 views/TensorRT；
- §5：four scenes、spp/resolution/probe count、search parameters、hardware/backend；
- §5.1, Fig. 10–12：RTRT/GPR/LFP/ISG/ULR protocols 与 qualitative results；
- §5.2, Table 1：20 frames/scene 的 RMSE/DSSIM/LPIPS 完整数值；
- §5.3, Fig. 13–15：probe count、GM/LPIPS、temporal joint ablations；
- §5.4, Table 2–3：precompute/runtime cost 与设备；
- §6, Fig. 16：高光失败、per-scene model、dynamic update、probe placement 限制。

### `S` Supplemental

- 未获得独立 supplemental 文档；作者 video 的本地下载不完整且未验证，未用作事实来源。

### `C` Official code/config/data

- 第一方 publication entry 未提供 code/config/checkpoint/data locator；没有 `C` 事实。

### `A` Author page

- `https://lingqiyan.github.io/`：论文 PDF 与 video 入口可见；只确认 availability，不补充技术配置。

### `N` NeuralShading

- `docs/realtime_material_compilation.md` 开头：`prepare/evaluate/sample/pdf` 与线性 `f` runtime contract；
- `docs/research/experiment_framework.md` §0、§2、§5：当前 NVIDIA evaluator/sampler route、线性 `f`、部署成本登记；
- `docs/research/model_candidates.md` §1：local evaluator query/output 语义。

### `I` 本报告分析

- 第 13 节：容量位置、成功假设、可迁移机制与 local/scene 边界；
- 第 14 节：NVIDIA correspondence；
- 第 15 节：四个可证伪的 scene-level 假设。

## Evidence review

```text
author_worker: /root/lightprobes2022
reviewer: /root/lightprobes2022_review
reviewed_at: 2026-08-29
sources_rechecked:
  - official main PDF SHA-256 F189B422F6B4ED17CB29436414FB0AD8B149075CCD56EC9BD3B8B8652B275DD3; pdfinfo confirms 14 pages
  - rendered page-01.png through page-14.png, including Fig. 3-16, Eq. 1-9 and Tables 1-3
findings_closed:
  - restored vector direction markers in the Eq. 3 transcription
  - corrected Fig. 9 final decoder operation to the legend's 1x1 transposed convolution
  - separated the N_max-bounded basic iterations from the uncapped/underspecified shortened-step special-case search
  - rechecked all Table 1-3 and Fig. 8/10-12 numeric transcriptions against rendered pages
remaining_evidence_gaps:
  - the first-party supplemental-video locator exists, but the partial local video download is unverified and was not used
  - official code/config/checkpoint/data unavailable
  - Eq. 3 confidence versus Eq. 4 exp(-tau*C) semantics unresolved
  - curve-search update/step schedule, shortened-step cap/counting and invalid-probe handling unreported
  - ground-truth sampling, split/color/metric protocol unreported
  - optimizer/batch/schedule/seed and HDR log transform unreported
  - parameter/MAC/bytes/precision and full runtime methodology unreported
review_status: passed-with-explicit-gaps
```

### 完成检查

- [x] main paper 已完整阅读，关键公式/图/表/图注/脚注已视觉核对；
- [x] supplemental/appendix/勘误的可用性已检查；未验证的部分 video 没有用作证据；
- [x] official code/config/data 的可用性已检查；第一方入口未提供；
- [x] architecture、training、runtime 和主要结果均有 locator；
- [x] 失败尝试与较差消融正确分类；
- [x] paper/code gap 和“未报告”保留；
- [x] `I` 分析晚于事实层，没有改写作者结论；
- [x] NVIDIA 影响引用真实 `N` 证据；
- [x] 假设包含 matched control、部署类别和证伪条件；
- [x] 独立 evidence review 已完成；未解决项均是第一方材料未披露或 Eq. 3–4 的论文内部语义张力，未用猜测填补。
