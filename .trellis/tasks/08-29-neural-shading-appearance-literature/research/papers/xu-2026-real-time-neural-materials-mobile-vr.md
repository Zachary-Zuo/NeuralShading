---
paper_id: "xu-2026-real-time-neural-materials-mobile-vr"
title: "Real-Time Neural Materials on Mobile VR"
authors: "Zilin Xu; Yang Zhou; Yehonathan Litman; Matt Jen-Yuan Chiang; Lingqi Yan; Anton Michels"
year: "2026"
venue: "Computer Graphics Forum 45(2), Eurographics 2026, e70318"
doi: "10.1111/cgf.70318"
report_status: "evidence-reviewed"
main_source: "https://doi.org/10.1111/cgf.70318"
supplemental_status: "unavailable"
official_code_status: "unavailable"
official_code_commit: "not-applicable"
author_worker: "/root"
reviewer: "/root/rta2024"
last_verified: "2026-08-29"
---

# Real-Time Neural Materials on Mobile VR：移动 VR 上的神经材质

## 1. 研究对象与报告边界

本文解决的是：如何在 Meta Quest 3 这类低功耗移动 VR 设备上，以至少 72 FPS 的显示帧率渲染 measured BTF 神经材质。它同时压缩三种成本：把 decoder 降到极低容量；把 coarse MLP 的一次求值摊给 `2×2` texel；把 texture-space shading 的结果跨 `N` 帧复用。最终屏幕仍逐帧正常渲染，降低的是昂贵神经着色的更新频率。[P Abstract, §3–4, Fig.1–3]

本报告覆盖 Computer Graphics Forum / Eurographics 2026 正式论文及内嵌 Appendix A。论文提到 supplementary video，但截至核查日，DOI 页面、作者主页和可检索的第一方入口没有给出可取得的视频、supplemental、代码、配置或模型，因此所有正文未披露的实现细节保留为“未报告”。[P §5.5][A DOI/author pages]

本文属于 local spatial appearance 与部署系统研究。其被学习对象是每个 measured material 的 6D BTF，不是跨材质 compiler，也不学习 scene-level visibility/global transport；运行时只演示 point-light direct shading，不提供 `sample()/pdf()` 或环境光积分方法。[P Eq.1–4, §5, §7]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | [DOI `10.1111/cgf.70318`](https://doi.org/10.1111/cgf.70318)；作者主页链接的正式 14-page PDF | 2026-08-29 | SHA-256 `4CEBE863D4393C297DC7509FC37BC09C4F5E1F905B2AA6C1CCB82572F2DBBBCF` | 正式方法、训练、结果、限制和内嵌 Appendix A；已完整读取并视觉核对全部 14 页、Eq.1–7、Fig.1–17、Table 1–3 及图注。 |
| Supplemental `S` | 正文 §5.5 提及 “supplementary video” | 2026-08-29 | unavailable | Wiley、Lingqi Yan、Zilin Xu 与 Yehonathan Litman 的第一方条目均未给出可下载视频；第一作者页将独立 project page 标成 “coming soon”。不能用视频补足 latency protocol 或动态伪影。 |
| Official code/config/data `C` | Wiley、[first-author publication entry](https://starry316.github.io/)、合作者主页与 GitHub 定向检索 | 2026-08-29 | unavailable | 未发现本论文正式代码、shader、训练配置、checkpoint、导出资产或 data manifest；第一作者现有其他 neural-material 仓库不能代替本文 release，本报告不从 NeuMIP 或其他项目猜实现。 |
| Author page/talk/correction `A` | [Zilin Xu publication entry](https://starry316.github.io/)；[Lingqi Yan publication page](https://lingqiyan.github.io/)；[Yehonathan Litman page](https://yehonathanlitman.github.io/)；[Wiley article page](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70318) | 2026-08-29 | 页面无版本号 | 核对题名、作者、venue、DOI 与 2026-03-24 first-published date；第一方页面目前只链接 paper，未发现勘误、code 或 video。 |
| NeuralShading evidence `N` | `.trellis/tasks/archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md`；`docs/realtime_material_compilation.md`；`docs/research/experiment_framework.md` | 2026-08-29 | 当前 workspace | 只用于第 13–15 节的项目映射，不作为本文方法事实。 |

书目信息：Computer Graphics Forum 45(2)，article `e70318`，first published 2026-03-24；作者单位覆盖 MBZUAI、Meta Reality Labs Research 与 Carnegie Mellon University。[A Wiley]

## 3. 原论文的问题、假设与贡献边界

作者的目标平台同时要求每眼 `2064×2208` 显示分辨率、双目和至少 72 FPS。直接在每个屏幕 pixel 执行常见 `25×25` 或 `32×32` hidden-width 神经材质网络，即使在高端桌面 GPU 上也很困难；Quest 3 又没有作者所需的低比特网络硬件支持，所以量化不是本文采用的主路径。[P §1, §3]

正式贡献由三个耦合部分组成：

1. 两级 coarse-to-fine per-material BTF 表示，用一个 coarse inference 服务 `2×2` texel，再用四个很小的 fine inference 恢复各 texel 细节；[P §4.1, Fig.2]
2. 在对象 texture space 运行 compute shading，把输出 radiance texture 跨后续 `N` 帧复用，同时让 viewport/几何仍以满帧率绘制；[P §3.2, §4.2, Fig.3]
3. 先训练 128-wide teacher，再用 output 与两个中间 feature matching 蒸馏 8-wide student。[P §4.3–4.4, Fig.4, Eq.5]

作者的部署假设是 VR 头动和照明通常连续、相邻 texel 的 lighting 平滑，少量 shading latency 与 `2×2` 内共享角条件不易察觉。高镜面材质更敏感，正文将 `N=4` 写成折中建议而非普遍最优。[P §3.2, §4.2]

论文不声称支持：未知 height 下的 neural parallax、robust LoD、跨材质共享 decoder、动态材质合成、IBL、材质驱动 importance sampling、任意光源下的单次推断，或双目共享 inference。[P §4.1, §6–7]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | UBO2014 measured BTF；正文选择 7 个材质 | 每材质一个离散 6D table；正文称完整数据约 40 GB | [P §3.1, §5] |
| Runtime query | `BTF(u,ω_i,ω_o)` | `u∈R²`，`ω_i,ω_o∈R³` | [P Eq.1–4] |
| Angular condition | incident、outgoing 与 half vector `h` | `ω_i,ω_o,h` 合计 9 scalars；正文没有单列 `h` 的归一化公式 | [P Eq.3, Fig.2] |
| Spatial condition | coarse center `u_c=1/4 Σ_{k=1}^4 u_k`；四个 fine coordinate `u_k` | 一个 `2×2` texel square | [P §4.1, Eq.3–4] |
| Coordinate frame | runtime 从 triangle ID 取三角形数据并插值，把 angular inputs 变换到 local space | per dispatch/thread group | [P §4.2, Fig.3 caption] |
| Output quantity | `BTF := dL_o/(L_i dω_i)`；foreshortening conventionally factored into BTF | RGB reflectance；随后形成 object-space radiance texture | [P Eq.1, §4.2] |
| Light domain | 实验中的 one-to-six point lights；每盏灯完整执行一次 neural inference | direct point-light shading | [P §5, Fig.10, Table 2–3] |
| Domain restrictions | 6D measured BTF，不含 neural offset/parallax；无 robust LoD；不做 IBL angular integration | opaque measured appearance | [P §4.1, §6–7] |

需特别区分两个 target：训练/2D slice 实验直接使用离散 BTF 数据；O3DE 里的对齐实时图像通常以 128-wide `NeuMIP Max` 作为“compromised reference”。Appendix A 才使用原始 6D BTF 的离线 Monte Carlo 渲染作为 reference，并为所有方法复用相同 MC samples。[P §5 Obtaining Reference Images, Appendix A/Fig.17]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

对 `2×2` texel 的中心：

`u_c = (u_1+u_2+u_3+u_4)/4`

`z_c = M_c(ω_i,ω_o,h,U_c(u_c))`。[P Eq.3]

`z_c` 在该 square 内共享。随后四个 texel 分别执行：

`BTF(u_k,ω_i,ω_o)=M_f(z_c,U_f(u_k))`。[P Eq.4]

方向 `ω_i,ω_o` 也在 `u_c` 对应的位置计算，而不是分别在四个 `u_k` 处计算，因此 coarse lighting 较准确、fine detail 带有作者承认的 minor angular aliasing。这个误差不是训练随机性，而是空间摊销的确定性近似。[P §4.1]

部署时先离线生成 triangle-ID texture。compute shader 覆盖对象 texture，backface texel 提前退出，以 dispatch ID 为 UV 取 triangle ID，手工 fetch/interpolate triangle data 和 local directions，执行 coarse-to-fine material shading，把结果写入每眼一张 object-space radiance texture。该 texture 更新一次后，后续 `N` 帧复用；forward pass 只按 UV gather radiance texture。[P §4.2, Fig.3]

### 5.2 持久化表示

- 每材质一张 7-channel coarse neural texture `U_c`，在每个 `2×2` texel square 的中心查询。论文没有直接写出 `U_c` 的离散分辨率；`200×200` 只能由 `400×400` spatial grid、`2×2` 分组和 Table 1 存储算术共同推得，属于 `[I]`，不能作为作者显式配置。[P Fig.2, §4.1/4.4, Table 1][I]
- 每材质一张 8-channel fine neural texture `U_f`，在各 `400×400` texel 上查询。[P Fig.2, §4.1/4.4]
- 每材质一套 coarse/fine student MLP。正文没有报告跨材质共享权重，也没有从材质参数生成网络/texture 的 compiler。[P §4, Table 1]
- Table 1 报告 ours 总存储 `6.24 MB`，包含 neural textures 和 `1.6 KB` network parameters；modified NeuMIP 为 `4.48 MB` total、`6.6 KB` network。故“network 7.4× smaller”不能改写成“整个材质表示更小”；而且 `6.6/1.6≈4.1`，Table 1 的 network-storage ratio 本身并不等于正文/Fig.1 的 `7.4×` network-size claim，该 claim 的参数、MAC 或其他口径未报告。[P Fig.1, §5.2, Table 1]
- `U_c:7×200²` 与 `U_f:8×400²` 若按 FP32 计数，恰为十进制 `6.24 MB`；这能解释 Table 1，却仍只是算术一致性 `[I]`。正文另称 inference 使用 half precision，但没有说明 Table 1 是训练资产、磁盘资产还是 runtime texture accounting，也没有给部署 texture format。[P §5, Table 1][I]

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Student coarse `M_c` | `ω_i(3)+ω_o(3)+h(3)+U_c(7)=16` | `16→8→8→8` | Fig.2/4 画出 hidden activations，但正文未命名 activation/normalization | `z_c∈R⁸` | per-material | [P Fig.2, Fig.4, Eq.3] |
| Student fine `M_f` | `z_c(8)+U_f(8)=16` | `16→8→3` | activation 与 output transform 未报告 | RGB BTF | per-material | [P Fig.2, Fig.4, Eq.4] |
| Teacher coarse | 16D coarse input（Fig.4 同样画为 angular 9 + texture feature 7；是否与 student 共用 texture 未报告） | `16→128→128→128` | activation/normalization 未报告 | `z_{t1}∈R¹²⁸` | training-only、per-material | [P Fig.4, §4.3] |
| Teacher fine | `z_{t1}(128)+` teacher-side 8D fine texture feature `=136`；与 student `U_f` 是否共享未报告 | `136→128→3` | activation/output transform 未报告 | RGB BTF | training-only、per-material | [P Fig.4, §4.3] |
| Distillation transforms `T_1,T_2` | teacher hidden feature 128D | learnable linear transform；正文先定义 `T` dimension 为 `D_s×D_t`，按 `D_s=8,D_t=128` 应写 `8×128`，随后又把实际 size 写为 `128×8` | linear | student-aligned 8D feature | training-only | [P §4.3, Fig.4]；矩阵朝向/row-column convention 内部不一致，无法由代码消歧 |

图 4 将 teacher/student 各自画成完整 coarse-to-fine 网络：`z_{t1}/z_{s1}` 是 coarse MLP 的 128D/8D 输出，`z_{t2}/z_{s2}` 是 fine MLP 唯一 hidden layer 的 128D/8D feature；`λ_1L_2`、`λ_2L_2` 分别接这两对 feature，`σL_1` 接最终 RGB，另有 student 对 GT 的主 `L_1`。作者只给出“8-wide/128-wide”和图示拓扑，没有给 bias、具体 activation、weight initialization、参数量计算或 shader packing；这些不能从常见 MLP 默认值补全。[P Eq.5, Fig.2, Fig.4]

### 5.4 条件化、坐标变换与物理先验

本文没有 analytic BRDF core，也没有 NeuMIP 式 learned offset。对 UBO2014，作者依赖测量同时提供的 height field，并建议在常规 graphics pipeline 外接 parallax mapping 或 displacement；未知 height 材质因而不在能力范围内。[P §4.1]

显式加入 `h` 是作者为低容量 student 提供的 angular parameterization enhancement。它与 `ω_i,ω_o` 一起直接输入 coarse MLP；fine MLP 不再看方向，只经共享 `z_c` 接收方向信息。[P §1, Eq.3–4, Fig.2]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset/source assets | UBO2014 的 7 个 BTF：`Leather11, Fabric12, Wood06, Leather08, Fabric07, Carpet07, Fabric09` | [P §5, Table 1] |
| Spatial resolution | 每个 training step 取一个 `400×400` 2D BTF slice | [P §4.4] |
| Batch angular condition | 同一 batch/slice 中所有 spatial samples 使用同一 viewing 与 lighting angles | [P §4.4] |
| Neural texture sampling | 对 neural textures 使用 stratified sampling，以避免 discrete artifact；具体 jitter 分布和边界处理未报告 | [P §4.4] |
| Train/validation/test split | 正文多次使用 “testing L1/test loss”，但没有报告方向、slice 或材质层面的 split 构造 | 未报告 |
| 训练 GT | measured BTF table 的 2D slices | [P §4.4, §5.2, Fig.7/11/12] |
| O3DE aligned-image proxy | modified NeuMIP、hidden width 128 的 `NeuMIP Max` | [P §5 Obtaining Reference Images] |
| Real 6D reference | Appendix A 用 offline Monte Carlo path tracing 直接渲染原始 6D BTF；各方法复用相同 samples | [P Appendix A, Fig.17] |
| Lighting protocol | 默认三盏 point lights、`N=4`、每眼 `1200²` texture；skybox 只作背景，不发光 | [P §5] |
| LoD/filtering | 未使用 feature pyramid，也没有 robust LoD；texture resolution 是显式质量/成本轴 | [P §5.2, §6] |

正文没有披露：原始 BTF directional sample layout、方向抽样分布、训练/测试角划分、每 epoch 的 slice 数、数据 normalization、RGB color space/exposure、teacher 与 student 是否共享/重新初始化 neural textures，或正式 checkpoint 选择规则。[P §4.4]

## 7. Loss、optimizer 与训练 lifecycle

student loss 为：

`L = L1(y_s,y_gt) + σ L1(y_s,y_t) + λ_1 L2(z_s1,T_1(z_t1)) + λ_2 L2(z_s2,T_2(z_t2))`，[P Eq.5]

其中 `y_s/y_t/y_gt` 分别是 student、teacher 和 measured GT reflectance；两对 `z` 来自对应 hidden layers；`T_1,T_2` 与 student 联合训练。所有 auxiliary weights `σ,λ_1,λ_2` 均为 `0.1`。[P §4.3]

| 项 | 正式配置 | locator |
|---|---|---|
| Target/output transform | 直接称 RGB reflectance；没有报告 log/compression、clamp 或 inverse transform | 未报告 |
| Optimizer | Adam，`β1=0.9, β2=0.999, ε=1e-8` | [P §4.4] |
| LR | `5e-4 → 1e-7`；ReduceLROnPlateau，factor `0.5` | [P §4.4] |
| Scheduler condition | 作者表述为每个 epoch 的 testing L1 decrease 小于 `1e-4` 时减半；patience、window 与是否用 validation 代替 test 未报告 | [P §4.4] |
| Batch/query count | 每个 training step 取一个 `400×400` 同角度 2D slice “as one batch”，即 160,000 spatial samples；论文没有说明实现是否因显存做等价 microbatch/gradient accumulation | [P §4.4] |
| Teacher stage | 先训练约 60 epochs | [P §4.4] |
| Student stage | 从 scratch 训练，distillation 最多 150 epochs | [P §4.4] |
| Initialization/seed/model selection | 未报告 | 未报告 |
| Hardware/time | desktop RTX 4080；teacher + student 全流程约 19 hours | [P §4.4] |

没有官方代码可核对 “from scratch” 是否同时重置 neural textures、`T` 的方向、LR scheduler 对象、epoch 完整定义以及 loss reduction。故这里不把图中矩阵方向或框架常见默认值升级成实现事实。

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime path | triangle-ID texture → compute shader/local directions → coarse once per `2×2` → fine four times → per-eye radiance texture → forward UV gather | [P §4.2, Fig.3] |
| Update frequency | 更新后未来 `N` 帧复用，即 `N+1` frames share one radiance texture；默认 `N=4` | [P §4.2, Fig.3] |
| Network size | ours network storage `1.6 KB`；modified NeuMIP `6.6 KB`，按表内 bytes 只形成约 `4.1×` ratio；作者另称 MLP/network size `7.4×` smaller，但没有给 parameter/MAC/FLOP 口径来解释两者差异 | [P Fig.1, §5.2, Table 1] |
| Total stored material | ours `6.24 MB`；NeuMIP `4.48 MB`；4-lobe analytical fit `35.2 MB` | [P Table 1] |
| Runtime radiance texture | Table 3：每材质使用两眼各 `1024²` 的 radiance texture；作者称 “32-bit radiance texture”，合计 `8.0 MB/material`，数值与 `1024²×4 B×2 eyes` 一致。channel format、是否另有 update/read double buffer 未报告 | [P Table 3][I：bytes 算术] |
| Precision | inference 使用 half precision；具体纹理/weight/accumulator 各自 precision 未拆分 | [P §5] |
| Backend/hardware | O3DE；Meta Quest 3，ADB 从 desktop 记录 FPS；设备 cap 为 90 FPS | [P §5, §5.4] |
| Eye sharing | 当前双眼分别 inference；Fig.3 描述 one pass 生成 two radiance textures，但不复用同一 radiance texture/共同 view result | [P §3.2, §6] |
| Multi-light scaling | 每个 point light 完整执行一次 material inference，FPS 随灯数近似线性下降 | [P §5.4, Fig.10] |
| Sampling/integration | 无 `sample()/pdf()`；IBL 仍需大量 angular samples，被列为移动端不可承受的开放问题 | [P §7] |

“one pass”与“shared inference”应区分：实现可以在同一 texture-space pass 中写两眼 radiance textures，但论文 §6 明确当前仍对两眼独立推断，没有利用双目 view correlation。[P §3.2, §6]

## 9. 实验 protocol、baseline、指标与结果

### 9.1 表示误差与存储

Table 1 是 2D BTF slice / whole discrete BTF L1 语境，不是 Quest 3 rendered-frame 质量排名：

| Material | 4-lobe analytical | modified NeuMIP | Ours | 最低值 |
|---|---:|---:|---:|---|
| Leather11 | 0.04606 | **0.02285** | 0.02311 | NeuMIP |
| Fabric12 | 0.01914 | **0.00814** | 0.00840 | NeuMIP |
| Wood06 | 0.02930 | 0.01264 | **0.01078** | Ours |
| Leather08 | 0.04185 | **0.02811** | 0.02904 | NeuMIP |
| Fabric07 | 0.02589 | 0.00851 | **0.00847** | Ours |
| Carpet07 | 0.01570 | **0.01472** | 0.01668 | NeuMIP |
| Fabric09 | 0.01082 | 0.01108 | **0.00939** | Ours |
| Total storage | 35.2 MB | **4.48 MB** | 6.24 MB | NeuMIP |

作者结论是 low-capacity ours 与更大 modified NeuMIP 质量同一水平，而不是逐材质全面超过。七个 L1 中 ours 赢 3 个、NeuMIP 赢 4 个；ours 的 total storage 也更大，只是 runtime MLP 明显更小。[P §5.2–5.3, Table 1, Fig.7–9]

modified NeuMIP 做了两个正式变更：去掉 feature pyramid、固定使用 finest neural texture；去掉 offset module。前者因为本文不评估 LoD，作者报告不损失质量；后者因为 UBO2014 提供 height field、传统 parallax/displacement 可外接，且论文目标只拟合 6D BTF。[P §5.2]

这个 baseline 对论文限定的“单一 finest resolution、只拟合 6D BTF”任务是有意对齐的，但它不再代表原始 NeuMIP 的 LoD/parallax 全能力。论文没有给 modified NeuMIP 的完整 topology、optimizer/schedule、checkpoint selection、训练时间或 matched feature-pyramid ablation；“去 pyramid 不损失质量”是作者文字声明。Fig.8 的 `NeuMIP Eq. Size` 也只说明 `8×8 latent layer size`，没有给总参数、texture bytes 或 distillation/training parity，因此只能支持该 protocol 下的可视结果。[P §5.2, Fig.8]

### 9.2 Texture-space 与 temporal amortization

- Fig.5，Wood06、Quest 3/O3DE，按 §5 默认三盏 point lights 与 `N=4`：screen-space `2064×2208` 为 9 FPS；每眼 texture-space `1200×1200` 为 80 FPS，即 `8.9×`。两者相对 screen-space `NeuMIP Max` 的 PSNR 都是 `38.07`、SSIM 都是 `0.976`；FLIP 分别 `0.048/0.051`。这个 8.9× 同时包含 resolution change、texture-space path 与 temporal reuse，不是单一 amortization 因子的速度比。[P §5, Fig.5]
- Fig.6，Leather08，按默认三灯与 `N=4`，以 ours screen-space 为 reference：每眼 `800²/1200²/1600²` 分别为 PSNR `25.78/28.13/29.72`、SSIM `0.764/0.854/0.898`、FLIP `0.122/0.077/0.062`；800² 有可见离散 artifact。这里评估 shading-grid resolution，不是相对原始 BTF 的 representation error。[P §5, Fig.6]
- spatial amortization 打开、temporal 关闭只降低性能且没有质量损失；temporal 打开而 spatial 关闭在 screen-space path 上不可执行，所以没有独立 2×2 因素的完整 matched timing。[P §5.1]

Table 2 是 Quest 3、同一 Fig.5 scene、一盏 point light；表中的 update interval 就是 `N`，所以一张 radiance texture 覆盖 `N+1` 个 display frames；screen-space 只有 `23 FPS`：

| reuse interval | 800² | 1200² | 1600² | 2000² |
|---:|---:|---:|---:|---:|
| 0 frames | 80 | 42 | 36 | 27 |
| 1 frame | 90 | 67 | 57 | 43 |
| 2 frames | 90 | 82 | 71 | 58 |
| 4 frames | 90 | 90 | 82 | 76 |
| 6 frames | 90 | 90 | 89 | 81 |

作者选择 `1200², N=4` 作为默认质量—性能折中；注意 90 FPS 是设备 cap，不能把多个 90 当成相等的未截断运行时间。[P §5.4, Table 2]

### 9.3 灯数与材质数 scaling

- Quest 3、Fig.5 scene、每眼 `1200², N=4` 下，1–6 盏 point lights 分别 `90, 90, 81, 70, 61, 42 FPS`；前两点被 90 FPS device cap 截断。每盏灯都要完整 inference，图支持的是近似线性下降趋势，不提供 uncapped frame time。[P §5.4, Fig.10]
- 三盏 point lights、每材质每眼 `1024²`、`N=7` 下，同时 1–4 个不同材质分别 `81, 62, 51, 39 FPS`。作者称随总 shading texture resolution 近似线性；每材质 radiance texture `8.0 MB`。[P Table 3, §5.4]
- Fig.14 的极高分辨率例子：screen-space 每眼 `2064×2208` 为 15 FPS 且没有覆盖所有 pixels；texture-space 使用 `4×2400²` texels 为 14 FPS，输出约 5× 更多 pixels/texels。它证明吞吐范围，不是大 texture 仍满足 VR 帧率。[P Fig.14]
- Fig.16 用 screen-space `NeuMIP Max` 作 proxy reference，比较 modified NeuMIP screen-space 与 ours texture-space：Fabric12 为 `<2 FPS, 31.23/.943/.080` 对 `85 FPS@1200², 31.98/.877/.069`；Leather11 为 `<2 FPS, 30.98/.914/.083` 对 `72 FPS@1600², 27.69/.847/.100`；Fabric07 为 `<2 FPS, 33.20/.953/.061` 对 `84 FPS@1200², 32.37/.942/.064`，三元组依次是 PSNR/SSIM/FLIP。它证明移动端可运行性差异，但同时改变 network、screen/texture path 与 resolution；Leather11/Fabric07 的 NeuMIP 质量指标还更好，不能作为 matched representation-quality 胜出证据。[P Fig.16]

### 9.4 Distillation 与结构消融

- Fig.11 只展示 Leather11/Fabric12 的 whole-6D test-loss curves，distillation 收敛更好；没有原始曲线数值或多 seed。[P Fig.11]
- Fig.12，Fabric12 2D slice：无 distillation / 有 distillation 的 PSNR `35.11/35.12`，FLIP `0.088/0.080`。PSNR 几乎不变，但 FLIP 与 self-shadow visual 改善。[P Fig.12]
- Fig.13，Leather08：同容量 single-level / coarse-to-fine 的 PSNR `29.90/29.70`，FLIP `0.120/0.115`。作者归纳“不引入质量损失”，实际是两个指标给出相反的微小排序。[P Fig.13]
- Fig.15 是作者明确标注的 Fabric07 fine-detail failure：ours 为 PSNR `34.89`、FLIP `0.059`，NeuMIP 为 `35.00/0.083`；ours 的 PSNR 略低而 FLIP 更好，且两者视觉上都没有恢复 GT 极细结构，不能压成单一“谁更好”。[P §6, Fig.15]

### 9.5 对真实 6D BTF reference 的 Appendix A

Appendix A 用相同离线 MC samples 比较 analytical、modified NeuMIP、ours without amortization 和 NeuMIP Max。下表保留 `(PSNR, SSIM, FLIP)`，不把 NeuMIP Max proxy 当作原始 GT：

| Material | Analytical | NeuMIP | Ours w/o amortization | NeuMIP Max |
|---|---|---|---|---|
| Fabric12 | (28.65,.863,.116) | (38.37,.953,.053) | (33.11,.927,.122) | (40.31,.965,.035) |
| Wood06 | (32.15,.843,.088) | (40.77,.985,.050) | (34.92,.968,.101) | (43.66,.987,.033) |
| Fabric07 | (26.69,.859,.157) | (39.10,.972,.051) | (33.82,.952,.115) | (41.88,.975,.032) |
| Leather11 | (20.08,.873,.212) | (23.93,.919,.134) | (24.23,.912,.128) | (25.80,.948,.105) |
| Leather08 | (20.35,.820,.177) | (23.61,.856,.132) | (23.31,.827,.113) | (25.27,.893,.083) |
| Carpet07 | (27.14,.932,.111) | (29.27,.936,.079) | (28.60,.921,.087) | (30.79,.952,.065) |
| Fabric09 | (27.25,.816,.110) | (28.00,.848,.104) | (28.54,.829,.093) | (30.08,.889,.068) |

这些完整图像结果里 ours 相对 NeuMIP 是 mixed：Leather11 的 PSNR/FLIP、Leather08 的 FLIP、Fabric09 的 PSNR/FLIP 更好；其他材质/指标常由 NeuMIP 更好。论文自己的结论也是 similar quality。[P Appendix A, Fig.17]

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `author-negative` | 直接在 Quest 3 screen space 运行低容量网络 | Fig.5 只有 9 FPS | 每眼高分辨率与 per-pixel MLP 太贵 | 证明 temporal/texture-space 系统是达标条件，不能把结果归因于小 MLP 单项 | [P §3, Fig.5] |
| `ablation-inferior` | 直接训练 8-wide student，不蒸馏 | Fig.11 test-loss 收敛较差；Fig.12 PSNR `35.11→35.12` 近乎不变，但 FLIP `0.088→0.080`、self-shadow visual 由 distillation 改善 | teacher 的 output/feature hints 更充分利用有限容量 | 正文无多 seed，不能进一步声称降低 optimization variance | [P §4.3, §5.6, Fig.11–12] |
| `ablation-inferior` | NeuMIP 缩到与 ours 相同 8-wide capacity | Fig.8 的 highlight/self-shadow 保存失败 | coarse-to-fine parameterization + distillation 比等宽 NeuMIP 更有效 | baseline training parity 未完全披露 | [P §5.2, Fig.8] |
| `ablation-inferior` | 800² texture-space resolution | 出现离散 artifact；FLIP 0.122 | shading grid 太粗 | 属采样/重建误差，不是 BTF compression 单项误差 | [P Fig.6] |
| `author-negative`（baseline） | 4-lobe Lafortune analytical fit | 多数 L1/visual 较差，强 fabric sheen 捕捉失败 | 四个 analytic lobes 容量不足 | 只否定该固定 lobe 数与拟合 protocol，不否定解析/混合 BRDF 普遍能力 | [P §5.3, Fig.9, Table 1] |
| `author-negative` | 只开 spatial amortization、关闭 temporal reuse | 作者称只造成 performance drop，quality 不变；反向的 temporal-only screen-space 配置不可执行 | temporal reuse 依赖 texture-space radiance texture，两个系统轴紧耦合 | 没有数值 timing，不能分解 coarse `2×2` 本身的 speedup | [P §5.1] |
| `known-limitation` | 很细的 Fabric07 detail | ours PSNR 34.89/FLIP .059，NeuMIP 35.00/.083；两者仍未恢复 GT fine details | 小 MLP capacity 有限；更大 NeuMIP 也困难 | capacity failure 与 metric mixed，不能写成 ours 全面较差 | [P Fig.15] |
| `known-limitation` | very large texture | `4×2400²` 只有 14 FPS | 没有 robust LoD，close-up 要高分辨率 | 超过实时帧率门槛但吞吐仍高于 screen-space | [P §6, Fig.14] |

neural offset 与 low-bit quantization 都不进入上表：前者是目标只覆盖 6D BTF、已有 height field 且移动预算受限下的 `design exclusion`；后者是作者所需硬件在移动 GPU 上不可用的 `platform constraint`。正文没有报告两项的实现尝试或质量失败，不能标成 `author-negative`。[P §3.1, §4.1]

Fig.16 Fabric07 的 small black artifacts 被作者归因于 O3DE rendering pipeline，而非本方法；这是作者的诊断结论，没有公开代码或 isolate test 可独立复核。[P Fig.16 caption]

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Architecture | Fig.2/4 给出完整层宽、texture channel、coarse/fine dataflow 与两个 feature tap | 不可得 | 不可得 | topology/tap 可锁定；activation、bias、packing 不可锁定。`T:D_s×D_t` 与后文 `128×8` 的矩阵次序存在 paper-internal ambiguity |
| Data/query | 7 个 UBO2014；每 step 一张 400² 同角度 slice | 不可得 | 不可得 | split、direction sampler、epoch size 未报告 |
| Loss/training | Eq.5、Adam、LR/scheduler、60+150 epochs、19h | 不可得 | 不可得 | 足以重建高层 lifecycle，不足以逐步复现 |
| Runtime/export | O3DE compute/forward path、half inference、Quest 3 FPS | 视频 locator 不可得 | shader/export 不可得 | 无法核对 exact thread mapping、buffering 和 precision |
| Storage/cost | Table 1：ours `6.24 MB` total/`1.6 KB` network，NeuMIP `4.48 MB`/`6.6 KB`；正文另称 network `7.4×` smaller | 不可得 | 不可得 | total storage 可锁定；逐 texture format 未披露，且表内约 `4.1×` network-byte ratio 无法解释 `7.4×` claim |
| Assets/evaluation | UBO2014 名称、表格/图、NeuMIP Max proxy 与 Appendix real reference | 不可得 | checkpoint 不可得 | 2D、proxy image 与 real-6D result 必须分层解释 |

没有出现可供代码反驳/补充的 `paper-code-gap`；更准确的状态是 paper-only、code unavailable。正文 “one pass” 输出两眼 radiance textures 与 §6 “independently per eye” 可以同时成立：前者描述 dispatch/pass 组织，后者说明两眼不共享 view-conditioned neural result。真正未解析的是 `T` 的矩阵朝向，以及 Table 1 network bytes 与 `7.4×` claim 的计量口径。[P §3.2, §4.3, §5.2, §6, Table 1]

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. 没有 robust LoD；close-up 必须增大 texture resolution，性能迅速下降。[P §6, Fig.14]
2. 8-wide 容量仍会丢失 extreme fine-grained details，且较大 NeuMIP 对该类细节也可能失败。[P §6, Fig.15]
3. 不显式处理 tiling；作者只提出可能与 dynamic neural BTF synthesis 兼容。[P §6]
4. 当前双眼独立 inference，没有利用高度相关的 stereo directions；共享 one-pass result 是未来方向。[P §6]
5. point-wise evaluation 已解决，但 IBL 需要大量 angular samples；常见 split-sum 只适用于 analytic BRDF，神经材质的高效积分仍开放。[P §7]
6. neural offset 被排除，未知 height field 的 parallax 不支持。[P §4.1]
7. 多灯、多材质和高分辨率成本近似线性增长，Table 3/Fig.10/14 已直接显示帧率跌破 72。[P §5.4, §6]
8. 作者需要的 low-bit acceleration 在目标移动 GPU 上没有硬件支持，所以本文没有采用量化；这不是量化质量失败实验。[P §3.1]

### 12.2 未报告/材料不可得

- official code、shader、exporter、model/checkpoint、supplementary video locator；
- MLP activation、bias、初始化、output transform、parameter/MAC/FLOP counting convention；`7.4×` network-size claim 与 Table 1 `6.6/1.6 KB` ratio 的对应未解释；
- neural texture precision、layout、address/filter mode、训练与 runtime storage correspondence；Table 1 的 `6.24 MB` 与 half inference 的逐资源关系未报告；
- BTF train/test split、direction distribution、epoch query 数、seed、重复试验/置信区间、checkpoint selection；
- distillation 中 teacher/student neural texture 的共享/初始化关系，以及 `T` 的 row/column orientation；两处 feature tap 已可由 Fig.4 锁定；
- modified NeuMIP 与 `NeuMIP Eq. Size` 的完整 topology、训练 parity、总参数/texture budget 和 checkpoint selection；
- `N` 帧 reuse 的 double/triple buffering、左右眼调度、异步 compute/barrier、motion threshold 或 adaptive update；
- informal user study 的人数、参与者特征、protocol、问题文本、场景/运动轨迹、统计与伦理信息；
- FPS 的 frame-time distribution、power/thermal state、warmup、测量时长，以及 90 FPS cap 下的真实 headroom；
- Table 1 的 MB/KB 采用十进制还是二进制，以及 total storage 与 half inference 的逐资源 accounting。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

本文没有把所有容量塞进 8-wide MLP。`U_c/U_f` 保存逐材质、逐空间的主要自由度；teacher 在训练期提供额外优化信号；runtime 再用空间与时间复用减少执行次数。因此“极小网络”只是 decoder compute 的描述，不是完整 representation 容量或总显存描述。Table 1 甚至明确显示 ours 总材质存储高于 modified NeuMIP。[I]

coarse-to-fine 的核心是把方向相关计算集中进 `z_c`，再让 `U_f` 只补 per-texel detail。这个分工很接近项目 `prepare()` 的理想职责：在同一着色点/小区域先计算可复用 view/light-conditioned state，再让廉价 evaluator 消费它。但本文的共享范围是固定 texture-space `2×2`，且把 `ω_i` 也放进 coarse state；它不能直接等同于本项目希望跨多 `ω_i` 复用的 view-conditioned `prepare(wo)`。[I]

### 13.2 成功所依赖的假设

- 头动、物体和灯光变化足够平滑，`N=4` 的 shading latency 不明显；
- 物体有良好 UV atlas、可预生成 triangle-ID texture，且 texture-space shading 不产生过大无效区；
- 目标是少量 point lights；每增加一盏灯可以接受再跑一次网络；
- 400² BTF spatial resolution 和固定 `2×2` coarse sharing 足以覆盖实验材质；
- height field 可另行提供，LoD/IBL/sampler 不在当前验收面。

这些假设解释了为何 90 FPS 不能单独作为“神经材质 evaluator 已足够便宜”的证据：达标结果同时包含降低 shading resolution、`2×2` spatial reuse、跨帧 reuse 与 90 FPS cap。[I]

### 13.3 可迁移机制与不能迁移的部分

可迁移：low-capacity student 的 intermediate-feature distillation；明确把 coarse/fine 条件拆开；把 expensive shading 与 display rate 解耦；将 texture resolution、update interval、灯数与材质数作为独立成本轴。[I]

不能直接迁移：用 stale radiance texture 替代随机访问 `evaluate(wo,wi)`；把一盏灯一次 inference 的 direct-light system 当成 path-tracing evaluator；把 NeuMIP Max proxy 当正式 GT；把 texture-space `2×2` 的 angular aliasing 引入需要逐 shading-point 精确 query 的 reference/training path。[I]

### 13.4 与本项目 runtime contract 的关系

两个 MLP、固定 `2×2` 调度、固定 texture reads 都是静态有界的；作为 forward direct-light renderer 可部署。但论文没有 `sample()/pdf()`，且 fine output 是 BTF/point-light shading path，不是已经按本项目 ABI 验证的线性 `f` evaluator。它更适合作为 `deployment/amortization` 和 `capacity diagnostic`，不是可原样替换当前 NVIDIA evaluator/sampler 的完整产品候选。[I]

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

当前 NVIDIA 复现已经有 hierarchical z8、`prepare()` 的 footprint/LOD fetch、`20→64→64→64→3` evaluator 与 matched sampler，并在统一 package 中执行 `prepare/evaluate/sample/pdf`。[N `.trellis/tasks/archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md`]

当前实现以 NVIDIA 2024 方法为 faithful identity，并不复现本文 Mobile VR renderer；两者之间没有 paper/config/checkpoint correspondence。因此下表只判断 Mobile VR 机制是否已属于当前 identity。候选迁移属于 `[I]`，不能把“当前没有该机制”标成实现偏差或 defect。[N/I]

| Mobile VR 机制 | 当前 NVIDIA 对应 | 分类 | 影响 |
|---|---|---|---|
| 8-wide student + feature distillation | 当前 evaluator 是 64-wide，训练未用大 teacher feature matching | `not-applicable` | `[I] candidate-transfer`：不改 faithful baseline；另建 budget-matched student candidate |
| coarse `2×2` sharing | 当前 `prepare()` 在单 shading point取 z8/构造 state，没有跨 texel 共用 `ω_i` | `not-applicable` | `[I] raster deployment variant`：另设 spatial-coherence 路线，不能改变 evaluator semantic baseline |
| across-frame radiance reuse | 当前 viewer 每帧按 package query | `not-applicable` | `[I] renderer scheduling`：只可作为 viewer 可选部署轨，不是 compiler/evaluator 忠实性修复 |
| half inference | 当前 NVIDIA reproduction 已有 functional FP16 weights、latent、arithmetic parity | `not-applicable`（跨论文 correspondence） | 当前 FP16 是 NVIDIA 自身的 `faithful` 项；本文只能启发移动硬件实测，因无 shader/code 不能逐 bit 对照 |
| no LoD | 当前 NVIDIA 已实现 hierarchical latent + footprint LOD | `not-applicable` | 本文缺失 LoD 是作者限制，不应倒推删除当前已恢复的 NVIDIA formal capability |
| no sampler/IBL | 当前 NVIDIA 有 matched two-lobe `sample/pdf` | `not-applicable` | 本文不能替代 path-tracing/环境积分验证；其 point-light scheduler 可另作 renderer-only 实验 |

最有价值的影响不是把当前 64-wide decoder直接缩到 8，而是建立一个严格 matched 的“teacher→student 是否把小网络从坏 basin 拉回可用质量”实验；训练稳定性需与 Taming Optimization Variance 的多 seed 证据一起解释。[I]

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H-MVR-1：feature+output distillation 能改善项目 shader-budget student 的 median 与 worst-seed quality | 本文 Fig.11/12；8-wide 直接训练较差 | teacher features 对 LayerStack/MaterialX query 也提供有用 optimization signal | 同一 8-wide student、相同 online queries/steps/optimizer；GT-only vs output-only vs output+2-feature；≥15 seeds | source/query recipe、参数量、batch、LR、output transform | directional L1/relative、image FLIP、seed median/p90、失败率 | evaluator candidate | full distillation 未改善 median，或 p90/失败率恶化且 CI 排除有益效应 |
| H-MVR-2：把 direction-heavy work 放进可复用 state，可在质量可控时降低单 query cost | `z_c` 在 2×2 内共享并达到 Quest 3 性能 | 项目存在同一 `wo` 下多 `wi` 查询或 raster neighborhood coherence | matched parameter/fetch budget：现有 per-query decoder vs `prepare(wo)+cheap evaluate(wi)`；禁止跨点偷换 GT | latent bytes、训练预算、precision、sampler disabled/enabled 分开 | local quality、prepare/evaluate ns、reads、multi-light scaling | product-capable evaluator | 单 query/多 light 的 break-even 不出现，或质量损失超出 matched best candidate CI |
| H-MVR-3：固定 `2×2` coarse spatial sharing 只在特定 footprint/频率范围安全 | Fig.6/13 与作者承认 angular aliasing | 当前 spatial source 的高频变化能由 footprint 指标预测 | per-texel vs `2×2/4×4` sharing，使用相同网络和 texture budget | exact directions、filter footprint、update interval=0 | error vs spatial frequency/roughness、FLIP、latency | raster deployment | 在 target footprint 内出现稳定结构性 artifact，或 speedup 不足覆盖额外 fine pass |
| H-MVR-4：跨帧 shading reuse 可以作为 viewer 可选部署轨，而不污染 evaluator 质量结论 | Fig.5/Table2/内部 user study | 项目 viewer 的 camera/light motion 也足够连续 | evaluator 固定；`N=0/1/2/4/6`，记录 exact trajectory 与无复用 reference | resolution、lights、material、display rate、thermal state | temporal FLIP、latency、frame-time p90、主观阈值另报 | renderer-only deployment | N=4 在预注册 motion envelope 内产生显著 temporal error，或 72 FPS 无实际收益 |
| H-MVR-5：小 network 并不等于小总表示，应联合约束 decoder 与 texture bytes | Table1：ours MLP 更小但 total storage 更大 | 当前候选也可能把容量转移到 latent texture | Pareto 同时报 network MAC、texture reads/bytes、state bytes、package total | dataset/quality target/hardware | quality-time-memory Pareto | evaluation policy | 只按 MLP 参数排序会反转与完整 package bytes/runtime 的 Pareto 结论 |

## 16. 证据索引

- `P §1–3`：问题规模、BTF 定义、VR 与 texture-space 假设。
- `P §4.1 / Eq.3–4 / Fig.2`：coarse-to-fine topology、channel count、`2×2` sharing 与 angular aliasing。
- `P §4.2 / Fig.3`：triangle-ID texture、compute pass、radiance texture 与 `N` 帧复用。
- `P §4.3–4.4 / Eq.5 / Fig.4`：teacher/student、feature transforms、loss、optimizer、batch 与训练时间。
- `P §5 / Fig.5–13 / Table 1–3`：proxy reference、质量、存储、性能、distillation 与结构消融。
- `P §5.5`：informal user study；supplementary video 当前不可得。
- `P §6–7 / Fig.14–16`：LoD、fine detail、tiling、双目、IBL 与 O3DE artifact。
- `P Appendix A / Fig.17`：真实 6D BTF offline MC comparison。
- `A Wiley/Zilin Xu/Lingqi Yan/Yehonathan Litman pages`：正式书目信息与第一方 PDF 入口；独立 project page 仍标为 coming soon，未给 code/video。
- `N correspondence.md`：当前 NVIDIA faithful reproduction 的 runtime/training correspondence。
- `I`：第 13–15 节；不反写为作者结论。

## Evidence review

```text
author_worker: /root
reviewer: /root/rta2024
reviewed_at: 2026-08-29
sources_rechecked:
  - main PDF SHA-256 4CEBE863D4393C297DC7509FC37BC09C4F5E1F905B2AA6C1CCB82572F2DBBBCF, all 14 pages, Eq.1-7, Fig.1-17, Table 1-3, captions, and embedded Appendix A
  - Wiley DOI page and first-party Zilin Xu, Lingqi Yan, and Yehonathan Litman publication entries
  - current NVIDIA reproduction correspondence at .trellis/tasks/archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md
findings_closed:
  - exact student/teacher coarse-to-fine topology and coarse/fine distillation taps
  - N versus N+1 temporal reuse semantics and one-pass versus independent-per-eye distinction
  - Table 1 total/network storage context and unresolved 7.4x versus 4.1x paper-internal metric gap
  - Fig.5/6/10/12/13/14/15/16/17 values with scene, resolution, light, reference, and amortization context
  - modified NeuMIP capability changes and remaining training/budget fairness limits
  - author-negative, ablation-inferior, known-limitation, design-exclusion, and platform-constraint classification
  - official code and supplementary-video availability boundary
remaining_evidence_gaps:
  - official code/config/checkpoint unavailable
  - supplementary video locator unavailable
  - activation, bias, output transform, split, seed, runtime precision and buffering breakdown not reported
  - T matrix orientation and the metric behind the 7.4x network-size claim unresolved
  - modified NeuMIP/NeuMIP Eq. Size training parity and exact budgets not reported
review_status: evidence-reviewed
```

### 完成检查

- [x] main paper 已完整阅读，关键公式/图/表/图注/脚注已视觉核对；
- [x] supplemental/appendix/勘误的可用性已检查；Appendix A 已纳入，supplementary video 当前不可得；
- [x] official code/config/data 的可用性已检查；当前不可得；
- [x] architecture、training、runtime 和主要结果均有 locator；
- [x] 失败尝试与较差消融正确分类；
- [x] paper/code gap 和“未报告”保留；
- [x] `I` 分析晚于事实层，没有改写作者结论；
- [x] NVIDIA 影响引用真实 `N` 证据；
- [x] 假设包含 matched control、部署类别和证伪条件；
- [x] 独立 evidence review 已完成。
