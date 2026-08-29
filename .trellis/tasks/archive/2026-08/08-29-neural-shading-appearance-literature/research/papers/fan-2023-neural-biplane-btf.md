---
paper_id: "fan-2023-neural-biplane-btf"
title: "Neural Biplane Representation for BTF Rendering and Acquisition"
authors: "Jiahui Fan; Beibei Wang; Miloš Hašan; Jian Yang; Ling-Qi Yan"
year: "2023"
venue: "SIGGRAPH 2023 Conference Proceedings"
doi: "10.1145/3588432.3591505"
report_status: "evidence-reviewed"
main_source: "https://sites.cs.ucsb.edu/~lingqi/publications/paper_biplane.pdf"
supplemental_status: "available"
official_code_status: "audited"
official_code_commit: "e2add11c795e6003d0069d214df8c57ac4b9889b"
author_worker: "/root"
reviewer: "/root/rta2024"
last_verified: "2026-08-29"
---

# Neural Biplane Representation for BTF Rendering and Acquisition

## 1. 研究对象与报告边界

本文研究 6D bidirectional texture function（BTF）`ρ(u,ω_i,ω_o)` 的表示、逐材质压缩、运行时求值与稀疏实物采集。核心表示把 Rusinkiewicz half/difference 方向中的 half-vector 与二维空间位置分别放入两个 feature plane，再由一个跨材质共享、训练后冻结的 MLP 解码；可选 per-BTF offset module 处理 synthetic heightfield 的 parallax。[P §3.1–3.4, Fig.2]

本报告覆盖 SIGGRAPH 2023 11 页正文、1 页正式 supplemental 和作者项目页。论文的 acquisition 分支属于 inverse appearance acquisition，但它与核心表示共用 H/U planes 和 universal decoder，因而是本篇的 load-bearing 组成，不能只摘掉 acquisition 结果。它没有 LOD/footprint 轴，不等同于 NeuMIP 的 multi-resolution MBTF；它也不学习 scene transport 或 scene visibility。[P §3.5–§4.4]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | [author-hosted PDF](https://sites.cs.ucsb.edu/~lingqi/publications/paper_biplane.pdf)，SIGGRAPH 2023，11 pages | 2026-08-29 | SHA-256 `1D6FF25A62C8C14DBE6063C2F9D6DD9FB5D3BBDFFC42512A996321361BD9F73E` | 完整方法、训练、压缩、采集、结果与限制；已完整读取并视觉核对 Fig.1–9、Table 1、Eq.1–7。 |
| Supplemental `S` | [author-hosted supplemental](https://sites.cs.ucsb.edu/~lingqi/publications/supplementary_biplane.pdf)，1 page | 2026-08-29 | SHA-256 `E773A2004288D7E3F62D3A1A1DF4098AD71A2489AE62FDF2AFCD1F513E534E1A` | channel/layer 数的 validation-error ratio 消融；已视觉核对。 |
| Official code/config/data `C` | [first-author official repository](https://github.com/sssssy/Biplane_release) | 2026-08-29 | commit `e2add11c795e6003d0069d214df8c57ac4b9889b`（2025-03-12） | 已完整审计仓库中的 8 个 tracked files；确认 core decoder、biplane、adapter、dataset 与训练 loop。README 另给 pretrained checkpoints，但仓库没有 data、`render.py`、offset/normal-map implementation、环境声明或 license，不能仅凭仓库复现论文完整 pipeline。 |
| Author page/talk/correction `A` | [Beibei Wang project page](https://wangningbei.github.io/2023/BIPLANEBTF.html)；[Ling-Qi Yan publication entry](https://lingqiyan.github.io/)；[Jiahui Fan publication page](https://whois-jiahui.fun/) | 2026-08-29 | 页面无版本号 | 交叉核对标题、作者、venue、paper/supplemental/code 入口；未发现勘误。第一作者页面补出了官方代码入口。 |
| NeuralShading evidence `N` | `docs/research/prior_art.md` §3.1–3.5；`docs/realtime_material_compilation.md`；`docs/research/experiment_framework.md` | 2026-08-29 | 当前 workspace | 只用于第 13–15 节项目分析。 |

## 3. 原论文的问题、假设与贡献边界

作者认为已有 BTF neural representations 难以同时取得准确度、压缩速度、求值速度、跨材质 decoder generality 与 compression ratio。其关键假设是：去掉/另行处理 parallax 与 normal-mapping 式移动后，6D BTF 虽仍是 `ρ(u,h,d)`，但主要高频容量可放在二维 spatial `u` plane 与二维 half-vector `h` plane；difference vector `d` 的剩余影响交给小 MLP。[P §1, §3.1–3.2]

论文贡献边界：

1. H-plane + U-plane + universal MLP 的 biplane BTF 表示；[P Eq.2–4, Fig.2]
2. 对未见 BTF 冻结 MLP、只优化 planes 与 per-texel color adapter 的快速压缩 lifecycle；[P §3.2–3.3, §4.3]
3. 可选 per-BTF direct 2D offset network；[P §3.4]
4. 约 20 张手机同轴 flash/camera 图片的 lightweight acquisition，其中未观测方向由训练 BTF 的 H-plane basis prior 补全。[P §3.5, Eq.6–7]

“universal decoder”只表示同一 MLP 可解码训练分布内的多个 BTF；新材质仍需看到目标 BTF queries 并优化 per-asset representation。论文没有证明从原生材质参数零样本生成 latent，也没有保证 exotic angular-color、glints、transmission、energy conservation 或 reciprocity。[P §3.2, §5.3]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| BTF source | UBO2014 measured BTF、path-traced synthetic heightfields、手机捕获的平面实材质 | 训练集主要为 84 个 `400×400` BTF、7 categories | [P §4.1, Fig.1/6–8] |
| Runtime query | `ρ(u,ω_i,ω_o)`，改参数化为 `ρ(u,h,d)` | `u∈R²`；projected-hemisphere `h∈R²`；`d∈R²` | [P §3, §3.2] |
| Plane query | `V_u=U(u)`、`V_h=H(h)` | 两次 bilinear interpolation，各 6 channels | [P Eq.3, Fig.2] |
| Output | RGB BTF/ABRDF response `ρ`；正文 renderer 称其为 BRDF value，之后再与 stored direct lighting 相乘 | `R³`；Fig.2 base decoder final sigmoid。color adapter 位于 sigmoid 之后，是不受界的 `3×3+3` affine transform，故最终 adapted response 不保证仍在 `[0,1]` | [P §3, Eq.4, §4.4, Fig.2][C `model.py`:22–28, 43–46, 96–97] |
| Optional offset | `Δu(u,ω_o)=Θ_off(V_u^off|ω_o)` | direct 2D offset；offset plane 6 channels | [P Eq.5, §3.4] |
| Footprint/LOD | 未建模 | single spatial resolution | [P Method][I：方法 domain 直接边界] |

论文称每个 texel 的 ABRDF “可能满足也可能不满足”物理 BRDF 性质。正文 §4.4 把 network 求出的 `ρ` 称为 BRDF value，再与 stored direct lighting 相乘，因而形式上更接近 pre-cosine reflectance；但 stored-lighting buffer 是否已经包含 cosine、输入光的归一化 measure、RGB 线性空间与绝对 scale均未报告。release code 则明确在 loss 前计算 `output * cos`，其中 `cos` 是第二个方向的 z 分量；这只能作为代码 response contract，不能自动证明论文正式 checkpoint 的 ABI。[P §3, §4.4][C `dataset.py`:74–82; `trainer.py`:403–422]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

给定 `(u,h,d)`：

1. 在 `H∈R^{20×20×6}` 对 `h` bilinear fetch 得 `V_h`；
2. 在 `U∈R^{u×v×6}` 对空间位置 bilinear fetch 得 `V_u`；
3. 拼接 `V_h,V_u,d`，由共享 MLP `Θ` 输出 3-channel response；
4. 压缩新 BTF 时再做 per-texel affine color transform：`ρ=A_u Θ(V_u,V_h|d)+B_u`，其中 `A_u∈R^{3×3}`、`B_u∈R^{3×1}`；
5. 若启用 parallax，先从另一个 6-channel spatial plane 与 `ω_o` 求 `Δu`，用 `u_new=u+Δu` 查询 U-plane。[P Eq.2–5, Fig.2]

### 5.2 持久化表示

- shared：一个 universal 6-layer MLP；训练阶段跨 BTF 共享，压缩新 BTF 时冻结。[P §3.2]
- per-BTF：`20×20×6` H-plane、`u×v×6` U-plane，以及每 spatial texel 的 `3×3+3=12` color-adapter scalars。[P Eq.3–4]
- optional per-BTF：`u×v×6` offset feature plane + 一个 4-layer/32-hidden offset MLP；该 MLP不共享，每个 BTF快速 overfit。[P §3.4]
- Table 1 对 `400×400` BTF 报告 average latent dimension 18；这与 U-plane 6 + color adapter 12 + 平摊后几乎为零的 H-plane 相符。optional offset 的 6 channels 和网络显然没有包含在该 18 中，故不能用 18 表示 full parallax asset cost。[P Table 1, Eq.3–5][I：按论文尺寸算术展开]
- precision、quantization、mip chain 与 texture format 未报告。

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| H-plane | projected `h` | `20×20×6` bilinear | 无 | 6 | per-BTF | [P Eq.3, Fig.2] |
| U-plane | spatial `u` | `u×v×6` bilinear | 无 | 6 | per-BTF | [P Eq.3, Fig.2] |
| Universal MLP | `6+6+2=14` | `14→256→128→256→128→256→3`，共 6 FC；两个 `256→128→256` additive residual blocks | 前 5 个 FC 后为 LeakyReLU；final FC+Sigmoid；无 normalization | RGB base response | shared | [P Fig.2 caption/legend, §3.2][C `model.py:Decoder`, `utils.py:fc_residual_block`] |
| Color adapter | base RGB | per-texel `3×3` matrix multiply + 3-vector add；论文只称 average initialization；代码把矩阵九项全部初始化为 `0.3333`、bias 为 0，使三个输出通道初始都等于 base RGB 的通道平均，并用 bilinear interpolation | 无；affine 后无 clamp/activation | adapted RGB | per-BTF/per-texel | [P Eq.4, §3.3][C `model.py`:35–46, 64–97] |
| Offset MLP | offset feature 6 + projected `ω_o` | 4 layers，32 hidden units；确切 width sequence/activation 未报告 | 未报告 | direct 2D `Δu` | per-BTF | [P Eq.5, §3.4]；官方 release 没有 `Offset` class，无法由 `C` 补齐 |

### 5.4 条件化、坐标变换与物理先验

half/difference parameterization 把随入射/出射移动的 specular structure 部分驻定。与把完整 4D direction function 全交给 MLP 相比，H-plane 对 half-vector 高频提供显式 2D capacity；difference vector 只作 2D conditional input。[P §3.2]

论文从 NeuMIP 借用 offset 思路，但选择直接预测二维 `Δu`，没有 NeuMIP 的 scalar depth + fixed geometric `H`。作者没有提供这两种 offset parameterization 的 matched 对照，不能说 direct 2D 在 Biplane 上更优。[P §3.4][I]

color adapter 是每 texel 的一般 3×3 affine color transform，不是白平衡的三个 scale。它能显著扩展色域，也用 12 scalars/texel 成为 per-asset 最大容量之一。[P Eq.4, Fig.5]

release 的 H/U plane 与 adapter bilinear 实现用 modulo 取得右/下邻居，即在边界采用 periodic wrap；论文只写 bilinear interpolation，没有披露 address mode。这是 `C` 的可复核实现细节，不应回填为正式方法唯一规定。[C `model.py`:64–73, 163–173]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Shared-decoder training data | UBO2014：84 BTFs、`400×400` texels、7 categories | [P §4.1] |
| Split | Fig.7 与 Rainer 2020 比较时沿用其 91% training proportion；图中明确测试 `Fabric04/Stone04/Felt03/Carpet07/Wallpaper12/Carpet12`，后两项标为 unseen。完整 84-BTF split manifest、validation 比例和其他实验 split 未报告 | [P §4.1, Fig.7] |
| Dense-to-random resampling | 每个 BTF 重采样 `6.4×10^7` random queries；UV 与 incoming/outgoing directions 独立随机 | [P §4.1] |
| Synthetic GT | simulated heightfield 在 renderer 中 path tracing；renderer、spp、light、footprint 未报告 | [P Fig.6 caption] |
| New-BTF compression | sparse input；real BTF comparison 中约 400 random samples/texel | [P §5.1, Fig.7] |
| Mobile acquisition | flat sample，hand-held phone，collocated flash/camera，约 20 images；marker frame 校准与 rectification，单图变为 `400×400` queries；distance falloff 记为 `G(I_j(u))` | [P §3.5, §4.3, Eq.6] |
| Acquisition prior | target H-plane 不直接自由优化，而是训练 BTF H-planes 的全局线性组合 `H_t=Σ_i w_i H_i` | [P Eq.7] |

论文没有说明方向 random sampling 在 solid angle、projected disk 还是 half/difference domain 上均匀，也没有给 grazing oversampling；作者反而把 rough conductor 的 dark edges 归因于 grazing samples 不足。[P §5.1]

## 7. Loss、optimizer 与训练 lifecycle

| 项 | 正式配置 | locator |
|---|---|---|
| Loss | 论文没有给精确 norm/颜色空间；§3.3 只说明第一阶段 loss 使用 averaged value、第二阶段调 color。release code 先让 adapter 的全 `1/3` 矩阵把 base RGB 变为三通道相同的灰度预测，再对 `output*cos` 与 clamp 后 RGB radiance 使用 L1；它没有显式构造 average-RGB target。compression clamp 为 `[1e-5,1-1e-5]`，shared-training clamp 为 `[0,1]`。该 code-only 配方是否生成正式结果未建立 checkpoint correspondence | [P §3.3, Eq.6][C `model.py`:43–46, 96–97; `trainer.py`:160–168, 403–422] |
| Shared optimizer | AdamW | [P §4.2] |
| Shared batch | 每 step 同时取 4 个 BTF，每个 160,000 random queries，共 640,000 | [P §4.2] |
| Release batch composition | 每 dataset item 读取一个 `400×400=160,000` query 文件并随机保留一半；batch size 8，故仍为 640,000 queries/step，但 8 个 file blocks 经全局 shuffle 后不保证恰好来自 4 个不同 BTF。checked-in `train_materials` 只有 `synthetic_rock2`，所以它是 example/default，不是 formal 84-BTF batch | [C `config.py`:76–78, 96–97; `dataset.py`:50–85] |
| Shared LR/schedule | planes `1e-3`，network `3e-4`；两者 exponential schedule，`0.9`/epoch | [P §4.2] |
| Shared lifecycle | 正文为 30 epochs、RTX 2080 Ti 11 GB、约 18 h。release 的 shared-training 分支先设 `max_epochs=40`，与论文 30 不同；但 checked-in `compress_only=True` 随后把实际 entry-point 流程覆盖成 20 epochs，并加载 `epoch-35.pth`。因此 release 没有一个不改配置即可重现论文 30-epoch shared training 的 default | [P §4.2][C `config.py`:96–127; `main.py`:9–31] |
| Compression stage 1 | 15 epochs；biplane/optional offset texture 初始化为 0；planes LR `1e-2`；optional offset MLP scratch LR `3e-3`；所有 planes 做 2D Gaussian blur，kernel size `20→0`。release 以 `start_radius=20`、每 epoch 乘 `0.75` 后取 floor，并在 epoch 15 切换参数 | [P §4.3][C `config.py:RepConfig`, `trainer.py:get_radius/train`] |
| Compression stage 2 | 5 epochs；只优化 color adapter，LR `1e-2`；stage 1 adapter 冻结在 average initialization。release 确实在前 15 epochs 令 adapter LR=0，随后冻结 planes/offset 并启用 adapter | [P §3.3, §4.3][C `trainer.py:init_checkpoint/train`] |
| Acquisition lifecycle | batch 40,000；`35+15` epochs，其他设置沿 compression；capture/data prep 10 min，optimization 3.5 min/BTF | [P §4.3] |
| Seed/model selection | 论文未报告。release 的实际 entry point 调 `set_global_random_seed(seed=None)`，用当前 microsecond 生成 seed，并非固定 0；`config.py` 独立打印入口才显式传 0。默认又硬编码加载 epoch-35，但没有 checkpoint selection protocol或 formal identity | [P][C `main.py`:36–45; `utils.py`:12–20; `config.py`:127, 177–180] |

作者明确说同时优化 feature planes 和 adapter 会造成 noticeable reflectance-distribution differences，因此把 intensity/structure 与 color 拆为两阶段。这是正式 `author-negative`，不是普通实现偏好。[P §3.3]

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime path | Mitsuba 生成 UV/方向/direct-light buffers；PyTorch GPU 批量求 BTF，再乘已存 lighting | [P §4.4] |
| Time | 1920×1080、1 spp，作者实现约 1 s；硬件未在该 timing 段重述，不能自动套用训练的 2080 Ti | [P §4.4] |
| Network cost | Table 1 报 Ours `13.5M FLOPs`，NeuMIP `0.2M`、Rainer 2019 `3.5M`、Rainer 2020 `3.8M`、NLBRDF `105M` | [P Table 1] |
| FLOP boundary | release code 可核对 core network shape，但仍没有 Table 1 的 FLOP convention、texture interpolation/color adapter/offset inclusion说明；该表不能转成 shader latency排名 | [P Table 1][C `model.py:Decoder`][I] |
| Sampling | complex lighting 用 MIS，但所有 BTF 的 BRDF proposal 均用 Lambertian；作者提出 Gaussian/Blinn–Phong lobe 为未来扩展 | [P §4.4] |
| Fetches | core 至少 H-plane bilinear + U-plane bilinear；optional offset 再加一次 spatial bilinear。native texture fetch 数和cache未报告 | [P Fig.2, §3.4] |
| Precision/bytes | 未报告；Table 1 的 18 是 scalar dimension，不是 alignment/precision 后的 bytes | [P §3.1, Table 1] |

该实现离实时 shader 还有明显工程距离：1 秒是 Mitsuba-buffer + PyTorch 路径，而不是优化 kernel；但 core query 的读取数和 MLP shape静态有界，存在部署可能。它应登记为 `bounded-capacity candidate / unoptimized runtime evidence`，不能因 Python timing 否定 representation，也不能用理论 FLOPs宣称实时。[I，依据 N method constraints]

## 9. 实验 protocol、baseline、指标与结果

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| Synthetic BTF | path-traced simulated heightfields；平面/场景图 | Rainer 2020；ours no/full offset；reference | visual | no-offset 比 Rainer 清楚；offset 在有 parallax 两例最接近 reference；flat case 不启用 offset | [P Fig.6, §5.1] |
| vs NLBRDF | 一个 synthetic rough conductor BTF | Fan 2022 NLBRDF | MSE、compression time、storage | ours MSE `1.5e-4` vs NLBRDF `7.1e-5`；5 min vs 200 h/约 8 days；作者称 ours 5× smaller。protocol 只有单例。 | [P Fig.3, §5.1] |
| UBO real BTF | 6 BTFs，后两个 unseen；ours 约400 random queries/texel，Rainer 用22,801 uniform queries | Rainer 2020 pretrained | per-image MSE | Rainer→ours：Fabric04 `5.4e-4→9.0e-5`；Stone04 `2.3e-2→9.4e-3`；Felt03 `1.9e-4→2.2e-5`；Carpet07 `2.5e-4→1.6e-4`；Wallpaper12 unseen `4.1e-4→9.4e-5`；Carpet12 unseen `5.9e-4→1.7e-4` | [P Fig.7] |
| Mobile acquisition | 每材料约20 collocated images；5 leather/fabric examples；novel light/view renders | input captures；无 quantitative full-BTF GT baseline | visual | 作者判为 plausible/faithful；无方向积分误差或几何/parallax GT | [P Fig.8, §5.1] |
| H-plane prior ablation | acquisition direct H-plane vs linear combination | 两配置 | visual feature continuity/artifacts | direct optimization 产生 incomplete discontinuous H-plane 和 highlight artifacts；linear combination 更可靠 | [P Fig.4, §5.2] |
| Color adapter ablation | 2 synthetic materials，with/without adapter | reference | visual | 无 adapter 有明显 color bias；adapter 改善 | [P Fig.5, §5.2] |
| Capacity supplemental | channel `{3,6,9,12}`；layers `{4,6,8,10}`；default 6/6 | ratio to default | validation-error ratio | 从柱图读取：channels 约 `1.50/1.00/0.98/0.82×`；layers 约 `1.08/1.00/1.14/0.95×`。8 layers 比 6 layers 更差，说明结果非单调；无绝对值、error bars、seed 或读取表格 | [S Fig.1] |

Table 1 的 dot ratings 是作者定性汇总，不作为跨论文 metric。BTF主结果没有 seed、置信区间、direction-weighted integral、energy/reciprocity 或 matched hardware latency；Fig.7 还同时改变了 query distribution/count，因此只能支持论文 protocol 下的 sparse-input result，不能隔离“架构优于 baseline”的单一因果。[I]

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `author-negative` | compression 同时优化 planes 与 color adapter | reflectance distribution 出现 noticeable differences | color freedom干扰 structure/intensity fit | adapter 容量很大，必须冻结轴并分阶段 | [P §3.3] |
| `ablation-inferior` | acquisition 直接自由优化 H-plane | 未观测区域不连续，highlight artifacts | collocated captures 缺 directional information | H-plane basis 是数据先验，不是由 observation 唯一恢复 | [P §3.5, Fig.4] |
| `author-negative` | collocated acquisition 中训练 offset module | 难以收敛，acquisition 不纳入 parallax effect | input information不足 | 约20张 `d=0` 图片不能约束 view-dependent spatial warp | [P §5.3] |
| `author-negative` | rough conductor synthetic BTF | ours 有 darker grazing edges，MSE `1.5e-4` 高于 NLBRDF `7.1e-5` | random sampling 的 grazing samples不足 | 这是单例 matched 不充分的负结果；可先检验 query recipe，而非立刻增大模型 | [P Fig.3, §5.1] |
| `known-limitation` | high-specular materials | 精度低于更大 NLBRDF | universal small MLP的 trade-off | H-plane 不能完全替代 decoder capacity | [P §5.3] |
| `known-limitation` | exotic angularly varying colors | Fig.9 明显失败 | UBO2014 训练集没有同类样本 | universal decoder 的 prior 受 cohort support 限制 | [P Fig.9, §5.3] |
| `ablation-inferior` | 3 plane channels | validation error约 default 1.5× | capacity/storage trade-off | 6 只是作者 operating point，不是 universal optimum | [S Fig.1] |
| `ablation-inferior` | 8-layer MLP 相对 default 6 layers | validation-error ratio 约 1.14×，反而比 default 差；10 layers 约0.95× | supplemental 只给总体 accuracy/complexity trade-off，没有解释非单调原因 | 无 seed/error bars，不能断言更深必然更差；只能记录该次消融 | [S Fig.1] |

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Architecture | 6-channel H/U，6-layer 256/128 bottleneck MLP，adapter，optional offset | 确认 default 6 channels/6 layers | `Decoder` 为 `14→256`、两个 `256→128→256` residual block、`256→3→sigmoid`；H/U 分别为 `20×20×6` 与 `400×400×6`、默认 zero-init。adapter 为全 `1/3` 的 `3×3` matrix + zero bias。release 的 bilinear 索引使用 modulo wrap，正文未披露 address mode | core shape 对应 Fig.2；release 不含 `Offset`/`NormalMap` class，无法核对 optional branches。 |
| Loss/response | 未给精确 norm；第一阶段称按 RGB average 算 loss。正文把网络值称为 BRDF/ABRDF，并说渲染时再乘 stored direct lighting，但没有说明该 lighting 是否已含 cosine | 未补 | `L1(output*cos, clamp(RGB))`；adapter 的 average initialization 使 base output先混成三通道相同的灰色预测，但 code 没有显式 average-target loss。dataset 把文件前一对方向命名 `wi`、后一对命名 `wo`，loss 的 `cos` 取后一方向 z；这些 code 命名是否与正文入射/出射约定一致未建立 | `paper-code-gap`：code 明确是 pre-cos output contract，但 loss、方向约定和正式 checkpoint 对应都不能从 release 补成论文事实。 |
| Training | AdamW；每 step `4 BTF×160k` queries；30 epochs | 未补 | 每 item 读取 `400×400=160k` 后随机保留一半，batch 8，合计仍为 640k，但 shuffle 不保证恰好 4 个不同 BTF。shared-training 分支配置为 40 epochs；checked-in `compress_only=True` 又把实际 entry point 覆盖为 20-epoch compression，并加载 `epoch-35.pth` | `paper-code-gap`：总 query 数巧合对应；material composition 不对应，且 release 没有不改配置即可复现论文 30-epoch shared training 的 default。 |
| Compression | 15+5 epochs、blur20→0、两阶段 | 未补 | `start_radius=20`、`0.75^epoch` floor；epoch 15 后 planes/offset LR 清零并开启 adapter LR | lifecycle 大体对应，并补出确定 blur schedule；仅 core release 可审计。 |
| Runtime | Mitsuba+PyTorch约1s，Table 1 FLOPs | 未补 | `trainer.py` import 的 `render.py` 未纳入仓库；无 optimized kernel | 完整 runtime 与 FLOP convention仍不可审计。 |
| Evaluation | Fig.3–9 | size/accuracy ratios | README 只链接 pretrained checkpoints；无原始 data、split manifest、metrics/render scripts，且 `render.py` 缺失 | 无法复算图表与 metric。 |

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

- universal small MLP 对 high-specular 不如大型 NLBRDF；[P §5.3]
- exotic angularly varying color 超出 UBO2014 support，表示失败；[P Fig.9, §5.3]
- collocated acquisition 无法让 offset 稳定收敛，因此不恢复 parallax；[P §5.3]
- Lambertian proposal 不匹配 specular BTF；[P §4.4]
- mobile acquisition 是 plausible recovery，不是从 20 张图唯一确定完整 BTF。[P Abstract, §3.5]

### 12.2 未报告/材料不可得

- 正文正式实验的精确 loss、颜色/response measure、optimizer beta、seed 与 checkpoint selection；release code 的 L1/clamp、weight decay 与从当前 microsecond 派生的非固定 seed 不能自动回填到正式结果；
- UBO 全部 84 个 BTF 的 split manifest、validation protocol、random direction density；Fig.7 只明确给出六例及后两例 unseen；
- synthetic path-tracer configuration与spp；
- optional offset activation/逐层实现与正式 checkpoint 对应；core residual endpoints 已由 release code 核对；
- precision、texture bytes、quantization、mip/LOD；
- Table 1 FLOP convention与是否包含 adapter/offset；
- 官方 release 缺失的数据、`render.py`、offset/normal-map、手机 calibration、环境声明与完整运行说明；
- acquisition H-plane weights 是否有 sum-to-one/nonnegative regularization；
- MSE 聚合/颜色空间、energy/reciprocity 与 sampling variance。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

本方法不是“两个 6-channel plane”这么简单。每个 spatial texel 有 U-plane 6 scalars 和 affine adapter 12 scalars；adapter 的容量是 U-plane 两倍。H-plane 只需 `20×20×6`，把跨空间共享的 specular-angular结构以极低 per-texel摊销存下。shared MLP 仍明显大于 NeuMIP，并承担 difference-vector correction。[I，依据 P Eq.3–4, Fig.2]

### 13.2 成功所依赖的假设

- half-vector 是主要 angular high-frequency axis，difference dependence 可由 shared MLP平滑修正；
- 训练 cohort 足以约束 universal decoder prior；
- per-asset direct optimization允许看到大量 target queries；
- per-texel 12-scalar color adapter 的 memory 可接受；
- spatial resolution单一，不要求 footprint-correct LOD；
- acquisition的未观测方向可由已知 H-plane span 表示。

### 13.3 可迁移机制与不能迁移的部分

可迁移：H-plane/U-plane 非对称 factorization；shared decoder + per-asset planes；half/difference encoding；两阶段 structure→color optimization；basis-constrained unseen-direction completion；grazing-aware query adequacy test。

不能直接迁移：手机 acquisition 不能替代权威 source reference；per-texel adapter若用于1×1 LayerStack几乎只剩12个全局色变参数，意义与BTF不同；H-plane linear combination只证明采集 prior，有全量 reference queries 时未证明优于自由 H-plane；没有 LOD 与 matched sampler，不能直接成为完整产品方法。

### 13.4 与本项目 runtime contract 的关系

`prepare(u,wo)` 可缓存 U-plane、adapter 和 optional offset后的 spatial state；H-plane取决于 `h(wo,wi)`，每次 `evaluate(wi)`仍需一次 directional texture lookup + shared MLP。读取数静态有界，但比只读 latent后跑 MLP 的 evaluator多一个 wi-dependent texture fetch。[I，依据 N `realtime_material_compilation.md`]

它最适合作为 spatial asset/BTF 的 factorized evaluator candidate，以及 1×1 LayerStack 的 directional-plane capacity diagnostic。正式晋级必须按 `B_asset` 把 adapter、H/U/offset全部计入，并在同 query/budget 下与 NVIDIA hierarchical latent和纯 MLP比较。[I，依据 N `experiment_framework.md`]

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

当前 NVIDIA reproduction 使用 source-parameter encoder、hierarchical z8 latent、learned-frame `3×64` evaluator、`3×32→9` matched sampler，并按 bootstrap→latent finetune 的 lifecycle 训练；正式接口以 typed evaluator/sampler route 输出线性 `f`，sampler 使用 forward-KL 目标。[N `docs/learning.md`]

当前代码与实验并未以 Biplane 为复现目标，也不存在 Biplane paper/config/checkpoint correspondence。因此下表描述的是候选机制是否已经进入当前 NVIDIA faithful identity，而不是把两种不同方法之间的差异误记成 implementation deviation；所有潜在收益均属于 `[I] candidate-transfer`，不是已发现 defect。[N/I]

| Biplane 机制 | 当前 NVIDIA 对应 | 判定 | 影响 |
|---|---|---|---|
| H-plane angular texture | 当前 learned frame 后仍由 MLP直接解码方向；没有 H-plane | `not-applicable` | `[I] candidate-transfer`：在相同 asset bytes、单次查询时间和训练预算下检验显式 half-vector field 是否比网络容量更划算。 |
| U-plane + shared decoder | 当前 hierarchical z8 是 source-state latent，不是 BTF spatial U-plane | `not-applicable` | `[I] candidate-transfer`：若以后扩展 spatial asset，再比较 single-scale U-plane 与项目的随机访问/LOD 合同；当前不能声称实现或偏离 Biplane。 |
| per-texel affine adapter | 当前无显式 `3×3+3` color head | `not-applicable` | `[I] candidate-transfer`：可作为结构/色彩解耦消融，但必须完整计入 asset bytes，并检查它是否掩盖 directional error。 |
| H-plane basis completion | 当前 encoder 从 native source parameters 生成 latent，不做稀疏图像逆采集 | `not-applicable` | `[I] candidate-transfer`：一个是 inverse-capture prior，一个是 source compiler；只有另设共同 target-visible direct-fit 轴时才可比较。 |
| Lambertian sampler | 当前是与 evaluator 配套的 learned `3×32→9` sampler；Biplane 使用 Lambertian proposal | `not-applicable` | `[I] future integration`：若将 H-plane 引入候选，仍应遵守项目的 matched sampler policy；这不是当前 NVIDIA identity 的 deviation。 |

该论文没有指向当前 NVIDIA 复现的实现 defect；它提出的是新的 representation axis。任何“改进 NVIDIA”结论都需保持当前 formal source/query/lifecycle，加入 H-plane 后做 matched Pareto，而不是拿论文 Fig.7 的不等 query count 比较。[N `experiment_framework.md` §2/§5][I]

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：显式 half-vector plane 可在 iso-byte/iso-time 下改善窄峰 | Biplane H-plane动机；NBRDF/当前项目均重视half/difference | LayerStack方向高频可跨state共享低分辨率basis | NVIDIA direct vs `H-plane+smaller decoder`，两组预算对照 | source/query/train steps/precision/sampler | solid-angle L1、peak location/height、bytes、GPU time | evaluator candidate | 主质量CI无改善或texture fetch使Pareto被支配 |
| H2：structure-first、color-second 比联合优化更稳定 | paper联合优化 author-negative | 当前native source的chromatic scale也会干扰direction structure | joint vs two-stage，same total updates | initialization/query/optimizer work | seed variance、convergence AUC、energy/color error | training lifecycle | two-stage没有降低variance或损害最终quality |
| H3：grazing-aware sampling比增大模型更能修复dark edge | paper粗导体失败归因grazing样本不足 | 当前难例也存在query-density不足 | uniform recipe vs matched grazing mixture，再与larger model对照 | total queries/model bytes/steps | grazing-bin error、global metrics、cost | query recipe | reweight后grazing误差无显著下降或全局显著退化 |
| H4：低秩/稀疏 H-plane dictionary 可替代自由plane | acquisition linear combination成功 | 跨材质half-vector结构存在共享basis | free H-plane vs full linear basis vs top-2 sparse basis | shared basis bytes与asset coefficient bytes明确配平 | quality、compression time、G2/W、runtime fetches | compiler/direct-fit candidate | basis方案被free plane在同bytes/time下支配或未见材质崩溃 |
| H5：per-state affine RGB adapter只需低秩/diagonal即可 | full 3×3+3 adapter修复color bias但占12 scalars/texel | 大部分source色差不需要任意channel mixing | no/diagonal/low-rank/full adapter | decoder/latent/query fixed | chroma error、direction error、bytes | capacity diagnostic | full adapter的优势无法由更小adapter保持，或adapter改善只是泄漏/过拟合 |

## 16. 证据索引

- `P`：§3.1–3.5（Eq.1–7、Fig.2）；§4.1–4.4（data/training/compression/acquisition/rendering）；§5.1–5.3（Fig.3–9、negative results/limitations）；Table 1。
- `S`：1 页 supplemental Fig.1，channels/layers validation-error ratio。
- `C`：官方 repository commit `e2add11c795e6003d0069d214df8c57ac4b9889b`：`README.md`；`model.py:10–28, 35–46, 64–97, 101–200`；`utils.py:12–20, 59–89, 112–116`；`config.py:76–127, 177–180`；`dataset.py:50–85`；`trainer.py:16, 160–168, 403–422`；`main.py:9–45`。README 的 Google Drive checkpoint 未下载；仓库只有 8 个 tracked files，且 `trainer.py` 无条件 import 的 `render.py`、optional modules、data、环境声明与 license 均缺失，按 clone 状态入口不可直接运行。
- `A`：Beibei Wang project page；Ling-Qi Yan publication entry；Jiahui Fan publication page（官方 code locator）。
- `N`：`docs/learning.md`、`docs/realtime_material_compilation.md`、`docs/research/experiment_framework.md`、`docs/research/prior_art.md`。
- `I`：第 8–15 节的成本边界、项目映射和迁移假设。

## Evidence review

```text
author_worker: /root
reviewer: /root/rta2024
reviewed_at: 2026-08-29
sources_rechecked: [main PDF SHA-256 1D6FF25A62C8C14DBE6063C2F9D6DD9FB5D3BBDFFC42512A996321361BD9F73E all 11 pages/full text/Fig.1-9/Table 1/Eq.1-7, supplemental PDF SHA-256 E773A2004288D7E3F62D3A1A1DF4098AD71A2489AE62FDF2AFCD1F513E534E1 one page/Fig.1, first-author official repository commit e2add11c795e6003d0069d214df8c57ac4b9889b all 8 tracked files, N docs/learning.md and docs/research/experiment_framework.md]
findings_closed: [H/U planes and decoder exact layer structure, adapter all-1/3 initialization and two-stage lifecycle, paper 30 epochs vs shared branch 40 vs checked-in 20-epoch compression entry point, paper 4 BTF×160k vs code batch 8×80k, L1×cos response and direction-name caveat, missing Offset/render/data/environment/license, all six formal Fig.7 numbers and supplemental ratios, negative/ablation/limitation classification, NVIDIA N/I boundary]
remaining_evidence_gaps: [pretrained checkpoint not downloaded and formal-result correspondence unknown, formal loss/response/stored-lighting measure unresolved, offset implementation unavailable, full split/data/render/mobile-calibration assets unavailable, Table 1 FLOP convention unreported]
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
