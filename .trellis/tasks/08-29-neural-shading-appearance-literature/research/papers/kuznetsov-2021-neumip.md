---
paper_id: "kuznetsov-2021-neumip"
title: "NeuMIP: Multi-Resolution Neural Materials"
authors: "Alexandr Kuznetsov; Krishna Mullia; Zexiang Xu; Miloš Hašan; Ravi Ramamoorthi"
year: "2021"
venue: "ACM Transactions on Graphics 40(4), SIGGRAPH 2021, Article 175"
doi: "10.1145/3450626.3459795"
report_status: "evidence-reviewed"
main_source: "https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP/assets/neumip_final.pdf"
supplemental_status: "available"
official_code_status: "audited"
official_code_commit: "c1e2f2aa3488b7460cbf19f5bf6d1c4343926178"
author_worker: "/root"
reviewer: "/root/nbrdf2021"
last_verified: "2026-08-29"
---

# NeuMIP：多分辨率神经材质

## 1. 研究对象与报告边界

NeuMIP 要表示的是带空间变化、方向变化和空间 footprint 的复杂表面外观。论文把目标写成一个 7D multi-resolution bidirectional texture function（MBTF）：二维表面位置 `u`、二维入射方向 `ω_i`、二维出射方向 `ω_o`，以及一个空间过滤尺度 `σ`。它用逐材质优化的 neural texture pyramid 保存主要容量，用两个很小的 MLP 分别完成 view-conditioned 坐标偏移和 RGB reflectance 解码。[P §3, Eq.1–8, Fig.2–4]

本报告覆盖作者项目页所链接的 TOG/SIGGRAPH 2021 正式 PDF、项目页、公开数据/模型入口以及 `straintensor/NeuMIP` 官方代码当前固定 commit。它属于本任务的 local/spatial neural material 主线，并直接关联 LOD、prefilter、`prepare()` 复用和有界 shader 求值；它不是跨材质 compiler，也不学习 scene-level visibility/transport。论文中的 microgeometry path tracing 是生成单个材质 MBTF reference 的手段，不等于运行时仍追踪该微结构。[P §3.1, §4–5]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | [作者项目页 PDF](https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP/assets/neumip_final.pdf)，TOG 40(4), Article 175, 13 pages | 2026-08-29 | SHA-256 `70504A05BD201FD4E1F8D5C77A4978DCC792CB1F8461221BC433BC4E13D3073A` | 正式方法、公式、实验和限制；已完整读取并视觉核对 Fig.2–11、Eq.1–8、Table 2–4。 |
| Supplemental `S` | 作者项目页链接的 [`assets/neumip_sig2021_final.mp4`](https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP/assets/neumip_sig2021_final.mp4)；正文多次要求结合动画观察视差 | 2026-08-29 | HEAD 为 HTTP 200、`video/mp4`、173044121 bytes；未下载，故无本地 hash | 已确认真实可用性；未把视频目测作为网络配置或定量结果证据。没有发现单独的 supplemental PDF/appendix。 |
| Official code/config/data `C` | [GitHub `straintensor/NeuMIP`](https://github.com/straintensor/NeuMIP)，`master` | 2026-08-29 | commit `c1e2f2aa3488b7460cbf19f5bf6d1c4343926178`，2022-07-12 UTC；BSD-2-Clause | 审计 README 指定命令、网络、texture pyramid、loss、optimizer、训练 loop 和 export 路径。README 的 Google Drive datasets/models locator 在本次复核时对匿名 GET 返回 HTTP 404，不能登记为当前可用资产，也没有做端到端复现。 |
| Author page/talk/correction `A` | [NeuMIP project page](https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP/) | 2026-08-29 | 页面无版本号 | 交叉核对标题、作者、SIGGRAPH 2021、paper/arXiv/demo/code/data入口。没有发现勘误。 |
| NeuralShading evidence `N` | `docs/research/prior_art.md` §3.2；`docs/learning.md`；`docs/realtime_material_compilation.md`；`docs/research/experiment_framework.md` §1/§4/§7；`.trellis/spec/project/method-constraints.md` | 2026-08-29 | 当前 workspace | 只用于第 13–15 节的项目映射，不反推论文事实。 |

访问边界：作者项目页仍显示 “Code (coming soon)”，但作者本人仓库和项目 README 已公开；因此以仓库作者身份、README 回链项目页和源码内容共同判作 official code。论文正文没有 optimizer、learning rate、seed、split 或数据资产逐项清单；公开代码补充的是 2022 年的后续公开实现快照，不能静默覆盖正文。README 虽保留 datasets/models 链接，但该链接当前为 404，故只记录 locator 与失效状态。[A project page][C README.MD]

## 3. 原论文的问题、假设与贡献边界

作者的问题设定是：传统 mipmap 可以过滤 diffuse color，却难以对 normal、self-shadowing、fiber、复杂 microgeometry 与方向 reflectance 做统一预过滤；直接离散 7D MBTF 的存储又不可接受。论文假设材质在 `uv` 平面上可平铺，运行时查询的 footprint 可用二维 Gaussian 的标准差 `σ` 表达，光照是从 `ω_i` 入射到 reference plane 的单位 irradiance，输出是朝 `ω_o` 的 RGB reflectance。[P Abstract, §3.1]

论文贡献边界有三层：

1. 定义并拟合 `M(u,σ,ω_i,ω_o)`，把 BTF 扩展为显式的多尺度查询；[P Eq.1]
2. 用每一级独立优化的 neural texture pyramid 加 4-layer material MLP，避免存储完整 7D 表；[P §3.2, Eq.2–3]
3. 用只依赖 `u,ω_o` 的 neural offset 先移动 feature lookup，再解码方向 reflectance，以较小网络表达 parallax/occlusion。[P §3.3, Eq.4–8]

作者没有声称：跨材质使用同一 decoder、从原生材质参数一次前向编译未见材质、产生与 evaluator 匹配的 learned sampler、严格保持 reciprocity/energy conservation，或支持 transmissive/transparent output。MBTF 因遮挡和多次反射通常不 reciprocal，这是目标函数的属性而不是作者报告的实现 bug。[P §3.1]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | 合成 microgeometry（多数由高分辨率 heightfield mesh + albedo/roughness/metallicity/micro-normal textures 驱动）；Basket Weave 使用真实 fiber geometry；另支持 UBO 2014 measured BTF | 每个材质单独生成/拟合 | [P §4, Fig.7–8, Fig.12–13] |
| Runtime query | `q=(u,σ,ω_i,ω_o)` | `u∈R²`；`σ∈R+`；`ω_i,ω_o` 在上半球 | [P Table 1, Eq.1–2] |
| Direction encoding | 每个方向用 projected hemisphere 上的二维点，合计 4 scalars | `R⁴`；公开代码直接拼接两方向的 `x,y` | [P §3.2][C `angular.py:AngularSimple.convert`] |
| Spatial filter | `G(u,σ;x)` 为归一化二维 Gaussian；continuous LOD `l=log2(σ)` | 两个相邻 pyramid levels 空间 bilinear、scale linear | [P Eq.1, Eq.3] |
| Output quantity | 单位 distant-light irradiance 下的 RGB reflectance，即 MBTF value | `R³_+`，paper 在 `log(x+1)` domain 学习 | [P §3.1–3.2] |
| Validity/domain restrictions | infinite tiled `uv`；opaque reflectance；正文未给 transmission/alpha；方向投影假定上半球 | texture lookup wrap-around；offset 的 `ω_o,z` 分母 clamp 到 `0.6` | [P §3.2–3.3, Eq.6–7] |

方向符号存在可定位的论文内部错误，而不是另一套 convention：Table 1 定义 `ω_i` 为 incoming/light、`ω_o` 为 outgoing/view；Eq.4–8 与 §3.3 右栏正文都用 `ω_o` 驱动 neural offset。相反，Fig.2 的图内 `O(u,ω_i)` 和输入圆、Fig.2 caption 的 “incoming direction”、Fig.4 的图内 `ω_i` 与 caption 的 “incoming direction”，以及 Fig.9 Stage 1 图内 `F_off(ψ,ω_i),H(ω_i,r)` 都写成了 `ω_i`。Fig.9 caption/§6 又明确 Stage 1 只依赖 camera direction、与 light direction 无关；公开代码也把 `camera_dir` 送入 offset network。因此本报告采用 Eq.4–8 的 `ω_o`，并把 Fig.2/Fig.4/Fig.9 标作 published-figure-notation conflict。[P Table 1, Fig.2, Fig.4, Fig.9, Eq.4–8, §6][C `NeuralMaterialSavable.evaluate: camera_dir`]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

reference 先定义 finest BTF `B(x,ω_i,ω_o)`。给定 footprint，正式 target 是：

`M(u,σ,ω_i,ω_o)=∫_{R²} G(u,σ;x) B(x,ω_i,ω_o) dx`。[P Eq.1]

baseline 从 neural pyramid 取 feature `v=P(u,σ)`，再求：

`M=F(P(u,σ),ω_i,ω_o)`。[P Eq.2]

连续 LOD 令 `l=log2(σ)`，分别在 `T_floor(l)` 与 `T_ceil(l)` 做 bilinear fetch，再按 fractional `l` 线性混合。每个 level 都是独立参数，不由 finest level downsample 得到。[P Eq.3, §3.2]

full model 先从单层 offset texture `T_off(u)` 取 7D feature，与 view direction 一起输入 `F_off`，得到 scalar ray depth `r`。固定函数把它转换为：

`O(u,ω_o)=r/max(ω_o,z,0.6)·(ω_o,x,ω_o,y)`，`u_new=u+O(u,ω_o)`；

最终 `M=F(P(u_new,σ),ω_i,ω_o)`。[P Eq.5–8]

### 5.2 持久化表示

- 一个材质拥有一套独立的 feature pyramid `P={T_s}`，每级分辨率 `2^s×2^s`，每 texel 7 channels；各级独立优化，运行时用相邻两级查询。[P §3.2, Fig.2]
- 一个材质另有一个 7-channel neural offset texture `T_off`；正文称其为 bilinear-only、无 pyramid。[P §3.3, Fig.4]
- 两个 per-material MLP：offset MLP 与 material MLP。论文没有 shared-across-material decoder；Table 4 只在部署计数时忽略比较方法的 encoders，不代表 NeuMIP 有训练期 encoder。[P Table 4, §7 Network Size]
- 表中合计 texture channels 为 14；network weights 为 3332；不含随分辨率增长的 texel 参数字节。论文没有报告量化，代码用 FP32 training tensor，CUDA runtime precision 未报告。[P Table 4][C `NeuralMaterialLive.get_type_device`]

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Pyramid lookup `P` | `u_new,σ` | 每个 level bilinear；`log2(σ)` 上邻级 linear interpolation；wrap | 无 normalization | 7D feature | per-material | [P Eq.3][C `MipmapTexture.evaluate_at_points`] |
| Offset lookup `T_off` | `u` | 单层 7-channel texture bilinear lookup | 训练时随 `σ(t)` Gaussian blur | 7D feature `ψ` | per-material | [P §3.3][C `NeuralOffset.get`] |
| Offset MLP `F_off` | `ψ(7)+ω_o,xy(2)=9` | `9→25→25→25→1`，四个 affine/1×1 conv | 三个 hidden ReLU；paper 写“in between” | scalar `r` | per-material | [P §3.3][C `FullyConnected1`, `NeuralMaterialSavable.init`] |
| Fixed offset `H` | `r,ω_o` | `r/max(ω_o,z,0.6)*(ω_o,x,ω_o,y)` | hard clamp in denominator | 2D `uv` offset | fixed/shared | [P Eq.6–7][C `NeuralOffset.calculate_offset`] |
| Material MLP `F`（paper 描述） | feature 7 + directions 4 = 11 | `11→25→25→25→3` | paper：包括 final 在内均 ReLU；对 reflectance 另施加 `log(x+1)` compression | 3-channel、非负的 compressed RGB；论文未单列逆变换公式 | per-material | [P §3.2, final PDF p.4] |
| Material MLP（官方 README 指定的公开训练配置，code-only） | feature 7 + directions 4 = 11 | `11→25→25→25→6`；前 3 为颜色，后 3 为逐 RGB shadow-mask logits | hidden ReLU；无 final ReLU/normalization；前三通道先 YUV→RGB、逐通道 `exp-1`，再乘 `0.1+sigmoid(mask)`；inference 最后 clamp | RGB | per-material | [C README.MD][C `experiments/simple.py:StandardRawLongShadowMaskOnly`][C `NeuralMaterialSavable.evaluate`] |

Table 4 的 `3332` 计数可逐项复核（代码的 Conv2d 皆有 bias）：

- offset MLP `9→25→25→25→1`：`(9×25+25)+2×(25×25+25)+(25×1+1)=1576`；
- 公开配置的 material MLP `11→25→25→25→6`：`(11×25+25)+2×(25×25+25)+(25×6+6)=1756`；两者合计恰为 `3332`；
- 若严格按论文文字的 3-output material MLP，末层为 `25×3+3=78`，material MLP 为 `1678`，总计只有 `3254`。

所以 `3332` 与公开代码的 6-output head 有唯一的数值对应，而与正文 3-output architecture 不相容。[I] 这强烈支持 Table 4 的计数包含额外三通道，但论文和仓库都没有提供 Table 2/3 checkpoint identity，仍不能进一步断言 README 配置就是生成论文全部图表的配置。该项保留为 load-bearing `paper-code-gap`，不以推断抹平。[P §3.2, Table 4][C `FullyConnected1`, `StandardRawLongShadowMaskOnly.shadow_mult=3`]

另一个较小但明确的 correspondence 差异是：公开代码把 offset MLP 的 raw scalar 先乘 `0.1` 再交给固定函数 `H`；论文 Eq.5–7 直接把 `F_off` 输出定义为 `r`，没有单列这个尺度因子。它可被吸收到最后一层权重中，因而不改变函数族，但忠实复现代码参数化时必须保留。[P Eq.5–7][C `NeuralMaterialSavable.evaluate: neural_offset_aux * .1`]

### 5.4 条件化、坐标变换与物理先验

关键结构先验不是网络深度，而是 view-conditioned coordinate warp：一个 scalar depth 加局部平面几何公式约束了二维 offset 的方向。作者明确报告，直接预测 unconstrained 2D offset 的结果更差；固定 `H` 让模型更 geometry-aware。[P §3.3]

该先验不使用 height/normal supervision。对 fiber 或 participating microstructure，作者也不要求存在严格的 heightfield；`r` 是为 appearance regression 服务的隐变量，不应解释成已恢复真实表面。[P Fig.3 caption, §3.3]

方向没有 half/difference warp、Fourier/SH encoding，也没有 analytic BRDF core。[P §3.2][I] 从数据流看，feature pyramid 承担空间/尺度容量，MLP 承担方向解码；这是本报告对表示分工的分析，不是作者给出的容量归因消融。

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset/source assets/scenes | 合成 heightfield/material textures；Basket Weave fiber geometry；UBO 2014 measured BTF。正式 train/test 材质逐项文件清单未报告。 | [P §4, §7, Fig.7–8, Fig.12–13] |
| GT/reference renderer or measurement | CPU standard path tracer + custom camera-ray generator；smoothed directional distant light，在 reference plane 上单位 irradiance；多数 synthetic 数据使用 commercial renderer，Basket Weave 用 PBRT；UBO 使用原始 measured BTF。commercial renderer 名称未报告。 | [P §3.1, §4] |
| Reference sampling | synthetic MBTF 每 query 64 samples；约 32-core CPU 30 min。distant light 有有限 directional smoothing kernel，以改善收敛并可做 MIS；kernel 宽度未报告。 | [P §3.1, §4] |
| Train/validation/test split | 未报告正式 split、holdout rule 或独立 seed。结果按若干具名材质展示。 | [P §7] |
| Spatial/directional sampling | 训练输入是 random 7D queries；论文结论处称约 `200–400` random queries/texel。`u`、方向、LOD 的确切概率分布未报告。 | [P §4, §8] |
| Filtering/LOD/footprint | GT 是 Gaussian footprint；network 输入 kernel radius；独立 mip levels 联合训练。 | [P Eq.1, §3.2, §4] |
| Online/offline generation | 先离线预生成 MBTF query 数据，训练代码读取 dataset；不是 runtime 在线追踪 microgeometry。公开代码数据为 HDF5/PyTorch 文件路径。 | [P §4][C `dataset/dataset_reader.py`] |

论文的“200–400 queries per texel”与“一个 batch 约 `2^20` queries”不是矛盾：前者描述总输入数据库相对 finest texel 的密度，后者描述训练时一次图像式 batch 覆盖的像素 queries。正文没有给数据集总 query 数、每个 LOD 的比例或 query 重采样次数。[P §4, §8]

## 7. Loss、optimizer 与训练 lifecycle

| 项 | 正式配置 | locator |
|---|---|---|
| Target transform/output transform（paper） | 对 RGB reflectance 施加 `log(x+1)` compression；final ReLU 保证非负。loss 的具体 norm 未报告。 | [P §3.2] |
| README 推荐 loss（code） | `--loss comb2`；`MSE(log1p(clamp(pred,-0.1)), log1p(clamp(gt,-0.1))) + 0.1·L1(pred,gt)`。README 也允许 `l1`，称 `comb2` 对 specular 更好。 | [C README.MD][C `NeuralMaterialLive.train_step`] |
| Code output transform | 前三通道 YUV→RGB，`exp(raw)-1`；乘逐通道 `0.1+sigmoid(aux)` shadow mask。训练期允许负值到 loss 内 clamp；只在 inference clamp RGB ≥0。 | [C `NeuralMaterialSavable.evaluate`] |
| Optimizer | paper 未报告；README 指定的公开配置使用 Adam，constant base LR `1e-3`，未见 scheduler。 | [C `StandardRawLongShadowMaskOnly.learning_rate`][C `NeuralMaterialSavable.init`] |
| Batch/query count | paper：每 batch 约 `2^20` queries；代码 README：`--batch 4`，dataset item 是整张 `H×W` query image，因此真实 queries/batch 取决于 dataset resolution。 | [P §4][C README.MD][C `Dataset.__getitem__`] |
| Steps | 通常 30,000 iterations until convergence。 | [P §4][C README.MD] |
| Latent regularization | neural textures 训练初始 Gaussian `σ_i=8` texels，按半衰期 3333 iter 的 `8·2^{-t/3333}` 衰减。 | [P §4] |
| Code blur lifecycle | README 配置令 `sigma_start_radius=8, sigma_1_time=10000`，实现为 `σ(t)=8·exp[-t/(10000/ln 8)]=8·2^{-t/3333.33}`：与 paper 的 3333 half-life 数值一致，并在 10k 达到 1。feature-pyramid lookup 继续使用逐渐小于 1 的 `σ`，全局只 clip 到 0.1；只有 offset lookup 把 `σ` clamp 到至少 1，并在 `iter_count>10000` 冻结 offset texture。另在前 3000 iter 每 100 iter 对每个 feature level 做一次 `0.9·T+0.1·GaussianBlur_{1.03}(T)`。 | [C `StandardRawLongShadowMaskOnly`][C `get_sigma`, `NeuralOffset.get_sigma`, `evaluate`, `MipmapTexture.fuse_blur`] |
| Initialization | paper 未报告；代码 feature levels 独立 `N(0,1)·0.1`，offset texture 为零；weights 使用 PyTorch Conv2d default。seed/model selection 未报告。 | [C `MipmapTexture.__init__`, `NeuralOffset.__init__`] |
| Hardware/training time | RTX 2080 Ti：max 512² 约 45 min，1024² 约 90 min；30k iterations。 | [P §4] |

代码每 step 从 `[5,5,5,7,max_levels]` 随机选择可访问的 maximum level，并对超过当前 maximum 的 footprint 用 mask 去除；这一 progressive/subsampled level recipe 未出现在论文正文。[C `NeuralMaterialLive.train_step`, `get_pixels_weight`] blur 主指数 schedule 不是 paper-code conflict：`sigma_1_time=10000` 只是把“从 8 衰减到 1 的时刻”参数化，代数上得到约 3333.33 的半衰期。真正的 code-only lifecycle 是 feature/offset 在 `σ<1` 后采取不同下限、10k 后冻结 offset，以及前 3k 的周期性 in-place feature blur；论文未说明这些细节。[P §4][C `get_sigma`, `NeuralOffset.get_sigma`, `evaluate`, `MipmapTexture.fuse_blur`]

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path | Mitsuba：逐 bounce 收集 material query buffer，批量送 PyTorch/GPU；OptiX：CUDA 直接实现，每个 shading event 无 batching 调用。 | [P §5] |
| Path-tracing sampling | indirect direction 只做 cosine-weighted hemisphere；每个 shading point 最多两次 evaluator（light sample 与 BRDF sample）。没有 learned/matched `sample/pdf`。 | [P §5, §7 Limitations] |
| Parameter count | 两个 MLP 合计 3332 weights；7+7 texture channels。texture texel 总量和 bytes 随最高分辨率变化，正文未换算总 bytes。 | [P Table 4] |
| Fetches | 正文 full query：offset texture bilinear；feature pyramid 在相邻两级各 bilinear，再插值。论文没有按 native texture sample 指令统计 fetch 数。 | [P Fig.2, Eq.3–8] |
| Precision/quantization | 未报告；作者明确说未用 Tensor Cores。公开 PyTorch code 为 FP32。CUDA storage/compute precision未报告。 | [P Table 4 note][C `get_type_device`] |
| 1920×1080 timing | RTX 2080 Ti，一 query/pixel 的 evaluator batch：正文报总 5 ms。Fig.9 的四个标签分别是 offset texture `0.3 ms`、offset MLP + fixed `H` `2.3 ms`、feature pyramid `0.3 ms`、material MLP `2.3 ms`；显示值因各自只保留一位小数而相加为 5.2 ms，不能反向当成更精确的总计。 | [P §6, Fig.9] |
| Amortization | Stage 1 只依赖位置、view 和 footprint，可缓存；多灯只重跑 Stage 2 material MLP。这里的 5 ms 是单 query/pixel；论文未单独给缓存命中、多灯数量和 end-to-end frame timing。 | [P §6, Fig.9] |
| Interactive path tracer | OptiX，1920×1080，one path/pixel/frame，60 FPS；场景、bounce depth、denoising、材质覆盖和完整 frame breakdown 未报告。 | [P §6] |

5 ms 是 dense evaluator evaluation；60 FPS 是 OptiX path tracer 的完整帧率陈述（one path/pixel/frame）。两者是不同 measurement scope，不能直接相减推出其余 renderer 开销，也不能把 5 ms 写成 end-to-end frame time。论文还把 Rainer et al. 2019 的 92 ms 与本方法 5 ms并列，但没有重述比较实现的 precision/coherence；该数字只保留为论文内同图对照，不进入跨论文速度排名。[P §6]

从本项目合同看，NeuMIP 的 MLP 和 texture accesses 都静态有界；但 pyramid level 数和 per-asset texture bytes 随 asset resolution 增长，且“两个邻级 bilinear”需要编译器明确固定最大 level、资源布局和 LOD clamp。它是有界随机访问 program，不是常量 asset size。[I，依据 P Eq.3 与 N method constraints]

## 9. 实验 protocol、baseline、指标与结果

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| 合成材质主比较 | 5 materials 映射到平面，斜视，单 directional light 稍偏左；reference 为 synthetic microstructure path tracing | Rainer 2019、Rainer 2020、NeuMIP w/o offset、full NeuMIP | image MSE×`10^-3`、LPIPS | full NeuMIP 在 5/5 项的 MSE 和 LPIPS 都低于两个 Rainer baselines；相对幅度见 Table 2。 | [P Fig.6, Table 2] |
| Neural offset | 同一 5-material protocol，baseline vs full | w/o offset | MSE/LPIPS + visual | full 的所有 Table 2 MSE/LPIPS 均优于 w/o offset；例如 Wool Twisted MSE `11.31→3.98`、LPIPS `0.333→0.151`。这是 `author-positive`，不是独立跨资产 generalization 证据。 | [P Table 2] |
| Multi-resolution | Fig.11 的 5 materials，LOD 0–7 | reference | MSE×`10^-3` | 大体随更 coarse LOD 降低；并非严格单调，例如 Wool Twisted LOD4 `0.74` 后 LOD5/6 回升至 `1.16/1.17`。作者解释 coarse target 高频更少、较易优化。 | [P Table 3, Fig.11] |
| Network capacity ablation | 默认 7 channels/texture、4 layers；用 ratio-to-default MSE 曲线 | channel 3–13；layer 2–8 | MSE ratio | 低于 7 channels 或少于 4 layers 明显更差；增大到 9–13 channels 收益小；5–8 layers 围绕默认上下波动。论文未给每点绝对数或误差条。 | [P Fig.10, §7 Ablation] |
| Real measured BTF | UBO 2014 原始数据，非 tileable；平面与非平 cloth shape | reference images | per-image MSE | 5 个示例 MSE 约 `2.69e-4–6.39e-4`；未给 neural baseline 或 independent split。 | [P Fig.7–8] |
| Network size | deployment decoder weights only；比较方法的 encoders 不计 | Rainer 2019/2020 | weights, texture channels | 3332 vs 35725/38269 weights；14 vs 14/38 texture channels。不同方法的 texture spatial resolution/bytes 未在表中统一，因此只支持 paper 的 decoder-count claim。 | [P Table 4] |
| Runtime | 1920×1080，RTX 2080 Ti，一 query/pixel | Rainer 2019 | ms | 5 ms vs 92 ms；NeuMIP CUDA path 未用 Tensor Cores。 | [P §6, Fig.9, Table 4 note] |

论文没有跨随机 seed、置信区间、方向积分指标、能量误差或 reciprocity 指标。Table 2 是固定 render view/light 的 image metric；它不能单独证明全 7D domain 的误差分布。[P §7][I]

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `author-negative` | 直接回归 unconstrained 2D `O(u,ω_o)`，不经过 scalar `r` 与固定 `H` | 结果更差 | hard-coded geometry-aware constraint 减轻 offset regression | 这是“结构先验替代网络容量”的直接证据，但只对局部平面式 parallax warp 成立。 | [P §3.3 after Eq.7] |
| `author-negative` | 单独优化 neural feature vectors，不做逐步 Gaussian smoothing | texture 出现类似 Monte Carlo noise，offset texture 尤其明显 | 每个 texel 自由优化导致 noisy latent | 与高分辨率 latent initialization noise 高度相关；不能扩展成“所有 autodecoder 都失败”。 | [P §4 Training] |
| `ablation-inferior` | 去掉 neural offset | 5 个主示例的 MSE/LPIPS 都差，shadow/parallax 更弱 | baseline 难以用小 MLP 学习非平材质的 angular stability | offset 让 view-dependent spatial transport 先变成坐标对齐，再交给 decoder。 | [P Fig.6, Table 2] |
| `ablation-inferior` | 少于 7 channels/texture | Fig.10 MSE ratio 较高 | 容量不足 | 没有 absolute bytes/error bar，不把 7 写成通用最优。 | [P Fig.10, §7] |
| `ablation-inferior` | 少于 4 MLP layers | Fig.10 明显较差 | 容量不足 | 更深网络收益不稳定，4 层是该 protocol 的 speed/storage trade-off。 | [P Fig.10, §7] |
| `known-limitation` | hard shadow discontinuity | full model 仍有 minor loss in shadow contrast | discontinuity 难以完美学习 | `log` transform/blur 与确定性小 MLP 都会压低极尖锐变化，但具体因果未由作者拆分。 | [P Fig.6 caption, §7] |
| `known-limitation` | very specular/glinty material | 不 blur 会难处理；当前方法未展示 | 作者提出随机 decoder + GAN 生成统计细节 | 这会改变 deterministic `evaluate` 语义，不能直接进入当前 scattering ABI。 | [P §7 Limitations, §8] |

在已获得第一方材料中，未报告以下尝试是否做过：shared decoder、encoder initialization、quantization、half/difference directions、analytic BRDF residual、energy/reciprocity regularization、matched neural sampler、alternative footprint kernels。不能从最终设计推断这些尝试失败。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Architecture | 7+7 texture channels；两个 4-layer/25-wide MLP；material 3-output RGB；final ReLU；Table 4 报 3332 weights | 视频只展示外观动态，不补配置 | README 指定配置的 material head 有 6 outputs，其中 3 个为 shadow-mask logits；无 final ReLU，使用 YUV→RGB/exp/mask/inference-clamp path | `paper-code-gap`：§5.3 算术证明 3332 与 6-output code 精确对应、与 3-output 正文不相容；但缺 checkpoint identity，不能断言论文全部结果使用 README 配置。 |
| Offset conditioning | Eq.4–8 使用 `ω_o` | 动画展示 view-dependent parallax | code 使用 `camera_dir`，并额外把 raw scalar 乘 0.1 | code 与公式的方向语义一致；Fig.2/Fig.4/Fig.9 的 incoming/`ω_i` 是 published-figure-notation conflict；0.1 是 code-only 参数化。 |
| Pyramid | 每级独立，continuous LOD trilinear | 未补 | 独立 tensor，两个邻级 linear blend | 对应；代码另有 max-level stochastic training recipe。 |
| Offset texture | 单层 bilinear，无 pyramid | 未补 | `number_mip_maps_levels=1` | 对应。 |
| Blur | initial 8，half-life 3333，无后续细节 | 未补 | README 配置的指数式代数上是 half-life 3333.33、10k 到 1；feature 可继续降到 0.1，offset clamp 到 1 且 10k 后冻结；前 3k 另有周期性 feature `fuse_blur` | 主指数 schedule 数值对应；下限、freeze 和 in-place blur 是 code-only lifecycle，不能合称 schedule 冲突。 |
| Loss | `log(x+1)` compression，未给 norm/weight | 未补 | README 推荐 `comb2=log-MSE+0.1 linear-L1` | code 补充但不能称正文正式 loss。 |
| Optimizer | 未报告 | 未补 | Adam LR `1e-3` | code-only evidence。 |
| Runtime | Mitsuba PyTorch batching + CUDA/OptiX direct | WebGL/视频 demo | repo 含 model export helpers，但 README 未给 OptiX renderer source入口 | 公开仓库不足以重建论文完整 CUDA/OptiX benchmark。 |
| Assets/evaluation | synthetic + UBO，表/图结果 | supplementary video（MP4 locator 当前 HTTP 200） | README 保留 Google Drive datasets/models locator | Drive locator 本次匿名 GET 为 HTTP 404，资产当前不可用；也没有 formal split manifest。 |

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. importance sampling 只有 cosine hemisphere；更 specular material 需要 per-texel parametric PDF，作者建议 Lambertian + microfacet mixture，并强调可不增加 neural sampling network。[P §7 Limitations, §8]
2. very specular/glinty appearance 在不 blur 的情况下难以拟合；作者建议随机 latent input + GAN loss 生成统计细节。[P §7 Limitations, §8]
3. hard shadow 是 reflectance discontinuity，full model 仍损失 shadow contrast。[P Fig.6 caption, §7]
4. 当前输出不含 alpha；作者把 semitransparency 列为未来通过额外预测 alpha 的扩展。[P §8]
5. method 是逐材质训练；作者没有展示未见材质或未见原生参数状态的 feed-forward compilation。[P §3.2, §4][I：由训练对象边界直接推出]

### 12.2 未报告/材料不可得

- synthetic commercial renderer 名称、directional smoothing kernel、MIS 配置、bounce 数与 reference variance；
- 训练 query 的 `u/σ/ω_i/ω_o` 分布、dataset 总 query 数、train/validation/test split；
- paper 正式 loss norm、optimizer、LR、seed、checkpoint selection；
- 完整材质资产及每个材质的最高分辨率、全量 texture bytes；README 的 Google Drive locator 当前返回 HTTP 404；
- CUDA/OptiX 源码、precision、编译参数、scene/bounce/denoiser timing scope；
- energy conservation、reciprocity、direction-integrated error、sampling variance；
- code 额外 shadow-mask head 在论文算法中的定义、监督或消融；
- 对 code-only `comb2` 与 paper final-ReLU 模型，哪一个生成 Table 2/3 结果的可验证 checkpoint identity。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

NeuMIP 的主要容量不在 3332 个 MLP weights，而在每个材质的 14-channel spatial fields，特别是每个 LOD 都独立的 7-channel feature grid。MLP 更像一个小型局部方向 decoder；offset texture + fixed `H` 把 view-dependent geometry movement 先对齐，减少 decoder 同时记住多个位置/视角相关模式的压力。[I，依据 P §3.2–3.3, Table 4]

因此只比较 MLP 参数量会严重低估 asset cost。对当前项目，必须把 `B_asset` 按所有 mip texels、channels、precision 和 residency 计入，而不是只报 decoder weights。[I，依据 N `experiment_framework.md` §0/§4]

### 13.2 成功所依赖的假设

- `uv` 中复杂 appearance 在局部可由一个 view-conditioned scalar-depth warp 显著对齐；
- Gaussian footprint 足以代表目标 filter，并且训练 query 覆盖 runtime LOD；
- 每材质自由 latent 有足够数据和优化时间，不要求对未见材质直接生成；
- hard discontinuity 与 glints 不是主 workload，或可接受 smoothing/contrast loss；
- cosine sampling 对展示的 indirect transport 足够，不把 PT variance 当主目标；
- active material 数量、texture residency 和 MLP divergence 未超出 demo 场景。

这些是假设集合，不是 NeuMIP 对一般 layered BRDF、measured BRDF 或多材质 compiler 的普遍结论。[I]

### 13.3 可迁移机制与不能迁移的部分

可迁移：

1. 把 footprint/LOD 作为 reference query 的显式轴，而非对 latent 事后平均；
2. `prepare(u,σ,ω_o)` 中执行 offset 与 pyramid fetch，缓存 view-conditioned feature，多次 `evaluate(w_i)` 只跑后半 decoder；
3. 用 constrained coordinate warp 代替单纯扩大 MLP，并通过 matched no-warp/2D-warp/scalar-depth-warp 消融验证；
4. 高分辨率 free latent 需要结构化初始化或 continuation，paper 的 decaying blur 是可证伪 baseline；
5. LOD levels 的独立容量与跨级一致性应作为两个轴比较，而不是默认独立 levels 必然最佳。

不能直接迁移：

- LayerStack 1×1 方向峰没有 spatial parallax，`uv` offset 不会自动对齐随 `ω_i/ω_o` 移动的 specular lobe；其类比机制应是 learned frame/half-vector warp，而非虚构空间位移；
- 每材质 autodecoder 不能满足未见参数状态的 compiler 目标；需要 source-parameter encoder、target encoder、hypernetwork 或 bounded refinement 的独立实验；
- stochastic GAN glint output 与 deterministic、可复现的线性 `evaluate()` 不兼容，除非先定义随机状态、期望量和 matched `sample/pdf` 合同；
- cosine-only sampler 不满足本项目最终的 matched sampler 目标。

### 13.4 与本项目 runtime contract 的关系

NeuMIP 可被编译为静态有界 runtime：固定两级 feature fetch、一个 offset fetch、两个 4-layer MLP 和固定 warp。最自然的映射是：

`prepare(u,σ,ω_o) → {u_new, v=P(u_new,σ), optional view state}`；

`evaluate(state,ω_i) → F(v,ω_i,ω_o)`。

这与当前 `prepare/evaluate` amortization 直接吻合。[I，依据 N `docs/realtime_material_compilation.md`]

但 paper runtime 没有 evaluator-matched `sample/pdf`，所以它更适合作为 spatial/LOD evaluator candidate，而不是完整 MethodBundle。其 per-asset pyramid 还必须进入 `B_asset`/texture fetch/packed precision 的 Pareto；在当前 P1/P2 局部方向研究中，它更适合作为后续 spatial phase 的产品候选，而不是当前 1×1 LayerStack 的替代器。[I，依据 N `experiment_framework.md` §7]

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

[N] 当前 NVIDIA functional reproduction 已有 native-parameter encoder → hierarchical z8 materialization → latent finetune、learned-frame `3×64` evaluator、matched GGX9 sampler 和 packed-FP16 runtime；MaterialX spatial 与 LayerStack 1×1 是明确的 source-domain adaptation。[N `docs/learning.md`]

对应关系如下：

| NeuMIP 机制 | 当前 NVIDIA 状态 `[N]` | 判定 `[I]` | 影响 `[I]` |
|---|---|---|---|
| per-asset spatial latent pyramid | NVIDIA 已有 hierarchical latent mip chain，但其来源/decoder/training recipe 属于 RTA 2024 | `not-applicable` 于论文忠实性；`interface-adaptation` 于空间 source | 不应把 NeuMIP 独立 levels 写成 NVIDIA 原论文机制；可作为 spatial LOD matched candidate。 |
| view-conditioned coordinate offset | 当前 NVIDIA 用 learned shading frames 对齐方向，不做 NeuMIP `uv` depth warp | `not-applicable` | 两者都在 MLP 前显式对齐难函数，但作用域不同：spatial parallax vs directional lobe。 |
| decaying latent blur | NVIDIA formal 有前 20k directional mollification，目标是减弱极窄方向峰 | `not-applicable` 但可做 optimization 对照 | 两种 continuation 不可合并命名；可在 spatial latent 上比较 encoder init、latent blur 和 directional mollification 的正交效果。 |
| `prepare` 缓存 view-only stage | 当前公共 ABI 已明确 `prepare()` 复用 view-conditioned state | `interface-adaptation` | NeuMIP Fig.9 提供了很直接的 stage split precedent，可用于空间 MaterialX 的 prepare cost 设计。 |
| cosine sampler | 当前 NVIDIA 已实现 learned GGX9 matched sampler | `intentional-deviation`（若把 NeuMIP evaluator 接入当前 MethodBundle） | 不应为“忠实 NeuMIP”退回 cosine；应把 evaluator 与当前 matched sampler配对视为新组合，并单独报告。 |
| per-material decoder | 当前 NVIDIA 可多材质共享 model，每资产保留 latent | `intentional-deviation` | NeuMIP 不能证明 shared decoder；需要 matched per-asset vs shared 对照。 |

[I] NeuMIP 本身不提供诊断当前 NVIDIA evaluator suspected defect 的同-domain 证据；它主要补上 spatial footprint、coordinate warp 与 `prepare` amortization 轴。依据当前冻结协议 `[N]`，任何对 RTA 复现质量的改进都必须保持 NVIDIA formal source/query/budget 做 matched control，不能用不同 BTF 数据与 image MSE 直接宣称更好。[N `docs/research/experiment_framework.md` §2/§5]

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：空间 source 的 `prepare` 中加入 scalar-depth neural offset，比同 asset bytes 的无 offset 或 unconstrained 2D offset 更好 | NeuMIP direct 2D offset author-negative；w/o-offset Table 2 inferior | MaterialX/BTF 的误差包含可由局部平面 warp 对齐的 view-dependent parallax | `{no offset, direct 2D, scalar depth+H}`，同 decoder MAC、texture bytes、train queries | source snapshots、7D query recipe、optimizer、steps、precision | solid-angle L1、image temporal stability、offset Jacobian/out-of-range、prepare/eval time | spatial evaluator candidate | scalar-depth 对主指标无 CI 改善，或相同质量下 prepare+bytes 更差，或大量 lookup 越界/不稳定 |
| H2：encoder/bake 初始化可替代 NeuMIP 的长程 latent blur，并保留更高频细节 | paper 报告无 blur 的 free latent 噪声；当前 NVIDIA encoder materialization 已存在 | source parameters 或 target tensor 能产生空间连续的初始 latent | `{random+blur, encoder no blur, encoder+short blur}`，相同 final architecture/steps | query stream、total updates、LOD、latent bytes | convergence AUC、最终 L1/p95、high-frequency recall、seed variance | compiler + spatial evaluator | encoder 路径在相同预算下不降噪/不加速，或 final quality 明显低于 random+blur |
| H3：每 LOD 独立 latent 比由 finest downfilter 更准确，但需要显式跨级 consistency 才不会 temporal pop | NeuMIP Table 3 与 independent-level design | 当前 spatial source 的 filtering 也包含跨尺度非线性相关 | `{independent, derived mip, independent+cross-level loss}`，iso-byte 或明确双预算 | GT footprint、LOD distribution、decoder | per-LOD L1、LOD sweep temporal error、bytes、fetch time | spatial/LOD candidate | independent levels 的质量优势消失，或 temporal/bytes Pareto 被 derived mip 支配 |
| H4：把 view-only spatial work放进 `prepare` 能在多灯/多 sample 下摊销 | NeuMIP Fig.9 Stage 1 独立于 light，单次各约 2.6 ms | 当前 renderer 能在同 shading point 复用 state | 同一 compiled model 比较 fused-every-query 与 cached-prepare，lights/samples sweep | backend、resolution、material coverage、precision | `C_prepare`、`C_eval`、frame time、state bytes、quality parity | deployment scheduling | 复用次数达到预设阈值仍无显著加速，或 state/带宽使总成本更差 |
| H5：NeuMIP spatial warp 与 NVIDIA learned directional frame是正交可组合机制 | 两篇方法分别处理 spatial parallax 与 moving directional peak | source 同时存在空间 microgeometry 与尖锐 BRDF lobe | 2×2 matched `{no/offset}×{no/learned-frame}`，同总或分项预算 | source/query/train lifecycle | 分层 spatial/directional error、peak metrics、cost/bytes | capacity diagnostic → candidate | interaction 项不正或组合被任一单机制 Pareto 支配 |

## 16. 证据索引

- `P`：正式 PDF §3.1–3.3（Eq.1–8、Fig.2–5）；§4（data/training）；§5–6（renderer/performance、Fig.9）；§7（Fig.6–13、Table 2–4、ablation/limitations）；§8（future work）。
- `S`：作者项目页 `assets/neumip_sig2021_final.mp4`（2026-08-29 HEAD：HTTP 200，`video/mp4`）；只确认动态效果材料存在，未用作配置证据。
- `C`：commit `c1e2f2aa3488b7460cbf19f5bf6d1c4343926178`（2022-07-12 UTC）：`README.MD`；`angular.py:AngularSimple`；`experiments/simple.py:StandardRawLongShadowMaskOnly`；`neural_rendering.py:FullyConnected1`、`NeuralMaterialSavable.NeuralOffset`、`NeuralMaterialSavable.evaluate`、`NeuralMaterialLive.train_step`；`mipmaptexture.py:MipmapTexture`；`dataset/dataset_reader.py:Dataset`。README Drive locator 于 2026-08-29 返回 HTTP 404。
- `A`：[作者项目页](https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP/)，标题/作者/venue 与下载入口。
- `N`：`docs/research/prior_art.md` §3.2；`docs/learning.md`；`docs/realtime_material_compilation.md`；`docs/research/experiment_framework.md`；`.trellis/spec/project/method-constraints.md`。
- `I`：第 8 节 runtime-class 推导、第 9 节 protocol 边界、第 13–15 节项目分析与假设。

## Evidence review

```text
author_worker: /root
reviewer: /root/nbrdf2021
reviewed_at: 2026-08-29
sources_rechecked: [作者正式PDF及SHA-256并视觉核对p.3-6/p.8-9, DOI/作者项目页与supplementary MP4可用性, official repo固定commit c1e2f2aa3488b7460cbf19f5bf6d1c4343926178, README指定StandardRawLongShadowMaskOnly命令, architecture/output/loss/blur/runtime源码, docs/learning.md与experiment_framework.md]
findings_closed: [确认paper 3-output+final-ReLU与code 6-output+YUV-exp-shadow-mask路径的冲突, 展开3332/3254含bias参数算术并限制推论, 证明code指数blur与paper half-life数值一致且分离code-only下限/freeze/fuse lifecycle, 定位Fig.2/Fig.4/Fig.9方向标注冲突并以Eq.4-8+camera_dir闭合, 将5ms evaluator与60FPS path-tracer measurement scope分离, 显式划分第14节N事实与I判断, 修正official commit日期与失效data locator]
remaining_evidence_gaps: [supplementary video未逐帧审计, Google Drive datasets/models locator当前HTTP 404, 未执行官方训练或OptiX benchmark, official repo未包含论文OptiX/CUDA renderer, code shadow-mask head缺少paper定义/消融, 缺Table2/3 checkpoint identity]
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
