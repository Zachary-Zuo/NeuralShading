---
paper_id: "1469-2026-volumetric-light-transport-inference"
title: "Real-Time Volumetric Light Transport Inference from Auxiliary Renderings"
authors: "anonymous submission 1469"
year: "2026"
venue: "Pacific Graphics 2026 submission; publication status unverified"
doi: "not-reported"
report_status: "evidence-reviewed"
main_source: "paper1469_1.pdf"
supplemental_status: "unavailable"
official_code_status: "unavailable"
official_code_commit: "not-applicable"
author_worker: "/root"
reviewer: "/root/taming2026"
last_verified: "2026-08-29"
---

# 从辅助渲染实时推断体积光传输

## 1. 研究对象与报告边界

本文研究的不是局部材质 `evaluate(wo,wi)`，而是场景/图像级的 participating-media transport surrogate：对每帧执行一组不含完整 path-traced radiance 的 primary-ray auxiliary operators，再用 attention-augmented U-Net 直接回归最终 scattered-radiance image。作者把它定位成对 MC denoising 的替代：输入是 variance-free、物理派生的 feature buffers，不是 noisy low-spp radiance，因此推断不需要 history/reprojection。[P Abstract, §1, §3]

本报告覆盖项目根目录的 `paper1469_1.pdf`。文件仍以匿名编号 `1469` 标署，PDF metadata 写有 “Pacific Graphics 2026, Short Papers”，正文页眉写 “Conference Paper”；截至 2026-08-29，按正式标题检索没有发现公开作者页、publisher record、DOI、supplemental 或 code。因此报告不猜作者身份，也不把投稿稿件当作已经正式录用的版本。[P p.1/metadata][A web identity search]

本文只在 cloud-like heterogeneous volume、Henyey–Greenstein phase function 与合成 illumination/density 数据上验证。它预测 full image radiance，不输出局部 phase/BSDF、path sampler 或可组合 transport operator，不能与本项目 local neural material 方法做同语义质量排名。[P §4.2, §6]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | repository root `paper1469_1.pdf`，10 pages | 2026-08-29 | SHA-256 `963292EAA1E5FB02677FE1774BEAE2FC682C8A70EDBE7BDBAEDBDC28F183A2F6` | 唯一方法来源；已完整读取并把 10 页全部 rasterize，视觉核对公式、Fig.1–10、Table 1。 |
| Supplemental `S` | 未提供 locator | 2026-08-29 | unavailable | architecture、feature count 与训练细节不能由 supplemental 补足。 |
| Official code/config/data `C` | 以完整标题、短标题定向检索 | 2026-08-29 | unavailable | 没有实现、配置、checkpoint 或 dataset manifest 可审计。 |
| Author/publisher/correction `A` | 标题检索；PDF metadata | 2026-08-29 | unpublished/unverified | 当前只能确认匿名稿自称 Pacific Graphics 2026；没有公开身份可回填。 |
| NeuralShading evidence `N` | `docs/realtime_material_compilation.md`；`docs/research/experiment_framework.md`；`docs/research/model_candidates.md` | 2026-08-29 | 当前 workspace | 只用于第 13–15 节的语义边界与假设设计。 |

文本提取的部分连字符/Unicode glyph 映射不可靠，报告中的公式与数值均回到 rasterized PDF 视觉核对；没有把乱码作为原文继续引用或回写。

## 3. 原论文的问题、假设与贡献边界

作者认为 volume path tracing 的每个 scattering sample 本身又递归依赖 transport integration，低 spp input 对体积 denoiser 既昂贵又高度不稳定。本文绕过 runtime MC path integration，把场景参数 `θ` 映射为一组廉价 auxiliary images `Î=({T_k(θ)})`，再学习确定性 decoder `S(Î;Ψ)` 逼近 full transport image `I=R(θ)`。[P §1, §3]

论文显式提出两个假设：

- A1 sufficiency：`p(I | Î, θ)=p(I | Î)`，即给定 features 后，原场景参数不再提供预测目标所需信息；
- A2 deterministic decoding：存在确定性 `S`，使 `p(I|Î)=δ(I-S(Î;Ψ))`。[P §3]

由此作者把问题解释成 heterogeneous transport projections 之间的 cross-operator reconstruction，而不是把 noisy estimate 滤成 clean estimate。贡献是：不需要 expensive path computations；给出一组可 ray-cast 的 volume features；以实时 attention network 获得 frame-independent 的 temporal stability。[P §1, §3]

“near-sufficient statistic”是经验主张，不是信息论证明。§3.1 的 inversion 只把 cloud density 当作 unknown latent，anisotropy 与 scattering coefficients 因直接暴露在 features 中而固定；作者也在 §6 明确 feature sufficiency 当前仅 empirical。[P §3.1, §6]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Scene input `θ` | volume density、phase function、environment illumination；形式化定义还包含 geometry/material | 每场景/帧 | [P §3, Fig.3] |
| Runtime query | image-space per-pixel primary ray | 每帧整张 image；正式 resolution 未报告 | [P Fig.3, §6] |
| Auxiliary input | first-event/interaction depth `f_d`、optical depth `f_t`、crossing statistics/layers `f_c`、direct/NEE `f_l`、environment illumination `f_e`，以及 Fig.3 caption 所称的 single-scattering response `f_s` | 多张 heterogeneous feature maps；§3.3 最后一组改称 first-event medium properties `g,φ`，`f_s` 与二者的精确 packing/channel 对应未报告 | [P Fig.3, §3.3, Fig.9] |
| Medium parameters | extinction `σ_t`、single-scattering albedo `φ`、HG anisotropy `g` | `φ` 可逐 RGB channel 变化；训练在 high-scattering regime | [P §3.2–3.3, §4.2] |
| Output target | scattered radiance image only | RGB image；background contribution 不在训练 target 中 | [P Fig.3, §4.2] |
| Runtime output | attention U-Net 的 full volumetric radiance reconstruction | frame-independent image | [P §3.4, §6] |
| Domain restrictions | cloud-like participating media，HG phase；合成 HDR-like environment 与 cloud volumes | smoke/fire/SSS/underwater 未验证 | [P §4.2, §6] |

background 可用 transmittance analytically recover，但正文没有把合成公式、tone mapping、exposure 或 alpha convention写成部署协议。[P §4.2]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

训练时同一 `θ` 走两条路径：

1. physically-based volume renderer `R` 用 delta tracking、NEE 与 MC phase-function integration，产生 4K-spp scattered-radiance reference `I`；
2. auxiliary operators `T_k` 沿 primary rays 计算低成本 feature maps `Î`。

网络只接收 `Î`，loss 只比较预测 `Î_out=S(Î;Ψ)` 与 scattered-radiance `I`；没有把 noisy MC radiance作为 ours 的输入。[P Fig.3, §4]

inference 时仍需执行 auxiliary rendering。作者称每 pixel 只 integrate 一条 ray through volume，网络成本与 density 无关；这不表示输入 features 免费，也不表示沿 ray 没有 volume marching/NEE/environment computation。[P §3.3, §5, Fig.8]

### 5.2 辅助表示

1. Optical depth `f_t`：`τ(d)=∫_0^d σ_t(x_t)dt`，`T(d)=e^{-τ(d)}`；为避免高 opacity 区域的 transmittance 饱和，输入改用 `τ̂(d)=1-1/(1+0.5τ(d))`。[P §3.3]
2. First-event/interaction depth `f_d`：先计算 `τ_c=1-τe^{-τ}/(1-e^{-τ})`，取第一个满足 `τ(t)>τ_c` 的距离 `t`，近似 conditioned on scattering 的 expected visible interaction depth。[P Fig.3, §3.3]
3. Crossing layers `f_c`：逐步降低 extinction，计算多张 first-conditioned-interaction depth，形成由 opaque 到 transparent 的 multi-scale occupancy profile。层数和 extinction schedule 未报告。[P Fig.3, §3.3]
4. NEE/direct illumination `f_l`：在 first interaction `x_1` 对 directional light 估计 `L_d(x_1,ω_l)=T(x_1↔x_l)L(x_l,ω_l)/(4π)`。[P Fig.3, §3.3]
5. Environment illumination `f_e`：`L_e(x_1,ω)=∫_{S²}T(x_1↔x_i)ρ(x_1,ω,ω_i)L(x_i,ω_i)dω_i`，即在 first interaction 对 environment map、transmittance 与 phase function 做球面积分。数值 quadrature/sample count 未报告。[P Fig.3, §3.3]
6. 第六组存在原文内部命名/packing 缺口：Fig.3 caption 把 `f_s` 称为 single-scattering response；§3.3 对应位置却以 “Medium properties” 结束，只明确说在 `x_1` 读取 HG anisotropy `g` 与 single-scattering albedo `φ(x_1)`，没有另给 `f_s` 公式或说明 `f_s` 是否就是 `g,φ` 的打包。Fig.9 又把 `-Scattering Albedo` 与 `-Anisotropy` 分开消融。因此可以确认六个概念组及 `g/φ` 的使用，不能确认第六张图/通道的精确 tensor schema。[P Fig.3 caption, §3.3, Fig.9]

Fig.9 的 qualitative leave-one-feature-out 列为 `All, -Depth, -Crossings, -Optical Depth, -NEE, -Environment, -Scattering Albedo, -Anisotropy`；它没有给每项的数值误差或 repeated trials。[P Fig.9]

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Backbone | 全部 auxiliary feature maps | U-Net + residual blocks | 未报告 | multi-scale features | dataset-shared | [P §3.4] |
| Local attention | intermediate resolution levels 3 and 5 | HaloNet-style local spatial self-attention | head/window/channel/normalization 未报告 | locally mixed features | shared | [P §3.4] |
| Global attention | bottleneck | full self-attention | token/channel/head 未报告 | global context | shared | [P §3.4, Fig.5] |
| Decoder/output | U-Net up path | 具体 layer count 与 skip topology 未报告 | output activation/transform 未报告 | RGB scattered radiance | shared | [P §3.4] |

论文没有给 input channel packing、每 level resolution/channel、convolution kernel、residual block count、attention head/window、activation、normalization、parameter count、MAC/FLOP 或 model bytes。故只能复现结构意图，不能从 “following HaloNet” 反推精确实现。

### 5.4 条件化、坐标变换与物理先验

物理先验位于输入 operators，不在 loss 中强制 transport constraints。network 学的是从 attenuation/interaction/lighting/local-scattering projections 到 RGB radiance 的 image-space mapping。global attention 的目的，是跨 image long-range 汇聚 multiple scattering、indirect illumination 与 global visibility 依赖。[P §3.3–3.4, Fig.5]

环境光先被 renderer 拆成：低于 saturation threshold 的 low-frequency residual map；以及代表 dominant high-intensity direction 的 directional delta，后者作为 NEE directional light。threshold、dominant-direction extraction 与 residual energy conservation 未报告。[P §4.1]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Volume assets | Cloudy Project 的 500 个 cloud formations | [P §4.2] |
| Illumination assets | Stable Diffusion 合成的 1000 张 HDR-like maps，覆盖 natural/stylized lighting | [P §4.2] |
| Dataset size | 22,000 reference images | [P §4.2] |
| GT/reference | delta tracking + NEE + phase-function MC indirect transport，4K spp | [P §4.1–4.2] |
| Randomized axes | camera poses、light intensities、HG anisotropy、scattering albedo；albedo 可逐 RGB 变化 | [P §4.2] |
| Target content | scattered radiance only；background omitted | [P §4.2] |
| Test | held-out 200 images | [P §5] |
| Train/validation split | 22,000 中的精确 train/validation/test 分配、scene/volume/environment disjointness未报告 | 未报告 |
| Render resolution | 未报告 | 未报告 |
| Reference convergence | 4K spp，但无 variance/error 或 renderer parity 检查 | [P §4.2] |

Stable Diffusion 只用于合成训练 illumination maps；它不是 runtime model。论文没有报告其版本、prompt、HDR reconstruction/calibration 或 train/test illumination 去重，因此“unseen lighting”边界无法精确判断。[P §4.2]

## 7. Loss、optimizer 与训练 lifecycle

正式 objective：

`L = λ_1 ||Î_out-I||_1 + λ_log ||log(1+Î_out)-log(1+I)||_1`。[P §4.3]

| 项 | 正式配置 | locator |
|---|---|---|
| Optimizer | AdamW；三方法使用 identical hyperparameters | [P §4.3] |
| AdamW `lr/β/ε/weight_decay` | 未报告 | 未报告 |
| LR schedule | exponential decay over 120k optimization steps | [P §4.3] |
| Decay endpoints/rate | 未报告 | 未报告 |
| Batch size | 4 images | [P §4.3] |
| Loss weights `λ_1,λ_log` | 未报告 | [P Eq. loss] |
| Initialization/seed/repeats/model selection | 未报告 | 未报告 |
| Training hardware/time | 结果硬件为 RTX 5090；正文未明确 training time，也未单独说明 training device | [P §5] |

“identical hyperparameters”只说明 Den/DenX/Att 的 optimizer policy 被作者视为 matched；CNN 与 attention U-Net 容量、input modality 和 compute 并不因此相等。[P §4.3–5]

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime stages | Vulkan auxiliary renderer → CUDA-exported device memory → neural inference | [P §5] |
| Synchronization | 避免 host-side synchronization/transfer | [P §5] |
| Hardware | NVIDIA RTX 5090 | [P §5] |
| Ours latency | Fig.10 七个案例为 25, 26, 25, 25, 26, 25, 28 ms | [P Fig.10] |
| Den/DenX latency | 包含 required 4-spp radiance condition；随 density 约 46–174 ms | [P Fig.8/10] |
| Ours input cost | auxiliary feature computation + attention model；Fig.8分开画 condition/model eval，但无表格原始数值 | [P Fig.8] |
| Precision/quantization | 未报告 | 未报告 |
| Model parameters/MAC/memory | 未报告 | 未报告 |
| History/reprojection | 无；每帧独立 | [P Abstract, §6] |
| Density scaling | 作者报告一条 primary ray per pixel，runtime 近似 independent of density；实验只覆盖 Fig.8 的 density settings | [P §5–6, Fig.8] |

Fig.1 的 “25 ms” 与 Fig.10 的每例总时间是整条方法的展示值；正文没有公开 resolution、timing aggregation、warmup、power state 或 condition/model 的 exact numeric breakdown，因此不能换算普遍 FPS，也不能与 local material 单 query ns 成本比较。

## 9. 实验 protocol、baseline、指标与结果

### 9.1 三个方法身份

- `Den`：受 prior/NVIDIA work 启发的 CNN denoiser，输入 noisy radiance estimates 与 statistics；正文未给 exact architecture/citation-to-implementation correspondence。[P §5]
- `DenX`：同一 CNN architecture，把 noisy auxiliary features 换成本文 auxiliary renderings，同时仍输入 4-spp MC radiance buffer。[P §5]
- `Att (Ours)`：attention U-Net，只输入 auxiliary features，不需要 path-traced radiance samples。[P §5]

三者在同一 200-image held-out test 上报告：

| Method | SSIM↑ | PSNR↑ | MAE↓ | `t-RMSE`↓ | LPIPS↓ |
|---|---:|---:|---:|---:|---:|
| Den | 0.9682 | 38.13 | 0.0048 | 0.0079 | 0.0474 |
| DenX | **0.9859** | **42.62** | **0.0030** | 0.0042 | **0.0161** |
| Att (Ours) | 0.9856 | 40.15 | 0.0043 | **0.0010** | 0.0193 |

按 Table 1 的数值，DenX 在 SSIM、PSNR、MAE、LPIPS 四个 spatial/perceptual metrics 上更好，Att 只在 temporal instability metric 上最好；table caption 明说的是 DenX 有最高 overall SSIM/PSNR，而“四项更好”是对表中数值的直接读取，不扩写成作者的统一综合排名。正文叙述处称该指标 `Root Mean Square of Instability (RMS-i)`，Table 1 列名写 `t-RMSE`，没有给逐帧序列构造或正式公式；本报告保留这个 published naming/definition gap，不把两名称自行定义为数学上等价。[P §5, Table 1]

### 9.2 性能与 corner cases

Fig.10：

| Setting | Ours | Den | DenX |
|---|---:|---:|---:|
| Teaser | 25 ms | 115 ms | 113 ms |
| High Density | 26 ms | 174 ms | 174 ms |
| Low Density | 25 ms | 46 ms | 47 ms |
| Colored | 25 ms | 85 ms | 85 ms |
| Isotropic | 26 ms | 120 ms | 122 ms |
| Anisotropic | 25 ms | 107 ms | 106 ms |
| Backlight | 28 ms | 77 ms | 78 ms |

这组结果支持的是同一 RTX 5090/Vulkan-CUDA pipeline 下，ours 避开 4-spp path condition 后对 density 更平；并不隔离 attention network 本身比 CNN 更快。Fig.8 反而说明 CNN model eval 很便宜，主要差异来自 denoiser condition buffer 的 path cost。[P §5, Fig.8/10]

### 9.3 Feature sufficiency 与 ablation

§3.1 用 Diffusion Posterior Sampling 从 reference cloud density 出发，逐步匹配 auxiliary subsets。匹配所有 features 后，优化得到的 density 与原 geometry 在 novel view 仍不同，但 main-view rendering 几乎相同。它表明 features 对该 view 的 radiance solution 约束较强，同时直接显示 features 不唯一确定 density。[P §3.1, Fig.4]

Fig.9 逐项移除 7 类 feature，给出 qualitative appearance；作者概括一部分影响 detail/proper shading，另一部分有 physically meaningful contribution。没有 numeric metric、样本集合或交互消融，因此不能排名各 feature 的边际价值。[P §5, Fig.9]

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `author-positive` | DenX = auxiliary features + 4 spp radiance | Table 1 综合质量最好 | features 给全局 transport 提供物理约束 | 同时保留 radiance，不能隔离 “features 足够替代 paths” | [P §5, Table 1] |
| `author-positive` | Att（正式主方法）不看 4-spp radiance | PSNR/MAE/LPIPS/SSIM 均略差于 DenX，但 `t-RMSE` 大幅更好，且 Fig.10 总时间显著更低 | 去掉 stochastic radiance input 获得稳定性并绕开 4-spp condition cost | 是 quality–stability–time tradeoff，不是全指标胜出；不能把正式方法误分为 inferior ablation | [P §5, Table 1, Fig.8/10] |
| `ablation-inferior` | leave-one-feature-out | Fig.9 出现 detail/shading变化 | features 各有用途 | 只有 qualitative single/example evidence | [P Fig.9] |
| `known-limitation` | feature inversion | all-feature match 仍得到不同 density/novel view | features constrain rendering but not geometry | 否定了 scene representation injectivity；对 unseen view 的 sufficiency 更弱 | [P §3.1, Fig.4] |
| `known-limitation` | cloud/HG only | 其他介质未验证 | 需要扩展 features/training | 不能声称 general volumetric transport compiler | [P §6] |

`Den` 是 Table 1 的比较基线，不是作者称作失败尝试或正式 ablation 的配置；它的五项指标均低于另外两法，但 exact baseline correspondence 未公开，结论只限本文实现。[P §5, Table 1] 正文没有报告训练失败史、多 seed optimization failure、pure-CNN-on-auxiliary 与 attention 的严格 capacity-matched architecture ablation，或 local/full attention 单独移除的数值结果；不能从最终设计倒推这些尝试发生过。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Architecture | U-Net/residual；local attention level 3/5；full bottleneck attention | 不可得 | 不可得 | 不足以精确重建网络 |
| Features | 给出 6 组概念 operator 与多数公式；Fig.3 的 `f_s` 命名和 §3.3 的 `g,φ` 描述未被精确对应 | 不可得 | 不可得 | input packing/channel count、crossing 层数、ray budgets、environment quadrature 未报告 |
| Data/reference | 500 clouds、1000 generated maps、22k/4K spp、200 test | 不可得 | 不可得 | split identity、assets、renderer config 不可复现 |
| Training | AdamW、120k、batch4、exp LR、双 L1 | 不可得 | 不可得 | LR/betas/decay/loss weights/seed 未报告 |
| Runtime | Vulkan/CUDA zero-copy、RTX5090、25–28ms cases | 不可得 | 不可得 | resolution、precision、model/feature timing raw data缺失 |
| Metrics | Table1；text `RMS-i` vs table `t-RMSE` | 不可得 | 不可得 | temporal metric definition不完整 |

这是 anonymous paper-only report。没有 code 时，不能把相关 HRS20/HaloNet/NVIDIA 实现中的默认 topology 或 metric 定义填入本文。

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. 仅训练/测试 cloud-like media + HG phase；smoke、fire、subsurface、underwater 未验证。[P §6]
2. frame-independent，虽因 variance-free input 获得稳定，但 rapid camera/light motion 可能受益于 temporal recurrence/cross-frame attention。[P §6]
3. sufficiency 只经验评估，operator parameterization 通过 discretized exhaustive grid search；未来可 joint differentiable optimization。[P §6]
4. feature inversion 不能唯一恢复 density，novel-view geometry 不一致。[P §3.1, Fig.4]
5. background/emission/general phase functions 与 mixed surface-volume scenes 不在正式实验里。[P §3.2, §4.2, §6]

### 12.2 未报告/材料不可得

- 作者、正式 publication status、DOI、supplemental、code/license；
- exact U-Net/HaloNet topology、输入 channels、parameter/MAC/model bytes、precision；
- auxiliary feature 的 channel count、crossing levels、step size、NEE/environment samples 与每项 cost；
- Fig.3 `f_s` single-scattering response 与 §3.3 first-event `g,φ` 的精确对应/packing；此外 §3.2 把 `L_e` 用作 emitted radiance，§3.3 又把 `L_e` 用作 environment illumination，原文符号发生复用；
- image resolution、train/val/test scene-level split、Stable Diffusion/HDR pipeline 与 data assets；
- AdamW/loss/scheduler完整配置、seed、repeat、training time/model selection；
- Den/DenX exact implementation、4 spp sample recipe、noise statistics；
- `t-RMSE/RMS-i` 公式、temporal sequence/protocol；
- timings 的 aggregation、warmup、condition/model raw measurements、memory/power；
- feature inversion 的 grid、DPS config、objective、view/error数值。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

容量由三部分共同承担：人为设计的 physical feature operators；整幅图像 U-Net 的 multi-scale convolution；bottleneck/global attention 的 scene-wide mixing。与 local neural material 的 per-query MLP 相比，本文把最难的 long-range transport 留给 image-space receptive field，代价是结果依赖固定 image sampling、camera和scene context。[I]

### 13.2 成功所依赖的假设

- 目标 domain 足够窄，500 cloud volumes 与生成式 illumination 覆盖测试分布；
- first-primary-ray features 对当前 view 的 multiple scattering 近似 sufficient；
- 屏幕空间邻域/全局 attention 可以从训练分布补出未观察 transport；
- 不要求从输出中恢复真实 density，也不要求跨 resolution/view 做随机访问查询；
- 25–28 ms 的 RTX 5090 image pipeline 满足作者的“real-time”语境。

Table 1 与 Fig.4 一起说明，deterministic mapping 可在 test images 上成立得很好，但不唯一、也不等价于物理 transport operator 被恢复。[I]

### 13.3 可迁移机制与不能迁移的部分

可迁移的是 evidence method：把便宜、确定性、物理含义明确的 auxiliary signals 与昂贵 GT 对齐；显式测量 feature sufficiency；把 temporal stability 作为独立 metric；通过 condition/model 分解解释性能来源。[I]

不能迁移到 local evaluator：整图 attention、camera-specific transport hallucination、fixed cloud distribution，以及“同 main-view image 即足够”的 criterion。它们不满足 `evaluate(wo,wi)` 随机访问、跨 light integration、sampler matching或未见 source parameter state 的 compiler semantics。[I]

### 13.4 与本项目 runtime contract 的关系

运行成本对固定 resolution/model 是有界的，但随 image resolution、attention tokens 与 feature-ray marcher 变化；它不是每次材质查询固定成本的 shader program。对本项目最合适的分类是 `scene/volume transport diagnostic` 与可选 image-space deployment track，不是 material evaluator、teacher 或 sampling proposal。[I]

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

当前 NVIDIA 方法的 latent、learned frame、evaluator 与 sampler 是局部 scattering program，`prepare/evaluate/sample/pdf` 必须在 PT/deferred 共用，并保留真实 UV footprint。[N `docs/realtime_material_compilation.md`; archived NVIDIA correspondence]

本文不提供可直接修复该实现的 architecture，因此对应关系是 `not-applicable`，而非 `intentional-deviation`。可借鉴的只有评测层：

- 将 source/reference 的廉价 auxiliary features 与 learned latent 做 sufficiency/inversion audit；
- 把无随机输入时的 temporal stability 单列，不能用单帧 PSNR替代；
- 成本必须拆成 `condition/prepare` 与 `model evaluate`，避免把 path/reference 开销混入 decoder timing；
- 对 scene-level surrogate 必须另设 identity，不能标成 NVIDIA local material faithful reproduction。[N/I]

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H-VLT-1：确定性 physical auxiliaries 可减少 scene transport 的 temporal instability | Ours t-RMSE .0010 vs DenX .0042/Den .0079 | 项目后续环境积分/scene track 存在稳定且廉价的 feature buffers | 同 network/capacity，noise radiance vs deterministic auxiliaries vs both；相同 frame sequences | scene split、resolution、training steps、camera/light trajectories | single-frame quality + 明确定义 temporal RMSE/FLIP | scene renderer | auxiliaries-only 在 matched quality 下不改善 temporal metric CI |
| H-VLT-2：feature sufficiency 应通过 collision/inversion 而非只看 regression error 验证 | Fig.4 all-feature match 不唯一确定 density | 项目 source features也可能把不同材质状态 alias 到同 latent | 优化两个 source states 使 features/latent接近，比较 dense directional reference | source family、optimizer budget、query grid | latent distance、reference function distance、edit-axis collision rate | compiler diagnostic | 大量低 feature distance 对应高 reference distance，且无法增加少量 feature 修复 |
| H-VLT-3：prepare/condition 与 decoder 成本拆分会改变 Pareto解释 | Fig.8 CNN eval便宜，4-spp condition随 density膨胀 | 当前候选也可能把成本藏进 reference/latent fetch/prepare | 对每候选分别计 source/prepare/evaluate/sample/integration | hardware、precision、batch/coherence | stage time、reads、bytes、quality | evaluation policy | 分解后排名与 end-to-end 相同且没有隐藏主导项，则该假设不提供决策增益 |
| H-VLT-4：global mixing 只对 scene transport有益，不应进入 local material shader | 本文以 bottleneck full attention捕获 non-local dependencies | local scattering GT 不需要场景全局上下文 | local MLP vs 加邻域/global context；严格相同 material/query split | parameter/compute budget、source family、directions | G1/G2/G2s、runtime/memory | capacity diagnostic | global context 在未见材质上稳定提高 local query 且仍满足固定成本/随机访问，则否定“无益” |

## 16. 证据索引

- `P Abstract/§1`：相对 MC denoising 的问题与贡献。
- `P §3/A1–A2`：sufficiency 与 deterministic decoding 假设。
- `P §3.1/Fig.4`：feature inversion、density non-uniqueness。
- `P §3.2–3.3/Fig.3`：volume equation、auxiliary operators 与输入分解。
- `P §3.4/Fig.5`：U-Net/HaloNet local/full attention。
- `P §4.1–4.3/Fig.6`：renderer、data、loss 与训练 lifecycle。
- `P §5/Table1/Fig.7–10`：baselines、quality、stability、density scaling、latency与feature ablation。
- `P §6`：domain、temporal、sufficiency/operator optimization 限制。
- `A web identity search`：截至核查日无公开正式身份。
- `N/I`：第 13–15 节；不反写为论文事实。

## Evidence review

```text
author_worker: /root
reviewer: /root/taming2026
reviewed_at: 2026-08-29
sources_rechecked:
  - local anonymous main PDF, SHA-256 963292EAA1E5FB02677FE1774BEAE2FC682C8A70EDBE7BDBAEDBDC28F183A2F6; all 10 pages visually rechecked
  - PDF metadata and visible author/proceedings headers for anonymous-identity boundary
  - exact-title and targeted public author/publisher/project/code searches as of 2026-08-29
  - project evidence-policy classification rules and N runtime-contract documents already cited in this report
findings_closed:
  - kept authorship at anonymous submission 1469; editor/header names were not misattributed as paper authors
  - rechecked A1/A2 equations and feature-inversion interpretation
  - rechecked all six auxiliary groups/formulas and recorded the unresolved Fig.3 f_s versus section 3.3 g/phi packing mismatch
  - bounded U-Net/HaloNet claims to residual U-Net, local attention at levels 3 and 5, and full bottleneck attention; all undisclosed topology remains unreported
  - rechecked dataset, reference renderer, training objective and every disclosed training setting
  - rechecked every Table 1 value and separated DenX spatial/perceptual wins from Att temporal-stability win
  - rechecked Fig.8 stage accounting and all 21 Fig.10 displayed runtimes
  - preserved the undefined RMS-i versus t-RMSE naming/definition gap
  - corrected success/failure classification: Att is the proposed author-positive result and Den is a comparison baseline, not an inferred failed ablation
  - confirmed no supplemental, official code/config, checkpoint or data release was discoverable
  - confirmed N/I treats this as a scene-level deterministic image decoder, not a local material evaluator, teacher, or matched sampler
remaining_evidence_gaps:
  - anonymous publication identity and DOI unavailable
  - supplemental/code/config/data unavailable
  - exact input tensor packing, architecture, feature/ray budgets, environment-light construction, training hyperparameters and splits remain underspecified
  - temporal metric formula/sequence protocol and raw Fig.8 timing values remain unavailable
review_status: evidence-reviewed
```

### 完成检查

- [x] main paper 已完整阅读，关键公式/图/表/图注/脚注已视觉核对；
- [x] supplemental/appendix/勘误的可用性已检查；当前不可得；
- [x] official code/config/data 的可用性与公开身份已检查；当前不可得；
- [x] architecture、training、runtime 和主要结果均有 locator；未披露项保留；
- [x] 成功、较差消融、比较基线与已知限制已按证据分类；
- [x] paper/code gap 和“未报告”保留；
- [x] `I` 分析晚于事实层，没有改写作者结论；
- [x] NVIDIA 影响引用真实 `N` 合同；
- [x] 假设包含 matched control、部署类别和证伪条件；
- [x] 独立 evidence review 已完成。
