---
paper_id: "xu-2025-improving-angular-parameterization"
title: "Improving Angular Parameterization for Compact Neural Materials"
authors: "Zilin Xu; Yang Zhou; Yehonathan Litman; Ling-Qi Yan; Anton Michels"
year: "2025"
venue: "SIGGRAPH Asia 2025 Posters"
doi: "10.1145/3757374.3771447"
report_status: "evidence-reviewed"
main_source: "https://starry316.github.io/sigas2025/sa2025_abstract_1014_final.pdf"
supplemental_status: "unavailable"
official_code_status: "unavailable"
official_code_commit: "not-applicable"
author_worker: "/root"
reviewer: "/root/dualband2025_review"
last_verified: "2026-08-29"
---

# Improving Angular Parameterization for Compact Neural Materials

## 1. 研究对象与报告边界

这是一篇两页 SIGGRAPH Asia 2025 poster extended abstract，研究 tiny neural-material MLP 在严格输入维度下应该使用什么方向参数化。它不是完整 conference paper：没有 supplemental、代码、训练配置或 runtime benchmark。因此本报告会详细恢复两页正文与正式 poster 的所有配置和结果，同时把未披露字段保留为缺口，不把图的 x 轴、网络 activation 或 texture resolution 自行解释出来。

它是 Taming 2026 明确引用的方向参数化证据，也是当前 NVIDIA reproduction 计划中 coordinate ablation 的 load-bearing related。[N] 论文只在 UBO2014 的 `Leather11` 和 `Fabric12` 两个 measured BTF/material 上实验。[P] 因此，同一排名能否适用于 layered BSDF、任意 neural material family 或本项目 G2/G2s，仍需另行验证。[I]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main abstract `P` | [作者托管两页正式 abstract](https://starry316.github.io/sigas2025/sa2025_abstract_1014_final.pdf)，DOI `10.1145/3757374.3771447` | 2026-08-29 | SHA-256 `F7E62188B9100E01910D0ECF20F77619F818BB73AA9F51FF7716F4F877FD3B90`；2 页 | 正式题名、作者、architecture budget、十种参数化、数据、Fig.1–3 与作者结论 |
| Official poster `A` | [作者托管正式 poster](https://starry316.github.io/sigas2025/sa2025_poster_template_1013_v5.pdf) | 2026-08-29 | SHA-256 `425415BD9758E525D8A8357638848A38C5972A13B001B65368B528AC391F0ABC`；1 页 | 对 abstract 的图例、数值和结论做视觉交叉核对；不增加训练细节 |
| Supplemental `S` | 未发现 | 2026-08-29 | 不适用 | 不可得 |
| Official code/config/data `C` | 本次核对的作者 abstract、poster 与 publication page 均未给 official code/config/data 链接 | 2026-08-29 | 不适用 | 没有可审计 locator；本次 review 未做 Git 网络探测 |
| Author page `A-meta` | [Zilin Xu publication page](https://starry316.github.io/) | 2026-08-29 | stable page | 论文身份与作者摘要 |
| NeuralShading evidence `N` | [当前 NVIDIA correspondence](../implications/current-nvidia-correspondence.md)；Taming/RTA 报告 | 2026-08-29 | 当前工作树 | 只用于 §13–15 的项目对应；不补论文配置 |

PDF 已以 Poppler 提取全文，并按 144 dpi 渲染两页逐页视觉核对；poster 以 96 dpi 整页核对。数学符号、图例颜色、十列标签和 Fig.1 数值均以渲染结果为准。

## 3. 原论文的问题、假设与贡献边界

作者的问题是：当 neural material 需要在 VR headset/phone 这类低功耗设备上 per-shader 执行，MLP 必须极小；在 `D<10` 的 angular input budget 下，方向坐标选择会怎样影响质量？[P §1–2]

论文比较的不是任意大模型中的“最佳坐标”，而是下列固定结构上的 parameterization study：

- 一个 7-channel neural texture；
- angular vector 维度 `D<10`；
- 三个 `8×8` hidden layers 的 MLP；
- 输出 RGB reflectance；
- 两个 UBO2014 measured materials；
- 每种 parameterization 从头训练。[P §2–3]

贡献是系统并列十种坐标输入，并指出这个预算下直接 Cartesian `ωo,ωi` 通常优于 spherical、1-level positional encoding、per-angle latent texture 和纯 half/difference；把 half vector `h` 再加入 direct Cartesian tuple 对复杂强反射的 `Leather11` 有利，但对 grazing sheen 主导的 `Fabric12` 反而有害。[P Figs.1–3, §3]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | UBO2014 `Leather11`、`Fabric12` 的 measured material data；模型输入包含一个7-channel neural-texture feature | feature取得方式与spatial coordinate未报告 | P §2–3 |
| Runtime query | 7-channel neural-texture feature与方向参数拼接后送入 MLP | `7+D`，`D<10` | P §2 |
| Direction coordinates | incoming `ωi`、outgoing `ωo`、half `h`、difference `d`；spherical/Cartesian/PE/latent variants | 见下表 | P Fig.2, §2 |
| Output quantity | RGB reflectance | 3 channels；是否含 cosine、单位与非负 transform 未报告 | P §2 |
| Validity/domain restrictions | 两个 measured materials、tiny MLP、最多9个 angular scalars | 不证明跨材质/不同 capacity | P §2–3 |

十种正式输入如下：[P Fig.1 legend; Fig.2]

| 编号 | Direction family | 表示 | `D` | 额外 angular 状态/读取 |
|---:|---|---|---:|---|
| 1 | half/difference | `(h,d)` spherical | 4 | 无额外 angular texture fetch |
| 2 | half/difference | `(h,d)` 1-level PE | 8 | 无额外 angular texture fetch；exact sinusoid ordering 未报告 |
| 3 | half/difference | `(h,d)` latent texture | 8 | 每个 angle 一个4-channel latent；作者明确总计2个额外 texture fetch，是否共享同一texture/storage未报告 |
| 4 | half/difference | `(h,d)` Cartesian | 6 | 无额外 angular texture fetch |
| 5 | half/difference + light falloff | `(h,d,cos θi)` spherical | 5 | 无额外 angular texture fetch |
| 6 | direct directions | `(ωo,ωi)` spherical | 4 | 无额外 angular texture fetch |
| 7 | direct directions | `(ωo,ωi)` 1-level PE | 8 | 无额外 angular texture fetch；exact sinusoid ordering 未报告 |
| 8 | direct directions | `(ωo,ωi)` latent texture | 8 | 每个 angle 一个4-channel latent；作者明确总计2个额外 texture fetch，是否共享同一texture/storage未报告 |
| 9 | direct directions | `(ωo,ωi)` Cartesian | 6 | 无额外 angular texture fetch；`ωi.z=cos θi`显式暴露light falloff |
| 10 | direct + half | `(ωo,ωi,h)` Cartesian | 9 | 无额外 angular texture fetch |

Fig.1 caption 明确所有 spherical inputs 都线性归一化到 `[0,1]`，不只 direct-direction spherical variant。这里的“无额外 angular texture fetch”不代表整条方法零读取；7-channel neural-texture feature 的物理texture布局与基础fetch数未报告。

论文没有说明 tangent frame 的 orientation、hemisphere convention、`h/d` 的 antipodal/singularity处理、`d` 的具体 rotation convention、latent angle texture parameterization/resolution/filter 或 invalid direction behavior。

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

每个 query 向 MLP 提供7-channel neural-texture feature，并按选定方案把 `(ωo,ωi)` 转成 `D` 维 angular vector；latent variants 另执行作者明确的两次 texture fetch，每次取得一个4-channel angle latent。随后拼成 `7+D` 输入并输出 RGB reflectance。[P §2, Fig.2–3] 正文没有说明7-channel feature如何取得、基础fetch数，也没有说明两次angle fetch来自同一还是不同texture/storage。

### 5.2 持久化表示

- 所有 variants 都有 7-channel neural-texture feature；texture对象数、基础fetch数、resolution、mip/filter、precision、每材质/共享边界未报告。
- latent-texture variants 额外为每个 angle 使用4-channel vector，正文明确付出 two extra texture fetches；两次读取的storage identity、texture resolution和寻址未报告。
- learnable-frame方法只列入相关参数化图，没有进入正式十列实验。作者说它需要额外 linear layer 和更高输入维度，超出本实验 budget。[P §2, Fig.2]

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Neural-texture feature acquisition | 未报告 | 论文只锁定输入是7-channel neural texture；lookup/layout未报告 | filtering 未报告 | 7D feature | 每材质/共享分界未报告 | P §1–2 |
| Angular parameterizer | `ωo,ωi` | 十种之一；spherical/Cartesian/Rusinkiewicz/PE/latent | spherical线性归一到`[0,1]`；其余 normalization 未报告 | `D<10` | fixed transform 或 learned texture | P Figs.1–2 |
| Reflectance MLP | `7+D` | 3个 `8×8` hidden layers；首/末层矩阵与 bias 未报告 | activation/normalization 未报告 | RGB reflectance | 每材质/共享未报告 | P §2 |

“三个 `8×8` hidden layers”是作者的 budget 描述，不能自行展开为某个带 bias 的精确 parameter count，也不能假设 ReLU/LeakyReLU。

### 5.4 条件化、坐标变换与物理先验

论文最核心的 prior 是让输入直接暴露 `cos θi`：Cartesian `ωi` 的 z 分量天然提供 light falloff。作者据此解释 direct Cartesian consistently best；给 spherical `(h,d)` 追加 `cos θi` 也显著改善。[P §2]

`h`是另一个 conditional clue：Leather11 的 complex strong reflection从 `h` 获益，Fabric12 的 grazing sheen并不从 `h` 获益。这里没有 learned frame，也没有 analytic microfacet core或 reciprocity constraint。

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset/source assets | UBO2014 `Leather11`、`Fabric12` | P §3 |
| GT/reference | measured material dataset；采集/预处理沿用 UBO2014，本文未展开 | P §3 + reference |
| Train/validation/test split | 未报告；只说 Fig.3 是 entire test dataset 的 average | P §3, Fig.3 |
| Spatial/directional sampling | 未报告 | P gap |
| Filtering/LOD/footprint | neural texture resolution/filter/LOD 未报告 | P gap |
| Augmentation | 未报告 | P gap |
| Online/offline generation | 每种 parameterization 从 scratch 训练；batch/data materialization 未报告 | P §3 |

只有两个材料意味着这是 targeted coordinate diagnostic，不是 broad family generalization benchmark。

## 7. Loss、optimizer 与训练 lifecycle

| 项 | 正式配置 | locator |
|---|---|---|
| Target/output transform | RGB reflectance；是否 linear/log/cosine-weighted、clamp/nonnegative 未报告 | P §2 gap |
| Loss terms | Fig.3 报 test `L1` difference；正文没有明确训练 objective 是否同为 L1 | P §3, Fig.3 |
| Optimizer/hyperparameters | 未报告 | P gap |
| LR schedule | 未报告 | P gap |
| Batch/query count | 未报告 | P gap |
| Steps/epochs/stages | Fig.3 横轴 0–150，但横轴单位未标，不能写成 epochs/steps | P Fig.3 |
| Initialization/seed/model selection | 各 variant from scratch；seed数、重复次数与 checkpoint rule 未报告 | P §3 |
| Hardware/training time | 未报告 | P gap |

由于没有 seed/variance，曲线和 Fig.1 是单次或未说明聚合的观察，不能支撑微小差异的统计显著性。

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path | per-shader tiny MLP读取7-channel neural-texture feature与angular input；feature acquisition与基础texture fetch数未报告 | P §1–2 |
| Parameter count/MAC/FLOP | 未报告；不同 `D` 改变首层权重，latent variant增加texture state/fetch | P §2 |
| Shared/per-asset/state bytes | 7-channel texture与latent texture bytes未报告 | P gap |
| Texture/feature fetches | 只锁定相对成本：latent-angle variants比fixed-coordinate variants多2次texture fetch；7-channel base feature需要多少物理fetch未报告 | P Fig.3 caption |
| Precision/quantization | 未报告 | P gap |
| Hardware/backend/coherence | 目标是 low-power device，但没有 runtime hardware/backend测量 | P §1 |
| Time/FPS/latency | 未报告 | P gap |
| Prepare/amortization | 未报告 | P gap |

因此“Cartesian最好”是该质量实验结论，不是 quality–time–memory Pareto 已测结论。D=6与D=9还不是严格 iso-parameter；latent texture也不是 iso-fetch/iso-byte。[I]

## 9. 实验 protocol、baseline、指标与结果

### 9.1 Fig.1 的完整 image-space FLIP 值

每个 material显示两个2D slices；列顺序与 §4 十种 parameterizations 一致。论文未说明两个 slice 的固定 angles，也未给跨slice聚合。[P Fig.1]

| Material/slice | `(h,d)` sph D4 | `(h,d)` PE D8 | `(h,d)` latent D8 | `(h,d)` cart D6 | `+cosθi` sph D5 | `(ωo,ωi)` sph D4 | direct PE D8 | direct latent D8 | direct cart D6 | direct+h cart D9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Leather11 / 1 | .475866 | .475390 | .453607 | .463470 | .248335 | .274507 | .293402 | .224207 | .233497 | **.186528** |
| Leather11 / 2 | .344900 | .336832 | .390140 | .340868 | .213227 | .229239 | .241053 | .172048 | .160973 | **.146292** |
| Fabric12 / 1 | .338956 | .342167 | .342674 | .337799 | .123376 | .132107 | .144884 | .115269 | **.112790** | .120870 |
| Fabric12 / 2 | .361621 | .373145 | .398609 | .376884 | .124347 | .143243 | .157078 | .115246 | **.107008** | .115999 |

这些数值显示两层事实：

1. 不显式暴露 `cos θi` 的 half/difference前四列在两材质都明显较差；给它追加 `cos θi` 后大幅下降。
2. `h`不是普遍收益：D9 direct+h在 Leather11 最低，但 Fabric12 的 D6 direct Cartesian更低。

### 9.2 Fig.3 training curves

Fig.3画 entire test dataset 的 average L1 difference。曲线视觉上 Leather11 最低的是D9 direct+half Cartesian，Fabric12 最低的是D6 direct Cartesian；作者caption则把结论概括为 Cartesian `(ωo,ωi)` overall outperforms，并指出latent texture可有相近loss但多2次texture fetch。曲线没有raw data/error bar，横轴单位未标，不能从图像自行录入精确终值或收敛速度比例。[P Fig.3, §3]

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释 | locator |
|---|---|---|---|---|---|
| `ablation-inferior` | spherical/PE/latent/Cartesian `(h,d)`，不含`cosθi` | Fig.1 FLIP约`.3378–.4759`，明显差于含light falloff的`.1234–.2483` | tiny input budget下advanced parameterization没有保住关键appearance信息 | chart新颖度不能替代目标量的低阶充分统计 | P §2, Fig.1 |
| `ablation-inferior` | 1-level PE | 两材质都没有超过direct Cartesian | tiny D约束下只能1 level，收益有限 | 不证明PE在更大D/网络普遍失败 | P §2–3 |
| `ablation-inferior / cost` | per-angle 4D latent texture | loss有时接近direct Cartesian，但多2 fetch；Fig.1并不一致占优 | indirect parameterization受budget限制 | quality与read/byte必须一起match | P Fig.3 caption |
| `excluded-by-budget` | learnable shading frames | 需要extra linear layer和更高input D，未进入实验 | 超出tiny MLP budget | 不是已执行失败实验 | P §2 |
| `material-dependent negative` | direct Cartesian再加`h` | Leather11改善，Fabric12变差 | Fabric sheen在grazing，h不help甚至negative | feature clue应按state stratify，不能全局宣布优越 | P §3, Fig.1 |

作者没有报告多seed、不同activation、不同texture capacity、iso-MAC width调整、优化器或更大material集合的失败历史。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Poster | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Architecture | 7-channel texture + `D<10` + 3×8×8 MLP→RGB | 同 | 不可得 | 高层一致；activation/bias/texture layout缺失 |
| Data/query | UBO2014 Leather11/Fabric12；十种方向输入 | 同 | 不可得 | 一致；split/sampling未知 |
| Loss/training | test L1 curves；from scratch | 同 | 不可得 | training objective/横轴/seed未知 |
| Runtime | low-power动机；latent多2个额外fetch | 同 | 不可得 | base fetch、真实backend/timing/bytes均未知 |
| Results | Fig.1 FLIP与Fig.3 L1 | poster复现相同数值/图 | 不可得 | 视觉核对一致 |

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

- 研究有意限制为 `D<10` 与three-8×8-hidden-layer MLP；作者结论适用于 compact budget。[P §2]
- 只用两个代表性 UBO2014 materials；正文没有声称跨全部BTF/material family统一最优。[P §3]
- learnable frame因额外linear layer/维度超预算而排除；这不是同成本质量失败。[P §2]

### 12.2 未报告/材料不可得

texture resolution/coordinate/filter/mip/precision、BTF query measure、train/test split、方向/spatial sampling、target units/transform、activation/bias、optimizer/LR/batch、横轴单位、steps、seed/repeats、checkpoint selection、parameter/MAC/bytes、hardware/backend/latency、official code/raw curves均未报告。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

论文把容量分成三处：7-channel spatial texture、极窄 MLP、direction input。结果说明当MLP容量极小时，input coordinate本身就是容量：direct Cartesian把light falloff作为近线性输入暴露，减少网络再学习球坐标到`cosθi`的负担；而latent-angle texture用存储/read换表达力。

### 13.2 成功所依赖的假设

收益依赖local tangent-frame directions可靠、material response包含由`ωi.z`直接解释的falloff、网络确实tiny、且目标与两个测试BTF相似。对 reciprocity、anisotropy、grazing singularity和层状多峰response的影响没有测试。

### 13.3 可迁移机制与不能迁移的部分

可迁移的是方法学：在 iso-budget下比较 coordinates，并显式包含物理上低阶但重要的量。不能直接迁移的是“永远用 direct Cartesian”结论：本项目当前 RTA有learned frames、20D input和更宽64层，且LayerStack source不是这两个BTF。

### 13.4 与本项目 runtime contract 的关系

在论文披露的angular branch内，fixed-coordinate variants没有额外angle-texture fetch，latent variant增加2次read与state；但7-channel base feature的物理读取数仍未闭合。direct Cartesian只需算/拼方向，因此值得作为fixed-read候选。`h`可在`prepare/evaluate`边界中按query计算；若只依赖fixed view的一部分，可考虑缓存，但不能增加scene-dependent state。[I]

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

当前 `nvidia-rta2024-functional-f@2` 保持2024 faithful identity：z8经两个learned frames投影direct directions，形成20D evaluator input；Taming stable half/difference是后续candidate而不是原结构。[N current correspondence §1–2]

本论文对当前复现的判定是：

| 轴 | 当前状态 | 论文影响 | correspondence |
|---|---|---|---|
| 2024 baseline coordinates | learned frames + direct dirs | 不应静默替换 | `faithful`保留 |
| Direct Cartesian control | 没有单独formal candidate | 应作为coordinate ablation的廉价control | new candidate，非defect fix |
| Stable direct+half/diff | 当前未实现 | 与Taming候选一起比较，但需iso-MAC | new candidate |
| Learned frames | 当前有 | poster因tiny budget排除，不证明当前64-wide结构不该用 | `not-applicable` to removal |
| Bare-f output | 当前项目接口适配 | poster只称RGB reflectance，measure未报告 | 无法裁决 |

它最重要的提醒是：coordinate实验必须把input D导致的首层参数/MAC差异和latent fetch算入预算，且按材质状态分层报告；不能只看平均loss。

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| `H-AP1`：显式direct Cartesian与`cosθi`能在同MAC小网络中优于只用half/diff | P Fig.1–3 | LayerStack queries也受light-falloff表达负担影响 | current learned-frame direct；raw tangent direct D6；half/diff D4；half/diff+cos D5；调width使总params/MAC matched | source/query、loss、optimizer、steps、seeds、texture/state bytes | stratified directional error、grazing/peak、energy、variance、query time | local evaluator | raw direct/`+cos`在matched成本与CI下不优，或只改善两种BTF相似state而损害总体Pareto |
| `H-AP2`：direct+h是specular-state条件性特征而非全局默认 | Leather改善、Fabric变差 | 本项目state可按roughness/peak complexity分层 | D6 direct vs D9 direct+h，另做iso-MAC width control；不同时换loss/activation | 同上 + state strata | peak alignment、grazing、diffuse、overall CI、latency | local evaluator | h在各state均无interaction，或平均/尾部损失抵消specular收益 |
| `H-AP3`：angle latent texture不值得其2 fetch | P Fig.3近似loss但多2 fetch | 当前GPU query也受read/state约束 | direct fixed coordinates vs two angle-latent reads；iso-param与iso-read分别报告 | texture resolution/filter、MLP、training、precision | quality/time/bytes/fetch Pareto | local evaluator fixed-read | latent在iso-byte/read控制下显著支配fixed coordinates |

## 16. 证据索引

- `P p.1 Fig.1`：十列坐标、D、四个slice的FLIP数值、spherical normalization。
- `P §1–2 pp.1–2`：low-power动机、7-channel texture、`D<10`、3×8×8 MLP、RGB output、十种参数化、PE/latent/frame预算。
- `P §3 p.2 Fig.3`：UBO2014 Leather11/Fabric12、from-scratch、test L1曲线、direct Cartesian与half-vector材料依赖结论。
- `A poster`：与P一致的图例、实验与数值；用于视觉交叉核对。
- `N`：[当前 NVIDIA correspondence](../implications/current-nvidia-correspondence.md)、[Taming report](bitterli-2026-taming-optimization-variance.md)、[RTA report](zeltner-2024-real-time-neural-appearance-models.md)。
- `I`：§13–15。

## Evidence review

```text
author_worker: /root
reviewer: /root/dualband2025_review
reviewed_at: 2026-08-29
sources_rechecked: [author-hosted 2-page formal abstract, author-hosted 1-page official poster]
findings_closed:
  - visually rechecked 10 configurations, all D values, 40 FLIP values and both curve legends
  - corrected Leather11/Fabric12 Fig.3 winner wording
  - narrowed texture-fetch claims to two extra latent fetches and kept base 7-channel fetch/layout unknown
  - separated author conclusions from project budget/Pareto inferences
remaining_evidence_gaps:
  - no supplemental, official code/config/data or raw curves
  - texture layout/filter/precision, target measure, activation, optimizer, split, sampling, x-axis unit, seeds and checkpoint selection unreported
  - no measured hardware timing, parameter/MAC/bytes or exact training lifecycle
review_status: evidence-reviewed
```

### 完成检查

- [x] 两页 main abstract 已完整阅读，三图、图注和十列数值已视觉核对；
- [x] supplemental/appendix/勘误的可用性已检查；
- [x] official code/config/data 的可用性已检查；
- [x] architecture、training、runtime 和结果均按两页材料的真实披露深度给 locator；
- [x] 失败尝试与较差消融正确分类，learned frame只标为budget exclusion；
- [x] paper/code gap 和“未报告”保留；
- [x] `I` 分析晚于事实层；
- [x] NVIDIA 影响引用当前identity；
- [x] 假设包含 matched control、部署类别和证伪条件；
- [x] 独立 evidence review 已完成。
