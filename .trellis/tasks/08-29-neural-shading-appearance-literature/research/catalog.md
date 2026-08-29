# 文献 Catalog 与研究状态

本文是本任务论文身份、来源状态、研究波次、关系和报告完成度的唯一共享索引。个体 worker 不直接修改本文；主会话根据 worker handoff 合并状态。

最后更新：2026-08-29。

## 1. 状态词汇

### 1.1 来源状态

| 状态 | 含义 |
|---|---|
| `discovered` | 只确认标题或候选身份，尚未锁定第一方正文 |
| `triaged` | 已找到至少一个第一方入口，尚未完成全文/补充材料/代码清点 |
| `sources-locked` | 正文、supplemental、项目页和代码可用性已逐项登记，版本/commit/hash 已锁定 |
| `blocked-source` | 已记录检索路径，但必要第一方材料当前不可得 |

### 1.2 报告状态

| 状态 | 含义 |
|---|---|
| `not-started` | 尚未分配 worker |
| `author-pass` | 已分配并正在完整阅读/写作 |
| `report-draft` | author pass 完成，尚未独立复核 |
| `evidence-reviewed` | 已完成独立证据复核，可进入综合 |
| `complete` | 复核问题已关闭，catalog 和关系已回写 |
| `source-audit-draft` | 必要正文不可得；已建立16节来源审计草稿，尚未独立复核；不能进入机制综合 |
| `blocked-source-audit` | 来源审计已独立复核，但完整方法报告仍被必要第一方材料阻塞；不能替代 `evidence-reviewed` |

`complete` 不表示论文公开了所有实现细节；缺失材料可以在明确证据边界后成为完成报告。

## 2. 初始必研集合

### 2.1 波次 1：局部 neural material / appearance

| paper_id | 正式标题 | class | 第一方入口 | source | report | 报告路径 |
|---|---|---|---|---|---|---|
| `zeltner-2024-real-time-neural-appearance-models` | Real-Time Neural Appearance Models | `local-material` | [NVIDIA project](https://research.nvidia.com/labs/rtr/neural_appearance_models/) | `source-locked` | `evidence-reviewed` | `papers/zeltner-2024-real-time-neural-appearance-models.md` |
| `bitterli-2026-taming-optimization-variance` | Taming Optimization Variance in Compact Neural Shading Networks | `local-material` | [author project](https://www.jannovak.info/publications/NAP-smallnets/index.html)、[official code](https://github.com/NVlabs/neuralappearance) | `source-locked` | `evidence-reviewed` | `papers/bitterli-2026-taming-optimization-variance.md` |
| `xu-2026-real-time-neural-materials-mobile-vr` | Real-Time Neural Materials on Mobile VR | `local-material` | [publisher/DOI](https://doi.org/10.1111/cgf.70318) · [author-hosted paper](https://lingqiyan.github.io/) | `source-locked` | `evidence-reviewed` | `papers/xu-2026-real-time-neural-materials-mobile-vr.md` |
| `kuznetsov-2021-neumip` | NeuMIP: Multi-Resolution Neural Materials | `local-material` | [author project](https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP/) | `source-locked` | `evidence-reviewed` | `papers/kuznetsov-2021-neumip.md` |
| `fan-2023-neural-biplane-btf` | Neural Biplane Representation for BTF Rendering and Acquisition | `local-material` | [author project](https://wangningbei.github.io/2023/BIPLANEBTF.html) · [paper](https://sites.cs.ucsb.edu/~lingqi/publications/paper_biplane.pdf) · [supplemental](https://sites.cs.ucsb.edu/~lingqi/publications/supplementary_biplane.pdf) | `source-locked` | `evidence-reviewed` | `papers/fan-2023-neural-biplane-btf.md` |
| `xu-2025-comprehensive-neural-materials` | Towards Comprehensive Neural Materials: Dynamic Structure-Preserving Synthesis with Accurate Silhouette at Instant Inference Speed | `local-material` | [author project](https://starry316.github.io/sig2025/index.html) · [renderer code](https://github.com/Starry316/ComprehensiveNeuralMaterial) · [training code](https://github.com/Starry316/ComprehensiveNeuralMaterial-Train) | `source-locked` | `evidence-reviewed` | `papers/xu-2025-comprehensive-neural-materials.md` |
| `fan-2022-neural-layered-brdfs` | Neural Layered BRDFs | `local-material` | [author project](https://wangningbei.github.io/2022/NLBRDF.html) · [author PDF](https://sites.cs.ucsb.edu/~lingqi/publications/paper_NLBRDF.pdf) | `source-locked` | `evidence-reviewed` | `papers/fan-2022-neural-layered-brdfs.md` |
| `2023-metalayer` | MetaLayer: A Meta-Learned BSDF Model for Layered Materials | `local-material` | [author PDF](https://sites.cs.ucsb.edu/~lingqi/publications/paper_siga23metalayer.pdf)、[DOI](https://doi.org/10.1145/3618365) | `source-locked` | `evidence-reviewed` | `papers/2023-metalayer.md` |
| `2021-neural-brdf-representation-importance-sampling` | Neural BRDF Representation and Importance Sampling | `local-material` | [publisher/DOI](https://doi.org/10.1111/cgf.14335) | `source-locked` | `evidence-reviewed` | `papers/2021-neural-brdf-representation-importance-sampling.md` |
| `bai-2023-bsdf-importance-baking` | BSDF Importance Baking: A Lightweight Neural Solution to Importance Sampling General Parametric BSDFs | `local-material` | [arXiv v3](https://arxiv.org/abs/2210.13681)；后续 [2025 CGF VOR](https://doi.org/10.1111/cgf.70286) 当前全文受阻 | `source-locked` | `evidence-reviewed` | `papers/bai-2023-bsdf-importance-baking.md` |
| `weier-2023-neural-prefiltering-lod` | Neural Prefiltering for Correlation-Aware Levels of Detail | `scene-transport / asset-prefilter` | [DOI/open paper](https://doi.org/10.1145/3592443)、[official code](https://github.com/WeiPhil/neural_lod) | `source-locked` | `evidence-reviewed` | `papers/weier-2023-neural-prefiltering-lod.md` |
| `2026-hybrid-neural-microfacet-brdf` | A Hybrid Neural-Microfacet BRDF Model for Real-Time Rendering | `local-material` | [DOI/VOR](https://doi.org/10.1111/cgf.70540)、[author project/demo](https://ubisoft-laforge.github.io/world/hybridrdf/) | `source-locked` | `evidence-reviewed` | `papers/2026-hybrid-neural-microfacet-brdf.md` |
| `2026-neural-material-adapter` | Neural Material Adapter: Transforming Complex Materials into Efficient Analytic BSDFs | `local-material` | [Disney Research](https://studios.disneyresearch.com/2026/07/01/neural-material-adapter-transforming-complex-materials-into-efficient-analytic-bsdfs/) | `source-locked` | `evidence-reviewed` | `papers/2026-neural-material-adapter.md` |

### 2.2 波次 2：场景级与体积 neural light transport

| paper_id | 正式标题 | class | 第一方入口 | source | report | 报告路径 |
|---|---|---|---|---|---|---|
| `sheng-2025-nelif` | NeLiF: Neural Lighting Function Generation for Real-Time Indoor Rendering | `scene-transport` | [publisher/DOI](https://doi.org/10.1145/3757377.3763958)；本地 `3757377.3763958.pdf` | `source-locked` | `evidence-reviewed` | `papers/sheng-2025-nelif.md` |
| `zheng-2023-nelt` | NeLT: Object-Oriented Neural Light Transfer | `scene-transport` | [publisher/DOI](https://doi.org/10.1145/3596491)；本地 `3596491.pdf` | `source-locked` | `evidence-reviewed` | `papers/zheng-2023-nelt.md` |
| `zheng-2024-superposed-deformable-feature-fields` | Neural Global Illumination via Superposed Deformable Feature Fields | `scene-transport` | [publisher/DOI](https://doi.org/10.1145/3680528.3687680)；本地 `3680528.3687680.pdf` | `source-locked` | `evidence-reviewed` | `papers/zheng-2024-superposed-deformable-feature-fields.md` |
| `mo-2025-dual-band-neural-gi` | Dual-Band Feature Fusion for Neural Global Illumination with Multi-Frequency Reflections | `scene-transport` | [publisher/DOI](https://doi.org/10.1145/3721238.3730733) · [author project](https://mshnb.github.io/dualbandfusion/) | `sources-locked` | `evidence-reviewed` | `papers/mo-2025-dual-band-neural-gi.md` |
| `ren-2024-lightformer` | LightFormer: Light-Oriented Global Neural Rendering in Dynamic Scene | `scene-transport` | [author project with paper/supp](https://wylighting.github.io/lightformer/) | `sources-locked` | `evidence-reviewed` | `papers/ren-2024-lightformer.md` |
| `guo-2022-neural-light-probes` | Efficient Light Probes for Real-time Global Illumination | `scene-transport` | [author PDF](https://sites.cs.ucsb.edu/~lingqi/publications/paper_nlp.pdf)、[author publication list/video](https://lingqiyan.github.io/) | `source-locked` | `evidence-reviewed` | `papers/guo-2022-neural-light-probes.md` |
| `1469-2026-volumetric-light-transport-inference` | Real-Time Volumetric Light Transport Inference from Auxiliary Renderings | `volume-transport` | `paper1469_1.pdf`（本地匿名 Pacific Graphics 2026 稿，SHA-256 `963292…2F6`）；正式作者/DOI 待身份公开后核对 | `source-locked` | `evidence-reviewed` | `papers/1469-2026-volumetric-light-transport-inference.md` |

### 2.3 已解除的 `blocked-source` 审计

2026-08-29，用户把三份正式正文放入项目根目录。它们只作为本次研究的本地输入，不复制到任务 `research/`、不登记为根仓库资产，也不需要账号、SSH token 或登录。三份正文均完成全文读取、逐页渲染、作者稿和未参与写作 reviewer 的独立证据复核：

| paper_id | 本地来源 | 锁定证据 | 仍未关闭的来源缺口 |
|---|---|---|---|
| `sheng-2025-nelif` | `3757377.3763958.pdf`，11页 | SHA-256 `07558A3B7D7CA47091337F3A5A41E4D57B0BB02F3EB5A0387D3DBF0F47D398DD`；Eq.1–11、Fig.1–12、Table1逐页复核 | 正文明确提到的 supplemental 当前不可得；official code/config/data、完整训练配置和runtime分账未公开 |
| `zheng-2023-nelt` | `3596491.pdf`，16页 | SHA-256 `56566C3D92F29EF6DF6D4B424DECB18CE1BFB9CF2582EC275D4213E028CA5ED5`；Eq.1–10、Fig.1–13、Table1–7逐页复核 | supplemental/video与official code/config/data不可得；exact topology、ratio稳定化、state bytes和representation-build成本未闭合 |
| `zheng-2024-superposed-deformable-feature-fields` | `3680528.3687680.pdf`，11页 | SHA-256 `4D5B6E0BB79D274735A8DD5BB4980665396B135241ED2D8476C2BAD8D31F8E0F`；Eq.1–9、Fig.1–12、Table1–2逐页复核 | supplemental/video与official code/config/data不可得；field尺寸、offset regularization、`α` schedule和完整runtime账本未闭合 |

旧的匿名访问失败记录仍解释为什么此前只能形成 `blocked-source-audit`；它现在是已解除的历史来源状态，不再限制波次2综合。报告没有用搜索摘要、第三方副本或相邻论文配置填补仍缺失的 supplemental/code 字段。

## 3. Load-bearing 候选队列

本节只登记从核心论文中实际触发的候选。一般 inverse rendering、NeRF/3DGS、生成式材质和 neural reconstruction 不因关键词相似自动进入完整报告。

| candidate_id | 来源核心论文 | promotion_trigger | 决定 | 理由/证据 | 状态 | 报告路径 |
|---|---|---|---|---|---|---|
| `guo-2018-position-free-layered-bsdfs` | `fan-2022-neural-layered-brdfs`、`2023-metalayer`、`bai-2023-bsdf-importance-baking` | `key-baseline` | 完整报告 | 三篇核心论文都把它作为 layered BSDF 的权威 evaluation/sampling reference 或直接质量—时间 baseline；不掌握其 position-free state 与 Monte Carlo operation，就无法判断 neural 方法究竟替代了哪段计算。 | `evidence-reviewed` | `papers/guo-2018-position-free-layered-bsdfs.md` |
| `belcour-2018-efficient-rendering-layered-materials` | `fan-2022-neural-layered-brdfs`、`2023-metalayer` | `key-baseline` | 完整报告 | 两篇 layered neural 方法都依赖它作为解析/低维层合 baseline；需要核对其统计表示、层合近似和失败边界，避免把“解析 baseline”误写成同一种 GT。 | `evidence-reviewed` | `papers/belcour-2018-efficient-rendering-layered-materials.md` |
| `zheng-2021-neural-process-brdfs` | `zeltner-2024-real-time-neural-appearance-models`、`bai-2023-bsdf-importance-baking` | `direct-inheritance` + `key-baseline` | 完整报告 | RTA 的 function-space latent、log objective/reciprocity讨论与 normalizing-flow sampler 都直接沿用或对照该方法；只读 RTA 转述会丢失 encoder/decoder、MERL/EPFL 与 sampler 配置。 | `evidence-reviewed` | `papers/zheng-2021-neural-process-brdfs.md` |
| `xu-2025-improving-angular-parameterization` | `bitterli-2026-taming-optimization-variance` | `direct-inheritance` + `failure-explanation` | 完整报告 | Taming 明确以它说明 tiny MLP 的 direction tuple 选择；当前 correspondence 又把 coordinates 列为独立候选轴。正式来源只有2页poster abstract，但其十种配置和负结果可完整恢复。 | `evidence-reviewed` | `papers/xu-2025-improving-angular-parameterization.md` |
| `xue-2024-hierarchical-neural-materials` | `bitterli-2026-taming-optimization-variance`、`xu-2025-improving-angular-parameterization` | `direct-inheritance` + `failure-explanation` | 完整报告 | Taming 的root/power loss把它作为直接precedent；它也提供multi-scale architecture、frequency encoding与gradient loss，能区分“loss启发”与完整representation。 | `evidence-reviewed` | `papers/xue-2024-hierarchical-neural-materials.md` |
| `xia-2020-gaussian-product-sampling` | `guo-2018-position-free-layered-bsdfs` | `failure-explanation` + `runtime-transfer` | 完整报告 | 它直接针对 position-free sequential proposal 的高variance，提出pair/multiple-product sampling；当前项目的matched sampler/reference轨需要知道它降低哪类path variance及付出何种近似/预计算。 | `evidence-reviewed` | `papers/xia-2020-gaussian-product-sampling.md` |
| `diolatzis-2022-active-exploration-neural-gi` | `mo-2025-dual-band-neural-gi`、`ren-2024-lightformer` | `key-baseline` + `runtime-transfer` | 完整报告 | AE/AE-Ref 是两篇scene方法解释高频transport和训练data coverage的关键baseline；其MCMC active data generation与sample reuse也是可迁移到online reference query的训练机制。 | `evidence-reviewed` | `papers/diolatzis-2022-active-exploration-neural-gi.md` |
| `granskog-2020-compositional-neural-scene-representations` | `ren-2024-lightformer` | `key-baseline` + `direct-inheritance` | 完整报告 | CNSR是LightFormer主要neural baseline与显式scene-parameter路线前序；LightFormer的“light-oriented而非object parameter vector”结论依赖正确理解CNSR输入、composition与成本。 | `evidence-reviewed` | `papers/granskog-2020-compositional-neural-scene-representations.md` |

`FieldGI`不是另一篇待建论文：Dual-Band 对 “simplified FieldGI” 的引用指向初始集合中的 `zheng-2024-superposed-deformable-feature-fields`。该身份已由正式正文和后续论文关系复核，不重复计数。

### 3.1 已登记但本轮不提升的 discovery-only 条目

| candidate | 来源 | 当前决定 | 未提升理由/重新触发条件 |
|---|---|---|---|
| MIPNet / LEAN mapping | RTA | `discovery-only` | RTA已给出本项目需要的hierarchical/filter correspondence；只有过滤综合需要复现原始moment/filter算法或将其变成candidate时，才按`direct-inheritance`提升。 |
| successive halving / Hyperband 原始算法论文 | Taming | `discovery-only` | 当前任务研究的是Taming正式固定schedule，不推导通用超参优化理论；Taming P/C已足以重建其预算。若设计自适应schedule，再提升。 |
| SmeLU 原始论文 | Taming | `discovery-only` | Taming正文与official code已经给出LeakySmeLU函数及常量；原始SmeLU不会改变本次neural-shading correspondence。若把activation单独发展成方法族，再提升。 |
| Rainer et al. 2019/2020 Neural BTF Compression | NLB / angular poster | `discovery-only` | 当前只承担BTF representation qualitative baseline；没有让本任务主要结论依赖其独立数值。若综合要比较unified BTF latent或grazing failure，再提升。 |
| LayerLab / Jakob et al. | Belcour | `discovery-only` | Belcour P/S已提供不同NDF下的storage/precompute语境；本任务不把高维tabulation实现为候选，也不直接排名。 |
| Weidlich & Wilkie layered model | Belcour / NMA | `discovery-only` | 当前只用于解释特定layer-decoupling baseline failure；Belcour/NMA一手材料已保留边界。若要实现production clear-coat control，再提升。 |
| Neural Shadow Mapping、classic Instant Radiosity、ONND/OIDN | LightFormer | `discovery-only` | 它们分别是clue灵感、classic control和denoiser controls，不是LightFormer neural representation的直接依赖；当前P/S已给足baseline protocol，独立全文不会改变local-material启发。 |
| 一般 inverse rendering、NeRF/3DGS、生成式材质与reconstruction | 多篇 related work | `discovery-only` | 只满足大领域相似性，没有改变本任务local evaluator、scene transport、filter/sampler/compiler或部署结论。 |

允许的 `promotion_trigger`：`direct-inheritance`、`key-baseline`、`failure-explanation`、`runtime-transfer`、`user-specified`。

## 4. 初始关系图

以下关系只是 source discovery 的阅读顺序，不是已经证明的机制结论；完成个体报告后才能升级为综合证据。

```text
Neural BRDF Representation ── evaluator / analytic proposal ──┐
NeuMIP ── spatial latent + offset + scale ────────────────────┼─> Real-Time Neural Appearance Models
                                                              └─> Mobile VR neural materials

Neural Layered BRDFs ── latent composition ──> MetaLayer ── feed-forward material compiler

NeuMIP ── spatial appearance ──> Neural Biplane BTF ──> Comprehensive Neural Materials

Active Exploration ── scene-network baseline（NeLT/SDF均只测AE uniform）──┐
NeLT ── ordered object transfer + hypernetwork ──> Superposed Deformable Feature Fields
                                                     └─ simplified FieldGI ──> Dual-Band Neural GI

LightFormer ── per-light observations + indirect module ──> NeLiF
NeLiF ── 把per-frame light aggregation前移成generated lighting field；NeLT/SDF是其per-scene边界对照

Neural Light Probes ── probe representation ──> scene transport 对照
Volumetric auxiliary inference ── image-space deterministic decoding ──> scene/volume transport 对照
```

## 5. 项目内 `N` 类证据

这些文件帮助检查 NeuralShading 当前方向和复现状态，但不能替代论文正文：

| evidence_id | 路径 | 用途 |
|---|---|---|
| `n-prior-art` | `docs/research/prior_art.md` | 发现论文、理解当前候选分类；其中摘要必须回到第一方来源复核 |
| `n-experiment-framework` | `docs/research/experiment_framework.md` | 把论文启发转成 matched 实验和指标时的项目合同 |
| `n-model-candidates` | `docs/research/model_candidates.md` | 当前候选与历史 empirical boundary；不作为论文方法事实 |
| `n-nvidia-faithful-task` | `.trellis/tasks/archive/2026-08/08-27-faithful-nvidia-neural-materials/` | 当前 NVIDIA 功能复现、correspondence 和预算适配证据 |
| `n-reference-candidates-task` | `.trellis/tasks/archive/2026-08/08-27-reference-material-candidates/research/` | NVIDIA 资产与 Lingqi Yan 外观路线的旧调研；只作为来源发现和项目证据 |

## 6. Worker 与 reviewer 记录

| paper_id | author worker | author pass | reviewer | review status | 未关闭问题 |
|---|---|---|---|---|---|
| `zeltner-2024-real-time-neural-appearance-models` | `/root/rta2024` | author-pass-complete | `/root/nbrdf2021` | evidence-reviewed | P/S 的 bitangent normalization 冲突未解；2024 原始资产、配置、KL estimator、stage split 与 seeds 未公开；当前 `functional-f@2` 的迁移说明已在专项 correspondence 闭合，但仍无300k formal artifact，旧200k `functional@1`结果不能验证新身份 |
| `bitterli-2026-taming-optimization-variance` | `/root/taming2026` | author-pass-complete | `/root/nbrdf2021` | evidence-reviewed | Table 1 的 15/24 trials、79.1%/74.4% 与 98.7%/99.4% 冲突无官方勘误；正式资产/query/raw logs/runtime 未公开 |
| `2021-neural-brdf-representation-importance-sampling` | `/root/nbrdf2021` | author-pass-complete | `/root/taming2026` | evidence-reviewed | AE/predictor code、anisotropic formal assets、`±`定义与曲线原始数据不可得 |
| `kuznetsov-2021-neumip` | `/root` | author-pass-complete | `/root/nbrdf2021` | evidence-reviewed | Drive 资产404、无OptiX/CUDA源码、Table 2/3 checkpoint identity 未解 |
| `fan-2023-neural-biplane-btf` | `/root` | author-pass-complete | `/root/rta2024` | evidence-reviewed | checkpoint 未下载；formal loss/response correspondence、Offset、完整数据/render/calibration 与 FLOP 口径未解 |
| `xu-2025-comprehensive-neural-materials` | `/root` | author-pass-complete | `/root/nbrdf2021` | evidence-reviewed | SharePoint assets/checkpoints 未审计；5.68 KB、2.43/3.4 ms、Fig.12 Ceramic `7.5×` vs `2.19/0.25≈8.76×` 冲突未解；无 formal figure manifest/split/metric/seed/output units；公开 helper 的 final activation、U16、scheduler、log response、metadata、Int8/export/loader gaps 仅完成 static correspondence，未端到端执行 |
| `fan-2022-neural-layered-brdfs` | `/root/taming2026` | author-pass-complete | `/root/rta2024` | evidence-reviewed | formal training/CUDA renderer/sampler code 不可得；optimizer、batch composition、seed 未报告；correction 撤回约 `5 ms` 与 faster-than-Belcour runtime 声明，但没有重新配平 Table 3；公开数据包另有 batch-slice defect |
| `guo-2018-position-free-layered-bsdfs` | `/root/rta2024` | author-pass-complete | `/root` | evidence-reviewed | 主文 14/14 页与两份 supplemental 各 3/3 页独立复核；修正草稿中 formulation/estimator/results 整体提前一节及 white-furnace Fig.9→Fig.10 的 locator 错误；paper `diffusePdf=0.1` vs Fig.8 XML `1`、formal 双深度界到单 `stochPdfDepth`、`pdf=TRT` 静态疑点和动态复现仍未闭合 |
| `belcour-2018-efficient-rendering-layered-materials` | `/root` | author-pass-complete | `/root/belcour2018_review` | evidence-reviewed | 正式主文15/15页、supplemental 8/8页、HAL code/data archive与作者talk已独立复核；Eq.17/Eq.24上标、正文↔talk↔代码折射公式、roughness映射、Dielectric TIR、`sigma_s=0`媒体分支、Fig.13 XML预算及无commit/license等冲突均保留为显式gap |
| `bai-2023-bsdf-importance-baking` | `/root` | author-pass-complete | `/root/nbrdf2021` | evidence-reviewed | 2025 CGF VOR 正文与 53.6 MB supporting information 被 Wiley 403 阻断，不得继承 2023 配置；2023 promised database、code/config/weights、exact disk/OT 配置、参数范围、split/seed、precision 与 single-query runtime 不可得 |
| `weier-2023-neural-prefiltering-lod` | `/root/taming2026` | author-pass-complete | `/root` | evidence-reviewed | ACM presentation 403、Drive 场景包未逐文件锁定；`log1p`、threshold 配置、训练步数、Arcade 参数量顺序与 SH degree 命名口径存在未闭合 paper↔code 差异 |
| `2026-hybrid-neural-microfacet-brdf` | `/root` | author-pass-complete | `/root/nbrdf2021` | evidence-reviewed | 正式 VOR/supp/artifact、32×3/UTIA 与 demo 范围已复核；`.0893`、300/312、five/four families、isotropic 11/15 vs 12/16 等内部冲突保留；训练/BRDFExplorer/CoopVec 源码、raw metrics/checkpoints、完整 optimizer、precision 与 formal MIS mixture 未公开 |
| `2026-neural-material-adapter` | `/root/taming2026` | author-pass-complete | `/root/adapter2026_review` | evidence-reviewed | 正文、supplemental、AvA/W&W 与 WebGL prototype 已锁定；方向语义、reciprocity、loss、PFMC sample budget、89/92 sets 和 lobe 数冲突已逐项复核并保留；官方训练/runtime code、config、checkpoint/data 与缺失 N=5 结果不可得 |
| `guo-2022-neural-light-probes` | `/root/lightprobes2022` | author-pass-complete | `/root/lightprobes2022_review` | evidence-reviewed | 正式 main 14/14 页已逐页复核；修正 Eq. 3 向量方向和 decoder 末层；Eq. 3/4 confidence 语义、shortened-step cap、搜索更新、GT/split、optimizer、precision、timing 与官方 code/config/data 仍未报告 |
| `2023-metalayer` | `/root/taming2026` | author-pass-complete | `/root/nbrdf2021` | evidence-reviewed | 无 official code/config/data；BSDFNet exact layer order、MetaNet activation、training batch/`K1/K2`/split/seed、sampler 实现与独立验证仍不可恢复；168D SH 与 “ninth order”歧义已按证据边界保留 |
| `xu-2026-real-time-neural-materials-mobile-vr` | `/root` | author-pass-complete | `/root/rta2024` | evidence-reviewed | official code/video 不可得；`T` 矩阵朝向及 `7.4×` 与 Table 1 约 `4.1×` 的加速口径冲突未解析；modified NeuMIP 完整训练 parity、activation、split、seed 与 runtime precision 未报告 |
| `1469-2026-volumetric-light-transport-inference` | `/root` | author-pass-complete | `/root/taming2026` | evidence-reviewed | 匿名身份/DOI、supp/code/config/data 不可得；精确 input packing/topology/feature budget/训练超参/temporal metric 与原始 Fig.8 数值未报告；Fig.3 `f_s` 与 §3.3 `g,φ` packing 未解析 |
| `mo-2025-dual-band-neural-gi` | `/root/dualband2025` | author-pass-complete | `/root/dualband2025_review` | evidence-reviewed | 正文11/11页、supplemental 4/4页与项目站点commit已独立复核；修正`1–2 ms`组件归属、final LeakyReLU、Eq.(3)/(9)符号索引和未报告的triplane分辨率/容量；code/config/checkpoint、stage长度、split、temporal/dynamic-light protocol、gamma merge与完整timing scope仍缺失 |
| `ren-2024-lightformer` | `/root/lightformer2024` | author-pass-complete | `/root/lightformer2024_review` | evidence-reviewed | 正文、supplemental、官方视频与viewer assets已独立复核；保留attention value concat/log-transform、encoder-vs-decoder skip、VPL跨帧复用/RNG/buffer更新、ONND/OIDN配置范围等证据边界；code/config/checkpoint/data、steps/schedule/seeds、参数/MAC/bytes、TensorRT precision/GPU、runtime breakdown与formal temporal metric仍不可得 |
| `sheng-2025-nelif` | `/root/nelif_full_report` | author-pass-complete | `/root/nelt_full_report` | evidence-reviewed | main 11/11页已复核；修正LightFormer身份、Fig.3 wiring、Eq.9字形与Eq.1记号gap；supplemental/code、1M↔400K run identity、逐层配置和runtime breakdown仍未闭合 |
| `zheng-2023-nelt` | `/root/nelt_full_report` | author-pass-complete | `/root/belcour2018_review` | evidence-reviewed | main 16/16页已复核；补齐Table5四个4-spp行，限定direct/indirect各自texture fetch并登记Eq.3近零分母策略缺口；supplemental/code、exact topology/state bytes仍不可得 |
| `zheng-2024-superposed-deformable-feature-fields` | `/root/belcour2018_review` | author-pass-complete | `/root/nelif_full_report` | evidence-reviewed | main 11/11页已复核；闭合两层superposition与AE/NeLT/Dual-Band关系，并补“整体framework而非新单模块”边界；supplemental/video/code和field/runtime配置仍不可得 |
| `zheng-2021-neural-process-brdfs` | `/root/lightformer2024_review` | author-pass-complete | `/root/dualband2025_review` | evidence-reviewed | Fig.3↔checkpoint encoder/aggregator topology、per-material↔per-observation latent与3.11 MB decoder↔3.10 MB total冲突已独立复核并保留；supplemental、formal training/NICE code、完整PDF measure与TensorRT协议仍不可得 |
| `xu-2025-improving-angular-parameterization` | `/root` | author-pass-complete | `/root/dualband2025_review` | evidence-reviewed | 十种配置、输入维度、40 个 FLIP 数值与三层 `8×8` hidden-layer 表述已复核；修正 latent texture/read 过度具体化及 Leather11/Fabric12 最优曲线归纳；训练、runtime 与基础 7-channel feature 布局仍未报告 |
| `xue-2024-hierarchical-neural-materials` | `/root/belcour2018_review` | author-pass-complete | `/root/dualband2025_review` | evidence-reviewed | 正式PDF、arXiv版本/source、official code与H5已独立复核；fourth-root↔Eq.(2)、Sobel↔gradient-L1、Fourier坐标、adaptive schedule/default等冲突保留；Wiley supplemental仍受403阻断 |
| `xia-2020-gaussian-product-sampling` | `/root` | author-pass-complete | `/root/belcour2018_review` | evidence-reviewed | pair/multiple-product数学、projected measure、internal proposal↔external sample/pdf和正式结果已独立复核；匿名bitstream直下返回401/403，未登录且无本地PDF hash/逐页render；official supplemental/code/config未发现 |
| `diolatzis-2022-active-exploration-neural-gi` | `/root` | author-pass-complete | `/root/lightformer2024_review` | evidence-reviewed | main 18/18页、supplemental 4/4页与official commit已独立复核；Sphere paper 11D↔code/checkpoint 12D、8-layer计数、proposal、ArchViz 24/36h与replay/resume gaps保留，fresh batch已修正为16–32 |
| `granskog-2020-compositional-neural-scene-representations` | `/root` | author-pass-complete | `/root/belcour2018_review` | evidence-reviewed | main 13页、supplemental 7页和official single commit已锁定并逐页/静态审计；独立复核修正gradient regularization、GQN 20M例外、null-loss归一化与cached-SPP身份，并保留formal dataset/config/runtime缺口 |

主会话在每次 dispatch、handoff 和 evidence review 后更新本表。worker 不直接修改共享表。
