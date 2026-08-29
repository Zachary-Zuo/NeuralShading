# 可复现实验假设：从文献机制到 NeuralShading 候选

## 1. 用途与排序原则

本文把已复核论文中的机制转成**可被后续planning冻结**的 matched experiment contracts。它是研究假设队列，不是已授权启动的训练、评测、scene基础设施或backend实现计划；任何 formal run 仍需在 `docs/research/experiment_framework.md` 下另行冻结 source locator/snapshot、online typed route/query recipe、method/config identity、seed策略、预算、checkpoint selection 与验收来源。

排序只反映三件事：

1. 第一方证据是否直接、是否有作者消融/负结果；
2. 与当前 `nvidia-rta2024-functional-f@2` **源码/配置身份**的数据、ABI、预算和基础设施迁移距离；
3. 能否用一个隔离变量的短matched实验否证。

它不按论文跨硬件质量/速度排名，也不把历史 observed result、论文胜负或 hypothesis falsification condition 变成任务 hard gate。所有 matched result 都用冻结 source state 为 bootstrap 单位，至少 1,000 次重采样；主指标沿用 solid-angle weighted normalized L1 与 energy relative error，结构 scorecard 另含 log、peak、top-energy recall、reciprocity 和分层 tail。部署按 `C_prepare/C_eval/C_sample/C_pdf`、`B_shared/B_asset/system buffers/prepared-state bytes` 与真实 GPU 时间登记。

当前 NVIDIA 证据必须分层：`functional-f@2` 是 bare-`f` ABI、online typed routes 与 300k/100k lifecycle 的**现行代码/配置 identity**，本任务没有产生其 formal 300k checkpoint/package；归档 200k artifact 属于旧 `nvidia-rta2024-functional@1`，只能作为旧 identity 的历史 observation，不能参与 `@2` checkpoint selection、质量 baseline、回归结论或任何 matched CI。下文若写“current NVIDIA baseline”，只指现行结构/config contract；真正运行前必须生成新的同 identity artifact，或显式登记另一个 adaptation identity。

runtime class 统一使用六类：`training-only diagnostic`、`local evaluator diagnostic`、`local sampling diagnostic`、`full local scattering program`、`deployment-only`、`separate scene renderer`。只有 `full local scattering program` 可按 `docs/contracts/scattering_backend.md` 进入 path tracing，并且必须由同一 `prepare` state 提供 `evaluate/sample/pdf`，连续 sample 满足 `weight=f|n_s·wi|/pdf` 与同proposal/event/measure合同；缺任一 capability 时 fail closed。只研究 evaluator 或sampler的候选可以存在，但在四入口合同闭合前必须标为不可部署 diagnostic。所有可晋级候选仍只能经 `MethodDefinition` 接入，并登记静态有界的读取、状态和control flow；不得为某 hypothesis 增加专用 runner/exporter/viewer 分支。runtime class只描述产物/验收边界，不构成run、基础设施或预算授权；同一hypothesis若包含training-only轴与runtime-function轴，必须分别登记class，不能用一个总标签掩盖。

## 2. 优先级总表

| Priority | hypothesis_id | 核心问题 | 排序依据：证据与迁移成本 | 执行依赖 | runtime class |
|---:|---|---|---|---|---|
| P0 | `H-O1` | power-map、stable coordinates、activation、successive schedule各自贡献是什么？ | Taming与当前compact family机制最接近；各轴可隔离，但runtime身份不同 | 冻结一个current-identity或显式adaptation baseline；不复用旧`@1` artifact | loss/schedule为`training-only diagnostic`；coordinates/activation为`local evaluator diagnostic` |
| P0 | `H-R1` | source-native/Belcour-inspired analytic core + bounded residual是否有同budget收益？ | Hybrid loss/core负结果、Belcour明确failure strata、历史M2死区机制可定位 | LayerStack online source/query冻结；direct/core-only配对 | `local evaluator diagnostic`，sampler补齐前不可部署 |
| P0 | `H-S1` | GGX9相对更小analytic proposals是否必要？ | NBRDF、Belcour、RTA直接proposal证据；不需改evaluator | 新的frozen evaluator checkpoint与完整event/support oracle | `full local scattering program` |
| P1 | `H-O2` | function-space监督能否稳定compiler/latent identity？ | NBRDF weight-loss负结果、NLB latent-only边界；迁移限于compiler loss | 稳定frozen evaluator与G2/G2s split | `training-only diagnostic` |
| P1 | `H-Q1` | error×optimizer-step active query allocation能否改善困难strata覆盖？ | Active Exploration有uniform与loss-only消融；迁移只改变training query policy | frozen test distribution、online reference accounting与可恢复selector state | `training-only diagnostic` |
| P1 | `H-F1` | independent levels + stochastic one-read是否优于derived/trilinear？ | NeuMIP/RTA直接机制；需要spatial source/footprint infrastructure | spatial source reference与filtered-output measure先闭合 | `local evaluator diagnostic`；sample/pdf共state后才可部署 |
| P1 | `H-D1` | view-only work移入`prepare`是否带来真实多query收益？ | 公式不变的runtime scheduling，因果隔离成本低 | 已有完整四入口program与代表性query sequence | `deployment-only` on full local program |
| P1 | `H-C1` | native parameters→program的pure compiler能否跨G2/G2s？ | MetaLayer/NMA family内证据，直接对应项目compiler目标 | typed native family、稳定decoder与pure/refined/control split | `training-only diagnostic`→`local evaluator diagnostic`；晋级需full local program |
| P2 | `H-R2` | direction/spatial planes是否值得额外random reads和bytes？ | Biplane/Comprehensive正例但measure、asset与runtime gaps较大 | spatial source与output correspondence；iso-byte/iso-cost分组 | `local evaluator diagnostic` |
| P2 | `H-S2` | direct learned sample tuple能否提供可认证MIS density？ | Importance Baking概念强但sample-map↔PDF未证明；oracle/实现成本高 | frozen evaluator、density identity与独立MIS correctness harness | `local sampling diagnostic`；认证后才是full program |
| P2 | `H-Q2` | Gaussian-product internal proposal能否降低online LayerStack reference的SE/time？ | Xia对layered estimator的pair/multiple-product正例；paper/code与measure缺口使迁移成本较高 | 冻结原random-walk expectation、solid-angle measure与独立oracle | `training-only diagnostic`（角色为`optimized-code control`） |
| P2 | `H-A1` | training-only transport auxiliary head能否保住hard components而不增runtime？ | CNSR shadow head有mixed result；local transport分量与导出能力均是待preflight的项目假设 | 同main evaluator、同reference work、可丢弃aux head，并先证明reference分解合法 | `training-only diagnostic` |
| P2 | `H-D2` | coherent/divergent路径是否需要不同kernel？ | NVIDIA系统证据强，但作者kernel不可得、当前backend不同 | 同checkpoint/package parity与项目backend实现 | `deployment-only` |
| P3 | `H-T1` | deterministic auxiliaries是否改善scene temporal stability？ | 四类scene证据互补但task/output不同；需新dataset与renderer协议 | 单独scene planning、动态trajectory与temporal reference | `separate scene renderer` |
| P3 | `H-T2` | per-light/per-band composition能否静态有界？ | LightFormer/Dual-Band直接消融；需per-light/band基础设施 | 单独scene planning、`L_max/k/ray/neighborhood`上界 | `separate scene renderer` |
| P3 | `H-T3` | ordered object transfer与order-invariant field superposition谁在matched成本下更稳？ | NeLT/Superposed DFF有直接谱系、消融与互补失败，但训练/bytes未matched | 单独scene planning、固定object partition与`O_max/field bytes` | `separate scene renderer` |
| P3 | `H-T4` | luminaire-generated field能否摊销per-frame light aggregation并保持novel-scene质量？ | NeLiF与LightFormer有matched subset/formal baseline，但field generation/update成本缺失 | 单独scene planning、统一luminaire observations与reuse/update轨迹 | `separate scene renderer` |

当前明确登记 **17 项**假设：P0 3项、P1 5项、P2 5项、P3 4项。证据入口为五份已复核综合：[representation/coordinates](../comparisons/representation-and-coordinates.md)、[optimization/loss](../comparisons/optimization-and-loss.md)、[filtering/LoD](../comparisons/filtering-and-lod.md)、[sampling/integration](../comparisons/sampling-and-integration.md)、[deployment/amortization](../comparisons/deployment-and-amortization.md)，以及各假设直接回链的个体报告；综合不替代个体报告的一手 locator。

P0/P1 优先不是预判方法质量，而是直接证据更接近当前 local program、迁移与因果隔离成本更低。Priority 也不是立即可执行顺序：例如 `H-S1` 必须等新的 frozen evaluator，`H-D1` 必须等完整四入口program。P3仍在研究范围内，但它不改变“local evaluator/compiler先闭合、scene transport后启动”的执行依赖；任何scene run、dataset、renderer protocol或基础设施扩张都需要独立planning和授权，并保留scene-level method identity。

## 3. P0：先分离优化与表示

### H-O1：Taming optimization bundle 的正交贡献

**命题。** 在同一 compact evaluator、online source/query identity与分别计账的训练成本下，Taming的target power map、stable direct+half/difference、LeakySmeLU与successive multi-instance schedule可能分别改变收敛分布；需要逐轴判断其收益是否保留到canonical bare-`f`指标与完整成本账本。

**直接证据。** Taming在与RTA同representation family的fixed-material experiments中分别报告log/power mapping、classic/stable/full coordinates、ReLU-family activation和single/multi-instance schedule；formal schedule固定每step `instances×per-instance batch=65,536` 次 sample–network evaluations，四phase为`64→16→4→1`与`1k→4k→16k→64k`，同一query batch在active instances间共享。因而理想四段各25k时，总network evaluations为6.5536B，而unique reference queries约2.176B；single `1×65,536` baseline的两者均为6.5536B。这里后两个绝对数是对论文schedule的算术展开，不是作者另报的预算。Hierarchical Neural Materials的正文与release只提供fourth-root exponent precedent；printed Eq.(2)/TeX却写`I^-4`，且root transform在其方法中与spatial gradient loss共同出现。因此linear control与fourth-root都属于项目adaptation，后者不能补成Taming公式或Xue Eq.(2)的faithful复现。Table 1内部百分比冲突、paper optimizer/LR缺失已保留，不能复用为项目成功率数值。[Taming §§7,9–12；Hierarchical §§7,11；Optimization comparison §§3,7.4]

| 合同项 | 冻结内容 |
|---|---|
| Baseline identity | 当前 `functional-f@2` 结构/config或显式budget adaptation；必须新生成同identity checkpoint，旧`functional@1` 200k artifact不得进入baseline/selection |
| Factor A：target map | Taming直接先例是log1p-like map vs cube-root power-map；linear L1为项目control，可选fourth-root只作独立adaptation identity，绝不照抄冲突的`I^-4`公式；runtime都输出同一bare linear `f`，map/offset/reduction逐式版本化 |
| Factor B：coordinates | current two-frame direct directions vs formal stable direct+half/difference；activation/loss不变，first-layer shape/MAC用预冻结padding/control匹配 |
| Factor C：activation | ReLU vs LeakySmeLU；坐标、loss、初始化与shape不变；`β/ε`按candidate identity冻结 |
| Factor D：schedule | single vs successive；固定steps与**sample–network evaluations**，shared-query语义一致；unique reference queries、reference生成时间、optimizer state与selection另报，不声称query-budget matched |
| Source/query | 每个matched block只用一个冻结family/snapshot/split；LayerStack G1与MaterialX W若都做，分别成run/CI，不跨family聚合；evaluator typed route、valid compaction、reference estimator、mip/filter/RNG一致 |
| Optimizer/init/selection | 一套预冻结optimizer与init分布；seed集合、phase cull statistic、checkpoint rule在test前冻结，不因结果追加seed |
| Metrics | 主指标与结构scorecard、per-seed convergence/AUC、best/median/worst分布、`E_net/Q_ref`、train/reference time、optimizer/prepared/runtime state bytes；“failure”若使用必须先给train/validation上的操作定义 |
| Runtime class | Factor A/D为`training-only diagnostic`；Factor B/C改变direction input/第一层或activation arithmetic，属于`local evaluator diagnostic`。两类都不自动产生path-tracing deployment claim；B/C若晋级还须重新登记MAC/latency、backend parity并补齐matched sampler |

**最小设计。** A/B/C分别做单轴matched control，任何组合都使用新method/config identity；D是独立的高成本schedule问题，只在planning另行批准相应`E_net/Q_ref/state`预算后执行。full bundle只用于机制组合验证，不是自动晋级步骤。Taming论文的100k/65,536 schedule只是correspondence候选，不构成本文件授权的预算。

**证伪。** 单轴在matched CI内无预期stratum/convergence收益；或收益只存在于training objective而canonical bare-`f`/energy/peak或成本Pareto变差；或successive schedule在分别计入`E_net/Q_ref`、selection、state与wall time后不优于single。这里的“证伪”只结束/降级该research hypothesis，不决定父任务成败，也不触发自动换seed/加预算。

### H-R1：Belcour/Hybrid analytic core + bounded residual

**命题。** 对native LayerStack，source-native simple core或Belcour-inspired bounded statistical core可能承担部分主能量/峰结构，bounded neural residual再拟合core gap；是否优于direct evaluator、以及`L_a`是否保留可解释core state，必须按roughness/grazing/TIR/tail分层而不是先假定low/moderate subset会主导总体结果。

**直接证据。** Belcour在low/moderate roughness近似较准，但Fig.19、Fig.22和supp HG sweep明确展示high roughness/tint/heavy-tail failures；Hybrid的`L_a+L_t`消融让analytic/full error都优于只用`L_t`，并报告更复杂Disney core数值不稳。迁移前P1 M2的`signed residual + clamp`在旧corpus identity上因44–98%截断产生死区；它只证明该parameterization危险，不是当前online candidate的quality baseline，候选必须使用新identity。[Belcour §§9–13; Hybrid §§7,10–13; N `model_candidates.md`/`experiment_log.md` historical boundary]

| 合同项 | Matched control / frozen axes |
|---|---|
| Evaluator groups | direct evaluator vs core-only vs core+nonnegative lobe/multiplicative-log residual；同shared host、direction frontend与canonical bare-`f` output |
| Core identity | direct-top/source-native simple core与Belcour-inspired 3-layer/2-lobe各自版本化；后者只在其plane-parallel isotropic family使用，FGD/TIR tables、TIR/refraction选择和layer/lobe cap计入identity，不当作GT |
| Loss axis | 所有neural组同full-function loss；`+L_a`只在同core/residual结构内单独比较，定义/权重在validation前冻结 |
| Source/query | 同一LayerStack snapshot、G1/G2/G2s split、online evaluator route、valid-domain/measure、reference samples、query strata与RNG；不把Belcour archive reference替代native random-walk reference |
| Budget groups | iso-MAC/state/read与natural-cost分开；analytic ops、table reads/bytes、`prepare` state、evaluator weights全部计入，不通过减少hard-strata queries配平 |
| Metrics/strata | 主指标+peak/tail/energy/reciprocity、analytic parameter fidelity；low/high roughness、grazing、TIR、layer order、HG、reflection/transmission分层；Slang/package parity、`C_prepare/C_eval`与bytes |
| Optimizer/selection | 同optimizer/init/steps/query-work/seed与checkpoint rule；core参数是否trainable、detach边界和`L_a`权重版本化 |
| Runtime class | `local evaluator diagnostic`；晋级为`full local scattering program`前必须提供与最终evaluator同state/measure的matched `sample/pdf`，缺失时fail closed |

**证伪。** residual在matched组无Pareto收益；收益只来自easy subset而预冻结hard strata显著更差；`L_a`只改善参数距离、不改善最终函数/工作流；或analytic/table/prepared-state成本计入后被direct evaluator支配。上述结果是正常empirical outcome，不是实现错误或自动改core/扩预算的触发器。

### H-S1：sampler proposal Pareto

**命题。** 对同一新生成的frozen evaluator，current-config GGX9、NBRDF-inspired 2-param analytic proxy与Belcour-inspired bounded path-lobe mixture会形成可测的variance–time Pareto；controls用于判定额外proposal容量的作用，不预设谁应胜出。

**直接证据。** NBRDF报告predicted Phong接近其fit并优于uniform，但predictor代码与完整matched数值不可得；Belcour证明完整mixture PDF修复selected-lobe fireflies；RTA给出bounded per-hit learned analytic mixture。三者proposal/evaluator均解耦，适合机制control，但项目实现分别是source adaptation，不能冒充论文artifact。[NBRDF §§8–10; Belcour §§5–6,10–12; RTA §§5,8,11]

| 合同项 | 冻结内容 |
|---|---|
| Evaluator | 同一**新生成**的`functional-f@2`或胜出local evaluator checkpoint完全frozen；旧`functional@1` 200k artifact只可另列historical observation，不进入matched组 |
| Proposal identities | cosine；NBRDF-inspired 2-param supervised/fitted proxy；Belcour-inspired bounded R/TRT/TT mixture；current-config GGX9。训练方式、lobe cap/safety component、roughness floor与parameter decode各自版本化，均用自身同源`sample/pdf` |
| Source/support | opaque reflection与reflection/transmission tracks分开；proposal不支持某event/domain时fail closed，不切generic proposal。Belcour control只用于其valid layered family，NBRDF proxy不被宣称覆盖transmission/multi-peak |
| Integrator | 同一NEE/MIS heuristic、scene/camera/light、SPP、seed、exposure、event handling与time scope；sample tuple按canonical `f·abs(cos)/pdf`，不得用rounded `wi`重建source-native tuple |
| Training/query | sampler训练的`method-sampler` route、conditioning/sample_u、target estimator、query work与seed一致；evaluator route不消费/替代sampler route |
| Correctness | support、solid-angle PDF integral、sample→external-pdf、finite tuple、forward/reverse/event/delta/null、white furnace；容差由precision/oracle在formal前校准 |
| Metrics | variance/RMSE vs spp与vs time、p99/tail weights、firefly count、bias、`C_prepare/C_sample/C_pdf`、prepared-state/asset/shared bytes |
| Runtime class | `full local scattering program` sampler track；descriptor缺任何capability时fail closed，不能把proposal-only图像实验称部署完成 |

**证伪。** 2-param proxy相对cosine无variance/time收益或在预冻结hard support失败；Belcour control在valid layered strata无收益；或GGX9在matched strata/成本上支配两者。后一结果只证伪“小proposal足够”的假设，controls仍保留为正确性与回归基线；任何低quality结果不放宽sample/pdf oracle。

## 4. P1：表示身份、filtering与compiler

### H-O2：function-space监督优于latent/weight-space距离

**命题。** compiler/encoder若只匹配optimized latent/weights，会受非唯一parameterization影响；通过frozen evaluator在query-space监督，可改善G2/G2s与workflow robustness W。

**直接证据。** NBRDF直接匹配675 weights无法恢复appearance；NLB layerer只学optimized latent而没有通过frozen evaluator回到function space；MetaLayer用function loss训练generated state但exact topology/alternating details有gap。[NBRDF §10; NLB §§7,13; MetaLayer §§7,10]

| 合同项 | 内容 |
|---|---|
| Matched controls | 同一compiler/decoder：`latent L1`、`function loss`、`latent+function`；optimized-code control固定，只由source-train/query-train得到，不从test重选 |
| Frozen axes | typed native source与G2/G2s split、target code生成/seed/canonicalization、function-query distribution、target transform、optimizer/init/steps/query work、checkpoint rule、runtime program/state bytes |
| Metrics | G2/G2s bare-`f`主指标、energy/peak/tail、latent跨seed对齐与编辑路径连续性、compile/reference-query latency、workflow robustness W与最终program成本 |
| Runtime class | `training-only diagnostic`；三组生成完全相同runtime schema，若该schema尚无matched sampler则仍不可部署 |
| Falsification | function supervision在matched CI内不改善function/generalization/W且reference成本更高；或预先canonicalized latent loss同样稳定，说明非唯一性不是当前瓶颈 |

本假设只比较supervision空间，不授权新增compiler run；若function loss需要扩大reference query预算，必须回planning并使用新config identity，不能从正式结果临时加queries。

### H-Q1：active online reference query allocation

**命题。** 在test distribution保持冻结的前提下，用当前error与下一步optimizer-update norm共同选择online reference queries，可能比uniform或loss-only replay更快覆盖rare-lobe、peak与grazing困难区；它是training query policy，不是runtime sampler。

**直接证据。** Active Exploration以`Loss·||ΔAdam||`驱动scene-configuration/patch MCMC，作者的same-time对照优于uniform；loss-only selector会卡在高误差但当前网络难以改善的mirror state，uniform+multi-res也没有稳定增益。其official code同时存在fresh batch `16–32`、small-step proposal、replay cap与resume-state correspondence gaps。因此直接证据只覆盖作者的per-scene image-patch protocol。把score改成local source/query bucket或candidate-pool influence estimator，是本项目迁移假设；论文的patch、128→600 resolution与replay常数都不是local recipe。[Active Exploration §§6–7,9–11；Optimization comparison §§4,7.7]

| 合同项 | 内容 |
|---|---|
| Matched controls | uniform fresh queries；uniform+loss-weighted replay；loss-only active；error×optimizer-step active。分成两个protocol：其一固定`Q_ref-new/Q_replay/E_net`并比较额外selector time/state，其二固定总wall time并允许各组完成不同work；不能声称同一run同时iso-work又iso-time，也不是固定相同query内容 |
| Frozen test boundary | G1/G2/G2s test source/query/strata与bootstrap单位在selector运行前冻结，永不参与proposal、score、checkpoint selection或reweight估计 |
| Selector identity | local迁移采用的source/query bucket、candidate pool、grouped/per-item influence近似、proposal state、large/local step、mutation width、acceptance、bootstrap、replay capacity/reuse bias、score refresh与checkpoint snapshot逐项版本化；不得把grouped approximation称为paper-faithful per-patch target。必须可保存/恢复RNG、chain、replay weights、EMA与optimizer状态 |
| Bias/coverage | training distribution变化单独登记；evaluation始终在原冻结分布。若训练loss需importance reweight，权重、clamp与normalization预冻结；另报source/query coverage和重复率 |
| Metrics/cost | G1/G2/G2s及peak/grazing/rare-lobe CI、seed variance、convergence AUC、coverage/ESS、`Q_ref-new/Q_replay/E_net`、candidate-score backward/optimizer simulations、proposal transitions、reference/network/selector time与history bytes |
| Runtime class | `training-only diagnostic`；不能替代`sample/pdf`，也不改变部署program |
| Falsification | held-out Pareto/coverage无改善；收益只在selector访问分布成立；reweight导致高variance；或selector/history/reference成本计入后被uniform/replay支配 |

该假设不授权根据训练中出现的困难区域临时扩充formal test strata；新的failure mode只能先登记，下一轮planning再决定是否版本化protocol。

### H-F1：filtered hierarchy 的bias–variance–reads

**命题。** 对spatial source，per-level independently learned state能比derived mip更准确；RTA式stochastic adjacent one-read与NeuMIP式two-level latent blend在bias、variance、temporal continuity和reads间存在可测tradeoff。

**直接证据。** NeuMIP与RTA分别给出independent level hierarchy及two-level latent blend/stochastic one-level-state机制；两者没有在共同source、decoder与预算下比较derived mip，因此A–D的横向排序仍是项目假设。Hierarchical Neural Materials只文字继承NeuMIP pyramid，没有重述层级配置；release每step随机选择的是training-time可访问最高level，不能补成formal runtime stochastic LoD，whole-buffer Inception也不是random-access runtime。Weier的voxel LoD属于scene/asset aggregate transport，不能作为local material hierarchy的同任务matched结果。[NeuMIP/RTA/Hierarchical/Weier reports；Filtering comparison]

| 组 | 结构 |
|---|---|
| A | finest latent→普通derived mip，nearest/trilinear |
| B | independent levels→latent trilinear，两level reads、一次decode |
| C | independent levels→stochastic one level，一level read |
| D | independent levels→two decoded outputs再linear blend，高成本control |

| 合同项 | 内容 |
|---|---|
| Matched controls | 上表A–D；每组重新matched训练，不把一个hierarchy checkpoint直接解释成另一种state语义 |
| Frozen axes | 先用output measure已闭合为bare `f`的spatial source（如当前MaterialX adapter）；source snapshot、Gaussian footprint/kernel/truncation、spatial/LOD/direction query、decoder、optimizer/init/steps/seeds与checkpoint rule一致。BTF仅在radiometric/cosine correspondence闭合后另立identity |
| Budget groups | iso-byte与iso-read/MAC分开，不同时伪装冻结；natural-cost另报。所有levels、texture formats、reads、decoder次数和prepared state计入 |
| State/randomness | evaluator diagnostic先比较per-prepare与per-evaluate RNG scheduling，并版本化stream/reuse语义；只有晋级full program时才要求同一event的`evaluate/sample/pdf`共享最终选择的state。不得把RTA未回答的项目ABI调度写成paper fact |
| Metrics | per-level/continuous-footprint主指标、energy/peak、level-boundary temporal spectrum、stochastic bias/variance/correlation、bytes/fetch/`C_prepare/C_eval`/time；sample/pdf parity只属于full-program晋级门，不伪装成evaluator diagnostic已有能力 |
| Runtime class | `local evaluator diagnostic`；只有hierarchy、state与matched sampler四入口一起通过contract后才晋级full program |
| Falsification | derived在同/更低成本不差；independent收益仅来自额外bytes；或stochastic temporal/transport variance抵消one-read收益且其它组占Pareto |

该实验只适用于reference真正接收footprint的spatial source；1×1 LayerStack只能做“not-applicable”控制，不能用来宣称plane/mip无收益。

### H-D1：`prepare`复用view-conditioned state

**命题。** latent/filter/warp/frame/sampler/analytic params只依赖material state+`wo`的部分移入`prepare`，能在同一点多`wi`/多灯/NEE下摊销，不改变数值。

**直接证据。** NeuMIP把约一半dense evaluator阶段放在view-only路径；RTA sampler params可同hit复用；NMA adapter与Belcour lobe propagation只依赖view/material。[NeuMIP §8; RTA §8; NMA §13; Belcour §13]

| 合同项 | 内容 |
|---|---|
| Matched control | 完全相同weights/formulas/precision：每个入口重复构造view-state vs `prepare(context, material)`一次缓存；`N=1,2,4,8...`只是报告轴，不预设产品hard门 |
| Frozen axes | 完整`MethodDefinition`、backend/precision、context的frame/`wo`/UV/derivatives/stochastic sample、memory layout、coherence/material mix、`wi/sample_u`序列、warmup/cache与调用次数 |
| Correctness | `evaluate/sample/pdf`使用同一prepared state；output bare `f`、solid-angle PDF、event与`f·abs(cos)/pdf` tuple一致。容差由precision与独立oracle预冻结；不得用rounded sampled direction重建native tuple |
| Metrics | `C_prepare/C_eval/C_sample/C_pdf`、N-query总时间、prepared/register/local-memory bytes、spill/cache/occupancy、coherent/divergent分层与parity |
| Runtime class | `deployment-only` on `full local scattering program`；不是新representation或quality run |
| Falsification | parity/tuple identity不成立；代表性reuse sweep无成本收益；或state spill/bandwidth使所有N上的总Pareto不优。quality低不属于该scheduling hypothesis的failure |

### H-C1：typed source compiler 的G2/G2s边界

**命题。** 在LayerStack等原生参数式family内，order-aware typed compiler可生成static program state；pure feed-forward若不足，bounded target-visible refinement可缩小与direct fit的差距。

**直接证据。** MetaLayer与NMA都在固定layer topology/family内从native params生成program，但没有证明arbitrary source；NLB layerer只在latent family内compose；当前M6定义了pure/refined/control三种role。[MetaLayer/NMA/NLB reports; N `model_candidates.md`]

| Role | 输入 | 允许claim |
|---|---|---|
| pure compiler | native typed source only | 即时编辑、G2/G2s前向泛化 |
| compiler + bounded refinement | compiler init + 新state query-train | 秒/分钟级cook，不是zero-query compiler |
| optimized/target encoder control | 可读完整target responses | best observed compression control |

| 合同项 | 内容 |
|---|---|
| Matched controls | 上表三种role；pure与refined分别报告，target-visible control不能被称为compiler。每种role使用同一decoder/runtime state上限 |
| Frozen axes | native typed family、参数范围与合法拓扑、G1/G2/G2s split、source snapshot、canonicalization、online query recipe、decoder、runtime state、optimizer/init/steps/seeds、query/refinement cap、checkpoint selection |
| Source boundary | source-native参数、图结构和资源是GT；不得先反演成LayerStack再称为其它family的GT。pure不得读target responses；refinement与control按上表显式标注target可见性 |
| Metrics | G1/G2/G2s的local quality/energy/peak与bootstrap CI、compile/edit latency、工作流稳健性W、program/state bytes、reads、`C_prepare/C_eval`与single-query backend time；晋级full program后补`C_sample/C_pdf`和tuple parity |
| Runtime class | compiler训练属于`training-only diagnostic`，生成state的local求值属于`local evaluator diagnostic`；只有以`MethodDefinition`生成静态有界且具备`prepare/evaluate/sample/pdf`的program，才可晋级`full local scattering program` |
| Falsification | pure与bounded refinement在冻结预算内均未缩小function gap；只记住train topology；或generated state/decoder被direct fit在quality—time—memory Pareto上支配。该结论不自动授权扩大family、steps或refinement cap |

## 5. P2：factorization、sample tuple与backend

### H-R2：direction/spatial planes 的random-access容量

**命题。** 对具有强spatial或half-vector相关的source，把高频轴放入plane可用额外texture reads换取更小MLP；对1×1 LayerStack则未必有效。

**直接证据。** Biplane把U/H放plane、difference进MLP；Comprehensive显式U/H/D三plane；NeuMIP与RTA把direction留给MLP。Angular Parameterization在固定7-channel neural texture、三层`8×8` MLP和两个UBO材质上报告Leather11由D9 direct+half Cartesian最低、Fabric12由D6 direct Cartesian最低；latent-angle配置多两次texture fetch，却没有跨两材质一致胜出。poster没有训练objective、measure、seed、iso-MAC或runtime，因而它只支持“coordinate clue/material dependence/read cost都须matched”的受限先例，不是plane efficacy或跨材质排序证据。[Biplane/Comprehensive/NeuMIP/RTA/Angular Parameterization reports]

**迁移假设。** U/H/D plane是否在当前已闭合spatial source上形成quality–reads–bytes Pareto，以及direct与direct+half clue在iso-capacity下是否仍有差异，均由下述项目实验裁决；不能把Angular的D6/D9原shape直接复制为formal candidate。

| 合同项 | 内容 |
|---|---|
| Matched controls | 先在generic `z8`内做direct Cartesian vs direct+half Cartesian的iso-first-layer parameter/MAC control；再比较generic `z8`、U+H plane、U+H+D plane。每个source family单独训练与统计，不把1×1 LayerStack和spatial source汇成一个CI |
| Frozen axes | 优先使用output measure/cosine correspondence已经闭合的spatial source（当前MaterialX）；source snapshot、train/val/test material split、spatial/LOD/direction query、coordinates、decoder head、optimizer/init/steps/seeds、checkpoint rule。BTF须等radiometric/cosine correspondence闭合后另立identity |
| Budget groups | iso-byte与iso-read/MAC是两组不同问题；natural-cost另报。plane resolution/channels/format/filter、MLP width/depth、fetches、prepared state和cache footprint全部入账 |
| Metrics | local quality/energy/peak与spatial-frequency分层、G1/G2、bytes/fetch/MAC、cache/occupancy、`C_prepare/C_eval`与backend time；若进入full program，再报告sample/pdf parity |
| Runtime class | `local evaluator diagnostic`；没有matched `sample/pdf`时不得作为可部署scattering program |
| Falsification | plane在相关source上无Pareto收益或cache/fetch使其被generic control支配；若1×1与spatial收益模式相同，只是否定“spatial相关轴是收益来源”的机制解释，不自动证明任一表示普遍更优 |

### H-S2：direct sample tuple 的可认证density

**命题。** Importance Baking式network直接输出`wi`与`f cos/q`可能减少hot evaluator调用；但项目的path-tracing ABI要求同一proposal同时提供可查询的solid-angle `pdf(wi)`。只输出sample tuple的形态只能做隔离诊断，不能注册为`ScatteringPackage`或进入MIS路径。

**直接证据。** Importance Baking分别训练sample map、cosine-weighted evaluator与独立PDF网络，但没有证明sample map诱导density与PDF网络归一匹配；其预烘weight也不能替代项目对同一proposal的external `pdf(wi)`查询。Neural Processes的NICE提供flow/change-of-variables先例，却没有公开sampler code/weights、完整solid-angle Jacobian或MIS审计。因此“independent PDF”与“可求Jacobian的flow”都是项目matched controls，不是已有论文已经通过本项目四入口合同的实现。[Importance Baking/Neural Processes reports；Sampling comparison §§4,6.4,8–11]

| 合同项 | 内容 |
|---|---|
| Matched controls | 固定同一frozen evaluator/target：current GGX9 analytic proposal；direct sample map+independent PDF net；具有可求Jacobian/flow density的sample map。另列sample-tuple-only为`local sampling diagnostic`中的fail-closed tuple ablation，不得把禁用MIS当作产品路径 |
| Frozen axes | source/package identity、prepared state、support/event、hemisphere/solid-angle measure、conditioning、random input、training query/optimizer/init/steps/seeds、parameter/MAC budget、integrator与sample count。sample和pdf必须来自同一proposal identity |
| Correctness | `sample()`返回source-native event、`wi`、PDF和`f·abs(cos)/pdf` tuple；`pdf()`复算同一proposal的solid-angle density。不得从rounded `wi`重建source-native tuple；unsupported event/capability fail closed |
| Metrics | PDF normalization/support、sample→pdf parity、tuple expectation、MIS bias、oracle integral、variance/time、invalid/tail、`C_sample/C_pdf/C_eval`、parameters/bytes/MAC与integrator调用数 |
| Runtime class | sample/PDF研究阶段属于`local sampling diagnostic`；前三者中通过完整四入口ABI者才是`full local scattering program`。sample-tuple-only不能注册`MethodDefinition` |
| Falsification | independent PDF在完整oracle与MIS检查内保持无偏且成本最低，则否定“必须显式同源density parameterization”的工程假设；若direct map在冻结预算下不优于analytic sampler，则该direct-map假设失败。任何bias都不是通过禁用合同所需路径来调和 |

### H-Q2：Gaussian-product optimized reference control

**命题。** 对当前plane-parallel isotropic LayerStack source，在不改变原random-walk reference期望的前提下，用Xia式pair/multiple Gaussian-product proposal采内部layer-path directions，可能降低online GT的SE/time；它是reference estimator control，不是目标neural representation，也不是external material `sample/pdf`。

**直接证据。** Xia为isotropic Beckmann/GGX slices拟合一/二分量bivariate Gaussian与低阶polynomial，runtime解析乘相邻proposal并以exact path contribution/proposal density和MIS保持估计；作者同时保留外部Guo-style stack `sample/pdf`为另一层接口。正式code/config不可得，GGX mixture weight、projected-vs-solid-angle density和部分MIS配置未闭合，因而项目实现只能叫source adaptation。[Xia §§4–7,11–15；Sampling comparison L/R/S/G taxonomy]

| 合同项 | 内容 |
|---|---|
| Estimator groups | current position-free random walk；pair-product proposal；multiple-product proposal；可选高成本exact/long-run oracle。所有组估计同一canonical bare-`f` expectation |
| Source validity | 只在Xia假设覆盖的thin/local-flat、isotropic surface-scattering LayerStack子域使用；anisotropy、BSSRDF横向位移或未支持event fail closed，不用近似proposal覆盖率冒充GT范围 |
| Frozen axes | LayerStack source snapshot/subdomain与G1/G2 split、outer `(wo,wi)` query/RNG、canonical bare-`f` measure、path termination/depth/event规则、native BSDF factors、reference precision与independent oracle在run前冻结；不得因某proposal失败临时缩窄source或query strata |
| Measure/correctness | slope→hemisphere Jacobian、projected/solid-angle density、forward/reverse/event、support、mixture normalization、MIS独立随机流与finite weights由独立oracle预冻结；不得把internal proposal PDF当external conditional `pdf(wi given wo)` |
| Fitting identity | `(η,α,θ)` grid、case split、polynomial degree、low-roughness anchor、GGX mixture weight、precision repair与coefficient bytes版本化；缺失paper配置由本项目另立adaptation identity |
| Matched work | 同target queries/source states/seeds；比较fixed path samples、fixed wall time与fixed reference SE三种口径，分别记录proposal fit/precompute、per-query reference time、path/BSDF eval数与state bytes |
| Metrics | 与long-run oracle的bias/CI coverage、per-query SE/tail/fireflies、throughput、training downstream gradient variance与最终G1/G2；proposal precompute摊销单列 |
| Runtime class | `training-only diagnostic`；角色记为`optimized-code control`。不会替换最终`evaluate/sample/pdf`程序，也不把“更低variance reference”称为neural method收益 |
| Falsification | correctness oracle失败；同时间SE不降；fit/precompute摊销后被current reference支配；或收益只在过窄strata且不改善downstream Pareto |

### H-A1：training-only transport auxiliary head

**命题。** 从同一online reference额外监督一个可物理解读的transport component，可能迫使compact latent保留主loss容易忽略的hard component；auxiliary head在deployment前删除，所以runtime shape、reads与ABI不变。

**直接证据。** CNSR让shared encoder另预测partial shadow visibility；作者观察geometry容量可改善，但material表现会变差，aggregate metrics基本不变。这个mixed result只说明scene-image encoder的auxiliary steering会重新分配有限latent容量，而不是免费增益。它不证明当前LayerStack reference已经暴露reflection/transmission、interface/volume或scattering-order标签，也不证明其中任一分解适合local bare-`f`训练。[CNSR §§5.4,9–10,13–15]

**迁移假设。** 先以read-only preflight证明某一个component能从同一reference path无歧义导出、与canonical bare `f`在measure上闭合，并明确是否增加path work；若做不到，本假设保持未就绪，不能为它自动修改reference ABI或扩张预算。component选择和auxiliary head均为项目新identity，不能借CNSR的shadow标签命名为论文复现。

| 合同项 | 内容 |
|---|---|
| Matched controls | preflight通过后比较main-only与main+一个预冻结aux component；同总reference paths/query work、shared evaluator/latent、optimizer/steps/seeds。若component需要额外reference work，再分iso-reference与natural-cost两组；aux head参数与training FLOP单独记账，runtime全部移除 |
| Frozen axes | source snapshot与G1/G2/G2s split、canonical reference/query/RNG、component attribution、main evaluator/latent/runtime schema、optimizer/init/steps/seeds、checkpoint rule、main/aux transform与预算在test前冻结；不得按test结果换component或追加head |
| Component identity | 只选preflight证明reference可无歧义返回的一个component；分解必须求和/measure对应canonical bare `f`，不根据test结果更换标签。source event、path-order attribution与invalid/termination规则逐项版本化 |
| Loss/gradient | aux target transform、weight、detach/gradient route、phase与head shape预冻结；主loss不变。另做same-parameter dummy head或减shared width control，防止把额外容量当机制收益 |
| Metrics | main G1/G2/G2s与hard-component CI、energy/peak/tail、latent capacity/gradient conflict、seed variance、train/reference time；runtime parity/bytes必须与main-only完全一致 |
| Runtime class | `training-only diagnostic`；不增加runtime output或修改`evaluate/sample/pdf`合同 |
| Falsification | aggregate或hard component无Pareto收益；另一个预冻结stratum显著恶化；收益由额外shared容量解释；或component/reference成本不可接受 |

### H-D2：coherent/divergent backend specialization

**命题。** 同一个compact MLP在coherent same-material packet与divergent path hit上的最佳kernel不同；动态选择可能优于单一路径。

**直接证据。** RTA commercial path分别使用tensor-core blocks与packed-FMA，并用SER/warp coherence调度；当前package只有regular FP16 functional path，不能沿用论文性能结论。[RTA §8; N correspondence]

正式论文所述tensor-core blocks与packed-FMA的商业实现未公开；因此下述实验只能检验项目自写specialization，不能声称paper-faithful性能复现。

| 合同项 | 内容 |
|---|---|
| Matched controls | regular FP16、项目自写coherent block、项目自写divergent packed implementation；selector开/关。所有路径读取同一新checkpoint/MethodBundle、precision policy、prepared state并输出相同bare `f`/sample tuple |
| Frozen axes | dense same-material、small material set、fully divergent workload的ray/hit/material顺序与规模；backend/compiler flags、warmup、cache、batch/packet、SER/coherence policy、hardware/clock、调用次数和完整program ABI |
| Correctness | 对独立oracle冻结数值/ULP或相对误差容差；`evaluate/sample/pdf`共享proposal/state且tuple一致。不得拿旧`functional@1` artifact或商业kernel数字替代当前`functional-f@2` package parity |
| Metrics | 每入口`C_prepare/C_eval/C_sample/C_pdf`、packet吞吐与single-query latency、selector overhead、occupancy/register/spill/cache、bytes与parity；coherent/divergent分别报告，不混成单一均值 |
| Runtime class | `deployment-only` on a `full local scattering program`；实现必须留在同一`MethodDefinition`静态有界路径，不得新增candidate-specific runner/exporter/viewer |
| Falsification | regular kernel在全部冻结workload不劣；selector开销抵消分路径收益；或specialized路径不能保持package parity。失败不推翻RTA论文的未公开商业实现，只否定当前项目实现 |

## 6. P3：scene transport 第二波

### H-T1：deterministic physical auxiliaries 与 temporal stability

**命题。** 在scene/volume renderer中，以variance-free physical auxiliaries替代noisy low-spp radiance，可降低flicker/temporal instability；但可能牺牲单帧quality，history仍是独立轴。

**直接证据。** 1469的aux-only Att单帧指标略差于DenX但`t-RMSE`更好；Light Probes用history+temporal loss仍有small-highlight flicker；Dual-Band无history且只有定性temporal观察；LightFormer公开network无history输入，但VPL/RSM temporal policy未报告。NeLT固定insertion order只为避免approximation order变化造成不一致，Superposed DFF和NeLiF也都没有formal temporal metric/history/update protocol，不能把结构上的order-invariance或generated field缓存升级成temporal稳定证据。CNSR只提供64²训练后query-resolution外推及定性artifact，Active Exploration的128→600是training curriculum；它们同样只能作为边界。[各scene报告]

这些论文的scene、输入构造与训练身份不同，因此只能提供机制动机，不能直接合并成matched quality排名；实验须在新的scene-track planning中建立共同身份。

| 合同项 | 内容 |
|---|---|
| Matched controls | noisy radiance only、deterministic auxiliaries only、both，各自做history off/on；iso-parameter/compute与natural-cost分开。输出任务固定为scene radiance，不与local `evaluate(wo,wi)`混同 |
| Frozen axes | scene/train/val/test split、camera/light/object/volume trajectories、reference renderer/spp与seed、resolution、color/exposure/tone mapping、current-frame ray/spp与auxiliary生成、history/reprojection规则、capacity/optimizer/steps/seeds、backend和buffer precision |
| Metrics | per-frame quality与bootstrap CI、warped temporal error、flicker spectrum、disocclusion/occlusion/highlight稳定性、ghosting、condition/model/total time、rays/spp与persistent/transient buffers |
| Runtime class | `separate scene renderer`；不继承或宣称local scattering ABI，也不改变当前evaluator/compiler研究顺序 |
| Falsification | auxiliaries-only在matched quality/time下不改善temporal CI；或history稳定地支配current-frame方案且无ghosting/成本代价。结论只适用于冻结scene protocol，不是产品hard gate |

1469仍有匿名实现与部分loss/config缺口；Neural Light Probes报告temporal failure；Dual-Band和LightFormer没有正式temporal metric。缺口须作为identity字段保留，不得用相邻论文配置补齐。

### H-T2：per-light/per-band bounded composition

**命题。** LightFormer的per-light encode→attention→single decode与Dual-Band的principal/mirror-secondary→feature-conditioned filter代表两种scene factorization；加bounded light culling/top-k与fixed secondary-ray budget后，可形成静态有界scene image program。

**direct evidence.** LightFormer attention优于capacity-near average pooling，full 840D key约3×计算，light-count线性是known limitation；Dual-Band的MLP替CNN、去self-tuned features、去zero branch均较差，single-bounce导致long specular path失败。[LightFormer/Dual-Band §§9–10]

训练correspondence也必须保留原边界：LightFormer是per-scene 20,000 random train configs/100 test configs、joint training、Adam `1e-4`/batch 4，但steps/schedule/seeds、逐项log/VGG与attention preprocessing未报告；Dual-Band是8,192 configs上的two-stage lifecycle（stage 1使用4,096个roughness `>=0.25`配置，stage 2启用secondary+fusion）、两阶段optimizer/batch已报告，supplemental锁定final head为LeakyReLU，但stage length/split/`γ` merge、mapped-output lifecycle、dynamic-light与temporal protocol未闭合。它们是各自scene identity，不是可以互相补齐或直接复制到本实验的共同recipe。[Optimization comparison §§2–6,8]

**最小实验.**

- light axis：full-light attention vs fixed`k` culling+attention vs average pooling；
- band axis：principal only、principal+mirror secondary concat、feature-conditioned fixed-neighborhood filter、full-screen CNN high-capacity control；
- 不把二者一次组合，先各自隔离；胜出后再做bounded composition。

| 合同项 | 内容 |
|---|---|
| Matched controls | light axis与band axis先按上列分别实验；胜出后才允许规划composition。每轴同时有iso-capacity/compute与natural-cost，equal-time PT/denoiser只作scene renderer control |
| Frozen axes | scene/light/object/material split、GT component定义与merge/color流程、light/VPL/RSM生成和更新、secondary-ray count、neighborhood、resolution、capacity/optimizer/steps/seeds、hardware/backend、trajectory与buffer precision |
| Static bounds | light axis预冻结`L_max`、top-`k`与overflow policy；band axis预冻结每pixel secondary rays、filter neighborhood与feature/buffer大小。不得在测试时随质量动态扩张 |
| Metrics | 按light count/roughness/path length分层的component/final error与bootstrap CI、missed-light energy、mirror/highlight/long-specular-path、temporal popping、rays/VPL/RSM更新成本、stage/total ms与persistent/transient bytes |
| Runtime class | `separate scene renderer`；不是local material `MethodDefinition`，也不得用其final-image收益替代local evaluator/sampler验收 |
| Falsification | fixed `k`随light数增长质量不可控；attention无matched收益；feature-conditioned filtering只在train scenes有效；或bounded program在冻结成本下被equal-time control支配。结果不是自动产品hard gate，也不授权把两个轴或dynamic-light/temporal缺口合并猜实现 |

### H-T3：object-oriented composition 与 superposed fields

**命题。** 在固定dynamic-object partition与总预算下，把scene transport拆成对象级状态可能改善未见transform/material组合；但NeLT的ordered ratio/residual composition与Superposed DFF的pair-latent/field两层求和具有不同表达与成本，必须分别与monolithic control比较，不能先验宣布order-invariant sum更优。

**直接证据。** NeLT把插入对象对背景direct写成乘法ratio、对indirect写成加法residual，并由object-specific hypernetwork/neural texture逐次复合；object数增加令frame time线性增长，composite transfer更快但牺牲对象级灵活性，novel-scene与specular/high-frequency仍失败。其formal fit是每个独立NeLT transfer unit约60 h/2×A6000；Table 4是RTX3090/PyTorch、排除G-buffer的full-image timing（256² `26.66–60.16 ms`、1024² `300.62–656.53 ms`），representation build/state bytes未报告。Superposed DFF用`r_i=Σ_j r_ji`与`Σ_i F_i`两层求和消除人工insertion order；feature field、C2F与static-vs-deformable消融支持对应模块，但没有分别消融pair-latent sum与field sum。其formal fit为15 h/scene on 4×A6000；RTX4090/PyTorch FP32的512² `18.9/19.0/26.6 ms`没有拆G-buffer、pair encoder、hypernetwork、fields与final decoder。两组training work、hardware、partition与timing scope都不matched；两篇对AE也都只使用uniform训练，不能把active allocation差异混入representation结论。[NeLT §§5,7–10,13；Superposed DFF §§5,7–10,13；Deployment §6]

| 合同项 | 内容 |
|---|---|
| Matched controls | A：monolithic scene latent/decoder；B：NeLT式ordered ratio/residual object transfer；C：完整pair-latent sum + order-invariant field sum。另在C内部做`C-pair`（保留field sum、替换`r_i=Σ_j r_ji`）与`C-field`（保留pair sum、替换`Σ_i F_i`）两项iso-budget消融，避免把两层sum合成一个原因。先做iso-parameter/bytes/compute，再报告各方法natural cost；AE selector统一关闭，另由H-Q1处理 |
| Frozen axes | scene/object/light/material/camera split、object partition与NeLT insertion order、Superposed DFF self-term policy、G-buffer、GT/component定义、NeLT ratio near-zero policy、training states/views/SPP、`Q_ref/E_net/GPU-hours`、loss/optimizer/steps/seeds、decoder/field precision、backend、resolution与trajectory；main未报告的self term/ratio policy必须作为项目adaptation identity预冻结 |
| Static bounds | 冻结`O_max`、是否含self pair及每frame最大pair encodes、field count、triplane levels/resolutions/channels、generated-weight/state bytes、每pixel field queries、composition passes和overflow/fallback；不得测试时扩object/field/state budget |
| Metrics | held-out object transform/material/light组合的image/component error与bootstrap CI、NeLT order permutation与fixed-order error、C的seed/partition稳定性、rare caustic/shadow/highlight strata、offline/fit work、per-object/state build、frame latency、persistent/transient bytes和`O`-scaling curve |
| Runtime class | `separate scene renderer`；object fields/ratio/residual均不得进入local material `evaluate/sample/pdf`，也不得把最终图像收益作为local fidelity证据 |
| Falsification | B/C在matched成本下不改善held-out组合，收益只来自额外bytes/training；C的代数order-invariance没有转化为seed/partition/held-out稳定性收益；pair或field消融表明另一层sum并非必要；或冻结`O_max`下质量/时间不可控，则拒绝对应factorization假设。代数上的element-wise sum可交换性本身不是待实验“证明”的结果 |

### H-T4：generated lighting field 与 per-frame light aggregation

**命题。** 对跨pixels/frames复用的luminaire，可把LightFormer式per-frame/per-pixel light aggregation前移到per-luminaire generated field；只有在field生成、更新、bytes和staleness全计入后，才可能形成更好的amortized scene-rendering Pareto。

**直接证据。** NeLiF用4D luminaire observations经交替spatial/angular attention与cross-attention生成spherical triplane，运行时再用G-buffer、RSM/VPL与kernel shadow解码；它复用LightFormer indirect module，并在共同400K subset、35 epochs的正式比较中主张novel-scene/light质量。完整corpus另为5,300 luminaires、10,000 scenes、1,000,000 training images；训练硬件为12×RTX4090D，但wall/GPU-hours、batch/steps及full-1M-vs-400K checkpoint identity未报告。Table 1的512² TensorRT half `10.56 ms`也没有说明是否包含field/3DGS generation、G-buffer/RSM/shadow passes、全部decoders、3DGS raster或transfer/sync。作者的关键动机正是避免LightFormer每帧重复聚合light effects，但field/3DGS generation time、bytes、update cadence与multi-luminaire composition均缺失，且monolithic neural radiance仍有作者声明的high-frequency spectral limitation。[NeLiF §§4–12；LightFormer §§4–12；Deployment §6]

| 合同项 | 内容 |
|---|---|
| Matched controls | A：per-frame per-light observation+pixel attention；B：相同observations生成per-luminaire field再query；C：同budget低频analytic/probe control。先单luminaire，再在固定`L_max`下测试bounded multi-light composition；full-1M corpus与matched-400K protocol不得混为同一checkpoint/control |
| Frozen axes | luminaire/scene train-val-test split、full/subset corpus identity、observation cameras/radiance/depth、G-buffer/RSM/VPL/shadow passes、decoder capacity、loss/optimizer/steps/seeds、training hardware/work、field coordinate/resolution/channels/precision、backend、camera/light update trajectories、reuse horizon与exposure |
| Static bounds | 冻结`L_max`、每luminaire observations/tokens、field/3DGS bytes、generation calls、per-pixel field reads、shadow levels/neighborhood、refresh/staleness policy与overflow fallback |
| Metrics | novel-scene/novel-light quality与bootstrap CI、near-field/high-frequency/shadow strata、field generation与update latency、steady frame time、amortized total over reuse horizon、persistent/transient bytes、light-edit latency、stale/temporal error |
| Runtime class | `separate scene renderer`；generated lighting field不是material latent，HDR scale迁移到local evaluator时必须另立training-only或bounded-state candidate |
| Falsification | 在冻结reuse/update轨迹下B无amortized成本收益、field memory超限、动态更新stale error不可控、novel-light/near-field质量不优，或收益只来自更大data/capacity/training work，则证伪；不得通过延长reuse、换full-corpus checkpoint或省略generation/observation stages挽救结果 |

## 7. 统一实验产物与停止规则

本文件只给出待planning冻结的假设合同，不授权启动formal run、生成新checkpoint、扩张scene infrastructure或修改backend。经`experiment_framework.md`完成source/query/protocol/budget freeze并取得任务授权后，每项formal run必须留下：

- method/correspondence/source-adaptation/recipe identity；
- source snapshots、online route/RNG、training work units、checkpoint selection；
- paper mechanism correspondence与项目adaptation清单；
- hard sanity、主指标、structure scorecard、bootstrap CI；
- `C_prepare/C_eval/C_sample/C_pdf`、shared/asset/state/system bytes与真实backend timing；
- 假设所属runtime class、验收合同来源，以及缺失capability的fail-closed记录；
- normal empirical outcome，包括quality较差、无显著差异或成本被支配。

Taming-style实验必须分别登记`E_net`（sample–network evaluations）与`Q_ref`（unique reference queries）；共享query batch不能写成二者相同。任何full local candidate必须由同一`MethodDefinition`提供静态有界的`prepare/evaluate/sample/pdf`，缺入口就fail closed；不得用candidate-specific runner/exporter/viewer绕过。旧`nvidia-rta2024-functional@1`归档只可支持旧identity观察，不能充当`functional-f@2`的checkpoint、baseline或quality证据。

pilot/diagnostic不得混入formal CI或候选排名。达到冻结budget后不自动扩大steps、seeds、data或model直到“过门”；本文件的证伪条件是经验结论分类，不是父任务或产品的hard gate。若失败属于implementation defect，修复后以新run identity重跑；protocol/design/resource defect返回planning；忠实且稳定的低quality直接登记为结论。

## 8. 当前建议的最短研究链

以下只是依赖顺序建议，不是执行授权。在不启动scene基础设施扩张的前提下，最短因果链是：

```text
静态确认 functional-f@2 code/config identity
  → planning冻结并新产出 @2 checkpoint（不得复用旧 @1 artifact）
  → H-O1 分离 loss / coordinates / activation / schedule
  → 选稳定 compact evaluator
  → H-Q1 active query allocation 与 H-A1 auxiliary head 分别做training-only诊断
  → H-R1 direct vs analytic-core residual
  → H-S1 在 frozen evaluator上比较 sampler controls
  → H-D1 部署 prepare/evaluate 复用
  → H-C1 扩到 G2/G2s compiler
  → spatial source 时做 H-F1/H-R2
  → evaluator/compiler稳定后再启动 H-T1–H-T4 scene track
```

H-O1的Taming schedule、65,536 batch与更长训练预算均须另行planning；它们不是由此表直接授权。这个顺序不排除scene transport；它只防止scene-level final-image improvement反向掩盖local evaluator、sampler或compiler尚未闭合的错误。

`H-Q2`不在最短链的必经路径：它只有在当前random-walk reference的SE/time被preflight证明为主导瓶颈、且Gaussian-product adaptation通过独立measure/correctness oracle后，才作为optimized-code control插入checkpoint生产之前。`H-Q1/H-A1`也必须分别立项，不组成自动联合bundle。

## Evidence review

```text
author: /root
reviewer: /root/nelif_full_report
reviewed_at: 2026-08-29
sources_rechecked:
  - research/comparisons/representation-and-coordinates.md
  - research/comparisons/optimization-and-loss.md
  - research/comparisons/filtering-and-lod.md
  - research/comparisons/sampling-and-integration.md
  - research/comparisons/deployment-and-amortization.md
  - load-bearing individual-report boundaries for Taming, Hierarchical, Angular Parameterization, CNSR, Active Exploration and Xia as linked by those syntheses
  - evidence-reviewed NeLT, Superposed DFF and NeLiF individual reports for H-T3/H-T4
  - docs/research/experiment_framework.md
  - .trellis/spec/project/method-constraints.md
  - docs/contracts/scattering_backend.md
  - research/implications/current-nvidia-correspondence.md
  - docs/research/model_candidates.md and experiment_log.md
remaining_evidence_gaps:
  - hypotheses remain planning inputs, not frozen formal configs or run authorization
  - functional-f@2 has current code/config identity but no new formal checkpoint/package from this literature task; old functional@1 artifact is historical evidence only
  - scene papers retain source/code/config, temporal, VPL/update and lifecycle gaps; no neighboring paper was used to fill them
  - priorities encode evidence strength, isolation value, dependency and migration cost, not an empirical quality ranking
  - H-A1尚未证明当前reference可无歧义导出同path、同measure的transport component；在preflight闭合前保持未就绪
  - H-T3仍缺NeLT/Superposed DFF的supplemental/code、exact field/state bytes与matched训练work；H-T4仍缺NeLiF field/3DGS generation、update、多灯与runtime breakdown
findings_closed:
  - confirmed the priority table contains exactly 17 unique hypotheses: P0=3, P1=5, P2=5 and P3=4
  - split H-T3 into ordered transfer, pair-latent aggregation and field-output aggregation controls; froze ratio/self-term adaptations, O_max, training work, state bytes and build/frame costs
  - recorded that algebraic field-sum commutativity is a fact, while any held-out/seed/partition stability benefit remains falsifiable
  - expanded H-T4 to preserve full-1M-vs-400K identity, 35-epoch/12x4090D training gaps and the unknown 10.56-ms included scope
  - confirmed H-T1-H-T4 remain separate-scene planning inputs after local evaluator/compiler closure and authorize neither scene infrastructure nor formal runs
  - confirmed all falsification conditions classify empirical outcomes only and do not authorize seed, data, model or budget expansion
review_status: evidence-reviewed
```
