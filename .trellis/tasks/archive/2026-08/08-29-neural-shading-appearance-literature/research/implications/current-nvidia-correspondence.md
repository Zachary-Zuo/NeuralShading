# 当前 NVIDIA Functional Reproduction：文献 Correspondence 与改进边界

## 1. 当前方法身份与证据边界

当前产品registry只发现`nvidia-neural-appearance`。稳定文档把它定义为`nvidia-rta2024-functional-f@2`：native-parameter encoder、hierarchical z8、两个learned frames、`20→64→64→64→3` evaluator、`11→32→32→32→9` analytic sampler，以及encoder bootstrap→materialize→latent finetune lifecycle属于同一个`MethodDefinition`。训练只经公共reference backend在GPU上online取得bare linear`target_f`，runtime evaluator也直接输出bare linear`f`，不乘/除cosine。[N `docs/learning.md`; `configs/learning/nvidia-rta2024-materialx-formal.json`，SHA-256 `53503C6E8A36D25B9EC09FB2128F6E538E6C70CCB2239DDA847D61D59F9E3CAF`]

本文件只做method correspondence和研究决策，不宣称执行了新的formal training或shader验证。当前有三层不能混用的证据：

| 层级 | identity | 可证明的内容 | 不可证明的内容 |
|---|---|---|---|
| 2024论文`P/S` | Real-Time Neural Appearance Models | 公开结构、hierarchy/frame/evaluator/sampler与300k lifecycle | exact log transform、KL estimator、stage switch、seed/stream、正式资产与output measure冲突的唯一裁决 |
| 当前源码/配置`N` | `nvidia-rta2024-functional-f@2` | 项目选择的bare-f ABI、formal recipe、typed routes、source adaptation与实现合同 | 未运行的300k结果、作者五材质图像复刻、未公开tensor-core/packed-FMA性能parity |
| 归档200k artifact`N-old` | `nvidia-rta2024-functional@1`、旧200k冻结结果 | 旧checkpoint/package在其旧correspondence下的方向/能量/viewer证据 | 当前`functional-f@2`质量、300k正式完成或新identity的回归结论 |

归档correspondence明确写旧200k结果不等于300k formal；该提前冻结是旧`functional@1`相对论文300k lifecycle的`budget-adaptation`，不是当前`functional-f@2`的已执行训练预算。当前稳定文档又明确取消runtime cosine adapter。因此任何研究报告不得把旧artifact的数值、checkpoint或package写成当前candidate的observed quality，也不得把它们用于当前identity的checkpoint selection。[N archived `08-27-faithful-nvidia-neural-materials/research/correspondence.md`; N `docs/learning.md`] 本文件就是研究综合层面的显式`functional@1→functional-f@2`迁移说明：它补齐此前个体报告所指出的文档identity gap，但不产生当前identity的checkpoint、package或formal结果。

本文统一使用六类correspondence标签：`faithful`表示公开定义已对应；`author-underspecified`表示一手材料缺值或彼此冲突；`source-domain-adaptation`表示换到项目的原生source family；`interface-adaptation`表示适配公共`evaluate()`语义；`backend-adaptation`表示更换执行后端；`budget-adaptation`表示改变训练或部署预算。同一行可在不同轴上同时拥有多个标签。`suspected-defect`只留给已经定位到实现违约的情形；个体报告曾把“缺显式`@1→@2`迁移说明”登记为文档/identity疑点，本文件已在综合层面闭合它，且该项从来不等于evaluator算术bug。本次回查没有建立新的实现defect结论。

## 2. 2024论文逐项 correspondence

| 论文一手定义 | 当前实现/配置身份 | 分类 | 结论与剩余边界 |
|---|---|---|---|
| native source parameters→`K→64×4→8` encoder | `native_feature_count=38` MaterialX adapter；统一reference session | topology `faithful`；`source-domain-adaptation` | encoder形态对应；MaterialX snapshot/38 slots不是作者五个私有材质的隐藏配置 |
| z8 hierarchy，两张RGBA textures；footprint LoD、adjacent stochastic level、level内bilinear | `4096²`、13 mips；`discrete-mip-bilinear-wrap@1`；两张RGBA16F package | `faithful` mechanism + `author-underspecified` recipe values | one-level随机访问与z8对应；4096/13、wrap、scale1、cap64是项目冻结选择 |
| coarse encoder input用LEAN，Gaussian footprint，sample数随area | `lean-first-second-moments@1`、`gaussian-area-proportional-cap-64@1` | `faithful` intent + `author-underspecified` exact filter | 论文没有sigma/截断/字段/分布参数；不得把当前值回填为作者配置 |
| z8→两个learned frames；n/t canonical residual+normalize，bitangent cross | 当前Torch/Slang frame path | topology/projection `faithful`；bitangent rule `author-underspecified` | 正文§9写normalized cross，supplemental/code不normalize；当前采用supplemental/later code的unnormalized cross，但这只是对一手冲突的版本化选择 |
| 两frame下fixed/query directions共12D + z8 =20D | current evaluator input | `faithful` to 2024 | Taming 2026 stable half/difference不是2024结构，不能无版本变化替换 |
| evaluator`20→64→64→64→3`、ReLU、`exp(raw-3)` | current max evaluator、bare-f target/output | topology/activation `faithful`；output measure `interface-adaptation` | P按BRDF叙述、S listing返回`f cos`；项目选择直接重训bare-f，不能称已裁决作者内部冲突，也不能复用旧`functional@1`的cosine-division checkpoint/package |
| sampler`11→32→32→32→9`，tilted cosine + non-centered anisotropic GGX | current learned GGX9 proposal | `faithful` topology/family | 仍需以当前package identity维护normalization、sample→pdf、reverse/event tests |
| supplemental同一`float2 u`，`u.x`选component后remap | current formal sampler contract | `faithful` | later2026 code改成float3不能覆盖2024identity；当前两随机数选择正确 |
| evaluator/sampler各65k，300k，online，simultaneous；sampler latent detach | two typed routes、`total_steps=300000`、batch65k/65k | `faithful` public lifecycle | 两route独立`seed_offset=0/1`是项目冻结选择，论文没有声明stream关系 |
| Adam`.9/.999/1e-7`、cosine`1e-3→1e-4` | formal config exact | `faithful` | — |
| 前20k、10°→0°、256 samples的outgoing mollification | formal config exact | `faithful` | reference logical estimator必须与dispatch tiling分开计数 |
| encoder sufficiently trained后materialize/fine-tune | `materialization_step=100000` | structure `faithful`；boundary `author-underspecified` | 100k取later-author default，不是2024隐藏事实 |
| BRDF log-space L1 | `log1p-l1@1` on bare f | `author-underspecified` | exact offset/reduction未披露；这是版本化项目选择 |
| sampler KL to current learned BRDF | `learned-sampler-forward-kl-score@1` | target/detach `faithful`；estimator `author-underspecified` | KL方向/normalization/gradient estimator非2024明文；当前选择受later-author code启发 |
| FP16 runtime与custom shader paths | current `ScatteringPackage@1` regular FP16 network contract | functional precision path `faithful`；`backend-adaptation` | 不声称复现作者未公开D3D12 tensor-core intrinsics、packed-FMA或SER性能；当前`functional-f@2`尚无formal package证据 |

主证据：[RTA个体报告](../papers/zeltner-2024-real-time-neural-appearance-models.md)、[当前学习文档](../../../../../docs/learning.md)、[formal config](../../../../../configs/learning/nvidia-rta2024-materialx-formal.json)、[归档correspondence](../../../archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md)。

跨论文机制与成本边界另回链五份已复核综合：[representation/coordinates](../comparisons/representation-and-coordinates.md)、[optimization/loss](../comparisons/optimization-and-loss.md)、[filtering/LoD](../comparisons/filtering-and-lod.md)、[sampling/integration](../comparisons/sampling-and-integration.md)、[deployment/amortization](../comparisons/deployment-and-amortization.md)。这些综合只用于分类迁移距离和cost domain，不覆盖个体报告的一手事实。

## 3. 当前已经明确不是 defect 的差异

### 3.1 Source family adaptation

作者五个layered assets、graphs、configs与checkpoints未公开。当前MaterialX `american_walnut_veneer`和LayerStack 1×1通过统一reference backend生成GT，属于可追溯的source adaptation，不是论文资产复刻。只要报告明确source/query identity，就不应把“没有作者资产”写成实现错误；同样也不能用当前source结果证明作者图像parity。

### 3.2 Bare-`f` ABI

项目runtime contract要求`evaluate()`返回不含cosine的linear`f`，训练reference也直接给`target_f`。这是公共接口选择；它解决了旧runtime适配的重复乘除风险，但没有消除P/S output measure冲突。因此分类是`interface-adaptation`，不是`faithful`或`suspected-defect`。

### 3.3 Regular FP16 backend

当前实现具备把同一checkpoint导出为FP16 weights/latent并执行Torch↔Slang/package oracle的路径；这描述的是实现合同和未来验收步骤，不是`functional-f@2`已经产出的package证据。它没有复现论文commercial renderer的custom tensor-core intrinsic。功能parity和性能复刻是两件事。缺少作者shader artifact时，regular FP16是合法`backend-adaptation`，但论文4090 frame timing不能被拿来给当前Slang path背书，旧`functional@1` package parity也不能迁移为新identity结论。

## 4. 由文献暴露的高价值疑点/缺口

这里的“疑点”表示值得matched诊断，不表示当前代码已被证明有bug。

| 轴 | 直接文献证据 | 当前状态 | 判定 |
|---|---|---|---|
| optimization coordinates | [Taming](../papers/bitterli-2026-taming-optimization-variance.md)证明classic half/difference singularity与stable shortest-arc输入影响compact network trial variance；[Angular Parameterization](../papers/xu-2025-improving-angular-parameterization.md)在两个tiny-MLP BTF上显示D6 direct Cartesian或D9 direct+half按材质占优 | 当前2024 identity使用两个learned frames下direct directions | `candidate mechanism`，坐标clue需按材质/预算matched，不能改写faithful baseline或宣布全局赢家 |
| target transform/activation | [Taming](../papers/bitterli-2026-taming-optimization-variance.md)的power map + LeakySmeLU + successive training；[Hierarchical](../papers/xue-2024-hierarchical-neural-materials.md)只有fourth-root prose/code先例且printed Eq.(2)写`I^-4`；RTA exact log未报告 | 当前`log1p` L1、ReLU、exp | `author-underspecified` baseline + 新ablation；fourth-root只能另立adaptation identity，没有当前defect结论 |
| latent initialization/continuation | [NeuMIP](../papers/kuznetsov-2021-neumip.md) no-blur free latent产生noise；[Biplane](../papers/fan-2023-neural-biplane-btf.md) progressive plane blur；RTA encoder+mollification | 当前encoder bootstrap+direction mollification，无spatial latent blur | `orthogonal diagnostic`；仅对spatial source有意义 |
| analytic core | [Hybrid](../papers/2026-hybrid-neural-microfacet-brdf.md)的`L_a+L_t`保护analytic state；[Belcour](../papers/belcour-2018-efficient-rendering-layered-materials.md) low/moderate roughness有效但tails失败 | 当前direct evaluator无analytic core | M2独立candidate；不能把Belcour/Principled变成公共GT词汇 |
| compiler generalization | [MetaLayer](../papers/2023-metalayer.md)/[NMA](../papers/2026-neural-material-adapter.md)做fixed-family参数→program；[NLB](../papers/fan-2022-neural-layered-brdfs.md)/RTA仍需per-asset optimization | 当前encoder用于单asset训练/bake，不证明G2/G2s feed-forward compiler | 当前claim边界；M6需独立unseen-state实验 |
| output filtering | NeuMIP/RTA独立filtered levels；RTA stochastic one-level | 当前formal config保留hierarchy/filter | mechanism faithful；exact filter choice需per-level/temporal审计 |
| sampler | NBRDF two-param proxy、Belcour R/TRT/TT、Importance Baking tuple与PDF gap | 当前GGX9是强learned baseline | 新sampler只作为proposal轴；current evaluator保持frozen做matched variance/time |
| online query allocation | [Active Exploration](../papers/diolatzis-2022-active-exploration-neural-gi.md)的error×optimizer-step selector优于其uniform/loss-only scene对照，但有proposal/replay/resume code gap | 当前formal recipe使用冻结online query distribution | `training-only diagnostic`；只能另立candidate比较coverage/reference work，不能改变test distribution或冒充runtime sampler |
| reference estimator | [Xia](../papers/xia-2020-gaussian-product-sampling.md)用Gaussian-product proposal降低layered internal estimator variance，且明确与external material sample/pdf分层 | 当前LayerStack以position-free random walk作权威reference | `optimized-code control`候选；只允许在同expectation/measure oracle下减少SE/time，不能改GT语义或替换runtime sampler |
| auxiliary supervision | [CNSR](../papers/granskog-2020-compositional-neural-scene-representations.md)的shadow auxiliary head改善部分geometry容量但material变差、aggregate metric近乎不变 | 当前evaluator只监督canonical bare `f` | mixed-evidence `training-only diagnostic`；只有source/reference本来就能提供的local物理分量才可测试aux head，runtime必须删除且同reference work配对；scene shadow标签不能迁成local材质真值 |
| scene transport | [NeLT](../papers/zheng-2023-nelt.md)、[Superposed DFF](../papers/zheng-2024-superposed-deformable-feature-fields.md)、[Light Probes](../papers/guo-2022-neural-light-probes.md)、[1469](../papers/1469-2026-volumetric-light-transport-inference.md)、[Dual-Band](../papers/mo-2025-dual-band-neural-gi.md)、[LightFormer](../papers/ren-2024-lightformer.md)和[NeLiF](../papers/sheng-2025-nelif.md)输出scene radiance/transfer components，并依赖object/light fields、probe/ray、history或G-buffer/VPL/RSM/shadow/3DGS等场景上下文 | 当前local material不处理scene visibility/GI | `not-applicable`，不是当前evaluator缺组件；object/luminaire-generated state只能另立scene identity研究 |

## 5. 文献机制→当前代码的迁移分类

| 来源方法/机制 | 当前NVIDIA关系 | 允许的项目动作 | 禁止的表述 |
|---|---|---|---|
| [Taming](../papers/bitterli-2026-taming-optimization-variance.md) stable direct+half/difference、LeakySmeLU、power-map、successive instances | same representation family的后续优化研究 | 分轴注册training/coordinate candidate；successive schedule只固定`E_net`，并单列实际`Q_ref`与selection成本 | “修复2024论文错误”、把successive写成fixed-reference-query对照，或不改identity直接替换 |
| [Angular Parameterization](../papers/xu-2025-improving-angular-parameterization.md) direct Cartesian与material-dependent half clue | tiny-budget coordinate diagnostic | 分列raw-tangent D6 direct与D9 direct+half；预先冻结iso-parameter/iso-MAC的width或padding规则及读取预算，按source strata报告 | 用两个UBO材质宣布全family最优，或假称D6/D9天然same-shape、忽略D/read/首层MAC差异 |
| [Hierarchical](../papers/xue-2024-hierarchical-neural-materials.md) fourth-root/gradient与whole-buffer Inception | loss/structured-query/full-buffer三类不同机制；whole-buffer网络不满足当前random-access runtime形态 | fourth-root仅作独立loss adaptation；Sobel只作材质空间邻域query诊断；whole-buffer Inception只能登记为独立full-buffer runtime class或high-capacity teacher | 复制冲突的Eq.(2)，把formal/code坐标与output gap猜成同一配置，或让当前`evaluate(wo,wi)`读取屏幕邻居 |
| [NBRDF](../papers/2021-neural-brdf-representation-importance-sampling.md) fixed `h,d` 与2-param Phong proposal | 坐标/超廉价sampler control | iso-MAC coordinate ablation；frozen evaluator sampler对照 | 把Phong变成evaluator vocabulary |
| [NeuMIP](../papers/kuznetsov-2021-neumip.md) view-conditioned UV warp/independent levels | spatial MaterialX/BTF candidate | `prepare`缓存warp；filtering matched experiment | 用1×1 LayerStack宣称spatial收益 |
| [Biplane](../papers/fan-2023-neural-biplane-btf.md)/[Comprehensive](../papers/xu-2025-comprehensive-neural-materials.md) direction planes | factorized M4候选 | iso-byte/iso-read与z8比较 | 只按MLP FLOP或KB宣称更轻 |
| [Hybrid](../papers/2026-hybrid-neural-microfacet-brdf.md) analytic core+residual | M2 candidate | 保留core-only/direct matched controls与`L_a`消融 | 把其measured-BRDF结论直接外推LayerStack |
| [MetaLayer](../papers/2023-metalayer.md)/[NMA](../papers/2026-neural-material-adapter.md) parameter→program | M6/compiler controls | 仅在native fixed family测G2/G2s和workflow W | 要求所有source先改写成layer/Principled GT |
| [Belcour](../papers/belcour-2018-efficient-rendering-layered-materials.md) statistical mixture | optimized-code control/proposal | low/high roughness stratified；完整mixture PDF | 当作随机游走GT或一般layer exact解 |
| [Guo](../papers/guo-2018-position-free-layered-bsdfs.md) position-free random walk | 当前LayerStack reference family | online GT、source-native sample/control | 当作固定runtime representation |
| [Xia](../papers/xia-2020-gaussian-product-sampling.md) internal Gaussian-product proposal | reference estimator adaptation | 在same expectation/measure oracle下做SE/time control，fit/precompute单列 | 替换external `sample/pdf`、把projected density直接接solid-angle ABI，或改变GT expectation |
| [Active Exploration](../papers/diolatzis-2022-active-exploration-neural-gi.md) error×optimizer-step selector | training query allocation inspiration | test分布冻结后比较uniform/replay/active，分别记`Q_ref/E_net/history` | 把MCMC训练selector称runtime sampler，或让test进入selection |
| [CNSR](../papers/granskog-2020-compositional-neural-scene-representations.md) auxiliary shadow head | mixed training-only prior | 仅在source/reference原生提供local物理分量时测试aux head，deployment删除 | 宣称免费收益、改变runtime ABI，或把scene shadow标签等同local component |
| [NeLT](../papers/zheng-2023-nelt.md) hypernetwork/neural texture与typed ratio/residual composition | separate object-transfer scene identity；只有“global condition→generated state”的系统形态可类比`prepare`，state是否静态有界必须由项目另行冻结 | 未来scene track比较fixed concat、generated weights/texture与object composition；object/light/sample/G-buffer、state build与bytes完整分账 | 把direct shadow ratio/indirect residual变成local BRDF measure，或以per-object image timing证明material shader成本 |
| [Superposed DFF](../papers/zheng-2024-superposed-deformable-feature-fields.md) object-pair sum、deformable triplane与field sum | separate per-scene field identity；C2F可作training diagnostic，coordinate deformation只适用于有spatial state的候选 | scene track冻结object partition/count、field/generated-state bytes和rare-event coverage；未来spatial source才允许matched static/deformable control | 把learned feature sum称精确transport线性分解，或把未冻结`O_max`/field growth接入local ABI |
| [NeLiF](../papers/sheng-2025-nelif.md) observation→spherical lighting field、kernel shadow与HDR scale | separate cross-scene lighting renderer；HDR scale只可作local training normalization假设 | scene track冻结luminaire observation/generation/update、多灯上界与field/3DGS bytes；local只允许独立scale-normalization diagnostic | 把lighting field当material latent，把screen-space shadow/RSM/VPL加入`evaluate`，或用10.56 ms证明single-query Pareto |
| [Dual-Band](../papers/mo-2025-dual-band-neural-gi.md)/[Light Probes](../papers/guo-2022-neural-light-probes.md)/[1469](../papers/1469-2026-volumetric-light-transport-inference.md)/[LightFormer](../papers/ren-2024-lightformer.md) scene auxiliaries | separate scene/image identity | 未来scene transport track；独立buffers/metrics/cost | 用final-image improvement证明local evaluator fidelity，或把scene buffers加入local `evaluate(wo,wi)` |

## 6. 当前最小 matched 诊断序列 `[N/I]`

这些是从文献得到的实验优先级，不表示本研究任务已经授权启动训练。

### 6.1 先确认当前identity本身

1. 用`functional-f@2` formal config做static/preflight，确认source snapshot、two routes、300k/100k lifecycle、loss和filter identities；
2. 以同identity完成smoke/checkpoint resume/package parity；
3. 只在正式冻结后运行300k，不让旧200k artifact参与checkpoint selection；
4. 当前MaterialX资产identity按G1与工作流稳健性W报告，并补充per-level、peak/tail/energy、sampler normalization/variance和single-query cost；G2/G2s只属于另建identity的参数式source-family compiler实验，不能从单资产run推出。

### 6.2 再做单轴训练改进

| 顺序 | 变量轴 | baseline | candidate | 必须冻结 |
|---:|---|---|---|---|
| 1 | target transform | current `log1p` L1 | Taming cube-root power-map；Hierarchical-inspired fourth-root只作单独adaptation；可加linear residual term | network、coords、queries、steps、seeds、optimizer；transform与aux loss分轴 |
| 2 | coordinates | two learned frames下的direct dirs | stable direct+half/difference；另列D6 direct与D9 direct+half tiny-budget controls | activation、transform、training queries、lifecycle、runtime state、texture reads；D6/D9以预注册iso-parameter/iso-MAC remap而非假设相同shape |
| 3 | activation | ReLU | LeakySmeLU | coordinates、transform、shape/MAC、training queries、lifecycle |
| 4 | optimization schedule | single current model | successive instances，固定`E_net=6.5536B` sample–network evaluations | formal四阶段理想`Q_ref≈2.176B`，而`1×65536` baseline为`6.5536B`；实际`Q_ref`、optimizer、selection、state/memory/time分别登记，明确不是fixed-reference-query或query-budget-matched对照 |
| 5 | analytic core | direct evaluator | Belcour/Hybrid core + bounded residual + `L_a` | total MAC/bytes、sampler、source/query |
| 6 | representation | z8 generic latent | plane/factorized或generated state | iso-byte与iso-read两组、filter target |
| 7 | sampler | current GGX9 | 2-param、Belcour mixture、direct tuple candidate | frozen evaluator、MIS/scenes/SPP/time、PDF oracle |
| 8 | training query policy | uniform online queries | replay-only、loss-only与error×optimizer-step active selector | frozen test distribution、`Q_ref/E_net`、proposal/replay/resume state、coverage |
| 9 | auxiliary supervision | main bare-`f` loss | 一个reference-native transport component aux head | main/runtime shape、reference work、loss weight/gradient route、deployment删除head |
| 10 | reference estimator | current random walk | Xia-inspired pair/multiple Gaussian-product optimized-code control | same expectation/measure、support/MIS oracle、fit/precompute与SE/time；formal multiple-product证据只覆盖三因子/两个internal directions，不外推任意层链 |

顺序的理由是先分离优化失败，再改表示容量；否则“新架构更好”可能只是loss/coordinate更可训练。每个变量轴仍需独立candidate identity和bootstrap CI，不自动采用。

## 7. 可证伪的 correspondence 假设

| Hypothesis | Direct evidence | Minimum matched control | Metrics | Falsification condition |
|---|---|---|---|---|
| C1：当前主要改善空间来自optimization，不是增加decoder容量 | Taming同compact family；当前3×64已是2024最大，但作者最强successive证据来自更小compact network | current vs Taming训练机制；3×64迁移实验与原尺度2×16复核分开登记，二者各自固定shape和`E_net` | seed success；资产式source用G1/W，参数式family才用G2/G2s；peak/tail、time、`Q_ref`、selection/state成本 | 同`E_net`无稳定改善，或改善只来自更多reference work/selection成本 |
| C2：bare-f direct training比loss内cosine weighting更安全且不损transport quality | 当前ABI与P/S冲突；NBRDF cosine-weighted loss提供alternative | 以新版本identity从头训练bare-f loss与loss内cosine weighting；runtime都输出bare f | bare-f、transport-weighted、energy/grazing、finite | cosine-weighted在全部相关指标更好且无grazing/ABI问题，否定direct-only优先假设 |
| C3：当前learned GGX9已接近analytic proposal的cost-quality Pareto | RTA sampler、NBRDF/Belcour controls | cosine、2-param、Belcour、GGX9；frozen evaluator | normalization、variance/time、state/MAC | 更小proposal同time/quality不劣，GGX9被支配 |
| C4：source-aware compiler可缩短edit cook但不能免费替代direct fit | MetaLayer/NMA fixed-family evidence | pure compiler、bounded refinement、direct fit | G2/G2s、compile/edit W、runtime program | pure/refined compiler在同program预算达到direct-fit CI且更快，否定“不能免费”中的gap假设 |

旧`functional@1` 200k checkpoint不是C2的matched control：它改变了output measure、runtime adapter、预算和identity。若要检验cosine weighting，必须建立新的版本化训练identity，并保证cosine只在loss中出现一次，不能在runtime从近零cosine反解`f`。

scene-level论文的runtime architecture不成为当前local NVIDIA evaluator候选：CNSR的Pool/global latent/G-buffer、Active Exploration的scene PixelGenerator、NeLT/Superposed DFF的object representations/fields、NeLiF的luminaire field/3DGS，以及probe/ray/history/VPL/RSM/shadow等whole-buffer上下文都不能进入当前ABI。它们的训练选择、normalization或capacity-allocation思想只有在target仍是source-native bare `f`、test distribution冻结、runtime shape不变且另立candidate identity时，才可作为training-only或spatial-source诊断；任何真正依赖scene buffer的模型必须进入独立scene/image identity，并用scene quality、跨场景边界和对应runtime成本验收。

## 8. 结论

当前NVIDIA reproduction的正确定位是：**2024公开方法的bare-f functional adaptation与当前面向部署的实现/配置基线**。它尚不是经过`functional-f@2` 300k、package parity和viewer证据确认的formal deployment result。文献没有证明它存在某个具体实现defect；真正明确的是若干`author-underspecified`选择和旧/新identity证据隔离。

后续改进应保留2024 baseline不动，以Taming optimization、Angular coordinates、Hierarchical loss/structured-query diagnostics、active query allocation、training-only auxiliary supervision、Xia-style optimized reference control、analytic-core residual、plane/factorized representation、typed compiler和sampler controls作为独立候选。NeLT的generated state、Superposed DFF的deformation/C2F和NeLiF的HDR scale只能按上表拆成新的matched轴；只有实验能决定其迁移价值。scene-level transport始终是第二条方法identity，而不是给local material evaluator补隐藏输入。

## Evidence review

```text
author: /root
reviewer: /root/nelif_full_report
reviewed_at: 2026-08-29
sources_rechecked:
  - docs/learning.md
  - configs/learning/nvidia-rta2024-materialx-formal.json (SHA-256 53503C6E8A36D25B9EC09FB2128F6E538E6C70CCB2239DDA847D61D59F9E3CAF)
  - current NVIDIA config validator, model/objective and FP16 shader paths
  - archived 08-27 correspondence, 200k recording decision and formal-report identity
  - docs/realtime_material_compilation.md and docs/research/experiment_framework.md
  - evidence-reviewed NeLT, Superposed DFF and NeLiF individual reports linked in sections 4 and 5
  - five evidence-reviewed comparison syntheses linked after section 2
findings_closed:
  - confirmed NeLT/Superposed DFF/NeLiF remain separate scene identities and do not establish any current NVIDIA implementation defect
  - tightened NeLT generated-state wording so static bounds, build latency and bytes must be frozen rather than inferred from the paper
  - confirmed Superposed DFF C2F/deformation and NeLiF HDR scale are isolated candidate/diagnostic axes, never faithful NVIDIA fields
  - confirmed no scene buffer, object/luminaire field, RSM/VPL/shadow/3DGS input is introduced into the local evaluate/sample/pdf ABI
remaining_evidence_gaps:
  - no current functional-f@2 formal 300k run, checkpoint, package parity or viewer evidence exists in this literature task
  - the authors' P/S output-measure conflict and exact log/KL/stage-switch details remain unresolved by public 2024 evidence
  - exact quality and runtime claims for functional-f@2 require a future formal execution under the locked identity
  - NeLT/Superposed DFF/NeLiF的supplemental/code、generated-state bytes/update与完整runtime scope未公开
  - 新增机制均为独立候选/diagnostic，不构成当前实现defect
review_status: evidence-reviewed
```
