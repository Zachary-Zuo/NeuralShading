# Optimization 与 Loss：跨论文证据综合

## 1. 证据边界与比较口径

本文只使用当前 `report_status: evidence-reviewed` 的 28 篇个体报告。其中 25 篇包含 learned component；Belcour 2018、Guo 2018 与 Xia 2020 没有 neural fitting，只作为“解析 control/reference、数值密度拟合或 table precompute 不应被误称为 neural training”这一边界。除 local neural material/appearance 方法外，本版已纳入 Weier 2023 的 asset-level neural prefiltering、Xu 2025 的 tiny-MLP angular parameterization diagnostic、Zheng 2021 Neural Processes BRDF、Xue 2024 Hierarchical Neural Materials，以及 Granskog 2020 CNSR、Guo 2022 Neural Light Probes、Diolatzis 2022 Active Exploration、Zheng 2023 NeLT、Zheng 2024 Superposed DFF、Ren 2024 LightFormer、Sheng 2025 NeLiF、Mo 2025 Dual-Band Neural GI 与匿名稿 1469 的 scene/volume optimization。CNSR、NeLT、Superposed DFF与NeLiF都输出scene/image radiance或其transfer components，不是 local evaluator；三份新解锁正文未披露的supplemental/code字段继续保持“未报告”，不由相邻方法补齐。

本文不把论文里的 `L1`、`log`、`batch` 或 `epoch` 按名称合并。至少要同时锁定：

1. 网络最终代表的物理量；
2. loss 前后的 target/output transform；
3. cosine、PDF、曝光或颜色空间是否已乘入；
4. query 如何抽样、一个 batch 的统计单位是什么；
5. 哪些参数同时更新、冻结或由 teacher 监督；
6. paper formal、supplemental、release example/default 是否真属同一 identity。

各事实回链到个体报告 §6–7；作者明确失败的尝试回链到 §10，paper↔code 差异回链到 §11。跨论文数值只用于描述配置，不构成质量或效率排名。

## 2. Target transform 与 loss 总表

| 方法 | 训练目标/输出变换 | 正式 loss | cosine/measure 边界 | 已锁定的关键缺口 |
|---|---|---|---|---|
| NBRDF 2021 evaluator | 网络表示 bare `f_r`；paper称 final exponential，Keras 为 `exp(raw)-1`；loss 内再做 `log(1+f_r cosθ_i)` | paper只写范数；Keras 对 RGB 与 batch 求 mean absolute difference | training objective 乘 incident cosine；旧 Mitsuba `eval()` ABI 返回 `f_r cosθ_o`，不能与网络函数值混名 | paper 未报告 optimizer；代码 Adam `5e-4`；seed 参数未接入 query 生成；论文 learned analytic sampler 不在公开 renderer 中 |
| NeuMIP 2021 | paper称 RGB reflectance `log(x+1)`；code 为 YUV→RGB、`exp(raw)-1`，再乘 learned shadow mask | paper 未披露 norm；README `comb2=MSE(log1p(clamp))+0.1 L1(linear)` | 输出是 filtered MBTF reflectance；code 允许训练期小负值后再 clamp | formal norm、paper checkpoint 与 `comb2` correspondence、split/seed 未锁定 |
| Neural Layered BRDFs 2022 | representation 正文拟合 raw scalar BRDF/channel；layerer拟合 optimized latent；sampler拟合 normalized GNDF | evaluator raw L1；layerer latent L1；sampler softmax-KL | release projection example却在 loss 前乘 `light_z` 并用 `y/(1+y)` | optimizer、batch、seed未报告；release只含 experimental projection，不是formal joint training |
| Neural Biplane BTF 2023 | 正文先学 averaged value，再单独调 color；精确 target transform 未报告 | formal norm/颜色空间未报告；release 对 `output*cos` 与 clamp radiance 用 L1 | code 的 cosine convention 与正式 checkpoint identity未闭合 | 论文没有 exact loss；release batch composition、epoch与默认入口均不同于正文 |
| MetaLayer 2023 | `μ`-law，`T(f)=sign(f)log(1+32abs(f))/log(33)` | reverse-Huber：`e≤0.1` 为 L1，否则 `(e²+0.1²)/(0.2)` | 分别训练BRDF与BTDF scalar；不是权重/latent loss | batch、`K1/K2`、`X1/X2`比例、seed和BSDFNet exact mapping未报告 |
| NVIDIA RTA 2024 | supplemental evaluator为 `exp(raw-3)`，学习 cosine-weighted RGB BRDF | BRDF log-space L1；sampler KL；albedo one-sample MC L2 | 正文把输出称 `f`，supplemental functional listing返回 `f cos`；必须保留ABI gap | exact log公式、KL方向/normalization/estimator、stage switch、seed/model selection未报告 |
| Comprehensive Neural Materials 2025 | 一般材质直接 RGB；极高动态 glossy target 用 `log(1+x)`，但 inverse/runtime correspondence 未报告 | output MAE；feature planes 与 QTP joint training | BTF radiometric unit、颜色空间与 cosine convention 未报告 | release default 的 plane channels、final activation、schedule 与 formal config 不同；helper 不能直接组成 paper-config build recipe |
| BSDF Importance Baking 2023 | sampler直接监督 direction 与 `f cos/p`；evaluator监督 `f cos`；PDF监督target density | sampler `L1(direction)+0.4 L1(log1p weight)`；evaluator stop-gradient normalized relative L1；PDF log1p L1 | 三个网络分别代表sample tuple、cosine-weighted evaluator与MIS PDF | sampler map与独立PDF网未被证明归一匹配；epoch内query数、Ranger细节、selection/seed未报告 |
| Hybrid Neural-Microfacet 2026 | analytic/full 两个分支都对 `log1p(cosθ_i f)`比较 | `L=L_a+L_t`，两项都是 L2；`L_a`约束analytic parameters，`L_t`约束完整输出 | runtime Eq.1 为 bare BRDF，cosine只在training objective | AdamW其余超参数、init/seed/model selection未报告 |
| Taming Optimization Variance 2026 | `M_pow(x)=3(x^(1/3)-1)`；formal decoder `f=exp(z)` | mapped-domain L1 | formal目标为nonnegative RGB BRDF；release default output是ScaledSigmoid而非formal exp | paper optimizer/LR未报告；code默认tuple/output/boundary offsets与formal实验不是同一identity |
| Neural Material Adapter 2026 | 对 source bin mean 与 analytic-target bin mean 做 `log(1+x)` 后 absolute difference | 正文正式配置为 log-domain L1；supplemental教学公式又写 squared error | network 输出 analytic target 参数，再由 target evaluator 生成 BRDF；不是直接输出 BRDF value | PFMC table recipe 与 supplemental “millions/bin”、89/92 set、lobe数、reciprocity、方向 encoding 均未闭合；无 code 裁决 |
| Mobile VR 2026 | student/teacher/GT 均称 RGB reflectance；是否 linear、颜色空间与 clamp 未报告 | `L1(student,GT)+0.1L1(student,teacher)+0.1L2(two hidden-feature pairs)` | local BTF response 经系统写入 object-space radiance texture；不是本项目 bare-`f` ABI | split、activation、seed/init、teacher exact architecture与 checkpoint selection 未报告 |
| Improving Angular Parameterization 2025 | 输出称 RGB reflectance；linear/log、cosine-weighted、clamp 与 nonnegative 均未报告 | 只报 entire test dataset 的 average `L1` difference；训练 objective 是否同为 L1 未报告 | 两种 BTF 上的 7-channel neural-texture feature + direction → RGB；exact scattering measure 与 cosine convention 未闭合 | 无 code/supplemental；split/query sampling、activation、optimizer、横轴单位、seed/checkpoint、MAC/bytes/runtime 均未报告 |
| Neural Processes BRDF 2021 | measured RGB BRDF连续做四次`log1p`；main decoder反变换后为 bare RGB BRDF；NICE另把其 luminance乘`cosθ_i`并归一化 | main NP 为`Σ_err=0.2I`的fixed-covariance Gaussian reconstruction term + `KL(q(z∣target)‖q(z∣context))`；hyper 在P中只报teacher-output difference而exact reduction未报告，C graph为`0.5`倍平方和；NICE为方向/归一化估计均未报告的KL | main NP、hyper 与 NICE 是三个分离生命周期；NICE target才含 incident cosine | P 对每材质/每 observation latent draw文字冲突；main training code与NICE code不可得；paper encoder/aggregator和release topology不同 |
| Hierarchical Neural Materials 2024 | 正文说对prediction/GT取fourth root，release的`4maploss`/部分stage branches实现该变换；printed Eq.(2)/TeX却写`I^-4`且缺显式双参数 | paper为 pixel L1 + squared Sobel-gradient；release为 grayscale gradient-L1 + pixel L1，并有多个未裁决schedule | 输出/measure correspondence未闭合；它监督整张 spatial-angular material buffer，不是已锁定的 bare-`f` random query | formal adaptive schedule、batch单位、LR/steps/seed与paper checkpoint identity未报告；README defaults不是论文配置 |
| CNSR 2020 | HDR input、prediction与loss均在`log1p` space，runtime再`expm1` | P为pixelwise L1 + DSSIM并经验缩放到近似同量级；C实现`2·L1 + (1-SSIM)`；另有factor-batch averaging/gradient correction、optional null penalty与shadow auxiliary loss | 三张scene observations编码global `r`，novel-view G-buffer + `r` 输出 image radiance；不是local scattering/evaluator measure | formal gradient-correction strength不明；release Pixel/save为`gradient_reg=0`、GQN/U-net为`1.0`；null `β=λ/256`但formal config缺失；paper/checkpoint 1M 与JSON 2M/20M、ArchViz 64²/128²冲突 |
| LightFormer 2024 | loss前对HDR values做`log(1+x)`，但没有逐项说明prediction/GT/VGG input的transform placement | direct shading、indirect shading、direct shadow三项per-pixel L1各权重1；direct-shadow VGG loss权重0.1 | per-scene、per-light component image reconstruction；不是local scattering measure | exact log/negative handling、VGG layer/normalization、attention projection、steps/schedule/seed/checkpoint与code均不可得 |
| Dual-Band Neural GI 2025 | HDR radiance 使用 `log1p` mapping，但论文未说明mapping在两项loss中的逐步位置；supplemental锁定final head为LeakyReLU | `L1 + L_SSIM`，两项系数均为1 | per-scene screen-space radiance reconstruction；不是local BRDF/transport operator，也没有cosine/PDF measure | final head输出mapped值还是radiance、inverse/clamp、LeakyReLU slope、stage length、split与`γ`多尺度merge均未闭合 |
| Neural Prefiltering 2023 | paper未报告transform；Arcade release用 `log1p(RGB)`、runtime `exp-1` | appearance relative L2 noise-to-noise；visibility BCE | target是voxel aggregate throughput/visibility，不是local scattering | paper的8k–12k与release max30k、paper无log与release log1p不能静默合并 |
| Neural Light Probes 2022 | HDR inputs 经过未给公式的 logarithm compression；网络输出 final image | `0.8 L1 + 0.03 LPIPS + 0.15 temporal L1`，原权重和为 `0.98`，论文未要求再归一化 | image-space scene reconstruction；没有 local BRDF cosine/PDF measure | loss 所在域/inverse、VGG layer/normalization、batch/sequence recipe、Adam超参、seed与 checkpoint selection 未报告 |
| Active Exploration Neural GI 2022 | GT与emission做`log1p`，preview再`expm1` | paper写`L1 + structural dissimilarity`；code为`2·abs(pred-gt) + (1-SSIM)` | per-scene G-buffer→radiance image；active MCMC改变的是训练配置分布，不是runtime BSDF sampler | 正文无总steps；small-step proposal、首步resolution、fresh batch、replay/resume状态均有paper↔code gap |
| NeLT 2023 | shading与light-sample radiance按mean expected power归一后逐RGB channel做`log1p`；背景direct target是ratio，indirect是residual | Eq.(9)四项L1：foreground diffuse/specular/indirect三项乘`ē`，background shadow ratio不乘；RGB channel独立训练/推理的参数复用方式未报告 | 同一模型的mask两侧不是同一measure：前景存radiance，背景direct存乘法ratio、indirect存加法change；near-zero ratio稳定化未报告 | Adam `1e-4`、batch64、200k iterations、2×A6000约60 h/object；supplemental/code缺失，exact topology、reduction、seed/model selection未闭合 |
| Superposed DFF 2024 | HDR target做`log1p`；inverse/negative/output activation未报告 | `L1 + structural dissimilarity`，式中权重均为1 | per-scene final radiance reconstruction；field feature与object-pair latent不是监督的物理transport component | Adam `1e-4`、batch16、4×A6000约15 h/scene；`α` C2F schedule、steps/epochs、offset regularization与state配置在不可得supplemental |
| Volumetric Transport 1469 | 输出 scattered-radiance image；分别在线性域与 `log1p` 域比较 | `λ_1 L1(linear)+λ_log L1(log1p)` | image-space participating-media transport；无 local BSDF/phase ABI 或 cosine measure | 两个 loss weight、AdamW超参、LR端点、split identity、完整网络/feature packing与 code 均不可得 |
| NeLiF 2025 | luminaire observations以跨views global max RGB归一，最终shading乘回同一intensity factor；该factor是scalar还是RGB、full composition的radiometric measure均未闭合 | 正文未报告正式loss、各component权重或optimizer；只给与LightFormer在400K subset共同训练35 epochs | direct×shadow+indirect再与Albedo/intensity/3DGS合成scene image；不是local scattering measure | 12×RTX4090D训练，但batch、steps、LR、时间与full-1M vs matched-400K run identity均未报告；supplemental/code不可得 |
| Belcour 2018 analytic control | 无 neural target/output transform；roughness fit 与 `FGD/TIR` table 是数值预计算 | 不适用 | official Mitsuba `eval` 是旧式 cosine-weighted ABI；reference 只承诺 direct-integrator sampling 边界 | 不得把数值 fit/table 称作 training；formal/reference/code 的 media、refraction与 TIR correspondence 另有 explicit gaps |
| Guo 2018 stochastic control | 直接 Monte Carlo 估计 layered BSDF；无 learned output | 不适用 | stochastic BSDF sample weight、scene-MIS PDF 与 approximate PDF 是不同 estimator roles | 无 neural optimizer；downstream NLB/MetaLayer 使用它生成 GT 的训练配置不属于 Guo 2018 |
| Xia 2020 Gaussian Product Sampling control | regular roughness把BSDF slice拟合为Gaussian mixture，low roughness拟合specular-anchored covariance/Cholesky precision；无 learned output | regular为maximum-likelihood/negative-log-probability fit，low-roughness为Frobenius fit；SLSQP而非neural optimizer | pair/multiple-product内部slope proposal、external layered `sample/pdf` 与 exact integrand是分离角色；projected-vs-solid-angle notation 仍有reconciliation gap | 无official code/supplemental；fit grid/range/tolerance/restarts/seeds/hardware未报告；23 min Beckmann fit不是neural training time |

证据：[NBRDF §7/§11](../papers/2021-neural-brdf-representation-importance-sampling.md)、[NeuMIP §7/§11](../papers/kuznetsov-2021-neumip.md)、[NLB §7/§11](../papers/fan-2022-neural-layered-brdfs.md)、[Biplane §7/§11](../papers/fan-2023-neural-biplane-btf.md)、[MetaLayer §7/§11](../papers/2023-metalayer.md)、[RTA §7/§11](../papers/zeltner-2024-real-time-neural-appearance-models.md)、[Comprehensive §7/§11](../papers/xu-2025-comprehensive-neural-materials.md)、[Angular Parameterization §7/§11](../papers/xu-2025-improving-angular-parameterization.md)、[Importance Baking §7/§11](../papers/bai-2023-bsdf-importance-baking.md)、[Hybrid §7/§11](../papers/2026-hybrid-neural-microfacet-brdf.md)、[Taming §7/§11](../papers/bitterli-2026-taming-optimization-variance.md)、[NMA §7/§11](../papers/2026-neural-material-adapter.md)、[Mobile VR §7/§11](../papers/xu-2026-real-time-neural-materials-mobile-vr.md)、[Neural Processes §7/§11](../papers/zheng-2021-neural-process-brdfs.md)、[Hierarchical §7/§11](../papers/xue-2024-hierarchical-neural-materials.md)、[CNSR §7/§11](../papers/granskog-2020-compositional-neural-scene-representations.md)、[NeLT §7/§11](../papers/zheng-2023-nelt.md)、[Superposed DFF §7/§11](../papers/zheng-2024-superposed-deformable-feature-fields.md)、[LightFormer §7/§11](../papers/ren-2024-lightformer.md)、[NeLiF §7/§11](../papers/sheng-2025-nelif.md)、[Dual-Band §7/§11](../papers/mo-2025-dual-band-neural-gi.md)、[Neural Prefiltering §7/§11](../papers/weier-2023-neural-prefiltering-lod.md)、[Neural Light Probes §7/§11](../papers/guo-2022-neural-light-probes.md)、[Active Exploration §7/§11](../papers/diolatzis-2022-active-exploration-neural-gi.md)、[1469 §7/§11](../papers/1469-2026-volumetric-light-transport-inference.md)、[Belcour §7/§11](../papers/belcour-2018-efficient-rendering-layered-materials.md)、[Guo §7/§11](../papers/guo-2018-position-free-layered-bsdfs.md)、[Xia §7/§11](../papers/xia-2020-gaussian-product-sampling.md)。

## 3. `log`/power compression 解决的是梯度动态范围，不是输出语义

### 3.1 三种不能互换的压缩

- NBRDF把 `f cosθ_i` 放进 `log1p`，因此峰值、掠射cosine和loss权重被共同改变；把它改成bare `f` 的 `log1p`不是同一objective。
- MetaLayer的 `μ`-law 带归一化分母 `log(33)`，再叠加 reverse-Huber；其大误差branch重新加强峰值，小误差branch保留long tail。它不是普通的log-L1。
- Taming的cube-root power map在零附近展开梯度、压缩大值，且 formal inference 仍用指数正输出；它针对compact网络的可优化性，而不是宣称改变BRDF measure。
- Neural Processes把 measured BRDF连续做四次`log1p`，并报告该次数稍优；它随后还用固定 covariance 的 Gaussian reconstruction term与latent KL，因此不能把“四重log”单独抽成与普通log-L1相同的对照。
- Hierarchical Neural Materials的正文和release `4maploss`/部分stage branches支持 fourth root，但正式 Eq.(2)/TeX 字面却是 reciprocal fourth power。它能作为“作者探索过更强 power compression”的证据，不能在没有勘误时被当作与 Taming cube-root formal map 完全一致的独立复现，也不能把release default称为fourth-root formal run。

NeuMIP release 的 `comb2` 同时保留 linear-domain L1；匿名稿 1469 也把 image-space linear L1 与 `log1p` L1 相加，但两个权重未报告。二者都说明压缩域与线性域可以承担不同误差区间，却不因此成为同一个 objective：NeuMIP 监督 filtered reflectance，1469 监督 scattered-radiance image。Dual-Band 给出 `log1p` 与 `L1+DSSIM`，却没标明mapped tensor如何进入两项loss、LeakyReLU final head之后何时inverse；LightFormer只说在loss前对HDR values做`log1p`，没有逐项对应三项L1与shadow VGG input；Neural Light Probes也只说明 HDR inputs 做 logarithm compression，没有给公式、inverse 或 loss 所在域。后三者都不能补成某一种标准 `log1p` loss。任何移植都必须把“target transform”和“loss norm/weight”作为两个轴，不得用一个 `log loss` 开关同时替代。

CNSR 的 deterministic generator 也在 `log1p` image space 使用 `2·L1+(1-SSIM)`，但它的三张observation、global scene latent、novel-view G-buffer 与factor-batch partition constraint都是scene-level identity。Improving Angular Parameterization 只报test L1 curve，连训练loss和target transform都未报告。前者不能作local evaluator的`log1p+DSSIM`直接先例，后者不能因为图中出现`L1`就被归入L1训练方法。

LightFormer 还明确报告“加入 AE 式 DSSIM 没有 performance gain”，而 Dual-Band 的正式 loss 正是 L1+DSSIM。两条第一方事实并不矛盾：二者的 scene representation、component decomposition、target processing 与 DSSIM protocol 不同，LightFormer也没有披露 rejected DSSIM 的权重。可迁移的结论只是把结构项作为独立、matched的loss轴，不能宣布DSSIM普遍有效或无效。

NeLT、Superposed DFF与AE formal都出现`log1p`及L1/structure项，但objective identity仍不同。NeLT额外按mean expected light power归一，并让direct背景ratio独立于乘回`ē`的三个radiance项；Superposed DFF只监督final image；AE还把scene configuration selector与replay耦合进训练。NeLiF正文甚至没有公开loss/optimizer，只有observation global-max normalization和35-epoch matched subset。因而不能从“都是HDR scene”推导四者共享可执行loss recipe，也不能用NeLiF Table1结果反填缺失objective。

### 3.2 非负输出不能单独保证稳定

NBRDF、NVIDIA RTA、Taming、NeuMIP 和 Hybrid 都用不同方式保证或鼓励非负，但其优化行为还取决于 raw activation、offset、mapped loss 与初始化。Taming formal 的 `exp`、2026 release default 的 ScaledSigmoid，以及 RTA 的 `exp(raw-3)` 不是可互换的“positive activation”。尤其指数的 raw offset 会直接改变初始输出尺度和梯度；若未冻结初始化/offset，就无法把 seed 差异只归因于 coordinate 或 activation。Dual-Band supplemental明确 final head 是 LeakyReLU，但没给negative slope、mapped-output/inverse/clamp lifecycle，因此不能把它写成非负radiance head。对 Mobile VR、LightFormer、Neural Light Probes 与 1469，正式材料也没有披露足以锁定相同问题的 final activation/clamp；本版不从“预测 radiance/reflectance”反推它们的非负参数化。

## 4. Query distribution 是 objective 的一部分

| 方法 | 训练 query recipe | 它实际强调的区域 | 风险/未报告信息 |
|---|---|---|---|
| NBRDF | 每材质总 `8×10^5` random `(h,d)`；公开代码约80/20 query split | half/difference域的总体函数 | seed参数未实际接入随机生成；anisotropic formal assets不可得 |
| Improving Angular Parameterization | UBO2014 Leather11/Fabric12；十种angular input均from scratch，共享7-channel neural-texture feature | tiny `3×8×8` MLP 在direct/half-difference、spherical/Cartesian/PE/latent下的coordinate burden | spatial/directional sampling、split、batch/data materialization与Fig.3横轴单位全未报告；只是两材质diagnostic |
| Neural Processes | 每 iteration取16个材质；每材质context size从1到16,200随机、target固定16,200；151 measured BRDF全部参加main training | 可变稀疏观测下的函数空间后验与训练集材质压缩 | context discrete distribution及是否嵌套target未报告；主结果不是held-out material；main training entry缺失 |
| NeuMIP | offline每texel约200–400 queries；训练batch约`2^20` queries；footprint/level随机 | spatial、view/light与continuous scale | GT split未报告；code progressive maximum-level sampling是paper未披露lifecycle |
| NLB | 每BRDF `25^4=390,625` direction pairs；600 base、12,720 two-layer、1,800 three-layer | 规则dense direction grid与固定layer family | batch内BRDF/query routing未报告；规则grid不等于当前online query recipe |
| Biplane | 84 BTF，每个约`6.4×10^7` random queries；每step formal为4 BTF×160k | spatial/half/difference总体分布 | grazing方向undersampling是作者确认的失败来源；完整split manifest不可得 |
| MetaLayer | 12k BRDF、10k BTDF；每BRDF `25^4`，每BTDF `4×25^4` | 固定two-interface/one-medium family | 每step材质/方向batch和phase-2拆分未报告 |
| RTA | online GPU reference；evaluator与sampler各65k/iteration；uniform UV、direction/half/difference采样并含biased mip recipe | spatially varying filtered response与sampler jointly | 两批RNG关系、sampler KL target estimator未报告 |
| Comprehensive | measured BTF使用完整角度表与`400²`空间图；synthetic 每个入/出方向各取 Shirley-map `10×10`，共`10^4`方向对、每图`800²` | 完整 same-angle spatial image 与离线 BTF table | formal split、radiometric unit、方向 held-out 未报告；release train/validation 实际复用同一 dataset |
| Taming | fixed-material trials；formal 每step保持 `instances×batch=65,536` 次 sample–network evaluations；shared query batch 在实例间broadcast，unique reference queries/step实际为`1k→4k→16k→64k` | 多初始化并行后逐步提高单实例batch，并降低早期reference generation量 | 不是未见材质泛化实验；固定的是network evaluations而非unique reference queries；formal optimizer/LR未报告 |
| Importance Baking | 正文对每个所称 material 生成 32,768 slices×128²；上下文是三个 parametric family，但 checkpoint 粒度未闭合；训练batch 1,048,576 | parameter/view/random-map 到 sample/eval/pdf | train/test split、epoch内query count、disk map与内存实现未报告 |
| Hybrid | 每step 1,024 cosine-weighted direction pairs，并应用到batch内全部BRDF | 接近cosine proposal的主体能量 | sharp/specular与grazing覆盖取决于数据；没有direction-stratified训练消融 |
| NMA | 每材质 819,200 次 PFMC random walk 形成 `40×1×40×80=128,000` bins；训练每step取1,000 bins，analytic target在每bin求20个方向对 | 固定三层 native parameter family 的离散 source-bin mean 与 analytic-target bin mean | 正文 PFMC 总量平均仅约6.4 samples/bin，与 supplemental “millions/bin”不相容；20个方向对不是新增 PFMC reference samples |
| Mobile VR | 每step取一个完整 `400×400` same-angle slice | 一个角度下覆盖完整空间纹理，利于GPU/data coherence | 没有LoD；train/test split和angle sampling细节未报告 |
| Hierarchical | paper称每个material 500 pairs、batch 30,000但单位未定义；release是full-buffer batch4，并随机选择可用mip level | spatial/angular material buffer、局部邻域梯度与跨mip训练 | source/query distribution、filter kernel、paper batch identity、formal schedule与checkpoint mapping未锁定；两个release H5不足以组成default batch |
| LightFormer | 每scene 20,000 random configs train、100 random configs test；GT为512²、2048 spp Falcor component images；每灯500 VPLs、environment 2,000 | 同一scene内camera、dynamic object/material/light变化与per-light attention | 无validation；参数范围、seed/stratification、VPL stream、crop/整图batch与temporal sampling未报告 |
| Dual-Band | 每scene 8,192组 random camera+dynamic-object configs；4,096组roughness `>=0.25`供stage 1，完整8,192供stage 2；GT为512²、1024 spp Falcor radiance再经offline denoiser | 固定scene的camera/object/roughness变化；stage 1先限制低角频外观，stage 2再加入mirror-hit secondary feature | split/seed、camera/object范围、denoiser bias、dynamic-light sampling、temporal sequence均未报告 |
| Neural Prefiltering | online先采 LoD0 active voxel再映射到均匀抽取的 LoD ancestor；appearance/visibility 分开生成；Arcade release每网262,144 candidate queries/step | 按细层 occupancy 加权的多 LoD voxel aggregate transport | paper未报告batch；release `log1p`、30k max steps与paper raw-throughput/8k–12k口径不能合并 |
| CNSR | 144k procedural scenes/dataset；每batch 16 scenes，每scene三张64² observations+一张query；batch内只变lighting/geometry/material一个factor | scene observation-set编码、factor partition 与novel-view HDR image synthesis | formal dataset SPP、seed/camera split、gradient-correction strength不明；release 9000/1000 batch-file defaults不等于README 90/10 smoke |
| Neural Light Probes | 每场景20张 path-traced training views；输入256 spp lightmap、通常256个128 spp glossy probes与当前 view G-buffer；temporal loss需要连续帧 | 固定场景、novel-view image reconstruction与temporal consistency | crop/batch、连续帧配对、camera trajectory、GT spp/color pipeline与validation未报告 |
| Active Exploration | 16条MCMC chains生成`32×32` online GT patches；以`Loss·norm(ΔAdam)`选择高价值scene configs，配合replay与`128→600` adaptive resolution | 当前模型仍可改善的shadow/reflection/caustic困难区域 | 它改变训练配置分布而非test distribution；fresh batch实际16–32、proposal family和resume/replay identity有code gap，正式总steps未报告 |
| Volumetric Transport 1469 | 500 clouds×1,000 generated illumination pool构成22,000张4K-spp reference images；随机 camera/light/HG/albedo；200张 held-out images | cloud/HG domain 的 scattered-radiance image | train/validation/test在 volume/environment 上是否 disjoint、resolution、reference error与 feature ray budgets未报告 |
| Xia control | regular roughness在`(η,α,θ_i)` grid上离散BSDF slice并拟合Gaussian mixture；low roughness数值求target covariance | offline analytic proposal 的density/covariance fit，不是neural query training | grid range/resolution/cell quadrature、held-out states、fit error distribution均未报告；runtime internal proposal与external `sample/pdf`不可合并 |

这里最重要的结论不是“随机优于规则网格”，而是训练 query 必须与测试 strata 和 runtime query recipe 共同冻结。Biplane 明确报告掠射角欠采样失败；NVIDIA 使用前 20k steps outgoing-direction mollification（cone 从 10° 收缩到 0°）平滑早期目标；NeuMIP/Biplane 又对空间 latent 施加逐步缩小的 Gaussian blur。这些机制都在训练早期改变有效 target distribution，不应被记成普通数据增强。Neural Processes的context/target集合、Hierarchical的整幅buffer、CNSR的factor batch与Active Exploration的MCMC patch同样是不同统计单位。尤其Active Exploration的selector是训练期配置搜索：它不改变runtime sampler，也不能不经reweight/held-out审计就宣称对目标query分布无偏。Neural Light Probes、CNSR、LightFormer、Dual-Band 与 1469 的统计单位是整幅 scene image，Weier 是 voxel aggregate path estimator；它们都不能用 local-BRDF 的“每 step 方向对数”直接换算 batch parity。Dual-Band 还用 roughness partition主动改变stage-1 target support，不能把两阶段只记成LR变化；LightFormer的20,000 configs也不能在缺少参数分布与VPL随机流时简写成“dynamic scene uniform samples”。

## 5. Optimization lifecycle：容量不是始终一起更新

### 5.1 主要阶段表

| 方法 | 正式阶段/更新对象 | 训练工作量与硬件 | 生命周期含义 |
|---|---|---|---|
| NBRDF evaluator | 每材质独立fit；diffuse约5、mirror-like最多90 epochs | 10 s–3 min/material；GPU未报告 | 简单per-material regression；不证明shared compiler泛化 |
| Improving Angular Parameterization | 十种coordinate variants各自from scratch | Fig.3横轴0–150但单位未标；optimizer、LR、batch、hardware/time均未报告 | 只能作coordinate diagnostic；test L1 curve不能补成training lifecycle或optimization-variance证据 |
| NLB | shared evaluator+per-BRDF latent joint 50 epochs；layerer 1000 epochs再three-layer finetune；sampler10 epochs | RTX2080Ti：40 h、10 h、<1 h | representation、composition与proposal是三个独立优化问题 |
| NeuMIP | 30k iterations；feature blur递减；offset在10k后冻结 | RTX2080Ti：512²约45 min、1024²约90 min | coarse-to-fine正则化与模块冻结共同决定最终解 |
| Biplane | shared decoder 30 epochs；新BTF compression先15 epochs只学planes/offset，再5 epochs只学color adapter | RTX2080Ti：shared约18 h | structure/intensity与color显式解耦；同时优化是author-negative |
| MetaLayer | Phase 1 joint；Phase 2交替冻结MetaNet/BSDFNet，用`X1/X2`分开更新 | 8×2080Ti：BRDF 225 epochs/50 h；BRDF/BTDF models 200 epochs/约48 h | 目标是稳定hypernetwork→generated-state mapping；不是test-time inner loop |
| RTA 2024 | encoder→decoder；materialize全texel/MIP并丢弃encoder；direct latent finetune；evaluator/sampler simultaneous | 300k iterations、约40B online samples；RTX4090 4–5 h/material | encoder是训练期locator，不是runtime state；exact switch未报告 |
| Comprehensive | feature planes 与 quantized tensor-product MLP joint train；formal 最多300 epochs、cosine warm restarts | RTX4090约18 h；可按loss提前停止或继续 | formal schedule 与 release 50-epoch plain-cosine default不同；public export/runtime helper也未形成端到端formal identity |
| Taming 2026 | 4 phases：并行instances `64→16→4→1`，per-instance batch `1k→4k→16k→64k`；每step保持65,536次sample–network evaluations，并在实例间共享query batch | 100k steps，共6.5536B sample–network evaluations；按理想4×25k phases派生约2.176B unique reference queries | successively eliminate差初始化并提高幸存实例batch；network-eval cost matched，但reference generation并非与`1@64k` baseline等量 |
| Importance Baking | sampling/evaluator/PDF三个网络各500 epochs | RTX3090：sampling约48 h，另两网各约12 h | 三个ABI函数独立监督，不能假定sample↔pdf自动一致 |
| Hybrid | shared network与每材质 analytic state/latent joint优化，200k steps；AdamW `5e-3` cosine decay、clip norm `0.01` | RTX5080：MERL 100材质约10 min | `L_a`约束可解释 analytic state，`L_t`约束完整输出；两项不是重复误差 |
| NMA | 先形成 PFMC source table，再训练5×64 adapter；Adam `1e-5`、OneCycle；steps/epochs未报告 | RTX3090：8,000材质正文约2 h、supplemental写under 3 h；table生成成本未计 | optimizer每个bin复用source mean并多次求analytic target；不能把runtime “no precomputation”回写成训练data lifecycle |
| Mobile VR | teacher 60 epochs；student最多150 epochs，用output+hidden feature distillation | RTX4080约19 h | teacher capacity与feature alignment是training-only prior |
| Neural Processes | main NP 40k iterations；再冻结decoder，独立训练hypernetwork 60k；NICE sampler另行post-train | RTX2080Ti：约40 h、20 h、2 h | 函数空间后验、per-material weight materialization与sampling density是三个独立阶段；NICE的“similar configuration”不足以补齐optimizer identity |
| Hierarchical | paper只报告Adam、约90 min/material；release含多个互异loss schedule、随机mip与blur/offset lifecycle | training GPU、formal steps/LR/stage/selection未报告 | 不能从README的30k/MLP/`comb1` defaults回填Inception+fourth-root formal方法；“adaptive”没有唯一可执行身份 |
| LightFormer | 各scene的component encoders/decoders end-to-end joint train；Adam lr `1e-4`、batch4；steps/epochs与schedule未报告 | 2×RTX A6000约50 h；每scene是否均为该时长、GT生成是否计入未闭合 | per-light representation与component supervision共同优化；不是跨scene model，无法从训练时长推断sample count |
| Dual-Band | stage 1只用roughness `>=0.25`子集，关闭dual-band fusion，principal feature直连final decoder并progressive unlock triplanes，Adam lr `1e-3`、batch32；stage 2用完整数据激活secondary+fusion并训练全框架，lr `1e-4`、batch20 | 4×RTX A6000约12 wall-hours，约48 GPU-hours；两阶段step/epoch与各自time未报告 | stage 1是受限support上的appearance prior，stage 2才是完整方法；不能把final checkpoint当作single-stage scratch identity |
| Neural Prefiltering | voxelization→appearance/visibility两网joint all-LoD训练→per-voxel threshold search | RTX3080；paper 8k–12k convergence，场景约7–40 min | noisy unbiased MC target与scene-specific field；release max30k不是paper formal总步数 |
| CNSR | Pool encoder+generator按factor batches共同训练；static/adaptive partition用forward averaging，paper还要求gradient correction；optional null/auxiliary另引入loss | Adam `1e-4`、16 scenes/batch；paper/checkpoint 1M batches=111 epochs，V100约8.5–10天 | release Pixel/save configs继承`gradient_reg=0`，GQN/U-net为`1.0`；JSON多为2M、Room GQN 20M，均不可回填formal 1M |
| Neural Light Probes | 每场景独立训练一个 image-reconstruction network 1000 epochs；L1、LPIPS与temporal项联合 | 训练约3.9–4 h/scene；GT与probe/lightmap precompute另计 | perceptual与temporal supervision绑定scene view sequence；不是shared cross-scene compiler |
| Active Exploration | online path-traced patches→aggressive MCMC selector→loss-weighted replay；每2000 iterations提升来源分辨率 | 单RTX6000；七个主scene约5–18 h；ArchViz比较的Fig./S标24 h而正文写36 h；总iterations未报告 | selector target随网络更新而变化；proposal、replay、resolution与optimizer state共同构成训练身份，official resume不能恢复全部身份；24/36 h冲突不代作者裁决 |
| NeLT | 每object 200k iterations；四个typed L1 terms joint train；AE baseline只复用network/scene representation并改用uniform data | 2×RTX A6000，约60 h/object；额外OptiX GT/data成本未计 | typed ratio/residual algebra与object-specific训练共同构成identity；没有测试AE active selector与NeLT组合 |
| Superposed DFF | per-scene end-to-end；低分辨率triplane先启用，`α`线性解锁高分辨率；uniform随机scene states | 4×RTX A6000约15 h/scene；6k–8k train states、100 test states | C2F同时改变有效capacity与optimization path；`α` schedule/steps不可得，不能从单次消融证明多seed稳定性 |
| Volumetric Transport 1469 | Den/DenX/Att用作者所称相同optimizer policy各训120k steps；AdamW、exponential decay、batch4 | 训练device/time未报告；结果硬件为RTX5090 | “same hyperparameters”不等于CNN/attention容量或input-cost matched；loss权重和完整schedule缺失 |
| NeLiF | 作者目标与generalization claim指向shared cross-scene model，但正文未明确所有数据是否只训练一个checkpoint；full数据称5300灯/10000场景/1M图，LightFormer公平比较用含全部训练灯具和4000+场景的400K subset、双方35 epochs | 12×RTX4090D；batch、step、LR、训练time均未报告 | full-1M与matched-400K checkpoint identity未说明；不能把35 epochs换算成统一query/reference work，也不能从硬件数推训练成本 |
| Xia control | regular-roughness Gaussian-mixture maximum-likelihood fit；low-roughness covariance/precision Frobenius fit；partial fits初始化更高维polynomial fit | SLSQP；Beckmann proposal model一次fit约23 min，hardware/function evaluations/tolerance/restarts/seeds未报告 | 数值density fitting而非neural lifecycle；runtime proposal仍与exact integrand、external stochastic PDF分离 |

### 5.2 共同训练、交替训练和蒸馏解决不同耦合

- NLB joint fitting让shared decoder与per-BRDF latent共同选择坐标系，但也使 latent identity依赖该decoder；后续只在latent space训练layerer，function-space误差不会自动约束composition。
- MetaLayer交替冻结是为了防止MetaNet与BSDFNet互相追逐；其`X1/X2`分开，但`K1/K2`未报告，不能根据算法框图猜更新比例。
- Biplane的两阶段不是一般“先几何后颜色”的口号，而是作者观察同时更新planes和adapter会改变reflectance distribution后做的约束。
- Mobile VR是teacher/student distillation；NVIDIA RTA的encoder→materialize不是distillation，Taming的多instance淘汰也不是ensemble inference。
- Neural Processes 的 main NP、hypernetwork 与 NICE 是先后冻结的三个模型；不能把NICE的KL和main NP的ELBO合成joint evaluator/sampler objective，也不能用release inference checkpoint补出缺失的main trainer。
- Hierarchical 的“adaptive loss”只有多个互异代码分支，没有formal schedule裁决；Active Exploration则同时改变配置选择、replay与训练分辨率。两者都证明lifecycle会改变有效objective，但不能据此任选一个release branch冒充paper faithful。
- CNSR 的formal factor-partition方法包含gradient replacement，但paper未给strength，release又按generator config分成`0`/`1.0`；不能把Pixel的forward averaging-only、GQN/U-net的full correction或某个JSON步数任选为formal identity。Null penalty只能锁定`β=λ/256`的归一化关系，formal `β=4×10^-4`对应`λ=0.1024`，不等于release default `.01`；且无formal null config可裁决。
- LightFormer 是per-scene、all-modules joint training；它没有披露stage boundary、adaptive sampling或distillation。其component losses与pixel-light attention属于同一end-to-end identity，不能从AE baseline的adaptive sampling或其它scene方法借配置。
- Dual-Band 的 stage 1 同时改变data support、graph与triplane可训练层级，stage 2再降低LR、改变batch并开启完整fusion；scratch-full较差只能支持该paper的定性 staged ablation，不能把收益单独归因于roughness curriculum、progressive unlock或初始化中的任一项。
- Superposed DFF 的C2F同样同时改变active triplane levels和offset auxiliary positional encoding；Table2只比较full、w/o fields与w/o C2F，不能把收益分解成frequency unlock、deformation学习或regularization。NeLT则没有active selector或C2F，只有typed components joint train；三者的lifecycle不能因共同使用hypernetwork/object state而合并。
- NeLiF的full dataset、400K matched subset与35 epochs只给出高层identity；没有公开loss、optimizer、batch或checkpoint对应。它能证明cross-scene data scale是方法的一部分，不能作为某种未披露optimizer/loss的成功先例。
- Neural Light Probes 的 temporal loss、1469 的 auxiliary-rendering supervision 与 Neural Prefiltering 的 noise-to-noise path target 分别依赖连续帧、deterministic scene features 与 stochastic transport estimate；它们都是 scene-level supervision，却不是可交换的 teacher/data lifecycle。

## 6. 已报告的优化失败与较差消融

| 来源 | 证据分类 | 失败/较差尝试 | 观察 | 可安全得出的结论 |
|---|---|---|---|---|
| NBRDF | `author-negative` | autoencoder 直接匹配输入/输出675个weights；独立NBRDF weights直接线性插值 | 前者无法重建原appearance，后者specular过渡不平滑 | 参数向量距离/插值不等于函数或图像空间对应；不证明所有 canonicalized weight-space regularizer 无效 |
| NeuMIP | `author-negative` / `ablation-inferior` | unconstrained 2D offset；完全去掉offset | geometric scalar-depth warp更好；无offset在5/5主材料的MSE/LPIPS均更差 | view-conditioned地址变换是有效prior；不证明其scalar latent是真实height |
| NLB | `author-negative` | representation 使用L2而非L1 | 作者报告L1更保色、少artifact | 只在该raw scalar BRDF protocol下成立；没有完整matched数值或多seed统计 |
| Biplane | `author-negative` | planes与color adapter同时优化 | noticeable reflectance-distribution differences | intensity/structure与color需要受控解耦；结论依赖该adapter parameterization |
| MetaLayer | `author-negative` / `ablation-inferior` | 预测BSDFNet全部weights；one-phase始终joint update | 前者被作者称难收敛；后者RMSE曲线更高且不稳 | generated-state规模与交替冻结都影响条件数；单曲线不是跨seed variance proof |
| RTA 2024 | `author-negative` | 直接优化4K latent，不经过encoder bootstrap | quality可达，但texel残留初始化噪声，训练近翻倍至约10 h | encoder改善高分辨率state的优化覆盖率/效率；不自动证明最终质量更高 |
| Comprehensive | `author-negative` | 对offset function做与color相同的position/direction decomposition；沿用TP exponential LR decay | offset feature模糊且合成失败；展示run约50 epochs后停在较高平台 | 不是所有6D分量都适合同一factorization；schedule结论只有单run、无seed方差 |
| Importance Baking | `author-negative` | marginalized inverse-transform map、hierarchical warping map、SOT-only teacher construction | 插值产生filament、层级binning产生grid artifact、SOT-only在低roughness留下crevice并导致dark region | sample-map连续性与teacher质量是网络之前的load-bearing条件；不是“generic sampler都失败” |
| Hybrid | `author-negative` / `ablation-inferior` | 更复杂Disney analytic core；只用`L_t` | Disney core joint fit数值不稳定；无`L_a`时analytic MAE `.0154→.0366`、full MAE `.0020→.0026` | analytic core必须可优化且显式约束其表示角色；不是“更多物理项一定更好” |
| Taming | `author-negative` / `ablation-inferior` | small single-instance、old Rusinkiewicz、log-mapped L1、ReLU；stable-half/difference-only与部分LeakySmeLU case也较差 | trial质量方差、perfect-reflection输入不连续、低值细节受抑、facet；但stable-only平均不优、LeakySmeLU也非全材质质量胜出 | 贡献是坐标、loss、activation与successive schedule的组合；Table 1/prose trial统计冲突仍保留，不能伪造精确成功率 |
| NMA | `author-negative` / `ablation-inferior` | constant single Principled BRDF直接fit复杂PFMC；固定grid center做1 sample/bin | 前者无法匹配colored/view-dependent lobes；后者漏窄峰且延长训练也不消除盲点 | target family与query estimator都可形成representation floor；这不是loss/sample/lobe文档冲突本身的“失败实验” |
| Mobile VR | `ablation-inferior` | 直接训练8-wide student、不使用完整distillation | test-loss收敛较差；PSNR `35.11→35.12`近乎不变，但full distillation使FLIP `.088→.080`并改善self-shadow visual | 只能确认完整output+feature distillation相对无distillation有益；论文未隔离两项各自贡献，也无多seed variance |
| Improving Angular Parameterization | `ablation-inferior` / `excluded-by-budget` / `material-dependent negative` | 无`cosθ_i`的half/difference、1-level PE、每角4D latent texture；learnable frame未进实验；direct Cartesian再加`h` | 前三者在两材质/tiny budget下未稳定超过direct Cartesian，latent多2 fetch；frame超预算；`+h`在Leather改善、Fabric变差 | 只是两BTF、无seed/无iso-MAC的coordinate evidence；frame是budget exclusion而非author failure，test L1也不支持optimizer结论 |
| Neural Processes | `ablation-inferior` / `author-negative` | max/sum aggregator、低维latent、hyper mainNet、novel-family投影与远距离extrapolation | mean通常较好；7D主版本；hyper quality `56.20→48.98 dB`换小state；novel color与extrapolated energy/semantics会失败 | 这些是set aggregation、capacity与training-support边界；非单调latent维数结果无seed证据，不能改写成optimization variance结论 |
| Hierarchical | `author-negative` / `ablation-inferior` | original/equal-neuron FC decoder及加深加宽FC；w/o Inception、input encoding、gradient loss、remapping；NeuMIP 10×/100×/300× | FC增容仍难表达复杂细节；Figure 4四项leave-one-component-out均有局部退化，其中w/o remapping的back yarn丢失；大NeuMIP仍漏sharp highlight/self-shadow | 这些是spatial hierarchy、encoding、loss/remap与capacity边界；Figure 4无numeric factorial separation，release的offset/disk-map/raw-input或多个loss分支不得倒推为author-negative |
| CNSR | `ablation-inferior` / `author-negative` / `mixed-result` | static/adaptive partition、`β=4×10^-4` null compression、GQN、U-net resolution extrapolation、Pixel w/o G-buffer、auxiliary shadow head、OOD gray wall | partition相对monolithic有quality penalty；R=256在该`β`下约余51 active dims并丢teapot color/shadow；各generator有不同failure mode；aux改shadow但material变差且aggregate metrics不变；OOD颜色失真 | 是scene global-latent/G-buffer的capacity与information-routing证据，不是local evaluator；auxiliary是mixed result，gradient/config gaps不是失败实验 |
| LightFormer | `author-negative` / `ablation-inferior` | 在final hybrid loss加入AE式DSSIM；去half-vector/light-direction/shadow clues或改unified decoder | DSSIM在作者cases中无performance gain；feature/decomposition消融在Gig多数指标变差，其中去shadow clues最明显 | DSSIM结论缺weight/protocol，不能推广；其余只在单scene且部分轴同时变化，不是全面因果证明 |
| Dual-Band | `ablation-inferior` / `author-negative` | full graph+完整dataset从scratch训练；稀疏screen-space reflection、single-bounce query的长specular path | scratch对照出现过锐/不自然/噪声reflection；thin spout highlight与ground mirror→car→environment内容丢失 | staged收益只有单个定性对照，不能证明一般local-minimum因果；后两项是query support/runtime path限制，不是多训可自动修复 |
| Neural Prefiltering | `author-negative` | Pandanus多次重新训练 | 受影响LoD随run变化，部分层transported energy不守恒 | 训练随机性是asset-field方法的真实风险；paper未给repeat raw logs/selection rule |
| Neural Light Probes | `author-negative` | network重建HDR highlights与小高光动画 | Search保留的Bathroom亮点经Final变暗/消失，小高光偶发flicker | 作者归因于log compression与G-buffer信息不足；这是network failure，不是probe/search失败 |
| Active Exploration | `author-negative` / `ablation-inferior` | uniform、loss-only MCMC、w/o position preconditioning、w/o multi-res、uniform+multi-res、Fourier features、128-feature network，以及ArchViz的512 features + resolution enhancement | uniform漏sharp transport；loss-only卡在高误差但不可改善状态；loss-only、w/o position与w/o multi-res均弱于full；Fourier/noisy-data与ArchViz enhancement产生artifact；128 features更快但更模糊 | selector收益来自覆盖与可改善性；不证明generator容量更高，也不允许把scene MCMC直接当local direction sampler |
| NeLT | `ablation-inferior` / `known-limitation` | w/o GI decomposition；direct分支只为diffuse预测background shadow；novel Room→Indoor scene | full typed decomposition在Table6优于统一形式；specular shadow与high-frequency detail失败，novel scene质量下降 | 支持typed image-transfer algebra在该scene task内有益；不能外推为local component head必胜，且AE active selector未参与该对照 |
| Superposed DFF | `ablation-inferior` / `author-negative` | w/o feature fields、w/o C2F、static field，即使static延长训练 | Table2/figures均劣于full；static variant仍难恢复动态shadow | 支持field/deformation/C2F组合在作者protocol内有益；无repeat/iso-memory，不能把收益只归因单轴或宣称跨seed稳定 |
| NeLiF | `ablation-inferior` / `known-limitation` | direct shadow与调整后的LightFormer shadow baseline；monolithic neural radiance捕获high-frequency lighting pattern | kernel-based shadow定性更稳；正文承认spectral limitation并建议multi-scale encoding | shadow比较无matched params/time/metric；spectral例子在不可得supplemental，不能量化或改写成已验证multi-scale修复 |
| Volumetric Transport 1469 | `ablation-inferior` | leave-one-feature-out | Fig.9中detail/shading发生退化 | 只证明该单例 qualitative feature ablation；Att相对DenX是quality–temporal stability–time tradeoff，不应误分为失败配置 |
| Xia control | `author-negative` / `ablation-inferior` / `robustness-repair` | forward sequential、long-path pair/multiple product、high-roughness multiple product；non-positive precision和`α<0.1`单一regular fit | adjacent product未被proposal覆盖或local Gaussian近似随chain length/roughness退化；作者用eigenvalue `ε`修复与specular-anchored分段 | 这些是analytic proposal/fit边界，不是neural optimization history；`ε`未报告且internal/external PDF不得混同 |

“论文没有采用某种方法”不等于负结果。表中只登记作者明确描述的 `author-negative`、较差消融或已定位的表示限制；paper↔code/config gap 仍留在 §2、§4–5 的 identity 列，不把“无 code 可裁决”本身改写成失败尝试。Belcour、Guo 与 Xia 的 sampling/MIS/analytic-fit 负结果属于 estimator control，不是 neural optimization history。

## 7. Optimization variance 的来源分解

### 7.1 表达坐标造成的目标不连续

Taming把classic Rusinkiewicz在perfect-reflection附近的`φ_h`不定性识别为compact网络的输入不连续，并使用shortest-arc stable half/difference；同时保留direct directions以应对normal mapping旋转。这个机制改变目标函数的平滑性，不改变query数量。它应与固定frame/raw directions、classic half/difference做同budget matched control，而不是和更大网络混在一起。

Improving Angular Parameterization 在两个BTF、tiny input budget下又观察到：direct Cartesian显式暴露`ω_i.z=cosθ_i`时通常优于不含该线索的half/difference，而direct再加`h`的收益随材质改变。这只是coordinate/capacity clue：D4/D6/D9并非iso-MAC，target measure、training loss、query recipe与seed均未报告，不能补成Taming的optimization-variance证据。

### 7.2 HDR target 与稀有峰值造成的梯度失衡

NBRDF、NeuMIP、MetaLayer、RTA、Taming、Hybrid 与 1469 都采用不同压缩/双域 objective，说明高动态范围是普遍困难；但各自是否乘 cosine、是否保留 linear-domain term、是否用 L1/L2、以及监督的是 local function 还是 full image，决定了峰值和 tail 的权重。Neural Light Probes 还把高光损失明确关联到未给公式的 HDR logarithm compression 与 G-buffer 信息不足。当前项目不能只比较 aggregate RGB error，还应预冻结 peak、tail、grazing 和 energy strata；同时不能拿 image-space HDR failure 直接证明 local-BRDF 某个 transform 失败。

Neural Processes 的四次`log1p`与latent ELBO绑定，Hierarchical 的 fourth-root又与spatial gradient项绑定；后者还存在 prose/code 对 printed Eq.(2) 的公式冲突。它们扩展了 power transform 的实证先例，却不能补成“指数越小越稳定”的单调规律。移植到当前项目时仍必须把 transform、norm、gradient/auxiliary term和output measure拆开，以matched control分别裁决。

CNSR虽同样使用`log1p`与DSSIM，但其误差是scene-image HDR reconstruction，又与factor partition、gradient correction和optional null/auxiliary loss耦合。它的失败边界是partition quality penalty、global/local information routing与OOD scene recipe，不能用来支持local BRDF target transform。

### 7.3 Shared state 与网络的可交换性

joint latent/network fitting存在尺度、旋转和置换等非唯一性。NLB的layerer只见optimized latent，MetaLayer生成partial weights，RTA在encoder后直接优化texel latent；如果表示身份没有额外约束，不同seed得到函数相近但latent几何不同是合理现象。对compiler评估必须同时测最终function-space质量与latent/program workflow robustness，不能只测训练loss。

### 7.4 Batch统计与实例淘汰

Taming 保持 `instances×per-instance batch=65,536` 次 sample–network evaluations/step，并逐步淘汰实例；但 formal/shared-data 路径把同一 query batch broadcast 给所有 active instances，因此 unique reference queries/step 是 `1,024→4,096→16,384→65,536`，并不恒定。按理想四段各25k steps派生，100k steps共6.5536B次network evaluations、约2.176B条unique reference queries；`1@64k` baseline则两者都是6.5536B。也就是说作者固定的是 evaluator compute，不是 reference-generation budget。若迁移到当前训练，必须分别登记 network evaluations、unique reference queries、optimizer steps、并行state bytes与selection rule；否则“同query预算”或“多seed更稳”都会把两种成本混名。

### 7.5 Noisy reference 与无偏性

Neural Prefiltering 正文以 noise-to-noise relative L2 拟合 raw-throughput 空间的 unbiased MC estimate；Arcade release 却先对单次 noisy estimate 做 `log1p`，非线性后不再自动保有 raw-throughput 空间的无偏性保证，二者必须分 identity。当前 LayerStack reference 也是在线 Monte Carlo，但 local scattering query 与 voxel aggregate throughput 的噪声结构不同。能迁移的是“固定每 query reference estimator 及其 SE/预算并避免离线 batch 选择偏差”，不能直接迁移其 relative-L2、release `log1p` 或 scene training steps。

### 7.6 Scene-image supervision 的可辨识性与时间轴

CNSR用三张scene observations生成global latent，再与novel-view G-buffer合成HDR image。其factor partition可以分配lighting/geometry/material信息，但partitioning相对monolithic有明确quality penalty，unknown gray wall也发生颜色外推失败。该方法无history/reprojection/temporal loss或formal temporal metric；per-pixel generator的resolution behavior与qualitative稳定性不能升级为时序保证。

Neural Light Probes 每场景只用 20 张 GT views，并把 LPIPS 与 temporal warp loss加入 image reconstruction；1469 从 22,000 张 cloud/light images 学习，正式 Att 不输入 stochastic 4-spp radiance，但相对 DenX 以略低的静态图像指标换来明显更好的 temporal metric 与更低总时间。前者的 temporal稳定性依赖 history/motion vectors，后者声称单帧 history-free；二者不能共享“temporal loss有效/无效”的结论。

LightFormer 的20,000 configs/scene覆盖作者定义的camera、light、material与animation变化，并由component L1+shadow VGG联合监督；公开architecture没有history/reprojection，作者的temporally-consistent结论来自visual evidence。VPL sample复用、RNG与buffer update未报告，也没有temporal metric，因此不能把“清晰输入”直接升级成冻结的无闪烁保证，或与Neural Light Probes的history-based temporal loss做数值比较。

Dual-Band 也按scene训练，但没有sequence training、history或reprojection；其两阶段先在roughness `>=0.25`子集学principal appearance，再启用single-bounce mirror-hit secondary feature与screen-space fusion。作者把scratch-full的定性退化解释为sub-optimal minima，却没有stage长度、loss curve或重复seed。因而它支持“完整graph的初始化/curriculum值得matched复核”，不支持把staged lifecycle写成已证明降低跨seed variance；缺少dynamic-light sampling与temporal metric也不能由定性视频补齐。

1469 的 feature-inversion 还显示：all-feature match 仍可对应不同 density 与 novel view，说明 auxiliary buffers 对 scene geometry 不具 injectivity。Neural Light Probes 的 per-scene 20-view supervision也未给 camera/split manifest。对 scene transport compiler，最终 image loss 很低不等于 scene state或novel-view transport被唯一约束；必须单独冻结 view/lighting/volume disjoint split 和 temporal protocol。

### 7.7 Active selection 会引入覆盖收益，也会引入非平稳与记账风险

Active Exploration把训练目标写成当前`Loss·||ΔAdam||`，并用持续变化的网络状态驱动MCMC与replay；因此它找到的是“此刻仍能推动optimizer的scene/patch配置”，不是固定的物理target density或directional sampler。论文显示loss-only selector会困在高误差但当前模型无法改善的mirror state，说明难例分数与可学习价值不可混名。与此同时，official code 的fresh batch（16–32 patches）、small-step proposal family、replay容量/记账和resume状态又与正文或内部记账存在差异。若迁移到online reference query，selector候选状态必须先能映射到小而可搜索的native training domain，且G1/G2/G2s的held-out assets/states不得进入候选池。实验必须分别记录新reference queries、replay evaluations、proposal transitions、候选target所需backward/optimizer evaluations、state bytes与wall time，并冻结no-resume或完整恢复selector/replay/optimizer/RNG的策略。否则所谓效率收益可能只是重复样本、测试泄漏、额外selection compute或不可恢复的history state。

NeLT和Superposed DFF都把AE作为正式baseline，却明确改用uniform sampling，仅复用其network/scene representation；作者都把adaptive training组合留作future work。因此两篇表格不能作为“完整AE active policy输给object-oriented field”的证据，反而构成一个干净边界：representation comparison与query-selection comparison尚未正交完成。NeLiF又因per-scene training成本排除AE、NeLT、Superposed DFF等方法的数值comparison，不能把排除项写成AE quality失败。

## 8. 对当前 NeuralShading 的执行约束 `[N/I]`

### 8.1 每个候选必须声明完整 objective identity

候选注册不应只写`loss=l1`，至少需要：

- canonical output是bare linear `f`，以及loss前是否乘`cosθ_i`；
- target/output transform的逐式定义、offset/epsilon、inverse与clamp位置；
- RGB/channel/query reduction顺序；
- query distribution、strata权重、reference spp/SE和online RNG recipe；
- optimizer全部超参数、initialization、seed集合、checkpoint selection；
- module freeze/detach、stage boundary与每stage总reference queries；
- formal candidate与任何paper/code adaptation的独立identity。

这既是复现要求，也是解释optimization variance的最小证据。缺少任一项时只能标`author-underspecified`，不能用常见default补全。

### 8.2 第一轮 matched controls

建议把训练轴和表示轴分开，按以下最小顺序冻结：

1. 同一compact evaluator、同一source/query/seed预算，比较 linear L1、`log1p` L1、Taming power-map L1；runtime output都还原为相同bare `f`。若另加fourth-root，必须预先定义本项目的transform/inverse、zero/epsilon/clamp，并作独立adaptation identity；不以Hierarchical冲突的Eq.(2)命名为faithful reproduction。
2. 固定最佳target transform，比较 raw directions、classic half/difference、stable direct+half/difference；网络shape、initialization和queries不变。Improving Angular Parameterization只作feature clue；不直接搬它的D4/D6/D9或未报告的loss，首层MAC/参数与texture reads必须matched。
3. 固定坐标后，比较单实例batch 65,536 与 Taming式总sample–network evaluations相同的successive multi-instance schedule；query在实例间broadcast，并把unique reference queries、额外state bytes、selection与wall time分别纳入成本，不声称reference-query matched。
4. 对analytic-core residual增加/移除`L_a`，保持完整output loss、MAC与state budget一致；检查analytic parameter fidelity是否真正换来G1/G2/G2s收益。
5. 对compiler候选同时报告function-space loss与workflow robustness W，避免latent/program内部距离替代真实外观误差。
6. 把Active Exploration式error/Adam-step selector限定为training-only query policy：候选池严格限于train split，test/G2/G2s holdout始终固定且不参与选择。uniform、replay-only与active selector分别做iso-new-reference-work与iso-wall-time对照，并单列candidate backward/optimizer target、replay、proposal transitions、state bytes与resume/history成本。

这些是待进入 `reproducible-hypotheses.md` 的实验方向，不是已经批准执行的训练run；正式运行前仍需按项目experiment framework冻结config、seed、预算和selection rule。

### 8.3 不能直接迁移的部分

- BTF整张same-angle slice、voxel noise-to-noise和当前local scattering online queries的统计单位不同；不能共享batch-size数字。
- CNSR的scene-observation global latent、Neural Light Probes 的per-scene image loss/temporal history、LightFormer的per-light component attention、Dual-Band的object-field+screen-space fusion与1469的cloud-image auxiliary features都不输出local `evaluate/sample/pdf`；它们只能进入后续scene-transport研究，不是当前bare-`f` evaluator的loss替代项。
- NBRDF/Biplane/NLB公开代码中存在cosine或response transform correspondence gap；不能把release loss直接命名为paper faithful。
- MetaLayer/NMA只覆盖固定source family/topology；其parameter→program训练不能证明跨任意native material family泛化。
- Taming formal实验是fixed-material optimization stability，不是G2/G2s unseen material generalization结果。
- Improving Angular Parameterization只在两个BTF上给出test L1/FLIP coordinate diagnostic，训练objective、measure、query distribution与cost均未闭合；不能以其替换当前loss或宣称Cartesian对所有材质普遍最优。
- Neural Processes 的主结果是151个参与训练材质的asset compression；其NICE是post-trained density，不是与evaluator匹配并联合训练的`sample/pdf`证明。Hierarchical依赖整幅spatial buffer的卷积邻域，其training/inference work unit都不能直接作为随机访问单query evaluator形态。
- Active Exploration的MCMC只选择scene training configs；它不输出runtime material sampler/PDF，且selector target随optimizer变化。迁移时只能作为小而可搜索的training-state/query-allocation实验，不能替换冻结的evaluation distribution，也不能访问任何held-out source state。
- Xia的SLSQP fit只生成analytic internal proposal，不是neural training；pair/multiple-product的internal integration directions、external material-direction `sample/pdf`、exact integrand 与 projected/solid-angle measure必须分开。它只能影响reference/proposal control，不能解释neural optimizer variance。
- Neural Prefiltering 的paper raw-throughput objective与Arcade release `log1p` objective必须作为两个identity；LightFormer缺逐项log/VGG preprocessing与steps/schedule，Dual-Band缺mapped-output/inverse lifecycle与stage length，1469缺loss权重，Neural Light Probes缺log公式，均不能用常见default补齐。
- 论文训练时长跨GPU、数据、reference和query count不可比，不进入候选排名或hard gate。

## 9. 可证伪的综合假设 `[I]`

| Hypothesis | Evidence basis | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|
| O1：power-map L1比linear/log1p L1降低compact evaluator seed variance | Taming formal ablation；Hierarchical的fourth-root prose/code只提供指数先例且与printed Eq.(2)冲突；其它HDR压缩只是跨measure背景 | linear、log1p、Taming cube-root；可选fourth-root必须另立project-authored identity；同bare-`f` ABI、network/query/steps/seeds | init、optimizer/schedule、coordinates、output parameterization、reference recipe、selection；每个transform的offset/epsilon/inverse/clamp | seed success rate、G1；参数式family才测G2/G2s；peak/tail/energy CI、train与runtime | local evaluator candidate，保持bare-`f` ABI | success CI与最终quality–time–memory Pareto无改善，或改善只来自牺牲peak/tail/energy |
| O2：stable direct+half/difference的收益独立于更大batch/网络 | Taming的singularity分析；Angular Parameterization的两BTF/tiny-budget clue不提供variance证据 | raw/direct Cartesian、classic、stable坐标；同shape/schedule并使首层MAC/参数、texture reads matched | target transform、queries、init、MAC、latent、output measure | convergence曲线、seed variance、boundary/grazing/peak slice error、runtime | local evaluator candidate | boundary/overall error与variance均无改善，或runtime成本使其被支配 |
| O3：successive multi-instance在固定总sample–network evaluations下优于单实例 | Taming `64→16→4→1`、固定65,536 network evaluations/step、shared query batch | single `1×65,536` vs successive；同100k steps与6.5536B network evaluations | optimizer/LR、init分布、selection、shared-query语义、network-eval总量；unique reference queries单独记账 | best/median/worst seed、selection bias、unique reference queries、bytes/time | training-only schedule | 同network-eval预算下无稳定收益，或reference/state/wall-time计入后Pareto更差 |
| O4：linear residual term可保护高亮而不破坏tail | NeuMIP `comb2`；MetaLayer reverse-Huber动机 | compressed-only vs compressed+linear项 | transform、network、queries、weight预冻结 | peak/tail/grazing error、energy、CI | local evaluator training objective | aggregate改善但peak/tail任一预冻结stratum显著恶化，或无Pareto收益 |
| O5：function-space supervision优于latent/weight-space距离用于compiler | NBRDF weight-loss author-negative；NLB latent-only layerer边界 | latent loss vs frozen-evaluator function loss vs joint | compiler shape、training queries、source split、evaluator | G2/G2s function error、latent stability、workflow W、runtime | compiler training diagnostic | function supervision无泛化/稳健性收益或引入不可接受reference成本 |
| O6：analytic auxiliary loss稳定physical core且保留residual容量 | Hybrid `L_a+L_t` ablation；Belcour/Hybrid analytic prior边界 | full loss only vs auxiliary+full；同core/residual预算 | core、latent、queries、steps、weights | analytic parameter error、full BRDF error、energy/reciprocity、CI | local evaluator candidate | analytic fidelity或完整Pareto无改善，或hard strata被core bias锁死 |
| O7：当候选状态可定义且严格限于train split时，error×optimizer-step selector能改善困难strata覆盖 | Active Exploration相对uniform及loss-only消融；迁移属于本项目假设，不是local-direction直接证据 | uniform vs replay-only vs active；分别做iso-new-reference-work与iso-wall-time对照，单列candidate backward/optimizer、replay和transition工作 | train-only candidate pool；held-out distribution不参与selection；proposal/perturbation、replay size、resolution、seeds、reweight、no-resume或完整state恢复 | G1；参数式family才在不泄漏前提下测G2/G2s；peak/grazing/rare-lobe CI、coverage、reference/selection-time/state成本 | training-only query policy | held-out Pareto/覆盖无改善，收益只在selector访问分布成立，或selection/history/resume/state成本使其被支配 |

## 10. 当前完成状态

- [x] 当前28篇`evidence-reviewed`报告的target transform、loss、query recipe与training lifecycle已分层比较；25篇learned reports与3篇non-neural controls的边界已明示；
- [x] paper formal、supplemental、release default/example和未报告项没有混为同一配置；
- [x] 作者负结果与本项目推断分开；
- [x] 没有用跨硬件训练时长做排名或hard gate；
- [x] Weier、Angular Parameterization、Neural Processes、Hierarchical、CNSR、Neural Light Probes、Active Exploration、LightFormer、Dual-Band、匿名稿1469与Xia control已按各自证据边界纳入；
- [x] NeLiF、NeLT与Superposed DFF已从正式正文恢复；不可得supplemental/code字段仍不由相邻方法补猜；
- [x] 本轮三份来源恢复后的独立 cross-document evidence review 已完成；NeLT、Superposed DFF、NeLiF 的 objective、sampling 与未报告边界均已回链个体报告复核。

## 11. Evidence review

- author update：`/root`
- reviewer：`/root/nelt_full_report`
- date：2026-08-29
- sources rechecked：28个唯一个体报告链接及其`report_status`；[NeLT §§6–7,9–11](../papers/zheng-2023-nelt.md)、[Superposed DFF §§6–7,9–11](../papers/zheng-2024-superposed-deformable-feature-fields.md)、[NeLiF §§6–11](../papers/sheng-2025-nelif.md)的独立review稿。
- findings closed：28篇计数确认为25篇含learned component的报告+3篇non-neural controls；NeLT锁定mean-expected-power归一、逐通道`log1p`、四项typed L1与`AE (uniform)`；Superposed DFF锁定final-image `log1p`、等权`L1+structural dissimilarity`、uniform scene states与C2F；NeLiF只保留已公开global-max normalization、400K/35 epochs与12×4090D，并将shared checkpoint、loss、optimizer维持未报告。
- remaining gaps：三个个体报告登记的supplemental/code、exact topology、NeLT ratio稳定化、Superposed DFF `α` schedule、NeLiF full-1M↔matched-400K checkpoint identity和runtime账本继续保留。
- status：`evidence-reviewed`
