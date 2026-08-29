# Sampling 与 Integration：跨论文证据综合

## 1. 证据边界

本文只把个体报告中已经达到 `evidence-reviewed` 的内容作为强证据；局部表选择有明确 sampling/integration 贡献或边界的论文，不声称穷举所有 local appearance 报告。Guo 2018 position-free random walk 与 Belcour 2018 statistical layered approximation分别作为 stochastic reference 和 analytic control；当前已复核的 scene/volume 报告进入独立矩阵，不再以“第二波尚未复核”为由整体排除。

本报告统一使用四类 query semantics；同一论文可以跨类，但每一项调用都必须单独登记：

| 记号 | Query | 返回量 | 不能混入的对象 |
|---|---|---|---|
| `L` local evaluator | 固定材质/状态与`(wo,wi)`查询局部方向函数 | bare `f`、`f cos`、BTF response等论文原生 measure | scene radiance、visibility、内部随机路径不能仅因最终用于着色就称为`evaluate()` |
| `R` reference-internal estimator/proposal | 固定外部`(wo,wi)`后，对layer/volume内部路径变量采样并积分 | stochastic `f` estimate及其内部path contribution/PDF | 它不是场景integrator向材质请求新`wi`的外部`sample()/pdf()` |
| `S` external material-direction sampler | integrator给定材质状态与`wo`，生成`wi`并查询实际proposal密度 | direction、solid-angle `q(wi\|wo)`，可另返回`f\|cos\|/q` | 独立MIS score、预烘weight、解析lobe外形都不自动等于实际proposal PDF |
| `G` scene integration/visibility | 查询ray、probe、VPL/RSM、G-buffer、体素segment或整幅auxiliary buffers | visibility、aggregate throughput、scene radiance/image | 这些量已经耦合geometry/light/visibility，不能注册为local材质`L/S` |

`proposal`只表示生成样本的密度近似，不要求采用evaluator的表示词汇；但一个可注册的`S`必须满足support、normalization、sample↔pdf identity与有限weight。评价方差时还必须固定evaluator、integrator、scene/light、SPP或time budget、seed与硬件/backend。各论文自己的FPS、rays/s、batch latency、SPP、RMSE/PSNR只在各自protocol内成立，本报告不把它们横向拼成排名。

Xia Gaussian Product Sampling在本次综合审查开始时仍是`report-draft`，因此初稿只把它当弱依赖；独立review随后在共享工作树中完成，front matter、review block与完成检查现已一致为`evidence-reviewed`。下文据此只纳入review已闭合的internal-estimator语义，不保留审查前的draft结论。[Xia报告](../papers/xia-2020-gaussian-product-sampling.md)

## 2. 方法总表

### 2.1 Local evaluator、reference estimator 与外部 sampler

| 方法 | 类别 | `L` evaluator / `R` internal estimator | `S` external proposal/sample | `pdf()`证据与边界 |
|---|---|---|---|---|
| NBRDF 2021 | `L+S` | per-material `6→21→21→3`，论文把输出称为RGB BRDF；训练loss作用于`f cos`，故移植时仍需独立审计runtime bare-`f` correspondence | 32D embedding经`32→8→2`预测monochrome Blinn–Phong参数；另测GGX labels | 解析proxy可构造matching solid-angle PDF，但predictor code未公开；官方Mitsuba plugin实际是NBRDF tabular sampler，不是论文Phong proxy |
| NeuMIP 2021 | `L+S(control)` | spatial/footprint MBTF evaluator | indirect path只用cosine-weighted hemisphere | cosine PDF可解析，但不匹配尖锐material appearance；light sample与BRDF sample均需另调evaluator |
| Neural Layered BRDFs 2022 | `L+S` | shared bare-BRDF evaluator | offline network把BRDF latent压成projected-half-vector Gaussian + Lambertian两参数proposal；最终参数在方向网格上平均，runtime不执行sampler MLP | supplemental给analytic mixture；input encoding、参数约束、normalization与formal code缺失，不能直接当实现oracle |
| Neural Biplane 2023 | `L+S(control)` | BTF/ABRDF response evaluator；论文与release的cosine位置需按各自合同处理 | complex lighting中使用Lambertian proposal并与light sampling做MIS | Lambertian PDF可解析；作者把Gaussian/Blinn–Phong proposal列为未来扩展，没有matched BTF sampler |
| MetaLayer 2023 | `L+S` | MetaNet生成state，BSDFNet给确定性layered BSDF estimate | 外接Belcour-style `R/TRT`或`R/TRT/TT` lobe mixture，按估计energy选lobe并采visible normal | 论文称按microfacet mixture计算；sampler不是BSDFNet输出，无code、normalization、sample↔pdf或独立variance实验，作者明确它并非arbitrary layered material的optimal sampler |
| NVIDIA RTA 2024 | `L+S` | compact learned functional evaluator；P/S functional path是cosine-weighted BRDF measure，当前项目bare-`f` adapter属于独立correspondence | latent + fixed view direction经`11→32→32→32→9`输出tilted cosine diffuse与non-centered anisotropic GGX mixture | supplemental给component remap、sample与完整mixture PDF伪代码；params可在同一hit复用；2024 KL方向、normalization、invalid handling与estimator未报告 |
| Neural Processes BRDF 2021 | `L+S` | 7D latent-conditioned isotropic RGB bare-BRDF decoder；post-trained hypernetwork可生成per-material compact decoder | two-coupling-layer conditional NICE从uniform 2D变量生成normalized half-vector，target正比于`luminance(f) cosθ_i` | flow有change-of-variables/log-Jacobian设计，但official repo没有NICE code/weights，论文未完整展开half-vector到incident-solid-angle Jacobian；16-spp图只支持其论文protocol内的qualitative/equal-sample结论 |
| BSDF Importance Baking 2023 | `L+tuple` | 可选独立evaluator输出`f cos` | neural map从材质/view/uniform 2D变量直接回归`wi`与RGB预烘weight | 另训的positive scalar network只作MIS query；未证明归一、未证明等于sample map Jacobian density，因此不能登记为matched `S` |
| Hybrid Neural-Microfacet 2026 | `L+S` | bare `f=f_c+f_g f_a` | formal论文以cosine与fitted-GGX lobe做BRDF-direction MIS | mixture selection probability、heuristic和formal host实现未公开；公开`.brdf`仅有GGX `sample/pdf`，Web demo则是GGX–environment MIS，均不能替代formal cosine–GGX correspondence |
| Neural Material Adapter 2026 | `L+S claim` | MLP按view生成两组Principled参数与blend，再由解析target评估 | 作者声明target具built-in/accurate importance sampling | blend sampler/mixture PDF、reverse PDF、MIS event与delta处理公式和代码不可得，只能保留architecture-level声明 |
| Belcour 2018 | `L+S control` | 用energy/mean/variance近似layered response并重建GGX lobe mixture | 按平均RGB energy选lobe、visible-normal sampling | opaque Forward/Symmetric路径汇总所有overlap lobe densities；Eq.40给balance contribution。transmission plugin仍需独立验证，不能由opaque correspondence代替 |
| Guo 2018 | `R+S` | 固定外部`(wo,wi)`时，uni/bidirectional position-free random walk对surface/volume内部路径积分，返回stochastic layered-BSDF estimate | 另有从外部方向出发的forward layer random walk，可生成退出方向/sample weight | external directional PDF本身又是path-generation probability integral，exact/short-path approximate估计必须与stochastic BSDF使用独立随机数；内部evaluation proposal与外部`S`不是同一query |
| Xia 2020 | `R` | 固定外部`(wo,wi)`后，以Gaussian slice product近似相邻两个BSDF因子，pair-product采一个内部方向；特定三因子component再以四维Gaussian联合采两个内部方向 | 不替换Guo-style external forward sample；也不替换external stochastic PDF | Gaussian只进入internal proposal/PDF，最终仍评估原始layered contribution；正式模型限isotropic surface slices，multiple-product不等于任意长path的外部sampler |

证据：[NBRDF §§4–8,10–13](../papers/2021-neural-brdf-representation-importance-sampling.md)、[NeuMIP §§4–8,12–13](../papers/kuznetsov-2021-neumip.md)、[NLB §§4–8,10–13](../papers/fan-2022-neural-layered-brdfs.md)、[Biplane §§4–8,10–13](../papers/fan-2023-neural-biplane-btf.md)、[MetaLayer §§4–8,10–13](../papers/2023-metalayer.md)、[RTA §§4–8,10–13](../papers/zeltner-2024-real-time-neural-appearance-models.md)、[Neural Processes BRDF §§4–8,12–13](../papers/zheng-2021-neural-process-brdfs.md)、[Importance Baking §§4–8,10–13](../papers/bai-2023-bsdf-importance-baking.md)、[Hybrid §§4–8,10–13](../papers/2026-hybrid-neural-microfacet-brdf.md)、[NMA §§4–8,10–13](../papers/2026-neural-material-adapter.md)、[Belcour §§4–8,10–13](../papers/belcour-2018-efficient-rendering-layered-materials.md)、[Guo §§4–8,10–13](../papers/guo-2018-position-free-layered-bsdfs.md)、[Xia §§4–8,10–13](../papers/xia-2020-gaussian-product-sampling.md)。

### 2.2 Scene integration / visibility矩阵

下表中的方法全部属于`G`；“sample”“query”“visibility”只描述scene pipeline内部工作，不表示存在local material `S`。

| 方法 | Scene query / integration入口 | Visibility处理 | 输出与local边界 |
|---|---|---|---|
| Active Exploration 2022 | MCMC只在训练期选择camera/object/material/light的scene configurations与patch；runtime PixelGenerator读取scene vector和first-hit G-buffer | visibility与multi-bounce transport隐式进入per-scene weights，没有独立visibility API | 输出pixel/整图outgoing radiance；active selector是training query policy，不是runtime evaluator或sampler |
| NeLT 2023 | foreground/background/light samples先编码成global representations；runtime按object insertion执行direct/indirect neural texture query和radiance composition | raster G-buffer给primary visibility；background direct以multiplicative ratio表达插入对象的shadow，indirect以additive residual表达transport change | transfer component只在对应mask分支是radiance/ratio/residual；无local direction sample/PDF，light importance samples只是representation输入 |
| Superposed DFF 2024 | 对每个pixel查询所有object/light deformable fields，field outputs相加后由final decoder恢复scene radiance | primary visibility来自G-buffer；dynamic shadow/caustic/indirect visibility隐式存入conditioned fields，没有显式visibility query | `F_i`是learned feature而非物理path contribution；两个sum都不是MIS estimator，也没有material-direction proposal |
| Efficient Light Probes 2022 | runtime读取diffuse lightmap与8个nearest glossy probes，做bounded reflection search、probe blending、temporal reprojection和full-frame reconstruction | probe候选有viewport/material-ID/visibility test；历史warp另有未完全披露的validity边界 | 输出1080p scene image；probe lookup/search不是material-direction proposal |
| Neural Prefiltering 2023 | voxel path tracer反复查询aggregate appearance `φ_V(wo,wi)`；碰撞后正式实现均匀采sphere direction继续path | 独立endpoint-pair visibility network决定segment是否通过，并用于traversal/RR | `φ_V`已聚合voxel内geometry/material/visibility/multiple scattering，不是bare local `f`；没有matched learned appearance sampler |
| LightFormer 2024 | emitter importance samples形成direct VPL；RSM texels形成indirect VPL；per-light observations经pixel-light attention后统一decode | RSM depth/shadow clue显式提供direct visibility；indirect VPL到shading point没有显式visibility test | 输出direct shading、direct shadow、indirect shading并合成scene RGB；VPL sampling probability不转换成path-tracer `pdf(wi)` |
| Dual-Band 2025 | 每pixel沿mirror correspondence追一条secondary ray，principal/secondary scene fields经screen-space learned kernels近似rough/glossy integration | 一次ray hit与zero branch隐式表达部分occlusion；无可查询visibility operator | 输出scene-dependent `L(x,wo)`；deterministic mirror direction没有proposal PDF |
| Volumetric 1469 2026 | primary ray生成first-event/optical-depth/crossing/direct/environment等auxiliary maps，再由attention U-Net回归full transport | global visibility/multiple scattering交给auxiliary operators与image-wide attention；无独立path sampler | 输出scattered-radiance image；匿名稿没有local phase/BSDF evaluator、sample或PDF |
| NeLiF 2025 | luminaire observations先生成spherical lighting field；runtime由G-buffer query direct、RSM/VPL query indirect、五级hard-shadow clues query shadow，再合成3DGS luminaire appearance | shadow mapping提供显式hard visibility clue，kernel network近似soft shadow；indirect visibility仍隐式进入RSM/VPL与decoder | 输出scene image components；VPL、shadow-map texel与field lookup均没有转换成path-tracer `pdf(wi)`，也不提供local `evaluate/sample/pdf` |
| Mobile VR Neural Materials 2026（system=`G`，内部BTF=`L`） | 对每盏point light在object texture space完整执行BTF shading，再将per-eye radiance texture复用未来`N`帧 | 只覆盖其direct point-light raster/texture-space path；没有environment/scene visibility积分接口 | BTF query本身属于local appearance，但系统产物是已乘当前light的radiance texture；无`sample()/pdf()`，多灯按每灯重复inference |

证据：[Active Exploration §§1,4–8,12–13](../papers/diolatzis-2022-active-exploration-neural-gi.md)、[NeLT §§1,4–8,12–13](../papers/zheng-2023-nelt.md)、[Superposed DFF §§1,4–8,12–13](../papers/zheng-2024-superposed-deformable-feature-fields.md)、[Efficient Light Probes §§1,4–8,12–13](../papers/guo-2022-neural-light-probes.md)、[Neural Prefiltering §§1,4–8,12–13](../papers/weier-2023-neural-prefiltering-lod.md)、[LightFormer §§1,4–8,12–13](../papers/ren-2024-lightformer.md)、[Dual-Band §§1,4–8,12–13](../papers/mo-2025-dual-band-neural-gi.md)、[Volumetric 1469 §§1,4–8,12–13](../papers/1469-2026-volumetric-light-transport-inference.md)、[NeLiF §§1,4–8,12–13](../papers/sheng-2025-nelif.md)、[Mobile VR §§1,4–8,12–13](../papers/xu-2026-real-time-neural-materials-mobile-vr.md)。这些论文的frame latency、resolution、GPU/backend、training corpus、GT与metrics均不同；矩阵只比较职责/接口，不比较质量或速度名次。

## 3. Proposal不必等于evaluator vocabulary

### 3.1 超小解析proxy

NBRDF用完整neural evaluator重建measured BRDF，但sampling head只预测Blinn–Phong的两个sampling-relevant参数：roughness/shape与specular–diffuse相对权重属于monochrome proposal，不拟合七个RGB reflectance参数。论文实验显示predicted analytic curves接近per-material fitted curves并优于uniform；作者还指出更准确的analytic BRDF fit不一定给更好proposal，因为它可能更精确追specular却漏掉sheen能量。[NBRDF §§3.3,4.4]

这支持的是“proposal可以是便宜的有support近似”，不是“材质应被编译成Blinn–Phong”。公开repo没有Figure 4 predictor，Mitsuba plugin改用对NBRDF建立tabular sampler，因此不能把plugin当作论文proxy oracle。[NBRDF §§8,11]

### 3.2 Latent→analytic mixture

NLB与NVIDIA都从latent生成解析mixture，但含义不同：

- NLB的sampling network离线把BRDF latent压成Gaussian+Lambertian proposal；runtime只保留解析参数，网络不在hot path。
- NVIDIA sampler MLP在每个hit从spatial latent与`wo`解出9参数；它能随位置/view改变，并在同一hit缓存供sample/PDF复用。

两者都让evaluator保持direct neural function，但sampler容量和调用频率不同。比较时必须把离线预计算、per-hit decode、state bytes和多次MIS PDF query分别计账。

### 3.3 Layer-path lobes

MetaLayer与Belcour按R/TRT/TT等主要layer paths构造lobes。Belcour的lobe state由adding-doubling统计算子直接产生；MetaLayer的neural evaluator并不输出这些sampler参数，只是另接Belcour-inspired proposal，而且没有说明如何从neural BSDF state稳定估计完整lobe energy/roughness。[MetaLayer §8.3; Belcour §§5–6]

因此“同样是R/TRT/TT”不能证明proposal与neural evaluator匹配。它们更适合强analytic control，供learned sampler在相同source/evaluator下比较。

### 3.4 Latent-conditioned flow不是analytic closure

Neural Processes BRDF的NICE sampler不把latent解成Phong/GGX参数，而是以`z7+θ_o`条件化两层coupling flow，把uniform 2D变量映射到normalized half-vector坐标，并累计log-Jacobian。它说明proposal也可以保留非解析、多峰容量，而evaluator仍独立输出bare BRDF。[Neural Processes BRDF §§5,7–8]

这仍不是可直接复制的`sample()/pdf()` oracle：official repo不含NICE，论文没有完整展开half-vector到incident-solid-angle的所有Jacobian，也没有PDF normalization、sample→pdf、MIS bias或single-query shader成本实验。其Figure 14–15是16 spp、论文自有MERL/interpolation与scene protocol下的equal-sample evidence，不得与NBRDF、RTA或Importance Baking的硬件/场景/时间数值横向排名。[Neural Processes BRDF §§8–9,11–12]

## 4. Sample tuple 与可查询PDF不是一回事

Importance Baking先为固定material/view slice离线构建从`[0,1]^2`到direction的transport map，再让sampling network回归direction和预烘sample weight。hot sample path可以直接返回`wi`与RGB`f cos/q_target`，无需再次运行独立evaluator；这确实是很强的tuple-oriented设计。[Importance Baking §§3–5]

但MIS还需要任意`wi`处的PDF。论文为此训练另一张PDF network，而不是从sampling map的Jacobian精确求密度。个体报告没有找到以下证据：

- sampling map是一一可逆且Jacobian density可计算；
- independent PDF network积分为1；
- 它对sample map实际生成样本返回同一密度；
- sample中预烘weight与runtime evaluator/PDF在所有参数、方向和precision下对应。

所以可以把它称为“learned sample tuple + independent MIS PDF approximation”，不能写成已经认证的matched `sample/pdf`。本项目若复用其思想，应优先把返回tuple与proposal identity版本化，并用sample→pdf、normalization和MIS oracle验证，而不是只看低spp图像。

## 5. Mixture PDF：selected component不等于proposal

若先以`α_k`选component，再从`p_k(w)`采样，则实际proposal是

\[
q(w)=\sum_k α_k p_k(w),
\]

不是被选component的`p_j(w)`。只在互不重叠support或特殊estimator下才可省略其它项。

Belcour明确报告：按energy选一个lobe、却只用selected-lobe PDF，会在overlapping lobes产生fireflies；正式Eq.40用所有lobe density形成balance contribution。官方Forward/Symmetric代码静态体现了汇总mixture density，但transparent dielectric路径另有TIR与sample实现冲突，不能由opaque correspondence替代验证。[Belcour §§5.5,10–12]

这一边界直接约束NVIDIA、NLB、MetaLayer、NMA等mixture sampler：

- component selection probability必须进入PDF；
- 所有对该direction有support的components都必须参与mixture PDF；
- sample与外部`pdf(wo,wi)`必须使用同一prepared params、direction convention和solid-angle measure；
- clamp/safety lobe会改变proposal identity，不能当作无影响的数值保护。

## 6. Sampler训练目标与detach边界

### 6.1 NBRDF staged supervised proxy

NBRDF先fit每材质evaluator，再训练跨材质autoencoder，最后由32D embedding监督预测analytic fit labels。proposal没有反向塑造evaluator或embedding的公开joint objective。这是一条稳定但多阶段的离线路线；predictor optimizer、batch、seed和代码均未公开。[NBRDF §§7.2–7.3,12]

### 6.2 NLB GNDF KL

NLB sampler把GT BRDF转换为normalized GNDF grid，并以softmax-KL训练sampling proxy。optimizer、batch与正式训练代码未报告。normalized proxy拟合的是proposal shape，不意味着neural evaluator本身归一或energy conserving。[NLB §7]

### 6.3 NVIDIA learned sampler KL

RTA 2024用KL让learned PDF逼近当前learned BRDF target，且从该loss path detach latent，避免sampler objective反向扭曲共享material representation；evaluator和sampler使用各65k batch同时训练。正式论文没有给KL方向、target normalization、invalid handling或gradient estimator。2026 official code能支持一种later-code-informed estimator，但不能回填成2024明文。[RTA §§7,11]

对当前项目，这形成三个独立实验轴：

1. sampler对frozen evaluator训练还是与evaluator同步；
2. sampler gradient是否进入shared latent；
3. target density用luminance`f|cos|`、RGB mixture、energy proxy还是source-native sample statistics。

任何比较都要固定evaluator checkpoint或明确同步策略，否则sampling改善可能来自evaluator被改变。

### 6.4 Neural Processes BRDF post-trained NICE

Zheng等人先完成Neural Process evaluator，再把frozen decoder给出的`luminance(f) cosθ_i`归一化密度作为NICE target；这比“同步更新但对latent detach”更强地隔离了sampler对evaluator的反向影响。论文只写KL divergence，没有报告KL方向、normalization estimator、direction count、optimizer exact config、seed或checkpoint；official repo也没有NICE implementation/weights。[Neural Processes BRDF §§7,11–12]

论文报告的TensorRT batch timing是RTX 2080 Ti上一次生成`512×512` samples的coherent batch，不包含single random sample、独立PDF query、BRDF eval、environment lookup或完整path tracing。它只能说明原protocol中的batch throughput，不能拿来与RTA shader latency、NBRDF CPU rays/s或scene方法frame time比较。[Neural Processes BRDF §8]

## 7. Reference sampling 与产品sampling不能混用

Guo 2018的position-free method沿内部layer/volume path随机行走，可同时构造stochastic BSDF estimate与directional sample。其PDF不是简单对最终direction的解析闭式：需要对所有可能生成同一external direction的内部随机路径概率做积分，并在随机函数/PDF与scene MIS结合时满足独立性条件。正式方法还提供unidirectional/bidirectional estimator与balanced MIS recurrence。[Guo §§4–5; S-MIS/S-BD]

它适合当前layer-stack source family的reference，因为保留native interfaces/slabs并估计完整multiple scattering；但它不适合作为最终small shader：path length、随机事件数与内部state可变，single query noisy，代码还保留paper/config的depth/PDF gaps。[Guo §§8,11–13]

正确关系是：

```text
native LayerStack parameters
  ├─ Guo/random-walk reference: 生成权威GT query与source-native stochastic sample
  ├─ direct neural evaluator: 编译成固定成本的bare f
  └─ analytic/learned proposal: 逼近target density并提供matched sample/pdf
```

reference sample可用于训练proposal或integration control，但不会自动成为部署sampler identity。

Xia 2020直接改进的是这条reference内部求积，而不是产品`sample(wo)`。固定外部端点后，一条含`n`个内部方向的path component有`n+1`个相邻BSDF因子，普通sequential proposal只用`n`次方向采样，至少漏掉一个可能尖锐的因子；Xia以Gaussian slice product让proposal同时贴近相邻因子，pair-product采一个内部方向，multiple-product只对正式定义的一类三因子component联合采两个方向。Gaussian approximation只改变内部proposal，原始BSDF contribution仍被求值。[Xia §§3–7]

因此Xia不能登记成learned material sampler，也不能替换Guo external sample/stochastic-PDF接口。它最直接影响的是`evaluation_samples`预算内的reference variance与work accounting；论文以`effective time`归一化proposal cost，且未公开code、formal termination、reference SE或per-scene measured timing，所以不能把其比值当本项目GPU wall-clock speedup预测。[Xia §§7–8,11–13]

## 8. MIS 与 output measure

当前项目canonical evaluator输出bare linear`f`，proposal PDF是solid-angle density，sample tuple为`f|n·wi|/pdf`。对论文实现至少要检查：

| 风险 | 典型来源 | 需要的oracle |
|---|---|---|
| training/runtime quantity可能已经乘cosine | NBRDF loss、RTA supplemental、Biplane/release、Importance Baking evaluator | 与source bare-f逐direction correspondence；cosine adapter只出现一次 |
| sampler返回预烘weight而非`f` | Importance Baking、Guo stochastic sample | 逐sample重算`f*cos/pdf`；确认weight期望与目标一致 |
| PDF measure不明 | projected half-vector、visible-normal、Mitsuba cosine-weighted ABI | Jacobian、hemisphere/solid-angle normalization、event side |
| mixture只返回selected component | 任意multi-lobe sampler | 对同direction汇总全部components；数值积分PDF≈1 |
| reverse PDF/event缺失 | transmission、BDPT/MLT、NMA analytic target | forward/reverse parity、reflection/transmission support、delta flags |
| stochastic state不一致 | stochastic mip、sampler params在sample/pdf间重解 | 同一`prepare` state复用，sample→external pdf逐项一致 |
| invalid方向clamp改变support | roughness/PDF floor/safety lobe | support sweep、finite weights、white-furnace与tail统计 |

整图RMSE下降不能替代这些identity checks；反之，PDF归一通过也不证明proposal方差好。正确性与empirical variance是两层验收。

## 9. 已报告的失败与限制

| 分类 | 方法/尝试 | 观察 | 项目边界 |
|---|---|---|---|
| `ablation-inferior` | NBRDF Phong vs GGX proxy | 作者slides称Phong最好；main指出更准确fit不保证更好sampling | 无完整matched数值表，不能称GGX普遍失败 |
| `known-limitation` | NBRDF two-param monochrome proxy | layered、transmission、multi-peak未验证 | 适合超廉价baseline，不应预设能覆盖当前全部source states |
| `known-limitation` | NeuMIP只用cosine hemisphere | 作者指出更specular材质需要per-texel parametric PDF，并建议Lambertian+microfacet mixture | evaluator质量与integration方差必须分报；未给matched sampler结果 |
| `design limitation` | Biplane只用Lambertian proposal | Gaussian/Blinn–Phong proposal只列为未来扩展，没有sampler variance实验 | 不能把“可能对sharp/glossy不匹配”升级成已观察失败 |
| `known-limitation` | MetaLayer analytic sampler | 作者明确非arbitrary layered material optimal | 无normalization/parity或sampler消融，不能从scene image反推matched quality |
| `author-negative` | Belcour只用selected-lobe PDF | overlapping lobes产生fireflies | 所有mixture proposal必须算完整density |
| `ablation-inferior` | Belcour不使用TIR factor | short TRT paths被高估，虽仍energy conserving | energy test不足以验证path decomposition或proposal quality |
| `evidence gap` | Importance Baking independent PDF | sample map/PDF consistency未证明 | 保留为sample-tuple候选，不能直接注册为已match sampler |
| `evidence gap` | NMA built-in sampling | blend/reverse PDF/event code不可得 | 可作analytic proposal候选，接入前必须重建并验证独立identity |
| `evidence gap` | Neural Processes BRDF NICE | flow设计有change-of-variables，但official code/weights与完整solid-angle Jacobian不可得 | 16 spp equal-sample images不能替代normalization、sample→pdf、MIS与equal-time审计 |
| `paper-code-gap` | Hybrid formal cosine+GGX MIS | 论文formal mixture的selection/heuristic/host code未公开；`.brdf`只实现GGX，Web为GGX–environment MIS | 可核查的demo sampler不能冒充formal external material proposal |
| `author-negative` / `known-limitation` | Xia Gaussian product proposal | multiple-product在超过三因子或高roughness/宽lobe时收益下降；pair-product随path变长收益递减；volume extension未实现 | 只作为reference-internal proposal候选，不能升级为任意长path或external material sampler |

## 10. 对当前 NeuralShading 的执行约束 `[N/I]`

### 10.0 Scene-level query单独登记为`G`

LightFormer/NeLiF的emitter VPL/RSM、Dual-Band的mirror secondary ray、NeLT的light/object samples、Superposed DFF的object-field queries、Efficient Light Probes的probe search、Neural Prefiltering的voxel visibility traversal、1469的volume auxiliary rays和Mobile VR的per-light texture-space shading都属于scene/system integration cost。它们不会因为内部出现“sample”“ray”“direction”或“sum”就变成local proposal；只有实际向材质抽取`wi`并能查询同一solid-angle density的路径才属于`S`。[§2.2；各个体报告 §§4–8]

Active Exploration还需再分一层：其MCMC在训练期选择**scene configuration/query**，训练完成后不进入runtime；PixelGenerator本身才是scene radiance surrogate。因此可迁移的是query-recipe diagnostic，不是给当前`sample()/pdf()`增加一个MCMC runtime分支。[Active Exploration §§5–8,13]

NeLT与Superposed DFF论文中的AE对照均禁用active selector，只用uniform training；因此它们不能证明active exploration在object-field方法上无效。NeLiF则因per-scene训练成本排除AE、NeLT、Superposed DFF的数值baseline，不能把“未比较”改写成sampler或integration失败。三条边界共同要求scene representation、training query allocation与runtime integration分别注册。[NeLT §9；Superposed DFF §9；NeLiF §9]

任何未来scene track都应单独冻结：scene/light/camera/material动态轴、visibility职责、input buffer generation、resolution、history/reprojection、GT integrator、frame-time breakdown与memory。local sampler实验继续固定同一个renderer/integrator；不能用scene surrogate的final-image improvement反向宣称材质proposal更match，也不能把local PDF oracle通过解释为scene visibility正确。

### 10.1 Sampler作为独立配置轴

对同一frozen evaluator/source/test scenes，至少保留：

- cosine/hemisphere control；
- Belcour-style bounded analytic mixture；
- NBRDF-style超小supervised analytic proxy；
- NVIDIA-styleper-hit learned analytic mixture；
- 若探索direct neural sample map，必须额外给可认证的density或明确只作non-MIS tuple path。

不得因proposal family不同而替换evaluator，也不得用各论文自己的scene、SPP和硬件数字横向排名。

### 10.2 `prepare()`应缓存什么

对同一shading point和`wo`，适合缓存：latent、learned frame、analytic lobe参数、component probabilities、roughness/correlation/tilt以及stochastic LoD选择。这样`sample()`与后续`pdf()`使用同一state，也能供NEE direction重复查询。若每次重解sampler network或重选mip，必须单独登记cost与随机一致性。

### 10.3 训练和评测最小协议

1. 先冻结evaluator checkpoint与bare-f correspondence；
2. 对每个proposal做finite/support/normalization/sample→pdf测试；
3. 以同一integrator、MIS heuristic、scene/camera/light、SPP、seed和time accounting渲染；
4. 同时报variance/RMSE随SPP与随time、tail weights/fireflies、sampler decode/PDF query成本；
5. 分layer state、roughness、grazing、transmission、multi-peak strata；
6. proposal训练只比较同等reference query预算，joint/frozen latent分别设identity。

## 11. 可证伪的综合假设 `[I]`

| Hypothesis | Direct evidence | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|
| S1：两参数analytic proxy可在大幅低于GGX9 decode成本时保留主要variance收益 | NBRDF predicted Phong接近fit且优于uniform | cosine、2-param、9-param；同frozen evaluator | training queries、scenes、MIS、SPP/seeds、backend | normalization、variance/time、tail weights、MAC/state | sampler proposal | 2-param接近cosine或hard strata显著失败，且成本收益不足 |
| S2：Belcour R/TRT/TT mixture是layer-stack learned sampler的强control | Belcour完整mixture PDF负结果；MetaLayer外接相同path-lobe family | cosine、Belcour、learned；同evaluator/integrator | source states、lobe cap、precision、MIS、time | variance/time、firefly、PDF identity、bytes | analytic control/proposal | Belcour相对cosine无收益，或learned无法超过它且成本更高 |
| S3：sampler loss不反传shared latent更稳定且不损evaluator | RTA latent detach；NBRDF与Neural Processes BRDF staged/post-train sampler | joint gradient、detach simultaneous、frozen sequential | evaluator/sampler shape、total queries、seeds、LR | evaluator drift/error、sampler variance/KL、failure rate | training mechanism | detach/frozen不降失败率或显著损proposal质量，joint同时更优 |
| S4：sample-tuple网络必须有同源density才适合MIS | Importance Baking evidence gap | independent PDF vs Jacobian/flow-certified density或non-MIS path | sampler map、target、training work、integrator | normalization、sample→pdf、MIS bias、variance/time | direct neural sampler | independent PDF在全部oracle/CI内无bias且成本更优，否定必须同源的工程假设 |
| S5：完整mixture PDF可消除selected-component firefly tail | Belcour author-negative | selected-only vs full mixture density | samples、lobes、evaluator、scene、seed | tail weight quantiles、firefly count、bias、PDF cost | mixture PDF correctness | full density不改善tail或selected路径经证明在support上等价 |
| S6：view-conditioned analytic target可在`prepare`摊销且保持sample/eval/pdf一致 | NMA/RTA参数仅依赖source state+view | uncached vs cached prepared params | convention、precision、source edits、sampler | dense f、sample/pdf parity、prepare+Nquery time/state | compiler + analytic proposal | cached与uncached超容差或state实际依赖逐`wi`无法复用 |
| S7：local proposal与scene observation是正交成本轴 | `L/S/R/G`矩阵显示LightFormer/Dual-Band/Probes/Prefiltering都不输出local PDF | 2×2 `{cosine,matched local sampler}×{reference PT,scene surrogate}`；分别冻结训练与runtime identity | material/evaluator、scene/light/camera、SPP/time、visibility、seed、hardware | local PDF/variance、scene bias、frame time、tail与visibility error | sampler × scene integrator | scene surrogate单独改变local sample→pdf结果，或local sampler收益完全来自不matched的scene/GT改变 |

## 12. 当前完成状态

- [x] local evaluator、reference-internal estimator、external material-direction sampler与scene integration/visibility四类query semantics已分开；
- [x] mixture PDF、cosine/measure与MIS identity风险已列为正确性门；
- [x] analytic与learned sampler按相同frozen evaluator比较，没有强制closure词汇；
- [x] 作者负结果与未报告实现边界未混淆；
- [x] 已用当前evidence-reviewed个体报告建立scene integration/visibility矩阵，并保留各自output/cost domain；
- [x] 所有跨论文数值保留原protocol/hardware边界，不形成未matched排名；
- [x] Xia在共享工作树完成独立复核后按`R`类纳入，并与Guo external `S`保持分离；
- [x] NeLT、Superposed DFF与NeLiF在正文解锁和独立复核后进入`G`矩阵，且没有把object/light sample、latent sum或VPL误写成local PDF；
- [x] 本轮三份来源恢复后的独立 cross-document evidence review 已完成；三者均只归入`G`，AE uniform与baseline-exclusion边界已逐项回查。

## Evidence review

```text
author_update: /root
reviewer: /root/nelt_full_report
reviewed_at: 2026-08-29
sources_rechecked:
  - current sampling-and-integration comparison, including the L/R/S/G semantics and all report links
  - evidence-reviewed NeLT §§4-11 and Evidence review
  - evidence-reviewed Superposed DFF §§4-11 and Evidence review
  - evidence-reviewed NeLiF §§4-11 and Evidence review
findings_closed:
  - NeLT object/light samples和typed ratio/residual composition只归类为G；无local S/pdf
  - Superposed DFF object-pair latent sums与per-pixel field sums只归类为G；不是MIS estimator或local S/pdf
  - NeLiF lighting-field lookup、RSM/VPL、五级shadow maps与3DGS composition只归类为G；无local evaluate/sample/pdf
  - NeLT与Superposed DFF的AE对照均为uniform training，没有把禁用active selector后的结果改写成完整AE policy结论
  - NeLiF按per-scene training/generalization协议排除AE、NeLT、Superposed DFF，不把未比较改写成quality、sampler或integration失败
remaining_evidence_gaps:
  - NeLT, Superposed DFF and NeLiF supplemental/code and exact runtime/integration scope remain unavailable
  - paper-specific missing PDF normalization, sample-to-pdf, reverse-event, runtime-breakdown and source-code evidence remain as recorded in the individual reports
review_status: evidence-reviewed
```
