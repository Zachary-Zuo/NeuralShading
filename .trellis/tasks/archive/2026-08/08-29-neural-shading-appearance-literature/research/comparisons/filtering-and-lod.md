# Filtering 与 LoD：跨论文证据综合

## 1. 证据边界与术语

本文只综合28篇 `evidence-reviewed` corpus 中与filtering/LoD直接相关的子集。local material证据来自 NeuMIP 2021、NVIDIA Real-Time Neural Appearance Models 2024、Neural Biplane BTF 2023、Towards Comprehensive Neural Materials 2025、Neural Layered BRDFs 2022、Hierarchical Neural Materials 2024 与 Mobile VR Neural Materials 2026；asset-level aggregate边界来自 Neural Prefiltering 2023；scene-level 证据纳入已经完成独立复核的 CNSR 2020、Active Exploration 2022、Superposed DFF 2024、Dual-Band Neural GI 2025、LightFormer 2024 与 NeLiF 2025。NeLT也已解锁，但正文没有material footprint、runtime level selection或scene-filter hierarchy；它只作为“neural texture不自动等于LoD”的边界，不为凑表而补不存在的filtering配置。

本文把容易混名的五件事分开：

1. **reference filtering**：GT/reference 对给定像素或纹理 footprint 真正积分什么物理量；
2. **representation hierarchy**：不同尺度存独立状态，还是由 finest state 下采样；
3. **runtime level selection/filtering**：一次query读取哪个level、如何做邻级混合；
4. **optimization continuation**：训练早期对target、latent或方向做平滑，最终可完全消失；
5. **system amortization**：降低昂贵着色更新频率，不等于产生一个正确的filtered material function。

如果这五层没有分别登记，就无法判断一个方法是在近似 `f(u,wo,wi)` 的 footprint average、一个asset体素内的aggregate transport，还是一张跨帧复用的radiance texture。

## 2. 方法总表

| 方法 | 被过滤的物理对象 | 层级状态 | runtime选择/插值 | 训练期平滑 | 已知边界 |
|---|---|---|---|---|---|
| NeuMIP 2021 | microgeometry/measured BTF的Gaussian-footprint MBTF response | 每个level独立优化7-channel feature grid；offset texture单层 | fractional `log2 σ`在相邻两level各bilinear，再linear blend | latent Gaussian radius 8、half-life 3333；code另有前3k periodic blur，offset在10k后冻结 | exact footprint/query distribution未报告；hard shadow与glints困难；无matched sampler |
| NVIDIA RTA 2024 | source material graph在Gaussian footprint下的filtered BRDF | 每level独立学习z8；两张RGBA FP16 mip textures | fractional LoD在相邻整数level间Russian roulette选一个level，再从该level的两张RGBA texture取得bilinear z8 | 前20k对`wo`做10°→0° directional mollification；coarse encoder input用LEAN | stochastic level有有界方差；native sample/tap与cache traffic、exact mip distribution/footprint/LEAN字段未报告；P/S output measure有gap |
| Neural Biplane 2023 | 单一spatial/directional BTF query；没有正式continuous footprint target | U/H planes；没有formal LoD hierarchy | 单level bilinear planes | 新BTF compression的plane blur kernel `20→0` | blur是优化continuation，不是runtime LoD；grazing undersampling与fine detail仍失败 |
| Comprehensive 2025 | BTF appearance的spatial U-plane average mip + 独立height/silhouette geometry path | U/H/D planes；U有average mip chain且dynamic synthesis只作用U；height另有coarse/fine max hierarchy | U/H/D bilinear lookup；appearance LoD查U average mip，但exact footprint、level selection/邻级插值未报告；height shell先coarse后fine | 正式训练helper不足以锁定mip构建与filtering lifecycle | U mip是derived average，不是NeuMIP/NVIDIA式每尺度GT独立学习；radiometric measure、address/filter与asset correspondence未闭合 |
| Hierarchical 2024 | camera-distance footprint下的spatial-angular material buffer；作者只称texture pyramid与offset沿用NeuMIP | paper不给channels/resolution/kernel/interpolation；release虽有7-channel texture与mip/offset实现，也不能反填formal配置 | full query buffer经neural offset/pyramid后进入两层Inception；runtime footprint→level mapping和邻级插值未报告 | release training每step随机选择可访问的最高level，并有前3k periodic blur、`σ`衰减与offset freeze逻辑；这些都没有formal run mapping | whole-buffer `3×3/5×5` neighborhood不是random-access single query；paper/code Fourier、output measure与loss均有gap；README不是formal配置 |
| CNSR 2020 | scene observations与novel-view G-buffer条件下的integrated HDR image；没有material footprint target | global scene representation + Pixel/U-net/GQN image generator；U-net scales是screen-space backbone，不是material mip | query时改变novel-view G-buffer分辨率；Pixel逐像素，U-net/GQN读取screen-space neighborhood/state；没有level选择 | formal generator comparison在64²训练并测试64/128/256；这是resolution extrapolation，不是逐级filtered target或curriculum | Pixel较稳定但texture模糊；U-net随分辨率改变出现shadow shrink/structure错位；无footprint kernel、level state或邻级策略 |
| Active Exploration 2022 | per-scene GI radiance patch；没有material footprint target | per-scene PixelGenerator + first-hit G-buffer；没有runtime LoD hierarchy | 每帧全分辨率G-buffer后逐像素MLP；没有level选择/插值 | formal保持32² patch，把完整训练渲染从128²每2000 iterations增加4至600²；official README/code组合的首个optimizer step实际为132² | curriculum改变patch覆盖的scene区域和active selector分布；Uniform+multi-res是mixed result，不能当作普遍filter收益 |
| Superposed DFF 2024 | per-scene image radiance；没有material footprint target | 每个field使用deformable multi-resolution triplane，levels/resolutions/channels未公开 | runtime拼接所有已开放levels的sampled features；没有footprint→level选择、邻级filter或LoD query | Eq.(8) windowed weights用线性增长`α`从低到高开放levels；同一windowed encoding也作用offset auxiliary features；`α`起止值与增长时长未报告 | 这是训练capacity continuation，不是runtime LoD；w/o C2F较差但无multi-seed/iso-work分解，large scene高分辨率triplane会显著增memory |
| Dual-Band Neural GI 2025 | scene radiance中的glossy reflection；按roughness与mirror-ray clue做screen-space近似低通 | per-object 8-level triplanes用于feature capacity；三尺度U-Net features | 三个尺度各预测`5×5` softmax kernel与level/interpolation权重，再过滤secondary feature | stage 1从低到高progressive unlock triplane levels；stage 2启用secondary/fusion | triplane没有footprint LoD语义；multi-scale `gamma` merge未报告；secondary clue仅来自每像素一条single-bounce mirror ray，long specular chain是已知失败区 |
| LightFormer 2024 | per-light scene observations；direct visibility与indirect attributes按不同空间分辨率采集 | depth RSM通常1024²、position/normal/flux 64²；large scene四级cascades | encoders把RSM/VPL/clues变为per-light screen maps，再由pixel-light attention聚合 | 无surface-footprint curriculum；20k configurations/scene共同训练 | high-res depth/low-res indirect是frequency allocation，不是material LoD；mirror/high-frequency indirect shadow仍失败 |
| NeLiF 2025 | 4D luminaire observations生成的lighting field，以及screen-space direct/shadow/indirect shading；没有material footprint target | spherical triplane把angular/radial density分配给近灯区域；shadow另有五级hard-shadow pyramid | triplane以`(θ,φ,r)`查询；shadow各级预测`5×5` filter与`4×4` learned upsample kernels，再做softmax level blend | 正文没有field-resolution curriculum；Fig.12的32/128只属定性triplane消融，不是formal LoD；shadow pyramid是runtime image filter | spherical density是coordinate prior，五级shadow是screen-space visibility discontinuity处理；均不定义filtered BRDF，field resolution/bytes、kernel normalization与energy correction均未报告 |
| Neural Prefiltering 2023 | 一个object-space voxel内部的visibility、BSDF、geometry和multi-bounce aggregate throughput | 7个discrete voxel LoD；每asset multi-resolution hash fields + threshold grids | 外部/手工选择一个global discrete voxel LoD，再做可变路径遍历；没有continuous pixel-footprint filtering或跨层blend | 所有LoD联合训练；appearance用noise-to-noise MC target，visibility用BCE；每voxel另搜索visibility threshold | target不是local BRDF；跨level能量漂移、长程相关性、锐利方向和dynamic geometry均有限 |
| Mobile VR 2026 | finest-resolution离散BTF；系统输出是per-eye object-space radiance texture | coarse 7-channel与fine 8-channel textures；没有feature pyramid | 一个`2×2` square共享coarse方向/MLP，四texel分别fine decode；更新后再供未来N帧复用（总计N+1帧） | teacher/student distillation，不是footprint continuation | 明确无robust LoD；大texture性能下降；minor angular aliasing与fine-detail failure |

证据：[NeuMIP §§5–7,10–13](../papers/kuznetsov-2021-neumip.md)、[RTA §§5–7,10–13](../papers/zeltner-2024-real-time-neural-appearance-models.md)、[Biplane §§5–7,10](../papers/fan-2023-neural-biplane-btf.md)、[Comprehensive §§5–8,11–13](../papers/xu-2025-comprehensive-neural-materials.md)、[Hierarchical §§4–8,10–13](../papers/xue-2024-hierarchical-neural-materials.md)、[NLB §§5–8,11–13](../papers/fan-2022-neural-layered-brdfs.md)、[CNSR §§4–8,10–13](../papers/granskog-2020-compositional-neural-scene-representations.md)、[Active Exploration §§4–8,10–13](../papers/diolatzis-2022-active-exploration-neural-gi.md)、[Superposed DFF §§4–8,10–13](../papers/zheng-2024-superposed-deformable-feature-fields.md)、[Dual-Band §§4–7,10–13](../papers/mo-2025-dual-band-neural-gi.md)、[LightFormer §§4–7,10–13](../papers/ren-2024-lightformer.md)、[NeLiF §§4–8,10–13](../papers/sheng-2025-nelif.md)、[Neural Prefiltering §§4–7,10–13](../papers/weier-2023-neural-prefiltering-lod.md)、[Mobile VR §§4–8,10–13](../papers/xu-2026-real-time-neural-materials-mobile-vr.md)。

## 3. Reference filtering：先定义积分对象

### 3.1 Local material footprint

NeuMIP把query显式写成位置、方向和Gaussian footprint；GT由synthetic microstructure path tracing或measured BTF形成。每个尺度目标仍是局部appearance/MBTF，而不是把场景visibility或lighting积分进去。论文未报告Gaussian截断、每LoD概率和完整query density，因此只能锁定“Gaussian-footprint filtered target”，不能复原exact online recipe。[NeuMIP §§4,6,12]

NVIDIA RTA也把每个level对应到Gaussian footprint，并让空间sample数随filter area增长；coarse encoder inputs使用LEAN预过滤。它训练的是source material graph的filtered response，运行时留下latent hierarchy。正式材料未披露Gaussian sigma/截断、LEAN具体字段、exponential mip分布参数和level cap，因此当前项目中这些都必须作为`author-underspecified`配置冻结，不能声称是论文隐藏常数。[RTA §§5.1–5.2,6,12,14]

两者共同支持“reference在目标footprint上定义、不同level直接学对应filtered function”，但不支持共享exact filter kernel或query recipe。

### 3.2 Asset transport LoD

Neural Prefiltering把每个voxel内部的path throughput聚合成`φ_V`，并另学point-to-point visibility。变粗的LoD会吞入更多geometry、material、visibility和多次transport；这不是对同一个local `f`做更大UV footprint平均。其far-field approximation主动删除voxel boundary positions，只保留center与directions，长程correlation仍由runtime traversal处理。[Neural Prefiltering §§3–5]

所以它对本项目最有价值的是“不同尺度目标必须按物理语义重新定义”的反例，而不是一个可直接移植的material mipmap。

### 3.3 System radiance reuse

Mobile VR先在object texture space执行neural BTF shading，再把已经乘入当前灯光/视角结果的radiance texture供后续N帧复用，即一次更新覆盖N+1个display frames。它降低的是求值频率；没有定义跨像素footprint的filtered BTF，也没有披露camera/light/object变化时的motion threshold或adaptive refresh。`2×2` coarse share还在square center计算统一方向，作者明确承认minor angular aliasing。[Mobile VR §§4.1–4.2,6,12]

因此temporal reuse不能被登记为LoD fidelity，必须以运动/lighting validity、更新调度与temporal error独立评估。

### 3.4 Screen-space learned reflection filtering

Dual-Band不是对material footprint预过滤。它先沿mirror direction取得secondary hit/feature，再由full-frame CNN读取roughness、view-normal cosine、reflection depth/emission和principal/secondary features，预测三个尺度的`5×5` kernels与插值权重，用邻域reflection observations近似rough glossy response。用MLP替换CNN会让反射边缘/shadow过锐；去掉feature self-tuning inputs会产生noisy reflection；去掉zero branch会在dynamic occlusion漏光。[Dual-Band §§5,10]

这个结果支持“scene reflection filter应读取被过滤的signal clues”，但不能直接迁移成local footprint filter：它依赖screen-space邻域、visibility、mirror secondary ray和当前帧scene content。triplane的8 levels又只是progressive feature capacity，论文明确没有footprint/LOD semantics；二者也不能因为都叫multi-resolution而合并。

### 3.5 Scene-observation frequency allocation

LightFormer用高分辨率depth RSM提供direct visibility边界，用低分辨率position/normal/flux提供相对低频indirect observation；Emernald Square另用四级cascades扩大覆盖。这是输入buffer的频率/空间覆盖分配，不是对最终radiance或local material做mip过滤。其已知失败恰好落在未充分观察的高频transport：间接阴影过硬/错误、highly glossy或mirror reflection不能准确恢复。[LightFormer §§4,7,10]

可迁移的实验问题是“同bytes/time下all-low、split-resolution、all-high的scene observation Pareto”，不能把1024²/64²直接写成当前material latent的最佳mip比例。

### 3.6 Output resolution extrapolation/curriculum 与 material filtering 不同

CNSR的formal generator comparison在64² query G-buffer上训练，再测试64²/128²/256²。Pixel generator逐像素运行，跨分辨率较稳定但texture detail仍模糊；U-net则在分辨率变化时出现shadow shrink等artifact。这是image generator对query-buffer resolution的外推，不是训练期curriculum，也不是给定material footprint的正确积分。[CNSR §§5,8–10]

Active Exploration的formal protocol保持32² patch，把完整训练渲染从128²逐步提高到600²，使同样patch覆盖的scene区域缩小，目的是让训练后期发现更细scene features。论文写每2000 iterations增加4像素；official README与code组合却会在epoch loop开头先增加一次，故未resume时首个optimizer step实际为132²。Uniform+multi-res的MAPE改善、MAE变坏，作者判断整体更差，说明resolution curriculum与active selector耦合；不能把“128→600”单独移植成普遍正项。[Active Exploration §§6–7,10–11]

二者都没有定义local footprint kernel、per-level state或邻级选择，不能登记为material LoD方案。

### 3.7 Field hierarchy 与 shadow pyramid 也不是material LoD

Superposed DFF的deformable multi-resolution triplane在runtime把各level feature拼接使用；Eq.(8)的window只决定训练何时开放高频容量，不接收pixel footprint，也没有选择单一level。NeLiF先由4D luminaire observations生成spherical lighting field；球面triplane把近灯区域分配更高coordinate density，但它服务于lighting function，五级shadow hierarchy则读取screen-space hard-shadow maps并学习filter/upsample。两者分别是field capacity prior与image-space visibility filter，不能登记为source-material filtering。[Superposed DFF §§5,7,10；NeLiF §§4–5,10]

NeLT虽然生成direct/indirect neural textures并做bilinear fetch，但正文没有texture mip、footprint或跨级语义；“有texture fetch”不能自行升级成filtering方案。这个反例要求后续catalog明确登记representation state与filter contract，而不是从数据结构名称推断LoD。[NeLT §§5,8,12]

三者也都没有验证本项目的local `prepare(P,wo)`合同：NeLT没有报告representation/texture/hypernetwork的重建cadence，Superposed DFF没有拆分scene-state encoder/hypernetwork与per-pixel field query，NeLiF没有报告lighting-field/3DGS generation的耗时、bytes或更新策略。把这些scene-level precompute类比为`prepare()`只能是后续迁移假设，不能由本文写成已证实的filter/LoD runtime机制。

## 4. 层级表示：independent levels 与 derived mip

### 4.1 NeuMIP

NeuMIP每个feature level独立优化；fractional scale查询两个相邻level，各做bilinear后linear blend。连续LoD来自**输出latent的跨层插值**，不是从finest latent生成普通mip。每level有自己的容量，能拟合该footprint下非线性过滤后的appearance，但asset bytes随各level总texel数增加。[NeuMIP §§3.2,6,11]

论文Table 3显示coarser levels通常误差更低，但并非严格单调，例如Wool Twisted从LOD4的`0.74`回升到LOD5/6的`1.16/1.17`（表内MSE×`10^-3`口径）。这说明“低频更易拟合”只是总体趋势，不能代替per-level质量检查。[NeuMIP §9]

### 4.2 NVIDIA RTA

NVIDIA同样让coarse level直接学习filtered BRDF latent，但拒绝在语义不同的latent之间做trilinear。fractional LoD在两个相邻整数level间按权重随机选择一个，只在选中level做bilinear；z8仍由该level的两张RGBA texture提供。作者解释trilinear会强迫两level之间的latent线性路径也解码为合理BRDF；stochastic selection则以小方差换取避免该插值bias。论文没有把bilinear硬件tap、cache transaction或整条shader traffic量化成统一fetch数。[RTA §§4.1,5.1,8,12]

这是一个明确的bias–variance–reads三角：

- one-level stochastic：取得一份level state（正式z8布局仍涉及两张RGBA texture），期望上混合两level的decoded outputs，但单query有level noise；
- two-level latent interpolation：取得相邻两份level state、确定性，但把latent线性混合交给非线性decoder；
- two-level output interpolation：需要两次decoder，成本更高，但数学上直接混合response。

论文只正式实现第一种，没有给三者equal-level-state/equal-MAC的完整matched消融；native texture sample/tap和cache traffic也未报告。当前项目不能把它写成普遍最优。

### 4.3 普通latent mip仍是独立候选

[NLB supplemental](../papers/fan-2022-neural-layered-brdfs.md)对96-float-per-texel latent建立普通mipmap并trilinear，但没有闭合coarse levels是由GT独立学习还是对latent downsample。Biplane没有footprint/LoD轴；Comprehensive则有U-plane average mip与独立height max hierarchy，但没有披露continuous footprint、U level选择/邻级插值或mip构建细节，不能与NeuMIP/RTA的independently target-supervised levels合并。因而“derived mip”在当前综合中是待测试control，不是假设已经被所有论文否定。

### 4.4 Hierarchical Neural Materials不能替NeuMIP补齐层级配置

Hierarchical正文明确说neural offset与texture pyramid保持NeuMIP原样，贡献集中在Inception decoder、encoding与loss；但它没有重述pyramid channels/resolution、filter kernel、interpolation、precision或level target。release代码能证明其训练路径会随机选择可访问的最高level，并存在blur、`σ`衰减与offset lifecycle，却没有formal command/config/checkpoint mapping，也没有由此闭合runtime footprint→level映射。因此本综合只能登记“作者沿用NeuMIP式多尺度入口”，不能把NeuMIP论文的隐藏常数、release defaults或NeuMIP自己的runtime interpolation直接复制成Hierarchical formal配置。

更关键的是，两层Inception的`3×3/5×5/max-pool`跨query-buffer邻域读取。它改善的是coherent full-buffer reconstruction，在稀疏query、tile seam、divergent LoD和单点shader下没有协议。即使它在Table/figures优于NeuMIP，也不能把该收益全部归因于hierarchy；spatial receptive field、Fourier basis和loss同时变化，且不是项目允许的random-access runtime形态。[Hierarchical §§3–5,7,9–11]

## 5. 训练 continuation 不是 runtime filter

### 5.1 Spatial latent blur

NeuMIP从8-texel Gaussian radius开始，按约3333 iteration half-life衰减；作者报告不做逐步smoothing时free neural texture会出现类似Monte Carlo noise，offset尤其明显。release进一步显示feature lookup可降到0.1、offset至少1且10k后冻结，前3k每100 step还会做一次`0.9T+0.1 GaussianBlur_1.03(T)`。主指数schedule与paper数值对应，但下限/freeze/periodic in-place blur是code-only lifecycle。[NeuMIP §§7,10–11]

Biplane在新BTF compression前15 epochs把plane/offset blur kernel从20减到0，再冻结这些状态、只训5 epochs color adapter。最终runtime不读取blurred neighborhood；blur只是让高分辨率free planes从低频解逐步释放容量。[Biplane §§7,10]

### 5.2 Directional mollification

NVIDIA前20k iterations围绕`wo`对每个target平均256 samples，cone从10°以cosine schedule收缩到0°。它平滑的是方向目标峰值，不是spatial latent或footprint hierarchy。最终runtime没有该mollification；把它与NeuMIP spatial blur合成一个“Gaussian regularization”会丢失作用域。[RTA §§6–7]

### 5.3 Distillation

Mobile VR teacher用128-wide coarse/fine网络监督8-wide student的最终RGB和两层hidden features。它补的是低容量优化信号，不定义footprint或runtime LoD。若以后把distillation用于filtered evaluator，teacher必须查询同一footprint GT并遵守相同output measure，否则teacher consistency可能只是在复制另一种aliasing。[Mobile VR §§5–7,10]

### 5.4 Hierarchical release continuation 是未裁决的 adaptation 集合

Hierarchical release包含多个互不等价的fourth-root/linear/gradient schedule，还会在每个training step从`[5,5,5,7,num_mips]`选择可访问的最高level、周期融合blur并衰减`σ`；default `sigma_1_time=100000`在default 30k steps内不会触发offset freeze。论文只称loss会随learning progress自适应，没有给阶段边界。因而这些branch只能作为待注册的adaptation候选，不能从中任选一个作为formal continuation，也不能把training-time最高level随机化改写成runtime stochastic LoD。它们同样没有把训练期`σ`衰减变成runtime footprint filter。[Hierarchical §7]

### 5.5 Superposed DFF 的 C2F 是scene-field continuation

Superposed DFF在训练早期只开放低分辨率planes，随`α`线性增长逐级加入高分辨率，并把相同windowed positional encoding作用于offset auxiliary feature。Table2中w/o C2F较差，支持“该完整continuation在Watercolor protocol内有益”；但`α`起止、stage长度、level尺寸和repeat均不可得，不能把它与NeuMIP blur、RTA direction mollification或Dual-Band two-stage初始化合成一个普遍schedule。最终runtime仍读所有active levels，故它也不是runtime filter。[Superposed DFF §§5,7,9–10]

## 6. 已报告的失败和限制

| 分类 | 方法/配置 | 观察 | 不能越界的解释 |
|---|---|---|---|
| `author-negative` | NeuMIP free latent不做decaying blur | latent出现MC-noise-like纹理，offset更明显 | 支持高分辨率free state需要continuation；不证明所有encoder初始化或regularizer无效 |
| `known-limitation` | NeuMIP hard shadows/glints | shadow contrast轻微损失；very specular/glinty未充分展示 | deterministic small decoder+blur难保稀有高频；作者的stochastic/GAN设想会改变evaluate语义 |
| `author-positive with cost` | RTA stochastic adjacent level | 作者观察优于trilinear latent，且只读一level | 未给equal-cost完整消融；单queryvariance与temporal spectrum仍需测 |
| `known-limitation` | Neural Prefiltering Pandanus跨LoD重训 | 某些level能量损失/精度漂移 | all-LoD共享容量可能不稳定；不是local material hierarchy的直接实证 |
| `known-limitation` | Neural Prefiltering long-range/high-frequency/dynamic content | 长程correlation未预解；SH难锐利高光；geometry变化失效 | asset-level field的domain边界，不能用更粗voxel自动解决 |
| `mixed ablation` | Mobile VR same-capacity single-level vs coarse-to-fine | Leather08 PSNR `29.90/29.70`偏single，FLIP`.120/.115`偏coarse-to-fine | 指标排序相反；只能说质量接近且不能压成全面不损失 |
| `known-limitation` | Mobile VR无robust LoD、very large texture workload | `4×2400²` texels仅14 FPS；fine Fabric07仍缺细节 | texture resolution是显式质量/成本轴；temporal reuse不修复spatial aliasing |
| `known-limitation` | Biplane grazing/fine feature | grazing方向欠采样，fine appearance受限 | sampling与plane capacity共同作用；不能只归因于是否有LoD |
| `ablation-inferior` / `known-limitation` | Hierarchical `w/o Inception` 与full method | `w/o Inception`时作者图示后侧纱线细节丢失；full method比NeuMIP稍慢，仍无importance sampling且只捕获direct single bounce | 支持Inception多尺度邻域分支在该whole-buffer protocol内有益；不证明它可在random-access shader或pointwise student中保留同样收益，也不锁定mip/filter lifecycle |
| `author-negative` | Active Exploration uniform+multi-res | MAPE改善但MAE变坏，作者判断整体更差 | resolution curriculum与active selector有耦合；不能把“渐增分辨率”单独移植成普遍正项 |
| `known-limitation` | CNSR generator resolution transfer | Pixel会模糊texture detail；U-net高分辨率出现shadow shrink等artifact | output-resolution泛化不等于footprint filtering，global latent/query G-buffer也未提供material LoD语义 |
| `ablation-inferior` / `known-limitation` | Superposed DFF w/o C2F与large-scene field | Watercolor的w/o C2F在L1/SSIM/LPIPS均较差；作者同时指出large complex scene需极高分辨率triplane并dramatically增memory | 支持该scene-field continuation，不证明它定义LoD；最终质量与persistent bytes必须共同记账 |
| `ablation-inferior` / `known-limitation` | NeLiF regular triplane与monolithic high-frequency lighting | spherical triplane在32/128定性对比更好；正文承认monolithic radiance仍有spectral limitation | 这是coordinate density与capacity边界，不是runtime mip；没有matched bytes/time/metric，也不能把建议的multi-scale修复写成已成功尝试 |

## 7. Filtering ABI 与成本账本

对当前 `prepare/evaluate/sample/pdf` 合同，一个local filtered evaluator至少需要静态声明：

| 项 | 必须冻结的内容 | 原因 |
|---|---|---|
| Footprint | 坐标空间、kernel、scale/anisotropy、截断、边界/address mode | 决定GT物理语义，不是renderer随意metadata |
| Level state | independent/derived；每level channels、precision、resolution、训练target | 决定asset bytes与跨层可比性 |
| Level selection | nearest、stochastic adjacent、latent interpolation或output interpolation | 决定reads、decoder count、bias与variance |
| `prepare` state | selected level、latent、frame/offset、随机数是否缓存 | 决定同一点多`wi`能否复用及sample/eval一致性 |
| Randomness | stochastic LoD RNG的stream、同一次sample/eval/pdf是否共用state | 避免同一shading event中函数/PDF选择不同level |
| Measure | filtered输出是bare `f`、`f cos`、BTF response还是aggregate throughput | 防止scene/radiance方法误接local ABI |
| Cost | 每querytexture reads、decoder次数、state bytes、asset bytes和cache/coherence | 避免只报MLP weights而忽略hierarchy |

尤其对stochastic mip，若`prepare()`为同一着色点选择一次level，则后续多次`evaluate(wi)`共享同一随机level，会产生相关噪声；若每次evaluate重选，又会改变single-query reads和sample/pdf correspondence。RTA论文没有用项目ABI回答该调度问题，因此必须作为本项目新增、可证伪的implementation choice。

## 8. 对当前 NeuralShading 的影响 `[N/I]`

### 8.1 当前1×1 LayerStack与未来spatial source要分开

当前层栈source是1×1材料状态时，没有空间footprint可平均；强行添加spatial mip只会制造无意义轴。但scale仍可以通过方向mollification、source parameter strata和scene integration分开研究。只有接入MaterialX/BTF等spatial source时，NeuMIP/RTA的footprint hierarchy才成为产品候选。

因此filtering实验应分两条：

- **local direction continuation**：Taming/RTA式目标平滑，只改变training，不增加runtime hierarchy；
- **spatial filtered program**：reference显式接收footprint，compiler生成per-level state，runtime固定读取。

Hierarchical的whole-buffer Inception不进入这组运行候选，因为它让一个query依赖query-buffer邻居，违反固定随机访问的local evaluator合同。只有未来接入具有规范material-space邻域的spatial source后，才能把这种结构**重新设计**为training-only teacher或offline diagnostic；论文没有做teacher/student distillation。任何“蒸馏邻域teacher到pointwise student”的方案都是本项目新假设，不能沿用论文方法名，也不能直接使用output measure尚未闭合的released checkpoint。

### 8.2 最小候选集合

对同一spatial source、相同reference footprint recipe至少比较：

1. finest latent derived mip + one decoder；
2. per-level independently learned latent + trilinear latent；
3. per-level independent + stochastic adjacent one-level-state；
4. per-level independent + two decoded outputs再linear blend（higher-cost control）。

候选必须分别按iso-byte和iso-read/MAC报告，因为independent levels与two-decode天然多占不同资源。Mobile VR式2×2 share/temporal reuse放在deployment axis，不参与material-function fidelity的首轮排名。

### 8.3 评测不能只看每level静态误差

除现有G1/G2/G2s外，spatial hierarchy需要：

- continuous footprint sweep的error/energy；
- level boundary前后的temporal spectrum/pop；
- stochastic LoD的单样本variance和相关性；
- anisotropic/minified footprint与grazing strata；
- per-level asset bytes、reads、prepare/evaluate时间；
- source parameter edit后recompile/refilter成本；
- sample/pdf若依赖filtered evaluator，验证同一prepared state下的parity。

Neural Prefiltering的Pandanus失败说明per-level aggregate指标可能掩盖能量漂移；Mobile VR的PSNR/FLIP mixed结果又说明单一图像指标不足以判断aliasing。

## 9. 可证伪的综合假设 `[I]`

| Hypothesis | Direct evidence | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|
| F1：independent filtered levels优于derived latent mip | NeuMIP/RTA都直接学习level target | derived vs independent；iso-byte与natural-byte两组 | source、footprint GT、decoder、queries、optimizer、precision | per-level/continuous-scale error、energy、bytes/time | spatial compiler/evaluator | derived在相同或更低成本下不差，或independent优势只来自更多bytes |
| F2：stochastic one-level比latent trilinear有更好quality–level-state Pareto | RTA作者观察；latent语义可跨level不同 | stochastic one-level-state、latent two-level-state、output two-decode；另记录真实RGBA samples/cache traffic | checkpoints或matched retraining、RNG、footprint、decoder、texture layout | bias、variance、temporal spectrum、samples/traffic/latency | prepare/evaluate filtering | trilinear在相同成本下同时更低误差/variance，或stochastic temporal噪声抵消读取收益 |
| F3：encoder初始化可替代长程latent blur | NeuMIP no-blur negative；RTA有source-aware encoder bootstrap | random+blur、encoder no-blur、encoder+short blur | final architecture、queries、steps、seeds、footprints | convergence AUC、fine-detail recall、seed variance、final Pareto | compiler training | encoder不降噪/不加速，或同budget final质量显著更差 |
| F4：方向mollification与spatial blur是正交的continuation轴 | RTA与NeuMIP作用域不同 | 2×2 `{on/off spatial blur}×{on/off direction mollification}` | source/query/steps/model/seeds | spatial-frequency、direction-peak、aggregate/stratified error | training-only | 交互完全冗余，或任一轴在预期stratum无独立收益 |
| F5：stochastic level应在`prepare`固定一次而非每evaluate重选 | RTA one-level query；项目允许prepare复用 | per-prepare vs per-evaluate RNG，sample/pdf共用state | model、queries、RNG sequence、reuse count | bias/variance/correlation、sample-pdf parity、cost/state bytes | runtime scheduling | per-prepare造成不可接受相关噪声且无成本收益，或per-evaluate同样满足所有parity且更优 |
| F6：Mobile VR系统摊销不能替代robust LoD | 作者明确无LoD且大texture降到14 FPS | proper footprint hierarchy vs fixed finest+2×2/N-frame reuse，分开报告 | model/asset bytes或明确双预算、motion/light sequences | spatial/temporal error、FPS、latency、memory | deployment system | fixed finest方案在完整distance/motion/light sweep下同成本不劣且无alias/pop |
| F7：规范material-space邻域teacher可在不改变runtime ABI时改善pointwise filtered evaluator | Hierarchical只证明Inception在作者whole-buffer protocol内的正消融；没有distillation、pointwise student或random-access证据，本行完全是迁移假设 | pointwise baseline、同一pointwise student+training-only teacher、固定material-space stencil的offline upper control；不得直接把query-buffer adjacency当material邻域 | 仅限未来spatial source；student architecture/runtime reads、reference footprint/output measure、teacher topology/checkpoint、neighborhood/address mode、distillation loss/weight/work、queries/optimizer/seeds全部冻结或单列 | G1/G2/G2s、spatial-frequency/edge与UV seam/tile strata、single-query time/reads、teacher training/reference成本 | training-only distillation；runtime仍为pointwise random access | student无Pareto收益，收益仅在runtime读取邻域时存在，layout/seam改变结果，或teacher复制错误filter/output measure |

## 10. 当前完成状态

- [x] reference filtering、hierarchy、runtime selection、training continuation与system amortization已分开；
- [x] local material footprint与asset aggregate transport已划界；
- [x] Dual-Band的screen-space learned reflection filtering已与material footprint/LoD分开；
- [x] stochastic/continuous level策略按bias、variance、level state与实际读取账本比较，没有预判赢家；
- [x] 失败案例保持作者分类与原protocol；
- [x] LightFormer已按review补入scene-observation frequency allocation；
- [x] Hierarchical的NeuMIP inheritance、whole-buffer convolution与未裁决continuation已分别登记；
- [x] CNSR/Active Exploration的resolution行为已与material LoD分开；
- [x] Superposed DFF的C2F、NeLiF的spherical field/五级shadow与NeLT的single texture fetch均已和material LoD划界；
- [x] 三份来源恢复仍不以不可得supplemental/code补入filtering配置；
- [x] 本轮三份来源恢复后的独立cross-document evidence review已完成。

## Evidence review

```text
author_update: /root
reviewer: /root/belcour2018_review
review_date: 2026-08-29
sources_rechecked:
  - 28篇evidence-reviewed报告的front matter与本文件引用子集
  - evidence-reviewed NeuMIP, RTA, Neural Biplane, Comprehensive, Hierarchical, NLB, CNSR, Active Exploration, Superposed DFF, Dual-Band, LightFormer, NeLiF, Neural Prefiltering and Mobile VR reports
  - NeLT报告§§5、8、12，用于确认neural texture不含未披露的mip/filter/prepare合同
findings_closed:
  - 已确认Superposed DFF的deformable triplane C2F只属training capacity continuation，不是runtime LoD
  - 已确认NeLiF由4D observations生成spherical field；球面密度与五级kernel shadow分别属于coordinate prior和screen-space visibility filter
  - 已确认NeLT neural texture没有被无证据提升为mip/filter hierarchy
  - 已明确NeLT、Superposed DFF、NeLiF均未提供material-local prepare合同
remaining_evidence_gaps:
  - exact NeuMIP/RTA footprint kernels, sampling distributions and native fetch/cache traffic remain unreported
  - Hierarchical formal pyramid channels/resolutions/kernel/interpolation, runtime footprint-to-level mapping and adaptive-loss schedule remain unreported; release branches cannot close them
  - Comprehensive U-mip construction, footprint mapping, level selection/interpolation and address/filter mode remain unreported
  - CNSR formal 64-square training versus released ArchViz 128 config remains unresolved; neither source defines material footprint semantics
  - Active Exploration formal 128-square start versus README/code first-step 132 correspondence remains unresolved; the curriculum is coupled to its active selector
  - Dual-Band multi-scale gamma merge, Eq.(9) indexing and reflection-ray miss/default encoding remain unresolved
  - LightFormer RSM/VPL refresh, cache policy, runtime breakdown and resource bytes remain unreported
  - Superposed DFF field levels/resolutions/channels, alpha schedule, persistent bytes and runtime scope remain unreported
  - NeLiF field resolution/channels/bytes, shadow-kernel normalization, eta formula, generation/update lifecycle and runtime scope remain unreported
  - Mobile VR buffering, motion/lighting refresh policy and exact runtime texture format remain unreported
status: evidence-reviewed
```
