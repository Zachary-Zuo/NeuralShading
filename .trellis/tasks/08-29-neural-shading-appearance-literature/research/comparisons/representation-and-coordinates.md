# Representation 与 Coordinates：跨论文证据综合

## 1. 本轮综合的证据边界

本文只综合当前28篇 `report_status: evidence-reviewed` 个体报告中与表示、坐标和query contract直接相关的证据。当前版本除波次1的13篇指定 local neural material/appearance 与 asset-level 方法外，已纳入 Guo/Belcour/Xia 的reference与analytic controls、Neural Processes BRDF、Hierarchical Neural Materials、Improving Angular Parameterization、CNSR、Active Exploration，以及完整复核的scene/volume方法。§2 的 local/analytic总表覆盖19篇，§9 的scene/asset/volume矩阵覆盖10篇，其中Neural Prefiltering因同时承担asset-level local边界与scene-path aggregate边界而重复出现；两表并集正好是28篇，不是29篇互异方法。NeLiF、NeLT与Superposed Deformable Feature Fields已由用户提供的正式正文解锁，并在逐页独立复核后进入事实矩阵。

本文件比较的是表示事实、query contract、坐标机制与容量分工，不把不同数据、硬件、质量 protocol 的论文结果拼成排名。表中 `P/C gap` 只表示个体报告已经保留的论文—代码差异，不由本综合重新裁决。

## 2. Query domain 与目标量总表

| 方法 | 原生 source / 拟合对象 | runtime query | 方向/空间坐标 | 输出量 | 表示身份 |
|---|---|---|---|---|---|
| NBRDF 2021 | 单个 measured BRDF | `(h,d)` | Rusinkiewicz half/difference，两个 Cartesian unit vectors，共6D | RGB BRDF；训练target为`log1p(f cosθ_i)` | per-material tiny evaluator；另有32D跨材质embedding和2参数analytic proposal |
| NeuMIP 2021 | microgeometry或measured BTF的filtered MBTF | `(u,σ,ω_i,ω_o)` | 2D spatial + continuous Gaussian footprint；方向各取projected-hemisphere xy；view-conditioned geometric offset | RGB reflectance/MBTF | per-material feature pyramid + offset texture + 两个MLP |
| Neural Layered BRDFs 2022 | BRDF与Guo layered reference的dense direction function | `(V_f,ω_i,ω_o)`；layerer另接`V_top,V_bottom,A,σ_T` | evaluator用raw Cartesian directions；sampling proxy在projected half-vector域 | per-channel bare BRDF；RGB独立 | shared large evaluator + per-BRDF 3×32 latent；latent-space binary layer compiler |
| Neural Prefiltering 2023 | 一个完整PBR asset的体素内aggregate transport | `(c(V),ω_o,ω_i)` + visibility `(p_i,p_o)` | object-space voxel center、两方向SH、两端点shared hash encoding；7个discrete LoD | voxel throughput `φ_V` 与segment visibility，不是local BRDF | per-asset transport/visibility fields；asset-prefilter/scene path component |
| Neural Biplane BTF 2023 | measured/synthetic/captured BTF | `(u,h,d)` | 2D U-plane、2D half-vector H-plane、difference 2D conditional；可选direct 2D offset | RGB BTF/ABRDF response | shared universal decoder + per-BTF planes/color adapter |
| MetaLayer 2023 | 固定 two-interface + one-medium layered family | `(Γ,ω_i,ω_o)` | physical parameters经MetaNet生成state；directions转`ω_h,ω_d`并各做84D SH | per-channel BRDF/BTDF scalar | family-shared hypernetwork/BSDFNet；per-material generated partial weights/features |
| NVIDIA RTA 2024 | spatially varying reference material graph的filtered response | `(x,footprint,ω_i,ω_o)`；sampler另接随机数 | 8D hierarchical latent；两个learned shading frames；每frame投影两方向 | paper/supp measure有`f` vs `f cos`边界；analytic 2-lobe sampler | per-asset baked latent hierarchy + compact evaluator/sampler |
| Comprehensive Neural Materials 2025 | measured/synthetic BTF + optional explicit height field | `(u,h,d)`；height shell traversal | U/H/D三plane，Shirley square map；dynamic synthesis只作用U；height另走coarse/fine grid | RGB BTF response；具体radiometric measure未报告 | per-material QTP + Int8 decoder + synthesis/height runtime |
| BSDF Importance Baking 2023/2025 | 三个固定parametric BSDF family的sampling/eval/pdf maps | `sample(ε,ω_o,ξ)`、`evaluate/pdf(ε,ω_o,ω_i)` | material encodings + unit-disk direction coordinates + Fourier/one-blob encodings | sample返回direction+预烘`f cos/p`；eval为`f cos`；独立PDF网仅供MIS | per-family sampling/evaluator/PDF networks；以sampler为中心 |
| Hybrid Neural-Microfacet 2026 | measured BRDF collection | `(p,z,ω_i,ω_o)` | raw Cartesian directions；无positional encoding | `f_c+f_g f_a`，bare RGB BRDF | shared shallow residual/gate MLP + per-material analytic parameters与4/8D latent |
| Taming Optimization Variance 2026 | NVIDIA式spatial reference material | `(z_x,ω_i,ω_o)` | final正式输入同时含direct directions与stable half/difference，均在两个learned frames下；8D latent | nonnegative RGB BRDF `exp(z)` | representation沿用NVIDIA；贡献集中在可优化性、coordinate continuity与activation |
| Neural Material Adapter 2026 | 固定三层PFMC native parameter family | `(P,ω_view)` | material native parameters + 一个view angle/direction；正式encoding未报告 | 两组Principled参数+blend，解析target给RGB BRDF/sample | class-shared material→analytic-program adapter；无PFMC per-material latent |
| Mobile VR Neural Materials 2026 | UBO measured BTF | `(u,ω_i,ω_o)`，按2×2 texel摊销 | coarse center共享`ω_i,ω_o,h`；fine stage不再读方向 | RGB BTF；写入object-space radiance texture | per-material coarse/fine planes+tiny MLP；spatial与temporal系统摊销 |
| Neural Processes BRDF 2021 | MERL100+EPFL51 measured BRDF function set | context observations→7D Gaussian latent；runtime `(z,x)` | `x=(θ_h,θ_d,sin2φ_d,cos2φ_d)`；current release另有input order/affine scale和dot-mask P/C gap | main decoder输出bare RGB BRDF；NICE target为归一化`luminance(f)cosθ_i` | set encoder+mean aggregator+shared decoder；另有per-material hyper-generated tiny mainNet与post-trained NICE |
| Improving Angular Parameterization 2025 | UBO2014 Leather11/Fabric12 | 7D neural-texture feature + `D<10` angular tuple | 十种direct/half-difference spherical、PE、latent与Cartesian；D6 direct Cartesian或D9 direct+half按材质占优 | RGB reflectance，cosine/linear convention未报告 | two-material tiny-MLP coordinate diagnostic；不是完整material representation release |
| Hierarchical Neural Materials 2024 | per-material spatial-angular appearance/NeuMIP式pyramid | 整幅2D buffer中的逐像素`(u,ω_i,ω_o,σ)`；offset/pyramid提供level-dependent latent | P：normalized 2D `u/ω` + `π/no-raw` Fourier；hemisphere map/frame未报告。C：恢复3D方向、with-raw/no-π；Inception跨邻域读取 | linear RGB reflectance；exact `f/fcos/radiance` measure未闭合，C另有optional cosine/output-transform gap | per-material offset+pyramid + double-Inception neighborhood decoder；依赖整幅buffer排列而非独立single query |
| Guo position-free 2018 | ordered interfaces + homogeneous slabs的native transport | stochastic `evaluate/sample/pdf(ω_i,ω_o)` | 只保留depth、direction、layer/media identity，删除横向位置 | layered BSDF/path weight/solid-angle PDF | source-family reference oracle；不是固定latent或部署表示 |
| Xia Gaussian Product 2020 | Guo式position-free layered BSDF的内部fixed-direction path integral | fixed external `(ω_i,ω_o)`与path topology下，pair-product采一个内部方向；multiple-product联合采两个；外部`sample/pdf`仍是Guo路径接口 | isotropic outgoing slope `s(ω)=(ω_x/ω_z,ω_y/ω_z)`；pair为2D、正式multiple为4D joint Gaussian；projected/solid-angle reconciliation有缺口 | Gaussian模块输出internal direction proposal/density；最终仍估计原始layered BSDF，不输出近似appearance value | per-BSDF-family polynomial Gaussian-slice proposal；reference variance control，不是neural/material representation或外部matched sampler |
| Belcour layered approximation 2018 | plane-parallel isotropic GGX interfaces + optional homogeneous slabs | view-conditioned adding-doubling后`eval/sample/pdf(ω_i,ω_o)` | projected-direction energy、2D mean、scalar variance；每个path group重建为GGX lobe | GGX mixture；realtime裁为3层/2个outgoing lobes | shared FGD/TIR tables + per-query lobe state；解析统计control/proposal，不是GT |

证据入口：各行分别回链到个体报告 §§4–5：[NBRDF](../papers/2021-neural-brdf-representation-importance-sampling.md)、[NeuMIP](../papers/kuznetsov-2021-neumip.md)、[NLB](../papers/fan-2022-neural-layered-brdfs.md)、[Neural Prefiltering](../papers/weier-2023-neural-prefiltering-lod.md)、[Biplane](../papers/fan-2023-neural-biplane-btf.md)、[MetaLayer](../papers/2023-metalayer.md)、[NVIDIA RTA](../papers/zeltner-2024-real-time-neural-appearance-models.md)、[Comprehensive](../papers/xu-2025-comprehensive-neural-materials.md)、[Importance Baking](../papers/bai-2023-bsdf-importance-baking.md)、[Hybrid](../papers/2026-hybrid-neural-microfacet-brdf.md)、[Taming](../papers/bitterli-2026-taming-optimization-variance.md)、[NMA](../papers/2026-neural-material-adapter.md)、[Mobile VR](../papers/xu-2026-real-time-neural-materials-mobile-vr.md)、[Neural Processes](../papers/zheng-2021-neural-process-brdfs.md)、[Angular Parameterization](../papers/xu-2025-improving-angular-parameterization.md)、[Hierarchical](../papers/xue-2024-hierarchical-neural-materials.md)、[Guo](../papers/guo-2018-position-free-layered-bsdfs.md)、[Xia](../papers/xia-2020-gaussian-product-sampling.md)、[Belcour](../papers/belcour-2018-efficient-rendering-layered-materials.md)。

## 3. 坐标机制并不是同一种“encoding”

### 3.1 Fixed canonicalization：把高光结构驻定

NBRDF、Biplane、Comprehensive 和 MetaLayer 都使用 half/difference 思想，但其容量位置不同：

- NBRDF 把完整 `(ω_i,ω_o)` 固定变换为 Cartesian `h,d` 后直接送入每材质 `6→21→21→3` MLP；没有direction plane或learned frame。坐标先验承担高光对齐，675个网络scalar承担剩余shape。[NBRDF §§4–5]
- Biplane把 `h` 放进显式 `20×20×6` H-plane，把 `d`保留为2D MLP condition；高频half-vector dependence不必全部穿过shared MLP。per-texel 12-scalar color adapter又承担大量色域容量。[Biplane §§4–5]
- Comprehensive把 `u,h,d`都放进8-channel planes，并用Shirley concentric map把方向半球变成方形lookup域；其decoder只是`24→32→32→32→3` Int8 QL。坐标plane已承担主体容量。[Comprehensive §§4–5]
- MetaLayer用`ω_h,ω_d`后仍各展开84D SH，共168D；该方向basis与MetaNet生成的partial weights/features共同进入BSDFNet。这里fixed basis换取了更规则的direction function，但正式BSDFNet exact layer mapping仍未被正文/代码锁定。[MetaLayer §§4–5,11]

所以“使用half/difference”不能直接推导网络更小或质量更高。它至少有四种不同作用：纯input canonicalization、方向plane地址、固定basis expansion、或与hypernetwork生成state共同条件化。

2025 Angular Parameterization poster把这个边界压到更小预算：固定7-channel neural texture和三层`8×8` hidden MLP，对两个UBO材质比较十种`D<10`输入。Leather11的D9 direct+half Cartesian最低，Fabric12则是D6 direct Cartesian最低；不给half/difference显式加入`cosθ_i`时明显较差。它支持“tiny网络应直接暴露低阶充分统计，而且half clue的价值随材质而变”，不支持单一坐标在所有材质普遍占优。poster没有seed、iso-MAC width或更多材质，因而只能作为当前coordinate ablation的受限先例。[Angular Parameterization §§3–5,9–10]

### 3.2 Stable canonicalization：连续性比语义优雅更重要

Taming指出经典Rusinkiewicz在perfect-reflection邻域因`φ_h`未定义而产生输入不连续；其shortest-arc `ω_d'`消除了该singularity。但normal mapping会把真实lobe旋离canonical alignment，因此正式建议并非“只用half/difference”，而是在两个learned frames下同时输入direct `(ω_i,ω_o)`与stable `(ω_h,ω_d')`。[Taming §4.1, §5.4]

这是一个重要设计边界：fixed coordinate的目标不是减少input dimension，而是让目标关于input更平滑；当source包含normal/tangent变化时，保留direct directions能防止canonicalization错误成为不可恢复的信息瓶颈。

### 3.3 Learned frames：把乘法式旋转从小MLP中拿出来

NVIDIA RTA与Taming的两个learned shading frames都由8D latent经无bias `8→12`线性层产生。每组normal/tangent分别normalize，frame不强制正交；2024正文与supplemental对bitangent是否normalize还有未解析冲突。两个frame下的方向投影不是可解释normal-map恢复，而是显式提供material-conditioned旋转，使小MLP不必用ReLU/LeakySmeLU重新逼近乘法和旋转。[RTA §§5.3–5.4; Taming §§5.3–5.4]

当前可得证据支持“learned frame是计算先验”，不支持“frame本身忠实恢复source normal/tangent”。因此后续candidate若监督frame语义，应视为新增假设而非NVIDIA faithful replication。

### 3.4 View-conditioned spatial warp：把parallax放进地址而非颜色MLP

NeuMIP从单层7-channel offset texture取feature，用`9→25→25→25→1`预测scalar depth `r`，再经固定几何函数

\[
\Delta u=\frac{r}{\max(\omega_{o,z},0.6)}(\omega_{o,x},\omega_{o,y})
\]

移动feature-pyramid地址。其优势证据是direct unconstrained 2D offset更差；`r`不被height supervision约束，不能解释成真实geometry。[NeuMIP §§5.1–5.4]

Biplane另有per-BTF direct 2D offset MLP，但没有与NeuMIP scalar-depth prior做matched对照；Comprehensive则拒绝把位移压进appearance network，改用显式height field与coarse/fine shell traversal。三者代表不同语义：appearance warp、自由2D regression、显式geometry intersection，不能用同一个“offset”标签合并。[Biplane §5.4; Comprehensive §§4–5]

### 3.5 Slope-product coordinates：它优化的是内部积分，不是外部材质query

Xia把isotropic microfacet BSDF在固定incident direction下的outgoing slice写到二维slope space `s(ω)=(ω_x/ω_z,ω_y/ω_z)`，用1–2个Gaussian拟合。pair-product对两个相邻factor的slope densities做解析乘积，sample一个internal direction；正式multiple-product只对三因子component构造4D joint Gaussian并sample两个internal directions。这里external `(ω_i,ω_o)`与path topology已固定，最终仍用native BSDF factors求原始layered integral。[Xia §§4–5]

因此它与NBRDF/NVIDIA等外部`sample(ω_i)→ω_o` proposal不是同一个query domain。Xia的external material `sample/pdf`仍是Guo-style forward random walk与独立stochastic PDF estimate；pair/multiple-product首先是`evaluate(ω_i,ω_o)`内部的reference-variance mechanism。slope density到solid-angle/projected measure的reconciliation也未由code闭合，不能把论文的internal density直接登记为本项目matched `pdf(ω_o|ω_i)`。[Xia §§4–5,11,13]

### 3.6 Object/voxel coordinates：这已不再是local material

Neural Prefiltering的appearance query用voxel center和两个方向，visibility query用object-space entry/exit endpoints；far-field approximation主动删除appearance的boundary positions。输出`φ_V`聚合体素内部visibility、BSDF、geometry term和multi-bounce transport。即使它也接收两方向，它的function domain不是`evaluate(wo,wi)` local scattering；把其hash/SH网络直接与NBRDF、NVIDIA decoder比较会混淆物理对象。[Neural Prefiltering §§4–5]

同理，Guo删除的是横向位置，但保留每条内部path的depth/direction/boundary state；其容量由随机path expansion提供，不是一个固定direction embedding。它适合source-family reference，不是可部署coordinate recipe。[Guo §§4–5,13]

Hierarchical Neural Materials表面上也接收空间与方向，但核心是whole-buffer double-Inception：当前query会读取周围像素，且ablation中去convolution会丢后侧纱线细节。这是一种spatial neighborhood decoder，不满足本项目“小MLP对任意单点随机访问”的运行合同；它可作为training/filtering与空间感受野证据，不能按一组独立direction features直接移植。Formal与release都锁定exact branch channels为`7:12:3:3`，论文只是另把它近似概括成`2:4:1:1`。真正未闭合的是direction hemisphere map/frame、继承NeuMIP的offset/pyramid/filter配置、P/C Fourier basis与output measure，不是Inception kernel/channel topology。[Hierarchical §§4–5,8,10–11]

Neural Processes BRDF又是另一类函数坐标：context observation set先被逐点encoder编码并mean aggregate成7D Gaussian latent，decoder再以latent和4D Rusinkiewicz query还原BRDF。latent因此代表“观测集合条件下的函数后验”，不是native material parameter或per-texel state。paper encoder/aggregator与release checkpoint的深宽不同，且main result让151个measured BRDF全部参与训练；把该latent用于G2 compiler前必须重建held-out source split，不能沿用其asset-compression数字。[Neural Processes §§4–6,9,11]

## 4. 空间、尺度与摊销的表示分工

### 4.1 Per-texel latent 与 independently learned LoD

NVIDIA RTA每个texel/level为8D latent，两张RGBA FP16 mip textures；每个coarse level学习对应footprint下的filtered response，而不是level-0 latent downsample。runtime对fractional LOD以Russian roulette选一个相邻整数层，只读一个level，避免在语义不同的latent间trilinear interpolation。[RTA §§5.1–5.2,5.4]

NeuMIP也让每个pyramid level独立优化，但在`log2 σ`上对两个相邻levels做linear blend；每level先bilinear。两者恰好体现相反取舍：NVIDIA用随机离散level避免latent interpolation bias，NeuMIP用确定性跨层插值得到continuous scale。不能只以“都有mip latent”归并。[NeuMIP §§4–5; RTA §§4–5]

NLB supplemental则对96-float-per-texel latent建立普通mipmap并trilinear；没有报告coarse levels是从GT独立学习还是对latent downsample。该缺口直接影响filter semantics，不能自动解释成NVIDIA/NeuMIP任一方案。[NLB §§5.1–5.2,11]

Hierarchical正式声称offset与neural texture pyramid保持NeuMIP原样，却没有重新披露channels、resolution、level interpolation或filter kernel；official release又包含额外blur/σ/mip switches，不能反填formal identity。它新增的可靠事实是whole-buffer多尺度convolution与Fourier/loss机制，不是一套已闭合的第四种LoD lifecycle。[Hierarchical §§5–6,11]

### 4.2 Plane factorization

- Biplane：`U(u)` + `H(h)` + conditional `d`，shared MLP跨84个BTF；per-BTF color adapter显著扩大per-texel容量。
- Comprehensive：`U(u)+H(h)+D(d)`三plane全显式；只有U参与dynamic synthesis；H/D不合成。
- NeuMIP：只有spatial multi-resolution pyramid，方向只进MLP；offset另有单层texture。
- NVIDIA：spatial hierarchy存generic latent，方向frame与MLP从latent共同解码；不单独为half-vector分配plane。

这四种factorization的关键差别不是plane数量，而是“哪个变量获得随机访问容量”。Biplane/Comprehensive把特定direction coordinates直接变成texture address；NeuMIP/NVIDIA把direction dependence留给MLP，texture只索引spatial/material state。

### 4.3 Spatial amortization 与 temporal amortization

Mobile VR把一个`2×2` texel square的方向计算和coarse MLP只做一次：`16→8→8→8`得到共享`z_c`，四个fine texel各用`16→8→3`恢复RGB。方向在square center计算，作者明确承认fine detail存在minor angular aliasing。这是确定性spatial amortization，不是model capacity本身。[Mobile VR §§4–5]

部署再把object-space neural shading写入每眼radiance texture并复用N帧；于是论文系统成本由三个域组成：per-square neural inference、per-eye object-space texture generation、跨frame reuse。该系统不能等价改写为单次`evaluate(wo,wi)` evaluator速度。[Mobile VR §§5,8]

## 5. 材质身份放在哪里

| 身份机制 | 方法 | 具体状态 | 泛化/编辑边界 |
|---|---|---|---|
| 每材质完整表示 | NBRDF evaluator、NeuMIP、Comprehensive | network或planes随材质保存 | 新材质通常需要optimization；没有source-param compiler |
| shared decoder + per-asset dense state | Biplane、NLB、NVIDIA RTA | planes、96D/texel latent、8D/texel hierarchy | decoder共享范围分别是dataset/BRDF族或单个baked asset训练；不能笼统称cross-material |
| physical parameters→generated program | MetaLayer、NMA | MetaNet生成293 scalars/channel；NMA直接输出两组Principled参数+blend | 只在各自固定layer topology/parameter family泛化；保留native parameter编辑但不覆盖任意source family |
| analytic core + free embedding | Hybrid | per-material`p={k_d,η,α}` + `z4/8` | 未见measured material仍需优化`p,z`，不是encoder/compiler一次前向 |
| source graph→baked latent hierarchy | NVIDIA RTA | training encoder→per-texel 8D latent hierarchy，部署丢弃encoder/source graph | runtime紧凑，但native编辑需重新bake/优化；不提供class-wide online compiler |
| baked latent，层级语义未完全披露 | Taming | 正式表示读取8D `z_x`；论文没有锁定mip层数，公开代码默认配置也不能反推正式层级 | 只能继承“需重新优化/烘焙”的身份，不能静默继承NVIDIA RTA的逐层GT与层级配置 |
| observation set→function posterior | Neural Processes | context set经encoder/mean aggregator得到7D Gaussian latent；可再materialize为2259-scalar mainNet | main训练覆盖全部151 measured BRDF；novel family会有color/energy/extrapolation失败，不等同native editable compiler |
| per-material spatial hierarchy | Hierarchical | multi-level feature/offset state + whole-buffer Inception neighborhood | 可利用空间邻域与mip线索，但formal hierarchy/filter identity未闭合，且不满足独立single-query shader形态 |
| parameterized family→sample program | Importance Baking | material encodings进入per-family sampling/eval/PDF networks | 固定三个family；主要输出sample tuple，不是统一appearance representation |
| native family→internal proposal coefficients | Xia Gaussian Product | reflection/refraction与roughness regimes分开的polynomial Gaussian-slice fits；runtime由native BSDF parameters求proposal | 没有per-material learned latent或appearance decoder；只优化position-free reference内部积分，外部`sample/pdf`身份不变 |

MetaLayer与NMA最接近“material compiler”，但也最能说明compiler的domain必须显式：MetaLayer只覆盖一个均匀介质夹在两个isotropic GGX interfaces间；NMA只覆盖固定三层PFMC topology，且output Principled schema/constraint未披露。它们不能证明所有native source都应归约为layer parameters或Principled lobes。[MetaLayer §§4–5,12; NMA §§4–5,12]

## 6. Analytic prior 的四种不同角色

### 6.1 最终 evaluator core

Hybrid的`f_t=f_c+f_g f_a`把diffuse+exact dielectric Fresnel GGX作为显式base；MLP输出positive correction与RGB gate。作者报告更复杂Disney core导致numerical instability，因此analytic prior不是“越强越好”，而是需要可优化parameterization。[Hybrid §§5.1–5.3,10]

NMA更进一步：网络不直接输出response，只输出两组Principled参数和blend；最终evaluate/sample完全由analytic target承担。它换取了runtime sampleability与强shape prior，但正式报告也保留了direction-conditioned non-reciprocity、parameter schema缺失和正文/补充材料冲突。[NMA §§4–5,10–12]

Belcour则把每个layer/path group压成投影方向分布的energy、mean与scalar variance，再重建为GGX mixture。它没有learned residual，低/中roughness时能高效传播layer effects，但high roughness、grazing skew、Fresnel color shift和HG heavy tails会系统偏离；因此更适合`optimized-code control`、proposal或hybrid low-frequency core，不能取代当前层栈random-walk reference。其realtime 3层/2-lobe版本静态有界，offline任意层版本则随layer/lobe count增长。[Belcour §§5,8,10,13]

### 6.2 仅作 proposal

NBRDF的32D embedding经`32→8→2`只预测monochrome Blinn–Phong proposal参数；evaluator仍是NBRDF。NLB的offline sampler把latent压成Gaussian+Lambertian mixture；MetaLayer外接Belcour-style R/TRT/TT lobes；NVIDIA另有learned 2-lobe analytic density。它们都体现同一数学边界：proposal可以近似目标shape，只要`sample/pdf`自身匹配，便不要求evaluator也变成analytic lobe。[NBRDF §5.1.3; NLB §5.4; MetaLayer §§4–5; RTA §5.4]

Importance Baking是更激进的sampler-centered design：sample网络直接回归direction与预烘RGB`f cos/p_target`，避免runtime再调用其independent evaluator/PDF网；独立PDF网只为MIS query训练，未被证明与sampling map归一匹配。这种source-native tuple思路值得研究，但不能把独立PDF网误称为sampling map的认证density。[Importance Baking §§4–5]

Xia的Gaussian product也是proposal-only，但domain更窄：它在固定external `(ω_i,ω_o)`后，为layered evaluator内部相邻BSDF factors生成一个或两个internal directions；final contribution仍用exact native factors。论文另行保留Guo-style external `sample/pdf`。所以它可作LayerStack reference的variance control，却不能与上述外部direction samplers合并成同一个`sample()/pdf()`候选，multiple-product也没有任意长chain证据。[Xia §§4–5,13]

### 6.3 仅作坐标/函数先验

Learned frames、stable half/difference、NeuMIP scalar-depth warp和SH basis都没有显式analytic BRDF output；它们约束的是输入geometry或function smoothness。将这些机制与Hybrid/NMA的analytic core合称“physics-informed”会掩盖实际约束位置。

## 7. Output measure 与 ABI 不能从“BRDF”一词推断

| 方法 | 个体报告已锁定的输出语义 | 对项目bare `f`的边界 |
|---|---|---|
| NBRDF | runtime输出RGB BRDF；loss监督`log1p(f cosθ_i)` | 训练target transform与runtime inverse必须对应；proposal另算 |
| NLB |正文目标为bare `f`；release projection example在loss前乘`light_z` | P/C gap，不能把公开example当正文唯一training ABI |
| NVIDIA RTA |正文以`f`叙述；supp Listing返回`f·cos(query-light)`；当前项目另有bare-f identity | 必须版本化cosine adapter或直接bare-f训练，不能混名 |
| Biplane |论文称BTF/BRDF value；release loss用output×第二方向z | 正式checkpoint measure未由代码唯一闭合 |
| Comprehensive |RGB BTF response，radiometric unit/cosine未报告 | 需与source/reference做数值correspondence |
| Importance Baking |eval明确为`f cos`；sample明确为`f cos/p` | 与项目ABI的tuple接近，但独立PDF网不自动匹配sampling map |
| Hybrid |Eq.1明确bare `f` | 最直接适配local evaluator，但analytic domain有限 |
| NMA |解析target给线性RGB BRDF；sample由target承担 | target code/schema不可得，仍需sample/eval/pdf parity |
| Mobile VR |BTF定义含其foreshortening convention，最终写radiance texture | 不可直接当bare local BSDF；需分离BTF与lighting pipeline |
| Neural Processes |main decoder为bare RGB BRDF；NICE另取`luminance(f)cosθ_i`并归一化为density | evaluator与density必须分ABI；NICE code/PDF normalization oracle不可得 |
| Angular Parameterization |只称RGB reflectance，cosine/linear convention未报告 | 只能用于coordinate ranking先例，项目adaptation必须独立冻结bare-`f` target |
| Hierarchical |paper称RGB reflectance，但release可选cosine multiplier且paper/code未闭合 | whole-buffer output与single-query bare-`f`均需独立correspondence，不能从代码变量名裁决 |
| Xia Gaussian Product | Gaussian只输出internal direction proposal/density；final estimator仍为原始layered BSDF。P在projected measure与slope-to-solid-angle density间留有reconciliation gap | 不能把internal PDF当外部`pdf(ω_o∣ω_i)`；若迁移到reference须显式对齐solid-angle measure，部署evaluator仍不由本文提供 |
| Neural Prefiltering |aggregate voxel throughput | `not-applicable`于bare local `f` |

因此当前项目候选注册必须把四件事分开：论文数学量、训练target transform、public code return convention、项目canonical ABI。只要其中任一未闭合，就不能以文件名或变量名完成correspondence。

## 8. 对当前 NeuralShading 方法方向的约束 `[N/I]`

### 8.1 没有证据支持一个强制统一的latent或closure词汇

已复核的可部署 learned methods 把容量放在完全不同的对象中：per-texel generic latent、direction planes、per-BRDF network weights、physical parameters生成的partial weights、analytic lobe parameters、spatial warp、aggregate voxel transport或sample map。它们的共同目标是把昂贵source/reference query转成更便宜的运行程序，而不是都输出同一种layer closure；其中 Guo 是 stochastic source-family reference，Xia只编译其内部积分proposal，Belcour是有偏解析statistical control/proposal，三者都不能硬归入“learned compilation”这一共同点。

这与项目根本目标一致：统一的是`prepare/evaluate/sample/pdf`调用与静态预算，不是把所有source native semantics改写成LayerStackIR或Principled参数。

### 8.2 `prepare(P,wo)`最有价值的三类复用

1. **坐标复用**：latent→learned frames、stable half/difference中只依赖`wo`的部分；
2. **地址复用**：view-conditioned offset、footprint/LOD level selection、plane/latent filtering；
3. **proposal复用**：analytic mixture parameters、lobe energies/roughness和完整mixture normalization。

Mobile VR的2×2 spatial share与texture-space N-frame reuse属于更外层系统摊销，不应塞进local `prepare` benchmark；scene-level history/reprojection则按§9的整帧state/cost域登记，不能反填local ABI。

### 8.3 最值得保留的竞争表示，而非过早选赢家

后续model candidates至少应保持五条互相独立、runtime class明确的最大形态：

- compact direct evaluator：NVIDIA/Taming式latent + explicit coordinate priors；
- analytic-core residual：Hybrid式base/gate/correction，但使用本项目native layer reference而非measured-only identity；
- compiler-generated state：MetaLayer/NMA式native parameters→bounded program，严格限制source family；
- plane/factorized evaluator：Biplane/Comprehensive式把high-frequency direction或space容量放进random-access planes。
- bounded spatial-context evaluator：只迁移Hierarchical的material-space固定stencil/多尺度感受野思想，把读取数、UV seam、footprint与`prepare()`复用静态冻结；不复刻依赖screen-buffer邻接的whole-image Inception runtime。

前四条是local evaluator/compiler表示；第五条只有在固定material-space读取且保持random-access ABI时才是产品候选，否则降为training/filtering diagnostic。每条必须在相同source/query/parameter state split下比较local scattering quality、single-query cost、state bytes和G1/G2/G2s；不以论文跨硬件数字预判排名。sampler作为独立轴使用cosine、analytic mixture与learned proposal matched controls；Xia首先留在reference variance-control轨，不冒充external sampler。

### 8.4 预先可证伪的表示问题

- stable direct+half/difference是否只改善optimization variance，还是也在最终同budget误差上稳定获益；
- learned frames在当前LayerStack是否提供超越fixed frame/half-difference的收益，还是只学到冗余旋转；
- analytic core能否降低MLP容量而不把high-roughness、multiple-scattering tails压回错误lobe；
- per-level independently learned latent与filtered GT是否优于latent mip downsample；
- half-vector direction plane是否值得其texture reads/bytes，尤其在spatial state已有8D latent时；
- native parameters→program compiler能否跨G2未见parameters泛化，同时保持G2s topology boundary而不偷换GT。
- Neural Processes式observation-set posterior在严格held-out material/source split下是否仍有价值，还是只复现参与训练的function manifold；
- Hierarchical式固定material-space stencil在静态读取预算内是否保留收益，还是依赖screen layout、邻居query排列或未计费的whole-buffer coherence。

这些问题将在 `reproducible-hypotheses.md` 中配成正式matched controls；本文只锁定representation事实与比较轴。

## 9. Scene/volume representation 轴

场景级方法需要新增、且不能由local material报告代替的字段包括：

- scene/object/light identity如何进入representation；
- geometry、material、visibility、direct/indirect transport的可编辑轴；
- feature field/probe/image-space buffer的空间坐标和cross-scene normalization；
- history/reprojection、shadow clues、G-buffer与temporal state；
- output是radiance、indirect lighting、lighting function、volume image还是local scattering；
- inference cost按pixel、probe、object、light、frame还是scene amortize；
- novel scene/light/geometry的泛化边界。

当前已经复核的十个 scene/asset/volume 方法构成下表；三份此前的 `blocked-source-audit` 已由用户提供的正式正文解除，但 supplemental/code 缺口仍按各报告保留：

| 方法 | scene identity/state | runtime coordinates/query | 输出 | history/temporal | 动态与泛化边界 | 成本单位 |
|---|---|---|---|---|---|---|
| Neural Prefiltering 2023 | per-asset七层稀疏voxel LoD occupancy/threshold，以及两套独立的appearance/visibility HashGrid+MLP | object-space voxel center+directions；visibility另查entry/exit endpoints；variable traversal | aggregate voxel throughput + segment visibility | 无image history；LoD离散 | 静态asset；无cross-asset holdout，geometry change失效 | network query + variable path/voxel traversal |
| CNSR 2020 | 三张observation的beauty+position+normal+ID与camera经Pool encoder/mean得到128–512D global `r`；`r`可分L/G/M/(optional null)连续partitions，generator weights另吸收dataset规律；正式observation SPP未报告 | novel-view camera16 + query position/normal/ID G-buffer + global `r`；Pixel每pixel独立但读取scene-global state | HDR beauty；application variant可输出indirect或auxiliary shadow image，均非local scattering | 无history/reprojection；Pixel的temporal/resolution行为仅定性 | constrained procedural scene family；新scene需三张observations，gray-wall OOD失败；非zero-shot arbitrary scene | observation render/encoding一次 + per-view query G-buffer + Pixel/U-net/GQN image generator；400 ms只锁定1k² indirect prediction而非full pipeline |
| Efficient Light Probes 2022 | fixed scene的2048² lightmap、regular-grid probes、per-scene CNN | current-view G-buffers；在附近8 probes做screen-space gradient reflection search；基本迭代`N_max=20`，material-ID/viewport缩步子搜索是否受同一上限约束未报告 | final 1080p scene radiance image | motion-warped history + temporal loss；仍有small-highlight flicker | fixed geometry/material/lighting bake；不支持dynamic glossy probe update或cross-scene | probe search + full-frame CNN + history |
| Active Exploration 2022 | per-scene约2.46M-weight MLP；normalized scene-variable vector `v`与G-buffer共同决定image；active selector/replay只属training data policy | 每帧首交点position/normal/reflectance/roughness/outgoing-direction G-buffer + broadcast `v`；C中position按scene bbox归一化、normal映射到`[0,1]`，position既走512D预条件支路也保留raw input；C把方向命名`wi`而P为`wo` | outgoing scene radiance `L_o`，训练在`log1p` domain；P称emission passthrough，C是thresholded log-domain merge，保留P/C gap | 无history/reprojection；训练selector/replay不是runtime state | 同一scene已定义变量内变化；新变量/对象通常重训，128D已stress，数千变量广播不可行 | G-buffer + per-pixel scene MLP；训练期另有online PT/MCMC/replay成本，不进入steady runtime |
| NeLT 2023 | foreground/background/light samples与cubemap observations编码成`z_f/z_b/z_g/z_l`；每个dynamic/composite object有object-specific hypernetwork、neural texture和transfer decoder；上一插入阶段的direct/indirect radiance也是composition state | 所有外部条件与query G-buffer变到foreground-object local frame；direct与indirect路径各自生成UV、一次bilinear fetch和pointwise decoder，再按固定object insertion order递推 | 前景direct/indirect为radiance；背景direct为乘法shadow ratio、indirect为加法residual，二者不是同一measure | 无history/reprojection；实际近似对insertion order有轻微依赖，固定order用于一致性，定量结果在不可得supplemental | rigid opaque objects、Lambertian+GGX且排除highly specular；同一object/context family可编辑，novel-scene quality下降；每个对象约60 h训练 | representation extraction/build + per-object full-frame passes；object数线性增长，1024² PyTorch时间不含G-buffer与若干representation成本 |
| Superposed DFF 2024 | source `i`对target `j`生成`r_ij`，再按target `i`对入边`r_ji`求和为`r_i`；每个dynamic object/light或static remainder有conditioned deformable multi-resolution triplane field；所有`F_i`再做第二次element-wise sum。C2F只控制训练期逐级开放容量 | pixel G-buffer转各object local frame；offset decoder将`x_i`变为`x_i+Δx`后查询所有已开放triplane levels，object decoder输出`F_i`，final decoder读summed feature+G-buffer；level尺寸/channels未报告 | final scene radiance；单个`F_i`只是learned feature，不是唯一物理radiance contribution | 无history/reprojection；uniform scene-state训练，rare high-frequency event仍会漏失 | per-scene固定partition；在三个训练scene内覆盖camera、rigid transform、material/light变化；无cross-scene checkpoint证据；field数随partition增长，large scene要求高分辨率triplane | 每pixel查询全部fields再sum-decode；RTX4090/PyTorch FP32为18.9–26.6 ms@512²，但G-buffer、scene-state/hypernetwork、memory与stage scope未闭合 |
| Volumetric inference 1469 | dataset-shared attention U-Net；scene/volume state实时进入auxiliary buffers | 每pixel primary ray生成depth/optical-depth/crossing/direct/environment等feature maps | full volumetric scattered-radiance image | 无history/reprojection；确定性features带来较低reported temporal instability | anonymous稿的split/novel-scene/view边界不完整；features不唯一确定density | auxiliary ray integration + full-frame attention decoder |
| Dual-Band Neural GI 2025 | per-scene、per-object 8-level dense triplanes；material/transform经hypernetwork生成object decoder | first-hit G-buffer + 每pixel一条mirror secondary ray；两次object-local field query；screen-space U-Net；ray miss encoding未报告 | final 512² RGB radiance-like image；`log1p`与head/inverse lifecycle未闭合 | 无history/reprojection，只有video定性temporal观察 | 同一训练scene内camera/rigid object/material dynamics；dynamic light与cross-scene未验证 | one secondary ray/pixel + object field queries + full-frame fusion CNN |
| LightFormer 2024 | per-scene model weights；per-light runtime direct/indirect VPL、RSM与840-channel full light embedding，更新频率未报告 | per-pixel G-buffer query跨lights做8-head attention；direct/shadow/indirect decoders | 三个RGB transport components与final radiance | 公开network输入无history/reprojection；VPL/RSM跨帧复用、RNG/refresh未报告 | 同一scene分布内camera/light/material/animated-object变化；无cross-scene checkpoint证据 | per-light observation/encoding + per-pixel attention + once-per-pixel direct/indirect decode |
| NeLiF 2025 | shared observation encoder/field generator；每个luminaire的多视图radiance/depth先形成4D observation tensor与light tokens，再生成spherical triplane；另有3DGS appearance与global intensity factor | G-buffer world position转`(θ,φ,r)`查询lighting field；direct decoder读G-buffer，indirect沿用LightFormer式RSM/VPL模块；五级shadow hierarchy对hard-shadow clues预测`5×5` filter与`4×4` upsample kernels | Fig.3组合direct、shadow、indirect、Albedo、intensity与luminaire 3DGS；正文未闭合严格radiometric semantics | 无history/reprojection；lighting field与3DGS的生成/更新频率及多luminaire composition未报告 | 5,300 luminaires/10,000 indoor scenes训练并主张novel scene/light，但“unseen”split规则未报告；monolithic radiance architecture仍有high-frequency spectral limitation | per-luminaire generation/bytes未报告 + full-frame decoders；RTX4090D TensorRT half precision为10.56 ms@512²，但field generation、G-buffer/RSM/shadow/3DGS范围未拆分 |

证据：[Neural Prefiltering](../papers/weier-2023-neural-prefiltering-lod.md)、[CNSR](../papers/granskog-2020-compositional-neural-scene-representations.md)、[Efficient Light Probes](../papers/guo-2022-neural-light-probes.md)、[Active Exploration](../papers/diolatzis-2022-active-exploration-neural-gi.md)、[NeLT](../papers/zheng-2023-nelt.md)、[Superposed DFF](../papers/zheng-2024-superposed-deformable-feature-fields.md)、[Volumetric inference](../papers/1469-2026-volumetric-light-transport-inference.md)、[Dual-Band](../papers/mo-2025-dual-band-neural-gi.md)、[LightFormer](../papers/ren-2024-lightformer.md)、[NeLiF](../papers/sheng-2025-nelif.md)。

### 9.1 场景方法的坐标先验是visibility/transport correspondence

Dual-Band先把first-hit与mirror-hit G-buffers变换到每个object local frame，再查canonical triplane；rigid motion后field地址仍稳定。它的principal feature来自first-hit object field，secondary feature来自single-bounce mirror-ray命中；三尺度`5×5` learned kernels根据roughness、view-normal cosine、reflection depth/emission和两类feature过滤secondary branch，再与principal/zero做softmax门控。这个高层机制已锁定，但 supplemental 中三尺度 `gamma` 的 upsample-and-merge没有公式，正文 Eq.(9) 又保留 `p/u` 索引不一致，因此不能从综合文档补猜精确融合实现。这里的“dual band”不是显式频谱基，也不是local half/difference coordinate，而是两种scene query机制产生的相对角频率线索。[Dual-Band §§4–5,11]

NeLT与Superposed DFF共享object-local canonicalization，但状态组织不同。NeLT先把background/foreground/light observations压成global representations，再由hypernetwork生成UV/decoder与neural texture；它按固定object insertion order递推，背景direct transfer是乘法ratio，背景indirect transfer是加法residual，前景两支则分别存direct/indirect radiance，不能把两条更新压成同一种加法。Superposed DFF先由source `i`面向target `j`产生`r_ij`，再以target为中心把`r_ji`求和成`r_i`；conditioned deformation查询每个object field后，才对所有field outputs `F_i`做第二次order-invariant sum。NeLT的固定order是近似一致性条件；Superposed DFF的两次求和处在不同语义层，且都只是learned algebra，均不能被解释成物理上精确线性的transport decomposition。[NeLT §§4–5,13；Superposed DFF §§4–5,11,13]

NeLiF把correspondence单位进一步换成luminaire：有限多视图radiance/depth先按spatial×angular轴形成4D observation tensor，交替attention得到light tokens，再由cross-attention生成以灯具为中心的spherical triplane；当前pixel才用G-buffer、RSM/VPL和五级hard-shadow hierarchy查询/解码。它与LightFormer的直接关系不是“都需逐scene训练”：NeLiF把LightFormer作为正式baseline，沿用其indirect module，并将LightFormer式per-frame/per-pixel light aggregation改为per-luminaire generated field。这个field仍与scene decoders共同决定最终radiance，不是可独立重组的local light transport operator；Table 1也没有证明二者参数量或完整prepare成本matched。[NeLiF §§4–5,8–9,13]

Efficient Light Probes则先用probe panorama和current-view depth/normal做gradient search，显式解决probe→view parallax correspondence，再让CNN修复低spp并融合buffer；其基本候选迭代以`N_max=20`停止，但material-ID/viewport special cases的连续缩步是否计入该上限、是否另有cap未报告，不能把完整search写成已证实的严格固定次数。CNSR用三张observation压成global scene latent，再让query G-buffer提供novel-view可见几何；Active Exploration则直接广播scene-variable vector，并把position同时放进预条件与raw支路。Neural Prefiltering把correspondence放进object voxel traversal；1469把它放进沿primary ray计算的deterministic auxiliary operators。LightFormer把representation单位改成“当前灯对场景的观察”：direct VPL描述emitter，RSM/indirect VPL描述受光表面，shadow/light-direction/half-vector clues提供高频坐标，再由pixel-light attention选择每个像素所需的lights。它们共同显示scene surrogate最主要的坐标问题是“当前像素如何找到包含所需transport的scene state”，而不是把两方向变成更平滑的local chart。[CNSR §§4–5; Active Exploration §§4–5; Light Probes §§4–5,8; LightFormer §§4–5]

### 9.2 Scene identity不可塞进material latent

十个方法的 representation/state dependency 分别绑定 asset geometry、observation-conditioned global scene latent、baked lighting、explicit scene-variable vector、object-transfer composition、per-object deformable fields、volume/camera auxiliary context、dual-band object fields、per-light scene observations或generated luminaire fields；其中1469的auxiliary maps是每帧临时输入，不是persistent latent。它们的输出分别是aggregate throughput/visibility或已经混入visibility、lighting与多跳transport的scene radiance，不能作为`evaluate(wo,wi)→bare f`的material latent替代物。可迁移的是factorization思想：observation/global latent、显式secondary ray、object-local field、probe search或physical auxiliaries先建立correspondence，再由learned component完成融合、聚合或重建；不能迁移的是scene-dependent output measure与整帧receptive field。

LightFormer的per-light VPL/RSM representation已由当前`evidence-reviewed`个体报告锁定；其attention把所有lights聚合后，每pixel只执行一组direct/indirect decoders，但per-light observation/encoding仍按pixel×light线性增长，论文也未冻结产品级`L_max`。attention value路径的未解释concat、projection width与VPL/RSM跨帧更新/缓存仍是公开证据缺口，不在综合中补猜。NeLiF的generated field可能把这部分工作前移到可摊销阶段，但其generation cost、field bytes、更新频率和多灯上界均未报告；NeLT/Superposed DFF的object数量也直接进入pass/field数量。三者都没有定义或验证material-local `prepare(P,wo)`：NeLT的representation/hypernetwork重建cadence、Superposed DFF的scene-state/hypernetwork stage、NeLiF的field/3DGS generation lifecycle均未闭合。因此三篇解锁论文扩展了scene factorization候选，却没有替任何方案证明local random-access ABI或产品级静态预算。

## 10. 完成状态

- [x] 波次1的13篇指定论文全部来自`evidence-reviewed`个体报告；
- [x] Neural Processes、Angular Parameterization、Hierarchical、CNSR与Active Exploration已按独立复核后的证据边界纳入；
- [x] 28篇计数已按19篇local/analytic +10篇scene/asset/volume −1篇重复Neural Prefiltering闭合；
- [x] query、coordinates、persistent state、output measure和analytic-prior角色已分开；
- [x] 未做跨硬件/跨protocol数值排名；
- [x] local material与asset/scene aggregate transport已划界；
- [x] Belcour 2018已按独立review补入analytic statistical control，并保留paper↔code公式与部署边界；
- [x] Xia 2020已补入internal evaluator proposal，并与external material `sample/pdf`分域；
- [x] 已用十篇`evidence-reviewed`报告建立scene/asset/volume矩阵；
- [x] LightFormer只按当前独立review锁定per-light representation，并保留attention与跨帧实现缺口；
- [x] 三份旧blocked-source audit已由用户提供的正式正文升级为完整报告，仍缺的supplemental/code不以相邻方法补齐；
- [x] 本地报告链接逐一解析且均存在；
- [x] 本轮三份来源恢复后的独立cross-document evidence review已完成。

## Evidence review

```yaml
author_update: /root
reviewer: /root/belcour2018_review
review_date: 2026-08-29
sources:
  - 当前28篇evidence-reviewed个体报告的front matter与§2/§9回链；计数复核为19+10-1=28
  - §9十篇scene/asset/volume报告的§§4、8、12及其Evidence review
  - NeLT、Superposed DFF、NeLiF报告的§§4–5、8、11–13与独立复核记录
findings:
  - 明确NeLT前景radiance、背景direct ratio乘法与indirect residual加法的有序组合，未把不同measure压成统一sum
  - 把Superposed DFF修正为source-to-target r_ij、target-side r_ji→r_i求和与field-output求和两层语义，并把C2F限定为训练continuation
  - 把NeLiF修正为4D observation tensor→light tokens→spherical field，并锁定LightFormer indirect-module谱系、五级kernel-shadow和未拆runtime scope
  - 十篇scene矩阵的output、动态/泛化边界与成本单位均按个体报告复核；未做跨protocol排名
remaining_gaps:
  - 个体报告已经登记的P/C output-measure、activation、split、temporal与runtime-scope缺口仍未闭合
  - NeLiF、NeLT、Superposed DFF的supplemental/code与完整state/runtime账本仍不可得
status: evidence-reviewed
```
