# Deployment 与 Amortization：跨论文证据综合

## 1. 证据边界与成本域

本文的事实层只回链到已经 `evidence-reviewed` 的个体报告。当前纳入波次 1 local neural material/appearance、Guo/Belcour 解析与 reference 对照，以及已独立复核的 CNSR、Neural Prefiltering、Neural Light Probes、NeLT、Superposed DFF、本地体积推断稿、Dual-Band、LightFormer 和 NeLiF。三份新解锁正文都缺少部分supplemental/code/runtime breakdown，因此只登记main-paper能够证明的成本域，不从相邻方法或示意图补state bytes与included scope。

跨论文最容易误导的是把不同成本域写成同一个“实时”数字。本文固定以下口径：

| 账本项 | 单位 | 必须包含/单列 | 不能静默合并到 |
|---|---|---|---|
| `R_offline` reference/observation acquisition | 每 material/asset/scene 一次 | 测量采集、path-traced GT、probe/lightmap/RSM bake、OT/threshold 等 teacher 构造 | optimizer wall time、runtime frame time |
| `C_fit` training/compile/export | 每 material/asset/scene 一次 | optimizer、latent fit、network training、materialization、量化与 export | `R_offline`；“训练 2 h”不等于完整 compilation 2 h |
| `B_asset` persistent storage | 每 material/asset/scene | shared weights/tables、per-asset fields/textures、mips、precision/alignment | 临时 workspace、只报 parameter scalar 数 |
| `C_prepare` shading point/hit | 一次 hit，随后复用 `N` 次 query | LoD/latent fetch、frame/warp、view-conditioned proposal/analytic state及其 state bytes/lifetime | 单 MLP `C_eval` |
| `C_eval` scalar query | 一次 `evaluate(wo,wi)` 或一次 `sample/pdf` | network、texture reads、analytic ops、output/sample measure | coherent dense batch、整帧 renderer |
| `C_dense` dense/coherent batch | 固定分辨率的一 query/pixel 或 query buffer | batching、coherence、transfer、launch与有效 pixel 数 | divergent path hit 的 scalar latency |
| `C_observe` current-frame scene observation | 每 frame、每 eye、每 light/object | G-buffer、ray query、RSM/VPL、auxiliary render、reprojection/search；说明是否缓存/刷新 | image network inference |
| `C_image` full-image inference | 每 frame/update | CNN/attention/decoder、输入分辨率、backend/precision/workspace | `C_observe` 或完整 renderer total |
| `C_system` frame/update total | 每 frame 或每次 texture update | raster/path tracing、observation、inference、denoise/upscale/history/reuse | 单 query、单 kernel 或 one-time bake |

只有 dataset/scene、resolution、SPP、hardware/backend、precision、coherence、调用次数、更新频率和 included stages 一致时，数字才可横向比较。下表并列数字只用于锁定各自 cost domain，不构成跨论文速度排名。

## 2. Local material部署总表

| 方法 | runtime program | state/asset主要容量 | 正式测量范围 | 静态有界性与主要缺口 |
|---|---|---|---|---|
| NBRDF 2021 | 每材质`6→21→21→3` evaluator + analytic proxy | 675 scalars、2.70 KB/material（正式 code 为 FP32）；proposal 另 2 参数 | i9-9900K、CPU Mitsuba，同论文 `NBRDF+PhongIS` 为 `12.50M rays/s`；不是纯 MLP latency 或 GPU shader | fixed ops；公开 Mitsuba sampler 与论文 proxy 不同，GLSL source/latency 不可得 |
| NeuMIP 2021 | offset texture/MLP + two-level feature pyramid + material MLP | 每材质7+7 spatial channels、多level；3332 MLP weights | RTX2080Ti，1080p一query/pixel evaluator总计约5ms；OptiX称1spp 60FPS | 单query读取/MLP有界；asset bytes随resolution；5ms与60FPS属不同measurement scope |
| NLB 2022 | shared large evaluator + per-texel RGB latent；proposal 离线预解 | 96 floats/texel + shared `1,074,689`-scalar evaluator；formal CUDA precision 未报告 | official correction 撤回正文 5 ms 和 faster-than-Belcour；有效值是 RTX 2080Ti fixed-batch/per-scene GPU time，例如 65,536 queries 为 23 ms | topology 有界但大 network；CUDA kernels/source、packing 与 scalar shader cost不可得 |
| Biplane 2023 | U/H bilinear + direction-conditioned decoder + color adapter | per-BTF planes/adapter；precision/bytes未报告 | 1080p/1spp Mitsuba-buffer+PyTorch约1s；非优化shader | reads/shape有界；Table FLOPs口径与texture/adapter inclusion未报告 |
| MetaLayer 2023 | material 参数经 MetaNet 生成 state；BSDFNet 逐 query；外接 analytic sampler | 每 RGB 材质 879 scalars generated state + shared nets；SV state 另按 texel 扩张 | i9-9900X CPU renderer；只给 scene/image timing，MetaNet 又有“<1 s”与“milliseconds”两种未对齐表述；无 single-query | 可编译 fixed topology，但 BSDFNet exact mapping、precision/MAC 与 shader 实现不可得 |
| NVIDIA RTA 2024 | 以 Russian roulette 从相邻两个整数 mip 中选一个→z8 bilinear fetch→two learned frames→compact evaluator；另解 9-param sampler | 两张 RGBA FP16 hierarchy；network 约 9.3–37 kB；4k z8 texture tile 约 256 MB | RTX 4090、1080p、1 SPP Falcor/DXR path tracer；coherent tensor-core 与 divergent packed-FMA 路径；frame result不可化成 scalar query | formal fixed reads/ops；per-query MAC、cache 与 2024 compiler artifact未报告 |
| Comprehensive 2025 | U/H/D QTP planes + Int8 decoder；可选 coarse/fine height traversal | Supplemental 表内合计约 5.04 MB/measured 或 19.69 MB/synthetic；network 表值 5.68 KB，但与 2,912 个 Int8 scalar 的算术未闭合 | **只在 desktop RTX 4090**：standalone full-screen QTP 为 `0.183–0.988 ms`，范围随 1K/2K/4K 和 synthesis 改变；不是 mobile 结果。另有 renderer inference/tracing 分项与 `2.43/3.4 ms` 内部 scope gap | decoder/plane reads 可有界；fine traversal 最多 32 steps，但完整 shell/control flow仍须登记；formal output/export correspondence 不完整 |
| Importance Baking 2023 | CPU dense MLP sample/eval/PDF networks | network binary；论文“each material”与三个 family 的 checkpoint 粒度未闭合，weights/precision/bytes 未报告 | i9-9900K CPU 整图 render；无 single-query。sampling/eval/PDF 训练分别约 48/12/12 h，但离线 BSDF slice/OT 生成时间另未报告 | network shape有界；sample-map、predicted RGB weight 与独立 MIS PDF 的同一 proposal identity不成立，不能直接作为项目 `sample/pdf` |
| Hybrid Neural-Microfacet 2026 | analytic diffuse+GGX core + shallow residual/gate MLP | one shared network/dataset + per-material analytic params与 4D/8D latent；32×3 shared network约14 kB | RTX 5080 BRDFExplorer、environment map、100 spp、no IS：32×3 hybrid `0.37 ms`，GGX-only `0.03 ms`；resolution/precision未报告。CoopVec另属多-bounce scene protocol | direct evaluator有部署可能；analytic core也有cost；formal host/CoopVec source与完整 MIS mixture不可得 |
| Taming 2026 | 沿用NVIDIA representation，换stable coordinates/power-map/activation与训练schedule | baked latent仍属NVIDIA family；direction tuple/activation改变runtime function，formal配置与release default有gap | 论文重点是optimization trials，不提供新的formalruntime benchmark资产 | stable coordinates/LeakySmeLU固定有界；不能从训练成功率推导runtime Pareto |
| NMA 2026 | 每 view 运行 5×64 MLP 生成两组 Principled 参数+blend，再解析 eval/sample | PFMC branch 为 class-shared weights + per-asset native params/textures、无 latent texture；MERL branch 另有 per-material embedding | 自研 Mitsuba 0.5 CPU full-render与 RTX 4090 full-table aggregate；CPU 型号/resolution、shader single-query/bytes/export均缺失 | PFMC adapter可放 `prepare(wo)`，但方向符号/encoding未闭合；两份 Principled 成本与 sample/pdf code未公开 |
| Mobile VR 2026 | object-texture compute：coarse 一次/`2×2` + fine 四次→per-eye radiance texture，跨 `N` 帧复用 | per-material neural textures + 1.6 KB network，总 material 6.24 MB；另有双眼 runtime radiance texture 8.0 MB/material@`1024²` | Quest 3/O3DE；FPS结果同时改变 texture resolution、light count、reuse与screen/object-space path，并受90 FPS cap截断 | neural call固定有界；无 robust LoD；不是单次 local evaluator benchmark，motion/light更新有效域未形式化 |
| Belcour 2018 | adding-doubling→GGX mixture；realtime 3-layer/2-lobe | shared约64 MiB FGD + 1 MiB TIR tables；per-query lobe state | GTX 980 commercial engine、1920×1080 full frame：layered shader `1.9–2.1 ms` vs standard `1.7–2.0 ms`；Mitsuba CPU Table 2 是另一 scope | realtime variant固定；offline arbitrary-layer loop/lobe数随层增长；production shader不可得 |
| Guo 2018 | position-free random walk/eval/pdf | native layer params；per-query可变path state | stochastic Mitsuba renderer；无neural storage或固定query time | source reference而非fixed shader；path length/reads/control flow可变 |

证据入口为各报告§8、§11–13：[NBRDF](../papers/2021-neural-brdf-representation-importance-sampling.md)、[NeuMIP](../papers/kuznetsov-2021-neumip.md)、[NLB](../papers/fan-2022-neural-layered-brdfs.md)、[Biplane](../papers/fan-2023-neural-biplane-btf.md)、[MetaLayer](../papers/2023-metalayer.md)、[RTA](../papers/zeltner-2024-real-time-neural-appearance-models.md)、[Comprehensive](../papers/xu-2025-comprehensive-neural-materials.md)、[Importance Baking](../papers/bai-2023-bsdf-importance-baking.md)、[Hybrid](../papers/2026-hybrid-neural-microfacet-brdf.md)、[Taming](../papers/bitterli-2026-taming-optimization-variance.md)、[NMA](../papers/2026-neural-material-adapter.md)、[Mobile VR](../papers/xu-2026-real-time-neural-materials-mobile-vr.md)、[Belcour](../papers/belcour-2018-efficient-rendering-layered-materials.md)、[Guo](../papers/guo-2018-position-free-layered-bsdfs.md)。

## 3. `prepare()`式摊销：同一点重复query

### 3.1 NeuMIP的view-only stage

NeuMIP Stage 1由位置、footprint与`wo`决定：offset texture/MLP产生warp，再取feature pyramid。Stage 2才把`wi`加入material MLP。Fig.9的四项显示offset texture约0.3ms、offset MLP约2.3ms、feature pyramid约0.3ms、material MLP约2.3ms；这些是一张1080p dense batch、各自一位小数的分解，总和5.2而正文总计写5ms，不应伪造更精确总值。[NeuMIP §8]

对当前接口，可把warp地址、filtered latent和view condition放进`prepare`，同一shading point的多个light/NEE queries只跑Stage 2。这是结构上的可复用性，不是论文已量化的多灯加速：paper没有给cache state bytes、复用次数或end-to-end multi-light timing。

### 3.2 NVIDIA learned frames与sampler params

RTA一次hit先选LoD并取z8，再从z生成两个frames；对固定`wo`，frame-projected view inputs和sampler 9参数都可复用。论文API没有叫`prepare`，但supplemental functional path支持按hit共享sampler params给`sample`/`pdf`。把这些每个`wi`重复计算会改变真实query cost，却不改变representation质量。[RTA §§4–5,8]

### 3.3 NMA/Belcour的view-conditioned analytic state

NMA PFMC adapter只接 native parameters 与论文所谓 viewing direction，输出两组 analytic 参数和 blend；若按论文 Background/Eq.(1) 的 convention 映射到项目 `wo`，可在 `prepare(wo)` 运行一次。可是正文别处又用 incident wording，且正式 direction encoding、parameter schema 与 sampler implementation 不可得，因此这个复用必须先做方向 convention/parity 审计。Belcour adding-doubling也从 material+view 构造 lobe coefficients/roughness/directions；实时 3-layer/2-lobe variant可注册固定 state，offline arbitrary-layer variant的 lobe 数与工作量则随层数增长。[NMA §13.4；Belcour §§6,8]

### 3.4 Filter、asset compile 与 `prepare` 不能互相改名

- RTA 的 stochastic adjacent-mip choice、z8 fetch、frame projection，以及 NeuMIP 的 warp/filtered latent，都确实发生在同一 hit 的多 query 之前；但论文没有给 prepared-state bytes 或复用 break-even。
- Comprehensive 只有 spatial `U(u)` lookup 可能进入 `prepare`；`H(h)`、`D(d)` 同时依赖 `wo/wi`，仍须逐 `evaluate` 查询。QTP 的 average-mip LoD 也不能补成 RTA/NeuMIP 的相同 filter semantics。[Comprehensive §13.4]
- MetaLayer 的 MetaNet 是 material/state change 后运行的 asset-compile stage；它不是每 hit 的 `prepare`。相反，NMA 把 source-parameter→analytic-state 映射留在每个 shading-point view 上，编辑立即生效但每 hit 支付 MLP 成本。
- Mobile VR 的 `2×2` coarse sharing跨 texel，temporal reuse又跨 frame；它们是 `C_system` scheduling，不是保持单一 shading-point函数不变的 `C_prepare`。

NeLT、Superposed DFF与NeLiF都存在“先把global condition变成可复用state”的系统形态，但它们不是当前local ABI的`prepare()`：NeLT由object/background/light representations生成neural textures和decoder weights，Superposed DFF由`r_i`生成offset/object decoders并查询per-object fields，NeLiF由luminaire observations生成spherical field与3DGS。三篇均未分报state-generation latency、bytes与update cadence；把它们类比成`prepare`只能形成后续scene候选，不能把未计成本藏进单次query或沿用论文frame time。[NeLT §§5,8；Superposed DFF §§5,8；NeLiF §§5,8]

## 4. Dense batch、coherence与divergent path

### 4.1 PyTorch/Mitsuba buffer路径

NeuMIP、NLB与Biplane都在Mitsuba中收集query buffer，再批量交给GPU/PyTorch/CUDA。这样的结果展示了表示可批量求值，但同时包含buffer construction、transfer、launch与高coherence；不能代表DXR中每个divergent hit的scalar shader cost。Biplane约1秒也不能仅因未优化就否定representation，反过来也不能称实时。

### 4.2 NVIDIA两条shader执行路径

RTA不是调用通用inference runtime，而把小MLP展开成Slang math：同warp material coherence高时走tensor-core 16×16 blocks，divergent时用packed 16-bit weights、128-bit loads和FMA，并按coherence动态选择，SER负责局部重排。这里“网络同shape”仍可能因material coherence得到完全不同吞吐；所以当前实验框架必须分coherent packet与divergent path，不用一个dense GEMM数字代表两者。[RTA §§7–8]

### 4.3 Quantized plane decoder

Comprehensive把大量容量放在U/H/D planes，decoder做Int8量化；这种设计把cost从MLP MAC迁到随机texture reads、dequant/adapter与cache。参数KB不能代表完整material bytes，尤其dynamic synthesis、height shell和silhouette另有资源。任何复现必须同时账记plane bytes/reads、decoderweights、height traversal和export/runtime format parity。

## 5. Spatial与temporal系统摊销

Mobile VR有两个在公开执行路径中耦合、但账本必须分报的轴：

1. 一个`2×2` texel square只计算一次方向、height与coarse MLP，四个fine texel共享`z_c`；
2. object-space radiance texture更新一次后跨N帧复用，forward pass只按UV gather。

第一个减少每texture texel的neural work，但在square center共享方向，产生minor angular aliasing；第二个减少更新频率，但旧texture已经绑定相机/灯光/对象状态。论文的8.9×等数字还同时改变screen-space/texture-space resolution与reuse，不能拆成单一MLP speedup。[Mobile VR §§4,8–10]

对当前项目，local `prepare/evaluate`保持每shading point语义；object-space update/reuse是renderer scheduling层。只有在冻结motion/light validity、per-eye资源、update policy和temporal metric后，才能作为部署轨道比较。

## 6. Scene/volume：offline、observation、image inference 与 total frame 分账

| 方法 | `R_offline` reference/data acquisition | `C_fit` training/compile | `C_observe` current-frame observation | generated/current state lifecycle | `C_image/C_system` 正式 timing | persistent/runtime bytes 与证据边界 |
|---|---|---|---|---|---|---|
| [CNSR 2020](../papers/granskog-2020-compositional-neural-scene-representations.md) | PrimitiveRoom/ArchViz各含144k procedural training instances，每instance为3张observation+1张query；formal GT SPP与生成时间未报告 | single V100上Pool encoder+Pixel/U-net约8.5天、GQN约10天；不含未报告的GT acquisition | 新scene先取得三张64² path-traced HDR observations及G-buffers；每个novel view另生成query G-buffer，indirect-GI demo还需direct buffer；各 acquisition timing未报告 | 三张observations由encoder编码/平均为per-scene global `r`；encoding latency、refresh与cache lifetime未报告 | RTX 6000、unoptimized PyTorch的1k² indirect prediction为`400 ms`；同demo的8k-spp direct buffer为`7 min`、indirect reference为`25 min`，是不同生成项。`400 ms`是否包含encoder、observations、query G-buffer、direct trace、transfer/sync未闭合 | `r`为128–512 scalars；shared Pool encoder约2.00M params，Pixel/U-net/GQN generator约4.20M/80.60M/147.74M params；precision/serialized bytes未报告。它是scene image query，不是local material evaluator |
| [NeLT 2023](../papers/zheng-2023-nelt.md) | 每dataset约6000 random scenes×8 views，256²/256 spp customized OptiX GT；dataset/foreground/background/light representation acquisition time与storage未分报 | **每个独立 NeLT transfer unit**（通常一个dynamic object；mutually stationary static objects可组成composite）用2×A6000训练200k iterations、约60 h；不是60 h/dataset | foreground surface 1000 samples、background six-face `64²` cubemap observations（64 spp）、area/environment light samples与当前view G-buffer；哪些输入按scene/object/light/frame更新未报告 | global representations驱动hypernetworks生成direct/indirect neural textures与UV/decoder weights；build latency、update cadence、cache lifetime未报告 | Table 4为single RTX3090/PyTorch且所有方法排除G-buffer：256² NeLT `26.66–60.16 ms`，1024² `300.62–656.53 ms`；Indoor包含三个object passes。tiny-cuda-nn 512²约20–50 fps只在不可得video，不能与Table 4合并 | per-transfer hypernetworks、neural textures、generated decoder/composition state；parameter/texture/state bytes、precision、activation与build workspace未报告。Table 4 time随独立NeLT object数线性增长；CNSR observation acquisition也不在其baseline time内 |
| [Superposed DFF 2024](../papers/zheng-2024-superposed-deformable-feature-fields.md) | 每scene 6000–8000 random states、512²/4096 spp Falcor GT并offline denoise；GT rendering/denoising time与storage未报告 | 4×A6000约15 wall-hours/**scene**；steps、precision、distributed strategy与checkpoint selection未报告 | 当前G-buffer position/normal/view转到各object local frame；G-buffer generation与transform timing未报告 | object-pair encoders聚合`r_i`，hypernetworks据此生成offset/object decoders，再查询per-object fields；这些state按scene change/frame/pixel的更新边界与生成时间未报告 | RTX4090、PyTorch FP32、512²为`18.9/19.0/26.6 ms`；Table 1没有拆G-buffer、pair encoding、hypernetwork、全部fields、final decoder、transfer或sync，不能称完整renderer total或单field query | Ajar/Watercolor/Hall可视化中的2/2/4 fields只锁定三个scene partition；levels/resolutions/channels/dtype、field/generated-state bytes与activation workspace未报告。成本随dynamic objects增加，large scene高分辨率triplane memory显著增长 |
| [Neural Prefiltering 2023](../papers/weier-2023-neural-prefiltering-lod.md) | per-asset voxelization与online MC/ray reference存在，但各自生成时间未单列 | reference+两网训练在RTX 3080上只报告合计`7m12s–40m16s`（五主场景），不能从中拆出纯fit；另有`20–120 s` threshold search | 无独立G-buffer/CNN observation pass；runtime沿sparse voxels反复做visibility query，命中后做appearance query，调用数随path/LoD/threshold变化 | 没有scene-change hypernetwork state；per-asset sparse grids/networks/threshold是预先固定的deployment state | Arcade LoD2 isolated traversal为stochastic on/off `41.2/83.5 ms/frame`；Fig.15/16其它FPS/收敛时间属于不同resolution/SPP scope | 每资产两网+7 LoD sparse grid共`19.31–131.44 MB`；单network固定，但完整path不是固定调用次数，不能换算local material query |
| [Neural Light Probes 2022](../papers/guo-2022-neural-light-probes.md) | per-scene low-spp lightmap/probes + GT + training总计`11.5–13.0 h`；其中GT `6.7–8.2 h`，low-spp probe/lightmap另为小于1 h的作者口径，设备job graph跨four Xeon Gold 5118与RTX 3090Ti | network training `3.9 h`；optimizer为Adam，但lr/batch/seed与逐设备归属未完全披露 | 1080p raster `0.4–0.6 ms` + raycast `4 ms` + eight-probe gradient search `4.4–6 ms` | per-scene probes/lightmap预存；two-frame history是runtime state，不是generated material state；其format/update bytes未报告 | TensorRT network `14.6 ms`；full pipeline `23.6–25.2 ms`/1080p on RTX3090Ti。以上为同论文Table 3分项，不是单material query | per-scene lightmap、256 probes的三类panoramas、network/history；formats、bytes、TensorRT precision未报告，不能把network time单列为system total |
| [Volumetric inference 1469](../papers/1469-2026-volumetric-light-transport-inference.md) | 500 clouds/1000 illumination maps产生22,000张4K-spp reference images；GT generation time/device未报告 | training time/device未报告 | Vulkan每帧生成deterministic auxiliary feature maps；feature packing、ray marching/NEE参数和exact condition time未报告 | 无独立cached generated field；current auxiliary buffers与attention activations的lifetime未报告 | RTX5090、Vulkan→CUDA pipeline总计`25–28 ms`/case；Fig.8只画condition/model分解而无原始数值。resolution、precision、warmup/aggregation未报告 | screen-space auxiliary buffers + attention U-Net；persistent/transient bytes未报告；不是3D local field、material `evaluate`或`sample/pdf` |
| [Dual-Band Neural GI 2025](../papers/mo-2025-dual-band-neural-gi.md) | 8192 configs/scene、512²、1024 spp并经offline denoiser；GT generation time/storage未报告 | 4×A6000约12 wall-hours/scene（约48 GPU-hours）；stage length、seed与checkpoint selection未报告 | first-hit G-buffer + 每pixel一条mirror ray取得reflection buffers；ray origin/sign/miss encoding与这些passes是否计入Table 1未闭合 | object representations经hypernetwork生成object decoders并查询principal/secondary fields；generated-parameter refresh/update timing未报告 | object-field双查询 + full-frame fusion CNN/final decoder；RTX4090/PyTorch FP32报`22.42–26.07 ms`/512²，但G-buffer/ray/hypernetwork/sync inclusion和component breakdown未报告；`1–2 ms`括注归属不明 | per-object dense 8-level triplanes、hypernetwork/object decoders、full-frame U-Net；bytes/object cap未报告，不能把情景推算64 MiB/object写成作者配置 |
| [LightFormer 2024](../papers/ren-2024-lightformer.md) | 20,000 configs/scene、512²、2048 spp component GT；GT generation time/storage未报告 | 2×A6000约50 h training，但是否每scene、是否含GT generation均未明确 | 每frame生成G-buffer；每盏灯取得direct/indirect VPL、RSM和clues。VPL重采样、RSM刷新/cache及observation timing未报告 | per-light 840-channel embeddings经attention聚合；embedding/attention state的refresh、workspace与cache lifetime未报告 | `Ours (TRT)`为`22.90–45.82 ms`/512²；TRT GPU/precision、G-buffer/RSM/VPL/network/DLSS inclusion与breakdown未报告。PyTorch `87.89–147.91 ms`是另一backend | per-scene weights + per-light VPL/RSM/embeddings/workspace；成本随pixels×lights线性，attention只避免逐灯完整decode。产品上界仍需冻结`L_max`或bounded culling/top-k |
| [NeLiF 2025](../papers/sheng-2025-nelif.md) | full corpus为5,300 luminaires、10,000 scenes、1,000,000 training images；renderer/SPP、GT/3DGS/Trellis generation time与storage未报告 | 与LightFormer正式比较只锁定相同400K subset与35 epochs；training使用12×RTX4090D，但optimizer、batch/steps、wall/GPU-hours及full-1M-vs-400K checkpoint identity未报告 | 每frame需要current G-buffer、RSM/indirect VPL与five-level hard-shadow hierarchy；各observation pass timing未报告 | 每luminaire由多视图radiance/depth生成tokens、spherical field、intensity factor与3DGS/HDR appearance；generation/update latency、cadence、cache lifetime与multi-luminaire composition未报告 | RTX4090D、TensorRT half precision、512² Table 1为`10.56 ms`；是否含field/3DGS generation、G-buffer/RSM/shadow maps、全部decoders/3DGS raster、transfer/sync均未拆分 | per-luminaire triplane/tokens/intensity/3DGS + shared decoders + per-frame feature maps；field/3DGS bytes、params/MAC、peak memory、多灯上界与update workspace均未报告 |

scene 方法是否“实时”必须按每 frame image size、light/object count、`R_offline/C_fit`、observation/state generation、ray queries、history与reuse horizon一起解释；不能拿其 FPS/ms 与 local MLP ns/query比较。NeLT/Superposed DFF的object count直接增加passes/fields，LightFormer的light count增加observation与attention，NeLiF虽把per-frame聚合前移为generated field，却未报告generation/update/bytes和多灯上界。Dual-Band 的 triplane只给 coarsest/finest `8/1024`，六个中间 resolution 和 storage precision均未报告；以上表格时间都只保留在各自论文protocol内。

## 7. Storage账本：容量经常不在MLP

### 7.1 Shared、per-asset、prepared state分开

| 类别 | 例子 | 应计入的内容 |
|---|---|---|
| shared weights/tables | shared decoder、Belcour FGD/TIR、class-wide adapter | weights、precision/alignment、tables、code constants |
| per-asset/material | NeuMIP pyramid、RTA z8 hierarchy、Biplane/Comprehensive planes、Mobile VR textures | 所有levels/planes/adapters/height、resident与streaming bytes |
| per-shading prepared state | selected LoD/z、frames、warp、sampler lobes、analytic params | state bytes、register/local memory、reuse lifetime |
| offline reference/observation assets | measured captures、BTF/BRDF tables、GT images、probe/lightmap/RSM bake、OT maps | 是否只作临时 teacher、是否进入交付、生成时间与原始 source是否仍需保留 |
| training-only | encoder、teacher、optimizer states、temporary GT/query buffers | 不计runtime，但计 peak compiler memory、训练可恢复性与交付可复现性 |
| system buffers | RSM/G-buffer/radiance texture/probes/history | 分辨率、eyes/lights/frames、format和double buffering |
| scene/object/luminaire generated state | NeLT neural textures/generated weights、Superposed DFF triplanes/decoders、NeLiF spherical field/3DGS | state生成时机、object/light上界、persistent/ephemeral bytes、更新延迟与跨frame复用 |

“tiny network”只描述shared decoder时可能严重误导：Mobile VR的network是1.6 KB但总material 6.24 MB，且双眼 `1024²` runtime radiance texture另为8.0 MB/material；NVIDIA 4k z8 tile约256 MB；NeuMIP主要容量在14-channel spatial fields；Comprehensive主要容量在QTP/height；Neural Prefiltering甚至由hash/threshold随asset复杂度增长。

### 7.2 静态有界不等于常量asset size

一个query可固定只读若干texels和执行固定MLP，但asset texture分辨率、mip count或scene grid仍可增长。项目runtime contract要求单次读取/state/control flow静态有界；asset budget则是Pareto轴，不应被误写成所有材质恒定字节数。

## 8. One-time reference、fit/compiler 与编辑性

下表只登记个体报告已锁定的生命周期；`R_offline` 为空不表示 reference 免费，而表示来源没有分报。

| 方法 | `R_offline` reference/data preparation | `C_fit` training/compile | 新 asset/edit 的实际边界 |
|---|---|---|---|
| NBRDF | measured BRDF table 已存在；800k directional query 的生成时间未分报 | 每材质 10 s–3 min，GPU 型号未报告 | 新材质重新拟合独立 675-scalar network；不是 unseen-material compiler |
| NeuMIP | synthetic MBTF：64 samples/query、约30 min on roughly 32-core CPU；measured BTF则使用现有数据 | RTX 2080Ti：max 512²约45 min、1024²约90 min | 新 spatial asset或 source edit需重新生成/拟合相应 pyramid/MLP |
| NLB | 12,720 layered BRDF的 Guo dense query corpus生成时间未报告 | shared evaluator约40 h、layerer约10 h、sampler<1 h，均为 RTX 2080Ti；新 BRDF projection约10–45 s | layerer可前向生成 layered latent，但递归编辑误差与正式 single-call latency未闭合；SV texel仍持久化96 floats |
| Biplane | measured/synthetic BTF data generation总时间未报告；mobile capture data prep约10 min | shared decoder 18 h on RTX 2080Ti；单例 compression报告约5 min，capture optimization约3.5 min | 新 BTF仍优化 planes/adapter；capture是 inverse observation workflow，不是 native-parameter compiler |
| MetaLayer | Guo 128-spp、`25^4/4×25^4`巨大 BRDF/BTDF query corpus，生成总时长未报告 | shared reflective/transmissive models各约48–50 h on 8×RTX 2080Ti | 均匀 material edit由MetaNet前向；`<1 s`与“milliseconds”未对齐。SV state texture生成按分辨率约27–80 s |
| RTA 2024 | reference在单GPU训练中 online生成，不保存 corpus；因此没有独立 `R_offline` wall time | encoder bootstrap→materialize→latent finetune合计约4–5 h/material on RTX 4090；direct optimization可接近10 h | runtime不留encoder；source graph/native edit通常需重新优化/bake |
| Comprehensive | measured BTF acquisition与synthetic Standard Material GT extraction cost未报告 | 最多300 epochs、约18 h/material on RTX 4090 | per-material QTP/planes；新 asset需独立训练。runtime synthesis只重排已训练 spatial content，不等于任意 source edit compiler |
| Importance Baking | 32,768×128² BSDF slices + GeomLoss/SOT teacher；Threadripper PRO 3995WX，生成总时间未报告 | sampling约48 h，evaluation/PDF各约12 h on RTX 3090+i9-7960X；“each material”/family checkpoint粒度未闭合 | 参数域内 query可复用训练网络，但新 family/material身份与再次训练范围未披露 |
| Hybrid | MERL/RGL/UTIA测量数据已存在 | per-dataset shared autodecoder；MERL 100 BRDF full train约10 min on RTX 5080 | held-out/new measured material固定 shared network后优化 `p,z`；steps/time未报告，不是前向 compiler |
| Taming | reference/query generator与 optimization benchmark合并执行 | RTX 5090 fully fused、四个 layered materials：multi-instance约54.40–59.00 s，single baseline约78.05–86.50 s；仅为论文 optimization protocol | 改善的是选择/训练可靠性；不生成 runtime ensemble，也没有新的 asset/export timing |
| NMA | PFMC先形成离散 tables；table generation time/storage未计入，且正文与 supplemental sample budget冲突 | PFMC class model约2 h on RTX 3090（supplemental写<3 h） | 未见 PFMC `P`/texture无需 per-material bake，但每 hit/view运行adapter；MERL branch另有per-material embedding，不能套用“无latent” |
| Mobile VR | measured BTF/UBO source已存在；采集成本未报告 | teacher约60 epochs + scratch student最多150 epochs，合计约19 h/material on RTX 4080 | 部署只留 student/textures；新材质重训，runtime temporal reuse不减少 one-time fit |
| Belcour | shared `64^4` FGD与`64^3` TIR table precompute算法可得，但 samples/hardware/time未报告 | 无 neural fit；runtime从 native layer parameters构造 lobes | material edit可立即重算 operators；offline arbitrary-layer 与 realtime 3-layer/2-lobe是不同 runtime identity |
| Guo | 无 bake/fit；每个 reference `evaluate/sample/pdf`或 render都在线执行 stochastic path estimator | 不适用 | 它是 reference generator；成本随 path/variance target变化，不能当 fixed deployment program |

scene 方法的 `R_offline/C_fit` 已在 §6 单列；CNSR 的144k-instance dataset与8.5/10天训练不能掩盖未报告的GT生成时间/SPP，Neural Light Probes 的“low-spp probes <1 h”也不能代替含 GT+training 的 `11.5–13.0 h` total，Dual-Band/LightFormer 同样不能用 training hours掩盖未报告的 GT generation。项目自己的候选必须同时报告 `R_offline`、`C_fit`、编辑后 recompile latency、`B_asset`、`C_prepare/C_eval` 与 system deployment cost。训练慢不自动否决方法，但会影响 native parameter interactive editing 与 workflow robustness W。

## 9. 对当前 NeuralShading 的部署约束 `[N/I]`

### 9.1 每个candidate的静态注册项

至少冻结：

- `prepare`输入、输出state shape/bytes、texture reads与最大control flow；
- `evaluate/sample/pdf`各自MAC/ops、reads、precision、是否复用prepared params；
- shared/per-asset/table/system-buffer bytes；
- coherent packet与divergent scalar两条cost；
- scene/image方法另列`C_observe`与`C_image`，说明G-buffer/ray/RSM/VPL/auxiliary generation、network、upscale/history分别是否计入；
- output measure与sample/pdf identity；
- `R_offline`、compiler stages、source edit后的更新范围；
- benchmark hardware/backend、warmup/cache、batch/query count和included stages。

### 9.2 当前最有价值的部署对照

1. current NVIDIA compact direct evaluator：作为learned-frame、z8 hierarchy与shader path基线；
2. Belcour 3-layer/2-lobe：作为bounded analytic control/proposal；
3. Hybrid analytic-core residual：同MAC/state budget比较quality与core cost；
4. MetaLayer-style asset compiler：只在严格固定source family下比较zero/low-refinement compile；
5. NMA-style per-view analytic adapter：作为`prepare(wo)`路径单列，不能把每-hit MLP成本记成one-time compiler；
6. plane/factorized evaluator：以asset bytes/reads换MLP容量；
7. Mobile VR outer scheduling：仅在local evaluator成形后作为texture-space/temporal部署轨道。

它们必须在同一source/query/parameter split下比较，不用论文各自FPS预判赢家。

## 10. 可证伪的综合假设 `[I]`

| Hypothesis | Evidence basis | Minimum matched control | Frozen axes | Metrics | Falsification condition |
|---|---|---|---|---|---|
| D1：把view-only工作移入`prepare`能改善多query摊销 | NeuMIP stage split、RTA/NMA/Belcour state依赖 | fused-every-query vs cached prepare；reuse count sweep | weights/state/precision/backend/queries | prepare time、evaluate time、state bytes、N-query total、parity | 在预冻结复用数下无加速或state/带宽使总Pareto更差 |
| D2：coherent与divergent路径需要不同kernel策略 | RTA tensor-core vs packed-FMA | dense/coherent、mixed-material、fully divergent三workload | model、precision、queries、hardware | median/p90 query、throughput、register/cache/occupancy | 单一kernel在全部workload不劣，动态分支只增成本 |
| D3：plane容量只在强空间/方向correlation source上值得其reads | Biplane/Comprehensive/NeuMIP容量分布 | plane vs generic z latent；iso-byte和iso-read两组 | source/query/decoder/trainwork/precision | G1/G2、bytes、fetch/time、cache strata | plane在空间与1×1 source均无Pareto优势或收益不依赖correlation |
| D4：analytic table可压缩而不破坏operator parity | Belcour regular FGD/TIR tables | exact interpolation vs bounded approximator | domain/grid/precision/backend | max/p95 table error、white furnace、grazing/tint、bytes/time | edge error破坏energy/tint，或saved memory被latency抵消 |
| D5：Mobile VR temporal reuse是独立系统轴，不替代single-query改进 | fixed finest/noLoD与跨N帧reuse事实 | same evaluator screen/object-space，reuse N sweep；另与proper LoD分开 | resolution/motion/light/eyes/hardware | FPS、latency、temporal/disocclusion error、memory | reuse对动态序列不可控，或single-query优化在相同质量下已支配 |
| D6：asset compiler前向生成state比per-material optimization改善workflow W | MetaLayer vs NLB/RTA lifecycle | forward compiler、bounded refinement、direct fit | source family/splits/runtime state/budget | G2/G2s、`R_offline/C_fit`、edit latency、W、query cost | compiler在未见states显著较差且bounded refinement仍不能恢复，或runtime program更重被支配 |
| D7：NMA式query-time adapter能用更低edit latency换取可接受的`prepare`成本 | NMA PFMC branch无per-material fit，但每view运行5×64 MLP+two Principled；MetaLayer则提前生成state | baked state vs per-view analytic adapter；两者使用同一native source family与解析target | source/split/target/quality、hardware/precision、reuse-count、native-texture reads | edit→first-frame latency、`C_prepare`、N-query total、state bytes、quality/sample-pdf parity | adapter的每-hit成本在预冻结reuse范围内被baked state支配，或未见state质量/合法参数域失败 |

## 11. 当前完成状态

- [x] `R_offline`、`C_fit`、`B_asset`、per-hit prepare、scalar eval、dense batch、scene observation、image inference与system frame成本域已分开；
- [x] `prepare`复用、coherence、quantization和outer-system amortization已划界；
- [x] 未用跨硬件FPS或训练时长做排名；
- [x] local material与已复核scene/volume成本语义已分开；
- [x] Dual-Band已按evidence review补入secondary-ray、object-field与screen-space fusion成本；
- [x] LightFormer已按review补入per-light VPL/RSM、attention与light-count scaling；
- [x] CNSR已按当前evidence-reviewed报告拆分three-observation acquisition、per-view G-buffer/direct input与full-image inference；
- [x] NeLT、Superposed DFF与NeLiF已拆分offline/fit、current observation、generated state与image timing，并保留所有included-scope缺口；
- [x] 本轮三份来源恢复后的独立cross-document evidence review已完成。

## 12. Evidence review

- `author_update`：`/root`
- `reviewer`：`/root/nelif_full_report`
- `date`：2026-08-29
- `sources`：逐项回链本文引用的local/analytic/reference个体报告，以及当前九份scene/volume成本报告；重点重查已`evidence-reviewed`的[NeLT](../papers/zheng-2023-nelt.md)、[Superposed DFF](../papers/zheng-2024-superposed-deformable-feature-fields.md)与[NeLiF](../papers/sheng-2025-nelif.md)，并与[representation/coordinates](representation-and-coordinates.md)、[optimization/loss](optimization-and-loss.md)、[filtering/LoD](filtering-and-lod.md)、[sampling/integration](sampling-and-integration.md)及本文的cost-domain边界交叉核对。
- `review_scope`：三篇的offline reference/corpus、fit、current observation、generated/current state lifecycle、image/system timing和persistent/runtime bytes是否被正确分账；不得把未报告scope当成包含项。
- `findings_closed`：已将`R_offline`与`C_fit`拆列，防止把GT/corpus acquisition静默计入optimizer；已把NeLT约60 h限定为每个独立transfer unit，并保留Table 4排除G-buffer、Indoor三object passes及tiny-cuda-nn video不可合并的边界；已把Superposed DFF约15 h/scene与RTX4090/PyTorch FP32的18.9/19.0/26.6 ms分别锁定到其原始scope；已把NeLiF full corpus与共同400K subset/35 epochs/12×RTX4090D训练身份分开，并保留10.56 ms included stages未知；三者未报告的state bytes、build/update与workspace均未补猜。
- `remaining_gaps`：各报告自身保留的hardware/precision/include-scope缺口继续保留，尤其是NeLT representation build与state bytes、Superposed DFF field配置/内存、NeLiF field/3DGS generation、多灯上界和10.56 ms scope。
- `review_status`：`evidence-reviewed`。
