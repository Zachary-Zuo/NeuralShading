# 相关工作：它们解决什么，以及本项目具体参考什么

本文只保留能影响当前 neural evaluator、compiler、latent 或部署决策的工作。每一项都区分“论文已经证明的事实”和“NeuralShading 要验证的迁移假设”。资料最后核对于 2026-08-24，优先链接论文、作者项目页、官方代码或平台文档。

## 1. 先用四类问题组织文献

| 类别 | 核心问题 | 本项目关注点 |
|---|---|---|
| 随机访问 codec | 如何压缩高维/多通道资产，又能局部解码 | latent 布局、codebook、量化、mip、tile 和 encoder |
| neural material | 如何直接求值方向外观，并可能提供采样 | 方向编码、共享 decoder、physical prior、输出变换 |
| LOD/过滤 | 如何让 footprint 变大时仍保持正确外观和有界成本 | 空间—方向相关性、filter supervision、层级表示 |
| 工业部署 | 如何把小网络放进 shader hot path | 权重布局、coherent/divergent execution、资产流送、真实帧时 |

这些类别互相连接，但不能互相代替。纹理 codec 不会自动学会 BSDF；一个高质量 BRDF MLP 也不会自动支持 mip/filter；能够运行 cooperative vector 只证明部署通道存在，不证明方法更快。

## 2. 随机访问压缩与高维分解

### 2.1 Random-Access Neural Compression of Material Textures，SIGGRAPH 2023

[论文与项目页](https://research.nvidia.com/publication/2023-08_random-access-neural-compression-material-textures)把一个材质的多张纹理和完整 mip chain 联合压缩。表示由每个 feature level 的高/低分辨率 latent grid、局部 positional encoding、LOD 标量和两层 64-channel MLP 构成；训练中模拟低 bit latent 量化，运行时只读取目标 texel 周围的固定 feature，因此支持随机访问。

本项目具体参考：

1. latent 不是单一平面，而是按目标 mip 分组的多级高/低频 grid；
2. 高频 grid 的四邻域直接拼接，让 MLP 学习插值，低频 grid 使用双线性插值；
3. 量化噪声、clamp 和冻结离散 latent 后微调 decoder 的 QAT 流程；
4. rate-distortion 必须包含所有材质通道、mip 和真实解码时间；
5. 小网络只解码局部查询，不解码整图。

不能照搬：原论文采用 autodecoder 形态，为每个材质把 feature grids 当作自由参数并与小网络联合优化；输出是纹理通道而不是 `f(wo,wi)`，方向峰、能量、互易性和 matched sampler 不在问题内。项目提出的“给 NTC 类表示增加 target encoder”是扩展候选，不是原论文方法。当前 [RTX NTC SDK](https://github.com/NVIDIA-RTX/RTXNTC) 还明确区分 on-sample 的未过滤 texel 与配合 stochastic texture filtering 的过滤路径，所以“有 mip 输入”不等于复杂 shading 已正确过滤。

### 2.2 Neural Graphics Texture Compression Supporting Random Access，ECCV 2024

[论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05476.pdf)使用不对称、确定性的 autoencoder。输入就是待压缩 texture set `T∈R^(C×H×W)`；global transformer `E(T)` 产生 bottleneck latent，再构造量化 grid。运行时只保留 grid、局部 feature sampler、positional input 和小型全连接 decoder，按坐标与 mip 重建单个 texel。论文明确按给定 texture set 单独训练 compression model，并报告相对 NTC 和 ASTC 的 rate-distortion 改进；它首先是 per-asset target encoder，不是已经训练好的跨材质通用 encoder。

本项目具体参考的是**target encoder 不会破坏随机访问合同**这一结构：encoder 只在训练/压缩时把完整目标 tensor 变成 latent；资产烘焙后将它丢弃，runtime 仍是局部 fetch + 小 decoder。这不是 VAE，也不是从源材质参数预测 latent 的 compiler。它与用户关于“确定性 target encoder 比随机初始化自由 latent 优化更快、噪声更少”的经验同向，应成为正式 baseline，而不是附加选项；但速度和最终质量仍需在同预算下实测。

不能照搬：卷积 target encoder 假定被压缩目标已经是对齐的规则 tensor。LayerStack/MERL 方向响应可以在冻结的查询网格上 tensorize，BTF 也天然接近高维 tensor；但不规则测量、连续方向查询和缺失数据需要 mask、set/coordinate encoder 或保留 autodecoder。更重要的是，它必须先获得完整目标 tensor，不能据此宣称未见原生材质可被即时编译。

### 2.3 Neural Dynamic GI，CVPR 2026

[论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Wu_Neural_Dynamic_GI_Random-Access_Neural_Compression_for_Temporal_Lightmaps_in_CVPR_2026_paper.pdf)压缩时序 lightmap 查询 `L(u,v,t)`。其 hybrid feature 包含高分辨率 `uv` plane、`ut/vt` planes、低分辨率 `uvt` volume 和时间 positional encoding；训练中用 endpoint/weight 模拟 BC7，并在 runtime 结合 virtual texture 流送。

本项目具体参考：

- 为不同相关性分配不同 resolution 和 channel budget，而不是让每个轴对称；
- QAT 要模拟最终 BC/量化资产，而不是训练 fp32 后再祈望无损压缩；
- tile/VT residency、decoder 和 latent 需要联合设计；
- decoder 输入应只含当前查询需要的局部 feature。

不能照搬：`t` 是一维、全局语义明确且通常连续的状态轴；材质的 `wo/wi` 是两个球面方向，并带随 view 移动的尖峰。NDGI 的 tri-plane/volume 是有价值的基线，但不是默认最优分解。

### 2.4 K-Planes、TensoRF 与 Dictionary Fields

- [K-Planes](https://openaccess.thecvf.com/content/CVPR2023/html/Fridovich-Keil_K-Planes_Explicit_Radiance_Fields_in_Space_Time_and_Appearance_CVPR_2023_paper.html)为 `d` 维信号建立全部两两 plane，便于给时间和空间施加不同先验。它适合做 plane factorization baseline，特别是验证“所有两两相关足以解释数据吗”。
- [TensoRF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920332.pdf)比较 CP 和 vector-matrix tensor decomposition，说明纯 rank-1 太受限时，把两个 mode 联合进 matrix factor 可以用稍多存储换更少 component。它为 `material state × wo × wi` 的低秩 oracle 提供直接对照。
- [Dictionary Fields](https://vlg.inf.ethz.ch/publications/Dictionary-Fields-Learning-a-Neural.html)把信号分为局部 coefficient field 和共享 basis field，并跨位置、尺度、信号复用 basis。它与用户 top-2 codeword 思路最接近，提示“共享方向 basis + 局部稀疏系数”可能比对称 plane 更符合材质数据。

这些工作解决的是通用 field factorization。材质实验必须额外检查：moving specular peak 是否导致 rank/codebook 急剧增长；若先做 half-vector warp 或 analytic-core 分离后 rank 明显下降，这本身就是重要结论。

### 2.5 用户提供的 K-means++ top-2 轨迹混合

该方案不是上述 NDGI 论文内容，而是项目经验提出的 K-means++ 轨迹字典：把每个像素的 26 帧 RGB 展平为 78 维向量，聚类得到共享 codebook；每个像素固定最近原型，再从第 2–5 近原型中选出与它做凸混合后 MSE 最低的一项，只保存两个 index 和一个标量权重。查询某个时间时只读取两个原型对应的 RGB 三元组并混合。其价值是一个非常便宜的 sparse dictionary decoder；完整算法、张量形状和重建公式见[问题定义](problem_definition.md)。

本项目要复现的不是整套竞赛资产流程，而是四个消融：top-1 VQ、top-2 凸混合、top-k soft mixture、top-2 + residual。先在固定方向 probe 上做 oracle，再把 codebook 移到 latent 空间，防止“显式存完整方向表”产生虚假胜利。

## 3. Neural BRDF 与 neural material

### 3.1 Neural BRDF Representation and Importance Sampling，CGF 2021

[论文](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14335)用浅层 MLP 直接从 Rusinkiewicz 方向参数输出 RGB BRDF，并对 cosine-weighted reflectance 使用 log loss；单材质网络还能通过 autoencoder 压到稳定 embedding，再从 embedding 预测可解析的 Blinn-Phong sampling proxy。

本项目具体参考：

- half/difference 方向参数化对 specular peak 的强先验；
- log-domain 监督和方向采样分布是表示质量的一部分；
- evaluator 与 sampling proxy 可以共享材质 embedding，但 `sample/pdf` 仍使用可解析分布；
- measured BRDF 应直接按方向值评测，不必先拟合统一 PBR 参数。

它不覆盖 spatial latent、mip、shared universal runtime 或 shader 内 tensor 加速，因此是函数建模基线，不是完整系统基线。

### 3.2 NeuMIP，SIGGRAPH 2021

[项目页与论文](https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP/)将 material 看成 7D 查询：二维位置、`wi/wo` 和 filter kernel size。它使用独立的 neural texture pyramid 与 MLP，并通过 view-conditioned neural offset 改变 texture lookup，以表现 parallax、self-shadowing 和复杂微结构。

本项目具体参考：

- spatial neural material 的监督必须显式包含 footprint/scale；
- 每个 LOD 独立学习比简单平均 latent 更有容量，但也增加资产和跨级一致性问题；
- 把难学的几何移动转为显式 coordinate offset，常比扩大 MLP 有效；
- 7D query 是空间材质阶段合理的最低维度描述。

不能照搬：NeuMIP 主要逐材质优化，不直接解决跨原生材质族 compiler；其 offset 对高度型/局部视差有效，但对一般 layered BRDF 的方向峰不是同一种移动。

### 3.3 Neural Layered BRDFs，SIGGRAPH 2022

[项目页](https://wangningbei.github.io/2022/NLBRDF.html)把 BRDF 压成 latent code，并学习两个 latent 的 layering operation。它证明“方向函数的 latent 可以参与组合运算”，而不是只能存最终外观。

本项目具体参考：层栈实验中增加“直接编译完整 stack”与“先编码原子层再组合”的消融，并检查组合次序、深度外推和误差累积。不能把它重新提升为项目唯一目标：当前源材质不都具有 layer 语义，最终 runtime 也不要求暴露 latent algebra。

### 3.4 MetaLayer，SIGGRAPH Asia 2023

[论文](https://sites.cs.ucsb.edu/~lingqi/publications/paper_siga23metalayer.pdf)包含表示 layered BSDF 的 BSDFNet，以及从物理层参数生成 BSDFNet 权重的 MetaNet；使用 Rusinkiewicz spherical-harmonic encoding 改善高频，目标是 feed-forward 生成未见 layered material 的专属网络。

本项目具体参考：

- 它是 LayerStack compiler 的强基线，直接回答“原生参数能否一次前向生成可求值表示”；
- hypernetwork 生成权重应与“shared decoder + generated latent”在 iso-byte、shader divergence 和编辑延迟下比较；
- 方向编码不能只比较 Fourier feature，还应纳入 half/difference 与球谐结构。

边界：每材质生成网络权重可能破坏大量材质并存时的 shared-code/coherence 优势；它也不适用于无层参数的 MERL、BTF 或任意图作为统一输入。

### 3.5 Real-Time Neural Appearance Models，TOG / SIGGRAPH 2024

[论文与项目页](https://research.nvidia.com/labs/rtr/neural_appearance_models/)是当前最接近本项目 runtime 形态的强基线。它从复杂 layered material 烘焙 hierarchical latent texture，用 neural decoder 直接输出 BRDF；decoder 从 latent 中提取 learned shading frames，再把方向变换后求值。独立 sampler decoder 预测可解析 two-lobe microfacet proposal 的参数。系统把小网络 inline 到 raster/ray-tracing shader，支持 anisotropy 和 LOD。

其训练流程尤其重要：先把每个表面位置的高维 source material parameters 输入 encoder，与 decoder 端到端训练；之后全图求值 encoder 得到 latent texture，再丢弃 encoder并可选直接 refinement。训练 query 不是先导出成固定 HDF5 再反复读取：论文在训练中从 UV、half/difference directions 等分布取样，在 GPU 上在线求 reference BRDF；报告规模为 300k iterations、每次两个 65k batch，共接近 400 亿个在线样本，单个材质在 RTX 4090 上约 4–5 小时。论文明确指出 encoder 改善 latent 结构和插值，并减少高分辨率直接优化残留的初始化噪声。[开源实现](https://github.com/NVlabs/neuralappearance)还支持多个 MaterialX/MDL 材质联合训练成各自 latent texture + 一个共享 neural model。这里的输入是 source parameters，不是 Qualcomm texture codec 中的完整目标 tensor；二者都在 runtime 前丢弃 encoder，但能支持的 compiler 结论不同。复现时，网络形态、online reference 生成、joint evaluator/sampler lifecycle 和训练规模必须分别登记；只对齐网络宽度不能单独称为论文复现。

本项目应直接参考/复现的组件：

1. source-parameter encoder bootstrap、latent bake 和 refinement 三阶段；
2. learned frame 是网络外的乘法图形先验；
3. log-L1、方向 mollification 和 half/difference query sampling；
4. evaluator 与 analytic-proxy sampler 共享 latent、但 sampler loss 不反向破坏 evaluator latent；
5. coherent 与 divergent material execution 的分开 benchmark；
6. source MaterialX 直接求值，而不是预生成固定表后失去编辑 provenance。

它已经覆盖“复杂 layered appearance → neural evaluator + sampler + LOD + shader deployment”，所以 NeuralShading 不能把这些单项重新包装为 novelty。当前新增问题应集中在：跨多种原生 source family 的统一 compiler、未见编辑状态、明确的 `prepare/evaluate` amortization、deferred 与 PT 共用资产，以及更严格的质量—时间—内存 Pareto。

### 3.6 2026 年的两个直接竞争方向

#### Neural Material Adapter

[Disney Neural Material Adapter](https://studios.disneyresearch.com/2026/07/01/neural-material-adapter-transforming-complex-materials-into-efficient-analytic-bsdfs/)让 analytic BRDF 参数随入射方向变化，并用轻量 adapter 从层材质配置生成这些参数。它用 differentiable analytic prior 从稀疏、带噪 reference 学习，并保持现有 renderer 兼容。

本项目具体参考：它是“编译为方向条件解析 BRDF”的强基线，尤其适合比较稀疏监督稳定性与 CPU/普通 shader 成本。边界是它主动把运行时输出限制到 analytic family；NeuralShading 的目标 evaluator 仍直接补全散射函数。

#### A Hybrid Neural-Microfacet BRDF Model

[Ubisoft / Inria 的 EGSR 2026 工作](https://ubisoft-laforge.github.io/world/hybridrdf/)用 GGX-type microfacet core 加共享 neural correction；每材质保存解析参数与低维 latent。在相同内存下，论文报告它比纯 neural measured-BRDF model 更准确，并保留编辑和重要性采样。

本项目具体参考：analytic core + neural residual 应是首轮候选，而不是失败后的补丁。要比较直接 neural output、加性 residual、乘性/对数域 correction，并用 `asinh` 等有符号变换处理 residual。边界是单一 GGX core 未必覆盖 LayerStack 的多峰和空间微结构，不能默认它必然胜出。

### 3.7 Toward Richer Material Generation via Procedural Data Enhancement，SIGGRAPH 2026

[论文项目页](https://research.nvidia.com/labs/rtr/publication/yu2026toward/)从普通 PBR material 程序化构造 dust、clearcoat、layered scattering 等多 lobe source，再用两个 RGB latent texture 和 pretrained universal MLP 表示 6D 非漫反射外观。

本项目具体参考：

- 普通 PBR 语料的外观支持太窄，训练集必须主动覆盖复杂非漫反射长尾；
- universal decoder 的 latent 要为跨材质生成/编译施加结构约束；
- 从基础材质程序化增强可用于构造独立的新 source family。

重要边界：增强后的 layered model 是一个**新的源材质定义**，不能被称为原始 PBR 资产的 GT。该方法适合丰富训练分布，不适合验证真实材料保真度。

## 4. LOD、过滤与 Ling-Qi Yan 相关工作

### 4.1 MIPNet，SIGGRAPH Asia 2022

[项目页](https://perso.telecom-paristech.fr/boubek/papers/MIPNet/)学习从 normal 与 roughness 等固定 SVBRDF channels 生成下一级参数，使过滤后的单次 BRDF lookup 更接近 footprint 内平均 radiance，并把 normal variation 转移成 anisotropic roughness。

本项目具体参考：它是传统 PBR channel 的强 LOD baseline，也说明 filter loss 应在 shaded response 上定义，而不是只匹配参数均值。它只输出固定材质参数，因此不能作为复杂 neural appearance 的容量对照。

### 4.2 Constant-Cost Spatio-Angular Prefiltering of Glinty Appearance，TOG 2022

[论文](https://doi.org/10.1145/3507915)把不同空间 footprint 下的 NDF 图预计算成 tensor，以 CP decomposition 压缩。rank-1 factor 使方向 point query 只需取各向量元素相乘，加入一维 summed-area table 后还能做常数成本 angular range query。

本项目具体参考：

- 把 LOD 问题明确写成 spatial-angular range query；
- 压缩表示必须支持 partial query，不应为一个方向解压整张 NDF；
- tensor rank 是可解释的容量/成本旋钮；
- point evaluation、area/environment angular range 与 sampling 可以共享同一压缩结构。

边界：论文只建模 microfacet NDF，固定 Smith 等其他项，不代表一般 layered/BTF 的完整散射。

### 4.3 Neural Prefiltering for Correlation-Aware Levels of Detail，SIGGRAPH 2023

[论文](https://doi.org/10.1145/3592443)用 sparse multi-level voxel grid 上的两个独立小网络，分别学习 voxel 内平均 appearance throughput 与点到点 visibility；目标是在远场 LOD 中保留会被独立参数平均破坏的几何—可见性相关性。

本项目具体参考：

- “相关性”应先指明是哪两个随机变量之间的相关性，不能只说 latent 能学习复杂外观；
- appearance 与 visibility 在查询语义不同的情况下应分开建模；
- 所有 LOD 联合训练，并用局部 Monte Carlo reference 监督；
- LOD 的正确性要用 temporal sequence 和 light leak 检查，而不只看单帧 PSNR。

它面向几何 asset prefiltering，不是局部 surface material codec。NeuralShading 当前仍把 scene visibility 留给 renderer；只有以后扩展 microgeometry/nonlocal capability 时才直接复用其 visibility 分解。

### 4.4 Towards Comprehensive Neural Materials，SIGGRAPH 2025

[论文信息](https://research.manchester.ac.uk/en/publications/towards-comprehensive-neural-materials-dynamic-structure-preservi/)同时处理 BTF quality、结构保持 synthesis、parallax/silhouette 和 runtime。其实现用 int8 权重/激活与 `dp4a`，feature planes 仍用浮点纹理以支持插值，并采用两级 displacement tracing。

本项目具体参考的是部署与压力测试：int8 QAT、feature/weight 不同精度、BTF 的 6D spatial-direction query、以及 displacement 不应被 surface BRDF 误吸收。其 synthesis 和 silhouette 是独立 capability，不进入当前 LayerStack evaluator 的第一轮目标。

### 4.5 Real-Time Neural Materials on Mobile VR，CGF 2026

[论文摘要](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.70318)以 extremely low-capacity coarse-to-fine neural material、distillation、texture-space shading 和时空计算摊销，在 Quest 3 上报告多灯 90 FPS 以上。

本项目具体参考：当 evaluator 已确定后，可以把 `prepare` 或部分 shading 迁到 texture space，并用时空缓存摊销；teacher-student distillation应纳入 mobile/low-end profile。它优化的是整个运行调度，不能用来替代当前函数容量与监督审计。

## 5. 工业实现与平台约束

### 5.1 RTX Neural Texture Compression SDK

截至核对日期，[RTX NTC SDK](https://github.com/NVIDIA-RTX/RTXNTC)提供三种有不同权衡的路径：

- on-sample：压缩资产常驻，shader 内逐 texel neural decode，VRAM 最省但 shading 成本最高；
- on-load：加载时解码并转成 BCn，运行时兼容性最好但失去 VRAM 压缩；
- on-feedback：通过 sampler feedback 只把可见 tile 转成 sparse BCn texture，在带宽、缓存和推理之间折中。

本项目应把这三种模式映射为 neural material asset 的部署 profile，而不是默认只有“每次 shading 都跑 MLP”。对于复杂 evaluator，on-feedback 还可用于预烘焙固定 LOD 或平台 fallback，但必须明确它是否仍支持参数即时编辑。

### 5.2 RTX Neural Shading SDK

[RTXNS](https://github.com/NVIDIA-RTX/Rtxns)提供 Slang、SlangPy、DirectX/Vulkan cooperative vector 和从训练到 shader inference 的示例。它最适合复用权重 packing、精度和 shader integration，不应成为研究架构本身；项目仍需在锁定的 Falcor/Slang 版本上验证实际支持范围。

### 5.3 DirectX shader 内线性代数

Microsoft 2025 年的 [Cooperative Vector 说明](https://devblogs.microsoft.com/directx/cooperative-vector/)展示 per-thread matrix-vector inference 进入 HLSL。随后官方说明原 experimental Cooperative Vector 设计将被统一的方案取代；2026 年公布的 [DirectX Linear Algebra](https://devblogs.microsoft.com/directx/evolving-directx-for-the-ml-era-on-windows/)把 vector-matrix 与 matrix-matrix 工作负载纳入同一方向。

当前结论是：shader 内神经计算有明确的平台路线，但 API 仍在演进。论文实验应以实际 Slang/D3D12/Vulkan kernel 和普通 ALU baseline 报告，不能仅按理论 Tensor Core FLOP 估算。

### 5.4 Unreal Substrate

[Substrate 概览](https://dev.epicgames.com/documentation/en-us/unreal-engine/overview-of-substrate-materials-in-unreal-engine)与 [GBuffer 格式说明](https://dev.epicgames.com/documentation/unreal-engine/programming-with-substrate-gbuffer-formats)提供工业对照：材质复杂度最终受 closure 数、GBuffer bytes、简化策略和灯光路径约束。

本项目具体比较对象是同一 workload 下的 bytes/pixel、prepare/evaluate GPU 时间、有效 light query 数和质量；Substrate 不是 reference GT，也不要求 neural program 输出 Substrate closure。

## 6. 当前优先级

### 现在必须精读或复现

1. Real-Time Neural Appearance Models：最强直接系统基线，尤其是 source-parameter encoder、learned frame、log loss、mollification、sampler 和 coherent/divergent execution。
2. Neural BRDF Representation：方向参数化、log-domain response 和 measured BRDF 基线。
3. Hybrid Neural-Microfacet BRDF：analytic core + neural residual 的直接竞争方案。
4. NTC autodecoder、ECCV target-tensor encoder、NDGI 与用户 top-2 字典：latent 获取、量化和字典候选。
5. MetaLayer：LayerStack feed-forward compiler 与 per-material weight generation 基线。

### evaluator 成形后再进入

1. NeuMIP、MIPNet、spatio-angular tensor filtering：spatial footprint/LOD；
2. Real-Time Neural Appearance 的 matched sampler、Neural Material Adapter 的 analytic compatibility：sampling 与部署；
3. Mobile VR texture-space amortization、RTX NTC feedback mode：系统调度；
4. Comprehensive Neural Materials 与 UBO BTF：高维空间外观、位移和 silhouette 压力测试。

这一路线让相关工作直接对应当前实验，不再以文献数量替代研究问题。
