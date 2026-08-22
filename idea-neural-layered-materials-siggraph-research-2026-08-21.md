# 面向现代实时渲染的多层材质神经压缩、采样与着色研究调研

> 调研日期：2026-08-21  
> 目标平台：**Falcor（D3D12/Vulkan）研究系统为主，Unreal Engine 为可裁剪的部署证据** [2026-08-21 修订，原为"Unreal Engine"]  
> 目标效果：在非 path tracing 的运行成本下，逼近离线多层材质的局部光学外观  
> 文档性质：研究方向分析、数据方案与可执行验证路线  
> 配套文档：`idea-neural-layered-materials-analysis-2026-08-21.md`（可行性深评）、`idea-neural-layered-materials-P0-任务清单-2026-08-21.md`（P0 任务拆解）

> **2026-08-22 实测修订：** 表示上界实验已经完成第一轮，实验前“LTC-K3 大概率够用”的预判不再有效。当前最好的结构是“精确顶层界面 + LTC 残差”，K2 的方向域 relative-L1 中位数/第 90 百分位为 6.73%/31.20%，K3 为 5.56%/25.24%。K2 继续作为实现基线，K3 说明增加容量有收益，但两者都没有解决导体基底和深层栈的长尾。项目当前先改进表示，再训练结构化编译网络。详见 `reports/oracle_ceiling_v0.md`。

### 修订记录（2026-08-21 会话定稿）

以下结论已逐项写回对应章节，正文中以 **[2026-08-21 修订]** 标出：

| # | 结论 | 写回位置 |
|---|---|---|
| R1 | raster 主着色中 ω_o 按像素已知 → view-conditioned closures 天然合法；次级光线需 view-averaged fallback closure | §2.2、§4.1 |
| R2 | 头条主张改为"未见（层类型×顺序×参数）组合零样本编译 + 改一层不重训"；5–8 层外推降为次要实验 | §1、§3.3、§10 Claim 1、§16 |
| R3 | 层组合以"学习方向基底上的散射算子 + Redheffer star product"为主线（结合律/能量守恒由构造保证），近镜面走 Belcour 式解析通道；学习式 compose 网络降为消融 | §4.4 |
| R4 | OpenPBR 不是 ground truth；主 teacher = pbrt-v4 `LayeredBxDF` 随机游走的 Slang 移植（GPU），PFMC 交叉验证，OpenPBR 仅作 Stage A sanity 与输入词汇 | §6.4、§9.1 |
| R5 | LOD 简化：footprint → 层统计量喂同一 encoder，pyramid 由聚合层重新 compose + mip consistency loss；dust/flake v1 只以统计 NDF 进入 | §4.5、§4.7 |
| R6 | closure 基 = Lambert / 各向异性 GGX / clearcoat GGX / Charlie sheen；与 Substrate 做 iso-byte 对比、画两条 Pareto | §4.6、§9.2 |
| R7 | 解码位置定为 pre-light compute pass；互易性只测量报告 | §4.8、§8.3 |
| R8 | 研究闭环全部在 Falcor 单仓库内完成；Substrate 成本基线在 stock UE 测量；UE 定制为可裁剪项 | §8、§9.2、§14 |
| R9 | PT-vs-raster 协议：同 GBuffer、同灯同 IBL、同 RayQuery 可见性；参考 = PathTracer(maxSurfaceBounces=0) 材质内随机游走；完整 PT 只作语境列 | §9.1 |
| R10 | 算力分工：4090 = 数据工厂 + 开发 + 实时 benchmark；A6000×5 = 纯训练农场；teacher 数据按版本预生成、A/B 半样本 + 计数存 bin；缓存 tile 足以算任意 IBL 下的 image loss；在线 teacher 只作泛化验证 | §6.5、§7.4、新增 §7.5 |
| R11 | 周期：不含 UE 8–10 个月，含 UE 10–13 个月；主目标 SIGGRAPH Asia 2027，EGSR 2027 兜底，完整版 SIGGRAPH 2028；难度 4/5 | §14 |
| R12 | 当时核实的背景：NMA = EGSR 2026；项目选择 Falcor 8.0；SM 6.9 于 2026-02 正式发布，Cooperative Vector 正被 DXLA 取代；NLBRDF/NA 代码公开。实际依赖以 `AGENTS.md` 的固定提交为准 | §3.1、§8.4 |
| R13 | **closure 表达力**：原文档未分析"K 个解析瓣够不够"。定稿：表达力问题不是"解析 vs 神经"二选一，而是"在存在廉价光积分算子的函数族里选哪一档"；closure 族从固定 GGX 参数族（①）升级为 **LTC 风格瓣（②）**；P1 之前先做**表示天花板（oracle）实验**用数据回答；全神经 g（④）只作上界消融列；学习字典（③）为 oracle 不达标时的升级路径 | §3.2、§4.6、§9.9、§10 Claim 2、§12.2、§14 P1 |

## 1. 执行摘要

这一方向仍然有冲击 SIGGRAPH/TOG 的空间，但研究问题不能再表述为“使用神经网络拟合多层 BRDF”或“压缩复杂材质图”。截至 2026 年，以下问题已经被较充分地覆盖：

- 使用网络表示或压缩 BRDF/BTF/SVBRDF；
- 在 latent 空间组合 layered BRDF；
- 将 MaterialX/MDL 复杂材质烘焙为实时神经材质；
- 随机访问式神经纹理压缩；
- 将复杂分层材质转换为轻量解析 BRDF；
- 对普通 PBR 数据进行 dust、clearcoat、fuzz 等多瓣外观增强。

最有希望的主方向不是再构建一个逐方向查询的 neural BRDF，而是：

> **构建面向 rasterization/deferred shading 的“可组合神经材质闭包代数”：将任意层数、顺序和空间参数的多层材质编译为固定数量、可过滤、可由 UE 光照循环直接消费的解析 closures。**

本文暂称该方向为：

> **Neural Closure Algebra: Composable and Filtered Layer Transport for Rasterized Real-Time Rendering**

其核心区别是：

1. 网络学习的是材质内部的局部光传输编译与组合，而不是最终像素颜色。
2. 运行时不进行 path tracing，也不为每个入射方向重复运行大型网络。
3. 网络每个可见像素只解码一次，输出固定的 diffuse、anisotropic GGX、clearcoat、fuzz、colored secondary lobe 等解析 closures。
4. 动态点光、聚光、区域光和 IBL 仍由传统实时光照管线完成。
5. 将像素 footprint、LOD 与高频微结构聚合纳入表示，避免远距离 glint、法线和粗糙度闪烁。
6. 层栈可以增删、换序和编辑，而不需要为每个材质重新训练一个网络。

**[2026-08-21 修订]** 头条主张不再是"训练 2–4 层、测试 5–8 层"（审稿人第一反应是"谁用 8 层"），而是第 6 条：**未见的（层类型 × 顺序 × 参数）组合零样本编译，改一层不重训**。层数外推保留为次要泛化实验。

若同时完成“可组合表示、可过滤闭包、真实 UE 系统、测量数据验证”四项贡献，该方向具有较强的 SIGGRAPH 潜力；如果只完成其中一项，更可能落在 EGSR、I3D、HPG 或工程型论文的范围。

**[2026-08-21 修订]** 四项全做按一人主力是 12–18 个月的量。定稿主线 = **结构化散射算子 + 固定 closure packet + Falcor 级 raster 系统 + 公开实测集验证**；UE 定制与自采 paired 实拍降为可裁剪项。该范围 8–10 个月可出 SIGGRAPH Asia / EGSR 质量稿，补 UE 证据后冲 SIGGRAPH。

---

## 2. 问题定义与范围边界

### 2.1 希望解决的问题

给定一个空间变化的多层材质栈：

- 每层具有独立的 base color、roughness、normal、height、IOR、absorption、thickness 等参数；
- 层之间可能包含反射、折射、吸收、粗糙界面、微粒或微纤维散射；
- 层数、层顺序和层参数允许变化；
- 材质需要在动态视角、动态光源、动态 IBL 和不同像素 footprint 下稳定显示；
- 运行时目标是 UE 的 raster/deferred 或 forward+ 光照，而不是 path tracer。

研究目标是将其转换为固定成本的实时材质表示，使材质内部复杂多次散射的外观接近离线参考，同时保持传统实时渲染的可部署性。

### 2.2 “接近 path tracing 效果”的准确含义

本方向可以逼近的是：

> **材质内部的多次反射、粗糙界面透射、吸收、微结构遮挡与尺度聚合所产生的 path-traced local appearance。**

典型效果包括：

- 粗糙清漆覆盖拉丝金属；
- 珠光或金属汽车漆；
- 上釉陶瓷；
- 灰尘或糖霜覆盖高光基底；
- 桃皮、织物 fuzz、细纤维逆反射；
- 湿石头、湿木材和透明涂层；
- 随距离正确聚合的 flakes、glints 和高频法线；
- 多层吸收引起的掠射角色偏移和高光分裂。

它不能单独替代场景级 path tracing，因此不应宣称能够独立生成：

- 物体之间的多次间接光；
- 场景级镜面互反射；
- 屏幕外可见性；
- 复杂焦散；
- 完整的全局照明；
- 透明物体内部的任意几何路径传输。

在 UE 中，合理组合应是：

- 本方法负责复杂局部材质；
- Virtual Shadow Maps 负责直接光可见性；
- SSR、reflection captures 或其他反射方案负责场景反射；
- Lumen 可选地负责场景级 GI；
- 材质本身不发射 path-tracing rays，也不依赖 spp 和 path-tracing denoiser。

**[2026-08-21 修订] 次级光线的边界。** 本方法的 closures 以主可见性的 ω_o 为条件（见 §4.1），因此 SSR / Lumen / 反射探针打到这类表面时需要另一个 ω_o。处理方式：closure packet 附带一个 view-averaged fallback closure 供次级光线使用，或在反射路径退化为 Substrate 参数混合。该限制必须写进论文 limitation，不得回避。

---

## 3. 相关工作与研究空缺

### 3.1 关键工作对比

| 工作 | 主要贡献 | 与本方向之间仍存在的空缺 |
|---|---|---|
| [Belcour 2018: Efficient Rendering of Layered Materials](https://belcour.github.io/blog/research/publication/2018/05/05/brdf-realtime-layered.html) | 使用能量、均值、方差等方向统计量与 adding-doubling，支持任意纹理化层栈和实时实现 | 少量方向矩难以稳定表示多峰、强各向异性、粗糙多层、glint、fuzz 和复杂掠射行为 |
| [Neural Layered BRDFs 2022](https://wangningbei.github.io/2022/NLBRDF.html) | 在 neural latent 空间表示 BRDF，并学习层组合操作，提出了 neural BRDF algebra 的雏形 | 最终仍是逐方向 BRDF 解码，主要服务 Monte Carlo/path tracing；缺少 raster closure、IBL 集成和 UE 成本验证 |
| [MIPNet 2022](https://perso.telecom-paristech.fr/boubek/papers/MIPNet/) | 将 normal 的尺度变化转化为 anisotropic roughness，生成可直接用于实时引擎的 mip | 输出仍受标准 SVBRDF 通道和固定解析模型的表达能力限制 |
| [Random-Access Neural Compression of Material Textures 2023](https://research.nvidia.com/publication/2023-08_random-access-neural-compression-material-textures) | 联合压缩整组材质纹理及 mip 链，实现 GPU 随机访问和实时解码 | 主要优化纹理重建，不直接编码经过多层传输后的光学响应 |
| [MetaLayer 2023](https://doi.org/10.1145/3618365) | 使用 meta-learning 将层参数映射为 neural BSDF 表示，保留材质编辑能力 | 空间变化材质涉及每 texel 预处理和较大中间表示；仍未面向固定成本 raster lighting |
| [Real-Time Neural Appearance Models 2024](https://research.nvidia.com/labs/rtr/neural_appearance_models/) | 将深层 MaterialX/MDL 图烘焙成 latent textures 和小型神经解码器，支持 LOD、各向异性和重要性采样，并在实时 path tracer 中运行 | 每个材质需要优化/烘焙；面向 ray/path tracing；动态层编辑、任意新拓扑和 raster/deferred 集成不是核心目标 |
| [Filtering After Shading with Stochastic Texture Filtering 2024](https://research.nvidia.com/publication/2024-05_filtering-after-shading-stochastic-texture-filtering) | 用随机纹理采样把过滤放到 shading 之后，适合神经或稀疏纹理 | 会引入噪声并依赖时空重建；本身不解决复杂多层材质表示 |
| [Improved Stochastic Texture Filtering Through Sample Reuse 2025](https://research.nvidia.com/labs/rtr/publication/wronski2025quadcomm/) | 复用相邻像素样本，改善放大时的 STF 误差和稳定性 | 仍然是过滤与样本复用问题，不提供多层传输闭包 |
| [Real-Time Neural Appearance / Neural Shading 工具链](https://github.com/NVlabs/neuralappearance) | 公开 MaterialX/MDL 到 latent texture、BRDF decoder 和 importance sampler 的研究管线；**参考材质直接由 Falcor 材质系统在 GPU 上评估，无需预生成数据集 [2026-08-21 核实]** | 是非常重要的实现与对照基础，但仍以神经 BSDF 查询和 Falcor/path-tracing 工作流为主。**与本项目同框架（Falcor），NA 基线可同仓库复现** |
| [Neural Material Adapter 2026](https://studios.disneyresearch.com/2026/07/01/neural-material-adapter-transforming-complex-materials-into-efficient-analytic-bsdfs/) **（EGSR 2026，Disney/ETH）[2026-08-21 核实]** | 使用轻量 MLP 将复杂材质参数映射到方向变化的 Principled BRDF，具有解析 sampling 和 energy control；CPU 推理、对未见层配置零样本 | 论文实验限制在固定三层、各向同性、仅反射的 PFMC 类；没有深层可组合拓扑、footprint-aware filtering 和 raster 系统验证。**最近竞品停在 symposium 级，说明"输出解析 BRDF 参数"本身已不够 SIGGRAPH，交叉点仍空** |
| [A Hybrid Neural-Microfacet BRDF Model 2026](https://ubisoft-laforge.github.io/world/hybridrdf/) | 使用 GGX 主体和小型神经 residual 拟合实测 BRDF，兼顾编辑与速度 | 重点是均匀测量 BRDF，不是空间变化、多层组合和 LOD |
| [Toward Richer Material Generation via Procedural Data Enhancement 2026](https://blaire9989.github.io/assets/4_DataEnhance/project.html) | 将普通 PBR 材质增强为 haze、dust、clearcoat、fuzz、scatter 等共享 latent 神经材质 | 目标是材质数据增强和生成，不是 UE raster 部署与任意层栈组合 |
| [Real-Time Level-of-Detail Rendering with ReSTIR 2026](https://research.nvidia.com/labs/rtr/publication/wang2026levelofdetail/) | 在几何 LOD 切换中复用路径/样本，提高实时 path-traced LOD 稳定性 | 说明随机残差和 ReSTIR 路线竞争激烈；本方向应以 deterministic raster closure 为核心 |

### 3.2 已经饱和或风险较高的题目

以下单独作为论文主题时，novelty 风险较高：

- 用 MLP 拟合多层 BRDF；
- 把 BRDF 压成 latent code；
- 为 neural BRDF 学一个 importance sampler；
- 用网络输出 GGX、Disney 或 Principled 参数；**[2026-08-21 修订]** 注意本方向的输出形态与此相近——区别不在"输出解析参数"这件事本身，而在（a）可组合的层算子、（b）ω_o + footprint 条件化、（c）raster 系统、以及（d）**closure 函数族经 oracle 天花板实验选定而非假设**（§4.6）。前三条是与 NMA 的划界，第四条是本文档此前缺失的表达力论证；
- 只优化 PBR texture compression；
- 只做 normal/roughness 的 neural mipmapping；
- 只展示单个材质球或 WebGL viewer；
- 只证明网络比复杂 shader 快；
- 只用 path-traced renderer 证明“实时”。

### 3.3 尚未被完整解决的交叉空缺

真正仍有价值的是以下交叉问题：

1. **Variable topology**：同一模型能否处理训练时未见的层数、顺序和组合？**[2026-08-21 修订] 重心是"未见的层类型 × 顺序 × 参数组合"，层数外推只是其中一个维度，不单独成为头条。**
2. **Raster-compatible output**：输出是否能被传统 clustered/deferred light loop 和 IBL 直接积分？
3. **Multiscale transport**：能否正确处理像素 footprint 内参数、法线、遮挡与多层传输的非线性平均？
4. **No per-material retraining**：新材质和编辑后的材质能否零样本编译，而非重新优化 latent？
5. **Fixed runtime cost**：复杂层数能否在运行时归约为固定 (K) 个 closures？
6. **Temporal stability**：动态视角、动态光照、缩放和运动时能否避免高光闪烁？
7. **Engine evidence**：能否在 UE 的真实 GBuffer、光照循环和 GPU 成本模型下成立？

---

## 4. 推荐主方向：神经闭包代数

### 4.1 核心研究假设

对于限定词汇内的多层材质，不需要在运行时保存完整高维 BRDF，也不需要为每个入射方向执行复杂网络。可以学习一个固定维度、顺序敏感但近似可结合的层传输 latent，并在给定视线和像素 footprint 后，将其解码为少量标准解析 closures。

换句话说，研究对象不是单点 BRDF 值：

\[
f_r(\omega_i,\omega_o)
\]

而是固定视线和 footprint 下，完整入射方向响应的紧凑可积表示：

\[
g(\omega_i)=f_r(x,\omega_i,\omega_o;\rho)\,\max(0,n\cdot\omega_i)
\]

其中：

- (x) 是材质空间位置；
- (\omega_o) 是视线方向；
- (\rho) 是像素在纹理/材质空间中的 footprint；
- (g) 是应与动态入射光积分的函数。

网络输出 (K) 个解析基函数：

\[
g(\omega_i)\approx\sum_{k=1}^{K}a_k\,b_k(\omega_i;\mu_k,\Sigma_k,\tau_k)
\]

其中 (b_k) 可以是 diffuse、anisotropic GGX、retroreflection、clearcoat、fuzz、LTC-compatible lobe 或其他固定 closure。

于是动态光照为：

\[
L_o\approx\sum_{k=1}^{K}\int_{\Omega}L_i(\omega_i)b_k(\omega_i)\,d\omega_i
\]

解析灯可以逐灯评估，IBL 可以使用预过滤环境贴图，区域光可以使用 LTC 或近似积分。网络成本与光源数量解耦。

**[2026-08-21 修订] 为什么 view-conditioned 在 raster 里是合法的。** deferred 主着色中 ω_o 在光照之前就已按像素确定（primary visibility），因此"以 ω_o 为条件输出 closures"不是妥协而是天然匹配：每像素只解码一次，对任意数量的动态光复用。这是相对 NLBRDF（逐方向解码）与 NA（逐材质烘焙、服务 PT）的根本优势，应写进论文 §1。代价见 §2.2 的次级光线边界。

### 4.2 系统数据流

```text
MaterialX / OpenPBR / 自定义层栈
              │
              ▼
       每层参数与空间纹理
              │
       Layer Encoder E
              │
              ▼
   z1, z2, ..., zn 传输 latents
              │
       顺序敏感组合算子 C
              │
              ▼
        z_stack latent pyramid
              │
      view + UV derivatives/footprint
              │
       Closure Decoder D
              │
              ▼
 固定 K 个 UE-compatible analytic closures
              │
      ┌───────┼─────────┐
      ▼       ▼         ▼
 clustered  IBL      area lights
  lights   prefilter     LTC
      └───────┼─────────┘
              ▼
        Rasterized shading
```

### 4.3 原子层编码

每个层具有统一但带 type token 的参数结构，例如：

- layer type；
- interface IOR 或 conductor eta/k；
- roughness x/y；
- tangent frame；
- thickness；
- absorption/scattering coefficient；
- phase 或 microflake 参数；
- spatial masks；
- normal/height statistics；
- 可选的 spectral/thin-film 参数。

层编码器输出传输 latent：

\[
z_i=E(p_i,T_i)
\]

需要避免把 latent 设计成无物理意义的任意向量。更稳妥的设计是分块保存：

- reflected energy summary；
- transmitted energy summary；
- directional moments 或 lobe parameters；
- anisotropic frame；
- high-frequency residual code；
- validity/uncertainty。

这使组合过程能够显式执行能量控制，并为消融实验提供可解释性。

### 4.4 层组合算子

多层组合是顺序敏感、非交换的：

\[
C(z_A,z_B)\neq C(z_B,z_A)
\]

但真实层传输在数学上应对括号方式保持一致，因此希望近似满足结合性：

\[
C(C(z_A,z_B),z_C)\approx C(z_A,C(z_B,z_C))
\]

训练中可以对同一层栈随机采样不同括号树，并加入 associativity consistency loss。这样能够：

- 支持任意层数递归组合；
- 使用平衡树从 (O(N)) 深度降到 (O(\log N))；
- 缓存子树结果；
- 编辑单层后只重算受影响的树节点；
- 测试训练层数之外的 extrapolation。

可考虑两种组合设计：

1. **Learned adding-doubling**：以 Belcour 的能量/方向统计和经典 adding-doubling 为物理先验，网络只预测解析近似的 correction。
2. **Learned scattering operator**：将每层表示为低秩 reflection/transmission operator，再使用类似 scattering-matrix 或 Redheffer star product 的结构进行组合。

~~第一种更容易实现并进入 UE；第二种理论贡献更强，但更容易出现维度、正定性和尖峰表达问题。~~

**[2026-08-21 修订] 定稿：以第二种为主线，第一种降为消融。** 理由：这是与 NMA/NLBRDF 拉开距离的**结构性**差异，而不是"我们也能跑 raster"。具体形态：

- 每层编码为学习到的方向基底（8–16 个基函数/半球）上的小矩阵 $\begin{bmatrix} R & T' \\ T & R' \end{bmatrix}$；
- 层组合 = Redheffer star product（含 $(I - R'_A R_B)^{-1}$ 项）：**结合律由构造保证、能量守恒靠算子范数 ≤ 1 结构保证**，不再依赖 associativity loss 与 energy penalty；
- 尖峰问题用混合解决：近镜面能量走 Belcour 式解析通道（方差/粗糙度闭式传播），低秩算子只承载平滑残差；
- 网络只做两件事：encoder（层参数 → 算子矩阵）、decoder（根算子 + ω_o + footprint → K closures）；
- 平衡树 O(log N)、子树缓存、编辑只重算到根路径全部由结构免费获得；
- 消融自然变为"结构化 star product vs 学习式 compose 网络"，以及"有/无解析近镜面通道"。

原 associativity consistency loss 保留为**评估指标**（对学习式 compose 消融项测量），不再是主线训练目标。

### 4.5 多尺度与像素覆盖范围感知表示

传统 mipmap 对 base color、normal、roughness 分别平均，不能保证 shading 后结果正确。多层材质中问题更严重，因为：

- normal 平均与 BRDF 平均不交换；
- roughness、IOR 和 Fresnel 非线性；
- coat 下的 base normal 分布会被再次折射和过滤；
- flakes、fibers 和 dust 的稀疏事件会随 footprint 改变分布；
- 多层之间的遮挡、吸收和高光分裂不能通过通道平均恢复。

建议为 (z_{stack}) 构建 transport-aware latent pyramid，而不是普通参数 mip：

\[
z^{l+1}=F(z^l_{00},z^l_{01},z^l_{10},z^l_{11},\rho_l)
\]

其中 (F) 通过 shading/rendering loss 学习聚合。运行时根据 UV derivatives 选择或插值 latent level，decoder 同时接收精确 footprint 形状或其简化描述。

为了控制研究范围，第一版可使用：

- isotropic footprint radius；或
- 主轴长度、次轴长度和方向组成的 anisotropic ellipse。

**[2026-08-21 修订] 不单独学习 F。** 完整的 transport-aware pyramid 本身是一篇论文的量（MIPNet、NA LOD）。定稿做法：footprint → 层级统计量（法线协方差 → 各向异性粗糙度 LEAN/MIPNet 式；coverage；高度方差）喂给**同一个** encoder，pyramid 由"对 footprint 聚合后的层重新 compose"得到，再加 mip consistency loss 微调。保留"transport-aware filtering"主张，机制便宜一个数量级；若 Phase 2 末仍有余量再考虑学习式 F 作为消融上界。

### 4.6 固定闭包参数包

建议第一版限制为 (K=2) 或 (K=3)，每个 closure 包含：

- type；
- RGB amplitude/F0；
- roughness x/y；
- lobe orientation 或 shading frame；
- optional tint/absorption；
- confidence 或 residual energy。

一个实用组合是：

1. stable diffuse/retroreflective closure；
2. primary anisotropic specular closure；
3. secondary coat/fuzz/colored closure。

当真实响应超过 (K) 个解析瓣能表达的范围时，可以：

- 由 decoder 动态选择 closure type；
- 将低能量瓣合并到 rough residual；
- 使用小型预积分 LUT；
- 在极少数高频区域启用 stochastic residual。

不建议第一版允许任意数量 closure，否则无法证明固定运行成本，也会与 UE Adaptive GBuffer 的复杂度问题重新重合。

**[2026-08-21 修订] closure 函数族：表达力是被测量的设计轴，不是假设。**

原文档默认 closure = 固定 GGX 参数族，但没有回答"K 个解析瓣表达得了目标材质族吗"。这个问题决定整个方法的天花板：若表示不够，后续 star product、pyramid、系统都是在错的表示上堆工程。

*(1) 先换问法：deferred 对表示的真实要求是"可积分性"。*

| 光源 | 需要的操作 | 解析瓣 | 全神经 g（latent + 小网络逐 ω_i 评估） |
|---|---|---|---|
| 点/聚/方向光 × N | 逐光评估 g(ω_i) | K 次 ALU | N 次 MLP——成本重新与灯数耦合 |
| IBL | ∫ L_i g dω | split-sum 预过滤 | MC（噪声回来）或第二张"神经积分器"网（与点光着色不一致风险） |
| 面光 | 多边形积分 | LTC 闭式 | 无闭式 |

因此"解析也由 neural 实现"在 deferred 语境里自毁前提，只配做上界消融列。真正的设计变量是：**在存在廉价光积分算子的函数族里，选表达力最强的一档。**

*(2) closure 函数族谱系。*

| 档 | 形态 | 表达力 | 光积分 | 定位 |
|---|---|---|---|---|
| ① 固定参数族 | K × {Lambert, GGX, clearcoat, sheen}，参数随 ω_o 变 | 单峰、对称、固定衰减形状 | 点光 ALU / split-sum / LTC 查表 | NMA 所在档；本文档原方案；降为消融下界 |
| **② 半参数瓣（候选）** | K × LTC 瓣（3×3 变换 + RGB 幅值 + frame） | 任意单峰形状：可拟合 GGX 粗糙度/各向异性，并能表达偏斜与非对称衰减；标准 transformed-cosine LTC 不是 GGX 的严格函数超集 | 面光闭式（LTC 原生）；IBL 用等效粗糙度查预过滤图；点光 ALU | bytes 相近（4–5 参数/瓣），是否优于①由 oracle 决定 |
| ③ 固定基展开 | g = Σ c_k B_k，B_k 为 SH / SG / ASG / **学习的固定瓣字典** | 任意形状（多峰、非对称、瓣内颜色随 ω_i 变） | 线性：点光逐基 ALU；IBL = 每基一张 load-time 预过滤图；SG 面光近似闭式 | oracle 不达标时的升级路径；学习字典版可成为表示级贡献 |
| ④ 全神经 g | latent + 小网络 | 最强 | 全部破（见上表） | 仅作上界消融列 |
| ⑤ 混合 | ①/② + ③ 或随机残差 | — | — | §4.7 |

*(3) 对 v1 材质族的表达力预判（ω_o 固定后 g 只是半球上的 2D 切片，比 4D BRDF 温顺得多）。*

| 现象 | ① | ② | ③ | 说明 |
|---|---|---|---|---|
| coat + base 两个不同粗糙度的同心峰 | ✅ | ✅ | ✅ | 2 个瓣 |
| 各向异性拉丝 | ✅ | ✅ | ✅ | |
| 多层导致的离镜面峰偏移 | ✅ | ✅ | ✅ | 每瓣独立 frame 随 ω_o 倾斜（NMA 同法） |
| 吸收导致的掠射色偏 | ✅ | ✅ | ✅ | 逐瓣 RGB 幅值随 ω_o 变 |
| Fresnel 随 ω_i 在瓣内变化 | ≈✅ | ≈✅ | ✅ | 窄瓣内半矢量变化小；宽瓣差异被漫反射项吸收 |
| 粗糙 coat 下 base 二次折射后的**偏斜/非对称瓣** | ❌ | ✅ | ✅ | GGX 形状固定，LTC 仿射变换可表达 |
| **瓣内颜色随 ω_i 变**（thin-film、强吸收路径长度依赖） | ❌ | ❌ | ✅ | thin-film 已在 v2；强吸收情形需 oracle 量化 |
| 离散 glint / flake | ❌ | ❌ | ❌ | 设计排除，只以统计 NDF 进入 |
| 大 footprint 聚合响应 | ✅ | ✅ | ✅ | 聚合后更平滑，对解析瓣更友好 |

实验前曾预判 LTC-K3 在 v1 范围内大概率够用。2026-08-22 的实测结果表明，这个判断过于乐观：通用 LTC-K3 的方向域 relative-L1 中位数为 14.72%；先精确计算顶层界面、再用三个 LTC 瓣拟合残差，可降到 5.56%，但第 90 百分位仍为 25.24%。因此 LTC 保留为残差基元，当前组合方式不作为最终表示。

*(4) 定稿与执行。*

- **当前工程基线 = 精确顶层界面 + 两个 LTC 残差瓣**：其 176-byte 布局已经完成 CPU、PyTorch 与 Falcor 一致性验证，适合继续搭建系统，但不等于最终闭包词汇。第三个残差瓣增加 48 bytes，能降低误差，却仍未消除长尾，暂不设为统一默认值。
- **表示上界实验已经完成第一轮**（§14 P1.0、P0 清单 C6）：结果表明纯 GGX-K3、通用 LTC-K3、SG-K8 和两个共享 SG 字典都弱于“精确顶层界面 + LTC 残差”的物理分解。当前任务是针对导体基底和深层多次传输改进残差表示；改进后仍用同一套方向域与真实 HDRI 指标复测，再决定最终闭包词汇。
- **④全神经 g 保留为上界消融列**：量化"去掉解析约束还能好多少"，是对"为什么不全神经"最有力的回答。
- 对"这不就是 NMA + 组合"的回答因此有三层：表示上界实验（族选择有据）+ 结构化组合算子（§4.4）+ raster 系统（§8/§9）。
- 3 个 closure 在 fp16 下约 50–60 B/px，落在 Substrate 参数混合（~28 B）与全 slab（~108 B）之间。论文必须画两条 Pareto：**同 bytes 比质量、同质量比 bytes**。
- IBL 用 split-sum（LTC 走等效粗糙度），实验把 **closure 表示误差**与 **split-sum 积分误差**分开报告：参考 BSDF MC 积分 vs closure MC 积分 vs closure split-sum。

### 4.7 可选的随机残差

对于极窄 glint、稀疏 flakes 和高频 fiber，完全确定性的 (K=3) closure 可能过度平滑。可以将响应分解为：

\[
g=g_{analytic}+g_{residual}
\]

- (g_{analytic})：由固定 closures 稳定渲染；
- (g_{residual})：只在少量像素、少量帧或高显著度区域采样。

残差采样可以使用：

- blue-noise spatial pattern；
- material-space sample reuse；
- motion-vector temporal reuse；
- 基于 residual energy、lobe width 和屏幕覆盖率的自适应预算；
- 小型 bilateral/neural temporal reconstruction。

但该模块应作为扩展或消融点，不宜成为主论文唯一创新，因为 stochastic texture filtering、实时 glint 和 ReSTIR-LoD 已经形成密集竞争。

**[2026-08-21 修订]** v1 中 dust / flake / glint **只以统计 NDF 形式进入**（SpongeCake 式 microflake 分布、footprint 聚合后的 flake 密度），不做离散 glint；stochastic residual 整体推到 v2。

### 4.8 物理与稳定性约束

建议至少包含以下约束和报告指标：

- non-negative closure weights；
- directional hemispherical reflectance 不超过能量上界；
- reflection-only 配置下的 reciprocity error；
- 层组合前后的 energy consistency；
- 不同括号顺序的 associativity error；
- 相邻 texel、mip 和 view direction 的 continuity；
- 对 glossy/high-dynamic-range target 使用 log-domain 或相对误差；
- 对 Monte Carlo teacher 使用 Average-vs-Average 或 bin-average supervision，避免网络拟合样本噪声。

需要注意：类似 NMA 的 view-dependent 参数化可能违反严格 reciprocity。若采用该路线，必须：

1. 显式测量而不是回避该误差；
2. ~~提供 reciprocal variant；~~ **[2026-08-21 修订] 不做 reciprocal variant**——view-conditioned 在 raster 主着色里必然不严格互易，测量、报告、在 limitation 里说明即可，不过度投入；
3. 说明 raster forward rendering 中该权衡的适用范围。

**[2026-08-21 修订]** 由于组合改为结构化 star product（§4.4），上表中的 energy consistency 与 associativity error 对主线方法应接近机器精度，主要用于核对实现与消融项；non-negativity、continuity、log-domain loss、Average-vs-Average 监督保持不变。

### 4.9 编辑与缓存

目标不应是每帧重新组合所有层。建议分为：

- **cook-time**：静态层栈离线生成 latent pyramid；
- **load-time**：上传或按需解压 latent tiles；
- **edit-time**：GPU compute 异步重算修改区域和受影响的 mip；
- **frame-time**：只进行 latent fetch、一次 closure decode 和普通光照。

组合树允许修改单层时只更新从该叶节点到根节点的路径。对于局部纹理编辑，可采用 tile dirty mask，只更新受影响的材质空间 tile。

---

## 5. 推荐的最小可行研究范围

### 5.1 第一版必须支持

- 2–6 个垂直层；
- opaque、reflection-only；
- 各向同性与各向异性界面；
- spatially varying 参数；
- base diffuse/conductor；
- rough dielectric coat；
- absorption medium；
- dust/fuzz 或 microflake 中至少一种；
- 动态方向光、点光和 HDR IBL；
- 像素 footprint 与 mip；
- 固定 2–3 closures；
- ~~UE desktop DX12 部署~~ → **[2026-08-21 修订] Falcor desktop D3D12 deferred 部署 + stock UE Substrate 成本列**；dust/fuzz/microflake 仅以统计 NDF 进入（§4.7）。

### 5.2 建议第二阶段再加入

- area light/LTC；
- stochastic glint residual；
- thin-film iridescence；
- 可编辑 layer cache；
- NTC 联合压缩；
- 动态材质参数动画；
- cross-vendor cooperative vector / DXLA backend；
- **[2026-08-21 修订]** 自采 paired layered materials（§6.7）；UE source-built 插件 / Substrate 扩展（§8）；学习式 transport pyramid F 作为 §4.5 的消融上界。

### 5.3 第一篇论文不建议同时承担

- 任意 transmission；
- 厚物体 SSS；
- wave-optics 全模型；
- fluorescence/polarization；
- 场景级 neural GI；
- 移动端；
- 无限制 MaterialX 图语义；
- 完整商业资产导入器。

“限定原子层词汇内的任意层栈”已经足够形成清晰贡献，不需要承诺支持所有材质节点。

---

## 6. 数据获取与构造

### 6.1 空间参数图来源（P2）：MatSynth

MatSynth 不是多层传输 ground truth，也不进入 P0/v0 的关键路径。它在 P2 用于提供 base color、normal、height、roughness 等空间变化参数，测试 footprint、mip 与材质类别泛化；核心监督始终由 §6.3–§6.4 的随机游走 teacher 生成。

首选 [MatSynth](https://research.adobe.com/publication/matsynth-a-modern-pbr-materials-dataset/)：

- 4,069 个 4K、tileable PBR 材质；
- 3,736 个原始材质加 332 个语义兼容 blend；
- 约 3,980 train 和 89 test 材质；
- Base Color、Diffuse、Normal、Height、Roughness、Metallic、Specular，以及部分 Opacity；
- 包含来源、许可证、类别、标签、生成方式、部分物理尺寸；
- CC0/CC-BY 等宽松许可；
- Hugging Face 完整下载规模约 431 GB；
- 还提供大量不同环境光和尺度的渲染，但本研究不必下载全部预渲染数据。

P2 开始时按 manifest 选择 500–1,000 个基础材质，不必使用全部 4,069 个。

#### 防止数据泄漏

MatSynth 同一原始材质可以产生大量 crop、scale 和 blend，因此不能随机按图块划分。正确顺序应为：

1. 按原始 material ID 划分；
2. 尽量按来源网站分组；
3. 再按类别平衡；
4. 最后生成 crop、scale、layer stack 和渲染样本。

建议设置：

- source-held-out split；
- material-held-out split；
- category-balanced split；
- topology-held-out split；
- illumination-held-out split。

### 6.2 补充基础纹理和 HDRI

#### Poly Haven

[Poly Haven](https://polyhaven.com/license) 的纹理、模型和 HDRI 为 CC0。其 [公开 API](https://polyhaven.com/sr/our-api) 可列出和下载资产。

注意：

- CC0 资产本身无需署名；
- 如果产品直接依赖在线 API，需要显示 Poly Haven 来源；
- 应设置唯一 User-Agent；
- 大规模训练最好保存资产快照和元数据，而不是在每次实验中动态抓取。

#### ambientCG

ambientCG 提供大量 CC0 PBR 材质，可作为 MatSynth 之外的独立来源测试集。由于 MatSynth 已包含部分 ambientCG 数据，必须根据来源链接去重，不能直接当作完全独立测试集。

### 6.3 多层 ground truth：程序化构造

公开 PBR 数据只有表面参数，通常没有真实的层结构标签。核心训练集是程序采样的层栈及其物理 teacher 方向响应；PBR 数据只在空间变化阶段提供参数图。

建议从 MatSynth base material 出发，生成以下层族：

| 层族 | 主要参数 | 典型外观 |
|---|---|---|
| Dielectric coat | IOR、roughness、thickness、tint、absorption | 清漆、上釉、湿表面 |
| Dust/haze | coverage、roughness、albedo、height-aware mask | 灰尘、粉末、糖霜 |
| Fuzz/fiber | density、orientation、color、phase | 桃皮、织物、绒面 |
| Microflake | density、NDF、orientation、color variance | 汽车漆、珠光、闪粉 |
| Rough medium | sigma_a、sigma_s、phase、thickness | 蜡质、浑浊涂层 |
| Thin film | IOR、thickness、thickness variation | 虹彩、油膜、氧化层 |
| Wet layer | coat + absorption + base darkening | 湿石头、湿木材 |

参数分布不能简单均匀采样。应结合：

- 物理有效范围；
- perceptually uniform sampling；
- log-space roughness/thickness；
- 对极窄高光和掠射行为的 importance oversampling；
- 真实材质类别的条件先验；
- 对无效或近似不可区分配置进行拒绝采样。

### 6.4 参考数据渲染器与参考实现

#### OpenPBR

[Academy Software Foundation OpenPBR](https://github.com/AcademySoftwareFoundation/OpenPBR) 是 Apache 2.0 的开放标准，适合作为输入材质词汇和作者参数空间。

[Adobe OpenPBR BSDF reference implementation](https://github.com/adobe/openpbr-bsdf) 提供：

- OpenPBR 1.1；
- coat、fuzz、thin film、transmission 等 lobes；
- eval/sample/pdf；
- multiple-scattering compensation LUT；
- C++、GLSL、CUDA、MSL、Slang 等目标；
- Apache 2.0 许可证。

~~它非常适合成为第一版 teacher 和跨语言验证参考。~~

**[2026-08-21 修订] OpenPBR 不能当 ground truth。** OpenPBR BSDF 参考实现是**解析分层近似**（albedo-scaling 式的 coat 组合），不是层间多次散射的物理模拟。若以它为 teacher，学生学到的就是 OpenPBR 的近似，论文"逼近 path-traced layered appearance"的主张不成立。OpenPBR 的定位改为：(a) 输入材质词汇与作者参数空间；(b) Stage A 单层 sanity 参考；(c) 解析基线之一（§9.2 的"直接拟合为固定 OpenPBR 参数"）。

#### PBRT-v4 —— 主 teacher [2026-08-21 修订]

[PBRT-v4 layered material](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Scattering_from_Layered_Materials)（`LayeredBxDF`，Apache-2.0）使用 Monte Carlo random walk 模拟任意上/下界面 + 中间介质，`f/Sample_f/PDF` 齐全。定稿做法：

- 把 `LayeredBxDF` 推广为 **N 层界面 + N−1 层介质**的随机游走，移植为 **Slang 模块**，在 GPU 上跑；
- 同一 Slang 模块被两处 `#include`：Falcor Python `ComputePass` 数据生成 kernel 与 Falcor 的 `IMaterialInstance` 实现（§8.3），训练端与渲染端 teacher 完全同源；
- Falcor 已内置 `PBRTCoatedConductorMaterial` / `PBRTCoatedDiffuseMaterial`（pbrt-v4 两层移植），作为 N 层推广的起点与两层情形的对照；
- 独立性交叉验证：pbrt-v4 CPU 原版（两层）、PFMC（任意层）。

任务拆解见 `idea-neural-layered-materials-P0-任务清单-2026-08-21.md` WS-A。

#### Position-Free Monte Carlo

[Position-Free Monte Carlo for Arbitrary Layered BSDFs](https://projects.shuangz.com/layered-sa18/) 提供任意层参考，并公开 Mitsuba 分支实现。

注意其代码仓库采用 GPL-3.0：

- 可以作为内部数据生成和学术比较工具；
- 不应未经审查直接复制到宽松许可证的 UE 插件；
- 如果计划公开代码，最好根据论文独立实现核心算法或明确隔离 GPL 组件；
- 数据和模型权利仍需结合实际生成流程与机构政策确认。

### 6.5 建议训练规模

核心数据从无纹理的局部层栈状态开始，再扩展到空间参数图：

- v0 不依赖基础 PBR 材质，直接采样 2–3 层局部层栈状态；
- v1/P2 再引入 500–1,000 个基础材质及其空间参数图；
- 每个基础材质 20–50 个层栈；
- 约 10,000–50,000 个 stack；
- 每个训练 step 在线采样位置、视线、光向和 footprint；
- 数千万级 BRDF/closure tuples；
- 30–50 个训练 HDRI；
- 10–20 个完全未见 HDRI 测试；
- 少量完整场景序列用于 image-space fine-tuning。

不建议预存每个 stack 的完整 6D/8D 张量。~~更合理的是：~~

~~- 对廉价 OpenPBR teacher 在线查询；~~
~~- 对高成本 PFMC teacher 缓存准蒙特卡洛 angular tiles；~~
- 对高动态范围 glossy 区域使用 bin averages；
- 只为最终 benchmark 生成高 spp 完整图像和视频。

**[2026-08-21 修订] 定稿：teacher 数据按版本在 4090 上预生成、A6000 只用缓存训练（详见 §7.5）。** 两档规模与体积（每 bin 14 B = RGB A/B 两组半样本均值 fp16 + 计数）：

| 版本 | 规模 | 随机游走次数 | 体积 | 4090 生成 |
|---|---|---|---|---|
| v0-oracle | 512 family × 1 local state × 4 ω_o × 128 bin × adaptive A/B | 自适应 512–65,024 spp/half | ~3.67 MB | 4090 实测，完成后回写 |
| v0-train 候选 | 5k family × 32 local state × 16 ω_o × 128 bin × A/B 各 64 walks | ~4.2×10^10 | ~4.6 GB | closure 定稿后重测新 teacher |
| v1 完整 | 50k × 64 texel × 8 ω_o × 256 bin × 64 walks | ~4×10^11 | ~92 GB | ~6–12 h |
| v1 瘦身 | 同上，half-vector 参数化 128 bin | ~2×10^11 | ~46 GB | ~3–6 h |

每版记录 teacher 代码 hash 与先验版本。2026-08-22 的 `1.223e8 walks/s` 与 8192-tile/3.568 秒结果属于加入多层 ballistic NEE 前的固定 64-spp kernel 标定，只证明批量调度可行；高精度 oracle 采用独立 adaptive 预算，不再用该数字外推。

### 6.6 实测材质验证集

#### OpenSVBRDF

[OpenSVBRDF](https://svbrdf.github.io/publications/OpenSVBRDF/project.html) 包含：

- 1,000 个近似平面材质；
- 9 个类别；
- 空间变化、各向异性反射；
- 高分辨率 GGX 参数与 local frames；
- 每个样本约 15 分钟采集；
- 2 台相机和 16,384 个 LED 的专业系统。

其代码仓库采用 GPL-3.0，但网页未清晰给出所有数据的统一再分发条款。建议将其用于内部评测，并在发布数据子集或模型前联系作者确认。

#### MERL BRDF

[MERL BRDF Database](https://merl.com/research/downloads/BRDF) 提供 100 个密集测量的各向同性 BRDF，适合：

- 测试多瓣和 color-shifting appearance；
- 与 NMA、Hybrid Neural BRDF 等工作比较；
- 测量 reciprocity/energy 和方向误差。

MERL 数据属于研究/非商业使用范围，不应作为商业训练资产直接重新发布。

#### RGL Material Database

[RGL Material Database](https://rgl.epfl.ch/pages/lab/material-database) 包含：

- isotropic/anisotropic BRDF；
- RGB 与 360–1000 nm 光谱数据；
- metal、paper、car paint、organic、fabric 等；
- 紧凑表示及 eval/sample/pdf API。

RGL 也接受外部样本测量。若有条件合作，可以制作受控多层样片并交由其测量，这会显著提高论文真实数据部分的可信度。

#### Bonn UBOFAB19

[Bonn SVBRDF / UBOFAB19](https://cg.cs.uni-bonn.de/btf/bonn_svbrdf_database.html) 提供大量实测织物和相关材质，包含 brocade、lycra、satin、velvet、glittering rubber 等，适合验证：

- fiber/fuzz；
- 强各向异性；
- glitter/glint；
- 织物尺度变化。

使用前应逐项核对数据许可证和发布限制。

### 6.7 自采 paired layered materials

公开测量数据通常不知道真实层参数，或者没有“同一基底在增加某一层之前/之后”的受控对照。建议制作 20–30 组 paired samples：

- 裸金属与不同厚度清漆；
- 裸木材与清漆/哑光涂层；
- 石材的干燥/湿润状态；
- 相同基底的不同粉尘密度；
- 同一底漆上的不同 flake 密度；
- 织物在压平、刷毛后的外观；
- 同一材质的两层与三层顺序交换。

低成本采集装置可包括：

- 电控转台；
- RAW/HDR 相机；
- 1–3 个可移动高 CRI LED；
- 线偏振片与交叉偏振；
- 灰卡和 radiometric calibration；
- 角度编码器或可重复机械刻度；
- 20–60 个离散光视组合；
- 连续视角和连续光照 sweep 视频。

自采数据主要用于最终验证，而不是承担大规模训练。对 SIGGRAPH 论文而言，20–30 组严格受控真实层栈通常比数千个来源不明的网络材质更有说服力。

### 6.8 许可证风险表

| 来源 | 建议用途 | 风险/注意事项 |
|---|---|---|
| MatSynth | 训练、公开 benchmark | 保留逐材质来源和 CC-BY attribution；先按原始材质划分 |
| Poly Haven | 训练纹理与 HDRI | 资产 CC0；在线 API 有署名和 User-Agent 条件 |
| ambientCG | 训练和 source-held-out 测试 | 与 MatSynth 数据去重 |
| Adobe Substance 3D Assets | **不要用于 ML 训练/测试** | 当前条款明确禁止使用资产直接或间接训练、测试或改进 ML/AI，包括学术研究；参见 [产品条款](https://www.adobe.com/go/substance3dassets) |
| OpenPBR/OpenPBR BSDF | 输入参数词汇、单层 sanity、解析基线 | Apache 2.0，适合公开工具链；不作 ground truth |
| PFMC code | 内部 teacher、参考比较 | GPL-3.0，注意公开插件和代码的许可证隔离 |
| MERL | 研究验证 | 非商业；不直接打包进商业资产或重新授权模型 |
| OpenSVBRDF | 研究验证 | 代码 GPL；数据再分发条款需单独确认 |
| RGL/UBOFAB19 | 研究验证 | 使用和再分发前核对各自条款 |

---

## 7. 训练方案

### 7.1 分阶段训练

建议采用 curriculum：

#### 阶段 A：单层重建

- 训练 layer encoder 和 closure decoder；
- 确认标准 diffuse、GGX、conductor、coat 能被正确重建；
- 建立 energy、reciprocity 和 directional loss 基线。

#### 阶段 B：双层组合

- 训练组合算子；
- 与 OpenPBR/PBRT coated materials 比较；
- 验证 clearcoat、absorption、base normal 的组合。

#### 阶段 C：2–4 层随机栈

- 训练 variable topology；
- 加入随机括号和 associativity loss；
- 引入 anisotropy、dust/fuzz/microflake。

#### 阶段 D：多尺度潜变量金字塔

- 加入 texture patches 与 footprint；
- 以 prefiltered shading/reference image 为监督；**[2026-08-21 修订] 监督信号直接由缓存的半球响应 tile 计算（§7.5），不需要渲染器在回路里**；
- 测试从近景 glint 到远景 rough aggregate 的连续变化。

#### 阶段 E：训练范围外的层数泛化

- 训练只见 2–4 层；
- 测试 5–8 层；
- 测试未见顺序和未见 layer parameter range；
- 决定是否需要 depth token 或 composition-tree normalization。

#### 阶段 F：渲染器感知微调 **[2026-08-21 修订，原“UE 感知”]**

- 使用与 **Falcor 部署路径**完全一致的 closure、IBL split-sum LUT、LTC 和色彩空间（UE 版本为可选项）；
- 直接优化渲染端输出与离线参考的 rendering loss；主路径仍用缓存 tile × HDRI 计算，Falcor 真在回路里只作为可选的最后一步；
- 加入量化、FP16 和 fixed packet packing。

### 7.2 损失函数

总损失可由以下部分组成：

\[
\mathcal{L}=\lambda_d\mathcal{L}_{dir}
+\lambda_i\mathcal{L}_{img}
+\lambda_e\mathcal{L}_{energy}
+\lambda_r\mathcal{L}_{reciprocity}
+\lambda_a\mathcal{L}_{assoc}
+\lambda_t\mathcal{L}_{temporal}
+\lambda_q\mathcal{L}_{quant}
\]

建议包括：

- log-domain RGB directional loss；
- relative SMAPE 或 tone-mapped HDR loss；
- FLIP-based image loss；
- 多环境光 image-space loss；
- energy integral penalty；
- reciprocity paired-direction loss；
- randomized-bracketing associativity loss；
- mip consistency loss；
- 动态视角 temporal warping loss；
- quantization-aware loss。

### 7.3 方向采样

均匀方向采样会浪费大量预算。建议混合：

- cosine hemisphere；
- teacher lobe importance sampling；
- grazing-angle oversampling；
- half/difference-angle stratification；
- anisotropic tangent-frame stratification；
- uniform safety samples；
- 对极窄 lobe 使用局部 angular bins。

对 Monte Carlo teacher，不能让网络拟合单次样本。可参考 NMA 的 Average-vs-Average 思路：在相同 angular bin 内比较两个独立样本集合的均值或统计量。

### 7.4 数据生成与训练存储

推荐存储：

- layer topology 和物理参数；
- base material ID 和 crop transform；
- random seed；
- sampled position/view/light/footprint；
- teacher mean、variance 和有效 sample count；
- 数据来源与许可证；
- renderer/commit/config version。

不建议只保存最终 RGB PNG，因为无法：

- 重采样方向；
- 分析 teacher variance；
- 复现实验；
- 检查能量和 reciprocity；
- 重建不同 UE 光照设置。

**[2026-08-21 修订] bin 存储粒度定稿：** 每 (stack, texel, ω_o) 一个 tile，tile 内每个 ω_i bin 存 **RGB A/B 两组独立半样本均值（fp16×6）+ 有效样本计数（u16）= 14 B**。A/B 分组同时提供 NMA 式 Average-vs-Average 监督与方差估计；ω_i 用固定的 half/difference-angle（或 concentric-map）参数化，便于后续重新分 bin 与重新加权。不存 loss、不存最终图。文件为 memmap fp16 shard（npy/zarr），随 shard 附 JSON 元数据（层拓扑、物理参数、base material ID、crop、seed、teacher hash、先验版本、许可证）。

### 7.5 算力分工与数据管线 **[2026-08-21 新增]**

**4090 = 数据工厂 + 开发机 + 实时 benchmark；A6000 × 5 = 纯训练农场。**

| 角色 | 机器 | 理由 |
|---|---|---|
| teacher 数据生成 | 4090（Windows，D3D12/CUDA） | 随机游走是 fp32 + 分支发散负载，4090 fp32 82.6 TFLOPS ≈ 2 张 A6000（38.7） |
| Falcor 开发与实时 benchmark | 4090 | D3D12 只能在 Windows；与 Codex@WinDocker 三层环境一致 |
| 训练（单 run / 5 路消融） | A6000 × 5（Linux） | 小 MLP 为带宽/launch 受限；48 GB 可让整个 shard 常驻显存，dataloader 瓶颈消失 |
| 泛化验证（在线 teacher） | A6000（Falcor Python，Vulkan） | 对从未缓存过的新层栈/新方向现算 |

关键机制：

1. **缓存 tile 足以计算任意光照下的 image-space loss。** 像素 = ∫ L_i(ω) g(ω) dω，把 HDRI 投到同一套 ω_i bin，image loss = bins × HDRI 矩阵乘。Stage D/F 都不需要 teacher 或渲染器在训练回路里。
2. **datagen kernel 写成可移植 Slang，由 Falcor Python `ComputePass` 调度**，不硬绑 4090；4090 忙于 benchmark 时 A6000 走 Vulkan 也能生成。
3. **保留在线 teacher 小通道，仅用于验证**：最终泛化实验必须在从未缓存的层栈/方向上现算，否则评估的是对固定 bin 集的记忆。
4. 数据集按版本生成（§6.5），1 Gbps 内网搬运 92 GB 约 15 分钟。

时间账（训练仍为数量级；teacher kernel 吞吐已实测）：

| 配置 | 单卡 4090 | 单卡 A6000 | 5×A6000 |
|---|---|---|---|
| v0-oracle | 512-family adaptive 高 spp，4090 实测完成后回写 | 不生成 | 不生成 |
| v0-train 一次迭代 | closure 定稿后重新标定 + 训练 ~1 h + 评估 ~0.5 h | ~3.5–4 h | 等 D3D12 closure 词汇定稿后再校准 Vulkan 生成 |
| v1 完整一次迭代 | 生成 6–12 h（缓存后省去）+ 训练 12–30 h → **~1–2 天** | 训练 15–40 h | DDP 单 run ~5–12 h（小模型有效加速约 3–3.5×）；或 **5 个消融并行 15–40 h** |

收益：tile 缓存让所有训练与消融复用同一版 teacher 结果；数据集版本切换的实际时间在多 tile writer 完成后以 wall-clock 实测为准。Stage F 若要 Falcor 真在回路仍需 4090。

---

## 8. Unreal Engine 部署设计

**[2026-08-21 修订] 本章定位变更：UE 从"最终论文系统"降为"可裁剪的部署证据"。**

- **研究闭环全部在 Falcor 单仓库内完成**（teacher、datagen、训练、viewer、基线、benchmark，见 §9.1 的项目结构）。Falcor 8.0 提供 Slang/GFX、D3D12+Vulkan、Python 绑定与 PyTorch 互操作、可微 Slang、内置 `PBRTCoated*` 材质、`FLIPPass`/`ErrorMeasurePass`；NA 基线本身建于 Falcor。近年 NA/NTC/STF 等 SIGGRAPH 论文均只用 Falcor，审稿人接受。
- **Substrate 成本基线在 stock UE 直接测量**：同场景导出、同 GPU、同分辨率，记录 bytes/px 与各 pass GPU 时间，零源码改动。我们的方法在 Falcor 中跑同一组资产，论文明说两者不在同一管线。
- source-built UE + Substrate 扩展是全计划最贵的单项（熟手 2–4 个月，且 Substrate 内部每版在变），仅在 P3 完成且有余量时启动；最低可行形态为 NVIDIA RTX branch 或 Material Custom 节点做一个 UE microbenchmark 列。
- Falcor 的已知短板：2024-08 后无新 release、主线无 cooperative vector、无生产级 clustered 光照（自写 1–2 周，v1 直接逐灯循环 ≤128 灯）。

下文 8.1–8.6 保留作为 UE 路径的设计参考。

### 8.1 为什么以 Substrate 为主要工业基线

UE 5.8 的 [Substrate Materials](https://dev.epicgames.com/documentation/en-us/unreal-engine/overview-of-substrate-materials-in-unreal-engine) 已经提供：

- slab-based principled BSDF；
- vertical/horizontal layering；
- coat、fuzz、anisotropy、rough refraction 等；
- Blendable GBuffer 与 Adaptive GBuffer；
- material simplification 和 closure count 控制。

但复杂多 slab 的代价集中在：

- base pass 编码；
- GBuffer bytes/pixel；
- closure count；
- 每盏灯重复评估多个 closures；
- shader permutation 和编译时间；
- 平台自动 parameter blending 带来的质量下降。

官方示例中，四个 slab 的复杂材质约为 108 bytes/pixel，启用 parameter blending 后可降到约 28 bytes/pixel，但真实层传输被近似。该质量—成本差距正是本研究应该攻击的位置。

### 8.2 不能只依赖 Material Custom 节点

普通 UE Material Custom 节点适合原型验证单个 forward shader，但不足以完成最终论文系统，因为它通常无法完整控制：

- GBuffer layout；
- deferred light loop；
- 自定义 closure packing；
- Substrate decode；
- clustered lights；
- IBL 与 area-light integration；
- per-pixel neural inference backend。

最终实现更可能需要：

- source-built UE；
- renderer module/plugin；
- 自定义 shading model 或 Substrate 扩展；
- HLSL/Slang include；
- cook-time asset compiler；
- GPU resource 和 latent streaming 管理。

### 8.3 推荐渲染路径

#### 快速原型

- forward shading；
- 一个方向光 + HDR IBL；
- closure packet 保存在寄存器中；
- 用于证明模型表达能力和每像素网络成本。

#### 最终论文系统

- desktop deferred/clustered；
- base pass 输出固定 closure packet 或 compact latent；
- lighting pass 解码或读取 closures；
- 动态点光、聚光、area light、IBL；
- 与 Substrate 完整层和 parameter blending 在相同场景比较。

~~需要通过实验决定 decoder 放置位置：~~

1. **Base pass decode**：只运行一次，但增加 GBuffer 存储；
2. **Lighting pass decode**：GBuffer 小，但可能被多 light pass 重复执行；
3. **Pre-light compute decode**：单独 compute pass 写 compact closure buffer，通常是最可控的折中。

**[2026-08-21 修订] 直接定为第三种**，不再花时间比较三种位置。在 Falcor 中即一个 `ClosureDecodePass`（compute），读 GBuffer 写 closure buffer，后接 `DeferredLightingPass`。

### 8.4 神经推理后端

#### 通用后端

- HLSL；
- FP16 或 INT8/DP4a；
- 2–3 层、宽度 32–64 的 MLP；
- 不依赖特定厂商 tensor core；
- 作为公平和跨硬件基线。

#### 加速后端

[RTX Neural Shading SDK](https://github.com/NVIDIA-RTX/Rtxns) 提供 Slang、SlangPy、DirectX/Vulkan cooperative-vector 示例，适合：

- shader 内执行小型 MLP；
- 训练到 shader 的一致实现；
- 使用 RTX Tensor Cores；
- 参考 neural shader 权重布局和推理代码。

[DirectX Cooperative Vector](https://devblogs.microsoft.com/directx/cooperative-vector/) 和后续 DirectX Linear Algebra 允许在普通 shader thread 中表达 vector-matrix/matrix operations。但相关 SDK、驱动和 UE 原生集成仍在演进，因此论文不能只在 preview driver 下成立。

**[2026-08-21 核实]** Shader Model 6.9 已于 2026-02 retail（Agility SDK 1.619、DXC 1.9.2602），但 Cooperative Vector **正被统一的 DirectX Linear Algebra（DXLA）取代**（2026-04 进入 public preview）。结论不变且更强：portable HLSL 是主后端、必须独立成立；coopvec/DXLA 只作加速列，并注明所用 API 版本。

建议同时报告：

- portable HLSL；
- cooperative-vector fast path；
- feature unavailable 时的 fallback；
- 不同 GPU 世代的相对收益。

### 8.5 与 Neural Texture Compression 的关系

[RTX Neural Texture Compression SDK](https://github.com/NVIDIA-RTX/Rtxntc) 可以作为可选资产压缩层：

- 压缩多通道 latent pyramid；
- load-time 解压到 BCn；
- on-sample neural inference；
- feedback-driven streaming。

但 on-sample 模式：

- 单次只产生未过滤 texel；
- 适合高性能 cooperative-vector GPU；
- 通常需要 stochastic texture filtering；
- 会把时间稳定性问题带回系统。

因此第一篇论文建议：

- 主方法使用普通 latent textures 或 load-time 解压；
- NTC 作为独立 ablation；
- 后续再研究 end-to-end shading-aware codec。

### 8.6 性能目标

建议设定公开、可失败的目标：

- desktop DX12；
- 1440p，60/90 FPS；
- neural material decode 预算约 1–2 ms；
- 固定 (K=2) 或 (K=3) closures；
- 复杂材质覆盖屏幕 25%、50%、100%；
- 1、8、32、128 个 clustered lights；
- 旋转 HDRI；
- 动态视角、缩放、相机切换；
- 1080p、1440p、4K 三档；
- 至少测试一块中端 GPU 和一块高端 GPU。

不要只报告单材质球的 shader time，也不要只使用没有阴影、没有 IBL、没有纹理和没有运动的 microbenchmark。

---

## 9. 实验设计

### 9.1 参考真值

- ~~OpenPBR high-spp reference；~~ **[2026-08-21 修订]** 主参考 = Slang 移植的 pbrt-v4 N 层随机游走 teacher（§6.4）；
- PFMC 任意层 reference（交叉验证）；
- PBRT coated-material reference（两层交叉验证，CPU 原版）；
- 实拍 controlled light/view sweeps（v2）；
- 至少一个独立 renderer 的交叉验证，排除 teacher-specific overfitting。

**[2026-08-21 修订] PT-vs-raster 对比协议。** 所有列共用同一 GBuffer、同一组灯、同一 HDRI、同一可见性（deferred 路径用 DXR 1.1 RayQuery 描影线，与 PT 的 shadow ray 完全一致），差异就只剩"材质表示误差 + IBL 积分近似"：

| 列 | 实现 | 作用 |
|---|---|---|
| 材质级参考 | Falcor `PathTracer`，`maxSurfaceBounces=0` + NEE + env，材质 = 随机游走 teacher，渐进收敛 | 排除场景级 GI 的 ground truth |
| ours | `GBufferRaster → ClosureDecodePass → DeferredLightingPass`（解析灯 + split-sum IBL + RayQuery 阴影） | 被测方法 |
| 解析基线 | 同上 deferred 路径，closure 来自 Belcour / Principled fit | 同管线公平对比 |
| 场景级语境 | Falcor `PathTracer`，`maxSurfaceBounces=N` | 说明本方法不承担的部分 |

三列同图即 teaser 骨架。

**[2026-08-21 修订] 单仓库项目结构：**

```text
neural-closure/
├── teacher/      Slang N 层随机游走 BSDF（pbrt-v4 移植，Apache）；PFMC CPU 交叉验证（GPL，隔离目录，仅验证）
├── datagen/      Python + Falcor ComputePass：采样层栈/位置/ω_o/footprint，GPU 查 teacher，写 A/B bin tile + 元数据
├── model/        PyTorch；closure 评估核用 slangtorch 与渲染端共享
├── viewer/       Falcor render graph（上表四列）+ 可复现 benchmark 脚本（固定相机路径、EXR、GPU 计时）
├── baselines/    Belcour 2018 移植、Principled/OpenPBR 参数拟合、NLBRDF（官方码）、NA（官方码）、NMA（复现）
└── ue/           可选：stock UE Substrate 场景 + 成本采集脚本
```

### 9.2 主要基线

必须至少包含：

1. UE Default Lit；
2. UE Substrate 完整多 slab；
3. UE Substrate Parameter Blending；
4. Belcour 2018；
5. MIPNet 或标准 normal/roughness mip；
6. Neural Layered BRDFs；
7. Real-Time Neural Appearance；
8. Neural Material Adapter；
9. 将复杂材质直接拟合为固定 OpenPBR/Principled 参数；
10. 若启用残差，加入 STF/glint baseline。

并非所有 baseline 都必须在 UE 中重写。可以分成：

- directional/function-space benchmark；
- offline image benchmark；
- real-time UE benchmark。

但 UE Substrate 完整层与 Parameter Blending 必须在相同 UE 版本、场景、分辨率和 GPU 下测量。

**[2026-08-21 修订] 三档基线的具体归属：**

| 档 | 指标 | 基线 | 运行处 |
|---|---|---|---|
| function-space | 方向域 SMAPE、能量、互易、峰值/宽度 | Belcour、Principled/OpenPBR fit、NLBRDF、NMA、ours | PyTorch / Mitsuba（NLBRDF 官方码） |
| image-space 离线 | 同 HDRI/灯下 HDR-FLIP、LPIPS | 以上 + NA | Falcor（NA 官方码同框架） |
| real-time | GPU 时间、bytes/px、光源数 scaling | Belcour、Principled、ours | Falcor 同 render graph |
| 工业成本 | bytes/px、GPU 时间 | Substrate 全 slab / 参数混合 | stock UE，同资产同 GPU 同分辨率 |

iso-byte 对比与两条 Pareto 见 §4.6。

### 9.3 材质族

测试集应覆盖：

- varnished wood；
- coated brushed metal；
- metallic car paint；
- dusty metal；
- glazed ceramic；
- wet stone；
- fuzz fabric；
- peach/sugar coating；
- rough colored coat；
- thin-film/iridescent material，若纳入第二阶段。

### 9.4 场景与动态序列

至少包括：

- 标准 sphere/teapot/blob，用于方向误差；
- 曲率和 UV 变化复杂的 hero asset；
- 汽车或工业产品场景；
- 多动态光室内场景；
- 远近尺度变化的材质平面/地面；
- 相机快速横移和旋转；
- 物体运动、光源运动、HDRI 旋转；
- 复杂材质占屏率变化场景。

### 9.5 质量指标

不能只报告 PSNR。建议包括：

- HDR-FLIP；
- LPIPS；
- SSIM/PSNR 作为传统补充；
- tone-mapped 与 linear HDR 两种误差；
- directional SMAPE；
- highlight peak/width/centroid error；
- temporal FLIP；
- motion-warped temporal error；
- flicker power spectrum；
- material-recognition/perceptual preference user study。

### 9.6 物理指标

- energy conservation violation；
- reciprocity error；
- associativity error；
- non-negative violation；
- integrated directional albedo error；
- roughness/IOR 极值稳定性；
- layer order sensitivity accuracy。

### 9.7 系统指标

- GPU frame time；
- 各 pass GPU time；
- neural inference time；
- light-count scaling；
- material-count scaling；
- occupancy；
- tensor/ALU utilization；
- memory bandwidth；
- GBuffer bytes/pixel；
- latent asset size；
- streaming bandwidth；
- cook/update time；
- shader permutation 和编译时间。

### 9.8 泛化测试

最关键的实验不是同分布重建，而是：

- 训练只见 2–4 层，测试 5–8 层；
- 测试未见层顺序；
- 测试未见 layer type pair；
- 测试未见参数范围；
- 测试未见材质网站来源；
- 测试未见 HDRI；
- 测试未见几何曲率；
- 测试未见 footprint 和倾斜角；
- 修改一层后无需重新训练；
- 在非 NVIDIA 或不支持 cooperative vector 的 GPU 上运行 portable backend。

### 9.9 必要消融

- 无 associativity loss；
- 无 energy constraint；
- 无 reciprocity constraint；
- 无 footprint conditioning；
- 普通参数 mip vs transport latent mip；
- (K=1/2/3/4) closures；
- fixed analytic closures vs dynamic type selection；
- **[2026-08-21 修订] closure 函数族 ① GGX 参数族 vs ② LTC 瓣 vs ③ 学习字典（oracle 上界与网络预测各一组）；④ 全神经 g 逐光评估作为质量上界列，同时报告其灯数 scaling 成本**；
- 无 stochastic residual；
- per-material training vs universal model；
- sequential vs balanced-tree composition；
- FP32、FP16、INT8；
- portable HLSL vs cooperative-vector backend。

### 9.10 用户实验

材质相似度不总能由像素指标正确刻画，建议做 matched-time pairwise study：

- 给出离线参考视频；
- 对比 ours、Substrate、Parameter Blending、NMA/其他最强 baseline；
- 问题为“哪个更像参考材质”，而不是“哪个更好看”；
- 分材质类别统计；
- 区分静帧和动态 sweep；
- 报告置信区间和参与者一致性。

---

## 10. 预期论文贡献与可证伪主张

### 主张 1：可组合性

~~一个共享、顺序敏感的层组合网络，可以在不针对新材质重新训练的情况下，处理训练层数之外的多层栈。~~

**[2026-08-21 修订]** 一个共享的层编码器 + 结构化散射算子组合（star product），可以在不针对新材质重新训练的情况下，零样本编译**训练时未见的层类型 × 顺序 × 参数组合**，并支持改一层不重训；层数外推为次要证据。

可证伪条件：

- 未见 layer pair / 未见顺序 / 未见参数范围无法泛化；
- 只能通过 per-material latent optimization 恢复质量；
- 5–8 层误差显著失控（次要）；
- ~~不同括号顺序差异过大~~（主线由结构保证，仅对学习式 compose 消融测量）。

### 主张 2：固定成本的光栅化闭包

当前研究假设是：复杂层栈可以归约为固定成本的解析闭包，在动态光源和环境光下优于普通参数混合并接近离线参考。第一轮表示上界实验支持“精确顶层界面 + LTC 残差”的分解方法，但尚未支持把瓣数固定为 K≤3；瓣数和残差函数族仍需由后续长尾实验决定。

可证伪条件：

- **[2026-08-21 修订]** oracle 拟合（无网络）在目标材质族上已不达标——表示本身不够，与网络无关；
- 网络预测与 oracle 之间差距过大——表示够但学不到；
- 必须使用过多 closure 才能匹配参考；
- IBL 下误差远大于点光；
- glint/fuzz 被不可接受地平滑；
- 固定 packet 的 GBuffer 成本接近完整 Substrate。

### 主张 3：多尺度稳定性

transport-aware latent pyramid 能在不 supersampling 的情况下保持动态缩放和运动时的材质外观稳定。

可证伪条件：

- temporal error 不优于普通 mip 或 MIPNet；
- 出现明显 popping、shimmering 或 lobe rotation；
- footprint decoder 成本超过质量收益。

### 主张 4：工业部署价值

~~在 UE 的 raster/deferred 管线中，~~ **[2026-08-21 修订]** 在完全 rasterized 的 deferred 管线（Falcor 系统）中，本方法以与 stock UE 中实测的 parameter-blended Substrate 相当的 bytes/px 与 GPU 成本，获得明显接近离线层传输的质量；UE 内部署为可选的额外证据。

可证伪条件：

- 只在自制 renderer 中快；
- UE integration 后 decode、GBuffer 或 light loop 成本过高；
- 只在 RTX 50 系 preview driver 上成立；
- 场景复杂度上升后收益消失。

---

## 11. SIGGRAPH 级别所需的完整证据链

仅有漂亮结果图不够。建议论文同时具备：

1. **方法贡献**：新的可组合 closure/transport 表示；
2. **物理结构**：能量、顺序、组合和过滤约束；
3. **泛化证据**：未见层数、顺序和材质；
4. **系统贡献**：UE 真实渲染路径；
5. **性能证据**：多 GPU、多分辨率、多光源；
6. **真实数据**：实测 BRDF/SVBRDF 和自采 paired layers；
7. **时域证据**：动态视频而非静态球体；
8. **感知证据**：FLIP 和用户实验；
9. **可复现性**：teacher、数据生成器、模型和 UE demo；
10. **诚实边界**：明确不解决场景级全局光传输。

如果缺少 UE 系统与真实数据，文章更像神经 BRDF 表示工作；如果缺少 variable topology 和物理结构，则容易被认为是 NMA/Neural Appearance 的工程组合。

---

## 12. 风险、失败模式与止损标准

### 12.1 Novelty 被已有工作覆盖

风险：Neural Layered BRDFs 已经提出 latent layering；NMA 已经输出解析 BRDF。

应对：论文贡献必须是两者都没有完整覆盖的交叉点：

- variable-topology layer algebra；
- fixed raster closure packet；
- footprint-aware transport filtering；
- no per-material training；
- UE real-time evidence。

### 12.2 固定 closure 数量表达不足

风险：复杂多层响应可能需要很多瓣，(K=3) 过于限制。

应对：

- 使用动态 closure type；
- 保留低能量 residual；
- 单独报告不可表示材质域；
- 将第一版范围限制为 reflection-only opaque stacks；
- 比较增加 (K) 带来的真实成本和边际收益。

止损：若 (K=4) 仍无法在目标材质族上明显优于 Substrate Parameter Blending，应重新评估主表示。

**[2026-08-21 修订] 该风险的处置前移到训练之前。** 原文把"closure packet 无法表达目标材质族"埋在 P1 止损条件里、放在训练之后，这会把表示不足与网络学不到混为一谈。定稿：P1.0 先做 oracle 天花板实验（§4.6 (4)、§14），把表示误差与预测误差分离。决策树：

- 第一轮实测选择“精确顶层界面 + LTC 残差”作为继续研究的基础；
- K2 只作为实现和编译器基线，不能据此关闭表示问题；
- 先针对导体基底和深层栈测试更细的解析路径拆分、不同残差族和按需附加瓣；
- 只有新表示在方向域长尾、点光图像和真实 HDRI 三类指标上都稳定后，才固定网络输出结构；若仍不达标，再收缩目标材质域并明确不可表示范围。

### 12.3 组合泛化失败

风险：训练 2–4 层，测试 5–8 层时 latent 分布漂移。

应对：

- normalized latent；
- 显式 depth token；
- balanced tree；
- randomized bracketing；
- recurrent stability loss；
- 物理能量通道与自由 residual 通道分离。

止损：若新拓扑必须重新训练，则不再宣称 algebra，改为“universal layered material compiler”。

**[2026-08-21 修订] 全项目最高风险项：通用模型在未见层栈上匹配尖锐高光。** NA 正是因为这点选择了逐材质训练。这应是 Phase 1 止损条件的第一条；结构化算子 + 解析近镜面通道（§4.4）正是为此设计，若仍不成立则退路是"逐材质族（per-family）微调"并如实降级主张。

### 12.4 UE 推理成本过高

风险：network decode 的理论速度在 UE 的真实寄存器压力、GBuffer 带宽和 shader divergence 下消失。

应对：

- 先做 portable HLSL microbenchmark；
- 比较 base-pass、lighting-pass、pre-light compute 三种位置；
- 批量按材质/网络分组；
- 固定 packet layout；
- quantization-aware training；
- cooperative-vector 仅作为加速而非唯一方案。

止损：如果 1440p 全屏 decode 在目标高端 GPU 上持续超过约 2–3 ms，且质量优势不能允许半/四分辨率执行，应缩小网络或改用 LUT/hybrid representation。

### 12.5 时域闪烁

风险：高频层通过 latent mip 或 stochastic residual 产生闪烁。

应对：

- 把 temporal metrics 纳入早期实验；
- 使用材质空间而非仅屏幕空间过滤；
- 对 mip transition 加一致性训练；
- 对 stochastic residual 提供 deterministic fallback。

止损：若方法依赖强 TAA/DLSS 才能隐藏噪声，不能宣称自身实现稳定材质过滤。

### 12.6 合成到真实域差距

风险：程序化 OpenPBR/PFMC 栈无法覆盖真实复杂材料。

应对：

- measured BRDF/SVBRDF validation；
- 自采 paired layer samples；
- OpenPBR 与 PFMC 双 teacher；
- 少量 real-data fine-tuning 与严格 held-out 测试；
- 用户实验。

### 12.7 数据许可证

风险：商业材质资产禁止 ML，或测量数据库不能重新发布。

应对：

- closure 编译器主干只依赖自生成层栈响应；MatSynth、Poly Haven、ambientCG 仅进入后续空间参数图与场景验证；
- Adobe Substance Assets 明确排除；
- 每个样本记录来源和许可证；
- measured databases 只作验证，默认不随项目重新分发；
- 发布前由机构进行许可证复核。

---

## 13. 两个次选研究方向

### 13.1 Shading-Aware Neural Material Codec

#### 核心想法

将 neural texture compression 与 neural appearance 端到端联合训练，不再要求重建 base color、normal、roughness 等 PBR maps，而是从压缩 latent 直接输出 raster closures。

优化目标从纹理 PSNR 改为：

- 动态光照下的 HDR-FLIP；
- 多视角方向误差；
- 多尺度时域稳定性；
- 每材质 bitrate 和 shader time。

#### 可能贡献

- layer-aware bit allocation；
- 对不同通道和空间区域按 shading sensitivity 分配码率；
- latent mip 和 closure decoder 联合优化；
- random-access streaming；
- UE 资产体积、显存和运行时间三目标 Pareto curve。

#### 判断

工程可行性最高，容易构建在 NTC 和 Neural Appearance 工具链上，但“把两者端到端联合”比较直观。要达到 SIGGRAPH，需要新的率失真理论、随机访问结构或显著的实时系统结果。

### 13.2 Analytic Core + Stochastic Material Residual

#### 核心想法

使用稳定解析 closures 表示主体，只对 glint、flake、fiber 等稀疏高频成分进行低率随机采样和时空重建。

#### 可能贡献

- material-space residual sampling；
- 根据 residual energy 和 perceptual saliency 分配采样；
- 与解析主体共享能量；
- 非 path-tracing 的稀疏材质事件采样；
- 动态缩放下无偏或低偏重建。

#### 判断

画面冲击力强，但与 STF、glint rendering、ReSTIR 和 temporal reconstruction 的竞争最激烈。更适合作为 Neural Closure Algebra 的可选模块，而不是单独承担整篇论文。

---

## 14. 推荐执行路线

**[2026-08-21 修订] 总体周期与目标会议**（一人主力 + 零星工程协助）：

| 阶段 | 内容 | 原估计 | 修订估计 |
|---|---|---|---|
| P0 基础设施 | Slang 随机游走 teacher、datagen、Falcor 双路径 viewer、Belcour/Principled 基线 | 2 周（只算文献） | **6–8 周** |
| P1 kill test | 2–3 层 reflection-only，K=2/3，方向光 + IBL，变拓扑初测 | 4 周 | 4 周 |
| P2 方法 | 结构化算子、spatial、footprint pyramid、function/image benchmark、NLBRDF/NA/NMA 对比 | 8–12 周 | 10–14 周 |
| P3 系统 | Falcor deferred + 逐灯/clustered + LTC + 性能；portable HLSL | 未单列 | 6–8 周 |
| P3' UE（可裁） | source-built 插件 + Substrate 扩展 | 未单列 | +8–12 周 |
| P4 评估写作 | 公开实测集、时域视频、用户实验、多 GPU、写作 | 未单列 | 8–10 周 |
| **合计** | | 隐含 ~7 个月 | **不含 UE 8–10 个月；含 UE 10–13 个月** |

投稿窗口（按历年惯例）：SIGGRAPH 2027（约 2027-01 下旬）完整版不可行；EGSR 2027（约 2027-04）精简版兜底；**SIGGRAPH Asia 2027（约 2027-05）为主目标**；加 UE 与自采数据的完整版投 SIGGRAPH 2028（约 2028-01）。难度 4/5，按难度排：① 通用模型匹配未见层栈的尖锐高光；② GPU teacher 基础设施；③ footprint pyramid 调到不闪；④ UE 集成；⑤ 实测/自采数据。

投稿前与开工前各做一次竞品扫描（SIGGRAPH Asia 2026 录用结果约 2026-09 公布）。

### 阶段 0：~~两周文献与基线锁定~~ 6–8 周基础设施 **[2026-08-21 修订]**

详细任务拆解见 `idea-neural-layered-materials-P0-任务清单-2026-08-21.md`。概要：

- WS-A：pbrt-v4 `LayeredBxDF` 推广为 N 层并移植 Slang；Falcor Python ComputePass datagen；与 pbrt CPU / Falcor `PBRTCoated*` / PFMC 交叉验证；生成 v0 数据集；
- WS-B：Falcor render graph——自定义材质类型、`ClosureDecodePass`、`DeferredLightingPass`（RayQuery 阴影 + split-sum IBL）、PathTracer 参考列、benchmark 脚本；
- WS-C：层栈描述 schema、Belcour/Principled fit 基线、stock UE Substrate 成本场景；
- 文献部分并行：运行 NLBRDF、NA 官方代码，确认 NMA 边界，选定 UE Substrate benchmark 场景。

### 阶段 1：四周止损实验

**[2026-08-22 状态] P1.0 第一轮表示上界实验已完成。** 结果没有支持直接固定 LTC-K3，而是支持保留“精确顶层界面 + LTC 残差”这一分解方式。进入网络 kill test 前，先完成针对导体基底和深层栈的表示改进，并补充点光图像指标；结果与决策见 `reports/oracle_ceiling_v0.md`。

- 只做双层和三层 reflection-only；
- 网络输出 (K=2/3) closures；
- 使用方向光和 IBL；
- 测量是否明显优于普通 Principled fit 和 Parameter Blending；
- 用 HLSL 跑全屏 decoder microbenchmark；
- 初步测试训练三层、推理五层。

#### 阶段 1 止损条件

- ~~closure packet 无法表达目标材质族；~~ → **[2026-08-21 修订]** 拆为两条：(a) P1.0 oracle 已不达标（表示不够，按 §12.2 决策树升级或收缩）；(b) 网络预测与 oracle 差距过大（表示够但学不到，属泛化风险，见 §12.3）；
- IBL 下误差远大于点光；
- variable-depth 泛化完全失败；
- decoder 成本不可能进入 1–2 ms 量级；
- 结果只能靠 per-material optimization 达成。

### 阶段 2：~~八至十二周~~ 10–14 周方法原型 **[2026-08-21 修订]**

- 结构化散射算子 + 解析近镜面通道（§4.4）；
- ~~associativity/energy 约束~~ → 由结构保证，改为消融对照；
- spatial material maps；
- footprint → 层统计量 + 重新 compose 的 latent mip（§4.5）；
- Falcor raster prototype（P0 已搭骨架）；
- 与强 baselines 的 directional/image benchmark；
- 5 路消融在 A6000 农场并行（§7.5）。

### 阶段 3：~~UE 系统~~ Falcor 系统（UE 为 P3' 可裁项）**[2026-08-21 修订]**

- ~~source-built UE renderer plugin~~ → Falcor deferred 路径性能化：逐灯 → clustered、LTC 区域光；
- pre-light compute decode（已定）；
- compact closure buffer 与 iso-byte packing；
- dynamic lights、IBL、area lights；
- portable HLSL 与 coopvec/DXLA 两个 backend；
- Nsight/PIX profiling，多 GPU、多分辨率；
- stock UE 中测 Substrate 成本列；
- P3'（可裁）：UE RTX branch / Custom 节点 microbenchmark，或 source-built 插件。

### 阶段 4：真实数据与论文实验

- OpenSVBRDF、MERL、RGL、UBOFAB19；
- 自采 paired layered materials；
- 时域视频；
- 用户实验；
- 多 GPU、多分辨率；
- 公共数据生成器和 UE demo。

---

## 15. 可能的论文标题与叙事

### 主标题候选

- **Neural Closure Algebra for Rasterized Layered Materials**
- **ClosureCraft: Composable Neural Layer Transport for Real-Time Rendering**
- **LayerCompile: Distilling Arbitrary Material Stacks into Real-Time Raster Closures**
- **From Layer Graphs to G-Buffers: Neural Compilation of Complex Materials**
- **Composable and Filtered Neural Closures for Real-Time Layered Appearance**

### 推荐的一句话叙事

> Existing neural materials accelerate BSDF queries in path tracers or bake individual material graphs, while real-time engines require a fixed, filterable representation that can be integrated against many dynamic lights. We introduce a composable neural layer algebra that compiles arbitrary material stacks into a small packet of analytic raster closures, preserving multilayer appearance, editability, and level-of-detail behavior at fixed runtime cost.

### 最强结果图应展示什么

一张真正有说服力的 teaser 应同时展示：

- 同一 hero asset；
- 5–8 层材质；
- 动态区域光或旋转 HDRI；
- 近景和远景；
- 离线 PFMC/OpenPBR reference；
- UE Substrate Parameter Blending；
- 本方法；
- 每帧 GPU 时间和 GBuffer bytes/pixel；
- 插图展示可实时改变 clearcoat、dust 或 layer order，无需重新训练。

---

## 16. 最终判断

该方向值得推进，但需要避免把目标降格为又一个 neural BRDF。~~最有 SIGGRAPH 潜力的研究命题是：~~

~~在限定原子层词汇内，学习可组合、可过滤的层传输 latent，将任意深度和空间变化的材质栈零样本编译成固定数量的 UE-compatible analytic closures，并在完全 rasterized 的动态光照管线中，以接近 Parameter Blending 的成本获得接近离线 layered transport 的局部材质外观。~~

**[2026-08-21 修订] 定稿命题：**

> **在限定原子层词汇内，把任意顺序与空间变化的层栈编译为固定 K≤3 个 raster closures；层组合由学习基底上的散射算子以 star product 完成（结合律与能量守恒由构造保证），未见的层类型 / 顺序 / 参数组合零样本成立、改一层不重训；在完全 rasterized 的 deferred 管线中，以接近 Substrate 参数混合的 bytes/px 与 GPU 成本取得接近随机游走参考的局部材质外观。**

与原命题的差异：去掉"5–8 层"头条、去掉"UE 必需"、teacher 从 OpenPBR 改为随机游走、结合律从 loss 改为结构。

决定论文级别的关键不是单帧材质球质量，而是以下结果能否同时成立：

1. ~~训练 2–4 层、测试 5–8 层仍然稳定~~ → **未见的层类型 × 顺序 × 参数组合零样本成立**（层数外推为次要）；
2. 动态点光、区域光和 IBL 都能直接使用；
3. 缩放和运动下无明显闪烁；
4. 修改层参数或顺序无需重新训练；
5. ~~UE 中的真实成本~~ → Falcor deferred 中的真实成本接近 stock UE 实测的 parameter-blended Substrate（iso-byte）；
6. 公开实测数据（MERL / RGL / OpenSVBRDF / UBOFAB19）仍明显接近离线参考（自采 paired layers 为 v2）；
7. portable HLSL 后端也成立，而不是依赖单一 preview 硬件路径。

若这七项中的大部分能够完成，该工作不仅是神经材质表示论文，也会成为连接离线材质图、神经着色和现代游戏引擎的一项完整实时渲染系统贡献。

---

## 17. 主要参考链接

### Layered materials 与 neural BRDF

- [Efficient Rendering of Layered Materials using an Atomic Decomposition with Statistical Operators](https://belcour.github.io/blog/research/publication/2018/05/05/brdf-realtime-layered.html)
- [Position-Free Monte Carlo Simulation for Arbitrary Layered BSDFs](https://projects.shuangz.com/layered-sa18/)
- [Neural Layered BRDFs](https://wangningbei.github.io/2022/NLBRDF.html)
- [MetaLayer: A Meta-Learned BSDF Model for Layered Materials](https://doi.org/10.1145/3618365)
- [SpongeCake: A Layered Microflake Surface Appearance Model](https://arxiv.org/abs/2110.07145)
- [Neural Material Adapter](https://studios.disneyresearch.com/2026/07/01/neural-material-adapter-transforming-complex-materials-into-efficient-analytic-bsdfs/)
- [A Hybrid Neural-Microfacet BRDF Model for Real-Time Rendering](https://ubisoft-laforge.github.io/world/hybridrdf/)

### Neural appearance、压缩与过滤

- [Real-Time Neural Appearance Models](https://research.nvidia.com/labs/rtr/neural_appearance_models/)
- [NVLabs Neural Appearance Pipeline](https://github.com/NVlabs/neuralappearance)
- [Random-Access Neural Compression of Material Textures](https://research.nvidia.com/publication/2023-08_random-access-neural-compression-material-textures)
- [RTX Neural Texture Compression SDK](https://github.com/NVIDIA-RTX/Rtxntc)
- [MIPNet](https://perso.telecom-paristech.fr/boubek/papers/MIPNet/)
- [Filtering After Shading with Stochastic Texture Filtering](https://research.nvidia.com/publication/2024-05_filtering-after-shading-stochastic-texture-filtering)
- [Improved Stochastic Texture Filtering Through Sample Reuse](https://research.nvidia.com/labs/rtr/publication/wronski2025quadcomm/)
- [Real-Time Level-of-Detail Rendering with ReSTIR](https://research.nvidia.com/labs/rtr/publication/wang2026levelofdetail/)
- [Toward Richer Material Generation via Procedural Data Enhancement](https://blaire9989.github.io/assets/4_DataEnhance/project.html)

### 开放材质标准与 teacher

- [OpenPBR](https://github.com/AcademySoftwareFoundation/OpenPBR)
- [Adobe OpenPBR BSDF](https://github.com/adobe/openpbr-bsdf)
- [MaterialX](https://materialx.org/Specification.html)
- [PBRT-v4 Layered Materials](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Scattering_from_Layered_Materials)

### 数据集

- [MatSynth](https://research.adobe.com/publication/matsynth-a-modern-pbr-materials-dataset/)
- [MatSynth GitHub](https://github.com/Code-SY95/MatSynth)
- [Poly Haven License](https://polyhaven.com/license)
- [Poly Haven API](https://polyhaven.com/sr/our-api)
- [OpenSVBRDF](https://svbrdf.github.io/publications/OpenSVBRDF/project.html)
- [MERL BRDF Database](https://merl.com/research/downloads/BRDF)
- [RGL Material Database](https://rgl.epfl.ch/pages/lab/material-database)
- [Bonn SVBRDF / UBOFAB19](https://cg.cs.uni-bonn.de/btf/bonn_svbrdf_database.html)
- [Adobe Substance 3D Assets Terms](https://www.adobe.com/go/substance3dassets)

### UE 与神经 shader 部署

- [Unreal Engine Substrate Materials Overview](https://dev.epicgames.com/documentation/en-us/unreal-engine/overview-of-substrate-materials-in-unreal-engine)
- [Programming with Substrate GBuffer Formats](https://dev.epicgames.com/documentation/unreal-engine/programming-with-substrate-gbuffer-formats)
- [RTX Neural Shading SDK](https://github.com/NVIDIA-RTX/Rtxns)
- [DirectX Cooperative Vector](https://devblogs.microsoft.com/directx/cooperative-vector/)
- [DirectX Linear Algebra Preview](https://devblogs.microsoft.com/directx/d3d12-linalg-preview/)
