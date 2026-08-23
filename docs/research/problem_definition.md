# 当前研究问题：受运行时约束的材质函数压缩

## 一句话结论

NeuralShading 的本质不是“用网络回归几个材质参数”，也不只是“把一个 BRDF 拟合得更快”，而是：**在保留源材质原生语义和编辑状态的前提下，压缩一个带材质状态、空间、尺度和双方向条件的查询函数，使它能够在 GPU shader 中随机访问，并让每次查询的时间、资产内存和共享运行时成本都有明确上界。**

## 1. 被压缩的对象是什么

对一个源材质资产 `M` 及其原生编辑状态 `θ`，reference 定义局部散射查询：

```text
f_M,θ(x, footprint, frame, wo, wi) -> RGB scattering
```

- `x` 是表面位置，当前局部材质通常用 `uv` 表示；
- `footprint` 是一个 pixel 或 ray cone 覆盖的纹理区域，不等价于单个 scalar mip level；
- `frame` 是几何/着色法线与切线约定；
- `wo`、`wi` 各自在局部半球上有两个自由度；
- 输出可能跨越多个数量级，并可能含极窄高光、颜色通道差异和 reference Monte Carlo 噪声。

编译后程序写成：

```text
A_M,θ = compile_material(M, θ)
h      = prepare(A_M,θ, x, footprint, frame, wo)
f_hat  = evaluate(h, wi)
```

`A_M,θ` 是 view-independent 的 latent 资产；`h` 只在当前着色点和当前 `wo` 下复用；`evaluate` 是逐 `wi` 的小型 neural decoder。需要 path tracing 时再从同一个 `h` 构造与 evaluator 匹配、PDF 可计算的 `sample/pdf`，但这不是当前 evaluator 容量实验的前置条件。

对常量局部 LayerStack，`x` 和 `footprint` 暂时退化，第一轮实际压缩的是：

```text
(material state, wo, wi) -> RGB f
```

这一步只验证方向函数与跨状态共享，不证明空间 latent、纹理过滤或 LOD 已经解决。

## 2. 这是一个带约束的率失真问题

候选方法不能只按 validation loss 排名。它同时决定：

```text
失真 D(reference, program)
材质专属 bytes B_asset
共享权重 bytes B_shared
prepare 成本 C_prepare
单次 evaluate 成本 C_eval
编译/编辑延迟 C_compile
随机访问的局部访存范围 R_access
```

研究目标是寻找这些量的 Pareto 前沿，而不是先固定网络再解释其成本。共享 decoder 的权重可以跨材质摊销，但报告单个材质和场景级成本时必须明确摊销方式；材质专属网络权重不能藏在“全局 runtime”里不计。

随机访问是硬约束：查询一个 `(x, wo, wi)` 不能要求解码整张纹理、整个方向表或相邻帧序列。允许读取固定数量的局部 latent texel、共享 codebook 条目和小型网络权重；读取数必须与分辨率、材质图复杂度和历史查询顺序无关。

## 3. 高维并不表示每一维都同样高频

这个函数的可压缩性来自多种不同相关性：

| 维度/结构 | 典型相关性 | 主要例外 |
|---|---|---|
| `u,v` | 邻近 texel、重复纹理、跨通道边缘相关 | mask、UV seam、离散图案和高频法线 |
| footprint/LOD | 粗级别是细级别外观的滤波结果 | 非线性 shading 使“先平均参数再着色”不等于正确滤波 |
| `wo,wi` | 互易性、各向同性/各向异性对称、half/difference 结构 | 极窄高光、retroreflection、多峰和遮蔽突变 |
| RGB/光谱 | 相同几何路径驱动通道，形状常相关 | 导体、薄膜、色散和窄谱效应 |
| material state `θ` | 小参数编辑通常产生连续响应 | 图拓扑切换、layer/mask 分支和离散资源替换 |
| material identity | 许多材质共享散射结构 | 测量异常、特殊光学效应和不同 source family 语义 |

因此，直接把所有坐标交给一个通用 MLP，或机械地为每对维度建立 plane，并不是唯一合理的表示。更重要的问题是：**哪些轴应被组成一个向量共同编码，哪些轴应保留为运行时查询，哪些高频应先通过坐标变换或物理 core 对齐。**

原始 `wi` 空间里的镜面峰会随 `wo`、法线和粗糙度移动；一个物理上平滑的材质状态变化，在固定方向网格上可能表现成高速移动的尖峰。Rusinkiewicz half/difference 参数化、learned shading frame、microfacet warp 或 analytic-core + residual 的价值，是把“移动结构”变成更容易共享的规范坐标，而不只是增加 positional encoding。

## 4. Neural Dynamic GI 与用户字典方案揭示了什么

Neural Dynamic GI（NDGI）压缩的是：

```text
L(u, v, t) -> RGB indirect lighting
```

它使用 `uv`、`ut`、`vt` feature planes，加一个低分辨率 `uvt` volume 和轻量 MLP，并在训练中模拟量化/BC7，运行时结合 virtual texture。它说明高维查询可以通过“显式局部 feature + 小 decoder + GPU 原生压缩/流送”成为实际系统；但材质额外包含两个方向、移动的高动态范围峰和 sampling/PDF，所以 NDGI 不是可直接复制的架构。[论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Wu_Neural_Dynamic_GI_Random-Access_Neural_Compression_for_Temporal_Lightmaps_in_CVPR_2026_paper.pdf)

用户提供的替代方案可以完整表述为一个 K-means++ top-2 轨迹字典 codec。对 26 个时间点的 RGB lightmap，先把每个像素的时间轨迹展平为：

```text
x_p in R^(3*26)
```

设全部像素轨迹组成 `H×W×78` 的输入，对它们做 K-means++，得到 `N×78` 的 codebook。对每个像素执行以下确定性编码过程：

1. 找到距离该 78 维轨迹最近的 5 个 codeword；
2. 固定最近的 codeword 为 `c1`，分别尝试以第 2–5 近的 codeword 为 `c2`；
3. 对每个 `(c1,c2)` 用一维最小二乘或等价标量优化求 `w∈[0,1]`，最小化 78 个分量上的 MSE；
4. 保存误差最小的两个 codeword index 和对应权重。

重建公式是：

```text
x_p_hat = (1 - w_p) * c[index_1(p)] + w_p * c[index_2(p)]
```

编码资产完整地由以下三部分组成：

```text
index map   : H × W × 2
weight map  : H × W × 1
codebook    : N × (3 × 26)
```

随机访问解码 `(u,v,t)` 时，从 index/weight map 读取 `i1`、`i2`、`w`，只取两个 codeword 在时间 `t` 对应的 RGB 三元组，计算 `(1-w)c[i1,t] + wc[i2,t]`。不需要重建其他像素，也不需要运行完整 78 维 decoder。用户经验是：该方案比 tri-plane 优化更快，表达效果也更好。这里把它记录为**项目经验与待复现实验假设**，不把它写成 NDGI 论文的方法或普遍结论。上述文字、形状和公式就是该方案在项目文档中的完整定义，不依赖外部示意图。

### 4.1 真正可迁移的机制

1. **按相关性选择向量化单元。** 一个像素的完整时间轨迹比互不相关的单点更适合聚类；对应到材质，应寻找“同一材质/texel 在规范化方向或编辑状态上的响应轨迹”。
2. **共享字典 + 稀疏局部系数。** 大部分信息进入少量共享原型，每个查询位置只保存少量 ID 和权重，天然支持有界随机访问。
3. **凸混合是低成本 decoder。** top-2 + scalar weight 比通用 MLP 更便宜、更容易量化，也能避免每个维度独立编码破坏跨维相关性。
4. **启发式初始化可以胜过纯梯度自发现。** K-means 先建立高覆盖原型，再优化局部混合，可能显著缩短训练。

### 4.2 不能直接照搬的部分

- 对固定 26 帧，codeword 可以显式保存完整轨迹；材质必须连续查询 `wo/wi`，不能为所有方向存完整 codeword 表后仍声称成本有界。
- top-2 线段只能表达 codebook 中两点之间的变化；多峰、尖峰移动和离群材质可能需要 top-k、residual VQ、product quantization 或一个可查询的 codeword function。
- RGB MSE 与凸混合都会偏向条件均值，容易降低峰值和对比度；材质的长尾更严重。
- 该 lightmap 方案没有主动利用相邻像素 index 的空间平滑性，实际收益还会受 index/weight map 的 BC 或熵编码方式影响。

### 4.3 对本项目的三个可测候选

| 候选 | 表示 | 用途 |
|---|---|---|
| 方向响应字典 oracle | 把一个常量材质在固定 `wo×wi` probe 上的变换后响应组成向量，再做 top-k 字典 | 只判断数据是否存在可聚类原型，不作为最终 runtime |
| latent dictionary | target encoder 或 source compiler 输出少量 codeword ID、权重和可选 residual；共享 evaluator 读取混合后的 embedding | 保留连续方向查询，是第一轮应实现的字典候选 |
| codeword function mixture | 每个 codeword 是共享的小型方向函数或 expert，`prepare` 选择 top-k，`evaluate` 混合其输出 | 表达力更强，但逐查询成本随 `k` 增长，应与单 MLP iso-time 比较 |

这三者必须与 dense latent + MLP、低秩 tensor/plane 和 analytic-core + residual 在相同 asset bytes、共享权重和 query time 下比较。

## 5. Target encoder、autodecoder 与 source compiler 不能混为一谈

### 5.1 被压缩目标作为输入的 encoder + decoder

这里的 encoder 首先是一种**训练期/压缩期 latent inference**，输入是要被压缩的目标 tensor 本身，而不是源材质参数。以一个很厚的多通道 texture 为例：

```text
X                    : C × H × W 的被压缩目标
Z = E_psi(X)         : encoder 生成的 latent tensor
x_hat(q) = D_phi(Z,q): decoder 随机访问坐标 q 并重建目标样本
```

`C×H×W` 只是 texture 例子；对本项目，`X` 也可以是把 `material state × wo × wi × RGB` 或 `uv × wo × wi × RGB` 冻结采样后排列成的高阶 response tensor。关键是 encoder 看到的是被压缩目标值，而不是生成这些值的原生材质定义。

训练时联合优化 `E_psi` 与 `D_phi`；压缩完成后把 `Z` 烘焙成资产，运行时只保留 `Z + D_phi`，不执行也不保存 encoder。decoder 不必一次重建完整 `X`，仍可只读取 `q` 附近固定数量的 latent feature，因此 encoder 不破坏随机访问合同。这里还有两种不同训练范围：

- **per-asset target encoder**：`E_psi` 与 decoder 针对一个 `X` 联合优化，encoder 是生成 latent 的结构化参数化，压缩结束即丢弃；
- **corpus-shared target encoder**：同一个 `E_psi` 在多个目标 tensor 上训练，可对未见但同形状/同语义的 `X` 摊销 latent inference。

用户描述和 Qualcomm 的 asymmetric texture codec 首先对应前一种；不能仅因存在 encoder 就自动宣称跨资产 amortized compression。

这默认是**确定性 autoencoder，不是 VAE**。固定 encoder 权重、预处理和推理模式后，同一个 `X` 映射到同一个 `Z`。这不等于从不同随机 seed 重新训练 encoder 后也必然得到逐 bit 相同的 `Z`；训练本身仍需确定性设置和实测 seed 方差。只有让 encoder 输出 latent 分布参数，并使用随机采样/重参数化、KL 等分布约束时才是 VAE；当前压缩目标并不需要这些机制。

相对把 `Z` 注册成自由参数并从随机值开始反向优化，target encoder 有三个待验证优势：

1. `Z` 由完整目标内容确定性地产生，不依赖每个 texel latent 的独立随机初始化；
2. encoder 的共享卷积/张量结构把邻域、跨通道和跨尺度相关性变成优化先验，可能更快到达高质量 latent；
3. 相同输入的 latent 初始化可复现，能减少稀疏 query 梯度直接更新自由 latent 时的噪声和 seed 敏感性。

这里的“减少反向梯度更新噪声”不是说训练不再反向传播：encoder 和 decoder 仍由 backprop 训练。减少的是把大量局部 latent 当作彼此独立自由参数、再由稀疏随机 query 直接更新时产生的噪声；encoder 改为通过共享函数 `E_psi(X)` 前向生成这些 latent，使更新受到结构和权重共享的约束。

这些是优化机制和项目经验支持的假设，不是“加 encoder 必然提高最终率失真”的定律。实验必须在相同 decoder、latent bytes、训练时间或收敛准则下比较。ECCV 2024 的 Neural Graphics Texture Compression 给出了明确的非对称 autoencoder 实例：`E(X)` 读取多通道 texture tensor，生成 bottleneck/grid，运行时的小 decoder 仍支持按坐标和 mip 随机访问。[论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05476.pdf)

### 5.2 Autodecoder

autodecoder 不计算 `E(X)`，而是把每个资产的 latent `Z` 直接注册为待优化参数：

```text
Z_star = argmin_Z sum_q loss(D_phi(Z,q), X(q))
```

Random-Access Neural Compression of Material Textures 的原始 feature grids 属于这类 per-asset 直接优化。它仍有不可替代的用途：提供 decoder/latent 的直接拟合上界；处理不规则、缺失或无法方便整理成固定 tensor 的观测；以及检查 target encoder 的 parameterization/inference gap。只有 corpus-shared encoder 才进一步存在通常意义上的 amortization gap。但 autodecoder 的收敛速度、噪声和最终质量会受 latent 初始化、query sampling 与优化器影响，因此“optimized latent”只有在预算、多次 seed 和收敛诊断充分时才是可信上界。

### 5.3 Target encoder initialization + optional latent refinement

压缩路径可以先执行 `Z0=E(X)`，再固定或联合微调 decoder，对 `Z0` 做有界 refinement。这兼顾确定性初始化和逐资产最终质量。报告必须分开列出 encoder-only 与 refinement 后结果，并计入 encoder 压缩时间、refinement 时间、优化状态和随机性；encoder 仍不会进入 runtime。

### 5.4 从原生材质输入生成 latent 的 source compiler

source compiler 解决的是另一个问题：

```text
Z = C(M, theta)
```

它读取源材质的原生参数、图或纹理资源，目标是在**没有先生成完整 reference tensor `X`**的情况下，为未见材质状态或编辑结果直接产生 latent。它才支持“即时编辑后前向编译”的结论。target encoder 即使能完美压缩 `X`，只要输入仍是完整 `X`，就不能据此宣称 source compiler 泛化，因为生成 `X` 本身可能正是最昂贵的 reference 求值步骤。

Real-Time Neural Appearance Models 中的 encoder 更接近 source-parameter encoder：它把每个位置的高维 layered-material 参数映射成 latent，随后全图烘焙并丢弃 encoder，再可选直接优化 latent。它证明“source-parameter encoder → bake → refinement”有效，但论文实践中通常仍为单个材质训练自己的 encoder；因此它与“完整目标 tensor `X` → target encoder”输入合同不同，也不单独证明跨资产通用 source compiler。[论文](https://research.nvidia.com/labs/rtr/neural_appearance_models/assets/nvidia_neural_materials_author_paper.pdf)

## 6. 低对比度与长尾不是一个单一问题

压缩结果“对比度偏低”可能来自四个不同来源：

1. L2/MSE 在多解或低容量下趋向条件均值；
2. 固定容量不足以表示尖峰和稀有模式；
3. 训练样本被大量低值区域主导，峰值很少获得梯度；
4. latent 插值、凸混合或过强量化本身产生平滑。

确定性的分布调整能改善第 3 项和部分优化病态，但不会凭空增加表示容量。当前应把 target transform 当成正式设计轴：

| 数据语义 | 候选确定性变换 | 逆变换/注意事项 |
|---|---|---|
| 非负 `f` 或 `response_cos` | `q = log1p(y / s)` | `y = s * expm1(q)`；`s` 只由 train split 统计 |
| 带正负号的 analytic residual | `q = asinh(r / s)` | `r = s * sinh(q)`；不能对负 residual 使用普通 log |
| 跨通道长尾 | 对变换后的值用预计算 `mean/std` 标准化 | 统计量按 reference family/通道版本化，并计入 runtime 常量 |
| 极窄峰与总能量并存 | 分解为积分能量 `E` 与归一化方向 shape | 分别预测 `log E` 和 shape，重建后检查能量与峰位 |

建议的首轮默认候选是：对训练监督 `y=response_cos` 使用 train-only scale 的 `log1p`，再按通道标准化；若使用 analytic residual，则改为 `asinh`。同时保留 raw-linear baseline，避免变换改善了 log 误差却损害线性能量。

必须遵守四条边界：

- validation/test 不参与任何均值、方差、分位数或 codebook 的估计；
- 每材质自适应 scale 若运行时需要，必须随资产保存并计入 bytes；
- evaluator 公共语义仍返回线性 `f`，target transform 只是训练/内部输出参数化；
- 同时报告线性域、变换域、峰值和能量指标，不能只选择对变换有利的 loss。

Neural BRDF Representation and Importance Sampling 使用 cosine-weighted reflectance 的 log loss 来保护高光；Real-Time Neural Appearance Models 使用 log-space L1，并在训练早期对极窄方向峰做逐步减弱的 mollification；RTX NTC SDK 对 HDR 通道提供 HLG 变换。这些都说明动态范围处理属于 codec 设计，不是无关的数据清洗。[NBRDF](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14335) [Neural Appearance](https://research.nvidia.com/labs/rtr/neural_appearance_models/) [RTX NTC 设置](https://github.com/NVIDIA-RTX/RTXNTC/blob/main/docs/SettingsAndQuality.md)

## 7. 当前可证伪问题

研究按以下问题推进，每一项都可以被实验否定：

1. **单材质容量：** 在固定 runtime 预算下，小型 evaluator 能否覆盖一个材质的完整 `wo×wi`，而不是只拟合一个 view slice？
2. **共享表示：** 一个 shared decoder 加有限材质 latent，能否覆盖 LayerStack 中未见结构/状态，并在 MERL/OpenPBR 上保持相同趋势？
3. **压缩期 latent inference：** target encoder 能否比 autodecoder 更快、更稳定地产生同等或更好的 latent，并避免直接梯度更新造成的噪声？
4. **编译泛化：** 不读取完整 reference tensor 的 source compiler 与 optimized/target-encoded latent 的差距是否足够小，且参数编辑后无需重新训练 shared decoder？
5. **长尾保持：** 坐标规范化、target transform、能量/shape 分解或 analytic residual，哪一种真正恢复峰值和对比度？
6. **字典假设：** top-k latent dictionary 是否比 dense latent、plane/tensor factorization 在相同 bytes/time 下收敛更快或误差更低？
7. **空间随机访问：** 进入 spatial 数据后，局部 latent fetch 能否支持 footprint/LOD，且不依赖整图解码？

在前五项没有形成稳定 evaluator 和 latent 获取路径前，不把多灯 scaling、path-tracing 方差、环境积分或 UE 接入设为当前 kill test。

## 8. 当前决策

当前方法研究应同时保留少量、差异明确的候选，而不是提前押注单一网络：

```text
dense latent + small MLP                         基础下界
target-tensor encoder + shared decoder           压缩效率主线
target encoder initialization + refinement       高质量离线压缩候选
source-state compiler + shared decoder            未见状态与编辑主线
source compiler + bounded refinement              实用 cook 候选
sparse latent dictionary / top-k mixture         用户经验驱动候选
analytic core + neural residual                  长尾与采样友好候选
plane/tensor factorization                       高维分解对照
```

当前 E1 证据没有改变这八类候选清单，但已经缩小三个适用范围：在 `alpha_x=0.002` 极窄单界面与冻结小 MLP 成本内，direct dense 因无法保持峰值能量而淘汰；在固定多界面 LayerStack 上，analytic core + neural residual 使用 energy/shape、multiscale half-slope、GELU 与 cosine 后通过数值容量 gate；raw-direction 六成对 plane v1 在 32² 时对未见 query 过拟合、16² 时欠拟合，当前淘汰。通过项仍是 optimized-latent 单材质上界，不能替代 E2 的 shared decoder、E3 的 source compiler 或 E4 的部署/视觉验证；淘汰的 plane v1 也不否定带物理 warp 或空间语义轴的新 factorization。逐项数值与 hash 见 `artifacts/research/learning-goal/e1/comparisons/`。

下一步的权威数据与实验设计见 [`data_and_experiments.md`](data_and_experiments.md)，各相关工作能提供的具体机制见 [`prior_art.md`](prior_art.md)。
