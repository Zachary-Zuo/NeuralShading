# Fidelity-first neural evaluator 与 compiler 设计

## 1. 文档状态与决策边界

本文是下一阶段模型工作的评审稿。它把当前最小实现暴露的问题转成可执行的模型设计，但在评审通过前不注册正式 contract/config、不实现代码、不启动训练，也不修改已经冻结的 E0 数据和已有实验产物。

本文只确定四件事：

1. 什么结果才有资格称为单材质或共享表示的“高保真可达性证据”；
2. 先实现哪三类 evaluator，分别解决方向高频、跨材质共享和结构化分解问题；
3. target response encoder 与 source compiler 应该怎样接入同一个 evaluator，而不混淆输入语义；
4. 何时重新加入资产 bytes、共享权重、`prepare/evaluate` MAC、Slang 和 GPU 时间约束。

本文不改变最终运行时合同。目标仍是：

```text
prepared = prepare(material_program, surface_state, wo, footprint)
response = evaluate(prepared, wi)
```

E6 冻结 evaluator 后再增加与其匹配的 `sample/pdf`。本文中的高容量 teacher、target encoder 和 source compiler 可以只存在于训练或 cook 阶段；只有保留下来的 student evaluator 才必须最终进入 Slang/MethodBundle。

## 2. 为什么现有结果不能继续称为上界

现有 E1–E3 完成了公共 reader、split、transform、checkpoint、独立 test、metrics 和 source adapter 的生命周期验证，这些工程证据继续有效；需要修正的是模型结论的适用范围。

### 2.1 E1 的适用范围

E1 的 direct evaluator 在极窄 `alpha_x=0.002` 高光上失败，说明冻结的约 65k-MAC 小 MLP、既有方向特征与 loss 组合不能保持峰值；它不证明更强的 neural directional representation 也不能拟合该函数。多界面 analytic-core residual 通过已有 gate，说明 direct-top core 加小 residual 网络在该状态上是一个有效的成本受限候选；由于最难的移动峰可能已经由 core 承担，它不能单独证明 direct neural evaluator 的完整容量。

因此，现有 E1 结果改称：

- `direct small MLP`：效率受限 baseline；
- `analytic core + small residual MLP`：成本受限单材质候选；
- `raw-direction pairwise plane v1`：只淘汰该坐标与分解方式。

### 2.2 E2 的适用范围

E2 主要固定为 16 维 latent、第一层 concat conditioning、width 108 小 MLP 和同一种 residual parameterization。width 123 与 latent 32 的局部对照没有改善长尾，只能说明在相同网络族和成本边界内继续小幅扩容价值低，不能说明：

- layer-wise modulation 或 partial-weight generation 无效；
- learned shading frame、multi-lobe expert 或物理 warp 无效；
- 64/128 维 latent 与高容量 decoder 无法形成共享表示；
- target response 的空间/方向结构只能由 DeepSets mean/max 编码。

因此，`dense16/width108`、DeepSets encoder、bounded refinement、hard top-k 和 rank-4 factor 均保留为“特定实现的 target-visible compression baseline”，不再称为共享表示上界。只有在更强 decoder 已达到 fidelity gate、optimized latent 多 seed 收敛且容量增加进入平台后，optimized latent 才能作为该 decoder 的 compression upper bound。

### 2.3 E3 的适用范围

当前 LayerStack source compiler 使用一个约一万参数的 token MLP + GRU，从零联合训练 decoder，并同时预测 latent 和九个逐状态 residual transform 常量。它的 smoke 证明 family adapter、native-token 输入、source-held-out partition 和独立 evaluation 路径能够工作；初步 validation 过拟合只说明这套最小 compiler/training design 不稳定。

在正式 gate、诊断和新设计评审前，不从这个 run 推导 source compiler 路线的质量结论。新的 compiler 不再负责外推任意 normalization statistics，而是预测有稳定语义的 evaluator code；transform 由 train-corpus 统计或显式能量/颜色 head 处理。

## 3. Fidelity-first 研究原则

### 3.1 先证明函数可达，再压缩

第一阶段取消 `B_asset`、`B_shared`、`C_prepare` 和 `C_eval` 的 kill threshold，但仍记录这些数值。模型先回答“固定监督与 reference noise 下，完整 `wo×wi` 是否能被连续表示忠实重建”。通过后才依次加入：

1. shared decoder 与 material code；
2. target encoder/source compiler inference gap；
3. latent、expert、rank 和网络宽深压缩；
4. Slang parity 与实际 GPU Pareto。

不能用“不计效率”规避最终部署目标。高容量模型的职责是建立 teacher、定位监督问题和给压缩提供质量参照，不自动成为 E4 候选。

### 3.2 先对齐移动结构，再增加通用容量

镜面峰会随 `wo`、粗糙度、各向异性、折射率和层结构移动。在 raw `wo/wi` 坐标上，这表现为一个高速移动的窄脊；盲目加宽 MLP 仍可能先学习低频平均。首选设计顺序是：

1. reflection/transmission half-difference chart；
2. Rusinkiewicz spherical-harmonic encoding；
3. learned shading frame 或可逆物理 warp；
4. multi-lobe/expert 或 tensor factorization；
5. 最后才是更大的通用 residual backbone。

这里的 chart、frame 和 lobe 是 evaluator 内部结构，不是要求所有 source family 暴露统一 analytic closure。

### 3.3 direct response 必须有独立高保真证据

analytic core + neural residual 是正式候选，但不能成为唯一 oracle。否则 core 已经解释的高频会掩盖 neural representation 自身的不足。每个代表状态至少保留：

- direct response oracle；
- 若 family 存在可信 analytic core，再增加 signed residual 对照；
- core-only 指标，量化 neural 部分的实际贡献。

### 3.4 test、validation 与对抗性 probe 继续分离

训练和 hard-query mining 只读取 source train × query train。validation 只选架构/config/checkpoint；held-out test 只在设计冻结后读取一次；adversarial probe 与 response-slice/capture 独立报告。任何 target transform、chart density、codebook、peak token 或 hard-example distribution 都不能从 validation/test 拟合。

## 4. 公共符号和张量合同

设：

- `m`：一个 source material state；
- `wo ∈ S²`：局部 frame 中的出射/观察方向；
- `wi ∈ S²`：局部 frame 中的入射/光照方向；
- `y(m, wo, wi) ∈ R³`：HDF5 保存的 RGB `response_cos`；
- `z_m ∈ R^D`：可部署 material code；
- `p_m(wo)`：`prepare()` 可被多个 `wi` 复用的 view-conditioned state。

所有 evaluator 继续满足：

```text
p_m(wo) = prepare(z_m, wo)
y_hat = evaluate(p_m(wo), wi)
```

训练实现可以一次处理：

```text
wo : [G, 3]
wi : [G, N, 3]
y  : [G, N, 3]
```

但公共模型不得依赖固定 `N`，也不得把一个完整方向表藏进 `prepare()`。单次随机 `wi` 查询必须成立。

reflection chart 在 `wo + wi` 有效时使用反射 half vector；transmission chart 使用 source 语义允许时的 generalized half vector。折射率只由原生 source 确实提供的 family adapter 使用；MERL 等不提供该参数的 family 采用 learned transmission frame/chart，不伪造 LayerStack 参数。

chart 计算必须返回显式 validity/mode flags，避免在 half-vector 退化、临界区域或反射—透射边界产生 NaN。所有归一化使用版本化 epsilon，并进入 Python/Slang parity 合同。

## 5. 模型 A：高保真双方向 oracle

计划 ID：`ncls.high-fidelity-warped-directional-oracle@1`。评审通过前它只是设计名称，不是已注册 contract。

### 5.1 它回答什么

模型 A 针对一个 material state 拟合完整 `wo×wi`。它没有 material latent 瓶颈，也没有部署成本上限，回答三个问题：

1. 当前 train supervision 是否足以重建 held-out continuous query；
2. 方向 chart/encoding 能否稳定表达极窄峰、掠射和 transmission；
3. direct response 与 analytic residual 的真实难度差距是多少。

模型 A 不是最终 shared runtime，也不能证明 source compiler 泛化。

### 5.2 输入编码

默认输入由四组特征组成：

1. **raw local features**：`wo`、`wi`、两方向的 cosine、hemisphere/mode/validity flags；
2. **reflection chart**：Rusinkiewicz `h_r/d_r`、half-vector slope 和 grazing-safe log-slope；
3. **transmission chart**：valid generalized `h_t/d_t`，或由 learned frame 形成的无 source-parameter chart；
4. **spectral features**：对 `h` 使用最高 9 阶 real spherical harmonics，对 `d` 初始使用最高 5 阶；阶数分别配置，不能写死在 reader 中。

高阶主要分配给 half-vector 轴，因为窄峰在 canonical reflection chart 中主要沿该轴变化。raw features 始终保留，防止 chart 在退化或非典型多散射区域丢失信息。

### 5.3 learned frame

每个单材质 oracle 初始允许 `J=4` 个 trainable shading frame。frame 使用连续 6D rotation representation，经 normalize + Gram-Schmidt 生成正交 normal/tangent/bitangent，禁止直接回归未约束 `3×3` 矩阵。

frame 本身是 material-static；`prepare(wo)` 只预测各 frame 的 view-conditioned 权重和 feature，不让 frame 随每个 `wi` 变化。这样同一个着色点的多个 evaluate query 可复用 frame，并避免网络通过任意 query-dependent rotation 记忆响应。

### 5.4 网络结构

初始容量 envelope：

| 模块 | 初始结构 | 输出 |
|---|---|---|
| frame/chart encoder | 每个 frame 共用 `2×256` residual block | canonical direction feature |
| `prepare` trunk | `3×256` residual block，GELU | `p(wo)∈R²⁵⁶` 与 frame weights |
| `evaluate` trunk | `6×256` pre-norm residual block，GELU | 每 query hidden feature |
| direct head | `256→128→3` | transformed RGB response |
| energy auxiliary head | `p(wo)→RGB × mode` | hemispherical reflected/transmitted energy |

每个 residual block 使用 linear → GELU → linear 和 identity skip；LayerNorm 只作用于 hidden feature，不跨 query 或材质估计统计。初始约百万级参数是刻意的：这一阶段要先排除小网络 spectral bias 和容量不足。

SIREN/periodic activation 只作为同输入、同参数量的单独 ablation，不作为默认模型。它可能提高高频拟合，也可能在稀疏区域产生振铃；没有 held-out peak 与积分证据前不混入主模型。

### 5.5 输出参数化

对 source working color space 中保证非负的 response，direct 版本预测：

```text
t_c = log1p(y_c / s_c)
y_hat_c = s_c * expm1(t_hat_c)
```

`s_c` 只由该 state 的 train query 拟合，作为 target-visible oracle 常量记录。为了确认这种自由度不会掩盖未来 compiler 问题，direct oracle 通过后必须增加 family-global/corpus-global `s_c` 对照；只有后者能直接迁移到 pure source compiler。

若 source/color-space contract 允许合法负通道，例如线性颜色空间转换产生的 out-of-gamut 值，则 direct 版本改用 signed `asinh`，或者优先在 source 的权威非负 working space 中学习后再执行固定颜色变换。不能对合法负值使用 `log1p`、softplus 或统一 clamp。

analytic residual 版本使用：

```text
r = y - y_core
t_c = asinh(r_c / s_c)
r_hat_c = s_c * sinh(t_hat_c)
y_hat = clamp_source_range(y_core + r_hat)
```

不能对 signed residual 使用 log，也不能把 source 合法的负色域通道统一 clamp 为零。范围约束由 family/color-space contract 决定。

### 5.6 loss

默认总 loss 是多个有清楚职责的项，而不是只优化总体 MSE：

```text
L = λ_transform L_robust_transform
  + λ_linear    L_solid_angle_linear
  + λ_energy    L_integrated_energy
  + λ_peak      L_top_energy
  + λ_recip     L_source_reciprocity
```

- `L_robust_transform`：Huber/Charbonnier transformed-domain error，保留暗部和长尾梯度；
- `L_solid_angle_linear`：按实际 proposal/solid-angle 权重计算的 linear response error；
- `L_integrated_energy`：每 `wo×mode×channel` 的积分能量误差；
- `L_top_energy`：只在 train query 的 top-energy support 上加权，防止窄峰被低值区域淹没；
- `L_source_reciprocity`：拟合相对 source reference 的 reciprocity deviation，不对已知 source 非互易项施加错误的绝对零约束。

各权重必须在读取 test 前通过 train/validation 因果对照确定并版本化。不能因为某个 metric 失败就做无边界 weight sweep。

### 5.7 训练与采样

每次训练分三段：

1. **coverage warm-up**：使用冻结的 uniform/cosine/microfacet-aware train mixture；
2. **peak curriculum**：逐步提高 train-only high-energy、掠射和 transmission-critical query 的 batch 比例；
3. **error refinement**：周期性在额外 train-only proposal 上求值，用模型误差与 reference SE 选择 hard query，并保存 proposal/hash。

hard-query mining 只能重采 train domain。所有 loss 仍使用 importance/solid-angle correction，不能把 oversampling 后的经验分布当成均匀半球指标。

每个正式设计至少运行三个 seed。只有 validation 同时进入平台、train/validation gap 有界、增加容量或训练预算不再产生实质改善，才读取 held-out test。单次最佳 seed 不能作为可达性证据。

### 5.8 最小因果对照

模型 A 不做无差别 sweep，只运行下列有明确问题的递增对照：

| 对照 | 唯一变化 | 回答的问题 |
|---|---|---|
| A0 | high-capacity raw/Rusinkiewicz ResNet，无 learned frame | 单纯解除小网络容量是否足够 |
| A1 | A0 + learned shading frames | 移动峰是否主要是坐标对齐问题 |
| A2 | A1 + reflection/transmission 分头和 energy head | 多 mode/能量耦合是否限制峰值 |
| A3 | direct 与 analytic residual 配对 | core 贡献与 neural 完整容量差距 |

代表状态至少覆盖：极低 roughness、固定多界面多散射、反射/透射临界区域、MERL 高光材质。前三项先用已有合格 H5；只有 fixed high-resolution probe 证明 coverage 不足时才针对性重生成数据。

## 6. 模型 B：canonical-lobe hyperdecoder

计划 ID：`ncls.canonical-lobe-hyperdecoder@1`。

### 6.1 它回答什么

模型 B 是主要 shared evaluator 候选。它把“材质差异”从第一层 concat latent 改成两种结构化条件：

1. `prepare()` 生成有限个 canonical lobe token；
2. material code 对每个 residual block 做 FiLM 和低秩 weight modulation。

它首先以 optimized latent 形态证明 shared representation fidelity；target encoder 与 source compiler 之后只负责产生同一 code，不更换 decoder。

### 6.2 material code 与调制

高保真起点使用 `z_m∈R¹²⁸`。共享 hyper trunk 从 `z_m` 生成：

- 每个 block 的 FiLM `scale/bias`；
- rank-8 low-rank modulation coefficient；
- material-static lobe descriptors；
- family-agnostic global color/energy context。

第 `l` 层的共享权重保持为 `W_l`，asset-specific 变化为：

```text
h_{l+1} = activation(
    gamma_l(z) * ((W_l + U_l diag(a_l(z)) V_l) h_l) + beta_l(z)
)
```

`U_l/V_l` 是共享参数，asset 只保存 `z_m`，不保存完整专属权重。这样比第一层 concat 更能改变函数族，同时保留未来 shared-code、coherent tile 和 `B_shared/B_asset` 分离的可能。

### 6.3 lobe token

初始使用 `K=8`、每 token 64 维。每个 token 通过受约束 head 表达：

- 6D shading-frame representation；
- reflection/transmission/low-frequency 类型 logits；
- 各向异性 bandwidth 与 roughness-like shape hints；
- RGB amplitude/energy hints；
- shared expert 的 FiLM/context feature。

这些字段是 neural evaluator 的内部坐标和条件，不是公共 analytic closure，也不要求 source adapter提供 lobe 参数。compiler 可以只输出 `z_m`，lobe token 由共享 `prepare` 网络产生。

### 6.4 `prepare/evaluate` 划分

```text
static_context = hyper(z_m)
p_m(wo) = prepare_trunk(static_context, wo)

for each lobe k:
    q_k = canonicalize(wo, wi, frame_k, type_k)
    v_k = shared_expert(q_k, p_m(wo), token_k)

y_hat = combine(v_1 ... v_K, low_frequency_head)
```

frame、token、material modulation 和 `wo` feature 在 `prepare()` 计算一次；`evaluate()` 只添加 `wi`、chart 与共享 expert。实现必须分别报告一次 query 和多个 query 的 amortization，不能把全部 K-lobe prepare 成本重复计入每个 `wi`。

### 6.5 mixture 与输出

direct 版本由三部分组成：

1. 每个 canonical lobe expert 输出非负 RGB contribution；
2. low-frequency head 表达 diffuse-like/multiple-scattering background；
3. 小型 signed correction head 修正 lobe 组合不能表达的局部差异。

gate 不是只对 latent 做凸平均。每个 expert 可以产生独立 contribution，组合后再按 source range 处理，避免 top-k convex mixture 把多个峰平均成宽峰。

analytic residual 版本保持 core 外置，并让 lobe experts 预测 signed residual。两版本共享方向 chart、prepare 和调制机制，以隔离 core 的影响。

### 6.6 初始容量与后续压缩

高保真起点：

| 组件 | 起始值 |
|---|---:|
| material code | 128 floats |
| lobe count | 8 |
| prepare width/depth | 256 / 4 residual blocks |
| shared expert width/depth | 256 / 6 residual blocks |
| low-rank modulation | rank 8 |
| direction encoding | 与模型 A 相同的 chart/RSH front end |

这组值不是部署承诺。只有 optimized-code 版本通过 fidelity gate 后，才沿一条有序 rate-distortion 路线压缩：

```text
K:       8 → 4 → 2
latent: 128 → 64 → 32 → 16
width: 256 → 192 → 128 → deployment envelope
rank:    8 → 4 → 2
dtype: fp32 → fp16/quantized
```

每次只改变一类瓶颈，并与 teacher distillation 对照。最小版本失败不能反向否定高保真结构。

### 6.7 与 sparse dictionary 的关系

dictionary 不再从随机 hard top-k logits 起步。先收集通过 fidelity gate 的 optimized `z_m`，仅使用 source-train states 做 K-means++/residual VQ 初始化，再比较：

- top-1 codeword；
- top-2 非凸/凸混合；
- top-k soft mixture；
- codeword + continuous residual。

codebook、whitening 和聚类统计不读取 validation/test。dictionary 只替换 `z_m` 的资产表示，decoder、query split 和 metrics 保持不变。

## 7. 模型 C：物理 warp 后的 tensor/plane field

计划 ID：`ncls.warped-half-difference-vm-field@1`。

### 7.1 它回答什么

模型 C 是 plane/tensor factorization 的重新设计。旧 v1 在 raw Cartesian `wo.x/wo.y/wi.x/wi.y` 的六个成对 plane 上分解，镜面峰仍会跨 plane 快速移动。新模型先把响应映射到 reflection/transmission canonical chart，再利用低秩与多尺度结构。

### 7.2 坐标和分支

- reflection branch：`q_r=(u_h,v_h,u_d,v_d)`，其中 half-vector slope 作为高分辨率二维轴；
- transmission branch：`q_t=(u_ht,v_ht,u_dt,v_dt)`，由 valid generalized half-vector 或 learned chart 构造；
- boundary/low-frequency branch：处理 chart invalid、临界混合和宽多散射背景。

每个 branch 返回 validity 与 Jacobian/weight metadata。warp 只改变表示坐标，不改变最终 solid-angle metric。

### 7.3 vector-matrix decomposition

每个尺度 `l` 使用双向 vector-matrix 形式：

```text
F_l(q) = sum_r [
    H²_l,r(u_h, v_h) * Dᵘ_l,r(u_d) * Dᵛ_l,r(v_d)
  + D²_l,r(u_d, v_d) * Hᵘ_l,r(u_h) * Hᵛ_l,r(v_h)
]
```

第一项让高分辨率 half plane 直接承载窄峰；第二项补充 difference-direction 的二维相关。各尺度 feature concat 后交给小 decoder 输出 direct response 或 signed residual。

初始 fidelity envelope 使用分辨率 `32/64/128/256` 和 rank `32`；如果最窄峰仍欠拟合，只在 half plane 增加分辨率/rank。不能同时扩大所有 plane 后把改善原因混在一起。

### 7.4 插值、边界和随机访问

- 2D plane 使用 bilinear fetch，1D factor 使用 linear fetch；
- chart seam 使用周期坐标或成对 seam feature，不能依赖 clamp 隐藏不连续；
- reflection/transmission branch 在物理 mode/validity 上显式路由；
- 单 query 只读取固定数量的邻域 texel/factor，保持随机访问；
- 资产 bytes、fetch 数和 decoder MAC 从第一轮起记录，但在 fidelity 阶段不淘汰。

### 7.5 与模型 B 的比较

模型 C 不是模型 B 的所有组合项。它只在模型 A 已证明正确 chart 后运行，并与模型 B 的 optimized-code 版本比较：

- 模型 B 更擅长共享和 compiler；
- 模型 C 更直接保存高频 canonical field；
- 若 C 显著更准但资产大，它可作为 distillation teacher 或高质量 Pareto 端点；
- 若 warp 后仍需极高 rank，说明多峰/状态相关性不能由该 tensor 结构充分解释。

## 8. Target response encoder

计划 ID：`ncls.response-chart-attention-encoder@1`。

### 8.1 为什么替换简单 DeepSets

现有 DeepSets 对 `[wo,wi,response,residual]` point tensor 做逐点 MLP 后 mean/max pooling。它能验证 permutation-invariant 生命周期，却会压掉：

- 峰之间的相对位置；
- 同一 `wo` 下完整 `wi` shape；
- reflection/transmission chart 的局部邻域；
- response 与 reference SE/solid-angle 权重的关系。

新的 encoder 仍接受无固定顺序的 train query，但通过 attention 显式建模点之间的关系。

### 8.2 输入 token

每个 train-only query token 包含：

```text
raw wo/wi
reflection/transmission canonical coordinates + validity
transformed RGB response or signed residual
reference standard error
solid-angle / proposal weight
energy-bin and train-only peak-support flags
```

peak-support flag 与任何 bin threshold 只由该 state 的 query-train 数据计算。validation/test point 不进入 encoder input，也不参与 token normalization。

### 8.3 网络结构

初始设计：

- point projection width 256；
- 64 个 learned inducing tokens；
- 4 个 induced self-attention blocks，8 heads；
- reflection、transmission 和 low-frequency 三组 learned pooling queries；
- 最终输出 128 维 `z_m`，以及仅用于诊断的 encoder confidence。

inducing attention 使计算随输入点数近似线性增长；训练时可分块累积 response tokens，但分块规则和归约顺序必须进入 fitted-state hash。

### 8.4 训练阶段

1. 固定通过 fidelity gate 的模型 B decoder；
2. 使用 optimized `z_m*` 做弱 latent-alignment loss；
3. 以完整 query-space reconstruction/distillation 为主 loss，避免 latent 对称性使直接 L2 误导；
4. 得到 deterministic `z_0=E(X_train)`；
5. 冻结 encoder/decoder，只对 `z_0` 做版本化 bounded refinement。

报告必须分开列出 encoder-only、refinement 后结果、compile time、refinement time、输入 bytes 和 seed 方差。target encoder 读取 reference response，因此永远不等同于 source compiler。

## 9. Typed source graph compiler

计划 ID：`ncls.typed-source-graph-compiler@1`。

### 9.1 family adapter 输出

公共 compiler runner 只接收 family adapter 的版本化 typed token/graph：

```text
node type
native continuous/discrete parameters
resource embedding references
ordered position / depth
typed edges and graph topology
source family and contract version
```

LayerStack adapter 保留 interface/medium 顺序和相邻关系；MaterialX/OpenPBR adapter 保留原生图拓扑与资源语义；MERL 没有可编辑 source parameters 时不伪造 LayerStack graph，而是明确属于 measured-table/target-visible 路径或单独的 measurement compiler。

### 9.2 初始 compiler 结构

LayerStack 第一版使用：

- type-specific parameter projection；
- width 256、6 层、8-head order-aware graph Transformer；
- relative depth、相邻关系和 top/bottom boundary edge encoding；
- attention pooling 得到 graph code；
- MLP 输出模型 B 的 128 维 `z_m` 与 compile-confidence diagnostic。

这是一条 offline compiler 路径，首轮不受 shader cost 限制；它不会进入 runtime bundle。不同 family 可以有不同前端/graph encoder，但必须输出同一个 evaluator code contract，公共 runner 不增加 family 分支。

### 9.3 不再预测任意 transform 常量

compiler 不输出每 state 的 residual scale/mean/std。允许的替代只有：

1. source-train corpus 固定的 family-global transform；
2. evaluator 内显式 energy/color head；
3. source 原生语义确实包含并能直接计算的物理量。

这样 compiler generalization 测的是材质函数 code，而不是同时外推一个依赖 target response 统计的坐标系。

### 9.4 训练阶段

训练按依赖顺序进行：

1. **teacher construction**：模型 B optimized code 在 source-train states 上达到 fidelity gate；
2. **canonicalization**：使用共同初始化、code regularization 和 decoder 固定，降低 optimized latent 的排列/尺度不确定性；
3. **code distillation**：compiler 接近 target encoder/optimized code，只作初始化；
4. **functional distillation**：固定 decoder，在 source-train × query-train 上最小化 response/energy/peak loss，这是主要目标；
5. **joint fine-tune**：只在有明确 validation 改善时有限解冻 shared decoder，且必须保留 optimized-code control；
6. **held-out evaluation**：分别测试未见连续状态、未见资产/图拓扑、跨 family。

latent L2 不能成为唯一 compiler loss，因为多个 latent 可能表示同一函数。最终 gap 始终在同一 decoder 的 query-space metrics 中计算。

### 9.5 pure feed-forward 与 bounded refinement

- pure feed-forward：`z=C(source)` 后不读取 reference response；
- compiler initialization + bounded refinement：只用新 state 的 query-train response 调整 `z`，validation 选固定预算内 checkpoint，test 独立；
- optimized/target-encoded control：使用相同 decoder，量化 compiler gap；它只有在 decoder 本身通过 fidelity gate后才是有意义的 control。

三者使用不同 manifest role，不能把 bounded cook 结果写成 pure source generalization。

## 10. Fidelity gate 与“上界”用语

### 10.1 两阶段 gate

新的 acceptance 分成两个不可互相替代的阶段。

**Fidelity gate** 只检查：

- solid-angle normalized L1、linear/log error；
- family/state median、p90、p95；
- peak ratio、peak-support angle、top-energy recall；
- reflection/transmission 分项积分能量；
- source reciprocity deviation、finite/range legality；
- model error 相对 reference SE 或 deterministic reference absolute floor；
- high-resolution response slice 与共同曝光的局部视觉对比；
- 多 seed 与 capacity/optimization convergence。

**Deployment/Pareto gate** 在 fidelity 通过后才增加：

- `B_asset/B_shared`；
- compile/refinement time；
- `C_prepare/C_eval` 和单次/多次 amortization；
- fp32/fp16/quantized parity；
- coherent/divergent tile、真实 GPU 时间、显存和带宽；
- MethodBundle、Slang 和 viewer capture。

### 10.2 阈值来源

本文不凭主观重新填写数值阈值。评审通过后，先从 E0 的 A/B replica、deterministic reference、现有 analytic control 和 fixed probes 生成 `fidelity-oracle-gates-v1`：

```text
threshold(metric) = max(
    k_noise * reference_replica_disagreement(metric),
    absolute_visual_tolerance(metric)
)
```

`k_noise`、absolute floor 和适用 family 在训练前冻结。reference SE 接近零的 deterministic case 不使用无穷敏感的 ratio 作为唯一 gate；Monte Carlo case 同时报告 absolute error 和 error/SE。

### 10.3 何时可以称为上界

一个 run 只有同时满足以下条件，才能在限定模型族内称为 upper bound：

1. 输入监督和 target transform 没有读取 validation/test；
2. 至少三个 seed 的 validation/test 分布稳定；
3. 训练已收敛，继续增加 step 不再改善；
4. 提高 latent、width、expert 或 rank 后结果进入平台，而不是只试一个更大点；
5. 峰值、能量、透射和视觉 slice 没有关键失败；
6. 明确写出上界适用的 decoder、chart、数据和训练预算。

不满足这些条件时使用 `baseline`、`smoke`、`capacity candidate` 或 `fidelity oracle attempt`，不得用“上界”替代证据。

## 11. 八类候选怎样接入，不做笛卡尔积

| 原候选 | 新设计中的位置 | 首个有意义的比较 |
|---|---|---|
| dense latent + small MLP | 保留为效率受限 baseline；模型 B 压缩末端 | 与通过 fidelity 的模型 B student 比较 quality/time/bytes |
| target-tensor encoder + shared decoder | response-chart attention encoder → 模型 B | encoder-only 与同 decoder optimized code |
| target encoder initialization + refinement | 同上，加固定预算 latent refinement | initialization gap 与 refinement gain |
| source-state compiler + shared decoder | typed source graph compiler → 模型 B | pure feed-forward 未见状态 |
| source compiler + bounded refinement | 同上，加 query-train bounded cook | compiler gap、cook time 与最终 fidelity |
| sparse latent dictionary / top-k mixture | 对通过 fidelity 的 optimized `z` 做 train-only K-means/VQ | dense `z` 与 top-k/residual VQ 的 rate-distortion |
| analytic core + neural residual | 模型 A/B 的 paired output variant | direct 与 residual 的 core contribution/长尾差异 |
| plane/tensor factorization | 模型 C | canonical warp 后的 VM field 与模型 B |

实验不跨所有 candidate × level。模型 A 先确定 supervision 与 directional front end；模型 B 通过后才投入 encoder/compiler/dictionary；模型 C 只验证与 MLP 明确不同的结构假设。

## 12. 审批后的最小执行顺序

### Phase F0：重新分类，不重跑

- 保留现有 E1–E3 run、checkpoint、hash 和失败指标；
- 把 small MLP、DeepSets、GRU compiler 结论限制为对应实现；
- 新建 fidelity gate 配置，不修改旧 gate 和历史 manifest。

### Phase F1：模型 A 高保真可达性

1. 极窄 reflection state：A0 → A1；
2. 固定多界面 state：direct A1 与 residual A3；
3. transmission/critical state：A1 → A2；
4. 只有 multi-peak 证据需要时再增加更多 frame/head；
5. 三 seed、固定 test、response-slice 人工检查。

如果模型 A 的 train 也无法接近 noise floor，先诊断容量、transform 和 sampling；如果 train 通过而 held-out 失败，检查 chart、coverage 和连续泛化。只有 fixed probe 证明监督缺口才修改 H5。

### Phase F2：模型 B 共享表示

1. optimized `z128/K8/width256/rank8`；
2. 与每 state 模型 A teacher 做 functional distillation；
3. source-state 与 family 分层 gate；
4. 通过后才沿单轴 rate-distortion 路径压缩；
5. 最后接 response encoder、dictionary 和 source compiler。

### Phase F3：模型 C 结构化对照

1. 复用模型 A 已确认的 reflection/transmission chart；
2. half-plane resolution 因果实验；
3. rank 因果实验；
4. direct/residual 配对；
5. 与模型 B 在 fidelity、asset bytes 和 random-access fetch 上比较。

### Phase F4：重新进入 E4

只有 F2/F3 留下的 Pareto candidate 才进行 student distillation、Slang/MethodBundle、GPU timing 和 viewer gate。高容量模型 A 不要求导出。

## 13. 配置、产物与报告位置

评审通过后计划使用：

```text
configs/research/fidelity-oracle-gates-v1.json
configs/research/f1-*.json
configs/research/f2-*.json
configs/research/f3-*.json

artifacts/research/learning-goal/fidelity-first/
  audits/
  runs/
  comparisons/
  response-slices/
  captures/
```

正式 config 必须保存完整 pipeline contract、dataset selection、seed、direction chart、SH order、frame/lobe count、latent/modulation 结构、loss、proposal 和 gate hash。单次结果留在 `artifacts/`；稳定设计结论回写本文与 [`data_and_experiments.md`](data_and_experiments.md)。

不为新模型建立第二套 reader、runner、checkpoint 或 metric。它们必须通过 candidate-neutral registry/config 接入已有 lifecycle；模型专属逻辑留在 representation、model、family adapter 或 pipeline composition 中，不能继续堆进公共 runner。

## 14. 评审时需要确认的设计决策

本文给出以下推荐，评审后才进入实现：

1. direct response 模型 A 必须独立通过 fidelity gate，analytic residual 不能替代它；
2. 模型 B 从 `z=128、K=8、width=256、rank=8` 起步，先证明共享 fidelity；
3. target encoder 使用 chart-aware induced attention，不再扩展 mean/max DeepSets；
4. LayerStack source compiler 使用 order-aware graph Transformer，并停止预测逐状态 transform statistics；
5. 模型 C 在模型 A chart 成形后执行，不与模型 B 的所有 inference 路径做笛卡尔积；
6. fidelity gate 通过前只记录效率，不用旧的 65k-MAC/512-KiB 门槛淘汰新结构。

## 15. 主要依据与迁移边界

- [Neural BRDF Representation and Importance Sampling](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14335)：支持 Rusinkiewicz coordinates、log-domain loss 和 specular-aware angular sampling；不证明 shared compiler 或 spatial LOD。
- [Neural Layered BRDFs](https://wangningbei.github.io/2022/NLBRDF.html)：支持高容量 residual evaluator、optimized BRDF latent 和 layering-space 学习；其 isotropic/layered 假设不能提升为所有 family 的接口。
- [MetaLayer](https://sites.cs.ucsb.edu/~lingqi/publications/paper_siga23metalayer.pdf)：支持 Rusinkiewicz spherical-harmonic encoding 与从 source parameters 生成部分 evaluator weights；完整 per-material weights 仍需与 shared modulation 的 coherence/bytes 比较。
- [Real-Time Neural Appearance Models](https://research.nvidia.com/labs/rtr/neural_appearance_models/)：支持 learned shading frames、source encoder → bake → refinement、evaluator/sampler 分头和 spatial LOD；本文先借用方向与 conditioning 机制，不提前声称 E5/E6 已成立。
- [An Adaptive Parameterization for Efficient Material Acquisition and Rendering](https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Dupuy2018Adaptive.pdf)：支持先把高光域 warp 到更平滑坐标；其采集/表格表示不是本文 runtime 合同。
- [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)：支持用 induced attention 表达无序点集间关系；它不是现成的 BRDF encoder，chart token 与 split 约束仍需本项目验证。
- [SIREN](https://www.vincentsitzmann.com/siren/)：支持周期激活表示高频隐式函数；本文只把它作为受控 activation ablation，不预设它优于物理 chart。
