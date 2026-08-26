# 统一散射方法与 viewer 闭环设计

> 状态：方法结构已由用户确认；本文件冻结首个目标方法、公共数学边界和跨层数据流。任务仍处于 planning，尚未获准开始实现。

## 1. 系统边界

本任务交付一个完整 MethodBundle 方法，而不是新的 source reference family。首个实现只支持 LayerStack upper-hemisphere、reflection-only、non-delta 事件，但 renderer 侧只依赖公共 `INclsScatteringBackend` 与 capability，不解释 LayerStack 或方法私有 state。

```text
LayerStack MaterialProgram
  -> LayerStack reference response（离线 cook）
  -> CompiledMaterial + shared Slang weights
  -> prepare(context, compiled)
      -> private ScatteringState
          -> evaluate(wi)
          -> sample(u)
          -> pdf(wi)

同一 MethodBundle specialization
  -> viewer method deferred
  -> viewer method PT
  -> GPU contract/parity tests
```

训练 loss 和 optimizer 留在 Python/Torch；方法前向、参数解码、NVIDIA diffuse+GGX/LTC sampling/PDF 和 physical core 只有一份 Falcor-free Slang 实现，由 SlangPy、Falcor GPU 测试与 viewer 共同编译。

## 2. baseline、目标 evaluator 与 sampler 轴

### 2.0 必做 NVIDIA baseline

`nvidia-frame-two-lobe-v1` 是当前目标的结构 baseline，不只是 prior-art 引用：

- evaluator：从 latent 提取两个 learned shading frames，把方向变换到两个 frame，再由 direct MLP 输出正的 BRDF；
- sampler：MLP 输出 9 个参数，形成 tilted-cosine diffuse 与 non-centered anisotropic GGX specular 的解析混合；
- training：BRDF 使用 log-space L1，窄峰使用 directional mollification curriculum；sampler PDF 相对当前 learned BRDF 做 KL，且 sampler KL 不回传到 latent；
- runtime：sample 与 PDF 使用同一 9 参数 proposal。

原规模 `64×64×64` evaluator 与 `32×32×32→9` sampler 是唯一正式 NVIDIA baseline。当前锁定 Slang 2024.1.34 路径没有本项目可验证的 cooperative-vector 加速能力，但普通 Slang 标量循环仍能表达同一个网络：它必须导出、加载并在 viewer 的 deferred/PT 路径中显示和测量。预计超出当前软预算时如实标记 runtime class；分类只影响成本声明和排序，不允许拒绝加载或替换成缩模。所有“优于 baseline”结论只在 matched data/role/evaluation 下成立，架构 bytes/time 是被比较的结果，不要求预先相同。

目标 evaluator ID 为 `core-frame-neural-v1`：保留 baseline 的 learned-frame direct MLP 主体，增加 exact top core，并让 MLP 预测 positive residual。sampler 配置轴为 `nvidia-diffuse-ggx9` 与 `ltc-k2`。最终 bundle identity 由 evaluator/sampler 组合和内容 hash 决定。

旧 `lobe-residual-k2-v1` 不参与选择；它把两个 LTC lobe 当 evaluator 主要输出词汇。新目标的逐方向 MLP直接表达多层 residual，LTC 只作为一个 sampler 候选和显式 analytic control 的公共原语。

### 2.1 CompiledMaterial

均匀 LayerStack 材质离线 cook 后保存：

| 字段 | 部署形态 | 作用 |
|---|---:|---|
| top-interface core | 64 B | 当前 LayerStack 的精确直接顶层 reflection；backend 私有，不进入公共合同 |
| latent `z` | 16×FP16 = 32 B | 由 reference response direct fit 得到 |
| normalization / flags | ≤ 32 B | 颜色尺度、版本、支持域与布局校验 |
| 合计 | 128 B 对齐 | 小于均匀材质 512 B 硬线 |

源参数编辑后重新运行 cook；本任务不实现 feed-forward source compiler。

### 2.2 prepare

输入为 `z16` 与七维 `wo` 特征，网络采用 `23→64→64` shared trunk，再分为 evaluator 与 sampler projection。激活函数必须能在 Slang 2024.1.34 中定长、无 LayerNorm/erf-GELU 地实现；最终选择由 SlangPy/Falcor 双编译与 matched quality 结果冻结。

`prepare` 输出 evaluator payload 与所选 sampler payload；最大配置为 27 个 FP16 标量：

| 内容 | 数量 | 说明 |
|---|---:|---|
| evaluator state `h` | 8 | 3 个非负 residual scale raw + 5 个 view-conditioned code |
| learned evaluator frames | 6 | 两个 frame，各用 2 个有界 slope + 1 个 tangent rotation 参数化 |
| NVIDIA proposal | 9 | `{w_d, mu_d.xy, w_s, alpha.xy, rho, mu_s.xy}`；仅该 sampler 配置存在 |
| LTC proposal transforms | 12 | 两个 LTC，各 2 个正 inverse scale、3 个有界 shear、1 个 rotation；仅 LTC 配置存在 |
| LTC mixture logit | 1 | 两个 learned lobe 之间的权重；仅 LTC 配置存在 |

部署 state 统一为 64 B 上限：NVIDIA proposal payload 较小，LTC 配置为 `materialSlot:4 + flags:4 + payload:54 + padding:2 = 64 B`。随机数不进入 `prepare`；同一着色点的多个 `wi` 查询复用该 state。baseline 与目标不能因未使用 bytes 获得隐藏的容量差异，matched 报告同时给实际有效 bytes 和统一 stride。

prepare MAC 纸面上界：

```text
23×64 + 64×64 + 64×27 = 7,296 MAC
```

满足 `C_prepare ≤ 10,000`。

### 2.3 evaluate

适用范围内，LayerStack reference 是非相干能量传输；顶层直接 reflection 是完整响应中的非负路径子集。因此目标参数化为：

```text
f_hat(wo, wi) = f_top(wo, wi)
              + softplus(s(h)) * softplus(g_theta(x(h, wo, wi, f_top)))
```

- `f_top` 复用通用 microfacet/interface 数学原语，负责已知的极窄峰；
- `g_theta` 是必选逐方向 MLP，不允许 `correction=none`；
- 两个 `softplus` 保证 residual 非负且处处保留梯度，不使用 `clamp(core + signed residual, 0)`；
- loss 始终作用于最终 `f_hat·cos`，不把带 Monte Carlo 噪声的 `reference - core` 静默截断成标签。

方向特征不使用高频 Fourier 编码。建议输入为：

```text
view-conditioned code 5
+ wi 在两个 learned frame 中的坐标 6
+ top half-vector slope 2
+ wi.z 1
+ log1p(f_top / scale) RGB 3
= 17 scalars
```

EvaluateMLP 采用 `17→32→32→3`，约 `1,664 MAC`；加 top core 和 frame 变换后仍需在 `2,000` 标量路径硬线内逐项记账。共享 FP16 evaluate 权重约 3.5 KB。

这个参数化保留 exact top core 的价值，又不要求多层 residual 能被两个解析 lobe 表达。它必须证明实现对应设计且训练稳定收敛；收敛后质量较低时记录结构归因，不删除逐方向 MLP、退回 lobe-only，也不围绕某个绝对误差线反复重跑。

### 2.4 learned tractable sampler 轴

sampler head 由同一 `prepare` 编码预测 proposal 参数。训练时 sampler loss 在 shared evaluator latent/encoding 处 stop-gradient，避免为了 proposal 方差破坏 evaluator；sampler projection 自身单独更新。运行时两者仍属于同一 state 和同一 Slang backend。

#### NVIDIA 9 参数 baseline

论文 proposal 为：

```text
p_nv(wi | h) = w_d * p_tilted_cosine(wi; mu_d)
             + w_s * p_noncentered_ggx(wi; wo, alpha_x, alpha_y, rho, mu_s)
```

- diffuse normal 由预测 slope `mu_d.xy` 倾斜；
- specular NDF 由 `alpha_x/alpha_y`、相关系数 `rho` 和平均 slope `mu_s.xy` 形成非中心椭圆 GGX，再经 half-vector reflection Jacobian 得到方向 PDF；
- `w_d/w_s` 由 softmax 归一；
- 为满足本项目对整个有效 upper hemisphere 的支持合同，外层加入固定 `epsilon=1/32` 的未倾斜 cosine safety component；这项适配必须在 baseline provenance 中明确。

NDF reflection 可能产生 below-surface/null event。数学空间定义为“有效连续方向 + 一个显式 null bin”：连续 PDF 的积分加 null mass 必须为 1；`sample()` 遇到 null 返回零贡献，不允许隐藏 rejection/resampling。

#### LTC-K2 候选

对 `wi.z > NCLS_MIN_COS`：

```text
q(wi | h) = epsilon * q_cos(wi)
           + (1 - epsilon)
             * [sigmoid(a) * q_ltc0(wi)
                + (1 - sigmoid(a)) * q_ltc1(wi)]

epsilon = 1 / 32
```

- `q_cos = wi.z / pi`；固定正权重给整个开放上半球 full support；
- 每个 `q_ltc` 是 normalized cosine hemisphere 经非奇异线性变换后的 push-forward density；正对角、有限 shear 与 rotation 由参数解码保证；
- `sample(u)` 用 `u.x` 选择并重映射 mixture 分量，用 `u.yz` 生成 cosine/LTC 样本；
- 返回方向后必须调用完整 mixture 公式计算 PDF，`sample.pdf` 与独立 `pdf(wi)` 共用一个函数；
- 非 delta `sample.weight` 只通过 `evaluate(wi) * wi.z / pdf(wi)` 形成。

LTC 线性变换保持 upper hemisphere，不产生 reflection null event。它是否比 NVIDIA diffuse+GGX9 更适合当前 LayerStack 只能由 matched 方差、成本和稳定性决定，不能因为公式复用方便而预设为默认。

两种设计都保证数学一致性和无偏性；sampler 训练得好坏只影响方差，不影响期望值。VNDF 继续作为通用 physical-interface/control 原语，但它与 NVIDIA 的 non-centered NDF proposal 是不同组件。

### 2.5 sampler loss

主训练遵循 NVIDIA 证据：sampler proposal 相对“当前冻结/当前步 evaluator 所定义的 `luminance(f_hat·cos)` 分布”优化 KL，并对 latent/shared evaluator encoding stop-gradient。这样 proposal 与实际 path-traced method 匹配，而不是与另一个 reference evaluator 匹配。

v5 shard 的 reference `response = f·cos` 和 solid-angle weight 用于 evaluator 训练、方向积分和 sampler oracle。对每个 `(state, wo)`，另构造 reference diagnostic：

```text
t_i = luminance(max(response_i, 0))
p*_i = t_i / sum_j(t_j * solid_angle_weight_j)
L_reference_oracle = -sum_i p*_i * solid_angle_weight_i * log(q_i)
```

零能量 group 显式回退为 cosine 目标。训练报告同时保存 evaluator-KL、reference-oracle cross-entropy、PDF/null 归一化误差、sample histogram 距离，以及相对 cosine proposal 的积分方差；不得只报告 sampler loss。

NVIDIA 还使用 directional mollification 让窄峰从宽到窄进入训练。它是唯一可能触发新增 reference response 的 baseline 核心机制：learned-frame evaluator 仍只需要 `(state, wo, wi, response)`，sampler KL 查询当前 learned evaluator，也不需要 source `reference_pdf`。数据 adequacy gate 要先判断当前 peak-aware v5 corpus 能否按明确邻域/权重构造同语义 curriculum；不足时才生成新 corpus identity，若需要新增 cone radius、anchor/group 和 curriculum level 语义则升级合同并重采，不能直接删掉该机制后仍称“充分复用 baseline”。

### 2.6 evaluator loss 与质量门

复用已审计的 appearance loss 与数据权重，删除任何 prediction clamp。训练和 selection 至少包含：

- transformed directional loss + 显式 linear/energy 项；
- tail guard checkpoint selection；
- signed energy ratio、`E_core/E_ref`、最差 state、bootstrap CI 和 leave-one-state-out；
- target-visible P1 回归，以及“shared decoder 未见该 state、但通过 reference queries direct-fit latent”的 offline cook 工作流测试。

进入部署轨道前分别检查：method correspondence/独立 oracle、训练稳定收敛、SlangPy/Falcor/packed parity、sampler 数学正确性。quality suite 的 sanity 仍拒绝非有限、负值或合同错误，但 directional/energy 的绝对数值只报告，不作复现 kill gate。

此外执行 evaluator `{NVIDIA direct, exact-core positive residual}` × sampler `{NVIDIA diffuse+GGX9, LTC-K2}` 的 2×2 matched 对照。比较按材质结构报告 evaluator quality、sampler variance、时间和内存的 paired evidence；不同结构组出现不同 Pareto 结论时保留多个非支配方法，不机械回退到 baseline，也不把比较失利写成复现失败。

## 3. 可复用 Slang 数学层

旧方法身份删除前，先将下列原语迁入不依赖 Falcor、MethodBundle 或 LayerStack IR 的公共 Slang 模块：

| 原语 | 公共语义 | 使用方 |
|---|---|---|
| cosine hemisphere | `sample/pdf`、方向与测度 | neural sampler、control、测试 |
| LTC transform | matrix decode、normalized `sample/pdf`、response basis | neural sampler、analytic control、GPU oracle |
| GGX | `D/G/lambda` | physical interface、control |
| GGX VNDF | visible-normal `sample/pdf`；reflection null event 显式保留 | LayerStack interface、analytic control |
| tilted cosine | slope-normal decode、sample/PDF、null/support 语义 | NVIDIA sampler baseline |
| non-centered anisotropic GGX NDF | 9 参数 range warp、half-vector sample/PDF、reflection Jacobian/null mass | NVIDIA sampler baseline、以后 proposal |
| frame/direction | safe normalize、局部/世界变换、learned frame decode | backend、reference、viewer adapter |
| finite mixture | component selection/remap、完整 mixture PDF | neural sampler、以后其他 proposal |

参数 head、RGB amplitude、method state pack 等仍是 backend 私有，不为了“复用”泄漏进公共合同。公共原语只保留一份公式，GPU 测试直接包含同一源码。

## 4. MethodBundle 与通用 Slang specialization

MethodBundle 需要把 backend Slang module、concrete backend type、contract version、entry capability、资源 layout 和所有内容 hash 描述完整。viewer 的 deferred/PT shader 只定义 bundle 给出的 module/type specialization，然后通过 `INclsScatteringBackend` 调用；loader 不按 `backend_id` 写 Film/new-method 分支，也不直接调用 backend 自由函数。

加载顺序保持：schema/hash → contract/platform → shader module/type → compiled-material layout → state stride/cost/capability → parity probe。任何不匹配都明确失败，不回退旧方法或 analytic control。

## 5. Viewer renderer path

右侧 MethodBundle 建立两个明确模式：

1. `Reference PT | Method Deferred`：右侧 G-buffer prepare 后，由通用 specialization 对显式灯光方向调用同一 `evaluate()`；
2. `Reference PT | Method PT`：右侧 ray hit inline `prepare()`，下一方向调用同一 state 的 `sample()/pdf()`，路径权重来自同一 `evaluate()`。

method PT 不是 source family dispatch 分支。source reference 仍按各族 adapter 路径执行；两侧只在 composite/capture 层形成对照。

数学 sampler 正确性先由同 evaluator 的确定性积分与 Monte Carlo 估计证明；source-reference PT 对照再测表示误差和最终图像，不混淆两类结论。

## 6. 数据与迁移

- 保留 `layer-stack-p1-v1` v5 corpus 和 corpus manifest；本方法无需 source `reference_pdf`。
- 先执行 directional-mollification adequacy spike：在 diffuse、窄导体峰、grazing 和四个既有尾部 state 上，对当前 corpus 的邻域重建结果与同 anchor 的新鲜 cone-averaged reference queries 做 matched 比较，并在子任务 planning 时冻结误差阈值。
- adequacy spike 通过则直接复用现有 v5 corpus；未通过时只新增覆盖 mollification curriculum 所需的 versioned reference-response corpus，不因 learned frames、sampler KL 或解析 sample/PDF 本身重采。
- 新数据若保存每个 cone 内的原始 jittered point queries，可继续使用 point-response 字段，但 manifest 必须记录 anchor/group、cone distribution/radius、sample count 与 curriculum level；若现有 schema 无法无歧义表达这些语义，则先升级 schema，再整体生成该新 corpus identity。
- v3/v4 shard、smoke 数据和旧 P1 方法产物按 `research/current-state.md` 的清单删除。
- 若 v5 reader 审计发现字段/measure 与合同不符，升版本并重采整个受影响 corpus；不得局部修补或重新解释。
- 旧 Film、旧 `lobe_residual` 注册/配置/backend/viewer 旁路在替代路径验收后同一任务批次删除；不会保留 fallback。

## 7. 显式对照

保留一个新身份的 `ltc-k2-analytic-control`：exact top + 两个非负 LTC evaluator lobe，并通过相同 MethodBundle/backend/viewer 接口执行。它只用于 optimized-code control、LTC oracle 和成本/质量对照，不能自动替代 neural 方法。

对照的存在不意味着 neural evaluator 要输出 LTC 参数；两者共享的只有公共数学组件和 renderer 合同。

## 8. 失败与回退规则

- sample/PDF 任一归一化、histogram、re-evaluation 或有限值测试失败：禁止进入 PT；修复公共数学原语。
- evaluator 实现或 convergence 未通过：禁止把该 run 写成复现成功；先修 method correspondence、梯度或训练生命周期。实现正确且稳定收敛但质量较低时保留结果并继续导出/viewer 证据，不围绕绝对质量线反复修正。
- sampler 数学正确但方差不优于 cosine：PT 仍无偏，但方法不能宣称 learned sampler 有效；继续改 proposal head/family后再收口。
- SlangPy 与 Falcor 2024.1.34 无法编译同一源：任务阻塞在单一源 gate，不维护第二套生产前向。
- viewer bundle 不兼容：明确报错；不回退 Film、旧 lobe-residual 或 analytic control。

## 9. 旧材料判定表

| 旧结论/实现 | 判定 | 新设计中的位置 |
|---|---|---|
| exact top core | 复用 | evaluator physical core、control |
| signed residual + final clamp | 淘汰 | 改为 nonnegative residual，无 final clamp |
| K=2 LTC 部署预算 | 修正复用 | sampler matched 候选与 analytic control，不限制 evaluator |
| NVIDIA learned frames / two-lobe sampler | 提升为必做 baseline | 原规模忠实复现 + convergence 证据 + 2×2 分组对照；缩模不是前置 |
| `f *= exp(Delta)` | 淘汰 | 不修改 exact core；逐方向 MLP直接预测 residual |
| `correction=none` 默认 | 淘汰 | 目标方法的 EvaluateMLP 必选 |
| learned/physical direction frame | 复用并加强 | 两个 learned evaluator frames |
| 高频 Fourier log-slope | 淘汰 | frame coordinates + bounded half-slope |
| 单一 Slang source | 复用 | training/GPU/viewer 共编译 |
| method 作为 reference 第五 family | 淘汰 | MethodBundle backend specialization |
| Film compatibility path | 淘汰 | 无 fallback 的通用 loader |
