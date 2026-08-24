# Evaluator-first 闭环建模与 compiler 设计

## 1. 文档状态与决策边界

本文是下一阶段模型工作的评审结论和执行设计。原稿提出的 directional chart、shared modulation、canonical field、target encoder 与 typed source adapter 都是有价值的候选，但原先“先建立高保真 oracle，再依次通过 shared representation、compiler 和部署 gate”的串行路线不作为执行计划。它会把共享、编译和运行时反馈推迟到很晚，也会诱导团队把大量时间花在证明某个模型族的“上界”上。

新的主线是：先用一个最小但语义完整的候选打通 `reference → train → evaluate → compiler control → MethodBundle/Slang → viewer/报告` 回环；之后每轮只针对已经观测到的主要失败增加一种机制，并把修复过的问题固化为回归项。高容量 teacher、learned frame、lobe expert、tensor field 和 attention encoder 都按问题启用，不按清单一次实现。

本文只确定五件事：

1. 如何先形成能反复运行、能定位失败的最小纵向回环；
2. 哪个 evaluator 是首个 walking skeleton，哪些结构只作为问题驱动的增量候选；
3. target response encoder 与 source compiler 怎样接入同一个 evaluator，而不混淆输入语义；
4. 数据、表示、优化、compiler 和部署失败怎样分开归因；
5. 哪些是始终生效的合同约束，哪些只是研究 scorecard 或阶段性晋级条件。

本文不改变最终运行时合同。目标仍是：

```text
prepared = prepare(material_program, surface_state, wo, footprint)
response = evaluate(prepared, wi)
```

sampler、环境积分和空间 LOD 仍在 evaluator 主体稳定后展开；这里的闭环不把多灯 scaling、PT 方差或 UE 集成提前变成 kill test。本文中的高容量 teacher、target encoder 和 source compiler 可以只存在于训练或 cook 阶段；walking skeleton 与保留下来的 student evaluator 必须尽早进入 Slang/MethodBundle，以便运行时布局、parity 和真实成本能够反向影响模型设计。

## 2. 对抗性评审结论

### 2.1 值得保留的部分

原设计有四个正确判断：方向移动结构需要 canonical chart；target-visible code 与 source compiler 不能混淆；direct 与 analytic residual 必须分开报告；质量、资产 bytes、共享权重和真实 GPU 成本不能压成一个指标。这些判断继续成立。

### 2.2 原设计不能直接执行的原因

1. **监督量与运行时输出没有闭合。** HDF5 保存 `y=f|cos theta_i|`，原稿却把 `y_hat` 直接写成 `evaluate()` 输出；公共 backend 合同要求 `evaluate()` 返回不含几何余弦的 `f`。如果运行时再除以接近零的 cosine，掠射区域会出现数值病态。新模型必须直接输出 `f` 或 `Delta f`，只在 loss/metric 内构造 `y_hat=f_hat|cos theta_i|`。
2. **路线过度串行。** 模型 A 未“通过”就不进入 B，B 未“通过”就不接 compiler，最后才做 Slang，会让 code 布局、材质分歧、`prepare/evaluate` 划分和导出限制太晚暴露。
3. **模型 B 同时改变太多机制。** FiLM、低秩 weight modulation、lobe token、mixture、signed correction 和新 chart 同时加入，成功时无法知道是谁起作用，失败时也无法知道该修哪里。
4. **容量与证据规模不匹配。** 当前 E1/E2 H5 足以做基础比较，但用约百万参数 teacher 或 shared decoder 对少量 state 和固定方向 probe 得出“连续函数已经可达”的结论仍然过强。需要按失败位置追加独立 probe，并扩大 state 组合，而不是只加模型容量。
5. **direct oracle 被赋予了不必要的否决权。** 高容量 direct fit 能帮助区分数据、坐标和共享瓶颈，但目标候选允许 analytic physical core。direct teacher 失败不应自动阻止一个实际 residual 候选进入 shared/compiler/Slang 回环。
6. **test 治理不足。** “每个设计冻结后读取一次 test”仍会在多轮公开实验中逐渐把同一 test 变成事实上的 validation；E2 仅有两个 source-test state 时，p90/p95 也不能稳定代表未见状态分布。需要日常 development holdout 和里程碑 sealed benchmark 分离。
7. **单一 fidelity gate 容易误导。** reference SE、视觉容差、峰值、积分能量和连续 sweep 的意义不同；把它们压成 pass/fail 会掩盖主要失败，也会诱导针对门槛调参。尤其 `model error / reference SE` 在 deterministic 或极低 SE 区域会过度放大微小但未必重要的绝对误差。
8. **缺少连续性与编辑路径证据。** 固定 query 指标不能发现 view sweep 峰位跳变、参数编辑抖动、lobe 路由切换和 viewer temporal artifact。compiler confidence 也只有在能触发追加采样、refinement 或拒绝部署时才有意义。

结论是：原稿可作为候选机制库，不能作为瀑布式审批表。以下设计按闭环执行重新组织。

### 2.3 现有 E1-E3 结果的证据边界

现有 E1–E3 完成了公共 reader、split、transform、checkpoint、独立 test、metrics 和 source adapter 的生命周期验证，这些工程证据继续有效；需要修正的是模型结论的适用范围。

#### E1 的适用范围

E1 的 direct evaluator 在极窄 `alpha_x=0.002` 高光上失败，说明冻结的约 65k-MAC 小 MLP、既有方向特征与 loss 组合不能保持峰值；它不证明更强的 neural directional representation 也不能拟合该函数。多界面 analytic-core residual 通过已有 gate，说明 direct-top core 加小 residual 网络在该状态上是一个有效的成本受限候选；由于最难的移动峰可能已经由 core 承担，它不能单独证明 direct neural evaluator 的完整容量。

因此，现有 E1 结果改称：

- `direct small MLP`：效率受限 baseline；
- `analytic core + small residual MLP`：成本受限单材质候选；
- `raw-direction pairwise plane v1`：只淘汰该坐标与分解方式。

#### E2 的适用范围

E2 主要固定为 16 维 latent、第一层 concat conditioning、width 108 小 MLP 和同一种 residual parameterization。width 123 与 latent 32 的局部对照没有改善长尾，只能说明在相同网络族和成本边界内继续小幅扩容价值低，不能说明：

- layer-wise modulation 或 partial-weight generation 无效；
- learned shading frame、multi-lobe expert 或物理 warp 无效；
- 64/128 维 latent 与高容量 decoder 无法形成共享表示；
- target response 的空间/方向结构只能由 DeepSets mean/max 编码。

因此，`dense16/width108`、DeepSets encoder、bounded refinement、hard top-k 和 rank-4 factor 均保留为“特定实现的 target-visible compression baseline”。后续 optimized latent 统一称为同 decoder、数据和优化预算下的 `optimized-code control`，不把有限实验包装成数学或经验“上界”。

#### E3 的适用范围

当前 LayerStack source compiler 使用一个约一万参数的 token MLP + GRU，从零联合训练 decoder，并同时预测 latent 和九个逐状态 residual transform 常量。它的 smoke 证明 family adapter、native-token 输入、source-held-out partition 和独立 evaluation 路径能够工作；初步 validation 过拟合只说明这套最小 compiler/training design 不稳定。

在正式 gate、诊断和新设计评审前，不从这个 run 推导 source compiler 路线的质量结论。新的 compiler 不再负责外推任意 normalization statistics，而是预测有稳定语义的 evaluator code；transform 由 train-corpus 统计或显式能量/颜色 head 处理。

## 3. 闭环优先研究原则

### 3.1 先打通回环，再逐步提高质量

第一版新 evaluator 不要求达到最终质量，但必须能经过完整生命周期：

1. 从同一 ReferenceDataset 训练并产生可复现 checkpoint；
2. 分别运行 optimized-code control 与一个不读取 target response 的 source-compiler control；
3. 导出 MethodBundle，在 Slang 中执行相同 `prepare/evaluate`；
4. 生成固定 response slice、view/state sweep、成本与 parity 报告；
5. 把最坏 state/query、失败层级和下一假设写入 failure ledger。

`B_asset`、`B_shared`、`C_prepare`、`C_eval` 从第一轮就记录，但早期不作为质量探索的统一 kill threshold。高容量模型只在无法判断问题属于监督、方向表示还是共享瓶颈时作为 teacher/诊断，不自动获得更高结论地位。

### 3.2 每轮只解决一个主问题

每轮先用现有证据把主要失败归入 `data/reference`、`directional representation`、`optimization/loss`、`shared code`、`source compiler` 或 `deployment`。随后只引入一个能证伪的变化，保留 matched control，并检查已解决项是否回退。没有具体失败证据时，不因候选清单存在就实现 learned frame、更多 lobe、更大 Transformer 或 tensor rank。

“一个变化”指一个可归因假设，不要求机械地只改一个标量。若新 chart 必须同时增加 validity flag 和 Slang 实现，或 mode head 必须配套路由 loss，应把它声明为最小机制 bundle，并在 bundle 内保留能识别主要贡献的消融。

### 3.3 先对齐移动结构，再增加通用容量

镜面峰会随 `wo`、粗糙度、各向异性、折射率和层结构移动。在 raw `wo/wi` 坐标上，这表现为一个高速移动的窄脊；盲目加宽 MLP 仍可能先学习低频平均。可检查的设计顺序是：

1. reflection/transmission half-difference chart；
2. Rusinkiewicz spherical-harmonic encoding；
3. learned shading frame 或可逆物理 warp；
4. multi-lobe/expert 或 tensor factorization；
5. 最后才是更大的通用 residual backbone。

这里的 chart、frame 和 lobe 是 evaluator 内部结构，不是要求所有 source family 暴露统一 analytic closure。

### 3.4 direct response 是独立诊断，不是统一门槛

analytic core + neural residual 是正式候选；direct response control 用来量化 core 承担了什么、shared decoder 是否在依赖 family-specific core。对代表状态保留：

- direct response control；
- 若 family 存在可信 analytic core，再增加 signed residual 对照；
- core-only 指标，量化 neural 部分的实际贡献。

direct control 的失败会生成方向表示或监督问题，但不会自动阻止 residual 候选继续打通 shared/compiler/Slang 回环。只有目标 claim 明确要求“完全不依赖 physical core”时，direct fidelity 才是该 claim 的晋级条件。

### 3.5 development、sealed test 与对抗性 probe 分离

训练和 hard-query mining 只读取 source train × query train。validation 与明确标记的 development probes 用于日常选架构/config/checkpoint；sealed source/query test 只在里程碑候选晋级时读取。若 sealed test 暴露的问题影响下一轮设计，该 test 已经成为开发证据，后续最终结论必须使用新版本或仍未公开的 sealed benchmark。adversarial probe、response-slice/capture 继续独立报告。任何 target transform、chart density、codebook、peak token 或 hard-example distribution 都不能从 validation/test 拟合。

当前 E1/E2 文档已经公开了既有 test 的逐项结果和最坏 state，并据此改变过设计；对新模型而言，它们只能作为 versioned development/regression evidence。R0 必须从 source corpus 另建未读取的 milestone split，且 E2 不再用 `2` 个 test state 的 p90/p95 支撑总体泛化 claim。

## 4. 公共符号和张量合同

设：

- `m`：一个 source material state；
- `wo ∈ S²`：局部 frame 中的出射/观察方向；
- `wi ∈ S²`：局部 frame 中的入射/光照方向；
- `f(m, wo, wi) ∈ R³`：公共 `evaluate()` 必须返回的、不含几何余弦的 RGB 散射值；
- `y(m, wo, wi)=f(m, wo, wi)|cos theta_i|`：HDF5 保存的 RGB `response_cos` 监督量；
- `z_m ∈ R^D`：可部署 material code；
- `p_m(wo)`：`prepare()` 可被多个 `wi` 复用的 view-conditioned state。

所有 evaluator 继续满足：

```text
p_m(wo) = prepare(z_m, wo)
f_hat = evaluate(p_m(wo), wi)
y_hat = f_hat * abs(cos_theta_i)  # 只在训练、审计和积分 metric 内构造
```

训练实现可以一次处理：

```text
wo : [G, 3]
wi : [G, N, 3]
y  : [G, N, 3]
```

但公共模型不得依赖固定 `N`，也不得把一个完整方向表藏进 `prepare()`。单次随机 `wi` 查询必须成立。输出 transform 只能是训练内部的 loss parameterization；不能要求运行时从 `response_cos` 除以接近零的 cosine 才恢复 `f`。direct head 预测 `f`，analytic residual head 预测 `Delta f=f-f_core`，随后在 loss 内乘一次 cosine 与 HDF5 对齐。

reflection chart 在 `wo + wi` 有效时使用反射 half vector；transmission chart 使用 source 语义允许时的 generalized half vector。折射率只由原生 source 确实提供的 family adapter 使用；MERL 等不提供该参数的 family 采用 learned transmission frame/chart，不伪造 LayerStack 参数。

chart 计算必须返回显式 validity/mode flags，避免在 half-vector 退化、临界区域或反射—透射边界产生 NaN。所有归一化使用版本化 epsilon，并进入 Python/Slang parity 合同。

注册 B0 前，chart contract 必须给出 reflection/transmission 的精确公式、方向约定、eta 使用方向、坐标域、seam/边界处理和退化 fallback，并用 near-grazing、critical transmission、方向交换和随机 probe 做 Python/Slang 测试。`learned transmission chart` 是单独的模型候选，不能作为同一 feature ID 下不透明的 family fallback；缺少 source 参数时先保留 raw features 和显式 invalid flag。

## 5. 模型 A：按需启用的双方向诊断 teacher

计划 ID：`ncls.warped-directional-diagnostic-teacher@1`。它只是设计名称，不是已注册 contract。

### 5.1 它回答什么

模型 A 针对一个 material state 拟合完整 `wo×wi`。它没有 material latent 瓶颈，不是 walking skeleton 的前置条件，只在主候选的失败无法归因时回答三个问题：

1. 当前 train supervision 加独立局部 probe 是否足以支撑连续插值；
2. 方向 chart/encoding 能否稳定表达极窄峰、掠射和 transmission；
3. direct response 与 analytic residual 的真实难度差距是多少。

模型 A 不是最终 shared runtime，也不能证明 source compiler 泛化。即使它在固定 H5 上表现很好，也必须通过新增的局部高分辨率 probe 排除大模型对离散方向表的记忆；不能把参数多于有效训练 query 的拟合自动解释为连续域证据。

### 5.2 输入编码

起点输入由前三组特征组成，spectral features 是按需增加的第四组：

1. **raw local features**：`wo`、`wi`、两方向的 cosine、hemisphere/mode/validity flags；
2. **reflection chart**：Rusinkiewicz `h_r/d_r`、half-vector slope 和 grazing-safe log-slope；
3. **transmission chart**：valid generalized `h_t/d_t`，或由 learned frame 形成的无 source-parameter chart；
4. **spectral features**：只有 chart probe 仍显示稳定高频残差时，才对 `h/d` 增加 real spherical harmonics；9/5 阶是诊断 envelope，不是默认值，阶数分别配置且不能写死在 reader 中。

若启用高阶，主要分配给 half-vector 轴，因为窄峰在 canonical reflection chart 中主要沿该轴变化。raw features 始终保留，防止 chart 在退化或非典型多散射区域丢失信息。固定 SH 阶数本身不能证明极窄峰已被解析；必须看局部 probe 的误差随阶数是否因果下降。

### 5.3 learned frame

learned frame 不是默认起点。只有 raw + canonical chart 的误差沿着随 `wo` 移动的峰轨迹集中时，才增加 `J=1` 的 trainable shading frame；证据表明确仍有多个不能由单 frame 对齐的峰时，再比较 `J=2/4`。frame 使用连续 6D rotation representation，经 normalize + Gram-Schmidt 生成正交 normal/tangent/bitangent，禁止直接回归未约束 `3×3` 矩阵。

frame 本身是 material-static；`prepare(wo)` 只预测各 frame 的 view-conditioned 权重和 feature，不让 frame 随每个 `wi` 变化。这样同一个着色点的多个 evaluate query 可复用 frame，并避免网络通过任意 query-dependent rotation 记忆响应。

### 5.4 网络结构

诊断容量 envelope 如下，但从 `width=128、prepare 2 blocks、evaluate 4 blocks` 起步；只有 train 与 development probe 同向改善才扩到表中最大值：

| 模块 | 初始结构 | 输出 |
|---|---|---|
| frame/chart encoder | 每个 frame 共用 `2×256` residual block | canonical direction feature |
| `prepare` trunk | `3×256` residual block，GELU | `p(wo)∈R²⁵⁶` 与 frame weights |
| `evaluate` trunk | `6×256` pre-norm residual block，GELU | 每 query hidden feature |
| direct head | `256→128→3` | RGB `f` 或 signed `Delta f` |
| optional energy auxiliary head | `p(wo)→RGB × mode` | 仅作 hemispherical energy 辅助/诊断 |

每个 residual block 使用 linear → GELU → linear 和 identity skip；LayerNorm 只作用于 hidden feature，不跨 query 或材质估计统计。百万级参数是诊断上限，不是默认实现。每个 run 同时报告参数数、有效 train query 数和新 probe 数；若只改善 train 而不改善局部 probe，优先判断为覆盖/连续泛化问题，不继续加宽。

SIREN/periodic activation 只作为同输入、同参数量的单独 ablation，不作为默认模型。它可能提高高频拟合，也可能在稀疏区域产生振铃；没有 held-out peak 与积分证据前不混入主模型。

### 5.5 输出参数化

direct 版本的运行时 head 预测 `f_hat`；HDF5 监督量只在 loss 内构造：

```text
y_hat_c = f_hat_c * abs(cos_theta_i)
t_c = log1p(y_c / s_c)
t_hat_c = log1p(y_hat_c / s_c)
```

`s_c` 只是 train-only loss scale，不参与运行时解码。单材质 teacher 可以用 state-train 统计；shared evaluator 使用 source-train corpus/family 统计。这样 pure source compiler 不需要外推 target-visible transform 常量。

在 source working color space 保证非负时，head 使用版本化的非负输出映射得到 `f_hat`，不能依赖 loss 内 clamp 掩盖非法预测；合法负通道则使用 unconstrained/signed head，并在对应颜色合同下评测。

若 source/color-space contract 允许合法负通道，例如线性颜色空间转换产生的 out-of-gamut 值，则 direct 版本改用 signed `asinh`，或者优先在 source 的权威非负 working space 中学习后再执行固定颜色变换。不能对合法负值使用 `log1p`、softplus 或统一 clamp。

analytic residual 版本预测 `Delta f`，仍然只在监督测度中变换：

```text
f_hat = f_core + Delta_f_hat
y_hat = f_hat * abs(cos_theta_i)
r_y = (f - f_core) * abs(cos_theta_i)
r_y_hat = Delta_f_hat * abs(cos_theta_i)
t_c = asinh(r_y_c / s_c)
t_hat_c = asinh(r_y_hat_c / s_c)
```

不能对 signed residual 使用 log，也不能把 source 合法的负色域通道统一 clamp 为零。范围约束由 family/color-space contract 决定，并直接检查最终 `f_hat`；禁止先预测 `response_cos` 再在 shader 中除以 cosine。

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
- `L_integrated_energy`：由 evaluator 的 `y_hat` 在同一组带权 `wi` 上积分得到的能量误差；独立 energy head 只能辅助 `prepare` 或诊断，不能替代 evaluator 积分而让主干逃避约束；
- `L_top_energy`：只在 train query 的 top-energy support 上加权，防止窄峰被低值区域淹没；
- `L_source_reciprocity`：在转换回 `f` 后、使用版本化掠射 mask 与 source event 语义比较 reciprocity deviation，不直接交换两个 `response_cos`，也不对已知 source 非互易项施加错误的绝对零约束。

各权重必须在读取 test 前通过 train/validation 因果对照确定并版本化。不能因为某个 metric 失败就做无边界 weight sweep。

### 5.7 训练与采样

每次训练分三段：

1. **coverage warm-up**：使用冻结的 uniform/cosine/microfacet-aware train mixture；
2. **peak curriculum**：逐步提高 train-only high-energy、掠射和 transmission-critical query 的 batch 比例；
3. **error refinement**：周期性在额外 train-only proposal 上求值，用模型误差与 reference SE 选择 hard query，并保存为有独立 provenance 的补充 probe/query pack。

hard-query mining 只能重采 train domain。所有 loss 仍使用 importance/solid-angle correction，不能把 oversampling 后的经验分布当成均匀半球指标。

每个进入里程碑比较的设计至少运行三个 seed。日常迭代只读取 validation/development probes；只有候选准备晋级时才读取 sealed test。单次最佳 seed 不能作为稳定性证据。

### 5.8 最小因果对照

模型 A 不做无差别 sweep，只运行下列有明确问题的递增对照：

| 对照 | 唯一变化 | 回答的问题 |
|---|---|---|
| A0 | high-capacity raw/Rusinkiewicz ResNet，无 learned frame | 单纯解除小网络容量是否足够 |
| A1 | 仅在峰轨迹错位时，A0 + 一个 learned shading frame | 移动峰是否主要是坐标对齐问题 |
| A2 | 仅在 mode 混淆时，matched candidate + reflection/transmission 分头 | 多 mode 耦合是否限制峰值 |
| A3 | direct 与 analytic residual 配对 | core 贡献与 neural 完整容量差距 |

代表状态从当前主线 LayerStack 的极低 roughness、固定多界面多散射与反射/透射临界区域开始。MERL 只在 LayerStack 回环稳定后作为跨 family 压力测试加入。已有合格 H5 用于初筛；局部 probe 证明覆盖不足时新增版本化 query pack，不重写旧 H5，也不把 development hard query 混入 sealed test。

## 6. 模型 B：逐级增强的 shared evaluator

计划 ID：`ncls.conditioned-shared-evaluator@1`；B3 lobe variant 再使用单独 ID。

### 6.1 它回答什么

模型 B 是 walking skeleton 和主要 shared evaluator 候选。它始终使用同一个 material-code contract，但不会在第一版同时加入 hypernetwork、lobe token 与低秩权重生成。增量层级如下：

| 层级 | 唯一新增机制 | 进入条件 |
|---|---|---|
| B0 | canonical direction front end + shared MLP + layer-wise FiLM | 首个纵向回环 |
| B1 | low-rank weight modulation | B0 对特定 state family 稳定欠拟合，且单纯扩宽被 paired control 支配 |
| B2 | reflection/transmission 分头 | 误差明确集中在 mode 竞争或临界边界 |
| B3 | canonical lobe token/expert | response slice 证明存在多个同时峰，单 head 发生平均化 |

optimized code、source compiler、MethodBundle 和 Slang 都从 B0 接入同一 code/evaluator contract。后续层级替换内部实现时，公共 runner 与证据格式不变。

### 6.2 material code 与调制

walking skeleton 从可配置的 `z_m in R^64` 开始，并保留 `128` 维 matched control。B0 的 shared conditioning 只生成每个 block 的 FiLM `scale/bias` 与小型 global color/energy context。只有进入 B1 后才生成 low-rank coefficient：

第 `l` 层的共享权重保持为 `W_l`，asset-specific 变化为：

```text
h_{l+1} = activation(
    gamma_l(z) * ((W_l + U_l diag(a_l(z)) V_l) h_l) + beta_l(z)
)
```

`U_l/V_l` 是共享参数。`hyper(z_m)` 的生命周期必须明确二选一：若在每个着色点的 `prepare()` 内执行，就完整计入 `C_prepare`；若在 compile/load 时展开为 material-static context，就把展开结果计入 `B_asset` 和编辑后重建时间。不能一边只按 `z_m` 计算资产 bytes，一边假设 shader 免费获得完整调制参数。进入 spatial latent 后还必须验证该机制能否按 texel/filter 后的 code 工作，不能默认 material-static hypernetwork 会自然推广到空间变化。

### 6.3 lobe token

只有 B3 才引入 lobe token。起点使用 `K=2`，证据需要时再到 `4/8`；每 token 可通过受约束 head 表达：

- 6D shading-frame representation；
- reflection/transmission/low-frequency 类型 logits；
- 各向异性 bandwidth 与 roughness-like shape hints；
- RGB amplitude/energy hints；
- shared expert 的 FiLM/context feature。

这些字段是 neural evaluator 的内部坐标和条件，不是公共 analytic closure，也不要求 source adapter 提供 lobe 参数。compiler 可以只输出 `z_m`，lobe token 由共享 `prepare` 网络产生。类型 logits、frame 和 bandwidth 必须有 usage、entropy、跨 view 连续性与专家消融报告；否则 token 可能退化为无语义的额外 hidden feature。

### 6.4 `prepare/evaluate` 划分

```text
static_context = condition(z_m)
p_m(wo) = prepare_trunk(static_context, wo)

q = canonicalize(wo, wi, validity)
f_hat = evaluate_trunk(q, p_m(wo))

# 只有 B3：
for each active lobe k:
    q_k = canonicalize(wo, wi, frame_k, type_k)
    delta_f_k = shared_expert(q_k, p_m(wo), token_k)
```

material modulation 和 `wo` feature 在 `prepare()` 计算一次；B3 的 frame/token 也在这里生成。`evaluate()` 添加 `wi`、chart 与 shared trunk/expert，并返回 `f`。实现必须分别报告一次 query 和多个 query 的 amortization，不能把 material-static condition 或 K-lobe prepare 成本重复计入每个 `wi`，也不能把它们从总成本中省略。

### 6.5 mixture 与输出

B3 direct 版本可以由三部分组成：

1. 每个 canonical lobe expert 输出非负 RGB contribution；
2. low-frequency head 表达 diffuse-like/multiple-scattering background；
3. 小型 signed correction head 修正 lobe 组合不能表达的局部差异。

组合不只对 latent 做凸平均。每个 expert 可以产生独立 contribution，避免 top-k convex mixture 把多个峰平均成宽峰。但 nonnegative expert、low-frequency head 与 unrestricted signed correction 是不可辨识分解；必须限制 correction 的能量/容量，并报告移除每个分支后的误差，否则 correction 可能学完整函数、lobe 只成为装饰。

analytic residual 版本保持 core 外置，并让 lobe experts 预测 signed residual。两版本共享方向 chart、prepare 和调制机制，以隔离 core 的影响。

### 6.6 初始容量与后续压缩

walking skeleton 起点：

| 组件 | 起始值 |
|---|---:|
| material code | 64 floats；128 为 matched control |
| lobe count | 0 |
| prepare width/depth | 128 / 2 residual blocks |
| evaluate width/depth | 128 / 4 residual blocks |
| conditioning | layer-wise FiLM |
| direction encoding | raw + 已通过 parity 的 reflection/transmission chart |

这组值不是最终部署承诺，而是为了尽快跑通训练、compiler control、导出和 viewer。所有轮次都记录 rate-distortion 与真实布局；根据 failure ledger 选择一个方向变化：

```text
capacity:  width/depth 或 latent 64 → 128
sharing:   FiLM → optional rank 2/4/8 modulation
structure: single head → mode heads → K=2/4/8 lobe experts
compression: latent 128 → 64 → 32 → 16
dtype: fp32 → fp16/quantized
```

每次只改变一类瓶颈，并与当前主候选做 paired control。teacher distillation 只在 reference loss 的优化明显不稳定或 teacher 已揭示可迁移局部结构时使用；最小版本失败不会否定所有后续结构，但必须先给出失败位置再决定增加哪一种机制。

### 6.7 与 sparse dictionary 的关系

dictionary 不再从随机 hard top-k logits 起步。当 B0/B1 已形成稳定 optimized-code control，且 failure ledger 显示主要问题是资产 bytes 或 code 聚类结构时，仅使用 source-train states 做 K-means++/residual VQ 初始化，再比较：

- top-1 codeword；
- top-2 非凸/凸混合；
- top-k soft mixture；
- codeword + continuous residual。

codebook、whitening 和聚类统计不读取 validation/test。dictionary 只替换 `z_m` 的资产表示，decoder、query split 和 metrics 保持不变。

## 7. 模型 C：物理 warp 后的 tensor/plane field

计划 ID：`ncls.warped-half-difference-vm-field@1`。

### 7.1 它回答什么

模型 C 是 plane/tensor factorization 的问题驱动备选，不是固定第三阶段。旧 v1 在 raw Cartesian `wo.x/wo.y/wi.x/wi.y` 的六个成对 plane 上分解，镜面峰仍会跨 plane 快速移动。只有 B 系列证据显示 canonical 高频场可以被局部表格直接保存、而 shared MLP 在相同查询成本下持续丢峰时，才实现物理 warp 后的低秩与多尺度结构。

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

第一项让高分辨率 half plane 直接承载窄峰；第二项补充 difference-direction 的二维相关。各尺度 feature concat 后交给小 decoder 输出 `f` 或 signed `Delta f`；训练时再乘 cosine 对齐 HDF5。

起点先用单一 `64/128` half-plane 与较低 rank 验证误差是否随 half-plane resolution 因果下降，再决定是否增加 `256` 或多尺度。原稿的 `32/64/128/256 × rank 32 × reflection/transmission` 资产可能远大于部署候选，不能在尚未证明局部表格确实解决主失败前一次分配。只在 half plane 增加分辨率/rank，不能同时扩大所有 plane 后把改善原因混在一起。

### 7.4 插值、边界和随机访问

- 2D plane 使用 bilinear fetch，1D factor 使用 linear fetch；
- chart seam 使用周期坐标或成对 seam feature，不能依赖 clamp 隐藏不连续；
- reflection/transmission branch 在物理 mode/validity 上显式路由；
- 单 query 只读取固定数量的邻域 texel/factor，保持随机访问；
- 资产 bytes、fetch 数和 decoder MAC 从第一轮起记录，但在 fidelity 阶段不淘汰。

### 7.5 与模型 B 的比较

模型 C 不是模型 B 的所有组合项。它复用已经通过 Python/Slang parity 和局部 probe 的 chart，并与当时的 B 系列 optimized-code control 比较：

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

新的 encoder 仍接受无固定顺序的 train query，但不能把所有点展平成一个无结构集合。第一层在同一 `wo` 的 query group 内编码完整 `wi` shape，第二层再聚合不同 `wo`；这样 query-group 语义是显式的，attention 不必从浮点相等关系中猜出哪些点共享观察方向。

### 8.2 输入 token

每个 train-only query token 包含：

```text
raw wo/wi
reflection/transmission canonical coordinates + validity
transformed RGB response_cos or signed response_cos residual
reference standard error
solid-angle / proposal weight
energy-bin and train-only peak-support flags
```

peak-support flag 与任何 bin threshold 只由该 state 的 query-train 数据计算。validation/test point 不进入 encoder input，也不参与 token normalization。

### 8.3 网络结构

只有 failure ledger 显示 DeepSets 的主要瓶颈是 target-code initialization，且 optimized-code control 明显更好时，才实现 attention encoder。起点设计：

- point projection width 128；
- 每个 `wo` group 内做局部 attention/pooling；
- 16/32 个跨 group learned inducing tokens；
- 2 个 induced self-attention blocks，4 heads；
- reflection、transmission 和 low-frequency 三组 learned pooling queries；
- 最终输出与当前 B 候选一致的 `z_m`。

inducing attention 使计算随输入点数近似线性增长；训练时可分块累积 response tokens，但分块规则和归约顺序必须进入 fitted-state hash。`encoder confidence` 只有在预先定义了低置信度动作，例如追加 target query、进入 bounded refinement 或拒绝发布时才实现；否则只保留可校准的 reconstruction diagnostic，不增加一个没有消费者的 head。

### 8.4 训练阶段

1. 固定当前稳定的模型 B decoder 与 code contract；
2. 使用 optimized `z_m*` 做弱 latent-alignment loss；
3. 以完整 query-space reconstruction/distillation 为主 loss，避免 latent 对称性使直接 L2 误导；
4. 得到 deterministic `z_0=E(X_train)`；
5. 冻结 encoder/decoder，只对 `z_0` 做版本化 bounded refinement。

报告必须分开列出 encoder-only、refinement 后结果、compile time、refinement time、输入 bytes 和 seed 方差。target encoder 读取 reference response，因此永远不等同于 source compiler。

## 9. Typed source compiler

计划 ID：`ncls.typed-source-compiler@1`；graph Transformer 若被证据触发，再注册结构变体。

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

LayerStack 的 typed adapter 仍先实现，因为 token 顺序和 interface/medium 邻接是 source 语义的一部分；但第一版 quality compiler 不直接从 6 层 graph Transformer 起步。闭环按两级实现：

1. **compiler control**：固定长度/有 mask 的 order-aware token projection + 小型 sequence encoder，尽早输出 B0 code、进入 MethodBundle provenance，并暴露 pure feed-forward gap；
2. **quality compiler**：只有 control 的误差随 layer count、拓扑或长程相互作用系统增长时，才比较 deeper sequence model 或 order-aware graph Transformer。

compiler 是 offline 路径，不受 shader cost 限制；它不会进入 runtime bundle。不同 family 可以有不同前端/graph encoder，但必须输出同一个 evaluator code contract，公共 runner 不增加 family 分支。`resource embedding references` 只定义引用关系，不等于纹理/测量表已经被编码；每个新 family 必须单独说明资源读取、filtering、空间 code 与编辑失效范围。

### 9.3 不再预测任意 transform 常量

compiler 不输出每 state 的 residual scale/mean/std。允许的替代只有：

1. source-train corpus 固定的 family-global transform；
2. evaluator 内显式 energy/color head；
3. source 原生语义确实包含并能直接计算的物理量。

这样 compiler generalization 测的是材质函数 code，而不是同时外推一个依赖 target response 统计的坐标系。

### 9.4 训练阶段

quality 训练可以复用以下步骤，但 compiler control 从首个 B0 回环就存在，不等待 decoder 达到某个总分：

1. **control construction**：保存模型 B 在 source-train states 上的 optimized code 与 functional metrics；
2. **canonicalization**：使用共同初始化、code regularization 和 decoder 固定，降低 optimized latent 的排列/尺度不确定性；
3. **code distillation**：compiler 接近 target encoder/optimized code，只作初始化；
4. **functional distillation**：固定 decoder，在 source-train × query-train 上最小化 response/energy/peak loss，这是主要目标；
5. **joint fine-tune**：只在有明确 validation 改善时有限解冻 shared decoder，且必须保留 optimized-code control；
6. **held-out evaluation**：当前 LayerStack 分别测试未见连续状态与未见层数/拓扑；跨 family 只有在多个 family 已各自具备 adapter、资源语义和足量训练资产后才定义，不能把“不同前端输出相同维度 code”写成零样本跨 family 泛化。

latent L2 不能成为唯一 compiler loss，因为多个 latent 可能表示同一函数。最终 gap 始终在同一 decoder 的 query-space metrics 中计算。

### 9.5 pure feed-forward 与 bounded refinement

- pure feed-forward：`z=C(source)` 后不读取 reference response；
- compiler initialization + bounded refinement：只用新 state 的 query-train response 调整 `z`，validation 选固定预算内 checkpoint，test 独立；
- optimized/target-encoded control：使用相同 decoder，量化 compiler gap。早期可用于定位，里程碑结论则要求 decoder 自身已满足该轮的 promotion criteria。

三者使用不同 manifest role，不能把 bounded cook 结果写成 pure source generalization。

## 10. 不变量、scorecard 与晋级条件

### 10.1 三类判据不能混用

**硬不变量** 每轮都必须通过，失败表示结果无效：

- dataset contract、split、hash、train-only fitted state 与 provenance；
- `evaluate()` 返回 `f`、cosine 只在监督/积分中乘一次；
- chart finite、mode/validity、source range 与 reciprocity 测度语义；
- checkpoint/config 可恢复；进入 bundle 后的 Python/Slang parity；
- 已声明的固定循环、state bytes 与随机访问边界。

**研究 scorecard** 用来发现主问题，不压成一个总 pass/fail：

- solid-angle normalized L1、linear/log error 与 family/state 分布；
- peak ratio、peak-support angle、top-energy recall；
- reflection/transmission 分项积分能量；
- absolute model error、reference SE、replica disagreement 及其联合置信区间；
- high-resolution response slice、固定 HDR light/view sweep、参数编辑轨迹与共同曝光视觉对比；
- `B_asset/B_shared`、compile/refinement、`C_prepare/C_eval`、单次/多次 amortization；
- fp32/fp16、coherent/divergent tile、真实 GPU 时间、显存和带宽。

**晋级条件** 只回答一个具体问题，例如“B0 是否值得扩大 state corpus”“B1 是否替换 B0”“某 student 是否进入实时 Pareto”。它由该轮 claim 所需的质量、回归预算和成本共同定义，不要求所有研究 metric 在所有阶段同时过线。

### 10.2 阈值与比较方式

E0 replica、deterministic reference 和 fixed probes 用来解释可分辨误差，不直接生成一个统治所有 family/metric 的公式阈值。正式比较优先使用同 query 的 paired difference、bootstrap confidence interval、每 state 分布和 effect size；同时报告 absolute error 与 error/reference-SE。reference SE 接近零时，ratio 只作诊断，不作为唯一否决项。

每轮维护 regression budget：已经解决的能量、峰位、合法范围、连续 sweep、parity 或成本项不能在没有明确权衡说明时回退。视觉容差必须绑定具体 render probe、曝光和显示变换，不能从 response-space 数值主观推导。

### 10.3 结果命名

后续稳定文档不再把有限容量实验称为“上界”。统一使用：

- `baseline`：明确配置和预算下的基线；
- `optimized-code control`：同 decoder 下直接优化 material code 的对照；
- `high-capacity teacher`：用于定位或 distillation 的训练期模型；
- `best observed candidate`：当前证据集上观测到的最好候选；
- `promotion candidate`：满足某个明确晋级条件的候选。

历史 schema、manifest 或归档报告中的 `ceiling/upper bound` 字段为复现可以保留，但新报告要标注为历史命名，不把它延续为研究结论。这里不改变“运行成本有明确上界”这一工程含义：实时 backend 的单次执行、访存和状态大小仍必须静态有界。

## 11. 八类候选怎样接入，不做笛卡尔积

| 原候选 | 新设计中的位置 | 首个有意义的比较 |
|---|---|---|
| dense latent + small MLP | 保留为效率受限 baseline；B0 walking skeleton 的历史对照 | 与当前 B 系列 candidate 比较 quality/time/bytes |
| target-tensor encoder + shared decoder | response-chart attention encoder → 模型 B | encoder-only 与同 decoder optimized code |
| target encoder initialization + refinement | 同上，加固定预算 latent refinement | initialization gap 与 refinement gain |
| source-state compiler + shared decoder | typed source compiler → 模型 B | pure feed-forward 未见状态 |
| source compiler + bounded refinement | 同上，加 query-train bounded cook | compiler gap、cook time 与最终 fidelity |
| sparse latent dictionary / top-k mixture | 当 asset bytes 成为主问题时，对稳定 optimized `z` 做 train-only K-means/VQ | dense `z` 与 top-k/residual VQ 的 rate-distortion |
| analytic core + neural residual | 模型 A/B 的 paired output variant | direct 与 residual 的 core contribution/长尾差异 |
| plane/tensor factorization | 模型 C | canonical warp 后的 VM field 与模型 B |

实验不跨所有 candidate × level。B0 先打通纵向回环；模型 A、attention encoder、dictionary 和模型 C 分别只在监督/方向归因、target-code inference、资产 bytes 或 canonical 高频存储成为当前主问题时启用。source compiler control 与 Slang parity 不等待质量完美，quality compiler 与实时 Pareto 晋级则需要相应证据。

## 12. 闭环式执行路线

### Loop R0：固定语义与观测面

- 保留现有 E1–E3 run、checkpoint、hash 和失败指标；
- 把 small MLP、DeepSets、GRU compiler 结论限制为对应实现；
- 为新 evaluator 锁定 `f` 输出、loss 内 `response_cos`、chart validity 和 Python/Slang feature parity；
- 增加统一 failure ledger、per-state/query error slice、view/state sweep、成本与 regression budget 报告；
- 把当前 test 角色区分为 development evidence 与仍未读取的 sealed milestone set；不修改旧 gate 和历史 manifest。

failure ledger 的最小记录为：`issue_id`、数据/config/code hash、最坏 state/query/sweep、绝对与相对误差、所属层级、当前假设、唯一变化、matched control、回归项、结论与下一动作。它保存证据和决策，不是只列待办事项；一个问题只有在新增回归 probe 后才标记为已解决。

### Loop R1：B0 walking skeleton

1. 实现 raw + canonical chart、FiLM-conditioned shared MLP，先支持 LayerStack direct 与 analytic residual paired variant；
2. 在现有 E1/E2 合格 H5 上运行 optimized-code control，输出 development metrics 和 sweep；
3. 接入最小 LayerStack source compiler control，量化 pure feed-forward gap；
4. 导出一个可运行 MethodBundle/Slang 路径，验证 `f`、chart、fp32 parity、state layout 与 prepare/evaluate 成本；
5. 在 viewer 固定 directional-light capture 中确认误差表现与 Python slice 一致；
6. 不管质量是否达标，都形成第一份完整 failure ledger。

这一步的完成条件是回环可重复、失败可定位，不是所有 fidelity metric 通过。

### Loop R2：问题驱动迭代

每一轮固定使用下面的模板：

```text
观察：哪一组 state/query/sweep/运行时指标是当前主要失败？
归因：data/reference、direction、optimization、shared code、compiler 还是 deployment？
假设：哪一个最小机制能证伪该归因？
对照：只改变这一机制，保留 matched baseline 和 regression budget。
回环：train → development evaluate → compiler gap → Slang parity/cost → viewer slice。
记录：接受、拒绝或保留未知；把新失败固化成 probe/regression。
```

典型路由为：

- train 与 development 都丢窄峰：检查 chart/proposal，再按需启用 A0/A1；
- train 好、局部新 probe 差：增加 query coverage 或改连续表示，不继续盲目加宽；
- 单材质好、shared state 尾部差：B1 modulation 或扩大 state corpus；
- mode 竞争明显：B2 分头；多个峰被平均：B3 lobe；
- optimized code 好、source compiler 差：改 source adapter/compiler，不动 evaluator；
- quality 好、asset bytes 或 GPU cost 差：dictionary、student、量化或模型 C；
- Python 好、Slang/viewer 差：先修语义、布局或数值 parity，不回头调训练 loss。

现有 E2 的 20 个 source-train state 只用于 B0 回环和回归。任何关于 shared decoder 未见状态分布、compiler 泛化或 p90/p95 的里程碑结论，都必须先扩大独立 family/state 数并给出 bootstrap interval；不能用增加 query 数替代增加 source state 多样性。

### Loop R3：里程碑晋级与扩展

当一个候选在 development scorecard、连续 sweep、compiler gap 和部署 telemetry 上没有关键未知项时，冻结 config/code/data hash，运行多 seed 与 sealed test，决定它是 best observed candidate、Pareto 端点还是失败证据。sealed test 暴露的新问题进入下一轮时，必须更新最终 benchmark 治理，不能继续把同一 test 当作未见证据。

LayerStack 回环稳定后再按 source family 逐个接入 MERL/OpenPBR/MaterialX；每次新增 family 都重新检查 adapter/resource/颜色/事件语义与 failure ledger，不预设一个 decoder 天然跨 family。evaluator 与 compiler 的局部闭环成熟后，再按既定顺序扩展 matched sampler、环境/面光积分、spatial LOD 和系统工作流。

## 13. 配置、产物与报告位置

计划使用：

```text
configs/research/evaluator-loop-scorecard-v1.json
configs/research/r1-*.json
configs/research/r2-*.json

artifacts/research/learning-goal/evaluator-loop/
  audits/
  runs/
  comparisons/
  failure-ledger/
  sweeps/
  response-slices/
  captures/
```

正式 config 必须保存完整 pipeline contract、dataset selection、seed、direction chart、SH order、frame/lobe count、latent/modulation 结构、loss、proposal、scorecard 与 regression-budget hash。单次结果留在 `artifacts/`；稳定设计结论回写本文与 [`data_and_experiments.md`](data_and_experiments.md)。

不为新模型建立第二套 reader、runner、checkpoint 或 metric。它们必须通过 candidate-neutral registry/config 接入已有 lifecycle；模型专属逻辑留在 representation、model、family adapter 或 pipeline composition 中，不能继续堆进公共 runner。

## 14. 本轮确认的设计决策

后续实现按以下决策推进：

1. B0 shared evaluator、最小 source compiler control、MethodBundle/Slang 和 viewer evidence 先形成纵向回环；回环完整不等于质量通过；
2. 新 evaluator 直接返回 `f` 或 `Delta f`，`response_cos` transform 只存在于 loss/metric；
3. direct 模型 A 是按需诊断 teacher，不是 residual/shared 路线的统一前置 gate；
4. B0 只使用 canonical chart + layer-wise FiLM；low-rank modulation、mode heads 和 lobe token 按 failure ledger 逐项加入；
5. LayerStack compiler 先用 typed order-aware control 打通 code contract，再由拓扑/长程误差决定是否升级 graph Transformer；停止预测逐状态 target transform statistics；
6. target attention encoder、dictionary 和模型 C 只在各自对应的问题成为主瓶颈时实现，不做候选笛卡尔积；
7. 硬不变量、研究 scorecard 和晋级条件分开；运行成本从第一轮记录，但旧 65k-MAC/512-KiB gate 不提前否定新结构；
8. 新报告使用 `optimized-code control`、`teacher`、`best observed candidate` 等限定表述，不再追求或宣称有限实验“上界”。

## 15. 主要依据与迁移边界

- [Neural BRDF Representation and Importance Sampling](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14335)：支持 Rusinkiewicz coordinates、log-domain loss 和 specular-aware angular sampling；不证明 shared compiler 或 spatial LOD。
- [Neural Layered BRDFs](https://wangningbei.github.io/2022/NLBRDF.html)：支持高容量 residual evaluator、optimized BRDF latent 和 layering-space 学习；其 isotropic/layered 假设不能提升为所有 family 的接口。
- [MetaLayer](https://sites.cs.ucsb.edu/~lingqi/publications/paper_siga23metalayer.pdf)：支持 Rusinkiewicz spherical-harmonic encoding 与从 source parameters 生成部分 evaluator weights；完整 per-material weights 仍需与 shared modulation 的 coherence/bytes 比较。
- [Real-Time Neural Appearance Models](https://research.nvidia.com/labs/rtr/neural_appearance_models/)：支持 learned shading frames、source encoder → bake → refinement、evaluator/sampler 分头和 spatial LOD；本文先借用方向与 conditioning 机制，不提前声称 E5/E6 已成立。
- [An Adaptive Parameterization for Efficient Material Acquisition and Rendering](https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Dupuy2018Adaptive.pdf)：支持先把高光域 warp 到更平滑坐标；其采集/表格表示不是本文 runtime 合同。
- [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)：支持用 induced attention 表达无序点集间关系；它不是现成的 BRDF encoder，chart token 与 split 约束仍需本项目验证。
- [SIREN](https://www.vincentsitzmann.com/siren/)：支持周期激活表示高频隐式函数；本文只把它作为受控 activation ablation，不预设它优于物理 chart。
