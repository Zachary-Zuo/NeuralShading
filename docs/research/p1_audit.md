# P1 v1 审计：容量、成本、长尾与部署路线的修正

> 历史记录：2026-09-05 架构重置前的命令、格式和结果只用于理解当时实验，当前接口见 docs/learning.md。旧权重不迁移、不兼容读取，旧 viewer 图像原地保留。

本文是对 P1 v1（`experiment_log.md` 中 2026-08-25 的七个正式 run 与 `film-m1-direct-neural@1` 部署）的代码级审计。它回答四个问题：P1 实际放宽了哪些约束；当前模型和 Slang 实现离实时目标有多远；质量差、长尾差的原因在代码里对应什么；按 [`prior_art.md`](prior_art.md) 已整理的结构，evaluator、sampler 与工程接口应改成什么形态。审计结论分三类标注：**[代码事实]** 可直接在仓库中核对；**[实测]** 来自 `experiment_log.md` 登记的远程机结果；**[假设]** 由代码推断、尚未在远程机验证，附验证方法。

## 1. 一句话结论

P1 v1 把 E2 的 concat 小 MLP 放大为 256 宽、9 个 residual block 的共享 trunk，并解除了研究期成本预算。它换来的是 test median `0.053 → 0.045`、p95 `0.169 → 0.118` 的边际改善（E2 与 P1 数据不同，只作量级对照），代价是单次 `evaluate` 约 `8.6e5` MAC，比 [Real-Time Neural Appearance Models 2024] 的 decoder 高约两个数量级；部署实现又比同 MAC 数的算力上限慢约两个数量级。同时 M2（解析 core + 残差）median `0.018` 这一明显更强的信号被 p95 爆炸掩盖；远程机冻结推理审计（`artifacts/audits/p1-v1/`）证实该爆炸集中在 4 个 core 覆盖不足、残差被 `clamp` 大面积截断的多层 state 上，而 core 精确的 15 个单层 state 上 M2 的 L1 ≤ `0.013`、M1-S 为 `0.006–0.126`。问题在残差参数化，不在 core 路线。

## 2. P1 v1 实际放宽了什么

| 事项 | 位置 | 后果 |
|---|---|---|
| 解除研究期 `C_eval ≤ 65k MAC`、`B_shared ≤ 512 KiB`、`B_asset ≤ 512 B` | commit `8e517e3`，[`experiment_framework.md`](experiment_framework.md) §0 | S/M/L 档 0.2M / 1.3M / 6M 参数；[`model_candidates.md`](model_candidates.md) §1.5 自述「旧 65k MAC 预算约等于 S 档以下」，即最小档已超过旧预算 |
| 废除 `model error / reference SE ≤ 6` gate | 同上 §5.4 | 长尾不再有任何硬门 |
| 参考线 `0.05 / 0.15 / 0.03` 定义为「参考，不是 kill gate」 | `configs/evaluation/quality-v1.json` | M1-S median `0.0506` 未达线仍登记为「效率优先默认」 |
| train reference 相对 SE 上限 0.25 | §2.3 | 训练目标最多含 25% 噪声，见 §4.3 |
| 单 seed、30 个 state 上报 p95 | `experiment_log.md` seed 决策 | 30 个 state 的 p95 就是第 2 差的 state；框架 §2.5 自身要求 test state ≥ 50 |
| partition = `target-visible-v1` | `src/ncls/learning/pipelines/p1_evaluator.py` descriptor | 30 个 state 全部参与训练（autodecoder latent），P1 只考方向泛化 G1 |

最后一条决定了 P1 数字的含义：**这是所有材质都见过、仅对未见方向泛化的拟合实验**。在这个设定下 `8.6e5` MAC 的网络仍留 4.5% median / 12% p95，而 [NBRDF 2021] 用 `6→21→21→3`（约 `6e2` MAC）逐材质拟合 MERL 的误差远低于此。放宽约束本身不产生容量；它只是让「长尾没有被表示」这一事实不再触发任何判定。

## 3. 当前模型与实现的成本

### 3.1 静态成本 [代码事实]

按 `src/ncls/learning/models/p1_evaluator.py` 与 `shaders/ncls/backends/film_m1/film_m1.slang` 的常量计算：

| | `evaluate`（每 `wi`） | `prepare`（每像素） | 每像素 state | 每材质部署 bytes |
|---|---|---|---|---|
| legacy-ltc-k2（`shaders/ncls/backends/legacy_ltc_k2/`） | 1 次顶层 microfacet + 2 个 LTC lobe ≈ `2e2` FLOP | GRU compiler w=64、4 层 ≈ `1.3e5` MAC（可离线） | 176 B | 176 B |
| M1-S | `162×128 + 3×2×128²` ≈ `1.2e5` MAC | ≈ `8e4` MAC | 512 B | latent 32 float；bundle 实际烘焙 condition 1408 float ≈ 5.6 KB |
| M1-M（已部署） | `294×256 + 6×2×256²` ≈ `8.6e5` MAC，另加 7 次 LayerNorm、1536 次 erf-GELU | ≈ `4.6e5` MAC | 1 KB | latent 64 float；bundle 实际烘焙 condition 4864 float ≈ 19.5 KB |
| [Real-Time Neural Appearance 2024]（论文事实） | 2–3 层 64 宽、FP8/FP16 cooperative vector，约 `1e4` MAC 量级 | 极小 | latent texel 十几 float | latent texture |
| [NBRDF 2021]（论文事实） | `6→21→21→3` ≈ `6e2` MAC | 0 | — | 675 参数 |

`B_asset` 在 `parameter_costs()` 中只计 latent（M1-M 为 256 B），但 `src/ncls/bundle/film_m1.py::_serialize_runtime_weights` 把 `condition(latent)` 整体烘焙进 bundle，viewer 实际加载的每材质数据是 19.5 KB。成本记账与部署形态不一致。

### 3.2 实测成本与两个 `×100` [实测 + 代码事实]

M1-M 在 RTX 4090、320×240 输出（方法半屏 160×240）、单方向光下 prepare `82.52 ms`、lighting `112.72 ms`，即每个 pixel-light 查询约 2.9 µs。按 38k 像素 × `8.6e5` MAC ≈ 33 GMAC，4090 fp32 算力下计算时间不到 1 ms。差出的约两个数量级来自 `film_m1.slang` 的实现方式：

- 标量 fp32，每个 MAC 单独从 `StructuredBuffer<float>` 读权重，5.3 MB 共享权重（1,333,251 个 float32）在 lane 间没有任何共享；
- `float input[294]`、四个 `float[256]` 局部数组必然溢出到 local memory；
- 每个 block 两次全向量 LayerNorm 归约；GELU 使用 erf 近似（每神经元十余条指令）；
- 环境光路径对每像素执行 `gEnvironmentQueryBudget` 次完整 evaluate（`apps/viewer/shaders/Approximation.cs.slang`）。

结论：模型比应有规模大约 `100×`，实现又比该规模应有速度慢约 `100×`。两者都要改，但前者是根本；只优化实现仍剩 `8.6e5` MAC/query，无法进入实时 Pareto。

## 4. 质量差的原因

按证据强度排序。

### 4.1 「FiLM」条件化实际被钳到 ±10% [代码事实]

`FiLMResidualBlock.forward`：

```python
residual = (1.0 + 0.1 * torch.tanh(gamma)) * residual + 0.1 * beta
```

γ 只能把残差缩放到 `[0.9, 1.1]`，β 缩 0.1。[`model_candidates.md`](model_candidates.md) §2.2 批评 E2 concat「只在第一层线性注入 latent」，而 M1 的实际条件化路径是：context 拼进 prepare 输入（concat）、`prepared` 拼进 evaluate 输入（concat），外加一层几乎不动的 ±10% 调制。30 个异质 state（diffuse 栈、窄各向异性导体、sheen、多界面）共用一条近乎相同的 trunk，只能靠撑大 trunk 来「装下」全部材质。这与 M→L 反而变差、S→M 置信区间跨零的形态一致：E2「该条件化机制到头」的结论被原样复现了一遍。

### 4.2 M2 的 p95 爆炸：clamp 死区 × core 覆盖不足 [实测，2026-08-25]

`P1EvaluatorPipeline.predict_f` 对 M2 返回 `torch.clamp(core + prediction, min=0.0)`，`training_loss` 再次 `clamp`；`core + Δ < 0` 的 query 梯度恒为 0。远程机对冻结 checkpoint 的推理审计（`artifacts/audits/p1-v1/mechanism-audit.json`，report `c78f8951…`；`supplemental-test-audit.json`，`3a7d4332…`；远程结论 `conclusions.md`）给出：

| test state 分组（M2-S best，step 4500） | state 数 | 残差被 clamp 的 RGB 比例 | directional L1 | 能量比 Σpred/Σref |
|---|---:|---|---|---|
| 单层（direct-top core 精确，`E_core/E_ref = 1`） | 15 | 0–0.97（无害：core 已正确，残差被推负后截掉） | ≤ `0.013` | ≈ 1 |
| 多层、死区比例 = 0 | 11 | 0 | `0.023–0.33`，中位 `0.038` | ≈ 1 |
| 多层、死区比例 > 0 | 4 | `0.44 / 0.88 / 0.97 / 0.98` | `0.65 / 0.67 / 0.43 / 0.51` | `0.35 / 0.33 / 0.58 / 0.49` |

- 四个死区 state 就是 M2-S 最差四个 state，`E_core/E_ref` 为 `0.17–0.54`；在 15 个多层 state 内「死区比例 > 0」与「最差 4 名」完全重合（精确秩检验 `p = 1/C(15,4) ≈ 7e-4`）。M2-M/L best 上是同样 4 个 state、同样完全分离。
- 远程结论 §3 按全部 30 个 state 算 Spearman（`ρ = 0.15`，CI 跨零）判死区「不是全局主因」；该检验被 15 个单层 state 混杂——那里死区比例很高但无害。按 core 覆盖分层后，死区就是 p95 尾部的机制。远程结论中 `E_core/E_ref` 与 L1 的 `ρ = −0.77`（CI `[−0.88, −0.51]`）与此一致：core 覆盖越低，signed 残差要承担的能量越多，被 clamp 截断的后果越重。
- 同一 run 继续训练到早停（step 7500）后四个 state 的死区比例降到 `0.04–0.93`、L1 降到 `0.28–0.37`，p95 从 `0.586` 降到 `0.340`——梯度只能从尚未被截断的方向缓慢回流，与死区机制一致；「validation median 优先、p95 仅决胜」的 checkpoint 规则把更早、尾部更差的 step 4500 登记为 best。
- 同样 4 个 state，M1-S 的 L1 为 `0.06 / 0.06 / 0.09 / 0.26`：多层 base 能量并非小网络表示不了，是 `clamp(core + signed Δ, 0)` 这一参数化让它表示不出来。

结论：`experiment_log.md` 原「解析 core 只改善多数 state、显著损害困难尾部」不成立；成立的是「direct-top core 精确处 M2 几乎零误差，core 覆盖不足处 signed 残差 + clamp 死区失效」。冻结的 p95 观测保留为该实现的事实，不能被引用为 core 路线的证据。修正见 §5.3。

### 4.3 系统性偏暗：M1-M 确认约 0.8%，loss 因果未确认 [实测，2026-08-25]

主损失 `smooth_l1(log(pred/s + 1e-4), log(target/s + 1e-4))`，train reference 相对 SE 上限 0.25（两个 promotion state 为 0.75）。对右偏的 Monte Carlo 估计，log 域最优解低于均值（偏置约 `−σ²/2`）。远程审计的 signed 能量比 `R = Σ(pred·|cos|·w) / Σ(ref·w)`：

| checkpoint（test） | state-median `R` | `R < 1` 的 state 比例 | median log `R` 95% CI | 全局 Σ/Σ |
|---|---:|---:|---|---:|
| M1-S best | `1.0001` | 0.50 | `[−0.0024, 0.0012]` | `1.0033` |
| M1-M best | `0.9921` | 0.80 | `[−0.0123, −0.0045]` | `0.9939` |
| M1-M last | `0.9937` | — | — | `0.9972` |

M1-M 五个 role 全部系统性偏暗 0.6–1.1%，且与 reference SE 无关（`ρ = 0.05`，CI 跨零）；viewer 所用四层 diffuse state `6324e3…` 的 `R = 0.9935`。但 M1-S 用同一 loss、同一数据没有偏置，所以现有证据不能把原因归给 log loss 或 Jensen 偏差，也不能把 `capture-display.png` 的整体偏暗归结为全局偏置（0.65% 能量差解释不了该 state 2.9% 的 L1）。本轮不换主损失；新候选从第一轮起报告逐 state / 逐通道 / 逐 role 的 signed 能量比，loss 保留显式 linear 能量项，log 是否 debias 由新模型自己的 matched 对照决定。

### 4.4 direct 路线重蹈 E1 的失败 [代码事实 + 实测]

E1 已记录「极窄 conductor 的 direct 小 MLP 丢峰」（[`model_candidates.md`](model_candidates.md) §9）。半宽 < 2° 的峰在任何平滑坐标下都不适合由 MLP 表示；LayerStack 是参数式族，顶层 GGX 的 `α` 是已知量，legacy-ltc-k2 直接精确求值。P1 选择放大 direct 网络去追这些峰，长尾 p95 停在 0.12 是这一选择的自然结果。

### 4.5 Fourier(log-slope) 对平滑材质是纯噪声输入 [代码事实]

M 档 5 个 band，最高频 `16π` / 单位 log-slope。对 diffuse 栈，网络必须学会对全部 38 个方向特征（含高频正弦）保持不变，既浪费容量又产生低幅波纹。[Real-Time Neural Appearance 2024] 依靠 learned frame 让峰驻定，[NBRDF 2021] 依靠 Rusinkiewicz 参数化与 log 输出，都不依赖高频 Fourier 编码。

## 5. 应采用的建模形态

[`prior_art.md`](prior_art.md) §3.5/3.6 与 [`model_candidates.md`](model_candidates.md) M2 已经指出方向，P1 没有执行。以下是修正后的候选优先级。

### 5.1 主线：prepare 输出 lobe 参数，evaluate 解析

即 [Neural Material Adapter 2026]、[Hybrid Neural-Microfacet 2026] 与本项目 legacy-ltc-k2 共同的形态：

```text
prepare(z, wo):  ≤64 宽、2–3 层的小网络 → K 个 lobe 参数
                 lobe 0 = 顶层界面精确 microfacet（参数来自源材质，不学习）
                 lobe 1..K = GGX / LTC / cosine 型残差 lobe（α_eff、albedo、能量、frame）
                 + 可选乘性 log 修正 exp(Δ)，Δ 由 ≤32 宽网络给出
evaluate(h, wi): 解析求和，约 1e2–1e3 FLOP
sample/pdf:      lobe 混合的 VNDF / cosine 采样，pdf 精确匹配，随 evaluate 免费得到
```

对四层 diffuse 栈它退化为一个 cosine lobe，对窄导体是精确 GGX，对多界面栈是 2–3 个 lobe。E1 在 65k MAC 内以此形态达到多界面 state test median `0.046`（迁移前数据、单材质，与 P1 的 30 state 不可直接比较；但 evaluate 成本相差约 `4000×` 是确定的）。

### 5.2 备选：NVIDIA 规格的 direct neural evaluator

若论文 claim 必须保留 direct neural evaluator，其规格应对齐 [Real-Time Neural Appearance 2024]：2–3 层 × 32–64 宽、ReLU/leaky ReLU（RTXNS 形态，不用 LayerNorm 与 erf-GELU）、FP16/FP8 + cooperative vector、learned frame 或解析 warp 前端、log-L1 + mollification；latent 用真正的 FiLM 或低秩调制，不钳 ±10%。它必须与 §5.1 在同 bytes / 同时间下配对报告，不能单独存在。

### 5.3 残差参数化不得含死区

用 `f = core·exp(Δ₁) + softplus(Δ₂)` 或 `core·(1 + tanh(·))` + 非负残差 lobe；禁止 `clamp(core + Δ, 0)`。

### 5.4 sample/pdf 先于 deferred 验证

`shaders/ncls/contracts/scattering_backend.slang` 的 `INclsScatteringState` 已含 `sample/pdf`，`legacy_ltc_k2.slang` 已实现（cosine proposal，pdf 诚实匹配）。neural backend 只要走同一接口，`apps/viewer/shaders/ReferencePathTracer.cs.slang` 就可以把它作为第 5 个 family：**左侧 PT + 源材质 reference，右侧 PT + neural material，同积分器、同灯、同反弹深度**，差图只剩表示误差与 Monte Carlo 噪声。这比当前 deferred-vs-PT 的对比（[`../viewer_spec.md`](../viewer_spec.md) 自述混入 GI 与可见性差异）干净得多，也是 evaluate 进入 deferred 前应有的正确性检查。§5.1 的 sampler 是精确的；即便 §5.2 形态，先用 cosine 或顶层 GGX proxy 做 proposal 也已是无偏估计。

## 6. 单一 Slang 后端：训练、评测、viewer 共用

### 6.1 现状：一个模型写四处 [代码事实]

1. `src/ncls/learning/models/p1_evaluator.py`：Torch 前向（289 行）；
2. `shaders/ncls/backends/film_m1/film_m1.slang`：手写 Slang 复刻（318 行），权重偏移常量手工同步；
3. `src/ncls/bundle/film_m1.py`：手写权重序列化与 layout（397 行）；
4. `apps/viewer/MethodBundle.cpp:76-146` 硬编码 `"film-m1-direct-neural"`、entry 名、`architecture_id`、`ncls.film-m1-weights`；`Prepare/Approximation/Parity.cs.slang` 直接调用 `nclsFilmM1*` 自由函数，**绕过 `INclsScatteringBackend`**——只有 legacy 后端和 `tests/gpu/kernels/legacy_ltc_k2.cs.slang` 走合同。

另有 `src/ncls/core/representations/legacy_ltc_k2/torch_eval.py`（359 行）是 `shaders/ncls/reference/interfaces.slang` 微面元代码的 Torch 复刻。每换一个模型四处重写，parity 测试因此成为必需品。reference 侧则已经是单一 Slang 源：`random_walk_reference.slang` 同时服务数据采集（Falcor Python）与 viewer。缺的只是把模型侧纳入同一模式。

### 6.2 目标结构

每个模型一份 Slang：

```text
shaders/ncls/backends/<name>/<name>.slang
  struct Params        { DiffTensorView weights…; DiffTensorView latent; }   // 训练可微
  struct CompiledMaterial / State
  struct <Name>Backend : INclsScatteringBackend
      State prepare(ctx, material)                    [Differentiable]
  struct State : INclsSampleableScatteringState
      evaluate(wi)  [Differentiable]
      sample(ξ) / pdf(wi)
```

- **训练与评测**：SlangPy 加载同一模块，按 `(state, wo, wi)` 行批量调用 `evaluate`，`bwd_diff` 得到对 weights/latent 的梯度，张量即 `torch.Tensor`；loss 与 optimizer 留在 Torch，`training/runner.py` 只把 `pipeline.predict_f` 换成 Slang 调用。RTXNS 的 SlangPy 训练示例即此模式，其 Slang `MLP` / CoopVec 模块可直接作为 §5.2 形态的实现。
- **viewer**：Falcor 侧 `#include` 同一文件；`Prepare/Approximation/Parity` 三个 pass 改写为对 `INclsScatteringBackend` 的泛型，由 bundle 声明的 module 路径与类型名做 specialization；`MethodBundle.cpp` 不再硬编码任何后端字符串。
- **权重布局**：由 Slang `Params` 结构反射生成（SlangPy 提供 layout），删除手写 exporter；bundle 只存张量与反射出的 layout。
- **parity**：退化为「SlangPy 编译 vs Falcor 编译同一源」的数值探针。仍值得保留一个：Falcor 8.0 锁定 Slang 2024.1.34，SlangPy 携带更新的 slang，后端源码要避开仅新版可用的特性，或升级 Falcor 的 slang。

### 6.3 风险与适用范围

SlangPy autodiff 逐线程执行。对 ≤64 宽的小网络吞吐充足（RTXNS 以此训练）；对当前 256 宽的 M 模型会慢于 cuBLAS——但该模型本不应存在。迁移验证需在远程 GPU 主机完成。

## 7. 下一步与验证项

1. §4.2 与 §4.3 已由远程机冻结推理审计落定（`artifacts/audits/p1-v1/`，工具 `ncls learn audit-p1`）：M2 尾部由 clamp 死区 × core 覆盖不足解释，core 路线保留；M1-M 偏暗确认但 loss 因果未证，主损失暂不改域。由此固定的诊断与规则：signed 能量比、`E_core/E_ref` core 覆盖、achieved reference SE（group p95 与 integrated ratio）进入每份评测报告；checkpoint 选择加 tail guard（[`experiment_framework.md`](experiment_framework.md) §4.2）；30-state p95 只作 selection 诊断，正式 tail 结论用 ≥ 50 个 test state 并附 bootstrap CI 与 leave-one-state-out 范围。当前 P1 v1 不再重训。
2. 研究期成本线已按 [`experiment_framework.md`](experiment_framework.md) §0.1 恢复（标量路径 `C_eval ≤ 2e3` MAC、`C_prepare ≤ 1e4`、state ≤ 64 B、`B_asset` 含烘焙参数、`evaluate` 权重 ≤ 32 KB），硬线写入 [`../contracts/method_bundle.md`](../contracts/method_bundle.md)。§5.1 形态（精确顶层界面 + 无死区残差 + 匹配 sampler）作为 P1 重跑主候选；§5.2 只在 cooperative vector 工具链于远程机验证后作同预算对照。`parameter_costs()` 与 bundle `cost_claims` 的 `B_asset` 记账改为部署实际 bytes。
3. 单一 Slang 后端骨架：先把 legacy-ltc-k2（已实现 `sample/pdf` 与合同）迁到 SlangPy 训练路径打通闭环，再接新模型；viewer 增加「PT + neural material」comparison mode。
4. 把 §2 中的放宽项回写进 [`experiment_framework.md`](experiment_framework.md)：成本线、signed 能量诊断、p95 所需 state 数与 P1 子集规模的一致性。
