# P1 v1 审计：容量、成本、长尾与部署路线的修正

本文是对 P1 v1（`experiment_log.md` 中 2026-08-25 的七个正式 run 与 `film-m1-direct-neural@1` 部署）的代码级审计。它回答四个问题：P1 实际放宽了哪些约束；当前模型和 Slang 实现离实时目标有多远；质量差、长尾差的原因在代码里对应什么；按 [`prior_art.md`](prior_art.md) 已整理的结构，evaluator、sampler 与工程接口应改成什么形态。审计结论分三类标注：**[代码事实]** 可直接在仓库中核对；**[实测]** 来自 `experiment_log.md` 登记的远程机结果；**[假设]** 由代码推断、尚未在远程机验证，附验证方法。

## 1. 一句话结论

P1 v1 把 E2 的 concat 小 MLP 放大为 256 宽、9 个 residual block 的共享 trunk，并解除了研究期成本预算。它换来的是 test median `0.053 → 0.045`、p95 `0.169 → 0.118` 的边际改善（E2 与 P1 数据不同，只作量级对照），代价是单次 `evaluate` 约 `8.6e5` MAC，比 [Real-Time Neural Appearance Models 2024] 的 decoder 高约两个数量级；部署实现又比同 MAC 数的算力上限慢约两个数量级。同时 M2（解析 core + 残差）median `0.018` 这一明显更强的信号被 p95 爆炸掩盖，而该爆炸有很大可能来自一个梯度死区，不是 core 路线本身的问题。

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

- 标量 fp32，每个 MAC 单独从 `StructuredBuffer<float>` 读权重，3.4 MB 权重在 lane 间没有任何共享；
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

### 4.2 M2 的 p95 爆炸疑似梯度死区 [假设，可验证]

`P1EvaluatorPipeline.predict_f` 对 M2 返回 `torch.clamp(core + prediction, min=0.0)`，`training_loss` 再次 `torch.clamp(prediction_f, min=0.0)`。`core + Δ < 0` 的 query 梯度恒为 0：某个 state 的残差一旦被推负就永远回不来。三档 M2 全部在 5.5k–7.5k 步早停、median 极好而 p95 极差，与「多数 query 拟合很好、少数 state 整块死掉」一致。core 只取 `direct_top`（顶层界面），对顶层为 dielectric coat 的多层栈，core 反射很小、残差要承担整个 base；signed 输出加死区最容易在这些 state 上失效。

`experiment_log.md` 据此把 core 路线降为「机制对照」，本文认为该结论不成立。验证方法：加载 M2-S early-stop checkpoint，统计 train/test query 中 `core + Δ < 0` 的比例，并按 state 与 directional L1 做散点；若失败 state 集中在高死区比例上，则 p95 结论作废。

### 4.3 log 域 loss 与带噪 reference 的系统性偏暗 [假设，可验证]

主损失 `smooth_l1(log(pred/s + 1e-4), log(target/s + 1e-4))`，train reference 每 group 相对 SE 允许到 0.25。对右偏的 Monte Carlo 估计，log 域或中位型损失的最优解低于均值（Jensen 不等式），偏置约 `−σ²/2`，6%–25% 噪声下即 −0.2% ~ −3%。`capture-display.png`（左 reference、右 M1-M，四层 diffuse stack）右侧整体偏暗、饱和度略低而没有局部结构错误，正是全局偏置的样子；一个响应近乎常数的 state 用 `8.6e5` MAC 仍留 2.9% L1，最合理的解释是偏置而非容量。`quality.py` 的能量误差取绝对值，看不出符号。验证方法：在 quality 报告中增加 signed `Σ pred / Σ target` 按 state 汇总；若系统性 < 1，改用 linear 域主损失或 debias 的 log 损失。

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

1. 远程机验证 §4.2 与 §4.3：M2-S early-stop checkpoint 的 `core + Δ < 0` 比例按 state 统计；quality 报告增加 signed 能量比。两项结果决定 M2 结论是否推翻、主损失是否改域。
2. 恢复研究期成本线：至少「`evaluate ≤ 1e4` MAC 量级才能进注册表」；§5.1 形态（精确顶层界面 + 无死区残差 + 匹配 sampler）作为 P1 重跑主候选，§5.2 只作同预算对照。`B_asset` 记账改为部署实际 bytes。
3. 单一 Slang 后端骨架：先把 legacy-ltc-k2（已实现 `sample/pdf` 与合同）迁到 SlangPy 训练路径打通闭环，再接新模型；viewer 增加「PT + neural material」comparison mode。
4. 把 §2 中的放宽项回写进 [`experiment_framework.md`](experiment_framework.md)：成本线、signed 能量诊断、p95 所需 state 数与 P1 子集规模的一致性。
