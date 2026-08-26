# 03 Neural Baseline 与候选选择设计

## 1. 所有权与数据流

```text
02 frozen training entry (47ef…5a89)
  ├─ base v5 corpus (0513…64b7) ─ validation/test/dense + late training
  └─ mollification corpus (f693…e4f3) ─ early curriculum training
                         ↓
           UnifiedScatteringTrainingStore
                         ↓
  single Falcor-free Slang core ← SlangPy autodiff / Falcor GPU probe
       prepare → evaluator → sampler pdf
                         ↓
      evaluator checkpoints + sampler-head checkpoints
                         ↓
         packed compiled materials + matched 2×2 report
                         ↓
                 04 generic MethodBundle runtime
```

`src/ncls/learning/` 拥有训练生命周期、loss、checkpoint、比较与 selection；`src/ncls/data/` 继续独占 data-entry/corpus 校验。`shaders/ncls/backends/unified_neural/`（最终命名可按现有组织规范收敛）拥有唯一生产前向，只依赖 `shaders/ncls/scattering/` 和稳定 contracts。03 不让 bundle/viewer 解释模型私有 state。

## 2. 方法身份、参数与特征

### 2.1 CompiledMaterial

不同方法保留自己的原生形态，不再为了“同 bytes”伪造统一私有布局：

| 方法 | material code | 额外内容 | 说明 |
|---|---|---|---|
| NVIDIA 原规模 baseline | latent `z8` | version / flags 与必要的过滤元数据 | frame 由 `z8` 提取；不保留被忽略的 top core，也不扩成 `z16` |
| `core-frame-neural-v1` | 候选声明的 latent | exact top-interface core、normalization / flags | core 来自 LayerStack 原生参数，不由 neural 反演 |

每个 backend 的 `CompiledMaterial` / `ScatteringState` 都是私有布局，并分别记录真实 bytes。matched 比较冻结数据、角色与评测协议，不以填充字段强行制造相同成本。shared decoder 对未见 state 做 offline cook 时只更新该方法声明可拟合的 material code；test target 始终不可见。

### 2.2 prepare

NVIDIA baseline 的 frame extractor 是原方法的一部分：`z8 → 无 bias/activation 的 Linear(8,12)`，每 6 个输出按 supplemental Listing 2 构造 `N=normalize([s0,s1,s2+1])`、`T=normalize([s3+1,s4,s5])`、`B=cross(N,T)`。原文明确说明 `(T,B,N)` 不正交化。frame 只依赖 material latent，不能依赖 `wo`。`prepare(wo)` 负责加载/过滤 `z8`、构造两个 frame、把 `wo` 变换到两个 frame，并缓存 evaluator 与 sampler 可复用的数据；这只是把原方法算术沿运行时生命周期重排，不改变函数。

baseline sampler 以 `z8 + wo.xyz` 为输入，经原规模 `11→32→32→32→9` head 解码 proposal 参数。它不复用会改变原方法函数族的 view-conditioned shared trunk。若后续要研究 shared trunk 或 view-conditioned frame，必须注册为独立 adaptation。

`core-frame-neural-v1` 可有自己的 prepare trunk、latent 和 state，但必须在对应 method-correspondence 中冻结，不与 NVIDIA baseline 共用名称或假装参数匹配。所有方法都满足静态有界、`prepare()` 不消费方向随机数、同一着色点可复用 state 的公共合同。

### 2.3 evaluator

NVIDIA 原规模 direct evaluator 的输入和网络固定为：

```text
z8
+ wi 在两个 latent-conditioned learned frame 中的坐标 6
+ wo 在相同两个 frame 中的坐标 6
= 20
→ Linear/ReLU 64 → Linear/ReLU 64 → Linear/ReLU 64 → Linear 3 → exp(raw - 3)
```

忠实 core 输出是线性 RGB cosine-weighted response。方向映射为 `paper wi ← project wo`、`paper wo ← project wi`，所以监督量正好是 corpus 的 `f·project_wi.z`。公共 backend adapter 再返回裸 `f`，并锁定 `backend.evaluate(project_wi)*project_wi.z == response_core`；grazing 除余弦策略是显式 adapter 合同。自定义 half-vector slope、额外 dot feature、view-conditioned code、`softplus` 或最低输出比例都不属于该 baseline；需要它们时使用独立 adaptation ID 和 matched ablation。

`core-frame-neural-v1` 的 exact-core positive residual 输入可按候选自身设计使用：

```text
view code 5
+ wi 在两个 learned frame 中的坐标 6
+ top-interface half-vector slope 2
+ wi.z 1
+ log1p(f_top / response_scale) RGB 3
= 17
```

输出固定为：

```text
f_hat = f_top + softplus(scale_raw) * softplus(g_theta(x))
```

`f_top`复用LayerStack top interface的公共Fresnel/GGX数学；它是完整非相干路径响应的非负子集。不得对最终输出clamp，也不得把residual改为LTC evaluator。

### 2.4 成本分类与 viewer

原规模 baseline 只有一个方法身份。普通标量 Slang、SlangPy、Falcor 和 viewer include 同一 core；实际 scalar MAC、bytes 和 viewer 时间均从该实现测量。若它超过当前软成本线，descriptor 如实写 `deployment_candidate=false`，MethodBundle/UI 标出对应 runtime class，但 exporter/loader/viewer 不得拒绝它，也不得静默换成缩模 checkpoint。

缩模网络若以后需要，使用独立 pipeline/backend/config/hash，并与原规模 baseline 做明确的容量—成本实验；它不参与本任务的复现验收。

## 3. Sampler 设计

### 3.1 NVIDIA GGX9

```text
p(wi) = ε p_cos(wi)
      + (1-ε)[softmax(a)_d p_tilted_cos(wi; μd)
              + softmax(a)_s p_noncentered_ggx(wi; wo, αx, αy, ρ, μs)]
ε = 1/32
```

9 raw参数通过公共NVIDIA range warp解码。GGX NDF half-vector reflection的below-surface结果返回显式null；连续upper-hemisphere PDF积分加null mass等于1。component选择使用`u.x`，二维方向用独立`u.yz`；返回有效方向后重算完整mixture PDF。

### 3.2 LTC-K2

```text
q(wi) = ε q_cos(wi)
      + (1-ε)[sigmoid(a) q_ltc0(wi) + (1-sigmoid(a)) q_ltc1(wi)]
```

两个LTC各用正inverse scale、有限shear和rotation的非奇异变换，复用`01`的push-forward sample/pdf。LTC保持upper hemisphere，不产生null。参数数量为13，仍落在统一64 B state内。

### 3.3 数学验收协议（在正式sampler结果前冻结）

对selection的30 states、每state至少四个含grazing的`wo`：

- deterministic upper-hemisphere quadrature重建连续PDF；NVIDIA另从生成half-vector统计null mass。`|continuous_integral + null_mass - 1| ≤ 5e-3`。
- 262,144个固定seed samples；所有continuous sample满足`sample.pdf == pdf(sample.wi)`，`rtol≤2e-5, atol≤1e-7`，invalid/null三态计数与解析概率一致。
- equal-solid-angle 128-bin histogram与解析bin mass的total variation `≤0.03`，并保存逐state/wo worst case。
- 对同一冻结evaluator，以8,192-direction dense deterministic integral为中心，64个独立replica、每replica 16,384 samples做MC；RGB standardized error `|mean-reference|/SE ≤3.5`，且family-wise 99% bootstrap interval覆盖reference。此门只证明sampler/PDF无偏，不混入evaluator对source reference的误差。

## 4. 训练生命周期

### 4.1 数据路由

pipeline只接受`02`的training-entry JSON。reader先验证entry/base/supplement identity，再暴露：

- train：前`M=20,000` evaluator steps使用`progress=(step-1)/(M-1)`调`MollificationCurriculumStore`；`.875`后reader自身返回base-v5 0° target。`M`之后使用base v5 train groups。
- validation/test/adversarial/dense：始终从base v5对应role读取，不参与scale拟合或checkpoint优化之外的泄漏。
- curriculum batch显式带`target_source/radius/progress`，checkpoint/run manifest统计每种来源数量。

训练早期supplement的8×64 anchors是`02`对在线cone query的版本化离线适配；不与base batch混合后伪称连续schedule。

### 4.2 evaluator stage

- NVIDIA baseline 的正式配置先由 method-correspondence 审计冻结原规模结构、optimizer、joint evaluator/sampler schedule、response log-space L1 和 mollification。论文报告 300k iterations、每 iteration 两个 65k batch、FP32 master/FP16 inference；本项目离线 corpus 无法冒充相同的在线 40B sample 数据量，因此训练预算适配单列记录。smoke 可以缩短步数验证生命周期，但不能作为复现结果；此前 20k/25k 且自定义 loss/feature 的 run 只保留为实现诊断。
- 若 LayerStack noisy reference 需要 linear/energy/peak 等额外 loss，使用独立 adaptation config，并与原 loss 做相同数据/seed 的 ablation；不能悄悄修改 baseline loss 后仍称忠实复现。
- `core-frame-neural-v1` 使用自己的已登记 loss 与 optimizer 方案。不同方法可以有不同架构与训练目标，但正式比较使用相同 data entry、角色隔离和总预算类别，并完整记录各自训练成本。
- 正式复现默认采用预先冻结的主 seed。只有轨迹异常或用户明确要求时才追加 seed；不得在看到 test 或其他方法结果后用额外私有调参预算追逐质量。已完成的第二个 baseline seed作为补充复现证据保留，不把相同重复预算扩散到 candidate 与 sampler。

### 4.3 sampler stage

这里分成两个不同身份，不能混用：

1. **NVIDIA reproduction joint stage**：evaluator、latent 与 GGX9 sampler 同时训练；BRDF response 的 log-L1 更新 evaluator/latent，sampler KL 更新 sampler head，KL 对 latent detach，当前 evaluator target 不反向改变 evaluator。
2. **matched sampler comparison stage**：每个 evaluator 的 best checkpoint 冻结后，重新训练 GGX9/LTC 两个 head；evaluator、latent、frame全部冻结。该 stage 为公平 sampler 轴而存在，checkpoint/manifest 标为 adaptation，不替代 joint reproduction。

matched stage 的目标是当前 evaluator 的离散方向能量分布：

```text
t_i = luminance(max(f_hat_i * cos_i, 0))
p*_i = t_i / Σ_j(t_j * solid_angle_weight_j)
L_sampler = -Σ_i p*_i * solid_angle_weight_i * log(q_i)
```

零能量group使用cosine target。报告evaluator-relative KL、reference-oracle cross-entropy、entropy、null mass、cosine-relative MC variance；reference target只作oracle，不把source`reference_pdf=0`解释为sampler监督。

### 4.4 收敛证据

每个 evaluator/sampler stage 都生成独立 convergence report，并引用完整训练 trace 与 checkpoint identity：

1. 每个已记录 step 的 objective、梯度范数和参数统计有限；非有限、silent clamp、跳过 optimizer step 或 callable 梯度身份错误直接判 implementation/convergence 失败。
2. 用固定 validation groups 对初始化 checkpoint 与训练 checkpoint 做 paired 比较；改善结论基于差值相对零的区间，不设任何材质误差目标值。
3. 对训练后段的 validation trace 做稳健趋势/区间分析；若存在可信上升趋势则记 late divergence，不能用较早 best checkpoint 掩盖训练不稳。
4. best checkpoint 必须可恢复，并在相同 validation 输入上复算一致；test 只在配置、实现身份和 convergence report 冻结后运行。
5. 汇总预先冻结的多 seed 结果；所有 seed 分别报告，不因某个 seed 质量较差而删除。`convergence_status=passed` 要求这些运行对“有限、改善、无后期发散”的判断一致；最终误差数值只进入 quality report。

## 5. SlangPy 与参数 identity

- `slangpy==0.43.1`必须先安装到`neural-shading`，运行现有/更新后的minimal autodiff spike；固定记录SlangPy/Slang/Falcor版本、weight tensor写法、梯度误差和吞吐。
- thin`nn.Module`只持有FP32 master weights/latents；`predict_f`和sampler PDF调SlangPy。自定义autograd wrapper把`bwd_diff`返回给Torch optimizer。
- shader`Params`布局由反射结果生成packing manifest；不得在Python与Slang各维护一份手工offset。checkpoint identity包含pipeline descriptor、Slang implementation hash、layout hash、data entry ID、training config hash和fitted-state hash。
- Falcor与SlangPy对同一FP32/FP16-packed inputs双编译；FP32 evaluate parity `rtol≤2e-5`，FP16 packed deployment parity `rtol≤2e-3`，任何非finite均失败。

## 6. 评测与选择

### 6.1 复现 gate 与 quality report

复现 eligibility 只由三类证据组成：method-correspondence/独立 oracle 证明实现正确，convergence report 证明训练稳定收敛，SlangPy/Falcor/packed checkpoint parity 证明部署的是同一实现。quality suite 的 sanity 仍可拒绝非有限、负值或合同错误，但 directional/energy 的绝对数值不作复现 kill gate。

冻结 best checkpoint 后再运行 test、adversarial、dense、signed energy、core coverage、peak、bootstrap 与 leave-one-out。报告保留逐 state 和结构分组，test 只读取一次；低质量结果是有效实验结论，不触发“继续修到过某条线”的无限循环。

### 6.2 matched 2×2

四格固定为：

| | NVIDIA GGX9 | LTC-K2 |
|---|---|---|
| NVIDIA 原规模 direct | A（复现 baseline） | B |
| exact-core positive residual | C | D |

每格共享对应evaluator checkpoint，只更换sampler head。比较单位为相同state/wo/integrand；paired bootstrap至少1,000次，95% CI。

选择规则：

1. implementation、convergence、sampler correctness 或 parity 未通过的格不能形成有效比较；这与质量高低无关。
2. 每格都报告相对 A 的 directional/energy/variance/time/memory paired evidence，并按材质结构分组；CI 跨零记“无显著差异”。
3. 只有在声明的比较范围内具有可信 Pareto 改善且其他主项无可信退化时，才形成该范围的优选结论；不同结构组结论不一致时保留多个非支配结果。
4. 报告不得因为没有全局 winner 而把 A 写成机械 fallback，也不得因为候选质量较低否定其“正确实现且稳定收敛”的复现状态。

## 7. 产物与 rollback

- 正式运行写`artifacts/runs/unified-scattering-03/`，比较写`artifacts/comparisons/`，packed assets写`artifacts/compiled-materials/`；所有路径由manifest/hash追溯但不进Git。
- 根仓库版本化smoke/standard configs、schema、源码、tests和`docs/research/experiment_log.md`结论。
- SlangPy环境/双编译失败：先修兼容源或包装，不启用Torch生产fallback；安全范围内穷尽后才报告blocker。
- implementation 对应关系失败：先修方法语义、独立 oracle 或数据/方向合同，再生成新 implementation/config identity；旧 run 保留但不进入复现结论。
- convergence 失败：根据非有限、梯度、初始化对照和 late-trend 证据修训练生命周期；不得以最终 quality 数值或换一个幸运 seed 掩盖。
- 实现正确且稳定收敛但 quality 较低：停止“修到过线”的循环，登记当前观察结果，转入结构归因或后续候选设计。
- sampler correct但无方差收益：仍保留实验结果，选择另一sampler或A；不得把无偏性当低方差结论。
