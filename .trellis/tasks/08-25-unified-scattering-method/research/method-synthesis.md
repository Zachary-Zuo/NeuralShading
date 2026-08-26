# 首个完整散射方法的证据综合

## 结论

首个部署候选不应直接续写旧 `lobe-residual-k2-v1` 的“顶层 core + 两个解析 LTC lobe，逐方向 MLP 仅作可选乘性修正”形态。该形态适合作为 optimized-code control，但把复杂多层响应压进两个固定 lobe 会重新引入表达瓶颈，并且旧默认配置甚至没有逐方向 neural evaluator。

推荐研究设计由 baseline、目标 evaluator 和 sampler 配置轴组成：

1. NVIDIA 结构 baseline：two learned shading frames + direct BRDF MLP + tilted-cosine diffuse / non-centered anisotropic GGX specular proposal；
2. 目标 evaluator：保留同一 learned-frame direct MLP 主体，增加精确顶层 reflection core，由 positive residual MLP 预测其余散射；
3. sampler matched axis：忠实适配 NVIDIA 9 参数 proposal，并与 K=2 LTC proposal 比较；两者的 `sample()` 与 `pdf()` 都只执行同一参数化分布的解析公式。

这同时保留了 P1 审计中最强的物理归纳偏置、当前项目对 neural evaluator 的定义，以及 sampler/PDF 的数学可验证性。

## 仓库证据

### `p1_audit.md`

- P1 v1 的 direct MLP 把网络放大到约 `8.6e5` MAC，仍然丢失极窄顶层峰，说明已知 top-interface core 应保留。
- 解析 core 在 15 个单层 state 上准确；M2 的尾部失败集中在 core 覆盖不足且 `clamp(core + signed residual, 0)` 进入死区的 4 个多层 state。应删除的是 signed residual + clamp，而不是 physical core。
- 高阶 Fourier log-slope 对平滑材质会产生无益高频；learned frame 或解析 warp 更适合把移动峰驻定。
- 目标逐方向网络应回到 2–3 个小层、32–64 宽、无 LayerNorm/erf-GELU 的着色器形态，并使用真实条件化。

### `p1_v2_plan.md`

仍可复用：

- `z ∈ R^16` 的 target-visible offline latent；
- `23→64→64` 量级的 `prepare` 网络和 `≤64 B` state 预算；
- exact top-interface core；
- LTC 的 `exp/softplus/tanh` 有界参数化；
- `K=2` 的部署成本结论；
- SlangPy、Falcor GPU 测试和 viewer 共用一份 Slang 源；
- method correspondence、稳定收敛、signed energy、core coverage、tail guard 和 bootstrap 诊断；绝对质量只报告，不作复现门。

需要修正：

- LTC lobe 不再定义目标 evaluator 的输出词汇，只用于 learned tractable proposal 和显式 analytic control；
- 逐方向 `EvaluateMLP` 是必选主体，不再把 `correction=none` 设为部署默认；
- neural residual 只补充 top core 之外的非负能量，不再把乘性修正施加到已经精确的 top core；
- method PT 是 MethodBundle backend specialization，不是 source reference 的“第五个 family”。

### 当前 `lobe_residual` 骨架

- Falcor-free 的 `core/mlp/pack` 拆分、half state、定长循环和权重偏移思路可复用。
- 当前 `sample()` 固定失败、`pdf()` 固定为 0，pipeline 训练入口全部未实现；它没有可保留的运行时方法身份。
- `f *= exp(Delta)` 会同时修改 exact top core；`correction=none` 又让 evaluator 退化为解析 lobe 求和。这两点不进入新方法。

## 原始文献核对

- NVIDIA 的 [Real-Time Neural Appearance Models](https://research.nvidia.com/labs/rtr/neural_appearance_models/) 使用 two learned shading frames 后由小型 MLP 直接预测 BRDF；同参数量 ablation 显示 learned-frame full model 优于 vanilla decoder。其 sampler 不是泛称的 two-lobe，而是 9 参数 `{w_d, mu_d,x, mu_d,y, w_s, alpha_x, alpha_y, rho, mu_s,x, mu_s,y}`：tilted-cosine diffuse + non-centered anisotropic GGX specular。论文用相对当前 learned BRDF 的 KL 训练 sampler，并把 latent 从 KL 梯度 detach；这些都应成为本项目 baseline 的必做机制。
- [Neural BRDF Representation and Importance Sampling](https://diglib.eg.org/bitstreams/2dc6056b-680f-43a5-ba25-59d2cdcd6142/download) 也把 neural 表示和可采样的解析 proxy 分开，并强调针对高光的方向采样密度。
- LTC 原始方法说明：从 normalized clamped-cosine 分布作可逆线性方向变换后，解析表达、归一化和 importance sampling 性质会继承。见 [Linearly Transformed Cosines](https://eheitzresearch.wordpress.com/415-2/)。因此 LTC 适合作为 neural head 的 tractable proposal family。
- Heitz 的 [GGX VNDF exact sampling](https://jcgt.org/published/0007/04/01/) 是顶层界面 control/reference 可复用的数学原语。NVIDIA sampler 使用的是经非中心线性变换的 GGX NDF proposal，并非当前 reference 中的 VNDF 函数；两者必须分开实现和测试。任何 reflection mapping 的 null event 都要作为显式概率质量计入归一化测试，不能静默丢弃或重采样。

## baseline 与目标方法草案

baseline 标识为 `nvidia-frame-two-lobe-v1`；目标 evaluator 标识为 `core-frame-neural-v1`，sampler 后缀由 matched 结果选择：

```text
NVIDIA 原规模 baseline
  learned frames + direct EvaluateMLP
  + epsilon cosine safety component
  + tilted-cosine diffuse / non-centered anisotropic GGX specular

Target evaluator
  exact top-interface core
  + learned frames
  + positive-residual EvaluateMLP

Sampler axis
  A. NVIDIA 9-parameter diffuse+GGX proposal
  B. K=2 LTC proposal

Analytic control
  exact top + K2 LTC evaluator lobes（不是 neural target）
```

四个 matched 组合为 evaluator `{NVIDIA direct, core positive residual}` × sampler `{NVIDIA GGX9, LTC-K2}`。implementation/convergence 与 quality/cost comparison 分开；后者按材质结构报告 evaluator quality、sampler 方差、runtime 和 memory 的 Pareto，允许保留多个条件性非支配结果，不把比较失利写成复现失败。

当前 Falcor 8.0 / Slang 2024.1.34 没有 cooperative vector。论文 `64×64×64` evaluator 与 `32×32×32→9` sampler 仍作为原规模正式 baseline 在标量 Slang 上实现、导出、显示和测时；超过软线只改变runtime class。缩模不是复现前置，若以后研究必须使用独立身份。

## 数据结论

现有 v5 response shard 已含 `wo/wi/f·cos/proposal_pdf/solid_angle_weight`，足以：

- 训练 evaluator；
- 训练相对当前 evaluator 的 sampler KL，并对 shared latent/state encoding detach sampler 梯度；reference `luminance(f·cos)` 目标只作为独立 diagnostic/oracle；
- 做独立方向积分和 sampler 方差评测。

首个方法限定 reflection-only、non-delta，因此 `reference_pdf=0` 本身不要求改 schema。NVIDIA 的 directional mollification 与训练密度是否能从当前冻结 corpus 忠实构造仍需专项 adequacy 审计；不足时必须生成新 corpus identity，若需保存 cone/group 语义则版本化合同，不能为了沿用 P1 文件而删掉已证实有效的训练机制。

## 当前结论

NVIDIA 方案是必做结构 baseline；exact top core / positive residual 和 LTC sampler 都是相对它的候选改进，不是未经对照即可替换 baseline 的既定答案。
