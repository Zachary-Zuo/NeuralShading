---
paper_id: "zheng-2021-neural-process-brdfs"
title: "A Compact Representation of Measured BRDFs Using Neural Processes"
authors: "Chuankun Zheng; Ruzhang Zheng; Rui Wang; Shuang Zhao; Hujun Bao"
year: "2021"
venue: "ACM Transactions on Graphics 41(2), Article 14; presented at SIGGRAPH 2022"
doi: "10.1145/3490385"
report_status: "evidence-reviewed"
main_source: "https://projects.shuangz.com/npbrdf-tog22/npbrdf-tog22.pdf"
supplemental_status: "unavailable"
official_code_status: "audited"
official_code_commit: "c471c99f1665e036a5813731718162553347b4d2"
author_worker: "/root/lightformer2024_review"
reviewer: "/root/dualband2025_review"
last_verified: "2026-08-29"
---

# A Compact Representation of Measured BRDFs Using Neural Processes

## 1. 研究对象与报告边界

本文研究的不是“给每个材质单独拟合一个 tiny MLP”，而是把一组 measured BRDF 视为函数空间中的样本，用 Neural Processes 学得三部分共享表示：[P §3-4, Figs.2-3]

1. **set encoder `h` + aggregator `a`**：把一个 BRDF 的任意数量、任意顺序 observation pairs `(direction query, RGB reflectance)` 编成 7D Gaussian posterior；
2. **共享 decoder `g`**：输入 7D material latent 和一次方向 query，输出该材质的 RGB BRDF；
3. **两个 post-trained decoder**：一个 hypernetwork 把共享大 decoder 蒸馏成每材质 2259-scalar `mainNet`，另一个两层 NICE normalizing flow 为 latent BRDF 产生 importance samples 和对应 PDF。[P §§6.2,7.1, Figs.8,13]

论文以 100 个 MERL 与 51 个 EPFL isotropic measured BRDF 共同训练一个函数空间，重点展示 compression、latent interpolation/editing 和 BRDF importance sampling。它不处理 anisotropic BRDF、spatially varying material、footprint/LOD、transmission、source-native procedural parameter editing 或 scene visibility/transport。[P §2.2; §8.1]

本报告覆盖 DOI `10.1145/3490385` 的 15 页正式版本、作者 publication entry，以及官方 `Rendering-at-ZJU/NPs-BRDF` 在 2026-08-29 可见的最新 commit。公开仓库在论文之后才发布：2023 初次公开 NP inference，2024 再加入 hypernetwork；因此代码是独立 `C` 证据，不能自动当成 2021 formal training code。论文多次引用 supplemental，但当前作者 publication entry只给 paper，官方仓库也不含 supplemental/NICE 资产；这部分保持 unavailable，不由正文或代码补猜。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---:|---|---|
| Main paper `P` | [作者托管 PDF](https://projects.shuangz.com/npbrdf-tog22/npbrdf-tog22.pdf)；[DOI](https://doi.org/10.1145/3490385)；TOG 41(2), Article 14 | 2026-08-29 | SHA-256 `932B693D11BDB9F285D9D56AE7293D0FCB41775585863E773D5C93E872446C44` | 15 页；方法、公式、图表、实验与作者限制的最高优先级来源 |
| Supplemental `S` | P 在 §§4.1、5.4、6.3、7.1、7.2 引用的 supplemental/web tool/video | 2026-08-29 | unavailable | 作者 [publication index](https://shuangz.com/publications/) 对本条只列 paper；repo 无 supplemental；自动访问 ACM DOI 页面返回 403。无法核验 multi-log ablation、更多 novel-BRDF/importance/comparison 图和 editing video |
| Official code `C` | [Rendering-at-ZJU/NPs-BRDF](https://github.com/Rendering-at-ZJU/NPs-BRDF)，MIT | 2026-08-29 | commit `c471c99f1665e036a5813731718162553347b4d2`（2024-11-25）；codeload ZIP SHA-256 `F506A3CE372BDB051CD713994E313A4888CFD03D12E7706A4DB62B49A1CC402F` | 固定审计 inference、网络构造、坐标、MERL I/O、预训练 NP/hyper weights、151 个 latent 与 21 个 traits；没有 main training loop 或 NICE |
| Initial public code `C-old` | 同一 repo commit `2388078dd16b5dcee22749da34dbba2b1f567adb` | 2026-08-29 | 2023-04-04 | 首次公开 NP inference、checkpoint、latent/traits；说明 current head 不是论文提交时快照 |
| NP checkpoint `C` | `models/LOG4_Mean_Dim7/weight/Epoch40000_400000_weights_nps.h5` | 2026-08-29 | SHA-256 `738897DFD0CA2EA5037DDCCB193657408C17EAED77ADC06630E495DB89864845`；4,086,912 bytes | 确认公开 code topology；文件名第二个计数的语义未说明 |
| Hyper checkpoint `C` | `models/LOG4_Dim7_Post_Hyper7_5/weight/Epoch60000_600000_weights_nps.h5` | 2026-08-29 | SHA-256 `70EE0D30EFF4EA2ACF8BDBF4BBAC814131596B52D0F7B1CA5A66A9CF8C27D17C`；10,771,336 bytes | 2024 commit 后补；同时含 frozen original decoder 与 hypernetwork weights |
| Author material `A` | [Shuang Zhao publication index](https://shuangz.com/publications/)；official repo README | 2026-08-29 | fixed URLs | 交叉核对 publication identity、code ownership、tested TensorFlow/Python/GPU 和公开资产；未发现 talk 或 correction |
| NeuralShading evidence `N` | [当前 NVIDIA correspondence](../implications/current-nvidia-correspondence.md)；[学习与部署](../../../../../docs/learning.md)；[runtime/compiler contract](../../../../../docs/realtime_material_compilation.md) | 2026-08-29 | repo-local | 只用于 §§14-15，不能回填 P/C 事实 |

### 2.1 来源可用性结论

- Main paper 的 15 页、Figs.1-21、Table 1、Algorithm 1、Eqs.(1)-(17) 与所有图注/脚注已逐页视觉核对。
- 可公开访问的第一方材料没有提供 supplemental。P 明确说 supplemental 包含额外结果和 web/video 工具，因此不是 `not-applicable`；缺口会直接限制 log-count ablation、更多 importance 图与工具行为的复核。
- 官方 repo 是作者机构组织下、README 明确对应本文的代码，但初次发布晚于论文约 17 个月；latest commit 又晚至 2024。报告分别保留 P、C-old、C，不把 later code 静默提升为 2021 formal implementation。
- repo issue #1 仍公开询问 training source；current head 确实没有 main NP 训练入口。`models.py` 有 network graph 和 hyper trainer，`NPs.py` 只有 compress/decompress/edit/interpolation inference。[C commit `c471c99`, `NPs.py:L19-L74`; repo issue #1]
- current head 含 151 个 latent files，正好覆盖 P 的 100 MERL + 51 EPFL；公开数据不含原始 measured BRDF，README 要求用户另行下载 MERL，EPFL 下载/转换入口未给出。[C README:L12-L38; asset audit]

## 3. 原论文的问题、假设与贡献边界

作者指出 measured BRDF 的两类旧表示各有结构性代价：matrix/tensor factorization 依赖共同、稠密的 direction discretization，且 runtime 需要 interpolation；解析 fitting 极紧凑，但简单模型不足以表达真实材料，复杂 multimodal fitting 又容易不稳定或陷入 local minima。[P §1, pp.14:1-14:2]

论文的核心假设是：不同 measured BRDF 可以被视为同一随机函数族的样本。对材质 `k`，latent `z_k` 服从由 observation set 推断的 Gaussian：

```text
f_k(ω_i, ω_o) ≈ g(ω_i, ω_o; z_k)
q(z_k | X_k, Y_k) = N(z_k; μ_k, Σ_k)
(μ_k, Σ_k) = l(X_k, Y_k) = a({h(x_j, y_j)})
```

[P Eqs.(1),(4)-(8), pp.14:3-14:4]

这使一个共享 decoder `g` 同时服务多个材质，而 `h/a` 可以接受不同材质各自的采样 pattern，不要求把 EPFL adaptive measurements 重采样到 MERL grid。训练使用 context posterior 近似 full/target posterior，以 KL regularization 让少量 context 也能总结同一函数。[P §4.1-4.3]

作者声明的三项贡献是：[P §1, p.14:2]

1. 基于 Neural Processes 的 measured-BRDF compact representation 与 end-to-end latent-space learning；
2. 对 latent dimensionality、semantic organization、stochastic reconstruction 和 novel-BRDF generalization 的系统分析；
3. 两个 post-trained networks：hypernetwork 用于小规模 BRDF 集合的 per-material compact decoder，NICE 用于 latent-conditioned importance sampling。

贡献边界很重要：这是 **measured-BRDF asset compression**，不是从 source-native procedural parameters 零样本编译材质；latent editing 是 learned appearance-space interpolation/trait motion，也不是原生物理参数编辑。原始 decoder 的 compression 还依赖跨多材质摊销 3.11 MB shared network。[P §§6.1-6.2]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | 一个 isotropic measured BRDF 的 observation pairs `(x_j,y_j)`；不同 BRDF 可有不同 query pattern 与数量 | `x_j∈R^4`，`y_j∈R^3`；set 长度可变 | [P §4.1, Eq.(8)] |
| Direction coordinates | Rusinkiewicz half-difference angles `(θ_h,θ_d,φ_d)`；利用 reciprocity 令 `φ_d` 以 `π` 为周期 | `x=(θ_h,θ_d,sin(2φ_d),cos(2φ_d))`；isotropic reflection | [P §4.1, p.14:4] |
| Encoder query | 方向 `x_j` 与同点 RGB reflectance `y_j` 共同进入 per-observation encoder `h` | 4+3=7 scalars/observation | [P Fig.3(b), p.14:5] |
| Runtime evaluator query | 固定 material latent `z` 加一次 direction query `x` | `z∈R^d`, `d=2..7`；正式其余实验用 `d=7`；decoder input 11D | [P Eq.(7); §§5.1,6.1; Fig.3(b)] |
| Evaluator output | RGB BRDF `f(ω_i,ω_o;z)`；multi-log domain 内预测后反变换 | 3 nonnegative scalars；Eq.(16)另乘 `cosθ_i` 构造 sampling target，说明 evaluator 本身不是 `f cos` | [P Eqs.(1),(7),(16); §4.2] |
| Importance-sampler condition | outgoing zenith `θ_o`、7D BRDF latent `z`、`x∼Uniform([0,1)^2)` | 输出 normalized half-vector coordinates `(θ_h,φ_ho)`；`φ_ho`相对 outgoing azimuth | [P §7.1, Fig.13] |
| Importance target measure | fixed `ω_o,z` 下的 incident solid-angle density，正比于 BRDF luminance `f` 与 `cosθ_i` | upper hemisphere `H^2`；footnote 明确 `f` 取 luminance | [P Eq.(16), footnote 2] |
| Validity/domain restrictions | only isotropic reflective BRDF；`ω_i,ω_o`应位于 upper hemisphere | no BTDF、no spatial coordinate、no time/scene context | [P §2.2; C `models.py:L30-L34,L52-L55`] |

### 4.1 坐标编码的论文与代码语义

P 把 `φ_d` 映射到圆 `sin(2φ_d),cos(2φ_d)`，其目的是同时表达 `π` 周期 reciprocity 与 `φ_d=0/π` 连续性。[P §4.1]

current C 的 MERL inference 先把 `(φ_d,θ_h,θ_d)` 分别归一化，再把 `φ_d/π` 变成 `0.5 cos(2φ_d)+0.5` 与 `0.5 sin(2φ_d)+0.5`，最后排列为 `(cos-map,sin-map,θ_h-map,θ_d-map)`。[C `NPs.py:L31-L38`; `util.py:L9-L21`]

因此 P 与 C 的 **维度和周期机制一致**，但输入顺序与 `[0,1]` affine scale 在 P 中未写。它们不影响“4D reciprocal coordinate”这一事实，却会影响 checkpoint-compatible reproduction，必须以 code identity 单独锁定。

### 4.2 Output measure

P 的 decoder 输出是 RGB BRDF reflectance `f`，没有预乘 `cosθ_i`。四次 `log1p` 只是一种内部 target transform；C 在保存 `.binary` 前做四次 `expm1` 反变换。[P §4.1; C `util.py:L37-L40,L60-L65`] P/C 没有另行声明 RGB color-space 或物理单位 metadata；因此可以确定的是“inverse multi-log 后的 bare RGB BRDF 数值”，不能仅凭本文把它升级成带完整单位/色彩管理合同的 runtime ABI。

NICE 的 Eq.(16) 再用 `luminance(f) cosθ_i` 归一化 proposal。不能把该 cosine 乘积误写为 evaluator ABI，也不能把 NICE 的 half-vector density直接当作 incident solid-angle PDF；后者还需要完整 change-of-variables/Jacobian，而 P 只给框架，没有把所有球面与 half-vector Jacobian逐项展开。[P §7.1, Eqs.(15)-(16), Fig.13]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

#### 5.1.1 NP function-space training

```text
MERL 100 + EPFL-isotropic 51 measured BRDFs
  → 保持各自原始 direction observations
  → Rusinkiewicz x4 + RGB y
  → y 连续执行 4 次 log1p
  → context set C 与 target set T
  → h(x,y) per observation
  → mean aggregation + a(s)
  → q_c=N(μ_c,Σ_c), q_t=N(μ_t,Σ_t)
  → z~q_t
  → g(x,z)=transformed RGB BRDF
  → L2/Σ_err + KL(q_t||q_c)
```

[P §§4.1-4.3, Algorithm 1, Fig.3]

#### 5.1.2 Compression/evaluation

```text
arbitrary observation set (X,Y)
  → h + mean + a
  → (μ,Σ)
  → deterministic asset code z=μ
  → shared g(x,z)
  → inverse multi-log
  → RGB BRDF f
```

P 的 storage equation只计 `d`-scalar `z`；C 的 `.npy` 同时保存 shape `(2,1,7)` 的 mean 和 log-variance，但 `Config.test_mode=True` 令 decoder实际使用 mean。[P Eq.(13); C README:L20-L27; `models.py:L14-L21`; asset audit]

#### 5.1.3 Post-trained compact decoder

```text
random z in learned 7D space + random directions x
  → frozen original decoder g(x,z) as teacher
  → hyperNet(z) = z_hat (2259 mainNet weights/biases)
  → mainNet(x; z_hat) ≈ g(x,z)
  → discard hyperNet for intended deployment
  → persist z_hat once per BRDF
```

[P §6.2, Eq.(14), Fig.8]

这个 `mainNet` 是 post-trained distillation，不是 NP encoder 的一部分。它以少量 per-material weights 替换约 3.1 MB shared decoder，代价是 reconstruction PSNR 从 56.20 dB 降到 48.98 dB。[P §§6.1-6.3, Fig.10]

#### 5.1.4 Post-trained NICE sampler

```text
random latent z + outgoing zenith θ_o
  + u=(u1,u2)~Uniform([0,1)^2)
  → coupling T0, T1（各自由 small weight network 条件化）
  → normalized half-vector (θ_h,φ_ho)
  → incident direction ω_i
  → accumulated log-Jacobian J0+J1
  → matching learned density / PDF
```

训练 target 是由 frozen NP decoder 给出的 `luminance(f) cosθ_i` normalized density，不需为每个 BRDF另做 GGX fitting。[P §7.1, Eqs.(15)-(16), Fig.13]

### 5.2 持久化表示

| 表示 | Shared 内容 | Per-material 内容 | 论文 storage/cost 边界 |
|---|---|---|---|
| Original NP | decoder `g`，§6.1称约 3.11 MB losslessly compressed；§6.3另称 MERL total 3.10 MB | P：`d`-scalar mean latent；7D FP32 为 28 bytes `[I]` | 100 MERL 从 2D 增至7D只多约2 KB；encoder只在 compression 时需要；`3.11/3.10 MB` 是 P 内部舍入/文字不一致，不能据此恢复 exact bytes [P §§6.1,6.3, Eq.(13)] |
| Official latent asset | same shared decoder checkpoint | mean + log-variance，各7 FP32，共56 bytes | C 比 P storage equation多存 covariance metadata；inference仍取 mean [C README:L23; asset audit] |
| Hyper mainNet | architecture fixed，无需存 hyperNet | 2259 weights/biases，论文计9 KB/BRDF | intended runtime 只存 `z_hat`；公开 CLI 没有导出 `z_hat` 的独立入口 [P §6.2; C `models.py:L139-L169`] |
| NICE | shared 2-layer conditional flow，约37 KB | 7D material latent | sampler network 和 latent共同决定 distribution；public repo不含 sampler weights/code [P §7.1] |
| Traits | SVM-derived directions，不是 evaluator必需内容 | 13 author traits；C current head提供21个 `.npy` vectors | 用于 editing UI；C 包含 paper 13 traits之外的 additional color/diffuse vectors [P §7.2; C asset audit] |

没有 texture、plane、mip、LOD、quantization、spatial latent 或 analytic BRDF core。P 的 3.11/3.10 MB、9 KB 和37 KB没有给 precision/serialization明细；C checkpoint/latent arrays为 FP32，但这不能自动证明 TensorRT benchmark precision。

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| P encoder `h` | `x4+y3=7` | FC `7→400→400→400→64` | Fig.3 gold marker表示 ReLU；无 normalization | 64D observation feature | shared | [P Fig.3(b), p.14:5] |
| P aggregator `a` | set of 64D features | mean → FC `64→128→128→latent-dim` | ReLU；最终如何分支/约束 `μ,Σ` 未报告 | Gaussian `μ,Σ` | shared | [P §4.2, Fig.3(b)] |
| P/C decoder `g` | `x4+z7=11` | FC `11→400→400→400→400→400→400→3` | ReLU on every shown FC，尤其 output保证nonnegative；无 normalization | transformed RGB | shared | [P §4.2, Fig.3(b)]; [C `models.py:L36-L60`; H5 audit] |
| C encoder `h` | 7 | FC `7→400→400→64` | ReLU | 64 | shared | [C `config.py:L25-L30`; `models.py:L62-L69`; H5] |
| C aggregator | 64D mean | three FC `64→64`；two linear heads `64→7` | hidden ReLU；heads linear | `z_mean,z_log_var` | shared | [C `models.py:L71-L103`; H5] |
| P/C hyperNet trunk | `z7` | seven FC400 layers；final projections collectively output2259 mainNet scalars | ReLU trunk；C用每层weight/bias独立Dense head，等价于concatenated output | `z_hat` | post-trained shared generator | [P Fig.8]; [C `models.py:L139-L157`; H5] |
| P/C mainNet | `x4` | FC `4→16→32→32→16→3` | ReLU including final in C | transformed RGB | 2259 scalars/per-material | [P §6.2, Fig.8]; [C `models.py:L145-L157`] |
| P NICE | `u2,z7,θ_o` | two coupling layers `T0,T1`；每个 weight net接收 condition + passive coordinate，经 FC `→16→16→16→17` | Figure marks ReLU；17 parameters如何映射到 coupling transform由 Müller et al. 2019定义，本文未重述 | normalized `(θ_h,φ_ho)` + log-Jacobian | shared | [P §7.1, Fig.13] |

本次独立 H5 dataset-shape 复核确认：NP checkpoint 的 decoder 为 `(11,400)`、五个 `(400,400)`、`(400,3)`；encoder 为 `(7,400)`、`(400,400)`、`(400,64)`；aggregator 为三个 `(64,64)` hidden kernels 加 `(64,7)` mean/log-variance 双头。因而上表的 P decoder 与 C decoder 对应、P encoder/aggregator 与 C checkpoint 不对应，均是文件级证据，不是根据参数量反猜的 topology。[C fixed checkpoint H5 audit]

### 5.3.1 参数与 MAC 推导 `[I]`

- original decoder `g`：`808,003` trainable scalars；dense matmul为 `805,600` MAC/query。计算使用 Fig.3/H5 的 `11→400`、五个 `400→400`、`400→3`，不含 ReLU 与四次 inverse multi-log。
- hyper `mainNet`：`2,259` scalars、`2,160` dense MAC/query；以结构计，dense MAC约为 original decoder 的 `1/373`，但论文 full-table latency只有约23倍差距，说明 end-to-end kernel/launch/transform/coherence不能由 MAC比直接替代。
- C hyperNet generator：七个400-wide trunk层加总宽2259的projection heads，共 `1,871,459` scalars；intended deployment在每材质生成 `z_hat` 后丢弃它，所以这不是每次 `evaluate` cost。
- NICE：P只给约37 KB，没有足以无歧义复原 coupling transform parameterization 的代码或明细，不能从 `17` 输出反推 exact parameter/MAC。

### 5.4 条件化、坐标变换与物理先验

- **Permutation-invariant set conditioning**：`h`逐 observation 编码，mean aggregator让 observation顺序不影响 latent；作者称 mean通常优于 max/sum，但未给表格。[P §4.2]
- **Reciprocity**：isotropic `φ_d mod π` 与 `sin/cos(2φ_d)` encoding让交换 incident/outgoing方向保持相同坐标。[P §4.1]
- **Nonnegativity**：decoder最终 ReLU。[P §4.2]
- **Dynamic-range control**：BRDF value连续做四次 `log1p`；这是 function value transform，不是对数输出 measure。[P §4.1]
- **Energy**：没有 architectural guarantee。作者只在 latent interpolation区域实测，extrapolation可违反。[P §8.1, Eq.(17), Fig.20]
- **No learned frame/analytic core**：所有 direction canonicalization来自 fixed Rusinkiewicz coordinate；hyper mainNet仍拟合同一 transformed RGB function。
- **Sampler factorization**：NICE condition只用 `θ_o`而非 outgoing azimuth，因为材质 isotropic；输出 half-vector azimuth相对 `ω_o`。[P Fig.13]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset/source assets | MERL 100 isotropic BRDF + EPFL 51 isotropic BRDF；EPFL另有11 anisotropic，但本文明确不使用 | [P §2.2; §4.3] |
| GT/reference | 原始 measured RGB BRDF evaluations；MERL为 `180×90×90` table，EPFL保留 adaptive/different directions | [P §§2.2,4.1,6.3] |
| Train split | main NP 使用全部 151 measured BRDF共同训练；没有 train/validation/test material split | [P §4.3; §6.3] |
| In-distribution evaluation | compression PSNR对全部 MERL100报告；这些材质参与训练，因此不是 unseen-material test | [P §6.3, Figs.9-10] |
| External generalization | 306 个 Serrano et al. synthetic BRDF投影/重建，qualitative + per-example PSNR；不参与 main training | [P §5.4, Fig.6] |
| Context sampling | 每 iteration 对每材质 context size从1到16,200随机选择 | [P §4.3, p.14:6] |
| Target sampling | target size固定16,200；从原 observation set采样；context与target是否嵌套未明确 | [P §4.3; Algorithm 1] |
| Batch | 16 materials/iteration；每材质各自sample context/target observations | [P §4.3] |
| Coordinate conversion | EPFL在其原 sampling pattern上转成 Rusinkiewicz，不重采成MERL grid | [P §4.1] |
| Sparse-data experiment | 从 `180×90×90` 均匀移除29.8%、60.3%、90.0%、99.9% entries，保持其他training settings | [P §6.3, Table 1] |
| NICE training queries | 每 batch BRDF随机 `z`，采一个 `θ_o`和一组 upper-hemisphere random directions | [P §7.1] |
| Filtering/LOD/footprint | 不适用；homogeneous directional BRDF | method domain |
| Data generation | offline measured tables；不是每 optimizer step调用物理reference renderer | [P §2.2; §4.3] |

### 6.1 Full-set compression 与泛化边界

P 的主 PSNR对照在训练过的 MERL100 上完成；EPFL的主要作用是让同一函数空间覆盖不同 sampling patterns，而不是提供独立 held-out split。作者还明确说只在 MERL训练可能进一步改善 MERL result，说明 Figure 9/10 不是跨 dataset generalization证据。[P §6.3, p.14:9]

对 306 novel synthetic BRDF，作者观察 shape of highlight可恢复，但 color recovery会失败。建议少量 novel data可直接用或 fine-tune，大量 novel data应加入原 MERL+EPFL retrain；只用 novel data retrain虽可能，但会失去原 latent semantics。[P §5.4]

### 6.2 Official data/code 边界

C current head公开151个 latent而不公开 measured dataset或 main training batch pipeline。`config.py`含 `MERL_path`/`EPFL_path`，但 CLI `compress`只读 MERL `.binary`；`merlFunctions.py`虽有 UTIA reader，也不是 P 的 EPFL adaptive input loader。不能凭路径常量宣称 EPFL formal training可由 repo重跑。[C `config.py:L8-L41`; `NPs.py:L52-L56`; `merlFunctions.py:L5-L42`]

C inference对 full MERL grid生成1,458,000 queries，使用预计算 `NdotL/NdotV`把 lower-hemisphere entries mask为零。这个 mask和 `global_data_std=0.15`未在 P 正文报告，是 checkpoint-specific code contract。[C `NPs.py:L31-L47`; `util.py:L23-L46`; `models.py:L30-L34`]

## 7. Loss、optimizer 与训练 lifecycle

### 7.1 Main Neural Process

| 项 | 正式配置 | locator |
|---|---|---|
| Target transform | RGB reflectance连续执行四次 `log1p`；P称比其它log次数稍好 | [P §4.1] |
| Reconstruction loss | `0.5 E_z[ΔY^T Σ_err^-1 ΔY]`，`Σ_err=0.2 I` | [P Eq.(12); §4.3] |
| Latent regularizer | `KL(q(z|target)||q(z|context))`，Gaussian KL analytic | [P Eqs.(11)-(12); Algorithm 1] |
| Latent sampling | reparameterized Gaussian；Algorithm写每材质一个 `z_k`，随后的正文写每 observation draw `z_k,j`，两处冲突 | [P Algorithm 1:L14-16; p.14:5 lower-right] |
| Optimizer | Adam，`lr=1e-4` | [P §4.3] |
| LR schedule | 未报告 | P |
| Batch/query count | batch16；context size从1..16,200随机选择（精确discrete distribution未报告）；target16,200 | [P §4.3] |
| Steps | 40,000 iterations | [P §4.3] |
| Hardware/time | Nvidia RTX 2080 Ti；约40h | [P §4.3] |
| Initialization/seed/model selection | 未报告 | P/S unavailable |

C `sampling()`为每 batch material draw一个 7D epsilon并向所有 queries broadcast；`Config.test_mode=True`则完全取mean。这支持 Algorithm 的“per-material z”路径，但 main training entry缺失，不能裁决 P 同页的 `z_k,j`文字冲突。[C `models.py:L14-L28,L105-L131`]

P 的 `Σ_err=0.2I`与 C 的 `global_data_std=0.15`不是同一字段：前者是 likelihood covariance，后者是 current checkpoint的 transformed-data scale。没有训练代码证明 C 如何把两者组合，不能互相替换。[P Eq.(12); C `config.py:L35-L38`; `util.py:L37-L40`]

### 7.2 Post-trained hypernetwork

| 项 | 正式配置 | locator |
|---|---|---|
| Teacher | frozen original decoder `g(x,z)` | [P §6.2, Eq.(14)] |
| Inputs | random latent `z` from learned space；每material随机directions | [P §6.2] |
| Loss | teacher BRDF output与 `mainNet` output差异；P称supervising BRDF value，exact reduction未报告；C为sum of `0.5*(y-y_hat)^2` | [P §6.2]; [C `models.py:L171-L197`] |
| Optimizer | Adam，`lr=1e-4` | [P §6.2] |
| Batch | 16 | [P §6.2] |
| Steps/time | 60,000 iterations，约20h，RTX 2080 Ti | [P §6.2] |
| Schedule/seed/checkpoint selection | 未报告 | P/S unavailable |

current C包含 hyper trainer graph和 2024 checkpoint，但不含正式 60k loop/data sampler；checkpoint filename `Epoch60000_600000`的第二计数未解释。[C commit `c471c99`, `models.py:L139-L197`]

### 7.3 Post-trained NICE sampler

| 项 | 正式配置 | locator |
|---|---|---|
| Target | frozen NP BRDF luminance乘 `cosθ_i` 后对 hemisphere归一化 | [P Eq.(16), footnote 2] |
| Training samples | random latent `z`、one `θ_o`、upper-hemisphere random direction set / batch BRDF | [P §7.1] |
| Loss | KL divergence；方向、normalization estimator、sample count未报告 | [P §7.1] |
| Optimizer/batch | 只说 configuration similar to hypernetwork；不能升级为 exact Adam/lr/batch identity | [P §7.1] |
| Training time | 约2h；small network；hardware沿同段语境为 RTX 2080 Ti，但未再次明写 | [P §7.1] |
| Code/checkpoint | 不可得 | [C commit `c471c99`, repo-wide audit] |

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Original runtime | per direction执行 shared `g(x,z)`；encoder只在compress新BRDF时运行 | [P Eq.(7); §6.1] |
| Original decoder storage | §6.1称约3.11 MB losslessly compressed；§6.3另称100个MERL的总表示3.10 MB，exact bytes未报告 | [P §§6.1,6.3] |
| Original decoder cost `[I]` | 808,003 scalars；805,600 dense MAC/query；四次 `expm1`额外 | [P Fig.3; C H5] |
| Per-material original state | P正式compression为7D mean latent；C asset另存7D log-variance | [P Eq.(13); C README:L20-L27] |
| Hyper runtime | intended：per material一次 `hyperNet(z)`生成并保存 `z_hat`，steady state只执行 mainNet | [P §6.2, Eq.(14)] |
| Hyper mainNet storage/cost | 9 KB/BRDF；2259 scalars、2160 dense MAC/query `[I]` | [P Fig.8; §6.2]; [C `models.py:L145-L157`] |
| Full-table decompression latency | `180×90×90` entries：original decoder74 ms，hyper3.2 ms，TensorRT，RTX 2080 Ti | [P §6.3, p.14:9] |
| NICE storage | roughly37 KB | [P §7.1, p.14:11] |
| NICE batch latency | 512×512 samples in8 ms，TensorRT，RTX 2080 Ti | [P §7.1] |
| Precision/quantization | 未报告；C H5/latent为FP32不等于 benchmark precision | P/C |
| Texture/feature fetch | 无 | method domain |
| Renderer FPS/single-query latency | 未报告 | P/S unavailable |
| Precompute included | full-table timings是 materialization/decompression；NICE timing是sample batch generation。I/O、PDF eval、BRDF eval、path tracing和environment lookup是否包含均未报告 | [P §§6.3,7.1] |

### 8.1 不能误读的时间数字

74 ms与3.2 ms是一次生成完整 MERL table的 coherent batch，不是 shader 中一次 random `evaluate(wo,wi)` latency；8 ms是262,144个 NICE samples的 batch，也不是完整 path-tracing frame。论文没有给 per-query launch、divergent rays、single-sample PDF、full renderer frame或显存数据，因此这些值只能保留在其 TensorRT/2080Ti/batch protocol 内。[P §§6.3,7.1]

### 8.2 Official CLI 与 intended deployment 的差异

P说 hyperNet在生成 mainNet weights后可以丢弃。C 的 `--backbone hypernet`却构造 `hyper_decoder` graph，每次 `decoder.predict`仍以 latent作为输入并在图内生成各层weights；repo没有“导出2259 weights为独立 runtime asset”的命令。[C `NPs.py:L42-L48,L57-L74`; `models.py:L139-L169`]

因此 C 能展示 hypernetwork functional inference，却不是 P 所述 steady-state storage/runtime路径的完整 exporter。README还把 hypernetwork误指为 §6.1（正文为§6.2），这是documentation gap，不改变论文方法身份。[C README:L12-L17,L37-L38]

## 9. 实验 protocol、baseline、指标与结果

### 9.1 Compression quality 与 storage

作者将 reconstructed BRDF 与 original tabulated BRDF分别放到球体上，在 St. Peter's Basilica、Uffizi、Grace 三个 environment maps下渲染，并以 image PSNR比较。Figure 9按 MERL material给 per-BRDF average rendering PSNR；Figure 10汇总全部 MERL。P 没有报告 image resolution、tone mapping/exposure、mask、multiple seeds或 confidence intervals。[P §6.3]

| Method/setup | Avg. PSNR (dB) | Storage context | 结论边界 |
|---|---:|---|---|
| Ours 2D | 47.15 | shared decoder约3.1 MB + 2 scalars/material | compression已很强，但作者认为低维 latent对novel data generalization较差 |
| Ours 3D | 53.78 | same shared decoder | observed result；未报告seed variance |
| Ours 4D | 53.24 | same | 比3D略低，说明这组结果不随dimension严格单调；P未解释原因 |
| Ours 5D | 55.68 | same | observed result |
| Ours 6D | 53.60 | same | 比5D低；P未解释 |
| Ours 7D | **56.20** | 100 materials从2D增到7D只约2 KB extra latent | 论文其余应用采用的主版本 |
| Ours 7D + hypernetworks | 48.98 | 9 KB/material mainNet | quality-storage/runtime tradeoff，不是等质压缩 |
| Bagher et al. 2016 non-parametric | 45.99 | 3.2 KB/material | analytic/non-parametric functional form限制quality |
| Hu et al. 2020 10D | 47.48 | decoder >11 MB | discrete learned representation baseline |
| Sun et al. 2018, 1 diffuse PC | 28.70 | model-specific | analytic/PCA baseline |
| Sun et al., 1 diffuse +3 specular PCs | 41.61 | model-specific | 同上 |
| Sun et al., 1 diffuse +5 specular PCs | 43.18 | model-specific | 同上 |
| Sun et al., 2-lobe GGX | 41.30 | model-specific | 同上 |
| Nielsen et al. 2015, 3/6/9 PCs | 23.30 / 34.00 / 40.20 | each PCA basis约33 MB | discrete PCA baseline |

[P Fig.10, p.14:10]

Figure 9显示7D NP对大多数 MERL materials高于上述 baselines，但这些是 **training-set asset compression** 结果，不是 unseen MERL split。Figure 1 的 `grease-covered-steel`例子为 NP `PSNR 51.32`、Sun `35.72`、Bagher `33.36`，只是单材质 illustration。[P Figs.1,9]

### 9.2 Resolution / sparse observations

| Training entries | Filter ratio | Full-resolution Avg. PSNR |
|---|---:|---:|
| `(180,90,90)` | 0.0% | 56.20 dB |
| `(160,80,80)` | 29.8% | 53.78 dB |
| `(133,66,66)` | 60.3% | 53.28 dB |
| `(83,42,42)` | 90.0% | 48.64 dB |
| `(18,9,9)` | 99.9% | 41.83 dB |

[P Table 1, p.14:10]

作者的正结论是移除60.3% entries没有出现“significant quality degradation”，90% removal仍与其引用 baselines大致同档；99.9%则明显下降。由于移除规则是 uniformly remove entries，不能把这一表外推成任意稀疏、偏置或 noisy measurement pattern的鲁棒性证明。

### 9.3 Latent space 与 stochastic reconstruction

| Experiment | Protocol | Metrics/observation | Result | locator |
|---|---|---|---|---|
| Latent semantics | 对151 measured BRDF的7D codes做PCA到2D | category clustering | fabric、paper、two-layer、metallic paint等形成局部簇；非监督出现 | [P §5.2, Fig.4] |
| Parametric sweep | 生成16,384个 diffuse+GGX BRDF，编码到7D再PCA | diffuse strength / roughness color map | PC2大致随 diffuse reflectivity，PC1大致随 roughness平滑变化 | [P §5.2, Fig.5] |
| Stochastic reconstruction | 对每个 MERL/EPFL BRDF多次sample `z~N(μ,Σ)`并渲染 | PSNR variance | 所有材质因 latent randomness产生的 PSNR variance `<0.001` | [P §5.3] |
| Novel synthetic BRDF | 306 Serrano synthetic BRDF投影/重建 | sphere PSNR + error map | highlight shape通常恢复，color bias可使quality显著下降；Fig.6示例PSNR约50.78、54.05、42.78、33.34 | [P §5.4, Fig.6] |
| Latent interpolation | linear mix两材质mean latent | renderings + BRDF slices | highlight shape和diffuse strength可平滑变化；不等同物理参数插值 | [P §7.2, Fig.16] |
| Hyper interpolation | original decoder vs hyper mainNet | error maps | qualitative相似但有small reconstruction loss | [P §6.3, Fig.12] |

P 的 `<0.001`写成“variances in PSNR”，没有说明重复次数或是 population/sample variance。不能重述为 PSNR standard deviation。

### 9.4 Importance sampling

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| MERL equal-sample | environment illumination，16 spp | cosine-weighted；Sun two-lobe GGX fit；NICE | rendered PSNR/noise | Fig.14第一例 `19.94/27.98/28.83 dB`，第二例约 `23.33/26.87/27.38 dB`（cosine/GGX/ours）；NICE略优GGX且不需per-BRDF fitting | [P Fig.14] |
| Interpolated BRDF | latent interpolation，16 spp | cosine-weighted；没有GGX | rendered PSNR/noise | NICE显著优于cosine；作者不列GGX，因为每个interpolated BRDF都需重新fit，不适合interactive editing | [P Fig.15] |

这些图支持 NICE 在作者的 learned latent family和16 spp scene上降低噪声；它们不提供 PDF normalization error、variance curve、equal-time renderer结果、MIS、tail weights或跨 dataset sampling。Figure 14把 NICE描述为“slightly better”于GGX，不能升级为一般性 sampler dominance。[P §7.1]

### 9.5 Physical plausibility

作者定义最大 reflected ratio：

```text
a(z) = max_{ω_o∈Ω} ∫_Ω f(ω_i,ω_o;z) (ω_i·n) dω_i
```

并在 latent前两 principal components的二维平面可视化。interpolation区域大体保持 `a(z)∈[0,1]`，但远离training samples的top-right extrapolation区域违反 energy conservation。[P §8.1, Eq.(17), Fig.20]

这只是 sampled 2D projection上的 empirical audit：P 没报告 integration quadrature、outgoing-direction resolution或全部7D domain证明。

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `ablation-inferior` | `log1p`次数变化 | four repeated `log1p`稍好 | 用于压缩 extreme HDR dynamic range | supplemental不可得，不能补写数值/对照次数 | [P §4.1] |
| `ablation-inferior` | aggregator用 max/sum而非mean | 作者说mean通常更好 | mean同时order-invariant且适合不同set size | 无表格/配置，不能写成formal失败 | [P §4.2] |
| `ablation-inferior` | latent 2D-6D vs7D | 2D compression仍强，但低维对novel BRDF generalization不利；3D>4D、5D>6D | 作者因此为多应用选择7D | 非单调结果可能含optimization variance，但P无seed证据，保持未知 | [P §5.1, Fig.10] |
| `ablation-inferior` | hyper mainNet替代original decoder | PSNR 56.20→48.98；9 KB/asset、full-table 74→3.2 ms | 接受small quality loss换小集合的storage/eval效率 | 正常 Pareto tradeoff，不是implementation defect | [P §6.3, Figs.10-12] |
| `ablation-inferior` | 减少training entries | 60.3% removal仍53.28；90%为48.64；99.9%为41.83 | NP能用continuous observations学习 | 仅验证uniform removal | [P Table 1] |
| `author-negative` | 直接应用于306 novel synthetic BRDF | highlight shape可恢复，color有bias；最差展示例33.34 dB | RGB baked latent没有decoupled color | 限制跨source family generalization | [P §§5.4,8.1, Fig.6] |
| `author-negative` | latent space远距离 extrapolation | 可能不保留任何training BRDF traits | encoder/decoder只对training neighborhood重建优化 | 不可把线性latent当全局可信edit domain | [P §8.1, Fig.19] |
| `author-negative` | extrapolated energy | top-right far region出现 `a(z)>1` | network无energy-conserving guarantee | interpolation观察不是物理证明 | [P §8.1, Fig.20] |
| `author-negative` | color-dependent interpolation/editing | color可能突然、非预期改变 | diffuse/specular albedo未分离，低维尤其明显 | 与source-native editable parameter语义不等价 | [P §8.1, Fig.21] |

在已获得第一方材料中，没有其它“训练失败”“NICE不收敛”或“hypernetwork不稳定”的正式负结果。未采用其它 normalizing flow、loss或coordinate不能反向写成作者尝试失败。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Main encoder `h` | `7→400→400→400→64` | unavailable | `7→400→400→64`；H5一致 | **paper-code-gap**：少一个400-wide layer |
| Aggregator `a` | mean→`128→128→latent-dim`，输出 `μ,Σ` | unavailable | mean→三个64-wide layers→两个7D linear heads `μ,logvar` | **paper-code-gap**：width/depth和variance representation不同 |
| Decoder `g` | `11→400×6→3`，ReLU | unavailable | same H5 topology，output ReLU，plus code-only dot mask | decoder topology对应；input scale/order与mask是later code contract |
| Multi-log | four repeated `log1p`；ablation在S | unavailable | `log_times=4`，forward `/0.15`，inverse four `expm1` | log count对应；0.15 scale未见于P |
| Main training | ELBO、Adam1e-4、batch16、40k、40h、context/target16200 | unavailable | main training loop、data sequence、loss compile均缺；repo issue公开索要training source | **paper-code-gap**：不可从 repo重跑formal training |
| Latent asset | P compression equation计 `d`-scalar mean | unavailable | 151 files均存 mean+logvar `(2,1,7)` FP32 | C保存额外 uncertainty metadata；steady decode取mean |
| Hypernetwork | Fig.8、60k/20h、9 KB mainNet | unavailable | 2024 commit加入matching topology和checkpoint；只有trainer graph/inference，无formal loop或mainNet exporter | functional partial release，晚于论文 |
| NICE | two coupling layers、37 KB、2h、TensorRT timing | unavailable | repo无 `NICE/coupling/importance` implementation或weights | **paper-code-gap**：sampler不可由official repo复现 |
| EPFL | 51 adaptive isotropic BRDF直接用于train | unavailable | 有51 EPFL-named latent；没有EPFL adaptive loader/training config | output assets存在，formal data path缺失 |
| Editing | 13 perceptual traits + SVM | unavailable | 21 trait vectors，CLI对mean latent加向量 | paper13功能可执行；extra vectors属later asset，不回填P |
| Hyper deployment | generate `z_hat` once，discard hyperNet | unavailable | CLI graph保留并执行hyperNet，不导出2259-scalar asset | **paper-code-gap**：intended steady-state exporter缺失 |

### 11.1 Latent draw conflict

Algorithm 1在每材质 target set上sample一次 `z_k`，Figure 3也画一个 `z'`广播给 target queries；同页训练文字却说为每个 observation pair随机draw `z_k,j`。C graph只支持每 batch material一个latent后broadcast。由于 S/training code不可得，本报告不裁决哪个是2021实际训练配置，只登记 C 选择。[P Algorithm 1, Fig.3; C `models.py:L23-L28,L119-L128`]

### 11.2 Public release identity

commit history只有四个 commits：2023-04-04 initial public release、README update，2024-11-25 add backbone hypernet。latest codeload并非隐藏的论文时期 snapshot；任何复现必须把 `P-2021`、`C-inference-2023`、`C-hyper-2024`分开命名。[C GitHub API commit history]

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. **只研究 isotropic BRDF**：EPFL 11个 anisotropic assets不进入本文方法实验。[P §2.2]
2. **Shared decoder amortization**：约3.1 MB shared network在大量BRDF时划算，小集合需要hyper mainNet且会损失quality；P 的 `3.11 MB decoder` 与 `3.10 MB MERL total`存在轻微内部不一致。[P §§6.1-6.3]
3. **Novel color generalization**：RGB baked latent没有颜色分解，novel BRDF color recovery可能较差。[P §§5.4,8.1]
4. **低维 latent generalization**：2D适合compression，但太低维难以泛化和插值；作者选择7D。[P §5.1]
5. **Extrapolation不可靠**：离training samples远时，appearance traits和energy conservation均可能失效。[P §8.1]
6. **Energy无理论保证**：half-difference coordinate与output ReLU只提供 reciprocity/nonnegativity，不保证 energy conservation。[P §8.1]
7. **Importance evidence局限**：NICE只在 learned isotropic latent family、少量 qualitative equal-sample renderings中验证；没有完整 variance/PDF audit。[P §7.1]
8. **新数据仍需 adaptation**：少量novel data可直接encode/fine-tune，大量novel data建议加入原datasets retrain；不是不看target observations的zero-shot compiler。[P §5.4]

### 12.2 由 method domain 直接推出的边界 `[I]`

- 没有 spatial/UV/footprint input，不能表示 SVBRDF/BTF 或提供 filtering/LOD。
- 没有 transmission/event/IOR contract，只能输出local reflection BRDF。
- latent interpolation和SVM trait edits不保持source-native参数语义，也没有全域physical validity。
- main decoder虽静态有界，但805.6k dense MAC/query远超本项目小 shader evaluator的目标形态；hyper mainNet更接近部署预算。
- NICE理论上有matching sample/PDF，但 public code缺失且 P未完整展开 half-vector到solid-angle measure，因此不能直接当实现oracle。

### 12.3 未报告/材料不可得

- main training seed、initialization、LR schedule、checkpoint selection、multiple runs、validation protocol；
- context size的精确discrete distribution、context是否为target subset、材质batch sampling、target reuse；
- P Figure3最终 aggregator如何从一个 `FC latent dim`同时构造mean/covariance，以及covariance positivity transform；
- Algorithm 1 per-material latent与正文 per-observation latent冲突的裁决；
- four-log ablation的对照次数、数值与 protocol；
- EPFL source filenames、adaptive measurement预处理、raw scale、invalid sample handling；
- Figure 9/10 image resolution、tone mapping/exposure、PSNR aggregation细节、seed/CI；
- stochastic reconstruction repeats与 `<0.001` variance estimator定义；
- hyper random-latent sampling distribution、teacher sample count、loss reduction、model selection；
- NICE KL方向、normalization estimator、direction count、optimizer exact values、coupling-transform 17参数语义、sample-to-solid-angle PDF完整公式、seeds/checkpoint；
- TensorRT precision、engine settings、batch warmup、I/O/transfer inclusion、single-query and full-renderer timing；
- §6.1 `3.11 MB` decoder与§6.3 `3.10 MB` MERL total的exact-byte裁决；
- supplemental web visualization tool、更多comparisons/novel/importance results与editing video；
- correction/errata/talk。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

这篇论文最容易被“7D latent”掩盖的事实是：高质量并不是由7个数独立完成，而是由一个约808k-scalar shared decoder、约151个training functions、每步每材质16,200 target queries、fixed reciprocal coordinate与四重log transform共同提供。7D只是对 shared function manifold 的地址。[P §§4.1-4.3,6.1]

容量分成四层：

1. **function-space prior**：MERL+EPFL共151个BRDF限定可学习分布；
2. **set compiler**：`h/a`把大量 observed response pairs压成latent；
3. **shared evaluator**：6×400 decoder承载绝大多数 runtime function complexity；
4. **post-training specialization**：hyperNet把shared decoder知识投影到2259-scalar per-asset mainNet；NICE再从latent statistics学proposal。

因此 `7D` 与 `9 KB`是两种不同 deployment point：前者每asset极小但依赖约3.1 MB shared decoder，后者每asset更大却可丢弃shared evaluator generator。Figure 10 的56.20→48.98 dB明确展示 specialization compression不是无损。[P §§6.1-6.3, Fig.10]

### 13.2 成功所依赖的假设

- source是 homogeneous isotropic measured BRDF，且允许先观察大量 `(direction,f)` pairs；
- compression结果在参与训练的 MERL family上评估，共享 decoder可跨足够多assets摊销；
- Rusinkiewicz coordinate与四重log能把峰和dynamic range变成400-wide ReLU MLP可拟合的形态；
- 7D neighborhood只需在training manifold附近有意义，远距离 extrapolation不是成功条件；
- importance target可由 frozen learned evaluator密集query并归一化；
- TensorRT coherent full-table/sample batches掩盖了 single random query的真实部署代价。

这些假设解释了论文为何同时在training-set compression上非常强、在novel color和extrapolation上明确失败。它不是泛化结论矛盾，而是 evidence scope不同。

### 13.3 可迁移机制与不能迁移的部分

**可迁移为独立候选/对照：**

- permutation-invariant observation-set encoder，可处理不同measurement patterns；
- fixed reciprocal coordinate与 periodic circle encoding；
- target-visible shared decoder → per-asset tiny mainNet 的 post-trained functional distillation；
- evaluator frozen后单独训练 sampler，避免 proposal objective反向改变 evaluator；
- NICE/flow proposal作为current analytic GGX9 sampler之外的expressive control；
- 对latent neighborhood做energy audit，而不是把smooth interpolation当物理保证。

**不能直接迁移：**

- 用 `(direction,f)` observations作compiler input会把本项目 pure native-parameter compiler变成target-visible fit；必须另建identity；
- measured-BRDF latent不能替代各source family的native parameters/resources；
- 7D latent/SVM direction不构成source-native可编辑语义；
- 约3.1 MB shared decoder和805.6k MAC不能因per-asset latent只有28 bytes就称实时材质程序很小；
- Figure 14/15不足以让 NICE成为无审计的sampler oracle；
- homogeneous BRDF结果不能证明spatial filtering、prepare reuse或multi-event BSDF。

### 13.4 与本项目 runtime contract 的关系

- **`prepare()`**：original NP可在cook时从observations获取latent；runtime `prepare`只需读取7D state，没有view-conditioned reuse。hyper mainNet可在cook时生成weights，不应在每shading point执行hyperNet。
- **`evaluate(wo,wi)`**：inverse multi-log后的P output是未乘 cosine 的 bare RGB `f`，与项目 ABI 的 output measure 对应；但P/C未单独声明RGB色彩空间/单位metadata，接入前仍要冻结数值尺度。fixed topology也静态有界。original decoder成本过大，mainNet 2160 MAC更接近部署candidate，但仍需本项目 source/query重新训练和FP16/Slang实测。
- **`sample/pdf()`**：NICE有bijective/change-of-variables设计，但缺 official implementation和完整measure细节；迁移必须先建立 independent PDF normalization、sample→pdf与transport oracle。
- **Filtering**：方法没有spatial coordinate/footprint，不能承担当前spatial assets的latent mip/filter职责。
- **Compiler role**：set encoder是 target-visible measurement compressor，不是pure source compiler；更适合作为 high-capacity/target-visible control或measured-asset workflow。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

当前 `nvidia-rta2024-functional-f@2` 使用 native-parameter encoder、hierarchical z8、两个learned frames、`3×64` evaluator和`3×32→9` analytic sampler；evaluator直接训练/输出bare linear `f`，sampler用detached evaluator构造 `luminance(f)|cosθ_i|` forward-KL。[N current NVIDIA correspondence §§1-2; `docs/learning.md`]

| Zheng 2021机制 | 当前 NVIDIA关系 | 分类 | 影响/边界 |
|---|---|---|---|
| Observation-set encoder `(x,f)→z7` | NVIDIA encoder读source-native features，训练后materialize spatial z8 | `not-applicable` + target-visible alternative | 若输入reference responses就不是pure compiler；需独立identity，不能无声替换native encoder |
| Fixed Rusinkiewicz `x4` | NVIDIA用latent预测两个 learned frames，再投影direct directions | `intentional-deviation` | 可作为fixed-coordinate matched control；Zheng没有证明它优于learned frames于layered/spatial source |
| Fourfold log transform + Gaussian L2 | current evaluator为single `log1p` L1 on bare `f` | `intentional-deviation` | 值得做单轴target-transform/loss ablation；不能把MERL PSNR直接当current改进证据 |
| Shared 6×400 decoder + per-asset z7 | current 3×64 decoder + spatial z8 mip chain | `not-applicable` | Zheng是大shared decoder极小state的另一Pareto点，不含filter/LOD |
| HyperNet→2259-scalar mainNet | current固定shared evaluator，不为每asset生成decoder weights | independent candidate | 可研究“shared teacher→bounded per-asset program”；必须freeze source/query和iso-byte/iso-MAC |
| Staged NP→hyper/NICE post-training | current evaluator/sampler simultaneous，但sampler target detach | partial shared principle | 论文支持frozen-evaluator post-training control，不证明current joint recipe有defect |
| NICE conditional flow | current sampler输出9个analytic two-lobe parameters | `intentional-deviation` | 更expressive sampler candidate，但code/PDF oracle缺失且runtime class不同 |
| Eq.(16) `luminance(f) cosθ_i` proposal target | current forward-KL target同样以detached `luminance(f)|cosθ_i|`构造 | `faithful` shared target principle | 只说明proposal target语义相近；KL estimator、proposal family和training schedule均不同 |
| Decoder bare BRDF output | current ABI bare `f` | `faithful` interface principle | cosine只属于proposal density，不需runtime division adapter |

本报告没有从 Zheng 2021建立当前 NVIDIA implementation defect。它提供的是三个独立研究轴：target-visible set encoder、post-trained per-asset compact decoder、conditional flow sampler。每个都改变 method identity，不能以“改进建议”覆盖 current baseline。

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：对 measured/irregularly sampled source，mean set encoder能在不同query patterns间稳定产生compact latent | [P §4.1-4.3]把MERL/EPFL共同编码 | 当前 measured source family也共享低维function manifold | 同一asset split比较 full direct fit、fixed-grid encoder、permutation-invariant set encoder；严格holdout assets与query patterns | reference、total queries、decoder、latent bytes、optimizer、seeds | G1/W、cross-pattern error、compile time、seed CI | set encoder在held-out patterns/assets不优于fixed/direct control，或需更多target-visible queries |
| H2：四重log比single `log1p`更适合极高dynamic-range compact evaluator | P称four logs稍好；Table/Fig结果使用该transform | layered peaks也受同类dynamic-range conditioning支配 | 只改变transform：linear、single log1p、four-log；runtime都反变换到bare `f` | network/coords/query/steps/seed/optimizer | bare-f、log、peak/tail、energy、finite、time | four-log只改善其own transformed loss而恶化bare-f/energy，或seed CI无收益 |
| H3：shared teacher post-distill成约2k-scalar per-asset mainNet，可改善bytes/latency Pareto | [P §§6.2-6.3] 2259 scalars、48.98 dB、3.2 ms full table | current source states可由小per-asset weights表达 | frozen teacher与reference下，direct same-shape fit vs hyper-generated weights vs current shared evaluator；iso-query budget | source/query、program bytes、precision、filter state、seeds | G1/W、peak/energy、compile time、single-query GPU、bytes | generated mainNet被same-shape direct fit支配，或quality loss超过任何runtime gain |
| H4：frozen latent-conditioned NICE可在复杂multi-peak BRDF上低于current GGX9的variance | [P §7.1, Figs.14-15] equal-sample MERL/interpolation gains | expressive flow可覆盖current layered proposal tails，且bounded runtime仍可接受 | frozen same evaluator；cosine、GGX9、NICE；分别实现exact sample/pdf oracle | evaluator、SPP、MIS、scene/env、seed、training queries | PDF normalization、sample→pdf、variance/time、tail weights、MAC/bytes | NICE在equal-time不优于GGX9，或tail/PDF错误、state/cost使其被支配 |
| H5：sampler完全post-train并freeze evaluator/latent能降低joint optimization variance | P先训练NP再训练NICE；current只detach sampler target但仍同步更新shared representation | moving latent是current sampler instability来源之一 | same evaluator initialization比较 simultaneous-detach 与 evaluator complete后 sampler-only | total evaluator/sampler queries、optimizer budget、seeds、proposal shape | evaluator drift、sampler seed success、variance/time、total cook | staged training不降低failure rate/variance，或相同总预算下proposal显著更差 |
| H6：latent neighborhood的energy audit可在不把analytic closure当target的情况下过滤unsafe edits | [P §8.1, Fig.20] interpolation大体合法、extrapolation违规 | current learned edit/compiler state也会在support外失真 | 对同一latent edit path做reference energy integration；对照无audit与bounded refinement | edit path、reference samples、threshold calibration、program | energy p95/max、appearance error、edit latency | audit不能预测unsafe state，或误拒绝大量reference-valid edits |

这些都是 future candidate hypotheses，所有 Zheng 数值只作 report-only prior。实施任一轴都需要新任务与新identity；本次文献任务不授权训练或产品改动。

## 16. 证据索引

### `P` Main paper

- pp.14:1-14:2：问题、贡献、MERL/EPFL isotropic scope。
- §3, Eqs.(1)-(7), Fig.2：function-space probabilistic representation、latent posterior与deterministic mean reconstruction。
- §4.1, p.14:4：Rusinkiewicz 4D circle encoding、four repeated `log1p`、arbitrary sampling pattern。
- §4.2, Fig.3：P encoder/aggregator/decoder topology、mean aggregation、output ReLU。
- §4.3, Eqs.(9)-(12), Algorithm 1：ELBO、KL/L2、context/target与latent sampling；同页latent draw冲突。
- p.14:6 implementation details：MERL100+EPFL51、Adam1e-4、batch16、40k/40h、context1..16200、target16200、`Σerr=0.2I`。
- §5, Figs.4-7：latent dimensionality/semantics、16,384 GGX sweep、stochastic variance、306 novel BRDF与color bias。
- §6.1, Eq.(13)：3.11 MB shared decoder与latent storage amortization。
- §6.2, Eq.(14), Fig.8：hyperNet/mainNet topology、2259/9 KB、60k/20h post-training。
- §6.3, Figs.9-12, Table 1：compression/storage/PSNR、sparse resolution、74/3.2 ms full-table TensorRT timing。
- §7.1, Eqs.(15)-(16), Figs.13-15：NICE topology、target measure、37 KB、8 ms/512² samples、2h training与equal-sample results。
- §7.2, Figs.16-18：latent interpolation、13 SVM traits与editing。
- §8.1, Eq.(17), Figs.19-21：extrapolation、energy和color limitations。

### `S` Supplemental

- P明确引用但当前不可得；无法验证 log-count ablation、更多 novel/compression/importance results、web visualization tool与editing video。

### `C` Official code/assets

- commit `c471c99f1665e036a5813731718162553347b4d2`：
  - `config.py:L4-L41`：40k、batch16、z7、code topology、four-log、0.15 scale、16200 cap；
  - `models.py:L14-L60`：mean/sample latent、11D decoder、dot mask；
  - `models.py:L62-L131`：current encoder/aggregator与per-material latent broadcast；
  - `models.py:L133-L197`：hyperNet/mainNet、teacher L2 trainer；
  - `NPs.py:L19-L74`：公开 CLI 只有compress/decompress/edit/interpolation；
  - `util.py:L9-L75`：coordinate ring map、four-log encode/decode、full-grid path；
  - `coordinateFunctions.py:L52-L177`：MERL/Rusinkiewicz conversion与reciprocity folding；
  - `merlFunctions.py:L5-L42`：UTIA/MERL readers；没有EPFL adaptive loader identity；
  - README：official asset/runtime说明、TF1.12/Python3.6/2080Ti tested environment。
- H5 audit确认 current code encoder/aggregator/decoder/hyper shapes；hash见§2。
- asset audit：151 latents、21 traits；latents为 `(2,1,7)` FP32 mean/logvar。
- GitHub API history：initial code `2388078...`（2023-04-04），hyper addition `c471c99...`（2024-11-25）。

### `A` Author material

- author publication index交叉核对TOG/SIGGRAPH 2022 identity，并显示该条当前只提供paper链接。
- official README将repo明确标为本文source code；没有supplemental/NICE/training release声明。

### `N` NeuralShading evidence

- `research/implications/current-nvidia-correspondence.md` §§1-3：current `functional-f@2` identity、bare-f ABI、old/new evidence isolation。
- `docs/learning.md`：current evaluator/sampler typed routes、GGX9 proposal、detach target与FP16 package合同。
- `docs/realtime_material_compilation.md`：`prepare/evaluate/sample/pdf`、source-native semantics、online reference与bounded runtime。

### `I` 本报告推导

- decoder `808,003` scalars/`805,600` dense MAC、mainNet `2,259` scalars/`2,160` dense MAC、hyperNet generator `1,871,459` scalars来自P/C标注层维度；不等于作者测量的runtime ops。
- §§13-15只在明确标注的 source/query/runtime边界内提出迁移判断，没有把MERL training-set PSNR外推为本项目结果。

## Caveats / Not Found

- 最大缺口是 supplemental、main training code与 NICE implementation/checkpoint均不可得。
- P Figure3与C checkpoint的 encoder/aggregator topology冲突尚无作者 correction裁决。
- P 同页 per-material/per-observation latent draw冲突尚未解析。
- P §6.1的`3.11 MB decoder`与§6.3的`3.10 MB MERL total`无法由未给出的exact serialization bytes裁决。
- C 是2023/2024 later public implementation，不能命名为精确2021 code snapshot。
- TensorRT timings没有precision、engine、single-query或full-renderer context。

## Evidence review

```text
author_worker: /root/lightformer2024_review
reviewer: /root/dualband2025_review
reviewed_at: 2026-08-29
sources_rechecked:
  - author-hosted formal paper, 15 pages, SHA-256 932B693D11BDB9F285D9D56AE7293D0FCB41775585863E773D5C93E872446C44
  - official source snapshot c471c99f1665e036a5813731718162553347b4d2 and README/config/models/inference paths
  - official NP checkpoint, SHA-256 738897DFD0CA2EA5037DDCCB193657408C17EAED77ADC06630E495DB89864845
  - official hyper checkpoint, SHA-256 70EE0D30EFF4EA2ACF8BDBF4BBAC814131596B52D0F7B1CA5A66A9CF8C27D17C
  - official 151 latent-vector and 21 trait-vector assets
findings_closed:
  - verified Figure 3 encoder, aggregator and six-by-400 decoder directly from the rendered formal PDF
  - verified the later checkpoint decoder matches Figure 3 while its encoder and aggregator do not; retained this as a paper-code gap
  - verified Algorithm 1 and Figure 3 use one target-posterior latent per material, while the same-page prose says one draw per observation; retained the contradiction without adjudication
  - verified current code draws one batch-material epsilon and broadcasts the latent across queries, with test_mode selecting the mean
  - clarified that the evaluator returns bare RGB BRDF values and that cosine belongs only to the NICE target; retained missing color-space/unit metadata
  - verified MERL/EPFL counts, context/target sizes, three training stages, Table 1, TensorRT batch scopes and author-negative results
  - recorded the paper-internal 3.11 MB decoder versus 3.10 MB MERL-total inconsistency instead of silently choosing one value
remaining_evidence_gaps:
  - paper-referenced supplemental/web tool/video not publicly located
  - main NP training entry and formal data pipeline absent from official repository
  - NICE sampler code, weights, exact KL estimator and complete PDF measure absent
  - Figure 3 encoder/aggregator differs from official later checkpoint with no author correction
  - per-material versus per-observation latent draw conflict unresolved
  - TensorRT precision, single-query cost and complete evaluation protocol unreported
  - exact serialization bytes behind the paper's 3.11 MB versus 3.10 MB statements unreported
review_status: evidence-reviewed
```

### 完成检查

- [x] main paper 已完整阅读，15页、Eqs.(1)-(17)、Algorithm 1、Table 1、Figs.1-21和关键图注/脚注已视觉核对；
- [x] supplemental/appendix/勘误的可用性已检查，并保留 unavailable 边界；
- [x] official code/config/data 已固定commit并审计architecture、assets、inference与later hyper release；
- [x] architecture、training、runtime和主要结果均有P/C locator；
- [x] author-negative、ablation-inferior和正常Pareto tradeoff已区分；
- [x] paper-code gaps与所有关键“未报告”已保留；
- [x] `I`分析晚于P/S/C/A事实层，没有改写作者结论；
- [x] NVIDIA影响引用current identity与项目合同；
- [x] 假设包含matched control、frozen axes、runtime class和证伪条件；
- [x] 独立 evidence review 已完成。
