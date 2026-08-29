---
paper_id: "2021-neural-brdf-representation-importance-sampling"
title: "Neural BRDF Representation and Importance Sampling"
authors: "Alejandro Sztrajman, Gilles Rainer, Tobias Ritschel, Tim Weyrich"
year: "2021"
venue: "Computer Graphics Forum 40(6), 332–346"
doi: "10.1111/cgf.14335"
report_status: "evidence-reviewed"
main_source: "https://doi.org/10.1111/cgf.14335"
supplemental_status: "available"
official_code_status: "audited"
official_code_commit: "e229dda3308c78f05e57dbc9455326884f766301"
author_worker: "nbrdf2021"
reviewer: "/root/taming2026"
last_verified: "2026-08-29"
---

# Research: Neural BRDF Representation and Importance Sampling

## 1. 研究对象与报告边界

- **Query**：完整重建 2021 年论文的单材质 neural BRDF evaluator、NBRDF autoencoder 与 analytic importance-sampling proxy，区分三者各自的输入、训练和 runtime 角色，并审计官方代码实际公开了哪些部分。
- **Scope**：external + internal。外部证据包括正式论文、两份结果 supplemental、作者项目页、EGSR 2022 slides、官方代码和预训练资产；内部证据只用于第 14 节的 NeuralShading correspondence。
- **Date**：2026-08-29。

本文研究的是 homogeneous measured BRDF 的紧凑表示和 BRDF-driven importance sampling。论文包含三个相关但不能混为一体的模块：[P §3]

1. 每个材质单独训练的 **NBRDF evaluator**：把方向直接映射为 RGB BRDF 值；
2. 以 NBRDF 的 675 个参数为输入/输出的 **autoencoder**：建立 32 维材质 embedding；
3. 从该 embedding 预测两项解析 BRDF sampling 参数的 **importance-sampling proxy**：采样时用解析模型的 inverse CDF，而非反演 evaluator。

本报告不把第三项写成“神经网络直接采样 NBRDF”，也不把 autoencoder 的 32 维 embedding 写成论文在最高保真 evaluator 路径中必然使用的 runtime 资产。论文自己指出：若要保持最大重建质量，仍需保存原始 675 参数 NBRDF；shared decoder 只有在约 105 个以上材质共同使用时才开始抵消其存储开销。[P §4.3, p.340]

论文只处理反射 BRDF；不覆盖空间纹理、footprint/LOD、层参数 compiler、透射 BTDF、场景 visibility 或 environment-product sampling。本报告将它归入 `local-material`，视为方向函数建模和解析 proposal 基线，而不是完整 neural material system。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---:|---|---|
| Main paper `P` | [DOI 正式入口](https://doi.org/10.1111/cgf.14335)；[作者版 PDF](https://asztr.github.io/publications/nbrdf2021/sztrajman2021nbrdf.pdf)，CGF 40(6), 332–346 | 2026-08-29 | SHA-256 `AB83FF0D27F8A46D34C15971965BC5A62085B28FE462778CBF3C2F9A264BDD64` | 方法、实验与结论的最高优先级来源；15 页 |
| Supplemental `S-MERL` | [Supplemental MERL](https://asztr.github.io/publications/nbrdf2021/supplemental/pdfs/supplemental_merl.pdf) | 2026-08-29 | SHA-256 `E32E2BA8F6A51330CFACBC654C87A4D8FF80D1225671CAC22A5EE8BCD2E16A7D` | 单个超长页面，给出全部 100 个 MERL 材质的 GT、各 baseline 重建、SSIM 和 polar plot；没有额外方法配置 |
| Supplemental `S-RGL` | [Supplemental RGL isotropic](https://asztr.github.io/publications/nbrdf2021/supplemental/pdfs/supplemental_rgl-isotropic.pdf) | 2026-08-29 | SHA-256 `948B884B03D5AAD9F5F3A3C6D676F259E5F07704BE3001A558BA924E8AAB8560` | 单个超长页面，给出 51 个 RGL isotropic 材质的 GT、NBRDF、SSIM 和 polar plot；项目页顶部链接被 HTML 注释，但页面下部仍有有效第一方链接 |
| Author project page `A` | [作者项目页](https://asztr.github.io/publications/nbrdf2021/nbrdf.html) | 2026-08-29 | 页面本地快照未作为交付；固定 URL | 论文、supplemental、slides、code、预训练数据和 WebGL demo 的第一方入口 |
| Author slides `A` | [EGSR 2022 slides](https://asztr.github.io/publications/nbrdf2021/sztrajman2021neural-slides.pdf) | 2026-08-29 | SHA-256 `6762A560CD4B9C0F4A1252F7F030AD8EE2A18C09E159DBCDE23130DCA8BFEA89` | 26 页；确认训练时间、模块边界，并明确作者测试 Phong/GGX 后认为 Phong sampling 表现最好 |
| Official GitHub `C` | [asztr/Neural-BRDF](https://github.com/asztr/Neural-BRDF) | 2026-08-29 | commit `e229dda3308c78f05e57dbc9455326884f766301`，2023-02-06；MIT | 审计 evaluator 训练、MERL query、权重导出和 Mitsuba plugin；该 repo 晚于论文，2023 年加入第三方 PyTorch 版本 |
| Publication code archive `C` | [nbrdf_code.zip](https://asztr.github.io/publications/nbrdf2021/supplemental/nbrdf_code.zip) | 2026-08-29 | SHA-256 `A2EE7E74571C3E85F3C20BCC7A7C62B1C841DA437D8A752907E6EA4B389FB716` | 无 commit metadata；核心 Keras/Mitsuba 文件与 GitHub 最新版一致，除 `fastmerl.py` 作者注释外；没有 autoencoder 或 sampling-predictor 代码 |
| Alternative PyTorch archive `C` | [pytorch_code.zip](https://asztr.github.io/publications/nbrdf2021/supplemental/pytorch_code.zip) | 2026-08-29 | SHA-256 `36B2041EE4D2DBEF3F9EC55F6A5485A2F4B05D7635AED7A9B95B6F6DC24B23E4` | 作者页明确称 Michael Fischer 的 alternative implementation；不是论文原始 formal implementation |
| Official pretrained data `C` | [MERL](https://asztr.github.io/publications/nbrdf2021/supplemental/data/merl_nbrdf.zip) / [RGL isotropic](https://asztr.github.io/publications/nbrdf2021/supplemental/data/rgl-isotropic_nbrdf.zip) / [Nielsen](https://asztr.github.io/publications/nbrdf2021/supplemental/data/nielsen_nbrdf.zip) | 2026-08-29 | 分别为 SHA-256 `D1678C0CC5059F315D3D3284F36F3082787474106D72E3CAE41FBAD76D8F271A`、`401F588F08A66714EBD9D1355298F85E4D2F3782C9A87F370538DC8657CF5389`、`FD034C57B2CBE0E976B088EA4BB7CCBDDA3782DAD90946243124E07316442166` | 分别含 100、51、8 个 Keras NBRDF；MERL 包另带一个材质的 6 个 `.npy` runtime 数组；不含 11 个 RGL anisotropic NBRDF |
| NeuralShading evidence `N` | [faithful NVIDIA correspondence](../../../archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md)；[scattering backend contract](../../../../../docs/contracts/scattering_backend.md)；[compiler contract](../../../../../docs/realtime_material_compilation.md) | 2026-08-29 | repo-local | 只用于第 14 节；不得回填成 2021 论文事实 |

### 2.1 来源可用性结论

- 正文、两份结果 supplemental、作者 slides、evaluator 代码、Mitsuba plugin 和预训练 evaluator 都真实可用。
- 可得 supplemental 是结果 mosaic，不是方法 appendix；没有新增 optimizer、autoencoder activation、sampling-predictor loss 或精确 sampler 公式。[S-MERL; S-RGL]
- GitHub 与发布 zip 都没有 NBRDF autoencoder 和 embedding→analytic-parameter predictor 的训练/推理代码；不能从 evaluator 脚本补猜这两个模块。
- 官方训练脚本只实现 isotropic MERL binary query。论文中的 anisotropic RGL 训练路径和预训练权重没有公开。[C commit `e229dda`, `binary_to_nbrdf/coords.py:L53-L61`; official data archives]
- 未发现作者勘误。项目页提供的 slides 晚于论文约一年，应作为 `A` 证据，不能静默覆盖 `P`。

## 3. 原论文的问题、假设与贡献边界

论文从 measured BRDF 的三方权衡出发：tabular data 准确但约 34 MB/材质且需要 angular interpolation；解析模型极小且快，但拟合 real-world appearance 的表达力有限，非线性拟合也可能慢且不稳定。作者目标是得到能直接放进 renderer 的连续、紧凑、高保真 BRDF 表示，并给这种表示补上可执行的 importance sampling。[P §1, pp.332–333; A slides 3–7]

作者声明三项贡献：[P §1, pp.332–333]

1. 一个单材质、浅层、fully connected NBRDF，支持 arbitrary angular samples，因此可以在训练时把样本集中到 specular highlight；
2. 一个“learning-to-learn”autoencoder，把 675 个 NBRDF 参数编码成 32 维 latent，并展示材质 embedding 与 interpolation；
3. 一个从 embedding 到可逆解析 BRDF sampling 参数的 learned mapping，使 sampler 使用已知 inverse CDF。

关键假设是 proposal PDF 不必精确等于 evaluator 的 BRDF；只要实际 sampling distribution 与报告的 PDF 一致，Monte Carlo estimator 仍保持无偏，差异只影响 variance。[P §2.2, p.333; §3.3, pp.336–338]

贡献边界：evaluator 不是一个跨材质共享 decoder；每个材质重新训练一份 675 参数网络。跨材质共享只发生在 autoencoder/parameter-predictor 这条辅助路径，而论文明确不建议为了最高 evaluator fidelity 丢掉原 NBRDF。[P §4.3, p.340]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | measured BRDF table；主体为 MERL，另测 RGL 与 Nielsen | MERL：100 材质，每材质 `90×90×180×3 ≈ 4.4M` 值、约 34 MB | [P §3.4, p.338] |
| Runtime evaluator query | incoming/outgoing hemispherical directions转换为 Rusinkiewicz half/difference variables | NBRDF 输入为 `h,d` 两个 Cartesian unit vectors，共 6 scalars | [P §3.1, p.334, Fig.1] |
| Isotropic coordinates | `θ_h, θ_d, φ_d`；`φ_h` 被各向同性省略 | `θ_h,θ_d∈[0,π/2]`，`φ_d` 训练采样覆盖 `[0,2π)`；MERL table 存 `[0,π)` 并由 reciprocity 补齐 | [P §3.1, p.334; §3.4, p.338]; [C `coords.py:L53-L77`] |
| Anisotropic coordinates | 额外依赖 `φ_h`，训练时均匀随机采四个 Rusinkiewicz angles | 仍以 `h,d∈R³` 输入 6D network；有效 BRDF domain 有四个角自由度 | [P §3.1, p.334] |
| Output quantity | RGB reflectance/BRDF value `f_r(h,d)` | 3 个非空间、非 filtered 的 RGB scalars | [P §3.1, Fig.1] |
| Loss target | cosine-weighted reflectance `f_r cos θ_i` 的 `log(1+x)` | RGB；正文写绝对范数，公开 Keras 代码对 batch/channel 求 mean absolute | [P Eq.(1), p.334]; [C `binary_to_nbrdf.py:L31-L54`] |
| Importance query | 对固定方向从一个近似解析 PDF 采另一方向，并可计算对应 PDF | 论文只在 isotropic MERL 上正式评测；解析 PDF monochrome | [P abstract; §3.3, pp.336–338] |
| Validity restrictions | incident/outgoing 均在 surface upper hemisphere | 训练脚本将无效组合 mask 为 0 后过滤；Mitsuba plugin 对任一方向 `cos≤0` 返回 0 | [C `binary_to_nbrdf.py:L80-L99`; `mitsuba/bsdfs/nbrdf_npy.cpp:L50-L54`] |

### 4.1 Rusinkiewicz 变换的实际编码

对一对局部方向，half vector 为 `h = normalize(w_i + w_o)`；difference vector 是把 `w_i` 依次绕 normal 旋转 `-φ_h`、绕 binormal 旋转 `-θ_h` 后得到的向量 `d`。公开代码的 `io_to_hd()` 明确实现这一变换。[C commit `e229dda`, `binary_to_nbrdf/coords.py:L6-L27`]

isotropic Keras trainer 不从任意 `w_i,w_o` 开始，而是直接均匀生成 `(θ_h,θ_d,φ_d)`，固定 `φ_h=0`，再构造：

```text
h = (sin θ_h, 0, cos θ_h)
d = (sin θ_d cos φ_d, sin θ_d sin φ_d, cos θ_d)
```

[C `binary_to_nbrdf/coords.py:L53-L61`]

正文称该参数化比 Rainer et al. BTF 工作使用的 direction stereographic projection 更适合 dense homogeneous BRDF 的 specular highlight；Figure 2 给出后者把 homogeneous BRDF 当 spatially uniform BTF 时的明显高光重建损失。[P §3.1, p.334, Fig.2]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

#### 5.1.1 Evaluator 路径

```text
measured BRDF table
  → random Rusinkiewicz angle queries
  → trilinear interpolation of measured values
  → (h,d) Cartesian 6D + RGB f_r
  → per-material MLP training with log1p(f_r cosθ_i) L1
  → persist 675 FP32 scalars
  → runtime MLP(h,d) = RGB f_r
```

[P §3.1; C `binary_to_nbrdf.py`, `fastmerl.py`]

#### 5.1.2 Embedding 路径

```text
pretrained 675-scalar NBRDF
  → flatten network weights/biases
  → encoder 675→675→32
  → z_NBRDF∈R^32
  → decoder 32→100→675
  → instantiate reconstructed NBRDF
  → differentiable 64×64 sphere rendering
  → gamma-space image MSE
```

[P §3.2, pp.335–336, Fig.3]

autoencoder 的监督不是对 675 个 scalars 做逐元素误差，而是把 decoder 输出重新解释成 NBRDF 权重，渲染 sphere 后计算 image loss。作者明确报告 weight-space loss 不能恢复原 appearance。[P §3.2, p.335]

#### 5.1.3 Sampling-proxy 路径

```text
z_NBRDF (32)
  → shallow predictor 32→8→2
  → monochrome analytic BRDF/PDF parameters
     (roughness, diffuse/specular relative weight)
  → analytic Blinn–Phong CDF^{-1}
  → sampled direction + matching analytic PDF
```

[P §3.3, pp.336–338, Fig.4]

这条路径没有对 NBRDF 做 inverse-CDF 数值反演。evaluator 仍是 NBRDF 或原 measured table；sample/pdf 是一个可解析 proposal。Blinn–Phong 完整 reflectance 有 7 个参数，但其 monochrome sampling PDF 只需与 roughness、diffuse/specular mixture weight 对应的 2 个参数。[P §3.3, p.338]

### 5.2 持久化表示

| 表示 | per-material 持久化内容 | shared 内容 | 质量/成本边界 |
|---|---|---|---|
| 原始 NBRDF | 675 个 FP32 weights+biases，论文计 2.70 KB | 无 | 最高质量 evaluator；每材质单独训练 [P Fig.1; Table 3] |
| AE embedding | 32 scalars | decoder `32→100→675` | decoder 内存约等于 105 个原始 NBRDF；decoded appearance 有不可避免的退化 [P §4.3] |
| Sampling proxy | 2 个解析 sampling 参数；其存储时点未报告 | `32→8→2` predictor | 只定义 monochrome proposal；不是 RGB evaluator [P §3.3, Fig.4] |

没有 texture/plane/grid、mip、LOD、quantization 或 spatial latent。论文没有给 per-material 32D embedding 和 2D sampling params 的确切 dtype/bytes；图与代码上下文均以普通浮点叙述，但不能据此补写 formal precision。

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| NBRDF | `(h,d)∈R⁶` | Dense `6→21` | ReLU；无 normalization | 21 | per-material | [P Fig.1]; [C `binary_to_nbrdf.py:L56-L64`] |
| NBRDF | 21 | Dense `21→21` | ReLU；无 normalization | 21 | per-material | 同上 |
| NBRDF | 21 | Dense `21→3` | paper：final exponential；Keras：linear 后 `exp(x)-1` | RGB `f_r` | per-material | [P §3.1, Fig.1]; [C `binary_to_nbrdf.py:L58-L61`] |
| NBRDF runtime | 3 preactivation | `max(exp(x+b)-1,0)` | 显式 nonnegative clamp | RGB `f_r` | per-material | [C `mitsuba/bsdfs/nn.h:L63-L89`] |
| AE encoder | 675 | Dense `675→675→32` | activation、bias 与 normalization **未报告** | `z_NBRDF∈R³²` | shared | [P Fig.3, p.335] |
| AE decoder | 32 | Dense `32→100→675` | activation、bias 与 normalization **未报告** | reconstructed NBRDF params | shared | [P Fig.3, p.335] |
| Sampling predictor | 32 | Dense `32→8→2` | activation、bias、output parameter transform **未报告** | 2 analytic params | shared | [P Fig.4, p.335] |
| Analytic sampler | 2 params + uniform random variates | Blinn–Phong inverse CDF | closed form；精确公式 **未报告** | direction + matching solid-angle PDF | analytic | [P §3.3] |

`6→21→21→3` 的 matrices 有 630 个 weights，bias 共 45 个，因此总计 675 trainable scalars；论文把它简写为 `6×21×21×3`。[P Figs.1,3,6; I 按图示维度求和]

Figure 3 显示的 full autoencoder 按 dense+bias 计算约 549,407 个参数，其中 decoder 约 71,475 个，等于约 105.9 个 675-scalar NBRDF；这一计算与作者“decoder roughly equivalent to 105 NBRDFs”的文字一致。[P Fig.3; §4.3; I 维度计算]

### 5.4 条件化、坐标变换与物理先验

- **高光先验**：half/difference coordinates 把镜面结构对齐到 `h,d` domain；作者用 Figure 2 说明一般 direction projection 对 dense BRDF 高光较差。[P §3.1]
- **动态范围先验**：对 `f_r cosθ_i` 做 `log1p` L1，而不是 raw `f_r` L2。[P Eq.(1)]
- **sample/eval 解耦**：解析 PDF 只逼近 BRDF 形状，sample 与 PDF 自身严格对应即可保持 estimator 无偏。[P §3.3]
- **没有显式物理约束**：论文/代码没有报告 reciprocity、energy conservation 或 white-furnace constraint。Rusinkiewicz parameterization 是坐标先验，不能自动当成这些物理性质的证明。
- **没有 learned frame/offset**：方向 canonicalization 完全由固定 Rusinkiewicz 变换完成。

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset/source assets | MERL 100 个 isotropic measured BRDF；RGL 51 isotropic + 11 anisotropic；Nielsen 数据用于额外 qualitative reconstruction | [P §3.1, §3.4, Fig.5]; [C official data archives] |
| GT/reference | measured table 的 interpolated RGB BRDF；MERL Keras reader 对 `θ_h,θ_d,φ_d` 做 3D interpolation | [P §4.1, p.338]; [C `fastmerl.py:L116-L162`] |
| Evaluator train/test queries | 正文：每材质总计 `8×10^5` random samples；公开代码分别生成约 640k train 与 160k test，均为独立 random draws，再剔除 invalid/all-zero 与 red<0 rows | [P §3.1, p.334]; [C `binary_to_nbrdf.py:L89-L99,L119-L127`] |
| Isotropic angular sampling | `θ_h,θ_d,φ_d` 分别 uniform random；这种“uniform in Rusinkiewicz angles”会相对集中到 specular region | [P §3.1; §4.1, p.339] |
| Anisotropic angular sampling | 对四个 Rusinkiewicz angles uniform random；样本总数增加 5 倍 | [P §3.1, p.334; A slide 11] |
| MERL table sampling | `θ_h` 90 个 inverse-square-root grid points，`θ_d` 90 uniform，`φ_d` 180 uniform；另一半 azimuth 用 Helmholtz reciprocity | [P §3.4, p.338] |
| Autoencoder material split | MERL 材质 80% train / 20% test；train 材质做 RGB channels 的全部 6 种 permutation | [P §3.2, p.335] |
| Autoencoder image supervision | `64×64` sphere；non-frontal directional light `θ_l=45°`；`γ=2.2`；低值 clamp 到 `10^-12`；pixel MSE | [P §3.2, p.335] |
| Importance-predictor split | MERL 的一个 subset 做训练，20 个 unseen MERL test materials 做 Figure 13–16；精确材质名单与是否复用 AE split **未报告** | [P §4.4, pp.341–343, Fig.14] |
| Importance labels | Blinn–Phong fitted parameters来自 Ngan et al. 2005；GGX labels 来自 Bieron & Peers 2020 | [P §4.4, p.341] |
| Filtering/LOD/footprint | 不适用；homogeneous directional BRDF，没有 spatial footprint | method domain |
| Online/offline generation | measured table 是离线数据；公开训练脚本在训练前一次性生成 DataFrame query/GT，不在每 optimizer step 在线重采 | [C `binary_to_nbrdf.py:L66-L99,L119-L127`] |

### 6.1 `adaptive` 与通常自适应采样的区别

论文把 uniform random **Rusinkiewicz angles** 称为 BRDF-aware adaptive angular sampling，因为坐标变换本身提高 specular vicinity 的密度；代码没有基于当前误差、材质 CDF 或训练迭代动态更新 sampling distribution。[P §3.1; C `generate_nn_datasets()`]

作为对照的 `NBRDF Uniform Sampling` 是在更常规/regular angular domain 取样，使 diffuse component 被更有效覆盖，但明显损失高光。正文没有给这一路径的完整代码或精确生成公式，因此不能把它补写为“uniform hemisphere”或某个固定 grid。[P §4.1, p.339]

### 6.2 公开代码中的随机性细节

Keras 函数 `generate_nn_datasets(brdf, ..., seed=...)` 接收 `seed`，但函数体完全没有使用它；`main()` 传入的 2 和 3 因而不起作用。`np.random.seed(0)` 又在 model 已构造、datasets 已生成之后才调用，因此不控制 query draw，也不明确控制 Keras initialization。[C `binary_to_nbrdf.py:L66-L75,L89-L99,L121-L126`]

2023 年加入的 PyTorch alternative 在文件开头设置 NumPy/Torch seed 0，并固定 uniform initializer `[-0.05,0.05]`；它是后续第三方实现，不应回填成 2021 formal seed protocol。[C `binary_to_nbrdf/pytorch_code/train_NBRDF_pytorch.py:L14-L26,L34-L57`]

## 7. Loss、optimizer 与训练 lifecycle

### 7.1 Per-material NBRDF evaluator

对 query `q=(h,d)`，论文 objective 为：

\[
\mathcal L(q)=\left\lVert
\log\bigl(1+f_r^{true}(q)\cos\theta_i\bigr)-
\log\bigl(1+f_r^{pred}(q)\cos\theta_i\bigr)
\right\rVert.
\]

[P Eq.(1), p.334]

公开 Keras 实现先由 `h,d` 反解 `θ_h,θ_d,φ_d`，计算 `w_i.z = cosθ_d cosθ_h - sinθ_d cosφ_d sinθ_h`，clamp 到 `[0,1]`，然后对 RGB 和 batch 求 mean absolute `log1p` difference。[C `binary_to_nbrdf.py:L31-L54`]

| 项 | 正式配置/可得代码配置 | locator |
|---|---|---|
| Target/output transform | target 为 `f_r cosθ_i`；network raw output 经过 paper 所称 exponential；Keras 实为 `exp(x)-1` | [P Eq.(1), Fig.1]; [C `binary_to_nbrdf.py:L56-L63`] |
| Loss | mean absolute difference in `log1p(cosine-weighted reflectance)` | [P Eq.(1)]; [C lines 31–54] |
| Optimizer | Paper 未报告；公开 Keras：Adam，`lr=5e-4, β=(0.9,0.999), epsilon=None, decay=0, amsgrad=false` | [C `binary_to_nbrdf.py:L25-L29,L62-L63`] |
| LR schedule | Paper/代码均未报告 schedule；代码固定 learning rate | [C] |
| Batch/query count | `8×10^5` total query；Keras `batch_size=512` | [P §3.1]; [C lines 26,89–99] |
| Steps/epochs | Paper：diffuse 约 5 epochs stabilise，最 mirror-like 最多 90；Keras 默认 100 epochs，无 early stopping | [P §3.1]; [C lines 27,66–78] |
| Initialization | Paper 未报告；Keras `random_uniform` kernels、zero bias；pretrained JSON 固定 Keras default range `[-0.05,0.05]`、seed null | [C `binary_to_nbrdf.py:L56-L61`; official pretrained JSON] |
| Seed/model selection | Paper 未报告；Keras seed wiring 如 §6.2，未提供 checkpoint selection | [C] |
| Hardware/time | GPU 型号未报告；每材质 10 s–3 min，diffuse→mirror-like | [P §3.1; A slide 6] |

### 7.2 NBRDF autoencoder

| 项 | 正式配置 | locator |
|---|---|---|
| Target transform | decoder 输出 675 scalars，被重新实例化为 NBRDF；不是直接重建 weights 的 loss | [P §3.2] |
| Loss | gamma-2.2 sphere rendering的 point-wise MSE，low values clamp `10^-12` | [P §3.2] |
| Optimizer/schedule | **未报告** | source gap |
| Batch/steps/epochs | **未报告** | source gap |
| Initialization/seed/model selection | **未报告** | source gap |
| Hardware/training time | **未报告** | source gap |

作者明确试过直接匹配输入/输出 NBRDF weights 的 non-image-based loss，并报告它无法重建 encoded materials 的原 appearance。这是有定位的 `author-negative`，不是本项目推测。[P §3.2, p.335]

### 7.3 Importance-sampling parameter predictor

| 项 | 正式配置 | locator |
|---|---|---|
| Target | fitted Blinn–Phong 的 2 个 sampling-relevant params；另做 GGX labels 实验 | [P §3.3; §4.4] |
| Loss | **未报告** | source gap |
| Optimizer/schedule | **未报告** | source gap |
| Batch/epochs/seed | **未报告** | source gap |
| Hardware/training time | **未报告** | source gap |
| Joint vs staged training | 论文叙述先有 pretrained NBRDF、再训练 AE、再从 embedding 预测；是否端到端回传到 AE **未明确报告** | [P §§3.2–3.3] |

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path | 每次 BRDF query 做固定 `6→21→21→3` MLP；Mitsuba plugin 把 `h,d` 送入 C++ `Net::forward` | [C `nbrdf_npy.cpp:L50-L72`; `nn.h:L63-L89`] |
| Parameter count | 675 trainable scalars；630 matrix coefficients + 45 biases | [P Fig.1/Table 3; I 维度计算] |
| MAC/ops | Paper 未报告；由 dense 维度计算为 630 MAC/query，另有 42 hidden ReLU、3 exponentials 和 coordinate transform | [I；P Fig.1] |
| Shared/per-asset bytes | NBRDF 2.70 KB/material；AE decoder 约等于 105 NBRDF，32D code bytes 未单列 | [P §4.3; Table 3] |
| Texture/feature fetches | NBRDF weights从普通 arrays 读取；无 neural texture/LOD | [C `nn.h`] |
| Precision/quantization | 预训练 Keras与 C++ arrays为 float32；没有量化实验 | [C pretrained JSON; `nn.h`] |
| Hardware/backend | CPU Mitsuba，Intel Core i9-9900K；NBRDF+PhongIS 为 unoptimized implementation | [P §4.5, pp.343–344] |
| Throughput | NBRDF+PhongIS `12.50×10^6 rays/s` | [P Table 3] |
| Prepare/amortization | homogeneous material 没有 per-shading-point `prepare()`；AE/predictor 是否 runtime 每次执行或离线预计算未报告 | source gap |

### 8.1 公开 runtime 与论文 sampler 不同

官方 Mitsuba plugin 的 evaluator 是论文 NBRDF，但 sampler 不是 Figure 4 的 learned Blinn–Phong proxy。构造函数把 neural evaluator 交给 `djb::tabular(..., 90, true)` 建表，`sample()`/`pdf()` 调用该 tabulated sampler。[C commit `e229dda`, `mitsuba/bsdfs/nbrdf_npy.cpp:L20-L32,L75-L105`]

因此公开 plugin 不能作为 Table 3 中 `NBRDF + PhongIS` 的完整 code correspondence；它只证明 NBRDF evaluator 可被 Mitsuba 调用。文件还明确写着 GPU shader implementation not implemented。[C `nbrdf_npy.cpp:L144-L147`]

作者项目页另有 WebGL demo，称把预训练 weights 存入 texture 并用 GLSL inference；但该 GLSL 源码不在官方 GitHub/release archive，且作者未报告 WebGL latency。[A project page]

### 8.2 Paper 内横向性能表

| Representation | Rays/s (`×10^6`) | Memory KB | 说明 |
|---|---:|---:|---|
| Bagher et al. 2012 | 10.64 | 0.13 | analytic SGD fit |
| RGL / Dupuy & Jakob 2018 | 10.66 | 48.0 | adaptive data representation |
| NBRDF + PhongIS | 12.50 | 2.70 | 675-scalar network，unoptimized |
| Cook–Torrance | 13.59 | 0.03 | analytic |
| Dupuy et al. 2015 | 14.05 | 2.16 | microfacet extraction |
| Low et al. 2012 | 15.13 | 0.03 | ABC |
| GGX | 16.82 | 0.03 | analytic |
| NPF 2016 | 未报告 | 3.20 | 表中无 throughput |

[P Table 3, p.344]

这些数值只在同论文的 CPU/Mitsuba/i9-9900K protocol 内可比，不能与现代 GPU shader 的 per-query latency 直接排名。

## 9. 实验 protocol、baseline、指标与结果

### 9.1 MERL reconstruction

Figure 6 使用 environment-map scene，把 measured MERL table interpolation 作为 GT；两种 NBRDF 都固定 675 scalars。Table 1 报告对全部 100 个 MERL 材质的 image-based aggregate：[P §4.1, Figs.6–7, Table 1]

| Method | MAE | RMSE | SSIM |
|---|---:|---:|---:|
| NBRDF Adaptive Sampling | **0.0028 ± 0.0034** | **0.0033 ± 0.0038** | **0.995 ± 0.008** |
| NBRDF Uniform Sampling | 0.0072 ± 0.0129 | 0.0078 ± 0.0134 | 0.984 ± 0.029 |
| NPF (Bagher et al. 2016) | 0.0056 ± 0.0046 | 0.0062 ± 0.0047 | 0.990 ± 0.008 |
| ABC (Low et al. 2012) | 0.0080 ± 0.0070 | 0.0088 ± 0.0075 | 0.986 ± 0.012 |
| SGD (Bagher et al. 2012) | 0.0157 ± 0.0137 | 0.0169 ± 0.0145 | 0.974 ± 0.027 |
| Dupuy et al. 2015 | 0.0174 ± 0.0143 | 0.0190 ± 0.0151 | 0.976 ± 0.021 |
| GGX | 0.0189 ± 0.0118 | 0.0206 ± 0.0126 | 0.969 ± 0.024 |

论文没有明确定义 `±` 是跨材质 standard deviation、standard error 还是另一统计量；报告保留原数值而不补写统计含义。Figure 10 显示 adaptive NBRDF 对几乎所有材质最好，但作者明确保留“少数 highly specular materials”例外。[P §4.1, pp.339–340]

作者对 baseline 的定性观察：[P §4.1, pp.338–339]

- GGX 容易 blur highlights；
- Bagher et al. 2012 可产生准确高光，但 diffuse albedo 偏低；
- NPF 总体次优，但 grazing angles fitting error 上升，并会出现 unusually long tails；
- NBRDF adaptive 能在 grazing-angle polar plots 中更接近 GT。

`S-MERL` 视觉核对了全部 100 个材质的 GT、七个 reconstruction columns、每材质 SSIM 与 polar plots；它没有提供 Table 1 之外的新 aggregate 数字。[S-MERL]

### 9.2 Autoencoder 与 PCA

同一 MERL 80/20 material split、同为 32D encoding 时：[P §4.3, Table 2]

| Method | MAE | RMSE | SSIM |
|---|---:|---:|---:|
| NBRDF AE | **0.0178 ± 0.013** | **0.0194 ± 0.014** | 0.968 ± 0.031 |
| Nielsen-style PCA | 0.0199 ± 0.008 | 0.0227 ± 0.009 | **0.982 ± 0.007** |

该结果不是单向“AE全面胜出”：AE 的 MAE/RMSE 更低，而 PCA 的 SSIM 更高且 `±` 项更小。论文随后用 t-SNE 展示类似 albedo/shininess 的 materials 在 32D latent 中聚类，并用 Figure 12 展示 embedding linear interpolation 产生视觉平滑的中间材质。[P §4.3, Figs.11–12]

### 9.3 RGL anisotropic 与跨数据集结果

- 11 个 RGL anisotropic materials 使用与 isotropic 相同的 675-scalar network；平均 SSIM 为 `0.981 ± 0.016`。作者称 anisotropic 结果的视觉差异更明显，但可通过加大网络降低误差。这个数值与 MERL isotropic 的 `0.995 ± 0.008` 来自不同数据集，不能据此作受控的直接排序。[P §4.2, Fig.9; I 跨数据集比较边界]
- Figure 9 的四个实例为 `copper-sheet 0.954`、`green-pvc 0.987`、`morpho-menelaus 0.987`、`sari-silk-2color 0.996`。[P Fig.9]
- `S-RGL` 展示 51 个 RGL isotropic materials 的 GT/NBRDF/SSIM/polar plots；没有新增 aggregate table。[S-RGL]
- Figure 5 对 RGL 和 Nielsen datasets 给出 qualitative rendering；paper 没有报告 Nielsen aggregate metric。[P Fig.5]

### 9.4 Importance sampling：Veach scene

Figure 13 把所有 reflectance evaluations 保持为 original tabulated MERL data，只改变 sampling strategy。GT 用 6400 spp，其他方法 64 spp；测试材质对 predictor 未见。[P §4.4, Fig.13]

| Material | Uniform RMSE | Dupuy et al. 2015 | Phong Fit | Predicted Phong (`NBRDF AE`) |
|---|---:|---:|---:|---:|
| gold-metallic-paint3 | 0.00671 | 0.00512 | 0.00388 | **0.00374** |
| dark-blue-paint | 0.00258 | 0.00258 | 0.00259 | 0.00258 |
| purple-paint | 0.00302 | 0.00302 | **0.00299** | 0.00303 |
| green-metallic-paint2 | 0.00551 | 0.00412 | 0.00367 | **0.00348** |
| aluminium | 0.0141 | 0.00853 | 0.00415 | **0.00334** |

[P Fig.13, visually verified]

这组实例既有 predicted params 优于 fit，也有相等或略差；不能从图中推成逐材质支配。

Figure 14 对 20 个 test materials 聚合 MAE、RMSE、MAPE、PSNR 随 spp 的曲线，baseline 包括 Uniform、Phong Fit、predicted Phong、GGX Fit、predicted GGX、Dupuy et al. 2015。两种 analytic-driven family 都明显领先 Uniform，predicted curves 接近 fitted curves；paper 没给曲线点的数值表。[P Fig.14]

作者说明更准确的 analytic BRDF fit 不保证更好的 proposal，因为它可能更精确追 specular lobe，却忽略 sheen 等其它能量；因此最终选 Blinn–Phong 是 sampling tradeoff，不是把 NBRDF 降为 Blinn–Phong evaluator。[P §4.4, p.341]

### 9.5 Importance sampling：kitchen scene

Figure 15 将场景大部分 materials 换为 20 个 MERL test materials；GT 6400 spp，比较图 64 spp。图中 crop RMSE 为：[P Fig.15]

- Uniform：`0.024`；
- Dupuy et al. 2015：`0.010`；
- predicted NBRDF/Phong proposal：`0.006`。

Figure 16 进一步画 MAE、RMSE、MAPE、PSNR 对 spp 与 render time 的曲线；predicted proposal 在该场景中保持最低 MAE/RMSE，并在时间轴上领先 Uniform 与 Dupuy。曲线没有公开 tabular data 或 confidence interval。[P Fig.16]

### 9.6 Size–quality tradeoff

Figure 17 同时画 average SSIM 与 memory weights。作者结论：[P §4.5, Fig.17]

- 约 100 scalars 的 NBRDF reconstruction 不准确，优先使用 parametric representation；
- 约 300 scalars 已优于 best parametric ABC，并与 NPF 相当；
- 675-scalar point 位于 NBRDF 曲线靠右但不是最大网络，网络尺寸可作为 quality/storage 旋钮。

论文没有给 Figure 17 每个 NBRDF 点的精确 layer widths，也没有公开生成所有 size points 的 config。

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `ablation-inferior` | adaptive Rusinkiewicz sampling → uniform/regular angular sampling | MERL MAE `0.0028→0.0072`，RMSE `0.0033→0.0078`，SSIM `0.995→0.984`；表中 `±` 数值更大，但其统计含义未报告 | regular samples 更高效覆盖 Lambertian component，却不足以解析 highlight | `[I]` query distribution 与 network size 同等 load-bearing；不能把两列差异归因于 MLP | [P Table 1; §4.1] |
| `author-negative` | 用 Rainer et al. BTF stereographic direction projection 表示 homogeneous BRDF | Figure 2 丢失/模糊 specular highlights | 该坐标更适合 anisotropy、inter-shadowing、masking，不适合 dense uniform BRDF 高光 | `[I]` 这是 coordinate prior 的失败，不是“neural BTF模型普遍失败” | [P §3.1, Fig.2] |
| `author-negative` | autoencoder 直接匹配 675 个 NBRDF weights | 无法重建输入材质 appearance | 不同 network parameter vectors 存在不对应 perceptual/function distance 的自由度 | `[I]` weight-space compiler 需要 functional alignment 或 canonicalization | [P §3.2] |
| `author-negative` | 直接线性插值两个独立 NBRDF 的 675 scalars | specular properties 过渡不平滑 | independently trained networks 不处于一致 parameter basis | `[I]` 与 permutation/symmetry of hidden units 一致，但论文未作此因果证明 | [P §4.3, Fig.12 bottom] |
| `ablation-inferior` | NBRDF AE 32D vs PCA 32D | AE MAE/RMSE 更好，SSIM 更差（0.968 vs 0.982） | 作者不把 AE 定位成无损 compression，主要价值是 embedding | `[I]` 不应只挑 MAE/RMSE 宣称全面胜出 | [P Table 2; §4.3] |
| `known-limitation` | NBRDF `<100` scalars | reconstruction inaccurate；作者建议用 parametric models | capacity 太小 | `[I]` 是该 dataset/protocol 的 observed crossover，不是通用 hard gate | [P §4.5, Fig.17] |
| `known-limitation` | 675-scalar anisotropic vs isotropic | average SSIM 降至 `0.981±0.016`，视觉差异增加 | additional DOF；训练 query 增加 5 倍 | `[I]` 固定容量对 4D domain 更紧张 | [P §3.1; §4.2] |
| `known-limitation` | 675→32 AE compression | decoder overhead约 105 NBRDF，并发生 appearance degradation | embedding/interpolation 是主用途，不是最高保真存储 | 无额外解释 | [P §4.3] |
| `ablation-inferior` (`A`) | 对 sampling proxy 测试 Phong 与 GGX | 作者 slides 称 Phong performed best | Phong 两参数、inverse CDF 简单；main paper另指出复杂 fit 不必然带来更好 sampling | `[I]` slides 未给 matched 数值表，不能写成 GGX 普遍失败或作者已证明的失败尝试 | [A slide 18; P §4.4] |
| `baseline-inferior` | NPF grazing lobes | high-grazing error 增加，部分材质 tail 过长 | NPF functional factorization 在这些方向失配 | 不扩展因果 | [P §4.1, Figs.7–8] |

在已获得第一方材料中，没有报告：多 seed instability、optimizer failure、不同 AE architecture、不同 latent dimensionality、different predictor losses、energy/reciprocity regularization 或 neural direct inverse-CDF 的失败实验。它们不能从最终方法反推为作者尝试过。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| NBRDF architecture | `6→21→21→3`、ReLU、final exponential、675 scalars | 无新增配置 | Keras精确实现 `exp(x)-1`；C++再 clamp nonnegative | 主结构对应；paper 的“exp”比代码欠精确，runtime 比 Keras 多一个 clamp |
| Autoencoder | `675→675→32→100→675`，rendering loss | 无新增配置 | **未公开** | `paper-code-gap`：不能复现 activation、optimizer、split identity 和 checkpoint |
| Sampling predictor | `32→8→2`，Blinn–Phong/GGX labels | 无新增配置 | **未公开** | `paper-code-gap`：论文主要 sampling claim 无 formal code/config |
| Direction query | isotropic 3 angles；anisotropic 4 angles/5× samples | 只展示结果 | trainer 固定 `φ_h=0`，只支持 MERL-style isotropic binary | isotropic correspondence基本成立；anisotropic path缺失 |
| Query split/seed | 800k samples，未给 seed | 无 | 640k/160k independently generated；`seed` 参数未使用 | `paper-code-gap`：公开脚本不可按参数复现 query stream |
| Evaluator loss | Eq.(1) log cosine-weighted L1 | 无 | Keras完整实现 | 对应 |
| Evaluator optimizer | 未报告 | 无 | Keras Adam `5e-4`、batch512、100 epochs | 只能标 `C`，不能称 paper 正式披露 |
| Pretrained framework | 未报告 Keras minor version | 无 | README称测试 Keras 2.2.5/TF-GPU1.13.1；预训练 JSON 写 Keras 2.2.4 | 版本小冲突，保留两者 |
| Runtime evaluator | 可直接替换 renderer BRDF | 无 | C++ CPU evaluator固定实现 | 对应，但 Mitsuba `eval()`按旧接口返回 `f_r cosθ_o` |
| Runtime sampling | learned analytic Blinn–Phong proxy | 无 | plugin 实际用 `djb::tabular` 对 NBRDF 建表采样 | **实质 `paper-code-gap`**；公开 renderer不复现论文 sampler |
| GPU/demo | paper只声称 practical rendering | 无 | C++写明 GPU shader 未实现；项目页称 WebGL/GLSL demo，但源码未公开 | 不能从 demo 外推 shader benchmark |
| Data assets | MERL、RGL、Nielsen；RGL含11 anisotropic | 展示 MERL100/RGL51 iso | 预训练包为 MERL100、RGL51 iso、Nielsen8 | anisotropic pretrained NBRDF 缺失 |

GitHub core code最早在 commit `595479d1114b17c436653f3bc58aec45b534ccd1`（2022-08-24）上传；最新 commit只对 `fastmerl.py` 作者注释、README、data 和 PyTorch alternative 有变化，Keras/Mitsuba核心与发布 zip一致。报告锁定 latest `e229dda...`，同时用 release zip hash保留论文发布资产身份。[C git history]

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. **每材质训练**：每个新材质重新训练一个 network；虽然小网络训练只需 10 秒至 3 分钟，但它不是 feed-forward unseen-material compiler。[P Conclusions]
2. **极小网络 crossover**：约 100 scalars 时不如 parametric representation。[P §4.5]
3. **anisotropic 容量更紧**：需要 5× queries，固定 675 scalars 有更明显视觉差异。[P §§3.1,4.2]
4. **AE 不是免费压缩**：decoder约等于 105 NBRDF，且 decoded appearance 会退化。[P §4.3]
5. **importance sampling正式实验限 isotropic BRDF**：abstract明确 reconstruction覆盖 isotropic/anisotropic，而 importance sampling只对 isotropic 映射到两个 analytic models。[P abstract]
6. **proposal是 monochrome approximation**：只预测 sampling-relevant roughness 和 spec/diffuse relative weight，不拟合七个 RGB reflectance参数。[P §3.3]
7. **空间变化尚属 future work**：作者只提出以后扩展 spatially varying materials 和 on-the-fly per-location sampling params；本文没有实现。[P Conclusions]
8. **复杂解析模型不保证更好 proposal**：更精确的 specular fit 可能忽略 sheen 等分量。[P §4.4]

### 12.2 由 method domain 直接推出、但非作者宣称的边界 `[I]`

- evaluator assets 不含 source-native editable parameters，不能保持原生参数编辑；改变材质要重新拟合或在 learned embedding 内插。
- 没有空间坐标与 footprint，因此不能直接承担 texture filtering、mip 或 spatial latent interpolation。
- 网络没有显式 reciprocity/energy constraints；需要独立验证，不能因输入是 half/difference 就视为已满足。
- sampling proxy可静态有界，但两参数 Blinn–Phong proposal 对 layered、multi-peak、transmission 或 retroreflective source 的 variance 可能不足；该预测需要 matched experiment 验证。

### 12.3 未报告/材料不可得

- AE 与 sampling predictor 的 activation、normalization、loss（predictor）、optimizer、learning rate、schedule、batch、epochs、initialization、seed、hardware、training time；
- 80/20 split 与20个 sampling test materials 的完整 identity和 seed；
- evaluator正式实验中每材质是否训练多个 seeds、如何选 checkpoint；
- `NBRDF Uniform Sampling` 的精确 angular distribution/实现；
- image metrics的 resolution、tone mapping、mask、`±` 统计含义；
- Figure 14/16曲线的 tabular data和 confidence intervals；
- Blinn–Phong/GGX inverse-CDF具体公式、两参数约束变换、predictor何时执行/是否持久化其输出；
- Table 3是否包含 embedding/predictor evaluation、sampler setup或只含 steady-state rays；
- anisotropic trainer、pretrained weights与 evaluation configs；
- GLSL demo source、GPU throughput、quantization、SIMD/vectorization；
- autoencoder/predictor official code和 formal configs。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

NBRDF 的表现并不只来自“675 参数 MLP”。容量和先验分布在三处：

1. Rusinkiewicz coordinates 让 specular peak在 query domain 中更容易对齐；
2. uniform-in-angle query recipe把有限 800k samples倾向 peak vicinity；
3. log cosine-weighted objective同时压缩动态范围并按渲染 throughput重加权。

Table 1 的 adaptive/uniform差距说明 query measure足以让同一网络从 `SSIM 0.984` 提升到 `0.995`。因此把该论文复现成“6→21→21→3 MLP + arbitrary hemisphere samples”会丢掉最 load-bearing 的设计。

autoencoder 的主要容量不在 32D code，而在约 71k 参数 shared decoder和 rendering-space loss。论文自己承认其 storage crossover约105 materials；对少量材质，它不是更紧的 evaluator资产。

sampling模块把容量放在 **跨材质统计 prior**：32D embedding与小 predictor用于选择一个廉价解析 proposal，而 proposal runtime本身只剩两参数。这与用大 flow/MLP直接生成方向是不同设计点。

### 13.2 成功所依赖的假设

- source是已测得、homogeneous、dense directional BRDF；query时不需要 source params、空间坐标或可见性。
- 每材质有最多 800k（anisotropic约4M）interpolated supervision，训练时间可被离线承担。
- specular structure在 half/difference coordinates下足够简单，使 21-wide shallow MLP有效。
- quality protocol主要是 image-based local reflectance reconstruction；没有能量、white-furnace、path throughput tail等物理指标。
- importance tests的20个材质来自同一 MERL family，analytic proposal labels也来自对该 family 的 fits；跨 layered/source-family外推未被证明。

### 13.3 可迁移机制与不能迁移的部分

**可直接成为 matched baseline 的机制：**

- fixed Rusinkiewicz `h,d` coordinate baseline；
- `log1p(f·|cosθ_i|)` loss与当前 `log1p(f)`的同预算对照；
- evaluator与 sampler proposal解耦，proposal使用exact analytic sample/pdf；
- 用 frozen material representation监督一个小解析-parameter head，而不是让 sampler gradient改变 evaluator representation；
- per-state 675-scalar NBRDF作为“坐标+loss能否在极小容量拟合”的capacity diagnostic。

**不能直接迁移：**

- 32D AE不接收原生 source parameters，不能充当未见材质 compiler；
- measured-BRDF interpolation不能替代当前 source-family reference；
- 两参数 Blinn–Phong不应被提升为 evaluator target vocabulary；它只是 proposal；
- homogeneous BRDF成功不证明 spatial latent、filtering或 multi-material shared decoder；
- Figure 14/16的CPU/MERL数值不能作为本项目 GPU/LayerStack hard gate。

### 13.4 与本项目 runtime contract 的关系

- **`evaluate`**：NBRDF固定层数、固定weights和有限ops，满足静态有界；需要把旧 Mitsuba `f·cos` convention适配成本项目要求的线性裸 `f`。[N `docs/contracts/scattering_backend.md`]
- **`prepare`**：homogeneous NBRDF没有可摊销 latent fetch或 view encoding，`prepare`可以为空；因此它无法证明本项目 `prepare` 设计的收益。
- **`sample/pdf`**：解析 Blinn–Phong proposal天然有匹配的 sample/pdf且静态有界，适合作为 sampler baseline；公开代码缺失意味着实现前必须回到解析定义，而不是照抄 plugin tabular sampler。
- **部署定位**：原 NBRDF适合 evaluator capacity diagnostic；analytic proxy适合 proposal candidate；AE更适合研究 embedding/alignment，而不是当前产品 runtime主表示。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

当前仓库已经实现 NVIDIA 2024 functional reproduction：online half/difference evaluator route、`log1p` L1、共享 spatial latent、learned frames，以及与 evaluator共享 latent但对 sampler target detach 的 two-lobe learned proposal。[N `archive/.../correspondence.md:L19-L36`; `src/ncls/learning/methods/nvidia.py:L524-L551`; `src/ncls/learning/producer.py:L165-L176`]

| NBRDF 2021机制 | 当前 NVIDIA复现 | 分类 | 影响 |
|---|---|---|---|
| fixed Cartesian `h,d` 输入 | NVIDIA先由 latent生成两个 learned frames，再在每帧中编码方向 | `not-applicable` + 可新增matched baseline | 应比较“固定Rusinkiewicz对齐”与“learned frame对齐”，不能把两者统称half/difference |
| `log1p(f cosθ_i)` L1 | 当前冻结为 `log1p(f)` L1 | `intentional-deviation` | 两者优化不同 measure；需要同source/query/budget对照，不能引用本论文直接证明当前loss |
| per-material 675-scalar evaluator | NVIDIA shared decoder + per-asset spatial latent | `not-applicable` | NBRDF仅作为函数容量/坐标诊断，不是系统替代 |
| 32D embedding→2 param analytic proposal | NVIDIA `11→32→32→32→9` two-lobe learned proposal，forward-KL；latent target detach | `intentional-deviation` | 可增加超廉价、supervised analytic-proxy baseline，测稳定性/variance/runtime |
| evaluator与proposal不同分布但sample/pdf一致 | 当前 scattering ABI要求proposal的 `sample/pdf`匹配，evaluator独立输出线性`f` | `faithful`（共享原则） | 该论文支持“proposal不必成为evaluator vocabulary”的设计边界 |
| code未公开 sampling predictor | 当前NVIDIA sampler有完整Torch/Slang parity | `author-underspecified`（NBRDF） | 不应把2021 proposal写成可直接复现的代码baseline；先显式冻结重建身份 |

最值得保留的差异是 cosine weighting：NBRDF objective拟合 render integrand中的局部 `f cos`，当前 NVIDIA objective拟合 bare `f`。项目 runtime必须输出 bare `f`，但 training loss可以选择另一 measure；这是一项实验轴，不是 ABI 冲突。

本报告没有发现当前 NVIDIA实现中可由该论文直接证明的 `suspected-defect`。它提供的是三项有价值的 matched alternative：fixed coordinate、cosine-weighted log loss、超小解析 sampler head。

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：在当前LayerStack方向查询中，fixed Rusinkiewicz Cartesian `h,d`能以更小网络达到learned-frame evaluator的相近高光质量 | [P Fig.2; §3.1]显示方向参数化决定高光重建 | LayerStack峰也能被同一half/difference canonicalization驻定 | 同一source/query/steps/seed，比较 raw local directions、fixed `h,d`、current learned frames；iso-MAC/iso-parameter | source split、half/difference query、loss、optimizer、budget | directional normalized L1、peak-region error、energy error、GPU query time | fixed `h,d`在iso-cost下未改善peak error，或为匹配质量需要更多MAC |
| H2：`log1p(f·|cosθ_i|)`比当前`log1p(f)`更稳地分配有限容量到实际transport贡献 | [P Eq.(1), Table 1]；但Table1还混入sampling差异 | cosine weighting对当前 layered source的variance/能量也合适 | 仅改loss measure的paired run；network/query完全一致 | target/reference、train/test directions、steps、seed、scale | bare-f error、cosine-weighted error、energy、grazing tail | weighted loss虽改善自身指标，却显著恶化bare-f/energy或未改善matched transport metric |
| H3：从frozen latent预测一个两参数analytic proposal，能以明显更小runtime成本达到当前learned two-lobe sampler的可接受variance | [P Figs.14–16] predicted Phong接近fit并领先uniform | 当前source的主要proposal需求可被单lobe+diffuse mixture覆盖 | frozen evaluator/latent下训练两参数head；对照current 9-param sampler和uniform，均使用exact sample/pdf | evaluator checkpoint、source/query、SPP、MIS、seed、training samples | PDF normalization、variance/RMSE vs spp/time、tail weights、sampler MAC/bytes | 在相同evaluator下variance接近uniform或tail显著差，且成本收益不足 |
| H4：把sampler supervision与evaluator latent优化解耦可降低训练不稳定，而不损害evaluator | [P staged NBRDF→AE→predictor]展示可从稳定embedding学proposal；公开资料未证明joint训练更优 | current sampler无需反向塑造shared latent即可学有效proposal | current joint/detach recipe vs 完全freeze latent后单独训练analytic或9-param head | evaluator checkpoint、sampler samples、optimizer budget、seed set | sampler KL/variance、evaluator drift、失败seed率 | frozen方案没有降低variance/失败率，或proposal质量显著退化 |
| H5：每state 675-scalar NBRDF可作为坐标/目标变换的廉价capacity diagnostic | [P Table1/Fig17]在MERL上极小网络有效 | 当前单个LayerStack state的方向函数复杂度相近 | 对冻结state集合逐state训练NBRDF；不称compiler；对照同字节generic MLP | state/query split、800k上限或匹配query budget、loss、seeds | per-state quality、peak/energy error、train time | 多数state即使充足query也明显欠拟合，说明该容量只适合MERL family |

所有预期数值均为 report-only；这些假设不能把 Table 1、Figure 15 或约 300-scalar crossover变成本项目 hard gate。

## 16. 证据索引

### `P` Main paper

- §1, pp.332–333：问题、三项贡献、每材质 evaluator定位。
- §3.1, p.334, Fig.1, Eq.(1)：`6→21→21→3`、Rusinkiewicz Cartesian input、log cosine-weighted loss、800k samples、5–90 epochs、anisotropic 5× samples。
- §3.2, pp.335–336, Fig.3：`675→675→32→100→675` autoencoder、80/20 split、RGB permutations、64×64 rendering loss、weight-loss负结果。
- §3.3, pp.336–338, Fig.4：32D embedding到两参数解析proposal、unbiasedness边界。
- §3.4, p.338：MERL grid和34 MB。
- §4.1, Figs.6–10, Table 1：reconstruction baselines、aggregate结果、grazing-tail与adaptive sampling分析。
- §4.2, Fig.9：anisotropic结果 `0.981±0.016`。
- §4.3, Figs.11–12, Table 2：AE/PCA、latent interpolation、decoder约105材质开销。
- §4.4, Figs.13–16：importance baselines、20-material Veach/kitchen结果。
- §4.5, Table 3, Fig.17：i9-9900K/Mitsuba throughput、memory、network-size crossover。
- §5：每新材质训练、future spatial extension。

### `S` Supplemental

- `S-MERL`：100材质全量重建、SSIM、polar plots；已完整视觉核对单页长图。
- `S-RGL`：51个RGL isotropic材质GT/NBRDF/SSIM/polar plots；已完整视觉核对单页长图。

### `C` Official code/data

- commit `e229dda3308c78f05e57dbc9455326884f766301`：
  - `binary_to_nbrdf/binary_to_nbrdf.py:L23-L64`：输入字段、loss、MLP、Adam；
  - `binary_to_nbrdf/binary_to_nbrdf.py:L66-L99,L116-L149`：dataset draw、100 epochs、seed wiring；
  - `binary_to_nbrdf/coords.py:L6-L27,L53-L77`：half/difference变换与isotropic固定 `φ_h=0`；
  - `binary_to_nbrdf/fastmerl.py:L116-L162,L183-L249`：MERL interpolation/grid mapping；
  - `mitsuba/bsdfs/nn.h:L38-L89`：fixed network与runtime nonnegative output；
  - `mitsuba/bsdfs/nbrdf_npy.cpp:L20-L105,L144-L147`：Mitsuba evaluator、tabular sampler、GPU未实现；
  - `binary_to_nbrdf/pytorch_code/train_NBRDF_pytorch.py:L14-L57,L124-L204`：2023 alternative实现差异。
- release/data SHA-256见第2节；pretrained counts为MERL100、RGL-isotropic51、Nielsen8。

### `A` Author material

- project page：source inventory、WebGL/GLSL demo声明、预训练数据说明。
- EGSR 2022 slides 6–7：10 s–3 min、2.7 KB；slide 11：anisotropic 5×；slides 13–18：32D AE与Phong/GGX sampling；slide 18明确“Phong performed best”。

### `N` NeuralShading evidence

- `archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md:L19-L36`：当前NVIDIA encoder/latent/frame/evaluator/sampler/training correspondence。
- `src/ncls/learning/methods/nvidia.py:L524-L551`：当前 `log1p` evaluator loss与sampler latent detach。
- `src/ncls/learning/producer.py:L165-L176`：当前 half/difference online evaluator route。
- `docs/contracts/scattering_backend.md`：`prepare/evaluate/sample/pdf`、bare linear `f` 与 matched PDF合同。
- `docs/realtime_material_compilation.md`：目标compiler/runtime边界。

### `I` 本报告推导

- 675 parameters、630 MAC、AE/decoder parameter count均由已标注层维度直接计算；不是论文报告的运行计数。
- 第13–15节的迁移判断与假设只把 `P/S/C/A/N` 当直接证据，不把MERL结果外推成LayerStack结论。

## Caveats / Not Found

- 最大证据缺口是 autoencoder 与 sampling predictor code/config完全未公开；独立 reviewer应重点核对本报告是否在任何地方误把图示维度扩写成未披露的activation或训练配置。
- 官方 Mitsuba plugin的tabular sampler与论文analytic proxy不对应；若后续实现baseline，不能引用该plugin作为proxy oracle。
- RGL anisotropic训练/权重不可得；`S-RGL`只覆盖isotropic集合。
- Table 1/2的 `±`、image metric细节和Figure 14/16原始曲线数据未报告。

## Evidence review

```text
author_worker: nbrdf2021
reviewer: /root/taming2026
reviewed_at: 2026-08-29
sources_rechecked:
  - main paper PDF AB83FF0D27F8A46D34C15971965BC5A62085B28FE462778CBF3C2F9A264BDD64；全文及渲染后的 Figs.1-4、9、13-17 与 Tables 1-3
  - MERL supplemental E32E2BA8F6A51330CFACBC654C87A4D8FF80D1225671CAC22A5EE8BCD2E16A7D；完整长页渲染
  - RGL-isotropic supplemental 948B884B03D5AAD9F5F3A3C6D676F259E5F07704BE3001A558BA924E8AAB8560；完整长页渲染
  - 作者项目页与 EGSR 2022 slides；视觉复核 slides 17-19
  - 官方 asztr/Neural-BRDF commit e229dda3308c78f05e57dbc9455326884f766301；trainer、loss、C++ evaluator、Mitsuba tabular sampler 与预训练资产
  - 第 14 节引用的 NeuralShading correspondence、evaluator/sampler 训练路径与 half/difference online query route
findings_closed:
  - 将无依据的 sampling predictor 拓扑 32→32→8→2 修正为 Figure 4 所示的 32→8→2
  - 移除 RGL anisotropic 与 MERL isotropic aggregate SSIM 的非受控直接排序
  - 将 Table 1 的统计显著性措辞降为原表数值观察，并保留 ± 统计含义未报告的边界
  - 将 Phong/GGX 选择由 author-negative 修正为 ablation-inferior，避免把缺少 matched 数值表的备选写成作者失败尝试
  - 对照论文与代码边界确认 isotropic/anisotropic 坐标、675-scalar evaluator、cosine-weighted log-L1 query target 与 anisotropic 五倍 query 数
  - 确认论文的两参数 analytic Blinn-Phong proxy 未出现在官方仓库；Mitsuba CPU 路径改为对由 NBRDF 构造的 djb::merl 辅助对象建 tabular sampler，GPU shader 路径未实现
  - 确认 Tables 1-3 与 Figs.9、13、15 的正式数值，未发现其它数值错误
  - 对照当前项目 locator 复核第 14 节 N/I；没有以 N 回填论文事实，假设保留了显式证伪条件
remaining_evidence_gaps:
  - autoencoder implementation/config 不可得
  - importance-predictor implementation/config 不可得
  - anisotropic training code/assets 不可得
  - metric ± 统计含义与曲线数据未报告
  - GLSL demo 源码与精确 WebGL runtime 配置不可得
review_status: evidence-reviewed
```

### 完成检查

- [x] main paper 已完整阅读，关键公式/图/表/图注/脚注已视觉核对；
- [x] supplemental/appendix/勘误的可用性已检查；
- [x] official code/config/data 的可用性与 commit 已检查；
- [x] architecture、training、runtime 和主要结果均有 locator；
- [x] 失败尝试与较差消融正确分类；
- [x] paper/code gap 和“未报告”保留；
- [x] `I` 分析晚于事实层，没有改写作者结论；
- [x] NVIDIA 影响引用真实 `N` 证据；
- [x] 假设包含 matched control、部署类别和证伪条件；
- [x] 独立 evidence review 已完成。
