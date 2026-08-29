---
paper_id: "fan-2022-neural-layered-brdfs"
title: "Neural Layered BRDFs"
authors: "Jiahui Fan, Beibei Wang, Miloš Hašan, Jian Yang, Ling-Qi Yan"
year: "2022"
venue: "ACM SIGGRAPH 2022 Conference Proceedings"
doi: "10.1145/3528233.3530732"
report_status: "evidence-reviewed"
main_source: "https://wangningbei.github.io/2022/NLBRDF_files/paper_NLBRDF.pdf"
supplemental_status: "available"
official_code_status: "audited"
official_code_commit: "904538a1197bcf4583caa748a5162119ae82a944"
author_worker: "/root/taming2026"
reviewer: "/root/rta2024"
last_verified: "2026-08-29"
---

# Neural Layered BRDFs

## 1. 研究对象与报告边界

本文研究一种面向局部表面反射的“neural BRDF algebra”：先用共享 evaluation network 把单个 BRDF 表示成 latent，再用第二个网络把 top BRDF latent、bottom BRDF latent 与中间介质参数编译成 layered BRDF latent。它属于 `local-material`，不是 scene-level light transport 方法；场景 path tracing 只用于验证已编译 BRDF 的渲染效果。[P §3]

本报告覆盖 DOI 正式版正文、5 页 supplemental、作者项目页、官方 timing correction，以及作者公开的单 commit PyTorch release。时间结论一律以 correction 为准：正文中的 5 ms、supplemental Figure 9 的旧 GPU 柱形，以及“shading 比 Belcour [2018] 快”的结论均不能继续作为有效事实。[A-corr §§1–2, Tables 1–2, Fig.1]

报告把四件事分开：

1. shared decoder 与 per-BRDF latent 的函数表示；
2. 把已存在 BRDF latent 组合为新 latent 的 layering compiler；
3. supplemental 中的 latent interpolation/mipmap；
4. supplemental 中预计算两参数解析 proposal 的 importance sampler。

该方法没有证明任意材质图、任意 source family 或显式 BTDF 的通用编译；也没有公开正式训练、CUDA renderer 或 sampling network 实现。[P §5.3; C README]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | [official PDF](https://wangningbei.github.io/2022/NLBRDF_files/paper_NLBRDF.pdf)，DOI `10.1145/3528233.3530732` | 2026-08-29 | SHA-256 `FA3D8E4696647D459EFD370E1B32C742D6DF187C7A4D71B38EE846B6A2E7C3D6` | 正式方法、数据、训练和主实验；8 页 |
| Supplemental `S` | [supplementary PDF](https://wangningbei.github.io/2022/NLBRDF_files/supplementary_final.pdf) | 2026-08-29 | SHA-256 `DB5919DEE4167DBAFABFABB71C5AE132E2643024F5CFA6410E1550D78F78C42B` | interpolation、mipmap、GNDF sampler、额外 representation 对照和 4/8/16-block 消融；5 页 |
| Official correction `A-corr` | [Correction to the Timings](https://wangningbei.github.io/2022/NLBRDF_files/correction_NLBRDF.pdf) | 2026-08-29 | SHA-256 `C9AA8892728CAB1AF92EF3E4C75995795032D0DC47B324EA306F601624B979FF` | 撤回 5 ms 和对 Belcour 的速度结论，给出正确 batch/scene GPU time；1 页 |
| Author project page `A` | [project page](https://wangningbei.github.io/2022/NLBRDF.html) | 2026-08-29 | 固定 URL | 论文身份、摘要、下载入口；网页摘要不能覆盖 correction |
| Official code `C` | [sssssy/pytorch-mitsuba-NLB_Release](https://github.com/sssssy/pytorch-mitsuba-NLB_Release/tree/904538a1197bcf4583caa748a5162119ae82a944) | 2026-08-29 | commit `904538a1197bcf4583caa748a5162119ae82a944` | 单 commit experimental PyTorch release；审计 network、latent-only compression 和 example layering；无 LICENSE 文件 |
| Official checkpoints/data `C` | 同一 commit 的 `saved_model/`、`data/` | 2026-08-29 | `RepreNetwork.pth` SHA-256 `FD2B9D741FCFE86E29BBA6B5904C6F849B70BA602E8026D07CA0DC8D9B16B7EB`；`LayeringNetwork.pth` SHA-256 `A2DC0E351A637EDE87222BA09857F23B827A9D69338FFE835C50BD06A6133DD1` | 两个 serialized bound-`state_dict` method checkpoint 与两个 `.npy` example；不是训练集或 formal config；在锁定 audit 环境中调用后与当前 evaluator/layerer source 全部 keys matched |
| NeuralShading evidence `N` | [NVIDIA correspondence](../../../archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md)、[runtime contract](../../../../../docs/contracts/scattering_backend.md)、[model candidates](../../../../../docs/research/model_candidates.md) | 2026-08-29 | repo-local | 只用于 §§13–15；不得回填成 2022 论文事实 |

作者提到 accompanying video 会给出复杂场景的详细材质参数，但该视频没有锁入本 worker 的证据包，本报告不从视频补写配置。release 只有 README 所称的“suboptimal and experimental”脚本，没有 formal training script、Mitsuba integration、Cutlass/CUDA kernel、sampler network、mipmap builder 或 benchmark harness。[C `README.md:L1-L36`]

## 3. 原论文的问题、假设与贡献边界

### 3.1 问题

精确 layered material 要追踪界面与介质中的多次反射、透射和散射。作者把 Guo et al. [2018] 的 position-free Monte Carlo random walk 当作 reference：它不引入模型近似，但计算贵且有 Monte Carlo noise；Belcour [2018] 等统计传播方法更快，但用统计摘要近似完整函数。[P §§1–2]

### 3.2 方法假设

作者提出两个连续假设：

- 大量单层/层状/测量 BRDF 可以共享一个 decoder，而每个 BRDF 只需一个短 latent；
- 在这个 latent 空间中，物理 layering 可以近似为一个确定性神经算子：

\[
f(\omega_i,\omega_o) \xrightarrow{N^{rep}} V_f,
\qquad
\{V_{top},V_{bottom},A,\sigma_T\}
\xrightarrow{N^{layering}} V_{layered}.
\]

[P Eqs.1–2, §3.1]

这里的 `N^rep` 不是 encoder，而是“固定 decoder 后优化 latent”的投影过程。训练 shared decoder 时 network weights 与训练 BRDF latents 联合反向传播；压缩新 BRDF 时冻结 decoder，只优化新 latent。[P §3.2, Fig.4]

### 3.3 贡献边界

正式 layering operand 是：top rough dielectric BSDF、bottom 任意 BRDF、两者之间 unit-thickness homogeneous medium 的 single-scattering albedo `A` 与 extinction `σT`。top 的 transmission 被假定为其反射能量的补集；界面不补偿 microfacet energy loss，介质采用 isotropic scattering。多层通过从 bottom 到 top 递归调用二元 layerer 获得。[P §§3.1,3.3,4.1, Figs.5,8]

论文只建模 reflection hemisphere。top interface 的 BTDF 从训练关系中隐式推断但从不显式构造；这不等价于一个可独立查询、采样或组合的 neural BSDF。[P §§3.1,5.3]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| BRDF source | analytic rough conductor/dielectric、Guo layered BRDF，投影测试另含 MERL/BTF | isotropic reflection BRDF | [P §§3.2,4.1,5.1; S §2.3] |
| Evaluator material input | 每个 RGB 通道独立的 `V_f` | `32` floats/channel；RGB 共 `96` | [P Fig.2, §3.2, Table 1] |
| Evaluator direction query | incoming/outgoing Cartesian direction | `ω_i,ω_o∈H²`，图示合计 6 scalars | [P §3.1, Fig.2] |
| Release direction layout | 文件存 `(view_x,view_y,light_x,light_y)`，代码从单位圆重建两个正 `z` | 4 stored → 6 decoder scalars；顺序为 4 个 xy 后跟两个 z | [C `README.md:L83-L97`; `utils.py:L76-L85`] |
| Evaluator output | 对单通道输出 BRDF 值，RGB 逐通道求值 | scalar/channel；正文训练 target 不含 cosine | [P §§3.2,4.1–4.2] |
| Layerer input/output | `{V_top,V_bottom,A,σT}→V_layered` | 每通道 `32+32+1+1→32` | [P Eq.2, Fig.3] |
| Sampling input/output | BRDF latent + incoming direction → `σ,w`；最终对 40×40 directions 的预测取平均 | 两参数 Gaussian+Lambertian mixture；最终参数与 incoming direction 无关 | [S §1.2, Eqs.1–2] |
| Validity/domain | upper reflection hemisphere；isotropic；individual layers 无 normal map | 不支持 two-sided/full-sphere BSDF、anisotropic layer/medium | [P §§3.1–3.3,5.3] |

正文没有使用 Rusinkiewicz half/difference 坐标作为 evaluator 输入；只有 sampler 的 analytic proxy 在 projected half-vector `(h_x,h_y)` 上定义 Gaussian。training directions 由每个方向的 elevation `θ` 与 azimuth `φ` 分层采样，再转为 direction query。[P §4.1; S Eq.1]

正文明确说训练/存储的是不含 cosine 的 `f`。release 的 `compress.py` 却在 loss 前把 decoder scalar 乘以最后一个重建 `z`；按 README 的 `(view,light)` 顺序，它是 `light_z`。因此公开 latent-projection example 不能无条件称为正文 raw-BRDF L1 的对应实现。[P §4.1; C `compress.py:L81-L95`; `utils.py:L82-L85`]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

训练与使用分成三个阶段：

```text
BRDF dataset + dense direction queries
  → jointly optimize shared evaluator D and per-BRDF latents V_f

known/new BRDF
  → freeze D; optimize one 32D latent per RGB channel

(V_top, V_bottom, A, σT)
  → shared layerer L, independently per RGB channel
  → V_layered
  → D(V_layered[channel], ω_i, ω_o)
  → RGB BRDF
```

layerer 不直接读取原生 six-parameter layer stack；它读取两个已经落在 decoder latent 空间中的 operand，再加当前 binary operation 的 `A,σT`。two-layer target latent 是先用 Guo reference 的完整 BRDF 投影出来的 optimized code，layerer 以该 code 为 supervision。[P §§3.2–3.3,4.1–4.2]

对于 SVBRDF，每个 texel 存 3×32 latent。supplemental 先对 latent texture 建 mipmap，再按 shading footprint 选相邻 levels 并做标准 trilinear interpolation。sampling network 仅在预处理阶段把 latent 压成 `σ,w`；渲染时 sample/pdf 使用解析 mixture，不运行 sampling MLP。[P §4.3; S §§1.1–1.2]

### 5.2 持久化表示

| 资产 | 持久化内容 | 正式大小/边界 | locator |
|---|---|---|---|
| 单个 RGB BRDF | 3 个独立 32D latent | `96 floats`；论文未给量化 | [P Fig.2, Table 1] |
| Shared evaluator | 所有材质共用 decoder | 论文称约 `1M floats` | [P §3.2, Table 1] |
| SVBRDF | 每 texel 96 floats | 4K base level 为 `4096²×96` floats，即 `[I]` 6 GiB FP32；mip overhead 未报告 | [P §3.2, Table 1; I arithmetic] |
| Layerer | shared network；产生新 latent | 不需要按 texel持久化 layerer weights 的副本 | [P §3.3, Fig.3] |
| Sampling proxy | per BRDF `σ,w` | 精度、存储格式和 per-texel/SVBRDF 构建方式未报告 | [S §1.2] |
| Latent mipmap | multi-channel latent mip chain | level construction 除“建立 mipmap”外未报告；无 bytes/format | [S §§1.1,2.1] |

没有 codebook、quantization、sparse grid、learned frame 或 per-query material network generation。对比 Sztrajman et al. [2021] 时，作者强调自己的 decoder shared，但这不消除每 texel 96-float base latent 的实际成本。[P Table 1]

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Evaluation network（单通道） | 32 latent + 6 direction | `38→256`，16 个 width-256 transform stages，嵌套 additive skips；`256→1` | hidden `FC+LayerNorm+ReLU`；residual/skip 在 LN 与 ReLU 前相加；final linear | scalar BRDF | shared | [P Fig.2, PDF p.3] |
| Release `model.py` evaluator exact topology | 同上 | initial FC；8 个 `256→128→256` residual blocks + 8 个 `256→256` FC blocks，带 nested outer skips；final `256→1` | LayerNorm+ReLU；无 final clamp/transform | scalar | shared | [C `model.py:L18-L111`] |
| Layering network（单通道） | top32 + bottom32 + `A` + `σT` | `66→512`；width-512 residual/FC tower；`512→128→32` | hidden LayerNorm+ReLU；final linear | layered latent32 | shared | [P Fig.3, PDF p.3] |
| Release layerer exact topology | 66 | initial `66→512`；4 个 `512→256→512` residual blocks +3 个 `512→512` FC blocks和 nested skips；`512→128→32` | LayerNorm+ReLU；final linear | latent32 | shared | [C `model.py:L117-L162`] |
| Importance network | BRDF latent + incoming direction | four hidden layers `128→512→128→32`，再输出 2 参数；输入 direction encoding 与 output constraints 未报告 | hidden ReLU；normalization 未报告 | `σ,w` | shared，offline | [S §1.2, PDF p.1] |

按 release 构造静态求和，evaluator 有 `1,074,689` trainable scalars，layerer 有 `1,954,208`；这与论文“约 1M floats”的 evaluator 量级一致。只读反序列化审计确认 bundled evaluator checkpoint 的 bound owner也是 `1,074,689` parameters，调用其 `state_dict` 后可严格载入当前 16-block source；layerer同样全部 keys matched。[C `model.py`, `saved_model/*.pth`; I parameter arithmetic and checkpoint audit]

对应 dense linear MAC（不计 bias、LayerNorm、ReLU）约为 evaluator `1,058,560/channel-query`、layerer `1,938,432/channel-operation`；RGB 分别要做三次，共约 `3.18M` 与 `5.82M` MAC。这是 release topology 的派生数，不是论文报告的硬件 FLOP。[C `model.py`; I arithmetic]

### 5.4 条件化、坐标变换与物理先验

- evaluator 只通过 concatenation 接收 latent 与 raw Cartesian directions，没有 half/difference warp、Fourier feature 或 analytic BRDF core。[P Fig.2]
- RGB 共享同一 scalar decoder，但每通道 latent 独立；这使 color variation 进入 latent，不由三通道 joint head 建模。[P §3.2]
- layerer 把 top rough dielectric 的透射、介质传输与多次散射全部隐式压进 latent mapping；没有显式能量、reciprocity 或 BTDF head。[P §§3.3,5.3]
- release `layering.py` 在调用 layerer 前后对 latent 做以 1 为中心的平移，并让 `A,σT` 经 `+1` 后再整体 `-1`，因而参数本身仍以原值进入。正文只披露 latent 初始化为 1，没有把该 normalization 声明为 formal training contract。[C `layering.py:L73-L87`; P §4.2]
- supplemental 的 sampler 是显式 analytic prior：projected-half-vector Gaussian lobe 与 cosine Lambertian lobe的 mixture；它近似 GNDF，不替代 evaluator。[S §1.2]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Base analytic BRDFs | 300 rough conductors + 300 rough dielectrics；GGX NDF | [P §4.1] |
| Two-layer corpus | 12,720 BRDFs：top rough dielectric、bottom rough conductor、unit-thickness isotropic medium | [P §§3.3,4.1] |
| Three-layer corpus | 1,800；bottom operand 本身已经 layered，用于后续 finetune | [P §4.1] |
| Parameter distribution | `α=[U(0.216,1)]³`；`η~U(1.05,2)`；`R0~U(0,1)`；`A=1-U(0,1)²`；`σT~V({0,1,2,5})` | [P Table 2, visually checked] |
| Medium/interface assumptions | HG phase `g=0`；unit thickness；no anisotropic scattering；不补偿 microfacet energy loss | [P §§3.3,4.1] |
| GT/reference | conductor/dielectric 用 microfacet model；layered BRDF 用 Mitsuba 中 Guo et al. [2018] Monte Carlo | [P §§2,4.1,5] |
| Direction sampling | 每个 `ω_i`/`ω_o` 的 `(θ,φ)` 各取 25 个 stratified samples；总计 `25⁴=390,625` direction pairs/BRDF | [P §4.1, PDF p.5; C README example shape] |
| Stored target | incoming direction、outgoing direction、BRDF，不含 cosine | [P §4.1] |
| Representation split | two-layer 12,000 train / 720 validation；base 600 不进入 evaluator training | [P §4.1] |
| Layerer split | 投影全部 12,720 layered BRDF 与其 600 base components；train/validation 比例沿用 representation；再以 1,800 three-layer finetune | [P §4.1] |
| Sampling-network split | 从 layered corpus 随机取 3,000 train / 300 validation；先投影 latent，再构造 GNDF | [S §1.2] |
| GNDF recipe | 对每 BRDF uniform 40×40 incoming directions，在 half-angle space 平均 2D BRDF lobes并归一化；loss 又在 GT/proxy 上各取 40×40 grid points | [S §1.2] |
| Extra test assets | unseen MERL measured BRDF；UBO2014 BTF；具体材质 split、数量和 projection budget 未报告 | [P Fig.9; S Figs.6–8,10] |
| Filtering/LOD | latent texture 预建 mip，footprint 选 level并 trilinear query；无 filtered radiance GT 或 scale loss | [S §§1.1,2.1] |
| Online/offline | 论文先生成并存储 dense BRDF queries；不是 optimizer-step 内 online reference | [P §4.1] |

`25⁴` 在 PDF 排版中容易被文本抽取成 `254`，但视觉核对正文 PDF p.5 与 release 的 `(1,390625,7)` example shape 后可确定是 `390,625`，不是 254。[P §4.1; C `README.md:L45-L51`]

两个 bundled `.npy` 的磁盘 shape 均为 `(1,2,734,375)`、dtype `float32`；`compress.py` 以 `reshape(1,-1,7)` 还原为 `(1,390,625,7)`。这进一步锁定了“每个 BRDF 有 `25⁴` 个七元 query record”的 release 数据口径，但两个 example仍不等于论文训练 corpus。[C `data/0-0.npy`, `data/1-0.npy`; `compress.py:L49-L50`; C array-header audit]

## 7. Loss、optimizer 与训练 lifecycle

### 7.1 论文与 supplemental 正式披露

| 项 | Representation network | Layering network | Sampling network | locator |
|---|---|---|---|---|
| Target | raw scalar BRDF per channel | target optimized latent | normalized GNDF grid | [P §4.2; S §1.2] |
| Loss | `L1 = (1/N)Σ_N |f_pred-f_gt|` | latent `L1` | `KL(S(f_gt)||S(f_pred))`，`S` 为 softmax | [P §4.2; S Eq.2] |
| 作者选择依据 | `L1` 比 `L2` 更保色、少 artifact | 未给 matched loss ablation | proxy 匹配 GNDF | [P §4.2; S §1.2] |
| Optimizer | **未报告** | **未报告** | **未报告** | source gap |
| Learning rate | network `3e-4`；mutable latent `1e-4` | initial `3e-3` | initial `3e-5` | [P §4.2; S §1.2] |
| Schedule | 两者每 epoch `×0.9` | 每 50 epochs `×0.7` | 每 3 epochs `×0.7` | 同上 |
| Batch | BRDF batch 的 `N` 未给具体值；每 BRDF方向子批规则未报告 | 未报告 | 未报告 | source gap |
| Epoch/stage | 50 epochs；network/latents joint | 1,000 epochs two-layer；随后 three-layer finetune，finetune epochs 未报告 | 10 epochs | [P §4.2; S §1.2] |
| Initialization | 所有 train latent 初始化为 1；network init/seed 未报告 | init/seed 未报告 | init/seed 未报告 | [P §4.2] |
| Hardware/time | RTX 2080Ti，约 40 h；新 BRDF latent projection约 10–45 s | RTX 2080Ti，约 10 h；finetune time 未报告 | RTX 2080Ti，<1 h | [P §§3.2,4.2; S §1.2] |
| Model selection | validation 存在；checkpoint criterion 未报告 | validation比例存在；criterion 未报告 | 300 validation；criterion 未报告 | source gap |

representation 是 autodecoder-like joint fitting，不是 encoder inference；新资产仍需 10–45 秒 gradient projection。layerer 随后只学 optimized latent 之间的映射，其训练 loss 没有通过 frozen evaluator回到 BRDF function space。[P §§3.2,4.2]

### 7.2 Release example，不是 formal training config

公开 `compress.py` 的**意图**是冻结 pretrained decoder并优化一个 RGB latent：Adam、batch `4096`、lr `1e-3`、`max_steps=max(floor(Q/4096)×10,1000)`、latent init 1、time-based microsecond seed。以 bundled `Q=390,625` 为例，`max_steps=1000`、decay interval 为 200；LR 在 step 200/400/600 分别乘约 `0.248`，从 `1e-3` 到 README 所示约 `1.53e-5`。它用 `L1(y/(1+y),t/(1+t))`，且先把 decoder 输出乘 `light_z`；这些都不同于正文披露的 raw-BRDF L1 projection，因此只能标 `C` example config。[C `compress.py:L27-L34,L49-L73,L86-L113`; I schedule arithmetic]

三个入口的 `load_state_dict(torch.load(path)())` 写法虽然反常，但不是零参数 `forward` defect：bundled `.pth` 序列化的是绑定的 `state_dict` method，第一次括号加载 method，第二次括号返回 state dictionary。锁定 audit 环境以 modern PyTorch 显式 `weights_only=False` 只读加载后，evaluator/layerer均 `All keys matched successfully`；evaluator owner有 `1,074,689` parameters。加载会因 archive 内嵌旧类源码与 release source不同而发出 `SourceChangeWarning`，而 README正式声明的旧 PyTorch 1.2环境没有 modern `weights_only`默认值问题；因此这里记录 compatibility/serialization caveat，不误标为 release defect，也不把本次只读 load称为完整 quick-start复现。[C `compress.py:L52-L55`; `layering.py:L40-L44`; `visualize.py:L87-L90`; `saved_model/*.pth`; C checkpoint audit]

release 不含 representation joint training、layerer training、three-layer finetune 或 sampling-network training。README 明确把这些文件称为 experimental/suboptimal scripts。[C `README.md:L1-L15,L17-L36`]

## 8. Inference、部署与成本

### 8.1 正文 runtime path

Mitsuba path tracer 先把 neural-hit 的 directions 与 lighting values写入 buffer；GPU 批量运行 evaluator，再由 CPU 组合 radiance。light sampling、BRDF sampling 和 MIS 路径还缓存相应 PDF。作者用 Cutlass 编写 CUDA inference、预编译多个固定 batch-size kernel，并按当前 neural pixels 选择/组合 kernel。[P §4.3; A-corr §§1–2]

layerer 在材质编辑后重新生成 layered latent；正文只说其代价相对 rendering 很小，没有单次 layerer latency。sampling network 的 `σ,w` 在 rendering 前预计算，因此 hot path 只有 analytic Gaussian/Lambertian sample/pdf。[P §5.2; S §1.2]

### 8.2 模型、资产与精度

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime evaluator | shared scalar decoder，RGB 三次调用 | [P Fig.2; C `visualize.py`] |
| Parameter count | paper约 1M floats；release-derived `1,074,689` | [P §3.2; C `model.py`; I arithmetic] |
| Layerer count | release-derived `1,954,208`；不是 per-shading-query evaluator | [C `model.py`; I arithmetic] |
| Per-BRDF/texel state | 96 floats = 384 B if FP32 | [P Table 1; I arithmetic] |
| Shared evaluator bytes | release params若 FP32约 4.10 MiB；formal CUDA precision未报告 | [C `model.py`; I arithmetic] |
| Texture fetch/packing | 96-channel texture的GPU packing、fetch count与alignment未报告 | source gap |
| Quantization | 未报告 | source gap |
| Renderer hardware | Ubuntu；Xeon E5-2650 v4 2.20 GHz 8 cores；64 GB RAM；RTX 2080Ti 11 GB | [P §5] |

### 8.3 Correction 后的唯一有效 CUDA timing

| CUDA evaluator batch | Corrected inference time |
|---:|---:|
| 1,024 | 1 ms |
| 4,096 | 2 ms |
| 8,192 | 4 ms |
| 65,536 | 23 ms |
| 131,072 | 45 ms |
| 262,144 | 88 ms |

[A-corr Table 1]

| Scene | Resolution | Neural pixels/spp | Corrected GPU time/spp |
|---|---:|---:|---:|
| Still Life | 1024×1024 | 336,519 | 113 ms |
| Ball | 512×512 | 187,928 | 68 ms |
| Shoe | 888×500 | 68,591 | 25 ms |
| Globe | 1024×1024 | 262,842 | 89 ms |
| Teapot | 720×480 | 69,680 | 27 ms |

[A-corr Table 2]

Still Life 的 336,519 neural pixels 由作者分到 262,144 与 131,072 两个预编译 batch，最终 GPU time 为两者之和。正文“1920×1080 每 pixel一次 BRDF evaluation只需 5 ms”已被作者明确判错；不能把 5 ms或 supplemental Figure 9 的旧 4/8/16-block GPU bars继续用于速度比较。[P §4.3, superseded; S Fig.9, time panel superseded; A-corr §§1–2]

## 9. 实验 protocol、baseline、指标与完整结果

### 9.1 Representation accuracy

正文 Table 1 是 capability table，不是 matched 数值 benchmark。作者将自己的方法归为 shared decoder + per-BRDF latent；Rainer et al. 2019/2020 在 sharp specular preservation 上被标为不足，Sztrajman et al. 2021 与本文均能保峰，但前者每 BRDF保留独立 decoder。[P Table 1]

supplemental Figure 10 在 6 个 analytic/layered examples 上比较 outgoing-radiance MSE；layered examples 的输入是 **GT BRDF projection**，不是 layerer prediction，因此只测 representation：[S Fig.10]

| Example row | Rainer et al. 2020 | Sztrajman et al. 2021 | Ours |
|---:|---:|---:|---:|
| 1 | `3.7e-4` | `4.8e-5` | **`2.5e-5`** |
| 2 | **`1.2e-4`** | `5.9e-4` | `7.5e-4` |
| 3 | `1.4e-3` | **`1.6e-4`** | `4.9e-4` |
| 4 | `1.4e-3` | **`4.8e-4`** | `9.7e-4` |
| 5 | `2.3e-3` | `1.5e-3` | **`7.4e-4`** |
| 6 | `1.1e-2` | `1.4e-2` | **`4.3e-3`** |

这个表是 mixed result：本文并非每行最低 MSE；作者还指出 Sztrajman 某些低数值结果有明显 color difference，但未给颜色误差指标。[S §2.3, Fig.10]

对 unseen MERL，supplemental Figure 6 的三个案例 MSE 为 `4.2e-3 / 3.9e-3 / 3.7e-3`。UBO2014 BTF 只给 qualitative grazing-angle比较，并统一对 inset 加 `+2.8` exposure；测试材质数量、projection samples 和统计聚合未报告。[S Figs.6–7]

reciprocity 没有显式 loss。作者把近似 reciprocal 行为归因于 `ω_i/ω_o` 对称采样，Figure 8 展示 swapped-direction render 与相对 MSE histogram，但未给 aggregate number或 worst case。[S Fig.8]

### 9.2 Layerer function quality

Figure 6 在 varying top/bottom roughness 与 medium albedo 网格中报告以下 MSE；每格对比本文 layerer 与高-spp Guo reference：[P Fig.6]

| top row → bottom row | Roughness 1 | Roughness 2 | Roughness 3 | Roughness 4 | Varying medium |
|---|---:|---:|---:|---:|---:|
| 1 | `4.8e-2` | `6.0e-3` | `2.2e-3` | `7.2e-3` | `7.9e-2` |
| 2 | `1.3e-4` | `3.9e-5` | `2.4e-5` | `1.7e-5` | `5.5e-2` |
| 3 | `1.0e-4` | `4.0e-5` | `1.8e-5` | `1.1e-5` | `5.5e-2` |
| 4 | `4.6e-5` | `2.5e-5` | `1.2e-5` | `9.2e-6` | `2.7e-2` |

结果范围跨 `9.2e-6` 到 `7.9e-2`，不能只用“close”掩盖最难格。Figure 8 将 binary layerer递归用于 3–6 layers，作者报告视觉仍接近 reference，同时明确 error 会累积；没有层数—误差曲线。[P Figs.6,8, §5.1]

Figure 9 把完全未见的 MERL measured BRDF作为 operand，不做 layerer finetune，给出 4 个 qualitative scene对照；没有数值 aggregate。[P Fig.9]

### 9.3 Complex scenes 与 timing correction

正文 Table 3 的原始 equal-time quality table为：[P Table 3]

| Scene | Resolution | Ours spp/MSE | Guo spp/MSE | Published time |
|---|---:|---:|---:|---:|
| Still Life | 1024×1024 | `512 / 1.0e-3` | `64 / 2.0e-3` | 4.82 min |
| Globe | 1024×1024 | `256 / 9.3e-4` | `24 / 2.3e-3` | 2.42 min |
| Teapot | 720×480 | `256 / 6.0e-4` | `12 / 5.4e-3` | 0.97 min |

correction 说明 CPU timings 正确，并保持作者对 Guo 的总体结论，但原 GPU overhead 被低估。它没有重发 Table 3 的 sample counts/MSE；因此这些质量值可作为正文报告值保留，却不能再不加说明地当作严格 matched total-wall-clock benchmark。[A-corr Table 2; I correspondence boundary]

Shoe 的正文图注中 `GPU 0.01 s` 已失效；correction 给出的有效值是 `25 ms/spp`。图中 CPU 时间仍为 ours `0.18 s`、Guo `9.77 s`、2048-spp reference `14.5 min`，作者说明 neural result 的剩余 noise来自 environment/indirect lighting。[P Fig.10; A-corr Table 2]

对 Belcour 的正确 equal-time结果是：ours `12 s / 170 spp / MSE 2.4e-3`，Belcour `12.2 s / 256 spp / MSE 7.7e-3`；Guo 2048-spp reference耗时 45 min。作者同时明确本文 neural shading **更慢** 于 Belcour，原正文“ours faster”被撤回。[A-corr §2, Fig.1]

### 9.4 Sampling、interpolation 与 mipmap

- GNDF Figure 3：Gaussian+Lambertian proxy 在一个 diffuse 与一个 specular example 上比单独 Lambertian/GGX更接近 reference；只有图像与中心横截线，无数值 divergence。[S Fig.3]
- BRDF-only Figure 4：三种 proposal 均 256 spp，作者称 learned proxy variance最低；没有 variance estimator或数字。[S Fig.4]
- Figure 5：大 area light + environment 与 point light 两种配置中，MIS视觉优于 light-only 与 BRDF-only；SPP与误差未报告。[S Fig.5]
- latent interpolation Figure 1：在 roughness跨度下产生一个中间宽度 lobe，而 BRDF-value mixture保留两个 lobes；这是 appearance-editing偏好，不是等价于物理 mixture 的证明。[S Fig.1]
- latent mip Figure 2：1 spp 下预建 latent mip + footprint trilinear query减少 checkerboard aliasing；没有 filtered-reference、能量或 temporal metric。[S Fig.2]

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `author-negative` | 正文 5 ms CUDA inference | 作者后续复核认定数字错误 | 固定 batch kernel 的真实 cost显著更高 | 原值必须撤回，不能“误差条修补” | [A-corr §1, Table 1] |
| `author-negative` | “ours shading faster than Belcour” | correction 明确改为 ours shading time更长 | 原 GPU cost低估 | quality 的 equal-time优势仍是单一场景结果，不恢复速度 claim | [P Fig.7 superseded; A-corr §2, Fig.1] |
| `ablation-inferior` | evaluator 4/8 blocks vs 16 blocks | 小网络更快但 highlights模糊、可能错色；16-block validation/test error最低 | 容量不足以表示 difficult BRDF | GPU time bars被 correction supersede，只保留质量趋势 | [S §2.4, Fig.9] |
| `author-negative` | representation `L2` loss | 相比 `L1`，颜色保持较差并有 artifacts | 作者据实验选择 `L1` | 无正式数值表，不能量化效应 | [P §4.2] |
| `known-limitation` | multiple separated lobes / highly specular BRDF | 更难预测 | 高频与多峰提高输入函数复杂度 | 与固定 16-block容量相关，但论文没有归因消融 | [P §5.3] |
| `known-limitation` | recursive binary layering | 层数增加时 error accumulation | 每次 approximate composition 累积误差 | 应与 direct whole-stack compiler matched比较 | [P §5.3, Fig.8] |
| `known-limitation` | BTDF、anisotropy、normal-mapped individual layers | 未显式支持/未训练 | 维度和训练数据进一步增加；normal mapping更易在 shading frame做 | “可能扩展”不是已证明能力 | [P §§3.2,5.3] |
| `known-limitation` | bias 与 energy conservation | neural error可能系统性变亮/变暗并破坏能量；作者称未观察到 artifact | 网络近似不是 MC-unbiased | 没有 energy test，不能把“未观察到”写成守恒 | [P §5.3] |
| `paper-code-gap` / release-code defect | `compress.py` batch indexing | slice start 是 `batch_num % reset_perm`，没有乘 `batch_size`；对 390,625 queries、batch4096，`reset_perm=95`，同一 permutation 周期的 95 个 batch只覆盖位置 `[0,4189]`，相邻 batch重叠 4,095/4,096 | 作者未讨论；README先声明脚本 experimental/suboptimal | `1000` optimization steps跨 10 个完整 permutation周期和第 11 个周期的前 50 步；周期间会重排，故全程 distinct query 数取决于 seed，但它不再具有“约10次完整数据 epoch”的通常语义。只影响 release latent-projection example；不得反推 formal training有同一 bug | [C `compress.py:L57-L62,L75-L83`; `README.md:L8-L15`; I index arithmetic] |

作者在 limitation 中列出“尚未尝试但可能有用”的方案时，本报告不把它们虚构为失败实验。sampling 的 Lambertian/GGX 图是正式较差 baseline，但没有数字；只能称 qualitative `ablation-inferior`，不能宣称统计显著失败。[S Figs.3–4]

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental/correction | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Evaluator architecture | 32+6→16-block width256→1，LN/ReLU/residual | 4/8/16 block quality comparison | `model.py`公开16-block nested topology；bundled bound-`state_dict` checkpoint严格匹配 | paper↔source↔checkpoint core correspondence成立；release/checkpoint params `1,074,689` |
| Layerer architecture | 32+32+2→width512→128→32 | 无新增 | exact nested topology公开 | core correspondence成立；release-derived params `1,954,208` |
| Direction representation | Cartesian `ω_i,ω_o`，dense stratified hemisphere | sampler另用 half-vector GNDF | stored xy→reconstructed positive z | evaluator无 half/difference warp；code data order更具体 |
| Representation loss | raw BRDF `L1`，joint weights+latents | 无 | example只优化 latent，使用 `x/(1+x)` L1 且 output×light-z | **实质 paper-code-gap**；release不是 formal recipe |
| Data batching | `25⁴`/BRDF；formal batch未报告 | sampler 40×40 grids | example batch4096；index slice defect | release defect不能外推论文；也不能作为 faithful projection复现 |
| Training lifecycle | evaluator 50 epochs、layerer1000+finetune | sampler10 epochs | 无 formal train scripts；两个 checkpoint可按 release的 bound-method serialization语义载入 | checkpoints支持example inference/projection，但不能恢复formal joint training或sampler training |
| Runtime CUDA | Cutlass kernels + buffered Mitsuba | old block GPU bars；correction重发真实 timing | CUDA/Mitsuba不存在 | **paper-code-gap**；不能从 PyTorch脚本复现 timing |
| Importance sampler | main只指向 supplemental | full GNDF/proxy/training描述 | 不存在 | **paper-code-gap**；sample/pdf没有官方 oracle |
| Interpolation/mipmap | main只声明可做 | qualitative method/results | 不存在 | 无 formal mip asset builder/filter validation |
| Timing | 5 ms、faster-than-Belcour | correction撤回并替换 | 无 benchmark | correction拥有最高优先级 |
| Assets | 12k/720、600 base、1800 finetune | 3k/300 sampler，MERL/BTF results | 2 `.npy` examples +2 checkpoints | 数据集、split identity、trained sampler均不可得 |

release README 测试依赖为 PyTorch 1.2.0 与 OpenEXR 1.3.2，没有环境锁、renderer版本、Cutlass版本或构建说明。checkpoint 使用旧式 bound-method pickle；在 modern PyTorch 中审计需显式允许 non-weights-only load，但 keys与source严格匹配。真正成立的 release-code defect是 batch-slice coverage，不是 checkpoint调用。[C `README.md:L12-L15`; C audit]

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

- 只训练 isotropic material 与 isotropic medium；不含 anisotropic layer或 normal-mapped individual BRDF。[P §§3.2,5.3]
- 不显式构造 BTDF；top reflection latent不能独立提供 transmission query。[P §5.3]
- multi-lobe/highly-specular BRDF更难；无 failure-rate 或 worst-case dataset统计。[P §5.3]
- recursive layering会累积误差，可能需要固定更多层输入或 dynamic-input architecture。[P §5.3]
- neural bias可能造成系统性亮暗偏差或 energy violation；没有显式约束和 energy benchmark。[P §5.3]
- latent interpolation/mip filtering只做 qualitative验证，不保证等于 footprint 内物理 BRDF/radiance平均。[S §2.1; I scope]
- corrected CUDA cost随 fixed batch 和 neural-pixel count变化，不能用单一 5 ms表示所有 scene/resolution/coherence。[A-corr Tables 1–2]

### 12.2 未报告/材料不可得

- representation/layerer/sampler 的 optimizer、batch composition、network initialization、seed与checkpoint selection；
- 12,720/1,800 BRDF 的公开数据、exact parameter seeds、Guo sampling rate与reference variance；
- three-layer finetune 的 epochs、LR、时间与是否重新划分 validation；
- evaluator final output的非负约束、负值处理、color space、HDR尺度与 metric tone mapping；
- formal CUDA precision、weight layout、kernel source、Cutlass版本、texture packing、fetch count与layerer latency；
- sampling network input direction encoding、`σ/w`约束变换、Gaussian normalization公式、optimizer/batch、sample/pdf代码；
- GNDF 与真实 conditional BRDF importance distribution之间的 quantitative divergence和 transport variance统计；
- latent mip 的 level construction、边界处理、memory overhead、filtered GT、energy与temporal stability；
- MERL/BTF测试清单、projection budget、统计聚合和 unseen protocol；
- accompanying video 的固定版本与场景详细参数；
- correction 后正文 Table 3 的严格 total-wall-clock equal-time重新配平；
- 官方 license；repo根目录没有 LICENSE 文件。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

这不是“小 decoder + 强 compiler”的部署形态。函数容量主要在每次 direction query 都执行的约 1.07M-scalar shared evaluator；layerer又有约 1.95M scalars，但它可以在材质编辑/编译期执行。材质差异放在每 RGB texel 96 floats，而不是紧凑的 shared RGB code。论文的主要压缩收益来自“全材质共享大 decoder”，不是当前项目意义下的低-byte shader program。[P Fig.2/Table 1; C model.py; I]

layerer 的真正价值是把 **已经投影到共同函数坐标系** 的 operands组合。它不直接解决原生 LayerStack parameters到 canonical latent 的首次编码，也没有证明跨 source family 的 latent具有同一代数。因此它最适合作为 compiler/teacher 候选：用大 decoder和 optimized-code target检查“composition in latent space”是否可学，再决定是否蒸馏到部署形态。[I]

### 13.2 成功所依赖的假设

1. corpus 是窄而结构化的 isotropic GGX dielectric/conductor + unit-thickness isotropic medium family；
2. 每个 BRDF有 `25⁴` dense direction samples和昂贵 Guo reference；
3. 训练 BRDF latents与 shared decoder联合优化50 epochs，形成了 layerer依赖的共同 latent坐标系；
4. layerer supervision读到 optimized target latent，而不是只靠 source参数和函数 loss；
5. RGB 可逐通道独立处理，且不同通道不需要共享色散结构；
6. GPU query可聚成 1k–262k fixed batches；低-coherence单次query并非论文性能目标；
7. 质量论证主要是 outgoing-radiance images/MSE，energy、tail、worst-state和跨结构泛化未覆盖。[P/S/A-corr; I]

### 13.3 可迁移机制与不能迁移的部分

可直接进入候选设计的机制：

- 在冻结 decoder 上比较 optimized latent 与 feed-forward layerer output，显式量化 compiler gap；
- 递归 binary composition 对深度外推的误差曲线；
- “latent L1 vs decoded function loss”作为关键消融；
- layerer作为大容量 teacher，蒸馏到当前 M6 typed source compiler；
- 解析 Gaussian+Lambertian proposal作为低成本 sampler baseline；
- latent mip interpolation作为需要 filtered-reference验证的 baseline，而不是默认正确答案。[I; N model_candidates M6]

不能直接迁移的结论：

- 96 floats/texel与 1M runtime decoder不满足当前 shader预算，不能作为产品候选默认形态；
- 对 restricted synthetic layer family成立的 latent algebra不能代表 MaterialX、MDL、MERL 或无 layer语义 source；
- latent线性 interpolation不等于物理 BRDF mixture，latent mip也不自动等于 footprint filtering；
- corrected RTX 2080Ti coherent batch time不能预测当前 Falcor/Slang单次随机 query cost；
- release `compress.py` defect与paper-code gaps不能被写成当前 NVIDIA复现的 suspected defect。[I]

### 13.4 与本项目 runtime contract 的关系

两个公开网络的控制流与矩阵尺寸都是静态有界的，理论上可编译；但 evaluator约 `3.18M` RGB MAC/query、shared FP32 weights约4.10 MiB、state 384 B/texel，远超当前小型 MLP与低-byte latent目标。它可作为 `high-capacity teacher` 或 compiler capacity diagnostic，不是当前 runtime Pareto候选。[C/I]

正文 evaluator语义是 bare `f`，与本项目 `evaluate()`一致；但 release example的 cosine乘法不能直接进入本项目 ABI。supplemental proposal有解析 sample/pdf思想，却没有代码和条件化完整定义，不能直接组成可验收的 `sample()/pdf()` backend。[P §4.1; C compress.py; N scattering contract]

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

Fan 2022 不是当前 NVIDIA 2024 functional reproduction 的规范来源。下表因此不把两篇论文的自然方法差异误标为 NVIDIA 对 Fan 的 `budget-adaptation` 或 `intentional-deviation`；除非 Fan 的证据直接暴露当前实现违反 NVIDIA 自身合同，否则分类均为 `not-applicable`，再单列可迁移的研究影响。

| 主题 | Neural Layered BRDFs | 当前 NVIDIA functional reproduction | 分类与影响 |
|---|---|---|---|
| Shared representation | shared decoder + per-channel z32；每 asset靠优化投影 | z8 hierarchical latent + shared `20→64→64→64→3` evaluator | `not-applicable`：可把 latent bytes/runtime capacity作为跨论文 matched轴；当前形态不是 Fan 的预算改写 |
| Latent acquisition | train-time autodecoder；new BRDF只优化latent | source-parameter encoder bootstrap→materialize→latent finetune | `not-applicable`：当前 lifecycle 由 NVIDIA 一手证据定义；Fan 只提供 autodecoder control候选 |
| Compiler/operator | top/bottom latents + `A,σT`→new latent | 当前 NVIDIA method没有 binary latent layerer；M6另定义 typed source compiler | `not-applicable`：Fan layerer可新增为 matched compiler/teacher 候选，不是现有缺陷 |
| Direction features | raw Cartesian pair | learned frames内的 fixed/query 6D features | `not-applicable`：保持 NVIDIA论文定义；Fan raw pair只提供容量/坐标对照 |
| Evaluator output | paper bare `f`；release example混入 light-z | 当前 runtime adapter输出 bare linear `f` | `not-applicable`：两篇正文的 bare-`f`语义相容；Fan release gap不要求当前 ABI 做适配 |
| Sampler | offline GNDF→2-param Gaussian+Lambertian，per BRDF预计算 | per-`wo` learned 9-param two-lobe proposal；latent detached | `not-applicable`：两者可作 matched proposal Pareto 对照，不互相定义正确性 |
| LOD | standard latent mip/trilinear，仅qualitative | NVIDIA hierarchical z8 +论文规定的mip/filter lifecycle | `not-applicable`：Fan 的 mip construction欠披露，不覆盖 NVIDIA 自身 filtering 合同 |
| Runtime evidence | RTX 2080Ti fixed coherent batch，已勘误 | 当前有 Torch/Slang/package/viewer parity | `not-applicable`：硬件与cost domain不同，不做速度排名 |

当前 NVIDIA 对应证据来自 archive correspondence：encoder `K→64×4→8`、z8 mip、evaluator `20→64×3→3`、sampler `11→32×3→9`、300k/100k lifecycle与 latent detach均有现有 locator。[N `archive/.../research/correspondence.md#逐项对应`; `src/ncls/learning/methods/nvidia.py:L85-L105,L499-L551`; `src/ncls/learning/producer.py:L165-L176`]

本报告没有提出任何新的 NVIDIA `suspected-defect`。Fan release 的 batch-slice bug只属于其 experimental projection script；两篇论文在 encoder、direction encoding、sampler和runtime规模上的差别均是独立方法差异，只有经 §15 matched control 后才可转化为候选选择证据。[N/I]

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：binary latent layerer可在未见 LayerStack states 上逼近 optimized-code control | [P Eqs.1–2, Fig.6] | 当前 decoder latent可 canonicalize到可组合坐标 | 同一 frozen decoder：`z_target`逐态优化 vs `z_pred=L(z_top,z_bottom,A,σT)`；G2 holdout | source snapshots、query recipe、decoder、latent bytes、layerer train states、seeds | bare-f normalized L1、energy、peak、compiler gap CI | offline compiler/teacher | G2 gap显著且 bounded refinement不能收敛，或只记住训练组合 |
| H2：decoded function-space supervision优于论文 latent L1 | [P §4.2]只用 latent L1；latent非唯一性未处理 | 多个 latent可表示近似同一函数，直接 latent距离会惩罚等价解 | 同 layerer init/budget：latent L1 vs frozen-decoder query loss vs 二者组合 | decoder、target latents、queries、optimizer、steps、seeds | function error、latent error、G2/G2s、seed variance | compiler training only | function loss未改善G2且更差于latent L1，或cost不可接受 |
| H3：direct whole-stack compiler比递归 binary composition更能控制深层误差 | [P Fig.8, §5.3]明确 error accumulation | 当前 G2s 深层 stack包含 binary operator训练外结构 | 2/3层训练后，在4–N层对比 recursive layerer与order-aware direct compiler；同decoder/bytes | layer family、train depth、query、compiler params/MAC、seeds | error vs layer count、energy drift、order sensitivity | offline compiler | direct compiler没有降低误差增长，或同预算明显更差 |
| H4：普通 latent mip不足以满足物理 footprint filtering | [S Figs.1–2]只有qualitative anti-aliasing | nonlinear decoder使 `D(avg z)`不等于`avg D(z)` | 同base latent/decoder：box-average latent mip vs scale-supervised mip vs dense filtered reference | spatial asset、footprints、sampling、decoder、mip bytes | filtered radiance error、energy、temporal shimmer、fetch/time | prefilter asset | ordinary mip在全部matched指标与cost上不劣于scale supervision |
| H5：两参数 Gaussian+Lambertian proposal是当前9-param sampler的低成本 Pareto端点 | [S Eq.1, Figs.3–5] | 当前 layered source主要是单 specular lobe + diffuse background | 同 frozen evaluator：uniform、2-param GNDF、current 9-param；全部实现exact matched sample/pdf | source/query、SPP、MIS、random streams、evaluator、training queries | normalization、RMSE/variance vs spp/time、tail weights、sampler bytes/MAC | bounded analytic proposal | 两参数proposal接近uniform或尾部恶化，且成本收益不足 |
| H6：RGB共享latent可显著压低 Fan 的96-float state而不损失色散外观 | [P Fig.2/Table1]逐通道z32是成本主体 | 当前 source的跨通道结构可由joint head共享 | per-channel 3×32 vs joint z32/z16/z8，matched decoder MAC与train budget | source states、queries、decoder class、precision、seeds | quality/energy/peak按通道、state bytes、MAC | evaluator capacity diagnostic | shared code在同bytes下产生系统性色偏或长尾峰丢失 |

所有假设都先属于 report-only研究输入。H1–H4 的 layerer可超过软线作 capacity diagnostic，但进入产品候选前必须重新压到静态 shader预算；H5必须通过本项目 `sample()/pdf()`数学合同，而不是只比较 proxy图片。[N `docs/research/experiment_framework.md`; `docs/contracts/scattering_backend.md`]

## 16. 证据索引

### `P` Main paper

- §§3.1–3.3、Eqs.1–2：BRDF latent projection、binary layering定义和top/interface/medium边界。
- Figs.2–4（PDF p.3）：evaluator/layerer逐层结构、joint weight/latent与new-latent投影路径；已视觉核对。
- Table 1、Fig.5、Table 2（PDF p.4）：storage/capability对照、layer configuration与parameter distributions；已视觉核对。
- §§4.1–4.3（PDF p.5）：`25⁴` queries、12k/720/600/1800数据、loss/LR/epochs/hardware、buffered renderer；已视觉核对。
- Figs.6–11、Table 3（PDF pp.6–8）：function-grid MSE、multi-layer、MERL、complex scenes和原始 timing；已视觉核对，timing按 correction覆盖。
- §5.3：BTDF、anisotropy/normal mapping、多峰/高光、递归误差、bias/energy限制。

### `S` Supplemental

- §1.1、Figs.1–2：latent interpolation、mipmap、footprint/trilinear与qualitative结果。
- §1.2、Eqs.1–2：Gaussian+Lambertian proxy、GNDF、40×40 recipe、3k/300 split、KLD/LR/epochs与analytic sample/pdf。
- Figs.3–5：proxy、BRDF-only sampling 256 spp与MIS qualitative对照。
- Figs.6–10：MERL、BTF、reciprocity、4/8/16-block质量趋势和六例representation MSE；全5页已渲染视觉核对。

### `A` Author material / correction

- project page：正式作者/venue、摘要和下载入口。
- correction §§1–2、Tables 1–2、Fig.1：撤回5 ms与faster-than-Belcour，给出6个batch times、5个scene per-spp GPU times和正确equal-time Belcour对照；整页已视觉核对。

### `C` Official release commit `904538a1197bcf4583caa748a5162119ae82a944`

- `README.md:L1-L36,L45-L51,L83-L97`：experimental身份、资产范围、390625-query example与文件格式。
- `model.py:L18-L111`：evaluator exact residual/LN/ReLU topology。
- `model.py:L117-L162`：layerer exact topology。
- `utils.py:L49-L85`：96D RGB latent config与4D xy→6D direction重建。
- `compress.py:L27-L34,L49-L113`：time seed、latent-only Adam、batch4096、schedule、`x/(1+x)` L1、light-z乘法、bound-method checkpoint调用和overlapping-slice defect。
- `layering.py:L40-L44,L73-L87`：bound-method checkpoint调用、RGB逐通道 layerer、latent centering与`A,σT`输入。
- `visualize.py:L37-L70,L87-L90`：RGB三次scalar decoder调用、cosine乘法与bound-method checkpoint调用。
- `saved_model/*.pth`（SHA见 source ledger）：反序列化类型均为绑定的 `state_dict` method；调用后可严格载入当前 evaluator/layerer source。evaluator bound owner为 `1,074,689` parameters；旧式 archive源码检查会产生 `SourceChangeWarning`，但不是 topology mismatch。

### `N` NeuralShading evidence

- `archive/2026-08/08-27-faithful-nvidia-neural-materials/research/correspondence.md#逐项对应`：当前 NVIDIA encoder、z8 mip、learned frames、evaluator/sampler与lifecycle身份。
- `src/ncls/learning/methods/nvidia.py:L85-L105,L499-L551`：当前参数布局、materialize与 detached sampler objective。
- `src/ncls/learning/producer.py:L165-L176`：当前 evaluator/sampler typed route边界。
- `docs/contracts/scattering_backend.md`：bare `f`、solid-angle PDF与matched `sample()/pdf()`合同。
- `docs/research/model_candidates.md#7-m6typed-source-compiler仅参数式族`：当前 compiler control/quality compiler的角色边界。

### `I` Derived/transfer notes

- release parameter/MAC totals与4K latent bytes为按公开shape逐项求和；不是paper-reported benchmark。checkpoint parameter与strict-key审计只用于确认公开资产对应关系，不替代正式 runtime benchmark。
- 论文 timing与supplemental旧柱形按 official correction supersede；未自行恢复未公开的正式配置。
- 对当前 NVIDIA 的所有对照都明确标为 `not-applicable`；没有新增 suspected defect。

### 建议提升的 load-bearing 论文

- **Guo et al. 2018, Position-Free Monte Carlo Simulation for Arbitrary Layered BSDFs**：`direct-inheritance` + `key-baseline`。它定义本论文 layered GT、训练数据和几乎全部质量结论；应完整复核 reference配置、variance、layer assumptions 与代码可得性。
- **Belcour 2018, Efficient Rendering of Layered Materials Using an Atomic Decomposition with Statistical Operators**：`key-baseline` + `failure-explanation`。官方 correction 唯一撤回的跨方法速度结论直接涉及它；提升后才能正确理解 corrected equal-time quality与cost domain。
- Rainer et al. 2020 只在本论文中承担 representation/BTF qualitative baseline；若后续综合要讨论 unified BTF latent或其 grazing failure，再按 `key-baseline` 提升，否则保持 discovery/load-bearing候选即可。

## Evidence review

```text
author_worker: /root/taming2026
reviewer: /root/rta2024
reviewed_at: 2026-08-29
sources_rechecked:
  - main PDF, 8/8 pages, formulas/tables/figures/captions/footnotes visually rechecked
  - supplemental PDF, 5/5 pages, formulas/figures/captions visually rechecked
  - official timing correction, 1/1 page, tables and corrected Figure 1 visually rechecked
  - official repo commit 904538a1197bcf4583caa748a5162119ae82a944, all tracked source/config/example data and bundled checkpoint binding/topology rechecked
findings_closed:
  - formal evaluator/layerer topology and release-source parameter arithmetic
  - 25^4 direction-pair interpretation and 390625-query release correspondence
  - correction precedence over 5 ms, supplemental timing bars and faster-than-Belcour claim
  - corrected batch/scene timings and corrected Belcour equal-time context
  - release batch-slice coverage defect scoped to example projection
  - release bound-state-dict serialization semantics and checkpoint-to-source strict-key correspondence
  - sampler hot-path boundary, failure labels and NVIDIA N/I boundary
remaining_evidence_gaps:
  - formal training, CUDA renderer and sampler code unavailable
  - optimizer, batch composition, seeds and checkpoint selection unreported
  - correction does not republish a strictly rebalanced Table 3 equal-time protocol
  - anisotropic/BTDF/deep-stack quantitative evidence unavailable
  - filtered-reference, energy and sampler-variance tables unreported
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
