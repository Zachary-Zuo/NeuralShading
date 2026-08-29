---
paper_id: "mo-2025-dual-band-neural-gi"
title: "Dual-Band Feature Fusion for Neural Global Illumination with Multi-Frequency Reflections"
authors: "Shaohua Mo, Chuankun Zheng, Zihao Lin, Dianbing Xi, Qi Ye, Rui Wang, Hujun Bao, Yuchi Huo"
year: "2025"
venue: "SIGGRAPH Conference Papers '25"
doi: "10.1145/3721238.3730733"
report_status: "evidence-reviewed"
main_source: "https://mshnb.github.io/dualbandfusion/static/pdfs/3721238.3730733.pdf"
supplemental_status: "available"
official_code_status: "unavailable"
official_code_commit: "not-applicable"
author_worker: "/root/dualband2025"
reviewer: "/root/dualband2025_review"
last_verified: "2026-08-29"
---

# Dual-Band Feature Fusion for Neural Global Illumination with Multi-Frequency Reflections

## 1. 研究对象与报告边界

本文研究的是一个 **scene-level neural global illumination** 方法：对一个已经离线训练的动态场景，运行时从当前相机的 first-hit G-buffers 出发，再沿理想镜面反射方向执行一次额外 ray-scene intersection；两组命中信息分别查询 object-centric feature fields，形成 principal/secondary features，随后用 screen-space 多尺度 kernel-prediction CNN 把二者融合并解码为当前像素的 RGB radiance。

这里的 `dual-band` 不是显式 Fourier/wavelet 分解，也不是把最终图像切成两个固定频带。作者按 **相对 view direction 的角频率**解释两个由不同查询机制产生的 feature band：object-centric fields 的 principal feature 倾向承载较低角频率但可有很高空间频率的外观；理想镜面方向的 secondary feature 提供更尖锐、更稀疏的反射线索。方法用 learned screen-space filtering 填补二者之间最难的中频 glossy reflection，而不是先给 radiance 做频谱变换。[P Fig.1、§3.1–3.4]

本报告覆盖 DOI `10.1145/3721238.3730733` 对应的 11 页正式论文、4 页正式 supplemental、作者项目页、作者链接的 supplementary video 入口，以及项目站点公开仓库的静态 source audit。边界如下：

- 它学习的是 scene-dependent `L(x, wo)`，不是 local material `f(wo, wi)`；输出已经包含 illumination、visibility、geometry 和 material 的耦合。
- 它没有提供 matched `sample()/pdf()`，也不输出可进入 path tracer 的局部 scattering law。
- 论文实验按 scene 训练，没有证明跨 scene、跨 object family 或新 geometry 的泛化。
- “dynamic scene”在正式实现中明确覆盖 camera、rigid object transform 和 variable material parameters；正文的宏观目标提到 lights，但公开的数据配置与网络输入没有给出动态 light parameterization，不能把动态灯光当作已验证轴。
- 不把 OIDN、AE、FieldGI 的完整方法实现归入本文；本文只提供它们在作者 protocol 下的 baseline 结果及一个作者增强的 `AE-Ref`。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---:|---|---|
| Main paper `P` | [作者项目页 PDF](https://mshnb.github.io/dualbandfusion/static/pdfs/3721238.3730733.pdf)，DOI `10.1145/3721238.3730733` | 2026-08-29 | SHA-256 `F66E9CEF82BE5E4E45D98B8339EB6E0EF359BAD3C5F76A424777FF486D5B60A3` | 正式论文，11 页；全文、公式、Table 1、网络/消融/失败图已逐页阅读，关键页视觉核对。 |
| Supplemental `S` | [作者项目页 supplemental](https://mshnb.github.io/dualbandfusion/static/pdfs/supp.pdf) | 2026-08-29 | SHA-256 `EE50C9612864BE33F75216C55AD18789511F0090FFACC1539311E0DB539FD0FB` | 4 页；给出 object field、hypernetwork/object/final decoders、multi-resolution CNN 的逐层图，以及 foliage/复杂几何/长 specular path 结果；全部视觉核对。 |
| Official code/config/data `C` | [公开项目仓库](https://github.com/mshnb/dualbandfusion/tree/05264274eaa54f9641a191dbe53c6b5d2d8051fe)，`gh-pages` commit `05264274eaa54f9641a191dbe53c6b5d2d8051fe` | 2026-08-29 | tree audit：23 个 entry，完整 recursive tree 未截断 | 仓库只包含 `index.html`、CSS/JS、teaser 与 main/supp PDF；没有训练/runtime code、config、checkpoint、scene/data manifest 或 license。该 commit 是项目页版本，不是 official implementation commit。 |
| Author/project page `A` | [项目页](https://mshnb.github.io/dualbandfusion/) | 2026-08-29 | HTML SHA-256 `32F6258F487045BAF7226F74DE9F70560198F694373476B8BBCF3F88A6964E9A`；页面仓库 commit 同上 | 作者、机构、摘要、main/supp/video 入口；没有 correction 或额外配置。 |
| Author video `A-video` | [作者项目页嵌入视频](https://www.youtube.com/watch?v=32rLtfqKauY)，YouTube oEmbed 将作者标为 Shaohua Mo | 2026-08-29 | video id `32rLtfqKauY`；oEmbed SHA-256 `0A21F2172A27E839DEF775E2F474943B1E77F09A2759A8035AB60F6A0A465B60` | 动态定性展示入口；没有可锁定的 slides/transcript/config，故不用于补写网络与训练数值。 |
| Bibliography metadata | 正式 PDF 首页、DOI/Crossref record | 2026-08-29 | Crossref cache SHA-256 `5C0D7403A6121CFA069AE812D7B586CA33E7988FB83B4FD40A278433E2569C9D` | 交叉核对正式标题、8 位作者、proceedings article、页码 `1–11` 与 2025 年出版信息；技术事实仍以 P/S 为准。 |
| NeuralShading evidence `N` | [`docs/realtime_material_compilation.md`](../../../../../docs/realtime_material_compilation.md)、[`docs/research/experiment_framework.md`](../../../../../docs/research/experiment_framework.md)、[`docs/research/model_candidates.md`](../../../../../docs/research/model_candidates.md)、[`configs/learning/nvidia-rta2024-materialx-formal.json`](../../../../../configs/learning/nvidia-rta2024-materialx-formal.json)、[`.trellis/spec/project/method-constraints.md`](../../../../spec/project/method-constraints.md) | 2026-08-29 | workspace current | 只用于 §13–15 的项目映射，不反向补全论文事实。 |

第一方入口的 main 与 supplemental 均可公开获取，不需要登录或凭据。项目页公开 tree 与项目作者名下的公开 repository list 中均未发现对应 implementation；公开 repository exact-title search 也没有结果。因而本报告把 official code/config/data 记为 `unavailable`，并保留所有只有图而没有 executable correspondence 的字段。

## 3. 原论文的问题、假设与贡献边界

作者把目标 radiance field 写为

```text
L(x, wo) = R(x, wo; S, L),
```

其中 `S`、`L` 分别是 scene objects 与 light sources，`R` 是依赖 neural representation/modules 的抽象 rendering process。困难来自 radiance 对空间与 view direction 的频率跨度很大：diffuse appearance、shadow、rough reflection、glossy reflection 到近镜面反射不能由同一种低容量连续网络同样容易地表示。[P Eq.(1)、§3.1]

论文的核心假设分为三层：

1. object-centric high-resolution feature fields 能表达复杂的 **spatial variation**，但只靠 MLP 从 first-hit buffer 生成 view dependence 时，angularly high-frequency glossy/mirror reflection 仍难学；
2. 沿当前 shading point 的 mirror direction 做一次显式 query，可以给出 reflected area 的高频位置线索；该线索对近镜面情况有用，但对 rough/glossy reflection 来说只是一个稀疏方向样本，不能直接等价于整段 BRDF lobe 积分；
3. 让一个 kernel-prediction CNN 根据 roughness、几何、反射命中和两类 feature 自适应聚合 screen-space 邻域，可以数据驱动地近似 rough reflection 对 incident lighting 的低通作用；再加入 principal feature 和一个 virtual zero vector，可以同时补回整体外观并表达 dynamic occlusion 的能量衰减。[P §3.1–3.4、Fig.2–3]

作者声明的贡献包括：

- 一个能在 dynamic scenes 中生成 multi-frequency reflections 的 neural GI pipeline；
- 将 principal/secondary features 转成连续多频 fused feature 的 dual-band fusion module；
- 先低频、再全频的 staged optimization，用于提高整体质量并减少反射伪影。[P Introduction contributions]

贡献边界是 **scene-specific image/radiance reconstruction**。论文没有声称从 source material graph 编译一个局部材质，也没有从任意未见 scene 直接运行；它以大量离线 rendered images 学习特定 scene 的 transport。`all-/multi-frequency` 是作者在该 scene/data domain 下对 appearance range 的描述，不是对任意 light transport frequency 的完整覆盖保证。

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/scene input | 每个 scene 的 object set；每个 object 有 object-space multi-resolution triplane、world-to-local matrix `M_i` 与 variable material parameters `v_i`。 | per-scene learned state；`r_i=(M_i,v_i)`。`v_i` 的字段数 `m` 未报告。 | `P` Eq.(5)–(7)、§3.2；`S` Fig.1–2 |
| First-hit runtime query | 当前 pixel 的 surface point `x`、view direction `wo` 与 first-hit G-buffers `g_f`。 | full frame，正式实验 `512×512`；object decoder 实际 G-buffer 是 position 3、normal 3、view direction 3、albedo 3、roughness 1，共 13 channels。 | `P` Eq.(2)；`S` Fig.3 |
| Reflection runtime query | 论文把 `omega_r` 称为由 `omega_o` 决定的 mirror reflection direction，但 Eq.(3) 原样写成 `x_r=I_{S,L}(x,-omega_r)`；Fig.2/caption 又口头称沿 mirror reflection direction 做 single-bounce ray tracing。 | 每个 shading pixel 一条 single-bounce mirror ray；`omega_r` 的入/出射符号约定、ray origin offset、miss/background behavior 均未报告，不能静默删掉公式中的负号。 | `P` Eq.(3)、§3.1、Fig.2 |
| Object/local coordinates | 所有输入 G-buffers 先由 `T_i` 转到 object `i` 的 local space，再查询该 object 的 triplane/decoder；各 object feature 在 screen space element-wise sum。 | object-space planes + screen-space feature aggregation。 | `P` Eq.(5)–(7)；`S` Fig.1 |
| Fusion CNN input | roughness、`dot(view,normal)`、reflection depth、RGB reflection emission、32-D summed principal feature、32-D summed secondary feature。 | `512×512×70`。没有 position/normal raw channels直接进入该 CNN；它们已参与 object feature query。 | `S` Fig.5 |
| Output quantity | 方法的概念输出是 RGB final color/radiance `L(x,wo)`，supplemental 的 network head 为 3 channels。 | 正式训练/实验均为 `512×512×3`；论文对 HDR target 使用 `log1p`，但没有闭合 head 输出所处域、inverse mapping、color space、exposure 或 direct/indirect 拆分。 | `P` Eq.(4)、§4.1、Fig.2；`S` Fig.4 |
| Dynamic/editable axes | camera、rigid object transforms、variable object material parameters；训练场景展示 moving/rotating objects 与 variable roughness。 | 同一已训练 scene 内。动态 light 的输入/采样 protocol 未报告。 | `P` §4.1、Conclusion；`S` Fig.1 |
| Validity/domain restrictions | opaque surface materials、rigid transformations；一条 mirror reflection query；依赖屏幕邻域能复用 reflection information。 | 不覆盖 deformable/translucent；复杂法线变化和多次 specular bounce 是明确失败区。 | `P` Conclusion；`S` Fig.7–8 |

论文的 `omega_r` 是为当前 `wo` 与 normal 构造的 mirror direction。它不是对 BRDF lobe 的随机样本，也没有对应的 sampling PDF；roughness 引起的 lobe integration 交给 learned screen-space aggregation 近似。[P §3.1、§3.3]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

对 object `i`，作者定义

```text
F_i(g_i, r_i) = D_i(g_i, r_i, mathcal_F_i(x_i)),
r_i = (M_i, v_i).
```

`mathcal_F_i` 是 object-space multi-resolution triplane，`D_i` 是由 hypernetwork 根据 `r_i` 生成参数的 lightweight object decoder。对同一套 G-buffers，方法把每个 object 的输出逐元素相加：

```text
F(g) = sum_i F_i(T_i(g), r_i).
```

分别用 first-hit 与 reflection buffers 查询同一个 scene field：

```text
f_p = F(x, wo, g_f)
f_s = F(x_r, omega_r, g_r).
```

随后 fusion CNN `C` 从 `g_f,g_r,f_p,f_s` 预测三尺度的 `5×5` kernel logits `z^l`、每尺度 level logits `beta^l`，以及 interpolation weights。正文 Eq.(8) 只写一张 full-resolution `gamma`；supplemental Fig.5 则在三尺度各输出 3 channels，再经未给公式的 upsample-and-merge 得到最终 `gamma`。每个尺度先对 kernel 做 softmax，再卷积 secondary feature：

正文 Eq.(9) 必须原样保留其索引不一致，而不能在转述时静默“修好”：

```text
tilde_f_s^l(p) = sum_{q in N(u)} w_u^l(q) f_s^l(q)
w_u^l(q) = exp(z_u^l(q)) / sum_{q' in N(p)} exp(z_u^l(q')).
```

紧随公式的文字又称 `N(p)` 是“centered around pixel u”。由上下文可推测作者想表达以同一中心像素归一化并卷积，但这只是意图解释；正式 `p/u` correspondence 在 §11 保持未闭合。

三尺度结果用 softmax-normalized level weights汇合：

```text
tilde_f_s = sum_l U_l(tilde_beta^l * tilde_f_s^l).
```

最后把 principal、filtered secondary 与 zero vector 做 softmax 权重混合：

```text
f_fused = tilde_gamma_p f_p + tilde_gamma_s tilde_f_s + tilde_gamma_e 0.
L_hat = G(g_f, f_fused).
```

`zero` 分支本身不增加 feature 内容，但让总的非零 feature 权重可以低于 1，作者用它表达 dynamic occlusion 造成的 attenuation。[P Eq.(5)–(11)、Fig.2–3]

### 5.2 持久化表示

- 每个 object 有 8-level、3-plane 的 dense multi-resolution triplane。每个 level 每 plane 有 4 channels；supplemental 只明确 coarsest resolution 为 `8`、finest 为 `1024`，没有列出六个中间 resolution，不能把逐级倍增静默当作正式配置。[S §1、Eq.(1)、Fig.1–3]
- 同一 plane 的 8 个 level 在采样点拼接成 32-D；三张 plane 合成 96-D，再经过一个 linear layer降为 32-D object feature。[S p.2–3]
- object representation `r_i` 输入 hypernetwork，生成 `10,240` 个 object decoder parameters。图中标为在 parameters update 时更新，但 update frequency、cache lifetime 与 state layout 未报告。[S Fig.1–3]
- final decoder、fusion CNN、hypernetwork、feature fields 都随 scene 训练。第一方材料没有说明其中哪些 neural weights 跨 object 共享，也没有证明任何部分跨 scene 共享。
- 没有 mip/LOD semantics、quantization、sparse storage、out-of-core streaming 或 asset package。虽然 triplane 是多分辨率的，它用于 progressive training/feature capacity，不等于论文已经提供 footprint-correct LOD。

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Multi-resolution triplane | object-local position `x_i` | 8 levels；每 level 三个 `XY/XZ/YZ` planes，coarsest/finest resolution 分别为 `8/1024`，4 channels/level/plane；采样后每 plane 32-D，三 plane 96-D，再 linear `96→32` | 六个中间 resolution、plane interpolation mode、linear bias 未报告 | 32-D grid feature | per object | `S` Eq.(1)、p.2–3 |
| Hypernetwork | 16-D world-to-local matrix + `m` material parameters | FC `(16+m)→512`；5 个 `512→512` hidden layers；输出层 `512→10240` | hidden blocks 图示 FC+bias+LeakyReLU；output 图示 FC+bias、无 LeakyReLU | 10,240 object-decoder parameters | sharing across objects 未明确 | `S` Fig.2 |
| Object input alignment | 13-D local G-buffer + 32-D triplane feature = 45-D | 图注另写需要一个 `45×64` FC 对齐 input channels | activation/bias 未单独说明 | 64-D | 未报告 | `S` Fig.3 note |
| Object decoder `D_i` | 64-D aligned feature | `64→64`、`64→64`、`64→32` | 前两 block 图示 LeakyReLU；图例未画 bias；最后一层无 activation | 32-D contribution | decoder parameters由 `r_i` 动态生成 | `S` Fig.3 |
| Final decoder `G` | 32-D fused feature + RGB albedo + roughness = 36-D | `36→256`；随后图示 4 个带 original-input skip 的 `(256+36)→256` layers；final `(256+36)→3` | supplemental 图例和 final block 均明确画为 FC+LeakyReLU；negative slope 未报告，且 `log1p`/inverse、额外 clamp 或 radiance-domain transform 未闭合 | RGB final color | per-scene weights | `S` Fig.4 |
| Fusion CNN encoder/decoder | `512×512×70` | symmetric U-Net：`512²×32 → 256²×64 → 128²×128 → 64²×128` encoder；transpose-conv/skip path 输出 `128²×256 → 256²×128 → 512²×64` | encoder/decoder 图示 `3×3 Conv/ConvTranspose + ReLU` 与 AvgPool；padding/boundary mode 未报告 | 三个 resolution 的 feature maps | per-scene weights | `S` Fig.5 |
| Per-scale prediction heads | U-Net 的 `128²×256`、`256²×128`、`512²×64` features | 每尺度 `3×3 Conv + ReLU` | ReLU；论文随后对 kernel/level/interpolation weights做 normalization | 每尺度 29 channels：25 kernel + 1 level + 3 interpolation | shared across pixels | `S` Fig.5 |
| Feature aggregation | secondary feature at 3 resolutions | 3 个 `5×5` softmax-normalized spatial kernels；三尺度用 softmax level weights upsample/sum | kernel softmax、level softmax | 32-D filtered secondary | per frame | `P` Eq.(8)–(10)；`S` Fig.5 |
| Zero-enhanced interpolation | principal、filtered secondary、zero | 3-way softmax-weighted element-wise blend | softmax | 32-D fused feature | per pixel | `P` Eq.(11)、Fig.3 |

逐层图仍不能替代 code：CNN skip merge operator 未直接标注（channel shape 与 concatenate 一致，但无文字确认）、卷积 padding、down/up sampling corner convention、LeakyReLU negative slope、final head 与 `log1p`/inverse 的关系、triplane interpolation、ray sign/origin/miss encoding 与 object count batching 均未公开。

### 5.4 条件化、坐标变换与物理先验

- **Object-centric conditioning**：将 geometry buffer 变换到每个 object local frame，使 rigid motion 后仍查询同一个 canonical feature grid；`M_i` 与 material params 通过 hypernetwork改变 object decoder。
- **Principal band**：first-hit query 的 field 受高分辨率 spatial grid 支撑，因此可保存 hard/soft shadows、caustic-like spatial pattern 等空间高频，但 view dependence 仍要经 compact decoder 表达。
- **Secondary band**：mirror hit 给 reflected object/area 的直接位置与 emission clue，随 view 变化很快；它在 screen space 稀疏，只覆盖每 pixel 一个 mirror direction。
- **Learned roughness filtering**：CNN 读取 roughness、`dot(wo,n)`、reflection depth/emission 及两类 feature，预测 3 个尺度的 `5×5` kernel；这是一种受物理过程启发的 screen-space approximation，不是显式 BRDF convolution，也没有 energy/reciprocity guarantee。
- **Occlusion gating**：virtual zero branch允许 principal/secondary 两个非零分支同时被衰减，用于 abrupt dynamic occlusion。它没有显式 visibility equation或 path throughput监督。
- **Progressive field weighting**：stage 1 用训练进度 `alpha∈[0,1]` 逐级解锁 triplane level：第 `j` 级在 `alpha<j/L` 为 0，在 `[j/L,(j+1)/L)` 线性增至 1，之后保持 1。[S Eq.(2)]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Scenes | `Hall`：可旋转、variable-roughness metallic-paint car；`Coffee`：两个 movable pots、不同 materials；`Bathroom`：大 mirror、variable roughness，用于反射物体动态变化。 | `P` §4.1 |
| Training configurations | 每 scene 8,192 个随机 camera + dynamic-object configurations；一半来自 stage-1 roughness interval，另一半来自相反 interval。 | `P` §4.1 |
| Roughness partition | stage 1 将所有 dynamic objects 的 surface roughness 设为 `>=0.25`；因此 4,096 个配置属于该较低角频 subset，另 4,096 个属于其相反区间。 | `P` §3.5、§4.1 |
| GT/reference renderer | Falcor path-traced radiance 与相应 G-buffers；每 image `1024 spp`、`512×512`。 | `P` §4.1 |
| GT denoising | 所有 GT radiance 还经过 Yu et al. 2021 的 offline denoiser 降低 rendering noise。denoiser checkpoint、输入 channels、residual noise 与 bias 未报告。 | `P` §4.1 |
| First-hit query data | position、normal、view direction、albedo、roughness；用于 object-field query。 | `S` Fig.3 |
| Reflection query data | single-bounce mirror hit 的同类 local G-buffers；fusion CNN 直接读取 reflection depth 与 RGB emission。miss/default values 未报告。 | `P` Fig.2；`S` Fig.3、Fig.5 |
| Train/validation/test split | 论文没有给出 8,192 configurations 的 train/validation/test 数量、random seed、holdout rule、evaluation frame 数或是否保持 config-disjoint。 | `P/C` unavailable |
| Camera/object/light sampling | 只写 random camera/dynamic objects；range、distribution、collision/visibility constraints 未报告。没有正式 dynamic-light sampling recipe。 | `P` §4.1 |
| Time/history sampling | 没有 sequence training、recurrent state、history buffer 或 reprojection；作者视频用于定性动态比较。相邻 frames 的 trajectory/metric 未报告。 | `P` architecture、§4.2；`A-video` |
| Filtering/LOD/footprint | fusion CNN 固定使用三尺度 screen-space aggregation；没有 texture footprint、mip selection 或 world-space LOD protocol。 | `P` §3.3；`S` Fig.5 |
| Online/offline generation | GT images/G-buffers 离线生成并用于 per-scene training；runtime 只做 current-frame G-buffer、one-bounce ray query 与 neural inference。 | `P` §3.5–4.1 |

`1024 spp` 再经过 neural denoising的图像是本文的 training target，不是严格无偏 path-tracing oracle。论文没有提供 raw 1024-spp images、pre-denoise reference、denoiser bias audit 或 GT uncertainty；因此 Table 1 的误差是相对作者这套 processed targets，而不是相对无限-spp radiance。

## 7. Loss、optimizer 与训练 lifecycle

| 项 | 正式配置 | locator |
|---|---|---|
| Target/output transform | 作者称对 HDR radiance 使用 `log1p` mapping `x_tilde=log(1+x)` 以降低训练不稳定；公式没有明确标出 mapping 位于两个 loss term 的哪一侧/哪一步。 | `P` §4.1 |
| Loss | `L = L1(L, L_hat) + L_SSIM(L, L_hat)`；`L1` 为 mean absolute error，`L_SSIM` 为 structural dissimilarity。两项系数按公式均为 1。 | `P` Eq.(12) |
| Optimizer | Adam。betas、epsilon、weight decay、gradient clipping 未报告。 | `P` §4.1 |
| Stage 1 data | 仅 roughness `>=0.25` 的 low-frequency subset。 | `P` §3.5 |
| Stage 1 graph | dual-band fusion disabled；principal feature直接进入 final decoder；triplane levels 从低到高 progressive unlock。 | `P` §3.5、Fig.2；`S` Eq.(2) |
| Stage 1 LR/batch | learning rate `1e-3`；mini-batch size `32`。 | `P` §3.5、§4.1 |
| Stage 2 data | 完整 8,192-config dataset，无 roughness restriction。 | `P` §3.5 |
| Stage 2 graph | 激活 secondary feature 与整个 fusion module，并训练 entire framework；从 stage-1 appearance prior 初始化。 | `P` §3.5、Fig.2 |
| Stage 2 LR/batch | learning rate `1e-4`；mini-batch size `20`。 | `P` §3.5、§4.1 |
| Steps/epochs/stage boundary | 两阶段的 step/epoch 数、各自 wall time、LR schedule、data traversal 与 stage transition checkpoint 未报告。 | `P/C` unavailable |
| Initialization/seed/model selection | weight/feature-grid initialization、random seeds、repeat runs、validation interval、early stopping、checkpoint selection 未报告。 | `P/C` unavailable |
| Hardware/time | 4× NVIDIA RTX A6000 上约 12 小时；Conclusion 另写每 scene 超过 40 GPU hours，二者在量级上相容（12 wall-hours × 4 GPUs 约 48 GPU-hours）。 | `P` §4.1、Conclusion |

“staged optimization 防止 sub-optimal minima”是作者依据 Fig.10 的定性解释；没有 loss curve、重复 seed 或 basin analysis，因此不能把它升级为一般优化定律。

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path | first-hit G-buffer pass → 每 pixel 一条 mirror ray 获得 reflection G-buffers → 两次 scene field query形成 principal/secondary → full-frame fusion CNN → final decoder。 | `P` Fig.2、Eq.(3)–(4) |
| Call frequency | object decoder parameters在 object representation 更新时由 hypernetwork生成；per-frame/per-object 的 update policy 未报告。field/fusion/final decode 至少随每帧当前 view 执行。 | `S` Fig.1 |
| Parameter count/MAC/FLOP | 未报告。supplemental 只给层宽/shape，没有总 parameter/MAC；object count、scene-specific field memory 也没有汇总。 | `S` Fig.1–5 |
| Shared/per-asset/state bytes | 未报告。dense triplane、hypernetwork、CNN、decoders 的 precision 与 serialization layout 不披露。 | `P/S/C` unavailable |
| Texture/feature fetches | 每 object、每 first/reflection query 都需要 multi-level tri-plane sampling，再做 object-wise sum；具体 interpolation fetch count 与 object culling 未报告。 | `P` Eq.(5)–(7)；`S` Fig.1 |
| Precision | Table 1 的 AE、AE-Ref、FieldGI 和 Ours 都在 PyTorch FP32 测量。没有 FP16/quantization/export。 | `P` Table 1 caption |
| Hardware/backend | NVIDIA RTX 4090；PyTorch FP32。ray tracing backend、Falcor/PyTorch synchronization、kernel warm-up、measurement repetitions 未报告。 | `P` Table 1 caption |
| Hall | Ours `22.42 ms`；OIDN `23.09 ms` at `22 spp`。 | `P` Table 1 |
| Coffee | Ours `26.07 ms`；OIDN `26.14 ms` at `25 spp`。 | `P` Table 1 |
| Bathroom | Ours `24.55 ms`；OIDN `24.57 ms` at `29 spp`。 | `P` Table 1 |
| OIDN timing scope | 明确包含 path tracing + denoising，并调 path-tracing spp 使整帧时间与 Ours 对齐。caption 在两项 process 后括注 `approximately 1–2 ms for all scenes`，但没有明确写它单独修饰哪一项；它显然也不是表中 `23–26 ms` 的 combined total。因此不能把 `1–2 ms` 硬归给 path tracing 或 denoising。 | `P` Table 1 caption |
| Ours timing scope | 表中只写 PyTorch FP32 timing，没有明确说明 first-hit/reflection ray pass、hypernetwork parameter update、G-buffer generation 是否都包含。 | `P` Table 1 caption |
| Temporal/history cost | 没有 history/reprojection/recurrent state；也没有 temporal metric。 | `P` architecture、§4.2 |

作者认为 kernel prediction/computation 若做 specialized optimization 还可显著提速，但没有提供 optimized backend 数据。该预测是 future optimization claim，不能当成本文已经测得的 speedup。[P Table 1 caption]

## 9. 实验 protocol、baseline、指标与结果

### 9.1 Baseline correspondence

- `AE`：Active Exploration 的 neural renderer，只用 first-hit G-buffers；作者称 vanilla MLP 无法准确处理 Bathroom mirror 等 high-frequency view dependence。
- `AE-Ref`：本文作者在 AE 上额外加入 reflection G-buffers 的增强实现，不是被引用 AE 论文的原始方法；它能生成尖锐反射，却会在 rough surface 上保留过于锐利的 reflected outlines。
- `FieldGI`：作者使用自己的 simplified object-centric field backbone来源；baseline 仍靠 MLP 表达 view dependence，主要困难是高光/镜面细节。
- `OIDN`：低-spp path tracing + Intel Open Image Denoise。作者逐 scene 调 spp，使 total runtime 与 Ours 接近；这不是 matched sample count，而是 matched observed frame time。

AE/AE-Ref/FieldGI/Ours 都在作者完整 dataset 上训练，作者称 neural baselines 的训练时间近似相等，但没有给出 exact architecture、parameter count、optimizer/config 或 checkpoint。[P §4.2]

### 9.2 Table 1 的完整作者报告值

全部数值来自同一 Table 1；time 在 RTX 4090，neural methods 为 PyTorch FP32。论文没有说明 L1/PSNR/SSIM/LPIPS 的 color transform、exposure、crop、frame count、per-image/whole-corpus aggregation，也没有 confidence interval。[P Table 1]

| Scene | Method | Time ms | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|
| Hall | AE | 62.55 | 0.0193 | 27.93 | 0.9402 | 0.0568 |
| Hall | AE-Ref | 67.68 | 0.0160 | 30.03 | 0.9549 | 0.0385 |
| Hall | FieldGI | 18.34 | 0.0147 | 30.00 | 0.9471 | 0.0465 |
| Hall | OIDN (`22 spp`) | 23.09 | 0.0120 | 33.85 | 0.9559 | 0.0292 |
| Hall | Ours | 22.42 | 0.0074 | 36.70 | 0.9811 | 0.0127 |
| Coffee | AE | 64.58 | 0.0125 | 28.41 | 0.9477 | 0.0504 |
| Coffee | AE-Ref | 68.87 | 0.0117 | 29.44 | 0.9554 | 0.0458 |
| Coffee | FieldGI | 21.43 | 0.0114 | 29.16 | 0.9552 | 0.0337 |
| Coffee | OIDN (`25 spp`) | 26.14 | 0.0104 | 32.59 | 0.9583 | 0.0466 |
| Coffee | Ours | 26.07 | 0.0060 | 36.32 | 0.9845 | 0.0112 |
| Bathroom | AE | 64.02 | 0.0082 | 35.46 | 0.9704 | 0.0575 |
| Bathroom | AE-Ref | 68.54 | 0.0078 | 36.81 | 0.9805 | 0.0285 |
| Bathroom | FieldGI | 20.60 | 0.0090 | 34.72 | 0.9695 | 0.0577 |
| Bathroom | OIDN (`29 spp`) | 24.57 | 0.0076 | 37.50 | 0.9746 | 0.0353 |
| Bathroom | Ours | 24.55 | 0.0045 | 41.33 | 0.9889 | 0.0108 |

在这三个作者 scene、这套 processed GT 与单次表格汇总中，Ours 的四个报告质量指标均为最佳，同时其 frame time 被设定为接近 OIDN。这是 `author-positive`，不能外推为任意 scene/hardware 或与其他论文 FPS 的无条件排名。

### 9.3 其他实验

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| Roughness sweep | Hall car、多个 roughness，从 low 到 high angular frequency | Reference、Ours | 定性 | Fig.1 展示连续 frequency range；无数值 sweep。 | `P` Fig.1 |
| Cross-method visual | Hall/Coffee/Bathroom | AE-Ref、FieldGI、matched-time OIDN | 定性 crop | Ours 在 rough、glossy、near-mirror 的作者示例中保留更准确 outlines/details。 | `P` Fig.4、p.10 |
| Intermediate features | Hall car | principal、three filtered-secondary scales、fused、3 interpolation weights | 可视化 | 三尺度 secondary 由 sharp 到 blur；principal/secondary/zero weights spatially varying。 | `P` Fig.5 |
| Stage progression | Hall/Coffee examples | end of stage 1、stage 2、reference | 定性 | stage 1 保留低频 appearance，stage 2 补反射细节。 | `P` Fig.6 |
| Complex visibility | foliage、intricate geometry、high-resolution environment lighting | Reference、Ours | 定性 | 大部分 hard/soft shadow detail 可重建；thin geometry 的 extreme shadow 稍模糊。 | `S` §2、Fig.6 |
| Temporal behavior | dynamic scene videos | OIDN、Ours | 作者视频定性 | 正文称 OIDN 有明显 flicker、Ours 更稳定；没有 temporal metric、sequence manifest 或 history ablation。 | `P` §4.2；`A-video` |

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `ablation-inferior` | 用 MLP 替换 multi-resolution kernel-prediction CNN；输入/输出与 inference time 保持近似 | rubber-duck reflection 的边缘和下方 shadow 过锐 | 没有邻域 reflection observations，无法模拟 surface reflection 的 low-pass effect | — | `P` §4.3、Fig.7 |
| `ablation-inferior` | CNN 不读 principal/secondary features，只用 G-buffers 预测 kernel | glossy sphere 出现 noisy reflection，shading 不够平滑 | 缺失 self-tuned target feature clues | — | `P` §4.3、Fig.8 |
| `ablation-inferior` | plain principal/secondary linear interpolation，去掉 virtual zero | moving sphere occlusion 下 table reflection 出现 light leakage | 两个非零 feature 不能充分表达 abrupt occlusion/energy attenuation | — | `P` §4.3、Fig.9 |
| `ablation-inferior` | 直接从 scratch 用完整 dataset 训练，不做 staged optimization | rough car 上过锐 reflection artifact；glossy sphere transition 不自然且 noisy | optimization 陷入 sub-optimal local minima | 只有单个定性对照，无重复 seed，原因仍是作者解释而非已证明因果 | `P` §4.3、Fig.10 |
| `author-negative` | intricate geometry / rapidly varying orientation | thin teapot spout 的 specular highlight 丢失 | reflection G-buffers 在 screen space 太稀疏，邻域不可复用 | — | `P` Conclusion；`S` Fig.7 |
| `author-negative` | long specular path：ground mirror → car → environment | 无法复现 car 上再反射的 environment content | runtime 只查询 single-bounce reflection G-buffer | — | `P` Conclusion；`S` Fig.8 |
| `author-negative` | complex thin-geometry shadows | extreme case 中 shadow 略模糊 | 复杂 visibility 的细薄结构仍难重建 | — | `S` Fig.6 caption |
| `known-limitation` | training scalability | 每 scene 需要 `>40 GPU hours` on A6000-class GPU | 为保留 realistic detail 使用高容量 scene representation；未来考虑 adaptive training | — | `P` Conclusion |
| `known-limitation` | scene/material domain | opaque surfaces + rigid transforms only | 需要更一般的 neural parameterization/query 支持 deformable/translucent | — | `P` Conclusion |

AE、FieldGI、OIDN 的较差 crop 是 baseline outcome，不是本文作者对自身方法的失败尝试。没有证据表明作者试过多-bounce query、non-local feature search、deformable/translucent extension 或 compact grids 后失败；这些只被列为未来方向。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Feature-field backbone | “simplified FieldGI”、object-centric multi-resolution fields，Eq.(5)–(7) | 明确删掉 FieldGI deformation；给出 local transform、8-level triplane、hypernetwork/decoder | 无 implementation | P/S 对齐；删 deformation 是正式方法，不应复现回去。 |
| Hypernetwork/object decoder | main 只给抽象 `D_i` | `16+m→512`、5×512、`→10240`；object input 45→64，generated decoder `64→64→64→32` | 无 implementation | 可锁 architecture diagram，不能锁 material input `m`、bias、activation slope、update/cache policy。 |
| Final decoder | main 写 `G(g_f,H(...))` | 实际 input 只列 RGB albedo + roughness + 32-D fused；给出带 skip 的 256-wide MLP | 无 implementation | main 的 `g_f` 是抽象记法；不能把全部 first-hit G-buffer 都拼给 final decoder。 |
| Fusion CNN inputs | main 用 `g_f,g_r` 抽象表示 | 实际为 70 channels：roughness、view-normal cosine、reflection depth/emission、两类 feature | 无 implementation | supplemental 收窄了正式输入；position/normal/view/albedo 通过 field query间接作用。 |
| Multi-scale interpolation weights | main Eq.(8) 把 `gamma` 写作一张 full-resolution 3-D map | Fig.5 在三个 resolution 各输出 3 interpolation channels，再“upsample and merge”成 finest map | 无 implementation | 合并运算未给公式；是 `paper-code-gap`，不能猜 sum/concat/learned head。 |
| Kernel indexing | Eq.(9) 的输出写 `tilde_f_s^l(p)`，求和邻域却写 `N(u)`，分母又写 `N(p)` | 未更正 | 无 implementation | `p/u` 记号不一致；实际 center/index 与 boundary handling 未闭合。 |
| Reflection-ray convention | Eq.(3) 将 intersection 写为 `I_{S,L}(x,-omega_r)`，同时把 `omega_r` 称为 mirror reflection direction | 无补充 | 无 implementation | 公式负号必须保留；出/入射方向约定、origin offset 与 miss encoding 均未闭合。 |
| Final output activation | main 只称 final radiance/color，并说明 HDR radiance 训练使用 `log1p` | Fig.4 的图例与 final block 明确给出 LeakyReLU | 无 implementation | activation 类型按 supplemental 锁为 LeakyReLU；negative slope、head 输出是否为 mapped target、inverse mapping 与额外 clamp 仍不可复现。 |
| Data/GT | 8192 configs/scene、1024 spp、512²、offline denoiser | 无 manifest/附加配置 | 无 data/config | split、sampling ranges、denoiser identity/checkpoint、GT bias 都缺失。 |
| Training | Adam、L1+DSSIM、log1p、两 stage LR/batch、12h/4 A6000 | 只补 triplane progressive weighting | 无 trainer/config | step/epoch/schedule/seed/checkpoint 缺失。 |
| Runtime | RTX4090 PyTorch FP32 scene timings | 只给 shape | 无 renderer/runtime | measured scope、memory、MAC、ray-pass inclusion 未闭合。 |
| Failure cases | Conclusion 描述 sparse reflection、long specular path | Fig.7–8 视觉化并给路径解释 | 无 repro scene | P/S 对齐；只有静态图，没有 scene/config。 |

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. **Sparse screen-space reflection reuse**：复杂 geometry、快速 normal variation 与 thin features 破坏邻域复用，specular highlight/阴影会漏失或模糊。[P Conclusion；S Fig.6–7]
2. **Single-bounce specular query**：一条 mirror ray 无法为多次 specular chain 提供足够路径信息；作者明确计划扩展为多 bounce 并记录 throughput。[P Conclusion；S Fig.8]
3. **Per-scene training cost**：每个 scene 超过 40 A6000 GPU-hours，限制 scalability。[P Conclusion]
4. **Opaque + rigid only**：不支持 translucent materials 与 deformable objects。[P Conclusion]
5. **公开验证域仅 `512²`**：CNN architecture 与全部正式实验都使用 512² 三尺度邻域；不同 resolution、off-screen reflected content 与 disocclusion 未验证。这是 method-domain/evidence boundary，不是作者报告的失败结果，也不能改写成网络必然只能运行在 512²。[P §4.1；S Fig.5]
6. **No cross-scene generalization evidence**：所有三个主要 scene 都单独建立 dataset/训练；第一方材料没有 unseen-scene test。这是 evidence boundary，不是作者声称跨 scene 失败。[P §4.1]
7. **No dynamic-light protocol**：虽然问题定义含 light set `L`、Introduction 提到 dynamic objects/lights，object representation、dataset sampling 与实验没有披露 light controls。因此只能确认 fixed-scene illumination 下的 camera/object/material dynamics，不能确认任意 dynamic light。这是 evidence boundary，不是 `author-negative`。[P Eq.(1)、§4.1]

### 12.2 未报告/材料不可得

- official code、formal config、checkpoint、raw/denoised GT、scene assets、split manifest、supplementary video source frames；
- object material parameter vector `v_i/m` 的字段、range、normalization与是否含 texture；
- hypernetwork是跨 object shared 还是每 object 一套、object count、triplane bounds与outside sampling；
- convolution padding、skip merge、triplane interpolation、ray direction/origin/miss/default encoding、depth scaling、emission preprocessing；
- final RGB LeakyReLU 的 negative slope、nonnegative/HDR constraint、log1p 在 L1/DSSIM 中的确切位置与 inverse/exposure；
- optimizer betas/epsilon、LR schedule、stage steps/epochs、seed、initialization、checkpoint selection、repeat variance；
- validation/test split、evaluation frame/config count、metric definition/color space/aggregation、error bars；
- Table 1 是否包含 G-buffer/reflection ray generation、hypernetwork updates与 CPU/GPU synchronization；
- parameter/MAC/FLOP、per-scene/per-object bytes、precision of grids、feature fetch count、peak memory；
- temporal sequence protocol、history/reprojection（architecture 中没有）、flicker metric；
- dynamic light parameterization、cross-scene/novel-object test、materials beyond opaque surfaces。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

方法的容量主体不是 `64→64→64→32` object decoder，而是 **per-object dense spatial fields + full-frame fusion CNN + per-scene decoders**。[I；依据 S Fig.1–5]

supplemental 没有列出六个中间 plane resolution。若仅作容量情景计算，假设采用常见的逐级倍增 schedule `8,16,...,1024`，一个 object 的三张 plane、8 个 resolutions、每 level 4 channels 将包含：

```text
3 * 4 * (8^2 + 16^2 + ... + 1024^2) = 16,776,960 scalars/object.
```

若再按 Table 1 的 FP32 execution 作存储假设，仅这些 plane 就约 `67,107,840 bytes ≈ 64 MiB/object`，还不含 networks、optimizer、temporary full-frame features 与双查询。[I arithmetic；倍增 schedule 与 FP32 storage 都是假设，不是作者报告 bytes，不能作为该实现的精确 memory]

因此 “lightweight scene query”主要指只追一条 reflection ray，而不是整体 representation 很小。这个设计用高空间容量避免 MLP 独自承担 geometry-dependent detail，再用 screen-space CNN 把单镜面样本扩散成 rough/glossy response。

### 13.2 成功所依赖的假设

1. scene 在训练时已知，且能承受每 scene 约 48 GPU-hours量级的离线 fit；
2. objects 以 rigid transform 运动，因此 object-local grids 可重用；material variation 落在训练范围；
3. surface orientation 与 reflected content 在多数 screen-space neighborhood 中足够连续，使相邻 pixel 的 mirror samples 可以近似一个 rough lobe；
4. 一次 mirror intersection 已命中决定视觉反射的主要 content；长 specular chain 不是主导；
5. 512² full-frame U-Net 与 dense fields 的 memory/latency 可接受；
6. processed 1024-spp targets 的 denoiser bias 没有系统性抹掉要学习的高频结构。

作者的成功结果只在这些假设、三个自建 scene 和未公开 split 上成立。特别是“低/中/高频”不是一个 scene-independent spectral contract；band location会随 roughness、geometry projection、pixel footprint 与 screen resolution 改变。[I]

### 13.3 可迁移机制与不能迁移的部分

可迁移机制：

- **先把最尖锐结构转为可对齐 clue，再让网络学带宽/融合**：本文用 mirror hit 对齐 reflected content，和 local scattering 中用 half/difference chart 或 analytic microfacet core 对齐 moving peak 是同一类表示原则，但 query semantics 完全不同。[I]
- **由易到难的 frequency curriculum**：先用 roughness `>=0.25`、低分辨率 fields，再开放 sharp reflections/high-resolution levels，说明监督频率与 representation capacity 同步解锁值得做 matched test。[I]
- **让 target feature 参与自己的 filter prediction**：self-tuned kernel 比只读 G-buffer 的 variant好，提示 filtering 权重应读取将被过滤的信号，而不只读 footprint metadata。[I]
- **显式 attenuation/gating branch**：zero branch避免两类非零 features 被迫解释“没有贡献”；在有真实 mixture-of-experts/residual branches 的模型中，可以测试显式 null/gate，但必须用本问题的物理/统计约束重定义。[I]

不能直接迁移：

- screen-space U-Net、mirror ray、reflection depth/emission、dynamic occlusion 都依赖 scene visibility，不是材质 intrinsic input；
- 输出 `L(x,wo)` 已积分 incident lighting，不能作为 `f(wo,wi)` 或 matched sampler target；
- per-object dense grids与按 object 求和的 query cost不满足本项目 scattering program 的 fixed-fetch/random-access 目标；
- SSIM/DSSIM 是 image-space loss，不能未经验证移到 independent directional BRDF queries；
- virtual zero用于 scene occlusion attenuation，局部 BSDF evaluator不应借它学习 visibility。

### 13.4 与本项目 runtime contract 的关系

本方法作为整帧 scene renderer 在 **固定 resolution、固定 U-Net 和给定 object cap** 下可以做到有界，但论文没有给出 object cap，Eq.(7) 的求和成本随 object count 增长。它不是 [`prepare/evaluate/sample/pdf`](../../../../../docs/realtime_material_compilation.md) 的 scattering program：

- `prepare()`不能合法读取当前 frame 的 screen neighborhood 和 ray-scene hit 来改变 source material intrinsic state；
- `evaluate(wo,wi)`必须随机访问单个 direction pair，而本文只接收一个 `wo` 并输出已经积分后的 radiance；
- 没有 `wi` query、`sample()` 或 `pdf()`；
- full-frame CNN 与 dense per-object grids不满足静态固定读取数的 material hot path。

因此它适合作为 **scene-transport related evidence / capacity diagnostic**，不适合作为 material evaluator 候选。真正可吸收的是 frequency alignment、curriculum 和 branch gating 的设计思想，而不是其 data flow。[I/N]

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

当前 NVIDIA functional reproduction 的公开合同是：online source `prepare/evaluate` 直接监督线性 RGB `f`，formal config 使用独立 evaluator/sampler route、300k steps、前 20k steps directional mollification、log1p-L1 evaluator loss，并冻结 spatial/LOD filter recipe。[N `docs/research/experiment_framework.md` §2；N `configs/learning/nvidia-rta2024-materialx-formal.json`]

| 主题 | 分类 | 对应关系与结论 |
|---|---|---|
| Query/output semantics | `not-applicable` | Dual-Band 输出 scene radiance `L(x,wo)`；NVIDIA evaluator 输出 local `f(m,wo,wi)`。二者不能直接做 method correspondence。 |
| Smooth-to-sharp curriculum | `interface-adaptation` | Dual-Band 用 roughness subset + progressive grids；NVIDIA formal 已有 20k-step directional mollification。前者支持“先低频后高频”这一原则，但不能证明当前 20k/10°/256-sample 配置正确或更优。 |
| Loss | `intentional-deviation` | 两者都提到 log1p，但 Dual-Band 是 full-image `L1 + DSSIM`，NVIDIA 是 per-query `log1p-L1` 并另有 sampling route。不存在 faithful loss correspondence。 |
| Data lifecycle | `intentional-deviation` | Dual-Band 先离线生成并去噪 8192 images/scene；本项目 formal 只用 GPU-resident online reference queries，不持久化 batch。本文不能作为恢复 offline corpus 的依据。 |
| Representation | `budget-adaptation` | Dual-Band dense per-object triplanes + U-Net 是 scene capacity point；即使表达力高，也不能在本项目 material budget下沿用同一 identity。 |
| Frequency alignment | `interface-adaptation` | mirror-hit clue 对 scene reflection做 alignment；本项目对应机制应是 half/difference chart、analytic core或lobe token，而不是 scene ray query。[N `docs/research/model_candidates.md` §1.2、§3、§5] |
| Filtering | `interface-adaptation` | Dual-Band 自调 screen-space filter读取 target features；NVIDIA formal 的 MaterialX filter是 footprint/normal/latent recipe。可提出 matched ablation，但不能声称已复现或已有 defect。 |
| Runtime evidence | `author-underspecified` | 论文未给 bytes/MAC/fetch 与完整 timing scope，无法用 22–26 ms 为 NVIDIA scattering query 设置 hard gate或速度预期。 |

这篇论文没有提供证据指向当前 NVIDIA implementation 的 `suspected-defect`。它主要提出两个可证伪迁移方向：frequency curriculum是否应从固定 directional mollification扩展为 source-aware band curriculum；filter predictor是否应读取被过滤的 latent/response feature。[I]

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| `H-DB1`：source-aware roughness/band curriculum 比仅固定角度 mollification 更能保住 narrow peaks | Dual-Band staged model 的定性结果优于 direct-full-data variant。[P Fig.10] | local scattering 的 roughness/peak bandwidth 可由 source metadata或reference probe可靠分层；不需要 scene G-buffer | 在同一 NVIDIA evaluator形态上比较 current 20k directional mollification 与 matched-work curriculum：先采宽 lobe/source-easy states，再恢复完整分布；不改总 queries/steps | source split、online query stream、seed、optimizer、schedule、model/MAC、loss、20k curriculum budget | directional normalized L1、peak position/height、top-energy recall、energy error、bootstrap CI | 同一 `prepare/evaluate`，无 runtime change | peak/long-tail 无显著改善，或 median/energy显著恶化；则不采用 source-aware curriculum |
| `H-DB2`：filter权重读取当前 latent/response feature，比只读 footprint/normal metadata 更能保持 spatial-directional correlation | 去掉 principal/secondary self-tune inputs 后 glossy reflection 出现 noise。[P Fig.8] | MaterialX/texture footprint filter中也存在“metadata相同但 feature content要求不同 filter”的情况 | 在相同固定 fetch cap 下比较当前 filter与 feature-conditioned fixed-radius weights；不能引入 full-screen CNN | same source assets、mip/footprints、fetch count、decoder、training queries/steps、seed | spatial/LOD error、directional error、temporal sweep continuity、bytes/MAC/fetch | `prepare` 内固定邻域/固定读取数 | matched fetch/cost下无显著 quality改善，或 feature-conditioned weights在 unseen assets/LOD不稳定 |
| `H-DB3`：把 analytic high-frequency clue 与 neural low-frequency/residual branch显式门控，可比单一 direct head 更稳地覆盖多带宽材质 | Dual-Band principal + aligned secondary + learned interpolation覆盖 rough-to-mirror；MLP fusion/AE-Ref 过锐。[P Fig.4、Fig.7] | local material中 analytic microfacet core/half-warp feature可扮演“aligned high-frequency clue”，且不需要 scene visibility | 在 M1 direct 与 M2 analytic-core+neural residual之间加入同预算、非负/能量受控 gate；与无 gate concat及 core-only配对 | source/query split、training budget、total MAC/bytes、core、chart、loss、sampler状态 | directional/energy/peak metrics、reciprocity、state-stratified CI、Slang cost | 静态有界 local evaluator | gate权重退化为常数/单支、matched指标不改善，或能量/reciprocity/cost劣化 |

这些假设都不把 Table 1 的 image-space PSNR/SSIM 或 RTX4090 frame time作为本项目 hard gate。每项都必须在本项目的 local output semantics 与部署成本下重新验证。

## 16. 证据索引

### `P` main paper

- `[P p.1, Fig.1]`：正式题名/作者/DOI；dual-band 是相对 view direction 的两个 feature frequency bands。
- `[P §1, p.2]`：问题、现有 neural GI 的 angular-high-frequency困难、贡献。
- `[P §3.1, Eq.(1)–(4), Fig.2]`：scene radiance目标、first/reflection queries、pipeline 与输出。
- `[P §3.2, Eq.(5)–(7)]`：object-centric feature fields、`r_i=(M_i,v_i)`、object-wise sum。
- `[P §3.3, Eq.(8)–(10), Fig.3]`：self-tuned multi-resolution kernel prediction与softmax aggregation。
- `[P §3.4, Eq.(11)]`：zero-enhanced linear interpolation。
- `[P §3.5]`：roughness `>=0.25` stage、progressive fields、stage LR `1e-3/1e-4`。
- `[P §4.1, Eq.(12)]`：三个 scenes、8192 configs、1024 spp、512²、offline denoiser、Adam、loss、batch、4×A6000/12h。
- `[P Table 1、§4.2、Fig.4]`：baseline protocol、RTX4090/PyTorch FP32 timing、完整指标、OIDN matched-time spp与定性比较。
- `[P §4.3、Fig.7–10]`：四项 ablation-inferior 与作者解释。
- `[P Conclusion, p.8]`：sparse reflection、long specular path、>40 GPU-hours、opaque/rigid限制。

### `S` supplemental

- `[S Fig.1、§1, p.1]`：object field query、hypernetwork parameter update、删除 deformation component。
- `[S Fig.2, p.2]`：hypernetwork input/layers/output与activation图例。
- `[S Fig.3, p.2]`：13-D G-buffer、32-D triplane、45→64 alignment、object decoder。
- `[S Fig.4, p.2]`：36-D final decoder input、256-wide skip architecture、RGB output。
- `[S Eq.(1)–(2), p.2–3]`：8-level 3-plane feature fields、4 channels/level、8→1024 resolution、progressive weights。
- `[S Fig.5, p.3]`：70-channel U-Net、三 resolution 的29-channel heads、`5×5` kernels。
- `[S §2、Fig.6, p.3]`：foliage complex visibility与thin-shadow模糊。
- `[S Fig.7–8, p.4]`：intricate geometry与long specular path失败。

### `C/A/N/I`

- `[C project tree commit 05264274eaa54f9641a191dbe53c6b5d2d8051fe]`：只有项目页/PDF assets，无 code/config/data。
- `[A project page, accessed 2026-08-29]`：作者摘要与 main/supp/video 入口。
- `[A-video id 32rLtfqKauY]`：作者视频入口；只作动态定性来源。
- `[N docs/realtime_material_compilation.md]`：本项目 static-bounded `prepare/evaluate/sample/pdf` 合同。
- `[N docs/research/experiment_framework.md §2]`：NVIDIA functional reproduction recipe。
- `[N configs/learning/nvidia-rta2024-materialx-formal.json]`：300k steps、20k mollification、online routes、filter/loss formal identity。
- `[N docs/research/model_candidates.md §1.2、§3、§5]`：half/difference chart、analytic core与 warped field 候选。
- `[I §13–15]`：capacity算术、适用假设、项目 correspondence 与迁移假设；均不改写为作者结论。

## Evidence review

```text
author_worker: /root/dualband2025
reviewer: /root/dualband2025_review
reviewed_at: 2026-08-29
sources_rechecked:
  - main paper 11 pages, SHA-256 F66E9CEF82BE5E4E45D98B8339EB6E0EF359BAD3C5F76A424777FF486D5B60A3
  - supplemental 4 pages, SHA-256 EE50C9612864BE33F75216C55AD18789511F0090FFACC1539311E0DB539FD0FB
  - author project page and public gh-pages tree at 05264274eaa54f9641a191dbe53c6b5d2d8051fe
findings_closed:
  - removed the unsupported attribution of Table 1's parenthetical 1-2 ms to path tracing; the component scope is ambiguous
  - resolved the supplemental final RGB head as LeakyReLU while preserving its missing slope and mapped-domain lifecycle
  - preserved Eq.(3) x_r=I(x,-omega_r) and recorded the unresolved direction/origin/miss convention
  - preserved Eq.(9)'s exact p/u mismatch instead of silently normalizing the indices
  - changed the unstated 8-to-1024 doubling schedule from paper fact to an explicit capacity-analysis assumption
  - verified the three-scale gamma merge remains genuinely underspecified
  - verified split, stage length, temporal protocol, dynamic-light protocol and full timing scope are unreported
remaining_evidence_gaps:
  - official training/runtime code, formal configs, checkpoints and scene/data manifests are unavailable
  - stage length, split, metric aggregation, temporal protocol and complete runtime scope are unreported
  - six intermediate triplane resolutions, gamma multi-scale merge, Eq.(9) p/u indexing, LeakyReLU slope/mapped-domain lifecycle and ray sign/origin/miss encoding cannot be resolved without code
review_status: evidence-reviewed
```

### 完成检查

- [x] main paper 已完整阅读，关键公式/图/表/图注/脚注已视觉核对；
- [x] supplemental/appendix/勘误的可用性已检查；
- [x] official code/config/data 的可用性与项目页 commit 已检查；
- [x] architecture、training、runtime 和主要结果均有 locator；
- [x] 失败尝试与较差消融正确分类；
- [x] paper/code gap 和“未报告”保留；
- [x] `I` 分析晚于事实层，没有改写作者结论；
- [x] NVIDIA 影响引用真实 `N` 证据；
- [x] 假设包含 matched control、部署类别和证伪条件；
- [x] 独立 evidence review 已完成。
