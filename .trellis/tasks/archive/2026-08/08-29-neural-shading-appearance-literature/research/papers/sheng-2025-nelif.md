---
paper_id: "sheng-2025-nelif"
title: "NeLiF: Neural Lighting Function Generation for Real-Time Indoor Rendering"
authors: "Hongtao Sheng; Yuchi Huo; Chuankun Zheng; Guangzhi Han; Yifan Peng; Shi Li; Bin Zang; Hao Zhu; Rui Tang; Yiming Wu; Rui Wang; Hujun Bao"
year: "2025"
venue: "Proceedings of the SIGGRAPH Asia 2025 Conference Papers"
doi: "10.1145/3757377.3763958"
report_status: "evidence-reviewed"
main_source: "local:3757377.3763958.pdf; https://doi.org/10.1145/3757377.3763958"
supplemental_status: "unavailable"
official_code_status: "unavailable"
official_code_commit: "not-applicable"
author_worker: "/root/nelif_full_report"
reviewer: "/root/nelt_full_report"
last_verified: "2026-08-29"
---

# NeLiF: Neural Lighting Function Generation for Real-Time Indoor Rendering

## 1. 研究对象与报告边界

NeLiF 是一个面向室内场景的**场景级整帧 global-illumination inference pipeline**。它从灯具的有限多视图观测生成与场景无关的 3D neural lighting field，再用当前场景的 G-buffers、shadow clues 与 reflective shadow map（RSM）中的 indirect virtual point lights（VPLs）查询该 field，分别预测 direct shading、shadow 与 indirect shading，最后与灯具自身的 3D Gaussian Splatting（3DGS）外观合成最终图像。[P Abstract, §3–5, Figs. 2–4, Eqs. 1–11]

论文把此前方法的主要障碍分开叙述：CNSR 从 sparse observations、AE 从 predefined variable parameters 得到 scene representation，再与 G-buffers 解码整帧 illumination；作者认为用单一 latent 承载全场景照明会形成容量压力。NeLT 与 Superposed Deformable Feature Fields 通过 object-centric/deformable fields 改善动态适应，但在 §6.1 被列为需 per-scene training 的 excluded baselines。LightFormer 则被正文明确描述为可面向 unseen dynamic scenes 的 light-centric 方法，并进入 Table 1 正式对照；NeLiF 对它的具体批评是每帧、逐 pixel 聚合 light effects 有冗余，而不是把它归入 §6.1 的 excluded list。NeLiF 的核心目标是从灯具观测**生成**可在未见场景中使用的 lighting representation，并把这次聚合缓存进 field。[P §1, §4.2, §6.1]

这里的“lighting function”不是局部材质的 `evaluate(wo,wi)`，也不输出 `sample()/pdf()`。它已经把灯具、visibility、direct/indirect transport、屏幕空间几何和阴影近似耦合到一个 image-space query 中，因此不能被当作 NeuralShading 当前 local neural material evaluator 的替代实现或质量对照。[P Eq. 1–2, §3–5][N `docs/realtime_material_compilation.md`][I]

本报告覆盖用户放在项目根目录的正式 11 页 proceedings PDF。论文多次把逐层 architecture、data generation、training、inference 与额外消融指向 supplemental，但当前项目目录、PDF attachment、作者 publication entry 与公开检索均未找到可读 supplemental；也未找到 official code/config/data。因此报告只恢复正文明确披露的 tensor、公式与实验，不用 LightFormer、NeLT 或常见 Transformer/U-Net 配置补全缺口。[P §4.1, §5.1, §6, §6.2–6.3][A WeLight publications][C unavailable]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | 项目根目录 `3757377.3763958.pdf`；正式 DOI [`10.1145/3757377.3763958`](https://doi.org/10.1145/3757377.3763958)；11 pages | 2026-08-29 | SHA-256 `07558A3B7D7CA47091337F3A5A41E4D57B0BB02F3EB5A0387D3DBF0F47D398DD`；53,842,675 bytes | 本报告的方法、公式、实验和限制主证据。11 页全部提取并按 150 dpi rasterize；逐页视觉核对 Fig. 1–12、Eq. 1–11、Table 1、图注、脚注与参考文献。PDF 无加密、无 embedded files。 |
| Supplemental `S` | 正文 §4.1、§5.1、§6、§6.2–6.3 多次指向 supplementary document/materials；项目根目录没有相应文件，主 PDF 无 attachment，作者 publication entry 与公开检索未给 supplemental locator | 2026-08-29 | unavailable | supplement 的存在由正文明确支持，但内容当前不可得。网络细节、数据生成、训练与 inference 细节以及额外 spherical-triplane/shadow 分析不能从正文反推。 |
| Official code/config/data `C` | [WeLight publications](https://hku.welight.fun/publications/) 仅链接 SIGGRAPH Asia presentation；以完整标题和 `NeLiF` 检索公开 GitHub/作者入口未发现 official repository | 2026-08-29 | unavailable；`official_code_commit=not-applicable` | 无法审计 exact topology、loss、formal config、TensorRT export、dataset manifest、checkpoint 或 benchmark implementation。没有执行 Git 网络探测、SSH、token 或登录。 |
| Author page/program `A` | [WeLight publication entry](https://hku.welight.fun/publications/)；[SIGGRAPH Asia 2025 official presentation](https://sa2025.conference-schedule.org/presentation/?id=papers_1990&sess=sess106) | 2026-08-29 | not-applicable | 只用于确认 publication identity 和作者摘要；正文已经可得后，不用摘要覆盖正文。未找到独立 project page、talk、slides 或 correction。 |
| NeuralShading evidence `N` | `docs/realtime_material_compilation.md`；`docs/research/experiment_framework.md`；本任务中的 LightFormer、CNSR、AE、Neural Light Probes 与 volumetric-inference 报告 | 2026-08-29 | 当前 workspace | 只用于 §13–15 的 query-semantics、部署类别和 matched-control 分析；不回填为 NeLiF 事实。 |

书目信息与 PDF 首页一致：Hongtao Sheng 与 Yuchi Huo 为共同一作，Hujun Bao 为通讯作者。[P p.1 footnotes] WeLight BibTeX 把 Bin Zang 排在 Yifan Peng、Shi Li 之前，而 main PDF 与正式 ACM reference 把 Bin Zang 排在两者之后；front matter 采用 main PDF 顺序，不从作者顺序差异推断贡献。[P p.1–2][A WeLight publications]

本地 PDF 的 metadata creation/modification date 为 2025-09-26，正文 proceedings 日期为 2025-12-15 至 18 日；这只是文件生成时间与正式会议日期的差异，不表示存在另一个方法版本。[P p.1–2]

## 3. 原论文要解决的问题、假设与贡献边界

作者把当前 camera ray 的最终结果写成：

\[
L(o,\omega)=\mathcal F(o,\omega;\mathcal S,\mathcal L)+\mathcal V(o,\omega),
\]

其中 `o,ω` 是 camera origin 与 view direction，`S`、`L` 分别是 scene objects 与 light sources，`V` 是灯具本身的可见外观，`F` 是灯具作用到环境后的 lighting effects。[P §3.1, Eq. 1] Eq. 1 后紧接的 prose 把“appearance”写成了 `L`，与公式中已经表示 light sources 的 `L` 冲突；本报告以公式与随后 3DGS 分支一致的 `V` 为准，并把该处保留为 `paper-internal notation gap`。[P p.4, Eq. 1 surrounding prose]

NeLiF 进一步把 `F` 分成 direct shading、shadow 与 indirect shading：

\[
\mathcal F(o,\omega;\mathcal S,\mathcal L)
=\mathcal D_d(E_S,E_L)\odot
\mathcal D_s(E_S,E_L,E_s)+
\mathcal D_i(E_S,E_L),
\]

`D_d,D_s,D_i` 分别是 direct、shadow、indirect decoder，`E_S` 是 scene encoding/G-buffer 信息，`E_L` 是 neural lighting field，`E_s` 是 shadow-clue encoding，`⊙` 为逐元素乘法。[P §3.1, Eq. 2]

作者的三项贡献是：[P p.2–3 contributions]

1. 从灯具观测生成 scene-independent Neural Lighting Function，并据此在未见室内场景中估计 lighting effects；
2. 用 advanced generative model 加 training-free Inverse HDR Splatting，把有限灯具图像变成带 HDR radiance 的 3DGS 外观，降低 view/luminaire update 时重新渲染的开销；
3. 用 multi-scale kernel prediction 处理 shadow 的空间不连续，并联合 lighting feature 预测随灯具形状变化的 soft/hard shadow。

方法的目标域是 complex indoor luminaires 与作者合成/重建的室内场景；论文没有声明支持室外太阳尺度、任意 spectrum、任意 participating media、任意透明/焦散 transport 或 local scattering query。[P §1, §3.2, §7]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Luminaire observations | 灯具近距离多视图 emitted-radiance image 与 depth；作者把它们解释为同时含空间与方向变化的 4D light field | 编码后 `W1×H1×W2×H2×C1`，`W1,H1` 是 angular resolution，`W2,H2` 是每视图 spatial resolution；原始 view 数、图像分辨率、depth/radiance channels 未报告 | [P §3.2, §4–4.1, Eq. 3] |
| Intensity normalization | 全部 views 的 radiance 用 global max RGB 归一化；输出 shading 乘回同一 intensity factor | scalar 或 RGB 的 exact 定义未报告；Fig. 2 画成一条 intensity factor | [P §4, Fig. 2–3] |
| Lighting-field generation query | 可学习 position field 同时编码 scene 中的 position 与 direction，作为 cross-attention Query；light tokens 为 Key/Value | position-field resolution、channel、direction parameterization 与 query 数未报告 | [P §4.2, Eq. 6, Fig. 2] |
| Runtime spatial query | 当前 G-buffer position `p=(x,y,z)` 转 spherical `(θ,φ,r)` 后查询 spherical triplane | `θ,φ` 经 equal-area projection 归一到 `[0,1]`；`r` 用 maximum lighting-influence radius `8` 归一化，单位和超域处理未报告 | [P §4.2, p.6] |
| G-buffers | direct/indirect decoder 使用当前 shading point 的 G-buffers；Figure 3 明画 Position 与 Albedo，正文只概称 surface-oriented geometric attributes | normal、roughness、view direction 等 exact channel schema/shape 未报告 | [P §3.2, §5.1, Fig. 3] |
| Indirect cues | RSM 中每个 texel 作为由 light source 发出的 indirect VPL；indirect decoder 合并 RSM 与 multi-bounce radiance estimation | RSM resolution、VPL 数、flux/channel 与 multi-bounce estimator 未报告 | [P §3.2, §5.1] |
| Shadow clues | `S={z-z_f, z/z_f, d, c_c}`：`z_f` 为到 emitter 的距离，`z` 为 emitter 到 occlusion point 的距离，`d` 为 camera-view depth，`c_c=n·ω_view` | 4 channels/pixel；没有显式 light direction、normal 或 penumbra width channel | [P §5.2, Eq. 7] |
| Hard-shadow hierarchy | 从 dynamic shadow mapping 取得 high-resolution hard shadow，再生成五级 multi-scale maps | depth precision `1×10^-3`；逐级 `2×2` downsample；base resolution 未报告 | [P §5.2] |
| Output components | direct shading、direct shadow、indirect shading；另有 luminaire 3DGS appearance `V` | RGB-like images；linear/HDR unit、颜色空间、exposure、tone mapping 未报告 | [P Eqs. 1–2, Fig. 3] |
| Final output | 按 Eq. 1–2 与 Figure 3 合成 final shading image | Table 1 按 `512×512` 测量 | [P Fig. 3, §6, Table 1 caption] |
| Domain restriction | unseen dynamic indoor scenes 与 novel luminaires；训练/测试的“unseen”split 具体规则未报告 | 非任意 local `wo,wi`，非可独立组合的 transport operator | [P Abstract, §1, §6.2–7] |

Equation 2 的形式只写 `direct×shadow+indirect`。按 Figure 3 的实际连线，direct shading 先与 shadow 相乘，所得结果再进入一个同时接收 Albedo 与 intensity factor 的乘法节点；该上支路随后与 3DGS 相加。indirect shading 在另一支路单独乘同一个 intensity factor，最后再与上支路相加。这只是图示 dataflow，不是论文给出的第三条 radiometric equation。正文没有解释 direct decoder 输出是 irradiance、已乘 BRDF 的 radiance，还是某种 learned shading，也没有定义 intensity factor 是 scalar 还是 RGB、3DGS 输出与 shading 的单位是否一致。因此不能把这些 image components 重命名为物理上严格可交换的 radiance/visibility/BRDF factors。[P Eq. 1–2, Fig. 3]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

NeLiF 的数据流分为三条相互关联、但不能混为一个 end-to-end MLP 的路径：[P §3.2–5.3, Figs. 2–4]

1. **Illumination Encoding**：采集灯具的多视图 radiance/depth；每个 observation 切 patch，经 MLP/PatchEmbed 得到离散 latent；Transformer 在 4D feature map 的 spatial 与 angular 轴交替 attention，输出 light tokens。
2. **Lighting Field Generation**：可学习 position field 提供 cross-attention Query，light tokens 提供 Key/Value，generator 产生 scene-independent spherical triplane。它把对 light tokens 的聚合缓存到 field，而不是像 LightFormer 那样每帧、每 pixel 重新跨 lights 聚合。[P §4.2]
3. **Shade Decoding**：G-buffer position 查询 triplane；direct decoder 合并 lighting feature 与 G-buffer，indirect decoder再合并 RSM/VPL；shadow decoder用 lighting feature 与五级 hard-shadow clues预测多尺度 filter/upsample kernels。Figure 3 的图示顺序是 `direct shading × shadow`，再与 Albedo、intensity factor 进入乘法节点并加上 3DGS；`indirect shading × intensity factor` 作为另一支路，最后与前者相加。该 wiring 的 radiometric quantity 未由正文定义。

Lighting Encoder 的正式 tensor 关系是：[P §4.1, Eqs. 3–5]

\[
\mathcal L_{input}\in
\mathbb R^{W_1\times H_1\times W_2\times H_2\times C_1},
\]

\[
F^0=\mathrm{PatchEmbed}(\mathcal L_{input})
\in\mathbb R^{W_1\times H_1\times(W_2/16)\times(H_2/16)\times C_2},
\]

\[
F^{l+1}=
\begin{cases}
\mathrm{SpatialAttention}(F^l),&l\bmod 2=1,\\
\mathrm{AngularAttention}(F^l),&l\bmod 2=0.
\end{cases}
\]

正文没有披露 `W1,H1,W2,H2,C1,C2` 的数值，也没有给 Transformer block 数、heads、QKV width、FFN、normalization、residual、activation 或 dropout。[P §4.1]

Lighting Field Generator 的 cross-attention 是：[P §4.2, Eq. 6]

\[
T_{output}=\mathrm{CrossAttention}
\bigl(Q(T_{position}),K(f_n^{out}),V(f_n^{out})\bigr),
\]

其中 `T_position` 来自 learnable position field，`f_n^out` 是 light tokens，`Q,K,V` 为各输入上的 linear layer。`T_output` 如何 reshape/project 成三张 plane、三 plane 的融合方式与 feature channel 均未报告。[P §4.2]

### 5.2 持久化表示与状态边界

- **shared learned weights**：Lighting Encoder、Lighting Field Generator、三个 shading decoder 与 shadow kernel-prediction network；论文没有报告是否所有数据只训练一个 checkpoint，但其 generalization claim 与“our model”表述指向跨数据集共享模型。[P Abstract, §6]
- **per-luminaire generated state**：light tokens、spherical neural lighting field、global intensity factor；生成/更新耗时、bytes、cache lifetime 未报告。[P §4, Fig. 2]
- **per-luminaire appearance state**：Trellis 生成的 LDR 3DGS，再由 Inverse HDR Splatting 写回 HDR scale/radiance；Gaussian count、SH degree、bytes、precision 未报告。[P §5.3, Eq. 11]
- **per-frame scene state**：G-buffers、RSM/indirect VPLs、hard-shadow hierarchy 与 decoder feature maps。[P §3.2, §5]
- **没有 material-local runtime state contract**：field 描述灯具对周围空间的作用，不是材质资产上的 latent grid；没有 material mip/LOD 或 `evaluate/sample/pdf` 状态。[P §4.2][I]

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Observation Patch MLP / PatchEmbed | 多视图 radiance/depth patch | Figure 2 称 MLP；Eq. 4 把空间轴各缩小 16 倍 | 未报告 | `W1×H1×W2/16×H2/16×C2` | shared | [P Fig. 2, §4.1, Eq. 4] |
| Lighting Encoder | 4D patch features | 2D Transformer blocks，spatial/angular attention 交替 | block 数、heads、FFN、norm、activation 未报告 | light tokens `f_n^out`；shape 未报告 | shared；per-luminaire tokens | [P §4.1, Eqs. 3–5] |
| Learnable Position Field | scene-wide positional/directional coordinates | learnable field + linear `Q` | field parameterization/resolution 未报告 | query tokens `T_position` | learned/shared；是否固定 world domain 未报告 | [P §4.2, Eq. 6] |
| Lighting Field Generator | position queries + light-token K/V | cross-attention + 未披露的 field projection | heads、width、layers、activation 未报告 | spherical light triplane | shared generator；per-luminaire field | [P §4.2, Fig. 2, Eq. 6] |
| Spherical triplane sampler | G-buffer world position | Cartesian→spherical；equal-area angular projection；三 plane projection/sampling | interpolation/fusion未报告 | per-pixel lighting feature | per-luminaire state | [P §4.2, p.6, Fig. 3] |
| Direct decoder | sampled lighting feature + G-buffers | neural decoder；exact layers未报告 | 未报告 | direct shading | shared | [P §5.1, Fig. 3] |
| Indirect decoder | lighting feature + indirect VPLs/RSM + G-buffers | 采用 LightFormer indirect module；exact adapted structure 在 supplemental | 未报告 | indirect shading | shared | [P §5.1, Fig. 3] |
| Shadow U-Net backbone | lighting feature + 4-channel shadow clues | U-Net-like multi-scale encoder；每层输出 `5×5` filter kernel；五级 pyramid | feature widths、layer count、norm/activation未报告 | per-level filtered shadow/features | shared | [P §5.2, Fig. 4, Eq. 8] |
| Filter & upsample block | level `l` hard/soft shadow + predicted kernels | `5×5` spatial filter；`4×4` learned upsampling；softmax scale blend | kernel normalization除scale blend外未报告 | high-resolution soft shadow | shared kernels；per-frame maps | [P §5.2, Fig. 4, Eqs. 8–10] |
| Energy correction | finest-level shadow feature | 预测 scalar/map `η` 并乘到 blended shadow | exact topology与公式未报告 | corrected final shadow | shared predictor | [P §5.2, Fig. 4] |

正文明确把网络细节移到 supplemental；因此不能从 LightFormer 的 released report 或常见 U-Net 结构替 NeLiF 补具体 channels。Figure 12 的 triplane resolution `32/128` 只属于可视化消融，不证明 formal runtime field 就使用其中某个 resolution。[P §4.1, §5.1, Fig. 12]

### 5.4 Shadow kernel、spherical triplane 与 HDR splatting 细节

Shadow filter 在 level `l` 的计算为：[P §5.2, Eq. 8]

\[
\hat S^l_{xy}=\sum_{uv} S^l_{uv}\,w^l_{uvxy}.
\]

各级再用预测的 `4×4` kernel `α` 上采样，并以 softmax 权重 `λ` 融合：[P §5.2, Eqs. 9–10]

\[
\tilde S^l_{xy}=\hat S^l_{xy}
+\lambda^l_{xy}\sum_{uv}4\tilde S^{l+1}_{uv}\alpha^{l+1}_{uvxy},
\qquad
\lambda^l_{xy}=\frac{\exp(z^l_{xy})}
{\sum_{k=1}^{L}\exp(z^k_{xy})}.
\]

Eq. 9 在 600 dpi raster 与 PDF text layer 中都明确印为 `4\tilde S`，即 `\tilde S^{l+1}` 前有一个字面标量 `4`；Figure 4 的上采样节点另用 `U` 标记，因此不能把这个 `4` 擅自转写成 upsample operator。论文没有解释该因子的来源、是否用于补偿 `2×2` 尺度变化，或它与 `4×4` kernel size 的关系。本报告保留印刷公式，不补其物理/数值含义。[P p.7, Eq. 9, Fig. 4]

论文强调 softmax 使跨 level blending weights 和为一；但没有说明 kernel `w/α` 自身是否归一化。最终 energy correction `η` 只在 Figure 4 与 prose 出现，没有进入 Eq. 9–10，exact application 与 `η` 的 range 均保留为正文内部缺口。[P §5.2, Fig. 4]

Inverse HDR Splatting 先用“existing models”在 LDR/HDR image 间转换。每个 HDR pixel 与其 LDR counterpart 的比值得到 scale `s_i`；一个 Gaussian 从多 pixels/views 收到多个估计，最终 scale 为：[P §5.3, Eq. 11]

\[
\gamma=\frac{\sum_i w_i s_i}{\sum_i w_i},
\]

`w_i` 等于该 Gaussian primitive 对第 `i` 个 pixel 的 rendering contribution。作者称这条写回无需额外训练或 per-object optimization；正文没有披露 LDR↔HDR model 身份、ratio epsilon/clamp、RGB scale 是按通道还是标量、occlusion/visibility filtering、Gaussian overlap 与 outlier handling。[P §5.3]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset scale | 5,300 luminaires、10,000 indoor scenes、总计 1,000,000 rendered training images；Fig. 1 写“over five thousand complex luminaires” | [P Fig. 1, §6] |
| Scene/luminaire source | 作者称 modern indoor scenes 与 complex luminaires；资产来源、license、scene construction、material/light parameter ranges 未报告 | [P Abstract, Fig. 1, §6] |
| Luminaire observations | synthetic luminaire 可由 rendering engine 生成多视图；real-world luminaire 可由 HDR camera；本文具体实验用 3DGS 渲染多视图，并可由 Trellis 从 multi-view 或 single view 生成 luminaire 3DGS | [P §5.3] |
| Single-image test | 使用 Internet 获得的灯具图像，先生成 3DGS 与 neural lighting function；没有 GT，只做 visual claim | [P §6.2, Fig. 10] |
| GT/reference renderer | 正文没有命名 renderer、integrator、reference spp、bounce depth、tone mapping 或 denoising；Fig. 5 只说 traditional path tracing `8000 spp` 仍因复杂灯具采样出现与 reference 的可见差异 | [P Fig. 5 caption, §6] |
| Train/validation/test split | 作者称在 entirely unseen test sets 同时测试 novel scenes 与 novel light sources；具体数量、asset identity、是否 joint-disjoint、validation split 与 seed 未报告 | [P §6.1–6.2] |
| Fairness subset | 为比较 LightFormer，双方训练 35 epochs，使用 `400K dataset`，包含 all training fixtures 与 `over four thousand scenes` | [P §6.1] |
| Full-vs-subset relation | 正文没有明确 Table 1 的 `Ours` 是否只来自 400K matched subset，或完整 1M model 是否用于其他图；两者不能静默合并成一个 formal training identity | [P §6–6.1] |
| Spatial/directional/light/time sampling | camera、scene、luminaire placement/view、multi-view angular grid、pixel crop、RSM VPL 与 dynamic-state sampling 均未报告 | [P §4–6; S unavailable] |
| Shadow-map recipe | high-res hard shadow depth precision `1e-3`，五级、每级 `2×2` downsample；resolution、shadow-map algorithm/light proxy 未报告 | [P §5.2] |
| Filtering/LOD/footprint | shadow pyramid是屏幕空间 visibility filtering，不是材质 footprint/LOD；lighting field没有公开 mip/scale supervision | [P §5.2][I] |
| Online/offline generation | 训练 corpus 预渲染；runtime 需要当前 G-buffer/RSM/shadow clues，并查询预生成 lighting field；field generation 与 3DGS generation 可视为 luminaire prepare，但缓存/更新规则未报告 | [P §3–6] |

“millions of unseen dynamic indoor scenes”只出现在 Figure 1 的作者概括，没有对应 test-scene enumeration、采样分布或统计置信区间；不能把它解释成逐个验证了数百万独立场景。[P Fig. 1][I]

## 7. Loss、optimizer 与训练 lifecycle

| 项 | 正式配置 | locator |
|---|---|---|
| Target/output transform | radiance input除以跨 views 的 global max RGB，shading output乘回同一 intensity factor；loss-space transform、clamp、tone mapping、negative handling未报告 | [P §4] |
| Loss terms/weights | 未报告；正文没有给 direct/indirect/shadow 的监督项、perceptual loss、HDR loss、kernel regularization 或 field loss | [P §4–6; S unavailable] |
| Optimizer/hyperparameters | 未报告 | [P §6; S unavailable] |
| LR schedule | 未报告 | [P §6; S unavailable] |
| Batch/query count | 未报告 image batch、crop、views/luminaire、pixels/query 或 distributed batch | [P §6; S unavailable] |
| Steps/epochs/stages | 与 LightFormer matched comparison 为 35 epochs；完整 model 是 joint/end-to-end 还是先 encoder/generator 再 decoders，正文未报告 | [P §6.1; S unavailable] |
| Initialization/seed/model selection | 未报告 | [P §6; S unavailable] |
| Hardware | training 使用 12× NVIDIA RTX 4090D；inference 测试使用 1× RTX 4090D | [P §6] |
| Training time | 未报告 wall-clock、GPU-hours、dataset rendering time 与 3DGS/Trellis generation time | [P §6; S unavailable] |
| Inverse HDR Splatting | training-free，不需 per-object optimization；它不是主 network 的 loss/lifecycle 说明 | [P §5.3] |

因此，35 epochs 不能单独恢复训练预算：没有 batch、images/epoch、optimizer、LR、distributed strategy 或 checkpoint selection。也不能从 12 张 GPU 推断 per-GPU batch、并行模式或实际训练时间。[P §6]

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path | luminaire field 已生成后：当前 scene raster/RSM 产生 G-buffers、indirect VPLs、hard-shadow clues；position 查询 spherical triplane；direct/indirect/shadow 三 decoder；最后与 Albedo、intensity、3DGS appearance 合成 | [P §3–5, Fig. 3] |
| Query frequency | 逻辑上按 output pixel 查询 field/decoders；shadow U-Net按整幅多尺度 map工作；是否 tile/crop、逐灯/多灯调用方式未报告 | [P Fig. 3–4] |
| Parameter count/MAC/FLOP | 未报告；缺 exact topology，不能可靠重算 | [P §4–6; S unavailable] |
| Shared/per-luminaire/state bytes | 未报告 network weights、light tokens、triplane、position field、3DGS、RSM、shadow pyramid 或 TensorRT workspace bytes | [P §4–6] |
| Texture/feature fetches | spherical triplane需三 plane 查询，但 interpolation、plane fusion、resolution/channel未报告；G-buffer/RSM fetch 数未报告 | [P §4.2–5] |
| Precision | Table 1 网络使用 half precision，TensorRT执行；具体 FP16 accumulation、TF32、INT8、quantization 未报告 | [P Table 1 caption] |
| Hardware/backend | NVIDIA RTX 4090D；network TensorRT half precision；OIDN 使用 OptiX 8.0 FP16 input，path tracing 由 RT cores 加速 | [P §6, Table 1 caption] |
| Resolution | 512×512 | [P Table 1 caption] |
| Runtime | LightFormer `10.47 ms`；OIDN `11.45 ms (13 spp)`；Ours `10.56 ms` | [P Table 1] |
| Runtime scope | Table 1 没有拆分 G-buffer/RSM/shadow map、field fetch、three decoders、3DGS rasterization、TensorRT、copy/synchronization；也未说明 luminaire field generation 是否计时 | [P Table 1 caption] |
| Prepare/precompute | Trellis/3DGS generation、multi-view rendering、Lighting Encoder/Field Generator 与 Inverse HDR Splatting 的耗时均未报告；作者只称 view/luminaire update 不需显著 rendering overhead | [P p.3 contribution, §5.3] |
| Memory | 未报告 GPU peak/resident memory、asset storage、streaming 或 multi-luminaire scaling | [P §4–6] |

在固定 output resolution、shadow levels、triplane resolution 和 network topology 后，NeLiF 单帧 inference 逻辑上有界；但论文没有披露这些关键静态上限，也没有报告多 luminaire composition 规则与成本 scaling，不能把 `10.56 ms` 改写成 local `evaluate()` 的 query cost。[P §4–6][I]

## 9. 实验 protocol、baseline、指标与完整结果

Table 1 是唯一的定量总表；三种方法都在 RTX 4090D、512×512 下测量。Neural networks 用 TensorRT half precision；OIDN 时间是 RT-core path tracing 加 OptiX 8.0 denoising，输入 FP16。正文没有披露指标颜色空间、tone mapping、逐图聚合、test image 数、方差或置信区间。[P Table 1 caption]

| Method | Time ↓ | PSNR ↑ | LPIPS ↓ | SSIM ↑ | sMAPE ↓ | protocol locator |
|---|---:|---:|---:|---:|---:|---|
| LightFormer | `10.47 ms` | `24.35` | `0.207` | `0.823` | `0.208` | [P Table 1] |
| OIDN | `11.45 ms (13 spp)` | `24.87` | `0.193` | `0.813` | `0.213` | [P Table 1] |
| Ours | `10.56 ms` | `29.33` | `0.084` | `0.882` | `0.167` | [P Table 1] |

作者的 formal comparison 边界：[P §6.1]

- LightFormer 与 NeLiF 都在同一 400K subset 上训练 35 epochs；正文没有报告二者是否 exact parameter/time matched，也没有给 repeats。
- Active Exploration、CNSR、Neural Scene Graph、Dynamic Neural Radiosity、NeLT 与 Superposed DFF 被排除，因为作者认为它们需 per-scene training，不能泛化到 unseen scenes。排除不是数值胜出，更不是这些方法“失败”。
- OIDN 使用作者所称 latest light-source implementation；exact source commit、integrator 与 sampling schedule 未报告。

其他结果均是定性证据：

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| Full rendering comparison | 四组 complex-luminaire scenes；Fig. 5 同时显示 Reference、OIDN、Ours，以及 direct-only 的 DirectShading/OursD/LightformerD | OIDN、LightFormer direct component | visual only；caption提到 8000 spp path tracing | 作者报告 Ours 更接近 reference；OIDN在复杂灯具采样下有 discrepancy；LightFormerD在 highlights/shadows 有明显差异 | [P Fig. 5, §6.1] |
| Novel luminaires | 同一 unseen scene/location 放置 3 个 novel luminaires | Reference | visual only | encoder捕捉不同 illumination patterns | [P §6.2, Fig. 6] |
| Position generalization | 同一个 novel luminaire 在 unseen scene 的 3 个位置 | Reference | visual only | transport module把局部多视图观测传播到全空间 | [P §6.2, Fig. 7] |
| Shadow generalization | 不同 luminaire/scene | Reference | visual only | shadow softness/hardness随 learned lighting cues变化 | [P §6.2, Fig. 8] |
| Inverse HDR Splatting | LDR luminaire→splatted HDR，与 ground-truth HDR、synthetic-HDR lighting、reference比较 | GT HDR/multi-view | visual only | splatted HDR 3DGS产生与 reference近似的 illumination | [P §6.3, Fig. 9] |
| Internet single image | 单张无 GT 的网络灯具图像→Trellis 3DGS→lighting function | 无 | visual only | 作者称3DGS与lighting visually convincing；不能形成精度结论 | [P §6.2, Fig. 10] |
| Kernel shadow | 复刻 LightFormer neural-shadow architecture并调 feature dims/layer count以保持 performance；另做 close-range occluder，与 direct prediction及五级 soft shadows比较 | LightFormer shadow、direct prediction、hard shadow | visual only | kernel module在极近 occluder 区域产生更合意的 soft result | [P §6.3, Fig. 11] |
| Spherical triplane | regular vs spherical triplane，resolution 32 与 128，展示 grid density 与 shading | vanilla triplane | visual only | spherical triplane对近光源/高频 lighting function 更好 | [P §6.3, Fig. 12] |

Figure 5 caption 同时说“traditional path tracing (8000 spp) struggle”与 OIDN/reference discrepancy，但正文没有给 reference 的 spp、8000 spp 图列身份或 error metric。该图不能支持“8000 spp path tracing 普遍差于 NeLiF”的跨方法结论，只能记录作者在所示复杂灯具采样场景中的定性观察。[P Fig. 5]

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `author-positive` | spherical triplane替代 regular triplane，分辨率32/128 | close-range/high-frequency lighting 的 qualitative shading更接近reference | 球坐标把更多grid density分配给近光源区域，符合 illumination frequency 随距离下降的结构 | 只证明几何分配机制有定性价值；没有 matched bytes/time/metric | [P §4.2, §6.3, Fig. 12] |
| `ablation-inferior` | direct shadow prediction baseline | close-range occluder区域不如 kernel-based shadow柔和/合理 | per-pixel features与不连续shadow clues难处理高频边界 | exact baseline topology和metric未报告，不能量化增益 | [P §5.2, §6.3, Fig. 11] |
| `ablation-inferior` | LightFormer neural-shadow architecture，经 feature dims/layer count调整到相近performance | 作者报告跨不同scene geometry的robustness不如NeLiF kernel design | multi-scale learned filter/upsample显式处理shadow discontinuity | “相近performance”的匹配准则、params、time未报告 | [P §6.3] |
| `author-positive` | Inverse HDR Splatting vs LDR 3DGS | 无训练地恢复 plausible HDR light source，relighting接近GT HDR multi-view输入 | Gaussian贡献权重提供HDR ratio反投影 | 仅qualitative；无HDR error/outlier/动态范围protocol | [P §5.3, §6.3, Fig. 9] |
| `author-negative` | monolithic neural radiance architecture捕获 high-frequency lighting pattern | supplemental所示pattern仍存在 spectral limitation | 单尺度/monolithic representation形成瓶颈；作者建议 multi-scale illumination encoding | supplemental不可得，无法核对失败图、频谱定义、严重度与受影响场景 | [P §7] |
| `known-limitation` | single-image Internet luminaire generation | 没有ground truth | 作者只给visual plausibility | 不能用于证明inverse HDR accuracy或真实域generalization | [P §6.2, Fig. 10] |
| `baseline-excluded` | AE、CNSR、Neural Scene Graph、Dynamic Neural Radiosity、NeLT、Superposed DFF | 未进入Table 1数值比较 | 需要time-consuming per-scene training，不能generalize到unseen scenes | 这是适用域/协议排除，不是author-negative或ablation failure | [P §6.1] |

正文没有报告训练不收敛、不同 optimizer/loss、field resolution、attention层数、观察视图数或不同数据规模的失败尝试。没有证据时不能从最终设计反推这些尝试过且失败。[P §4–7]

## 11. Paper ↔ supplemental ↔ code correspondence 与冲突

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Architecture | 给出4D tensor、`/16` patch spatial reduction、交替spatial/angular attention、cross-attention、spherical mapping、shadow kernel sizes/levels | 正文明确说有更多encoder、indirect module细节，但当前不可得 | unavailable | 只能恢复运算骨架，不能恢复逐层channels/heads/params |
| Data/query | 给出5,300 luminaires、10,000 scenes、1M images，以及matched 400K subset/35 epochs | data generation细节指向supplemental | unavailable | full 1M 与comparison 400K是两个口径；二者的run identity关系未明确 |
| Training | 只给35 epochs comparison、12×4090D与global-max normalization | training details指向supplemental | unavailable | loss、optimizer、batch、schedule、训练时间均未闭合 |
| Runtime | 512²、RTX4090D、TensorRT FP16、10.56 ms | inference details指向supplemental | unavailable | timing included scope、prepare/field generation、memory/bytes/MAC未闭合 |
| Eq. 1 notation | 公式用 `V(o,ω)` 表示 luminaire appearance；紧随 prose 却写 appearance `L`，而 `L` 已是 light sources | unavailable | unavailable | 按公式与 3DGS appearance 分支采用 `V`；记录为 paper-internal notation gap |
| Shadow energy | prose/Fig.4给 `η` energy correction；Eq.9–10只给scale blend | unavailable | unavailable | `η` exact公式、range、作用位置只可按图记录，不能补写 |
| Eq. 9 glyph | 公式在 `\tilde S^{l+1}` 前字面印有标量 `4`；prose只定义 `α` 为 upsampling kernel weight | unavailable | unavailable | 必须保留 `4`，但其来源/含义未定义；不能把它改成 Fig.4 的 `U` 或自行解释为归一化 |
| Final composition | Eq.1–2为`F+V`与`direct×shadow+indirect`；Fig.3实际画出 direct×shadow 后再接 Albedo/intensity，随后加3DGS；indirect另乘intensity，最终相加 | unavailable | unavailable | dataflow可逐线恢复，但 formal output quantity、单位与各节点对应未完整定义 |
| Inverse HDR | Eq.11给weighted ratio；“existing models”未命名 | unavailable | unavailable | ratio稳定化、RGB语义、Gaussian回写细节与Trellis配置未知 |
| Results | Table1唯一量化；Fig.5–12定性 | 额外shadow/triplane/spectral analysis不可得 | unavailable | main足以锁定作者主张，但不足以复现或独立审计 |

没有发现 main 与可得 official code 的直接冲突，因为 official code 不可得。以上条目是 `paper-internal-gap` 或 `paper↔missing-supplemental gap`，不能误写成 paper-code mismatch。

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

作者在 §7 明确承认：当前 monolithic neural radiance architecture 对 high-frequency lighting patterns 有 spectral limitation；作者提出 multi-scale illumination encoding 作为可能方向。[P §7] 这是正文唯一明确的 method limitation。由于承载例子的 supplemental 不可得，本报告不补猜其具体频率、error magnitude 或受影响灯具。

论文目标域是 indoor scenes/complex luminaires；这构成公开适用域，但不自动证明所有室外、频谱、透明/焦散或参与介质场景都已经尝试并失败。[P §1, §3.2]

### 12.2 复现所需但未报告/材料不可得

- 多视图数量、camera rig、angular/spatial resolution、depth/radiance format；
- PatchEmbed MLP、Transformer heads/layers/FFN/norm、Lighting Field Generator完整拓扑；
- learnable position field、spherical triplane resolution/channel/interpolation/fusion/边界处理；
- direct/indirect decoder逐层结构、G-buffer exact channels、RSM/VPL格式与multi-bounce estimator；
- shadow U-Net widths、kernel normalization、Eq. 9 字面因子 `4` 的来源、energy correction `η` 的公式和输出约束；
- loss/weights、optimizer、LR/schedule、batch、steps、seed、初始化、checkpoint selection；
- dataset来源、split manifest、reference renderer/integrator/spp/bounces、tone mapping与指标实现；
- full 1M 与 matched 400K runs 的对应、test样本数、variance/CI；
- network params/MAC、weights/field/3DGS/RSM bytes、peak memory；
- TensorRT engine/build/config、10.56 ms breakdown、field/3DGS generation与update时间；
- multi-luminaire composition、maximum supported lights、temporal behavior/history/reprojection；
- official code/config/data/checkpoint与可访问 supplemental。

这些缺口阻止 faithful reproduction，但不否定论文的 author-positive 结果；它们必须保持为 `author-underspecified`，不能由相邻论文或默认实现填充。[P §4–7][C/S unavailable]

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

NeLiF 的容量被拆到四个层级：[I]

1. shared Lighting Encoder/Generator 学会从观测到 light-conditioned field 的 compiler；
2. per-luminaire spherical triplane缓存灯具对周围空间的聚合影响；
3. shared direct/indirect/shadow decoder结合当前scene observations完成image-space transport；
4. 3DGS单独承载灯具自身的可见appearance，Inverse HDR Splatting补回emission dynamic range。

这比 CNSR/AE 的单 scene latent 更明确地把“灯具本体”“灯具对空间的作用”“当前场景visibility/material response”拆开；也比 LightFormer 每 pixel/per-frame跨lights attention更积极地把聚合前移到 per-luminaire field。但它没有把 transport 分解成可由任意 renderer重组的物理 operator：field与decoder仍在训练分布中共同决定最终shading。[I]

### 13.2 成功所依赖的假设

- 灯具的近场多视图 radiance/depth足以辨识其对周围空间的主要照明分布；不可见内部结构、非朗伯灯罩、频谱和极窄峰不能由观测唯一决定。[I]
- 把照明复杂度主要组织为距灯具的 spherical radial/angular variation，适合作者的室内灯具；远场环境光、超长条灯、强方向投射器未必服从同一grid-density先验。[I]
- G-buffer、RSM与hard-shadow clues由传统renderer稳定提供；NeLiF并未消除这些raster/shadow pass的成本与偏差。[I]
- shadow可由五级 screen-space learned filtering近似；跨屏幕、透明occluder、二次visibility或高度非局部penumbra可能超出clue domain。[I]
- 5,300灯具/10,000场景的合成分布覆盖部署域；论文没有公开split，因而此假设尚不能独立验证。[I]

### 13.3 与 AE、NeLT、Superposed DFF、CNSR、LightFormer 的关系

| 方法 | NeLiF 正文中的关系 | 可迁移机制 | 不可混同的边界 |
|---|---|---|---|
| Active Exploration for Neural Global Illumination（AE） | 早期方法从 predefined variable parameters提取scene representation；NeLiF把它列入per-scene-training excluded baselines | AE的error-guided training-query allocation可作为NeLiF/本项目训练策略研究，而非runtime表示 | AE仍是scene/config→image radiance；NeLiF不是AE architecture的轻量版 [P §1, §6.1][I] |
| CNSR / Neural Scene Graph | 从sparse observations提scene latent；single latent给full-scene illumination造成capacity pressure；因per-scene training排除 | observation encoder、capacity partition可作scene diagnostic | NeLiF生成per-luminaire field并以当前G-buffer/RSM解码，不是global scene latent [P §1, §6.1][I] |
| NeLT | object-centric decomposition改善动态适应，但仍per-scene training | object/light/environment factorization说明scene transport需显式拆容量 | NeLiF以luminaire observations为source并主张跨scene/light generalization [P §1, §3.2, §6.1][I] |
| Superposed Deformable Feature Fields | NeLT后续用deformable fields与implicit light transport增强；仍列入per-scene excluded methods | deformable/spatial feature placement可对照spherical radial allocation | 二者field的conditioning、持久化与generalization identity不同 [P §1–3, §6.1][I] |
| LightFormer | 正式baseline；NeLiF采用其indirect module；认为LightFormer每帧聚合light effects有冗余，并把它缓存为lighting field | G-buffer/RSM/VPL物理clues、direct/shadow/indirect decomposition有直接谱系 | NeLiF的field generation与kernel shadow是新增路径；Table1并非local material比较 [P §4.2, §5.1, §6.1][I] |

### 13.4 与本项目 runtime contract 的关系

NeLiF 适合归类为**scene-level auxiliary renderer / capacity diagnostic**，不是 material runtime candidate。[I]

- `prepare()` 可类比 per-luminaire field generation，但原论文 prepare cost、bytes与更新频率未报告；
- runtime按整图使用U-Net shadow、多级maps、G-buffer/RSM与3DGS，不能编译为每个shading point随机访问的小MLP `evaluate(wo,wi)`；
- 输出已经包含 visibility 与 global transport，不具备局部 `f` 的measure，也没有 matched `sample/pdf`；
- 可迁移的是 representation allocation、input normalization与component supervision思路，不是query semantics或10.56 ms数值。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

总体状态：`not-applicable`（method correspondence）+ `project-hypothesis`（训练/表示机制）。当前 NVIDIA reproduction 直接拟合材质局部线性 `f(wo,wi)`，NeLiF 预测整帧 scene lighting；二者没有 faithful architecture/loss/runtime correspondence。[N `docs/research/experiment_framework.md`][I]

| NeLiF机制 | 对当前NVIDIA方法的可用影响 | correspondence标签 |
|---|---|---|
| global-max RGB normalization + inverse scale | 可作为每个source state/asset的HDR conditioning与target-scale实验，但必须保证runtime仍返回线性`f`，并冻结scale定义 | `project-hypothesis`，非faithful requirement |
| spherical proximity allocation | 当前LayerStack local BRDF没有“距灯具半径”；不能直接套用。对未来spatial material/lighting cache可研究distance-adaptive grid | `not-applicable` to local evaluator |
| generated light field | 对应“compiler产生runtime latent/state”的一般结构，但NeLiF source是luminaire observations，target是scene transport，不证明材质source compiler能零样本工作 | `project-hypothesis` |
| direct/indirect/shadow分路 | 当前local scattering不应引入scene shadow/indirect heads；可借鉴的只是把已知不同频率/语义component做训练期辅助监督 | `interface-incompatible` / training-only hypothesis |
| multi-scale kernel shadow | 依赖screen neighborhood与整图U-Net，违反当前random-access evaluator热路径；不应作为本轮产品候选 | `not-applicable` |
| Inverse HDR Splatting | 是3DGS emission recovery，不对应材质latent；其contribution-weighted反投影可作为future target-encoder研究类比 | `not-applicable` / analogy only |

NeLiF 没有暴露局部方向parameterization、BRDF loss或sampler，不能用来证明当前NVIDIA half/difference coordinates、log-L1或GGX proposal正确/错误。[N `docs/research/experiment_framework.md`][I]

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| `H-NeLiF-scale`：按source state/asset显式预测或记录HDR scale，再让compact evaluator拟合normalized `f`，可降低跨state梯度尺度方差 | NeLiF用跨views global max RGB归一radiance并在输出乘回scale，作者称恢复能量且稳定optimization [P §4] | local `f`的dynamic range也可被一个静态有界scale分离，且near-zero/peak state不会被max统计放大噪声 | 当前target/loss vs相同network/queries/budget的frozen scale-normalized target；部署时乘回scale | source split、query recipe、network/latent、optimizer、steps、seeds、scale统计与epsilon | G1/G2/G2s主指标、peak recall、energy error、seed variance、scale bytes/time | evaluator runtime：增加固定1–3 scalars/state | 若无matched Pareto改善，或峰值/能量/未见state显著变差，证伪 |
| `H-NeLiF-aux`：训练期把local scattering分成可定义的component auxiliary heads，再丢弃heads，可提升shared latent对难分量的覆盖 | NeLiF direct/shadow/indirect分路与kernel shadow在其scene task有定性/定量正结果 [P Eq.2, Table1, Fig.11] | 当前source family存在有权威reference定义、且和runtime最终`f`一致可重组的component；不是凭经验发明layer closure | base evaluator vs相同runtime evaluator+training-only component heads；最终只输出相同linear `f` | authoritative component definition、loss weights、queries、network runtime core、training budget、seeds | final `f`主指标、component strata、peak/energy、训练时间；runtime parity | training-only auxiliary；runtime不增加读取/MAC | 若final `f`无改善、component tradeoff导致任一重要stratum恶化，或source无唯一component GT，停止 |
| `H-NeLiF-field`：对未来spatial scene-lighting支线，per-source field generation可把per-pixel跨source聚合前移到prepare阶段 | NeLiF明确把LightFormer per-frame aggregation缓存进generated lighting field [P §4.2] | lighting/source在多个pixels/frames复用，prepare成本可摊销，且field bytes有界 | per-pixel attention聚合 vs generated field；相同scene/source queries、quality与总wall time | source count、update frequency、field resolution/channels、prepare horizon、visibility inputs、hardware | quality、prepare+frame amortized time、bytes、update latency、temporal error | future scene-transport；不是current material evaluator | 若短复用窗口内总成本不降、field memory超预算或dynamic update造成明显stale error，证伪 |

上述假设只进入后续matched实验规划，不是本研究任务的quality hard gate，也不把NeLiF机制自动称为本项目novelty。[I]

## 16. 证据索引

- `P-title/abstract/contributions`：pp.1–3，正式身份、问题、two-stage outline、三项贡献、Fig.1规模概括。
- `P-problem`：§3.1，Eq.1–2，scene/light/camera映射与 direct/shadow/indirect decomposition。
- `P-pipeline`：§3.2，Figs.2–3，Illumination Encoding、Shade Decoding、3DGS/HDR并行路径。
- `P-encoder`：§4–4.1，Eqs.3–5，4D light-field tensor、PatchEmbed `/16`、spatial/angular alternating attention、global-max normalization。
- `P-field`：§4.2，Eq.6，cross-attention与spherical light triplane；p.6 spherical equal-area/radius-8 mapping。
- `P-decoders`：§5.1，direct/indirect decoder与LightFormer indirect-module继承。
- `P-shadow`：§5.2，Figs.4、11，Eqs.7–10，四channel clues、五级pyramid、5×5 filtering、4×4 upsampling、softmax blend与energy correction。
- `P-HDR`：§5.3，Figs.9–10，Eq.11，Trellis/3DGS与training-free Inverse HDR Splatting。
- `P-results`：§6–6.3，Table1，Figs.5–12，dataset/hardware、quality/generalization、定性消融与baseline排除。
- `P-limit`：§7，monolithic neural radiance architecture的high-frequency spectral limitation与multi-scale方向。
- `S`：正文确认存在并多次引用，但当前不可得；未使用二手材料补全。
- `C`：official code/config/data/checkpoint未找到；未执行Git网络探测、SSH/token或登录。
- `A`：WeLight publication entry与SIGGRAPH Asia official presentation，只作身份/摘要核对。
- `N`：`docs/realtime_material_compilation.md`、`docs/research/experiment_framework.md`和本任务已复核报告，仅用于§13–15。
- `I`：§13–15；没有写回作者事实。

## Evidence review

```text
author_worker: /root/nelif_full_report
reviewer: /root/nelt_full_report
reviewed_at: 2026-08-29
sources_rechecked: [local main PDF SHA-256 07558A3B7D7CA47091337F3A5A41E4D57B0BB02F3EB5A0387D3DBF0F47D398DD, PDF text layer, 11-page 150 dpi raster, Eq.9 600 dpi raster, project-local supplemental/code locator sweep, WeLight public publication entry, SIGGRAPH Asia official presentation locator]
findings_closed: [Eq.1-11 and Fig.1-12/Table1 locators and values rechecked, Eq.1 V-vs-L internal notation conflict recorded, Fig.3 final wiring recovered without assigning undefined radiometric semantics, Eq.9 literal scalar 4 preserved and left unexplained, LightFormer separated from per-scene excluded baselines, full-1M-vs-400K run identity kept unresolved, supplemental/code absence rechecked]
remaining_evidence_gaps: [supplemental, official code/config/data/checkpoint, exact architecture, Eq.9 factor-4 rationale, eta formula/range, final-composition radiometric semantics, loss/optimizer/batch/schedule, dataset split/reference recipe, full-1M-vs-400K run identity, parameter/byte/MAC/memory/runtime breakdown]
review_status: passed
```

### 完成检查

- [x] main paper 11页已完整阅读，Eq.1–11、Fig.1–12、Table1、图注、脚注与参考文献已逐页视觉核对；
- [x] supplemental/appendix/勘误可用性已检查；正文确认supplemental存在，但当前项目目录、PDF attachment与公开一手入口均未提供可读副本；
- [x] official code/config/data可用性已检查；未发现可审计locator；
- [x] 正文已披露的architecture、training、runtime与主要结果均有locator；适用但未披露字段明确写“未报告”；
- [x] 失败尝试、较差消融、baseline排除与known limitation已正确分开；
- [x] paper internal gap、missing-supplemental gap与“未报告”均保留；没有虚构paper-code冲突；
- [x] `I`分析晚于事实层，没有改写作者结论；
- [x] NVIDIA影响只引用真实`N`证据，并明确query/runtime不适用边界；
- [x] 假设包含matched control、冻结轴、部署类别与证伪条件；
- [x] 独立 evidence review 已完成；关键公式、图示、baseline 边界与缺口经主 PDF 和可得一手入口复核，报告状态更新为 `evidence-reviewed`。
