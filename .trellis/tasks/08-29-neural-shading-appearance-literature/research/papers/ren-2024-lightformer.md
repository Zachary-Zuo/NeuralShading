---
paper_id: "ren-2024-lightformer"
title: "LightFormer: Light-Oriented Global Neural Rendering in Dynamic Scene"
authors: "Haocheng Ren, Yuchi Huo, Yifan Peng, Hongtao Sheng, Weidong Xue, Hongxiang Huang, Jingzhen Lan, Rui Wang, Hujun Bao"
year: "2024"
venue: "ACM Transactions on Graphics 43(4) / SIGGRAPH 2024"
doi: "10.1145/3658229"
report_status: "evidence-reviewed"
main_source: "https://wylighting.github.io/lightformer/static/pdf/paper.pdf"
supplemental_status: "available"
official_code_status: "unavailable"
official_code_commit: "not-applicable"
author_worker: "/root/lightformer2024"
reviewer: "/root/lightformer2024_review"
last_verified: "2026-08-29"
---

# LightFormer: Light-Oriented Global Neural Rendering in Dynamic Scene

## 1. 研究对象与报告边界

LightFormer 是一个**场景级、整帧 radiance inference pipeline**。它每帧从当前相机的 G-buffers、每盏灯的 virtual point lights（VPLs，虚拟点光源）和 reflective shadow map（RSM，反射阴影图）构造 per-light neural representation，再以 pixel-light attention 为每个 shading point 聚合所有灯，最后分别预测 direct shading、direct shadow 与 indirect shading，并合成为最终 RGB 图像。[P Abstract, §3–5, Eqs. 1–7, Figs. 1–3]

本文所谓“fully dynamic”覆盖作者实验域内的 camera、灯光、材质参数、刚体变换、animated/skinned objects 与环境光变化；它不是无需训练即可接受任意新场景的 universal renderer。正式实验对每个 scene dataset 生成 20,000 个随机训练配置并训练约 50 小时，§6.4 又明确说训练集“consists of one scene only”。论文的泛化实验因此是**同一 scene/domain 内的未见动作、角色/对象和变换**，不是跨场景 zero-shot generalization。[P §6.1, §6.4, Fig. 9]

该方法不输出局部材质的 `f(wo,wi)`、方向采样 proposal 或 `pdf`；也不把 scene visibility 与 global transport 分解成可被 path tracer 任意组合的局部 operator。它属于本任务的 `scene-transport` 波次，不能与 local neural material evaluator 按同一输出语义排名。与 NeuralShading 当前方法的关联仅在 representation factorization、物理辅助 buffer、direct/visibility/indirect 分解和多灯摊销机制层面。[P §3–5][N `docs/realtime_material_compilation.md`][I]

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | [作者项目页 PDF](https://wylighting.github.io/lightformer/static/pdf/paper.pdf)，14 pages；DOI [`10.1145/3658229`](https://doi.org/10.1145/3658229) | 2026-08-29 | SHA-256 `B9214208DB8154B03885E1EDF6E057EA41E857053A964186517CC1C81E45C470`；73,486,283 bytes | 正式方法、数据、训练、实验、消融与限制的主证据。14 页已完整阅读；方法页 5–8、Table 1–2、Fig. 1–10、Eq. 1–8 均已 rasterize 并视觉核对。 |
| Supplemental `S` | [作者 supplemental PDF](https://wylighting.github.io/lightformer/static/pdf/supp.pdf)，3 pages | 2026-08-29 | SHA-256 `67C27A84017D852E2FDDDB44FDCFF8D649B5F565D6A3E23FA2CD3E84BB569F65`；4,452,212 bytes | 网络逐层结构、Interior Design 结果、Instant Radiosity 对照。3 页均已 rasterize；重点核对 Fig. 1–3 与 Table 1–2。 |
| Official code/config/data `C` | 项目页的公开导航只列 Paper、Supp、Video、Interactive viewer；以完整标题和作者组合检索公开 GitHub repository 返回 0 项 | 2026-08-29 | unavailable；`official_code_commit=not-applicable` | 没有官方训练/runtime code、config、checkpoint、scene assets 或 dataset manifest，不能审计 attention 实现、export、真实资源布局或 paper reproduction entry。 |
| Author page/talk/correction `A` | [作者项目页](https://wylighting.github.io/lightformer/)；[公开补充视频](https://pan.zju.edu.cn/share/555b20693638fcd67b7f8968c6?isCurrent=1&isopen=1&preview=455036531272&preview_side_active=1&scenario=share&share_type=file)；[结果 viewer](https://wylighting.github.io/lightformer/comparison_tool/index.html) | 2026-08-29 | project HTML SHA-256 `DA152208151463A38754D2568B9D5034A7E3505C1F269CE034CB261D42C9A033`；video SHA-256 `A5479C9E83F08FE690AED6A509C309A2CB155C8F45B91AC7DAD2550E0B45ED27`、398,856,314 bytes；viewer `data.js` SHA-256 `021D65DC7C37698904E0C7268DA0B7F8BAFF2216EFE61688A0E069902DDC7751` | 项目身份、公开视频/结果资产可用性和动态演示。视频为 697.7 s、1280×720、50 fps、无音轨；按 20 s 间隔抽帧并重点查看编辑、对照和 performance overlay。未找到独立作者 talk、slides 或 correction。 |
| NeuralShading evidence `N` | `docs/realtime_material_compilation.md`；`.trellis/tasks/08-25-03-neural-baseline-and-candidate/research/nvidia-method-correspondence.md` | 2026-08-29 | 当前 workspace | 只用于第 13–15 节判断 local runtime contract、当前 NVIDIA identity 与迁移边界，不回填为论文事实。 |

补充视频的项目页入口与浙江大学公开分享页均不要求账号；本轮只使用 public web/API/author files，没有使用 SSH token、Git 凭据或登录。项目页的 BibTeX 把 Hongxiang Huang 排在 Weidong Xue 之前，但 main、supplemental 与 DOI metadata 均是 Weidong Xue 在前；front matter 采用后三者一致的顺序。[P p.1][S p.1][A project page]

作者 main PDF 页脚仍写 `Article 1`，supplemental 更保留 `Vol. 1, No. 1` 与占位 DOI `10.1145/nnnnnnn.nnnnnnn`；正式 DOI 已由 main 首页和 publisher metadata 锁定为 `10.1145/3658229`。这些是作者公开版本的排版占位差异，不应被解释为另一篇论文。[P p.1][S p.1]

## 3. 原论文的问题、假设与贡献边界

作者把场景渲染写成

\[
L(o,\omega)=F(o,\omega;\mathcal S,\mathcal L),
\]

其中 `o,ω` 是 camera ray，`S` 是完整场景，`L` 是所有灯。作者认为把 object/scene parameters 直接烘进网络会让模型随变量数增长并限制 animated/unseen objects；仅用 screen-space G-buffers 又看不到屏幕外 transport。LightFormer 因而把“容量组织单位”从 object/scene 改成 light observation：每盏灯观察当前场景，生成 VPL、RSM 与 shading clues；网络主要学习如何把这些当前帧、与光传输相关的 observation 变成最终 radiance，而不是记住固定对象参数。[P §1, §3.1, Eq. 1]

作者提出的两阶段与 classic many-light pipeline 对应：

1. **Light Encoding**：direct VPL 描述 emitter，indirect VPL/RSM 描述被灯照亮的表面；另编码 shadow、light direction 与 half vector clues，得到每盏灯的 screen-space light embedding；[P §3.2, §4, Fig. 2]
2. **Light Gathering**：用当前 pixel 的 G-buffer 查询所有 light embeddings，以 pixel-light attention 代替显式 light culling/逐 VPL 累加，再由 direct/indirect decoders 预测 radiance components。[P §3.2, §5, Fig. 3]

作者贡献边界是：支持实验域内全动态场景；提出 neural reflective shadow map；提出 pixel-light attention，并在四个 scene datasets 与一个补充场景上展示实时结果。作者同时明确这项工作探索的是 neural rendering framework 的潜力，不是 production-ready GI；path tracing 对复杂材料、复杂 transport 和完全动态场景仍更一般。[P p.2 contributions, §6.2, §7–8]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Scene/source input | 当前帧可 rasterize 的几何、材质、相机和所有灯；训练时每个 scene dataset 随机化其声明的可编辑变量 | scene-specific dynamic configuration | [P §3, §6.1] |
| Runtime query | 对当前 camera ray 命中的每个 shading point `x` 预测 `L(x,ω)`；网络不是按任意 `wi` 查询局部 BSDF | 512×512 inference image in formal data pipeline | [P Eq. 1, §5–6.1][A video 08:20 title card] |
| G-buffer query `g(x)` | world position(3)、normal(3)、diffuse/albedo(3)、specular color(3)、roughness(1) | 13 channels/pixel | [P Eq. 5, §5.2][S Table 2] |
| Direct VPL | position(3)、normal(3)、由 emitted radiance 与 sampling probability 形成并按每灯平均 expected power 归一化的 power(3) | 9D/sample；每灯的样本集合 | [P §4.1][S Table 2] |
| Indirect VPL | RSM texel 的 position(3)、normal(3)、reflected flux(3)；depth 另用于 shadow clue | 9D/sample | [P §4.2][S Table 2] |
| Shadow clues | `S={d-d_f,d/d_f,c_e,c_c,p}`；`d` emitter-to-occluder depth，`d_f` pixel-to-emitter distance，`c_e,c_c` 是 normal 分别与 light/view direction 的点积，`p` 是 position | 7-channel screen map/per light | [P Eq. 2, §4.3][S Table 2] |
| Specular clues | 每 pixel 的 light direction 与 Blinn-Phong half vector；area light direction用 direct-VPL mean position 的 point proxy | 两个 3-channel maps/per light | [P §4.3] |
| Light coordinates/types | area、environment、point、directional；正文还说明 spot/directional RSM 用单纹理，omnidirectional/area 用 cubemap；无效字段填唯一常数 | per-light VPL set + light-view textures | [P §4.1–4.2] |
| Output components | `T^1` direct shading、`T^{1S}` direct shadow、`T*` indirect shading，均为 RGB | 3+3+3 channels/pixel | [P Eq. 6][S Table 2] |
| Final output | `L=T^1⊙T^{1S}+T*` | path-traced component supervision 对应的 RGB radiance-like image；绝对物理单位、exposure 与显示变换未报告 | [P Eq. 7, §6.1] |
| Domain restrictions | scene-specific training distribution；G-buffer material model 能表达的参数；有限且实验中较少的 lights；复杂 transport 未验证 | 非 arbitrary-scene、非 local scattering query | [P §6.1, §6.4, §7] |

正文和 supplemental 公开的输入/网络中没有列出历史帧、motion vector、reprojection 或 recurrent state。作者把 temporal stability 归因于 temporally consistent、clear 的当前观察与所设计的 pipeline，并用视频作定性证据；但 direct VPL 来自 random samples，论文没有说明跨帧是否复用样本、RNG 如何固定，官方代码也不可得，因此不能把整条 runtime 写成已证实的“确定性推断”。正文没有 temporal metric、sequence aggregation protocol 或 history ablation。[P §2.4, §3.2, §4.1, §6.2][A supplemental video][C unavailable]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

对每盏灯 `l_k`，运行时执行：

1. 在 emitter 上按 emitted-radiance distribution importance-sample direct VPLs，记录 position、normal、radiance、sampling probability；计算 expected power，并按该灯的平均 expected power 归一化。[P §4.1]
2. 从灯的 point proxy 渲染 RSM。directly illuminated surface texels 成为 indirect VPLs；depth 生成 shadow clues，position/normal/flux 生成 indirect-light observations。area light的 proxy 是 direct VPL mean position；area/omnidirectional light用 cubemap，directional/spot light用单纹理。[P §4.2]
3. Direct/indirect VPL MLP 对每个 VPL 编码，再 average-pool 成该灯的两个全局 vectors。light direction、half vector 与 shadow UNet 产生该灯的 screen-space feature maps；全局 vectors repeat 到 screen size 后与 maps concat，形成 `z_k`。[P Fig. 2, §4.1–4.3]
4. 对每个 pixel，G-buffer 经 MLP 形成 query `Q`；正文把每灯 full embedding `z` 定义为 value、把截取出的 direct-light embedding `z_d` 定义为 key。8-head pixel-light attention 对 lights 聚合成 `z~`。Fig. 3 的 value projection 前另画了一个 `C`（concatenation），但正文和图没有标明它的另一输入或投影宽度；因此这里只锁定正文的 Q/K/V 语义，不把图中未解释的 concat 猜成具体实现。[P §5.1, Fig. 3]
5. Direct decoder 读取 13D G-buffer 与 composed direct/half/light-direction/shadow embeddings，输出 `T^1,T^{1S}`；indirect decoder 读取 G-buffer 与 composed indirect embedding，输出 `T*`；Eq. 7 合成最终图像。[P §5.2, Eqs. 6–7][S Table 2]

以上是每个 runtime scene configuration 需要形成的逻辑数据流。论文没有报告 direct VPL random samples 是逐帧重抽、固定后随 emitter 变换，还是采用其他复用策略；也没有报告 RSM、VPL 与 clue 的更新/缓存粒度。因此不能从方法图直接推出精确的每帧采样频率、随机性或缓存成本。[P §4–5][C unavailable]

作者用 many-light 求和

\[
L(x,\omega)=\sum_j f(x)G(x,y_j)V(x,y_j)L_j
\]

解释物理动机，但 LightFormer 不显式逐 VPL 计算这条 estimator：同一灯的 VPLs 先压成一个 light embedding，visibility/geometry/BRDF contribution 由 clues、attention 与 decoders 隐式近似。[P Eq. 3, §5]

### 5.2 持久化表示

- **持久化**：一个 scene dataset 训练得到的 encoder/attention/decoder weights；论文没有报告 model bytes、checkpoint 格式或跨 scene 共享同一 checkpoint。[P §6.1]
- **每帧、每灯临时状态**：direct VPL set、RSM cubemap/texture、indirect VPL set、screen-space clue maps 与 light embedding。[P §4]
- **每帧、每 pixel 临时状态**：G-buffer、attention query/composed embedding、三个 radiance components。[P §5]
- **没有 per-object persistent latent**：object state 通过当帧 G-buffer/RSM 进入；这是 object 数不直接决定 neural composition 次数的原因，但 scene-specific weights 仍承载该训练场景分布。[P §1, §2.2, §3]
- **LOD/quantization**：没有 material mip/footprint、network quantization 或 neural LOD。RSM 的 depth/other channels 使用不同 resolution，Emernald Square 用四级 cascades；这是 scene observation 的频率/覆盖分配，不是局部材质 LOD。[P §4.3, §6.1]

按 supplemental Table 2 明列的组件相加，per-light full embedding 包含 `256+512+32+32+8=840` channels；论文没有单独给出这个总数，也没有报告 attention 内部 projection width。[S Table 2]

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Direct VPL Encoder | 9D `position+normal+power` | `9→256→256→256→256→256`；per-VPL 后 average pooling | 前四个 Linear 后 LeakyReLU；末层无标注 activation | 256D direct light vector | scene model 内共享；per-light runtime value | [S Fig. 3, Table 2] |
| Indirect VPL Encoder | 9D `position+normal+flux` | `9→64→64→64→128→512`；per-VPL 后 average pooling | 前四个 Linear 后 LeakyReLU；末层无标注 activation | 512D indirect light vector | 同上 | [S Fig. 3, Table 2] |
| Light Direction Encoder | 3D direction map | `3→64→64→64→32`，1×1-equivalent per-pixel MLP | 前三层 LeakyReLU；末层无标注 activation | 32-channel map | shared；per-light runtime map | [S Fig. 3, Table 2] |
| Half Vector Encoder | 3D half-vector map | `3→64→64→64→32` | 前三层 LeakyReLU；末层无标注 activation | 32-channel map | shared；per-light runtime map | [S Fig. 3, Table 2] |
| Shadow Encoder | 7-channel clue map | U-Net：`DoubleConv 8→Down 16→32→128→256→Up 128→32→16→1×1 Conv 8`，带 skip；`DoubleConv=3×3 Conv→ReLU→1×1 Conv→ReLU`，Down 为 `2×2 AvgPool+DoubleConv`，Up 为 `2×2 Upsample+DoubleConv` | DoubleConv 内 ReLU；final activation 未标 | 8-channel shadow map | shared；per-light runtime map | [S Fig. 3, Table 2] |
| Pixel-light Attention | G-buffer query；direct-embedding keys；840D full-embedding values | Fig. 3 画出 Q/K/V 的 MLP、attention 加权和输出 MLP，并在 value projection 前画出一个未解释的 concat；8-head attention；full `z` 作 K 的 variant 约为当前 K 的 3×计算 | concat 另一输入、activation、projection width、normalization、residual/softmax细节未报告 | per-pixel composed embedding | shared；lights 维聚合 | [P §5.1, Fig. 3] |
| Direct Decoder | 341D=`G-buffer13+256+32+32+8` | `341→256→256→256→35`，35 split为 direct RGB3 与 latent32；latent32 concat shadow embedding8，再 `40→32→32→3`；图示另有 input skip 到第三个 256 block | 主干四个 Linear 后 LeakyReLU；shadow支路两个 hidden 后 LeakyReLU、final Tanh | direct shading RGB + direct shadow RGB | shared | [S Fig. 3, Table 2] |
| Indirect Decoder | 525D=`G-buffer13+indirect512` | `525→256→256→256→3`；图示 input skip 到第三个 256 block | 三个 hidden 后 LeakyReLU；final无标注 activation | indirect shading RGB | shared | [S Fig. 3, Table 2] |

Supplemental prose 称“direct encoder and indirect encoder incorporate a skip connection”，但 Fig. 3 的 skip arrows 明确画在 Direct/Indirect **Decoder**；没有代码可确认 skip 是 concat、add 还是特定 pixel-generator operator。这一处保留为 source ambiguity，不自行补算精确参数量。[S §1, Fig. 3]

### 5.4 条件化、坐标变换与物理先验

- **light-oriented factorization**：per-light observation 是 representation 单位；object properties 不作为固定 object vector 编进网络。[P §3]
- **VPL/RSM prior**：direct VPL 统一不同 emitter types；RSM把被照表面变成 indirect VPL，并显式提供 direct visibility clue。[P §4.1–4.2]
- **HDR normalization**：direct VPL expected power 除以每灯平均值，作者沿用 NeLT 并认为这改善 HDR emission distribution 的泛化。[P §4.1]
- **specular coordinate hint**：light direction 与 half vector把随 view/light 移动的高频 specular structure 显式输入网络。[P §4.3]
- **transport decomposition**：direct shading、direct shadow、indirect shading使用独立输出与 supervision，最后按 Eq. 7 合成。[P §5.2, §6.3]
- **visibility boundary**：shadow clue显式编码 emitter-to-primary visibility；作者为性能不做 indirect VPL 到 shading point 的显式 visibility test，这正是高频 indirect shadow 的已知薄弱点。[P §4.2, §7]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset/source scenes | Chess、Gig、Emernald Square、Living Room；每个 dataset 20,000 random scenes/configurations 训练、100 random scenes/configurations 测试。没有 validation split。 | [P §6.1, Fig. 4] |
| Chess variables | Cornell-box-like scene；2 area lights；animated chess pieces；lights 可 transform；两侧墙颜色可调。 | [P §6.1] |
| Gig variables | 6 area stage lights + 1 large textured emissive wall light；intensity/color变化；2 stage lights持续旋转；墙 base color随机；28 animated characters。正文后称总计 7 lights。 | [P §6.1–6.2] |
| Emernald Square variables | 10.68M triangles；40 animated characters/buses；高保真树；environment map 可旋转和替换。 | [P §6.1] |
| Living Room variables | 2 animated characters；3不同形状 area lights；sofa/lights 可 translation、rotation、scaling。 | [P §6.1] |
| Supplemental Interior Design | 5件可自由移动家具、3 area lights；sofa/wall 的 color与roughness、light emission可编辑。该场景的训练样本数/随机范围未报告。 | [S §2, Fig. 1, Table 1] |
| GT/reference renderer | Falcor path tracer；512×512；2,048 spp；同时记录 direct shading、direct shadow、indirect shading components。 | [P §6.1] |
| Train/test variable ranges | camera views会变化；各 object/material/light 参数的数值范围、联合分布、train/test seed与是否 stratify 未报告。 | [P §6.1] |
| VPL sampling | 每灯 500 VPLs，environment light 2,000；direct VPL按 emitted radiance importance-sample；论文没有分开报告 direct/indirect各自确切 sample count或随机 seed。 | [P §4.1, §6.1] |
| RSM/shadow map | 多数 scene shadow map 1024×1024；每 cubemap face depth为1024×1024，position/normal/flux为64×64。Emernald Square 使用4 cascades，各 buffer不同分辨率，最大2048×2048。 | [P §4.3, §6.1] |
| Baseline data | CNSR与AE在每个 scene的同一 dataset上训练至相等或更长时间；CNSR用完整 G-buffers；AE获得 normalized animation timestamp，采用 uniform data sampling。 | [P §6.2] |
| Filtering/LOD | 无 surface footprint/mip supervision；RSM multi-resolution和CSM只用于scene observation。 | [P §4.3, §6.1] |
| Augmentation/distillation/teacher | scene randomization本身是数据生成；没有 distillation/teacher。AE adaptive sampling未用于正式 LightFormer，作者列为 future work。 | [P §6.1–6.2] |
| Online/offline generation | training data离线由 Falcor path tracing生成；runtime当前帧 G-buffer/RSM/VPL online生成。训练 corpus、scene assets和生成脚本未公开。 | [P §4, §6.1][C unavailable] |

## 7. Loss、optimizer 与训练 lifecycle

正式 loss 对三个 component 分别监督：

\[
\mathcal L=\mathcal L_1(\hat T^1,T^1)+\mathcal L_1(\hat T^*,T^*)+
\mathcal L_1(\hat T^{1S},T^{1S})+0.1\,\mathcal L_{VGG}(\hat T^{1S},T^{1S}).
\]

作者在计算 loss 前对 HDR values 做 `log(1+x)`，VGG term只加在 direct shadow上以改善 soft shadow；`λ=0.1` 是让 magnitudes 约相等的经验设定。正文没有明确 `log(1+x)` 作用于哪些 prediction/GT operands、是否也进入 VGG preprocessing，因此 Eq. 8 不能单独恢复逐项的 exact transform placement。[P §6.1, Eq. 8]

| 项 | 正式配置 | locator |
|---|---|---|
| Target/output transform | 正文只说在 loss 前对 HDR values 使用 `log(1+x)`；没有明确逐项作用于 prediction、GT 和 VGG input 的位置。network runtime仍输出并合成 radiance components；log clamp/epsilon、negative handling未报告。 | [P §6.1, Eq. 8] |
| Loss terms and weights | 三项 per-pixel L1各权重1；direct-shadow VGG loss权重0.1。VGG具体layer/feature normalization未报告。 | [P Eq. 8, §6.1] |
| Optimizer | Adam，learning rate `1e-4`；β、epsilon、weight decay未报告。 | [P §6.1] |
| LR schedule | 未报告。 | [P §6.1] |
| Batch/query count | mini-batch size 4；一个 sample是整图还是crop、gradient accumulation、每次跨几张GPU如何分片未报告。 | [P §6.1] |
| Steps/epochs/stages | all encoders/decoders end-to-end jointly trained；总 steps/epochs、warmup/stage boundary未报告。 | [P §6.1] |
| Initialization/seed/model selection | 未报告 initialization、random seed、validation/checkpoint selection或重复训练方差。 | [P §6.1] |
| Hardware/training time | 2× NVIDIA RTX A6000，约50 h；每个 scene是否都各50 h、GT生成时间是否计入未明确说明。 | [P §6.1] |
| Rejected formal loss | DSSIM按AE方式加入时，在作者 cases 中没有 performance gain，未进入最终 loss。 | [P §6.1] |

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path and frequency | 每个当前配置生成G-buffer，并为每灯取得direct/indirect VPL、RSM与三类clue，编码成per-light map；每pixel跨灯attention一次，direct/indirect decoder各一次。论文没有报告VPL重采样、RSM刷新或跨帧缓存频率。 | [P §3–5, Figs. 2–3][C unavailable] |
| Asymptotic scaling | light encoding对pixel数和light数都线性；latent-space compose后decoder主要按pixel数执行，不逐灯decode。 | [P §7] |
| Parameter count/MAC/FLOP | 未报告；supplemental不足以恢复attention projection/skip精确实现，因此不自行估算“正式参数量”。 | [S Fig. 3][C unavailable] |
| Shared/per-asset/state bytes | 未报告weights、RSM/VPL transient bytes、workspace峰值或checkpoint/model storage。 | [P §4–6][C unavailable] |
| Texture/feature fetches | depth RSM 1024²/cubemap face，other RSM 64²；large scene最多4 cascades/2048²；具体face数、discard后的VPL layout和fetch count未报告。 | [P §4.2–4.3, §6.1] |
| Precision/quantization | 未报告PyTorch/TRT precision、TF32/FP16、TensorRT engine settings或DLSS settings。 | [P Table 1 caption][A video 08:20] |
| Hardware/backend/coherence | `Ours`/CNSR/AE以PyTorch测量，caption只明确CNSR/AE在RTX A6000；ONND/OIDN为RTX4090上Falcor RTCore path tracing + denoise；`Ours (TRT)`的GPU型号没有在同一句中明确绑定。 | [P Table 1 caption] |
| Time/FPS/latency | PyTorch Ours：87.89–147.91 ms（四scene）；TRT Ours：22.90–45.82 ms；Interior Design TRT 38.19 ms。视频current-frame overlays会变化，例如Chess 00:20为25.2 ms，Interior 08:40为41.2 ms；视频值不是formal aggregate。 | [P Table 1][S Table 1][A video 00:20, 08:40] |
| Resolution/upscaling | training/reference与正式网络inference为512×512；公开视频明确写“Infer at 512×512, DLSS 768×768”。main Table 1没有说明是否计入DLSS或最终display resolution。 | [P §6.1][A video 08:20] |
| Precompute/prepare/amortization included? | per-scene约50 h训练不计runtime；Table 1没有清楚拆分G-buffer/RSM/VPL generation、network、TensorRT和upscaling，也没有说明首次engine build/I/O。 | [P §6.1, Table 1][C unavailable] |

在固定 `P` pixels、`L` lights、VPL数和RSM resolution后，单帧工作量是有限的；但论文没有给产品级 `L_max`，并明确总成本随 lights 线性增长，因此不能把“支持可变灯数”自动写成与本项目相同的静态 shader budget。[P §7][I]

## 9. 实验 protocol、baseline、指标与结果

Table 1 的 formal reference来自§6.1的512×512/2048 spp path tracing；质量指标是 LPIPS 与 Relative Square Error（RSE），但RSE公式、颜色空间、tone mapping、逐图/逐pixel聚合、置信区间均未报告。下表每格为 `time ms / LPIPS / RSE`，只在论文自己的 protocol 内解释。[P §6.1–6.2, Table 1]

| Scene | CNSR | AE | Ours (PyTorch) | ONND | OIDN | Ours (TRT) | 论文内结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Chess | `171.37/.0326/.0125` | `115.80/.0274/.0040` | `98.48/.0222/.0023` | `22.08(42spp)/.0209/.0012` | `22.29(32spp)/.0181/.0013` | `22.90/.0222/.0023` | Ours优于neural baselines，但同时间denoisers质量略优。[P Table 1] |
| Gig | `171.60/.2798/.1781` | `132.83/.3295/.2985` | `147.91/.1338/.0125` | `46.55(25spp)/.1998/2.0660` | `46.36(22spp)/.1441/1.2590` | `45.82/.1338/.0125` | Ours在该大/复杂scene显著保持更多detail；7 lights使其本身也最慢。[P Table 1, §6.2] |
| Emernald Square | `171.28/.4021/.3589` | `118.63/.3010/.1386` | `87.89/.1600/.0394` | `26.64(13spp)/.4652/.0668` | `25.96(10spp)/.3491/.0557` | `25.26/.1600/.0394` | Ours在低spp denoising难以覆盖的10.68M-triangle scene更优。[P Table 1, §6.2] |
| Living Room | `174.19/.1766/.0351` | `120.37/.1823/.0189` | `113.72/.1820/.0142` | `28.01(35spp)/.1406/.0056` | `27.85(28spp)/.1412/.0052` | `27.47/.1820/.0142` | Ours优于/接近neural baselines，但denoisers质量更优。[P Table 1] |

其他正式结果：

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| Complex luminaire | Chess把area lights替换为透明外壳内18 light bulbs；equal-time 23 ms；reference 8192 spp/92 s | ONND 22 spp | LPIPS | Ours `0.3751`，ONND `0.4843`；作者以复杂灯具/短时path budget解释差异 | [P Fig. 5, §6.2] |
| Interior Design | 5 movable furniture、3 area lights；与main同类baselines | CNSR、AE、ONND、OIDN | time/LPIPS/RSE | Ours PyTorch `107.72/.0583/.0070`；TRT `38.19/.0583/.0070`；ONND `38.09(66spp)/.0470/.0124`；OIDN `37.99(57spp)/.0440/.0100` | [S Table 1, Fig. 1] |
| Editable axes | Chess改变light distribution；Gig改变wall color；video连续操纵camera、objects、lights/material controls | reference components | visual only | direct/indirect components随编辑变化；无数值编辑保真或out-of-range protocol | [P Fig. 7][A video] |
| Novel action/object | Gig novel actions、unseen character；Emernald Square未进训练的tree/bus做translate/rotate/scale/move | CNSR visual baseline | visual only | Ours看起来更合理；没有跨scene checkpoint测试或quantitative metric | [P §6.4, Fig. 9][A video 10:40–11:20] |
| Temporal stability | dynamic supplemental sequences；ONND 8.0 明确启用 temporal 与 kernel-based extensions，OIDN 2.2.2 只明确为 GPU-accelerated | ONND/OIDN | visual only | 作者报告Ours较少flicker；没有temporal warping error、sequence LPIPS、history ablation或两种denoiser各自的history配置 | [P §6.2][A video] |
| Classic VPL | Mitsuba Instant Radiosity；2048/8192 VPL，clamping 0/0.3；正文只把 IR timing 明确绑定到 RTX4090 | Ours 23 ms（该值的GPU未在 supplemental 中重新绑定） | time + visual | 8192 VPL/23 s无clamp有bright blotches；clamp减blotch但丢能量；2048 VPL/6 s仍不理想 | [S §3, Fig. 2] |

不同表行的GPU/backend不同，且Table 1未报告统计聚合与runtime breakdown；这些数值不能直接与local neural material query time、其他分辨率或其他论文FPS做排名。[P Table 1 caption][N `docs/realtime_material_compilation.md`][I]

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `author-negative` | 在最终hybrid loss中加入AE式DSSIM | 在作者cases中“does not yield any performance gain” | 未进一步解释 | 不能据此推广为DSSIM对所有scene transport无效；protocol/weight也未报告 | [P §6.1] |
| `ablation-inferior` | average pooling替代pixel-light attention，并给decoder加层以接近capacity | Gig：L1/RSE/LPIPS=`.0137/.0141/.1442`，final=`.0132/.0125/.1338` | 每pixel不同lights重要性不同，attention能自适应组合 | 参数量并未精确matched；只在Gig单scene验证 | [P §6.3, Table 2, Fig. 8] |
| `author-negative/runtime` | full 840D light embedding同时作K和V | 作者称计算约为只用256D direct embedding作K的3× | direct part已足够提供attention key | 没有quality/timing table或projection实现，不能复算3× | [P §5.1] |
| `ablation-inferior` | 去掉half vector | `.0135/.0126/.1473`；高光变blur | half vector降低specular mapping复杂度 | 只证明当前scene/model/protocol内有效 | [P §6.3, Table 2, Fig. 8] |
| `ablation-inferior` | 去掉light direction | `.0143/.0146/.1417` | 失去per-pixel light orientation clue | LPIPS并非所有指标都单调；不得写成全面失败 | [P Table 2] |
| `ablation-inferior` | 去掉shadow clues | `.0151/.0234/.1624`；self-shadow无法恢复 | direct visibility是高频信息，G-buffer alone不足 | 是最强的clue ablation，但仍只在Gig | [P §6.3, Table 2, Fig. 8] |
| `ablation-inferior` | double-size unified decoder直接预测final radiance，L1训练，不做GI decomposition | `.0135/.0127/.1428`，劣于final | 分组件inputs/supervision简化学习 | loss与decoder结构同时变化，不是单轴因果证明 | [P §6.3, Table 2, Fig. 8] |
| `author-negative` | 高spp/一般性对照 | Interior中高spp path tracing + denoising的LPIPS更低；Chess/Living equal-time亦更好 | path tracing是成熟、更一般方法 | LightFormer优势依赖scene complexity压缩了equal-time spp，而不是普遍支配 | [P §6.2, Table 1][S §2, Table 1] |
| `author-negative` | high-frequency indirect shadow | grass上的indirect shadow呈direct-shadow-like高频pattern，不是期望的soft shadow | complex indirect visibility；当前indirect decoder缺visibility clues | 与“indirect VPL无explicit visibility test”直接对应 | [P §4.2, §7, Fig. 10] |
| `author-negative` | highly glossy/mirror reflection | Fig. 10不能准确恢复mirror reflection | 可能需要second-bounce G-buffer、AE data importance sampling或indirect-VPL attention | 当前13D G-buffer+low-res indirect observation不足以确定极高频多跳路径 | [P §7, Fig. 10] |
| `known-limitation` | lights增加 | Gig 7 lights latency变差；encoding cost随pixels×lights线性 | 建议light culling、VPL clustering、advanced shadowmaps | attention没有消除per-light observation成本 | [P §6.2, §7] |
| `author-negative/baseline` | classic Instant Radiosity无clamp/有clamp | 无clamp bright blotches；clamp显著能量损失；可接受质量需大量VPL与秒级时间 | neural shader隐式学习，避免classic IR artifacts | 该对照不是unbiased many-light或现代VPL clustering的全面否定 | [S §3, Fig. 2] |

未公开作者的模型开发日志、failed architectures、optimizer sweeps或seed variance；除上述明确文本/正式消融外，不从最终结构推测“作者试过并失败”。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Bibliography | main有正式DOI与2024 TOG卷期，但页脚Article 1 | Vol.1/No.1和占位DOI | 无 | 以main DOI/publisher metadata为身份；排版占位不改变方法版本。 |
| Architecture | Fig. 2–3、§4–5给模块与attention dataflow；正文定义Q=`g(x)`、K=`z_d`、V=`z`，但Fig.3在value projection前还画出未解释的concat | Fig.3/Table2给VPL/clue/decoder widths与activations | 无 | MLP/UNet/decoder基本可重建；attention concat/投影、skip operator和参数量不可完整重建。 |
| Skip connection | P只说decoder类似prior pixel generator | prose称direct/indirect “encoder”有skip，图却把skip画在Direct/Indirect Decoder | 无 | `source ambiguity`：报告按图登记decoder skip，但不猜concat/add。 |
| Data/query | 4 scenes、20k/100、VPL/RSM/GT配置 | 增加Interior Design和classic VPL comparison | 无scene/data | randomization ranges、split manifest、scene licensing与exact seeds不可审计。 |
| Loss/training | Eq.8、Adam 1e-4、batch4、50h/2×A6000 | 无新增 | 无 | scheduler、steps、VGG layers、log边界与checkpoint selection缺失。 |
| Runtime/export | Table1给PyTorch/TRT times；未拆RSM/network/upscale | Interior给同类times | 无TensorRT engine/export | precision、hardware binding、latency scope与workspace未知。 |
| Assets/evaluation | main figures/table | full-resolution viewer、Interior、IR | viewer只有结果images/data.js，不是训练数据 | viewer支持视觉核查，不能替代raw metric scripts/ground truth manifest。 |
| Temporal | 正文声称temporally consistent、clear inputs与pipeline带来stability | video提供动态序列 | 无 | 公开architecture没有列出history/reprojection，但VPL sample复用/RNG与buffer缓存策略也未报告；没有temporal metric，结论仍是作者visual evidence。 |

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. **高频 indirect transport**：复杂indirect visibility、mirror/highly glossy reflection不能准确恢复；direct shadow clue没有对应地进入indirect decoder。[P §7, Fig. 10]
2. **复杂 transport未覆盖**：subsurface scattering、caustics、volumetric rendering等没有实验；作者只说framework“theoretically”有潜力，并承认需要新clues和额外工作。[P §7]
3. **light-count scaling**：encoding成本随pixels和lights线性；当前attention只让所有lights共享一次decode，没有消除所有per-light work。[P §7]
4. **跨场景 generalization未证实**：§6.4只在单scene training set内测试新动作/对象，并说large-scale scene dataset preparation成本过高；path tracing不受这种训练集限制。[P §6.4]
5. **production boundary**：作者明确称rasterization/path tracing更一般，复杂材料与transport的production readiness不足。[P §1, §8]
6. **area-light point proxy**：mean direct-VPL position只是approximation；作者报告大多数cases无可见影响，但未给error bound或针对large/near-field emitters的压力测试。[P §4.2–4.3]

### 12.2 未报告/材料不可得

- official code、config、checkpoint、scene assets、training data、reference split manifest、metric scripts；
- per-scene是否独立50 h、总steps/epochs、schedule、Adam β/epsilon/decay、seed、initialization、model selection；
- VGG feature layers、preprocessing、log transform数值边界、output clamping/linear radiance units；
- attention value路径中concat的另一输入、projection widths、normalization、softmax/residual细节、decoder skip operator、精确parameter/MAC；
- PyTorch/TRT precision、TensorRT engine/export settings、TRT具体GPU、Table 1是否包含G-buffer/RSM/VPL/DLSS；
- weights、per-light embeddings、RSM/VPL buffers与peak workspace bytes；
- RSE公式、LPIPS版本、color/exposure/tone mapping、per-image aggregation、variance/confidence interval；
- temporal metric、frame sequence split、camera/object velocity distribution、direct-VPL sample跨帧复用/RNG与RSM/VPL/clue刷新/缓存策略；
- 每类light/VPL的exact sample recipe、indirect VPL selection、discard rate、randomization numeric range与out-of-distribution protocol；
- cross-scene single-checkpoint或leave-one-scene-out实验；
- author talk/slides/correction。项目页公开视频为无音轨supplementary demo，不是method talk。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

LightFormer不是把完整scene压进一个global latent，而是把容量分成三层：[I]

1. **runtime physical observations**承载大量“当前场景事实”：高分辨率depth RSM给direct visibility，低分辨率RSM channels给indirect distribution，G-buffer给当前pixel geometry/material，half vector/light direction给高频坐标；
2. **per-light encoders + attention**学习哪些light observations对当前pixel重要；
3. **scene-trained weights**学习这组scene/material/animation distributions中，clues到full radiance components的剩余映射。

因此“light-oriented”降低的是显式object parameter vector与逐object composition的负担，不等于网络没有scene prior。20k configurations/scene、50 h训练以及mirror/indirect-shadow失败共同说明：成功来自**足够确定的clues +受控scene distribution**，不是小网络凭空求解一般rendering equation。[P §4–7][I]

### 13.2 成功所依赖的假设

- 当前几何、材质和灯都能以干净、稳定、及时的G-buffer/RSM重新rasterize；screen/off-screen transport的重要信息能从light view覆盖；[P §3–4][I]
- direct visibility与specular peak可由显式clues降低学习维度；indirect transport大体低频，64² position/normal/flux仍足够；[P §4.3][I]
- 测试变化留在单scene训练分布附近；未见对象仍共享相似material/geometry/light-transfer statistics；[P §6.4][I]
- light数较小，per-light cubemap/UNet/attention的线性成本尚能接受；[P §6.1–§7][I]
- 2048-spp component GT、50 h/scene训练和direct/shadow/indirect分解足以把scene-specific mapping拟合到实时decoder。[P §6.1][I]

### 13.3 可迁移机制与不能迁移的部分

可迁移到未来 scene-transport 轨道的机制包括：[I]

- 按light而非object构造observation，并在latent space做一次pixel-wise composition；
- 把direct shading、visibility与indirect transport拆开监督和诊断；
- 让高频visibility使用高分辨率depth，低频indirect attributes使用低分辨率buffers；
- 用half vector/light direction等物理坐标提示代替只扩网络；
- 明确把per-light encoding与single decode分开benchmark，暴露light-count scaling。

不能直接迁移到local material compiler的部分包括RSM、scene position、occlusion、final-image loss、per-light attention和scene-specific image decoder。它们改变的是scene visibility/integration，不是source material自身 `evaluate(wo,wi)`。把这些量塞进material latent会破坏source/native semantics、random access和renderer/material责任边界。[N `docs/realtime_material_compilation.md`][I]

### 13.4 与本项目 runtime contract 的关系

LightFormer可在固定image size、`L_max`、VPL数、RSM resolution与network shape后形成一个有界的**scene image program**，但论文没有冻结`L_max`或资源bytes，并明确cost随lights增长。它不满足当前local contract：没有随机访问 `evaluate(wo,wi)→linear f`，没有同一shading state的多个`wi`复用，也没有matched `sample()/pdf()`。[P §5–7][N `docs/realtime_material_compilation.md`][I]

更合适的角色是未来 environment/multi-light/scene-transport renderer 的 architecture candidate 或 capacity diagnostic；它不是当前 NVIDIA neural material的产品候选、teacher BRDF或sampler proposal。其component GT与intermediate outputs值得借鉴，因为它们能把最终图像错误定位为direct、visibility还是indirect，而不是把所有失败混进一个LPIPS。[I]

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

当前 NVIDIA identity 的核心是：局部8D latent、learned frames、逐方向evaluator输出cosine-weighted response并由公共adapter返回bare linear `f`，以及11D条件的analytic-proposal sampler；项目还要求`prepare()`缓存同一shading point可复用的view-conditioned state。[N `.trellis/tasks/08-25-03-neural-baseline-and-candidate/research/nvidia-method-correspondence.md`]

| 对齐项 | 判定 | 影响 |
|---|---|---|
| Output/query semantics | `not-applicable` | LightFormer输出final scene radiance image，不是local `f`；不能用其LPIPS/RSE判NVIDIA evaluator质量。 |
| Visibility | `intentional-deviation` | 当前material contract把scene occlusion留给renderer；LightFormer把RSM/shadow clue作为核心输入。两者职责不同，不是当前实现漏组件。 |
| Multi-light amortization | `interface-adaptation` | LightFormer“per-light encode、latent compose、single decode”提示未来scene renderer可避免每灯完整image decoder；当前NVIDIA则应保持material `prepare`一次、多`wi`/多灯重复`evaluate`的既有合同。 |
| Half-vector/light-direction clues | `not-applicable` to faithful baseline | NVIDIA learned frame和20D方向输入由其自身论文定义；不能因LightFormer消融而向faithful baseline加入half-vector clue。若研究，只能注册独立candidate并matched预算。 |
| Component decomposition | `interface-adaptation` | 可在未来environment integration评测中分direct/material response、visibility、indirect；不应把final-image component loss混入当前bare-`f` formal training identity。 |
| History/reprojection | `not-applicable` | LightFormer公开的输入/architecture未列history，但其VPL sample复用等temporal implementation也未披露。visual temporal claim不能证明当前NVIDIA local evaluator或PT sequence稳定性。 |
| Suspected defect | `not-applicable` | 本文没有提供能指认当前NVIDIA evaluator/sampler correspondence defect的证据。 |

最直接的项目启示是**保持两个identity**：local material program继续只负责source scattering；若以后实现LightFormer式scene surrogate，注册独立scene-level method bundle、独立GT与time/memory domain，不把final-image improvement回写成NVIDIA材质复现成功。[N `docs/realtime_material_compilation.md`][I]

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：未来scene renderer中，per-light embedding + pixel-light attention在多灯时比average pooling更好地保持quality | Gig Table2的attention ablation | Light importance的pixel dependence在本项目scene set中同样存在 | 同一G-buffer/RSM/VPL、相同training data/optimizer/parameter class；attention vs capacity-matched average pooling | scene split、light count/distribution、buffer resolutions、GT spp、precision、decoder | component/final error、temporal warp error、median/p90 ms、bytes，按light count分层 | scene-level，固定`L_max`后有界 | 多scene/seed下quality无显著改善，或matched quality下time/bytes更差且随L斜率不可接受 |
| H2：direct/shadow/indirect分解监督比single final-radiance decoder更容易定位并降低错误 | Eq.7–8与Gig `w/o GI decomposition` | 本项目未来scene GT能稳定导出同measure components | 同一总参数/MAC、相同inputs；3-head component model vs single-head final model；loss transform matched | GT decomposition、tone/exposure、data/query、steps/seed、hardware | 各component error、final error、edge/highlight recall、training variance | scene-level decoder | 分解在matched预算下不改善任何component/final metric，或component定义导致final合成bias |
| H3：高分辨率depth + 低分辨率indirect attributes是比all-high更好的scene-observation Pareto点 | P采用1024² depth与64² P/N/flux；mirror/indirect shadow仍是失败 | visibility高频、indirect attributes相对低频在目标scenes成立 | all-low vs split-resolution vs all-high，网络/VPL/count完全相同 | scene/camera/light sequence、RSM projections、GT spp、training、precision | direct-shadow edge error、indirect error、mirror/highlight recall、RSM ms/bytes | scene-level fixed-resolution pipeline | split方案在matched bytes/time下不优，或indirect高频长尾显著恶化且all-high能恢复 |
| H4：只用current-frame clean physical buffers，可在不依赖history时获得更可控的动态稳定性 | LightFormer公开architecture未列history且视频声称较少flicker；但无formal metric，VPL跨帧采样策略也未报告 | 本项目scene buffers的determinism与coverage足以避免history ghosting | current-frame LightFormer-like surrogate vs equal-time low-spp+temporal denoiser；另加同network+history ablation | scene sequence、motions、spp/time、camera path、exposure、resolution、hardware | per-frame quality、warped temporal error、flicker spectrum、disocclusion error、latency/state bytes | scene-level current-frame vs temporal pipeline | current-frame在equal-time下temporal/quality都不占优，或RSM变化产生同等flicker且history显著改善 |
| H5：对多灯先做bounded culling/top-k，再attention，可把LightFormer线性light成本压到产品上界 | P §7明确light-count bottleneck并建议culling | 少量被选灯的latent能代表多数pixel贡献，且discard error可被显式控制 | full-light attention vs固定`k` culling+attention；保持per-light encoder/decoder不变 | `L`分布、culling rule、`k`、GT、training、resolution、hardware | error vs L/k、missed-light energy、temporal popping、ms/bytes | scene-level，固定k静态有界 | quality随L增长显著退化、temporal popping不可控，或culling成本抵消attention节省 |

这些假设都是后续scene-transport任务的report-only研究输入，不是当前NVIDIA formal config的hard gate，也不授权在本任务中实现或扩大训练预算。[N `.trellis/spec/project/research-execution.md`][I]

## 16. 证据索引

### `P` Main paper

- p.1–2、§1：问题、light-oriented动机、贡献、production边界；
- §3、Eq.1、Fig.1：camera-ray scene rendering domain与两阶段pipeline；
- §4.1：direct VPL fields、emission importance sampling、expected-power normalization、light types；
- §4.2：neural RSM、indirect VPL fields、area-light proxy、cubemap/single-texture与无indirect visibility test；
- §4.3、Eq.2、Figs.2–3：7-channel shadow clue、RSM resolutions、half-vector/light-direction clues；
- §5、Eqs.3–7：pixel-light attention、8 heads、Q/K/V选择、component decoders与final composition；
- §6.1、Eq.8、Fig.4：四scene、20k/100、500/2000 VPL、512²/2048spp GT、CSM、loss、Adam/batch/time；
- §6.2、Table1、Figs.5–7：baseline protocol、完整结果、complex luminaire、editable axes与runtime；
- §6.3、Table2、Fig.8：attention/clue/decomposition消融；
- §6.4、Fig.9：同scene的novel action/object generalization及单scene dataset声明；
- §7、Fig.10：indirect shadow、mirror/complex transport与light scaling限制；
- §8：结论及rasterization/path tracing更一般的声明。

### `S` Supplemental

- §1、Fig.3、Table2：五encoders、两decoders、逐层width/activation/input/output；
- §2、Fig.1、Table1：Interior Design变量、视觉与量化结果；
- §3、Fig.2：Mitsuba Instant Radiosity的VPL count/clamping/time/artifacts。

### `C` Official code/config/data

- 不可得；项目页无code/config/data入口，公开repository定向检索未找到正式实现。不得用第三方复现补成作者配置。

### `A` Author project/video/viewer

- project page：正式作者入口、main/supp/video/viewer与citation；
- video 00:00–11:37：无音轨动态演示；00:20 Chess performance，08:20声明512² inference + DLSS 768²，08:40 Interior performance，10:40–11:20大scene对象/相机变化；
- viewer `data.js`：Chess/Gig/Emernald Square/Living Room/Interior Design的Reference/Ours/CNSR/AE/ONND/OIDN color与RSE/1-SSIM/FLIP images；它不提供raw metrics、训练数据或代码。

### `N/I`

- `docs/realtime_material_compilation.md`：local `prepare/evaluate/sample/pdf`语义和source/runtime边界；
- `.trellis/tasks/08-25-03-neural-baseline-and-candidate/research/nvidia-method-correspondence.md`：当前NVIDIA evaluator、learned frame、sampler与训练identity；
- 第13节：capacity、success assumptions、可迁移/不可迁移机制与runtime class；
- 第14节：对当前NVIDIA的`not-applicable/interface-adaptation/intentional-deviation`判定；
- 第15节：五个scene-level matched、可证伪假设。

## Evidence review

```text
author_worker: /root/lightformer2024
reviewer: /root/lightformer2024_review
reviewed_at: 2026-08-29
sources_rechecked:
  - author main PDF, SHA-256 B9214208DB8154B03885E1EDF6E057EA41E857053A964186517CC1C81E45C470
  - author supplemental PDF, SHA-256 67C27A84017D852E2FDDDB44FDCFF8D649B5F565D6A3E23FA2CD3E84BB569F65
  - author project page, public video and comparison viewer assets
  - public exact-title/code search and DOI/bibliographic metadata
findings_closed:
  - preserved the supplemental prose-versus-diagram encoder/decoder skip ambiguity
  - recorded the unexplained attention value-path concatenation instead of inventing its operand or width
  - changed temporal claims from deterministic/no-history assertions to the actual disclosed-input boundary
  - recorded VPL/RSM refresh, cache and cross-frame RNG policy as unavailable
  - narrowed log-transform placement, TensorRT hardware/precision and runtime-scope claims to what sources disclose
  - separated ONND temporal settings from the less-specified OIDN setup and normalized the classic-IR negative classification
remaining_evidence_gaps:
  - official code/config/checkpoint/data/metric scripts unavailable
  - attention concat/projection, decoder skip operator, exact parameters/MAC/bytes unavailable
  - training steps/schedule/seeds/model selection and per-scene timing scope unavailable
  - TensorRT precision/GPU/runtime breakdown, VPL/RSM temporal update policy and temporal metric unavailable
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
