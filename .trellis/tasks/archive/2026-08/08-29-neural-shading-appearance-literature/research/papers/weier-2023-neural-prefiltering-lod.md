---
paper_id: "weier-2023-neural-prefiltering-lod"
title: "Neural Prefiltering for Correlation-Aware Levels of Detail"
authors: "Philippe Weier, Tobias Zirr, Anton Kaplanyan, Ling-Qi Yan, Philipp Slusallek"
year: "2023"
venue: "ACM Transactions on Graphics 42(4), Proceedings of SIGGRAPH 2023, Article 78"
doi: "10.1145/3592443"
report_status: "evidence-reviewed"
main_source: "https://sites.cs.ucsb.edu/~lingqi/publications/paper_neural_lod.pdf"
supplemental_status: "available"
official_code_status: "audited"
official_code_commit: "2035107a1b221386f1ab18a51acaef111a3122ac"
author_worker: "/root/taming2026"
reviewer: "/root"
last_verified: "2026-08-29"
---

# Neural Prefiltering for Correlation-Aware Levels of Detail

## 1. 研究对象与报告边界

本文研究的是**资产/场景级的相关性感知外观预过滤与离散 LoD 表示**：把带几何、纹理和材质的高细节资产体素化，在七个离散尺度上，用一个 appearance network 近似体素内部平均 throughput，用另一个 visibility network 近似体素边界两点之间的可见性；运行时仍通过稀疏体素遍历与 Monte Carlo 路径延续生成图像。[P pp. 1–2, 5–9, Fig. 4]

本报告覆盖作者公开的 16 页 TOG/SIGGRAPH 2023 正文、4 页 supplementary、作者/机构/Intel/ACM 入口、官方 1 分 21 秒结果视频，以及官方 `WeiPhil/neural_lod` 代码。代码固定在 2026-08-29 的 `main` HEAD `2035107a...`; 该 HEAD 相对 2023 初始提交只改动 README 中额外场景下载链接，算法代码仍是初始公开版本。[A/C source ledger]

方法边界必须先钉死：

- 它不是局部材质 `f(wo,wi)` codec。appearance 输出是某一**空间体素**在 far-field 假设下、已经聚合几何遮挡和多次散射的平均 throughput `φ_V(ω_o,ω_i)`；visibility 还需要体素边界两端点。[P §3.2–§4.3]
- 它不是确定性的 neural renderer。一次完整像素路径可能遍历可变数量体素、调用可变次数 visibility network，并在碰撞后随机采样新方向、调用 appearance network。[P Eq. 4, §4.5; C `wavefront_neural_throughput_visibility_lod.cu`]
- 它不解决连续像素 footprint、各向异性 footprint 或跨 mip 连续插值；论文实验按七个离散 LoD 手工/外部选择一个层级，release UI 的 `current lod` 也是全局离散选择。[P Fig. 16 caption; C `neural_lod_learning.h`:21–37, `neural_lod_learning_ui.cu`:101–111]
- 它没有把 source parameters 编译为共享材质 program，也没有测试未见资产、未见材质或未见参数状态；每个资产分别训练两张网络与阈值网格。[P Tables 1–2]
- 它没有提供与 appearance evaluator 匹配的 learned `sample()/pdf()`；正式运行时均匀采样球面方向，作者把 importance sampling 明列为未来工作。[P Fig. 17, §6 discussion; C `wavefront_neural_pt.cuh`:205–250]

因此，本报告将其归为 `scene-transport / asset-prefilter`，并把它对本项目的意义限定在 `prepare()` 中的层级 latent/feature filtering、位置条件化、相关性保留和质量—时间—内存评测设计；不会把它自动升级为 material compiler。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---:|---|---|
| Main paper `P` | [UCSB author PDF](https://sites.cs.ucsb.edu/~lingqi/publications/paper_neural_lod.pdf)，16 页；DOI `10.1145/3592443` | 2026-08-29 | SHA-256 `5CE8C794EE3C9347AE349458973211D6FA9EA4640140C2A1E1D25C8091B839D1` | 方法、公式、正式实验；正文所有页面、图、表、图注已视觉核对 |
| Supplemental `S` | [UCSB supplemental PDF](https://sites.cs.ucsb.edu/~lingqi/publications/supplementary_neural_lod.pdf)，4 页 | 2026-08-29 | SHA-256 `CAF2CB2DC05D475B4F616609B223B6889035DA90CBEBF51066220C6BFFB17B0A` | 复杂照明与 City Houses 追加结果；4 页逐页视觉核对 |
| Official code/config `C` | [WeiPhil/neural_lod](https://github.com/WeiPhil/neural_lod)，`main` | 2026-08-29 | commit `2035107a1b221386f1ab18a51acaef111a3122ac`；archive SHA-256 `B74F60880C033665BC2643444BC00253F5A9FE747E0ED69B69C05851112EBE51` | 逐路径 runtime、训练 query、网络构造、Arcade formal example；MIT 主许可，另含第三方许可 |
| Official config `C` | `neural_lod/scenes/arcade/arcade_world_final.ini` | 2026-08-29 | SHA-256 `011540ACDA0C3EDF132F6BE6DA533BF121D54A68977402DFEA74098AA2CDB5E1` | 唯一随 repo 发布的完整 scene config；是 release formal example，不等同于论文所有场景配置 |
| Serialized config/weights `C` | `arcade_world_final_512_throughput_config.json` 与两份 weights JSON | 2026-08-29 | throughput config SHA-256 `6F547DEF34FD1468DBBD35B9A4C1C0C119D194CE7988017FE1CC97469641BBBC`；weights 中 `n_params=3,814,400/3,800,064` | 核对 Arcade appearance 配置与两网参数量；visibility config JSON 未发布 |
| Official data `C/A` | repo 内 Arcade scene/纹理/预训练权重/七层阈值；README 链接的 [Google Drive 额外场景](https://drive.google.com/drive/folders/1Vc5gxQ-qmquV4LGHQgDFRDvzydhjN4_K?usp=sharing) | 2026-08-29 | repo 内文件受锁定 commit 覆盖；Drive bundle 未逐文件下载/hash | Arcade 可审计；其余论文场景仅确认官方入口可用，未锁定整个外部 bundle |
| Author page/talk/video `A` | [Saarland project record](https://graphics.cg.uni-saarland.de/publications/weier-2023-neural-lod.html)、[Lingqi Yan 页面](https://lingqiyan.github.io/)、[Intel media page](https://www.intel.com/content/www/us/en/developer/articles/technical/neural-prefiltering-for-correlation-aware.html)、[ACM record](https://doi.org/10.1145/3592443)、[official 1:21 video](https://github.com/lingqiyan/lingqiyan.github.io/releases/download/ucsb-archive/publications__video_neural_lod.mp4) | 2026-08-29 | 本地 author video SHA-256 `87F63F484F9F9DDDE972F125CF9C8203DD0D228648D8D0C5BB88178A15C35612`，30,874,521 bytes，1:21 | 元数据、官方入口、时间稳定性/结果演示；ACM 161.93 MB presentation locator 存在，但直接下载返回 403，未以其补事实 |
| NeuralShading evidence `N` | `docs/realtime_material_compilation.md`、`docs/material_scope.md`、`docs/research/experiment_framework.md`、`docs/learning.md`、当前 NVIDIA config/实现 | 2026-08-29 | repo-local | 只用于 §§13–15；不回填为 2023 论文事实 |

作者旧项目 deep link `https://weiphil.github.io/portfolio/neural_lod` 当前返回 404，但作者 portfolio 根页、Saarland record、Lingqi Yan 页面和 Intel media page 仍相互印证论文身份。ACM supplementary presentation 的下载接口 `action/downloadSupplement?...papers_728_VOD.mp4` 可定位但返回 403；这影响 talk 逐帧复核，不影响已由正文、supplemental 与 release code 闭合的技术事实。没有发现官方 erratum/correction。

## 3. 原论文的问题、假设与贡献边界

作者的问题设定是：当一个高细节资产投影到少量像素时，原始几何/材质路径追踪成本和存储仍随源复杂度增长；经典 mesh/volume LoD 又常把几何、法线、BRDF、density 等分解后分别过滤，有限解析模型难同时保留结构化表面与体积式 aggregate detail，尤其容易丢失沿射线的可见性相关性并产生 light leak。[P §1–§2]

核心假设有四层：

1. 空间可划分为轴对齐体素，LoD 由 `512^3 → 256^3 → … → 8^3` 的七层稀疏网格表示；高分辨率活跃体素向低分辨率聚合。[P §4.1; C `neural_lod_learning.h`:21–37]
2. 完整体素内 transport `T_V(p_o,ω_o;p_i,ω_i)` 太高维，粗体素也保留端点位置会挤占固定容量，因此 appearance 接受 far-field 近似，移除 `p_o,p_i`，学习 `φ_V(ω_o,ω_i)`。[P Eq. 2–3, §3.3]
3. appearance 去位置后会损失多次间接散射的部分 positional correlation；为避免把同样近似施加到遮挡，另学 point-to-point `V_M(p_o,p_i)`，再通过硬阈值维持一次 ray segment 内的相关事件。[P Eq. 4, §4.3–§4.4]
4. 两个小网络跨所有体素与 LoD 共享，通过 multi-resolution hash grid 存局部容量；每体素再存一个 classifier threshold 控制 light-leak/漏遮挡取舍。[P Fig. 4–10]

作者声明的贡献包括：

- 用 generalized voxel path space 从 path integral 中推导 appearance/visibility 分解，而不是先选一个经典 microflake/SGGX 参数化；[P §3]
- 位置无关的 5D appearance field（体素位置 + 两方向）与位置到位置的 6D visibility field；[P §4.2–§4.3]
- 对 visibility 设计共享参数的 recurrent hash-grid encoding，并用每体素 weighted F-score 阈值减少 correlated light leaks；[P Fig. 7–10, Eq. 5–6]
- GPU online noise-to-noise 训练、多 LoD 采样、体素边界 projected-area query 和 0.5% domain extrusion；[P §5]
- 用连续 visibility 作为保守 Russian roulette 信号，并在 wavefront 中反复 compact，降低空体素/可见体素遍历成本。[P §4.5]

作者没有声称：零样本新资产泛化、动态场景、连续 footprint filtering、局部 BRDF 替换、确定性单次 query、或完整的 importance sampler。

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material/scene input | 完整 PBR asset：几何、材质、纹理；先对 asset AABB 做多层稀疏体素化 | 每资产独立；LoD 0 `512^3` 到 LoD 6 `8^3` | [P §4.1, Tables 1–2; C `neural_lod_learning.h`:21–37, README]
| Appearance runtime query | 体素中心 `c(V)`、出射方向 `ω_o`、入射/下一跳方向 `ω_i` | normalized object/AABB position `R^3`；两方向位于 `S^2`，代码先映射到 `[0,1]^3` | [P Fig. 5, §4.2; C `wavefront_neural_pt.cuh`:205–245]
| Appearance output | 平均体素 throughput `φ_V(ω_o,ω_i)`，RGB | `R^3`；不是 bare local BRDF，也不单独输出 cosine/PDF | [P Eq. 3–4, §3.3]
| Visibility runtime query | 一条当前 ray 与体素边界的 entry/exit 端点 `p_i,p_o` | 两个 normalized object-space position，共 6 scalars；作者比较后采用 object/global coordinates | [P §4.3, Fig. 7–8; C `wavefront_neural_pt.cuh`:125–202]
| Visibility output | segment 无遮挡概率，运行时与 per-voxel threshold 比较为硬二值；可选 stochastic threshold 直接把概率当碰撞概率 | sigmoid scalar `[0,1]`；正式主方法为 binary decision | [P §4.3–§4.5, Eq. 5–6; C `wavefront_neural_throughput_visibility_lod.cu`:295–330]
| Direction coordinates | 方向在 voxel/object axes 中表示，3D Cartesian 输入后作 SH encoding | `ω∈S^2`; release 传入 `(ω+1)/2` 以满足 tiny-cuda-nn SH 输入约定 | [P §4.2; C `neural_lod_learning_kernels.cuh`:145–225]
| LoD/footprint | 体素中心在七层网格中唯一，作者称中心隐式编码 LoD；评测选择“voxel roughly pixel-sized”的离散层 | 7 层离散；未报告连续 footprint、层间 blend 或各向异性 footprint | [P §4.2, Fig. 16 caption; C `neural_lod_learning_ui.cu`:101–111]
| Runtime path state | camera ray、当前 ray origin/direction、path throughput、当前 voxel entry/exit、RNG、bounce | wavefront path tracer；每像素可变步数 | [P Fig. 4, Eq. 4, §4.5; C `wavefront_neural_throughput_visibility_lod.cu`:467–690]
| Validity/domain restrictions | 静态 asset；训练时将 voxel boundary 向外扩 0.5%，推理仍查原边界 | geometry/material edit 需重新训练相应表示；动态内容未支持 | [P §5.1, Fig. 13, §6 limitations]

`φ_V` 的精确定义不是“把 BRDF 对端点简单平均”。Eq. 3 的分子对体素边界端点的完整 intra-voxel transport `T_V` 乘两端 projected-area measure 积分，分母用从 `p_o` 沿 `-ω_o` 进入体素时真正命中几何的可见 projected area 归一化。因此它把体素内部 visibility、BSDF、多 bounce 和几何项聚合为一个类似 BSDF、但语义属于 aggregate voxel 的方向函数。[P Eq. 2–3]

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

预处理流程为：

1. 把 source asset voxelize 到最高 `512^3` 稀疏 occupancy，并逐层聚合到 `8^3`；NanoVDB 每个 active voxel 最终存 occupancy/threshold。[P Fig. 4; C `neural_lod_learning_init.cu`:160–380]
2. online 采样最高层 active voxel，再采 LoD 并映射到 ancestor voxel；从体素边界/方向域生成两网 query。[P §5.1]
3. 对 appearance query，在选定 `p_o,ω_o` 处进入原始 asset，递归 path trace；每个内部 hit 对固定 `ω_i` 做 shadow test 并累加贡献，BSDF importance sampling 继续内部路径，直到离开体素。[P §5.2, Fig. 14]
4. 对 visibility query，直接对 `p_o→p_i` 线段做原始几何 occlusion ray，形成二值 label。[P §5.1]
5. 同一 asset 的所有体素/LoD 联合训练一个 appearance network 与一个 visibility network；随后逐 active voxel搜索 weighted-F 最优阈值并写入 sparse grid。[P §4.2–§4.4]
6. runtime 从 camera ray 找下一个 active voxel，生成 entry/exit visibility query。若网络判定无碰撞就步进到下一体素；若判定碰撞，则在球面均匀采样 `ω_i`、查询 appearance、乘 `4π`，用该方向延续 path；直到 ray 逃逸或到达环境/灯。[P Eq. 4, Fig. 4; C `wavefront_neural_pt.cuh`:205–265]

这条数据流把“预过滤”解释为用一次 appearance inference 代替该体素内部可能很长的原始 surface path，而不是把图像/局部材质预先 bake 成固定照明颜色。

### 5.2 持久化表示

每资产持久化：

- 七层 sparse NanoVDB voxel grids；每个 active voxel 一个 float threshold。Arcade release 的七个 threshold `.bin` 共 `4,079,576` bytes，即 `1,019,894` 个 float，和正文 Table 2 的 `4.08 MB` 十进制口径一致。[P Table 2; C release asset]
- appearance network 的一个 3D multi-resolution hash encoding、两个 SH encodings 和一个 MLP；
- visibility network 的一个被两个端点重复调用、参数共享的 3D hash encoding和一个 MLP；
- 全部 network parameters 使用 half precision 存储；论文 Table 2 的 bytes 与 `2×parameter_count` 对应。[P §6, Table 2]

它没有 per-voxel dense latent tensor；容量主要在 shared hash table。`S=2^19` 时，2/4/8 features per level 分别决定约 3.8M/7.56M/15.04M 参数。由于两个网络的 hash tables 相互独立，visibility/appearance features-per-level 可不同。[P Table 2]

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Appearance voxel encoding | normalized `c(V)∈R^3` | 7-level HashGrid，base resolution 8，level scale 2，`S=2^19`；2/4/8 feat/level 按场景 | linear interpolation；无 normalization 报告 | `7×feat` spatial features | 同一资产/所有体素/LoD共享；网络与 hash per-asset | [P §4.2, Fig. 5, Table 2; C `arcade_world_final.ini`:179–188]
| Appearance direction encodings | normalized Cartesian `ω_o,ω_i` | 两个独立 SphericalHarmonics encoding；paper 写“coefficients up to degree 8”，release 设置 `degree=8`，tiny-cuda-nn 实际输出 `degree^2=64` channels | 固定解析 basis | release 为 64+64 direction features | 固定 basis，无训练参数 | [P §4.2; C `arcade_world_final.ini`:192–199; C tiny-cuda-nn `spherical_harmonics.h`:393–397]
| Appearance MLP | spatial + two direction encodings | FullyFusedMLP，4 hidden layers × 128 neurons，RGB head | hidden ReLU；output activation None | 3 scalars | per-asset | [P §4.2, Fig. 5; C `neural_lod_learning_init.cu`:580–660]
| Visibility repeated encoding | normalized object-space `p_i,p_o` | `RepeatedComposite(n_repetitions=2)` 对两个 3D point 复用同一个 7-level HashGrid；base 8、scale 2、`S=2^19`，2/4/8 feat/level | linear interpolation | concatenated endpoint features | encoder weights在两端共享；整网 per-asset | [P §4.3, Fig. 7; C `neural_lod_learning_init.cu`:820–895]
| Visibility MLP | 两端 shared-encoding features | FullyFusedMLP，4 hidden layers × 128 | hidden ReLU；output Sigmoid | 1 visibility probability | per-asset | [P §4.3; C `arcade_world_final.ini`:228–251]

Arcade release serialized weights 给出 exact total parameter counts：appearance `3,814,400`、visibility `3,800,064`；按两位小数应约为 3.81M/3.80M。正文 Table 2 却写 appearance 3.80M、visibility 3.81M，恰好相反。这是一个很小但真实的 paper↔release 数值 gap；正文 topology 与 release topology仍逐项对应，现有证据不能判定是论文表格对调、release版本漂移还是不同统计口径。[C weights JSON; P Table 2]

方向 encoding 另有一个命名口径 gap：按通常球谐记号，“up to degree 8”会包含 `l=0…8` 共 81 个系数；本文固定的 tiny-cuda-nn 接口把参数 `degree=8` 定义为 8 个 band，输出 `8²=64` 个通道，即实现到 `l=7`。因此复现 release 应使用 64+64 方向通道，但报告不能把 paper 的自然语言悄悄改写成“最高阶 7”，也不能以 paper 措辞改成 81+81 后仍沿用 release identity。[P §4.2; C serialized config and `spherical_harmonics.h`:393–397]

### 5.4 条件化、坐标变换与物理先验

- **Transport/visibility decomposition** 是最重要的先验：appearance 承担体素内聚合散射，visibility 保留一条端点线段的空间相关性。[P Eq. 4]
- **Far-field approximation** 主动删除 appearance 的 boundary positions。这减少高维容量，但作者明确承认它损失 multi-bounce indirect transport 的 positional inter-voxel correlation。[P §3.3]
- **Object-space visibility**：相邻/不同 LoD 体素共享边界点坐标，使相同几何遮挡模式可跨层共享；作者用 Fig. 7 展示 local-voxel coordinates 更易丢关联。[P §4.3, Fig. 7–8]
- **Implicit LoD coordinate**：HashGrid 的 7 个 resolution 与 7 个 LoD 对齐；voxel center 被作者用于隐式识别层级，不另输入 one-hot LoD。[P §4.2]
- **Hard threshold** 把 continuous classifier 输出变成 correlated binary event。weighted F-score 对 occluded class 用 `β=2`（更重 recall、减少 false negative/light leak），visible class用 `β=1`，再按两类样本数加权。[P Eq. 5–6]
- 没有 learned frame、half/difference、analytic BRDF core、per-material parameter encoder、mip interpolation或 view-conditioned latent preparation。

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Dataset/source assets/scenes | 主实验 Arcade、Bay Cedar、Rover、Pandanus、Fractal；Bay Cedar/Pandanus 来自 Disney Moana Island，Rover 由 vajrablue 提供。比较另用 Oak Tree、Bare Tree 与 Bako glossy tree；supp 增加 16.8M-triangle、1,026 MB City Houses | [P Fig. 16, acknowledgments; S pp. 3–4]
| GT/reference | 原始 source geometry/material path tracing；appearance training 对原始体素内部 geometry做 unbiased MC path estimator；visibility label 是端点线段 ray test | [P §5.1–§5.2]
| Train/validation/test split | 没有资产级 train/test split；每个资产单独优化。classifier threshold 使用每体素采样 test set，但是否与 network training RNG/样本严格独立、validation checkpoint selection 均未报告 | [P §4.4]
| Voxel/LoD sampling | 先均匀采 LoD 0 active voxel，再均匀采 LoD，右移 Morton index 找 ancestor；因此粗体素按其包含的高分辨率活跃体素数获得更多样本 | [P §5.1; C `neural_lod_learning_kernels.cuh`:25–100]
| LoD distribution | 正文称 LoD uniform；代码实现有 `pdf_strength/pdf_shift` 的可调 shifted-square distribution，Arcade formal config 均为 0，退化为 uniform | [P §5.1; C `neural_lod_learning_kernels.cuh`:31–69, `arcade_world_final.ini`:141–146,201–206]
| Boundary segment sampling | 均匀采 `ω_o∈S^2`；在与 `ω_o` 投影为正的 cube faces 中按 `dot(n,ω_o)` 选 face，再均匀采 face point `p_o`；沿 `-ω_o` 与对面边界交得到 `p_i`，从而使体素内部 segment density 均匀 | [P Fig. 12, §5.1]
| Appearance directions | `ω_o,ω_i` 均匀球面；`p_o` 若沿 `-ω_o` 未命中体素内几何，则 appearance 样本拒绝 | [P §5.2]
| Appearance MC target | 从 accepted `p_o,ω_o` 向体素内 path trace；内部方向用 source BSDF importance sampling；每个 hit 对固定 `ω_i` 做 shadow ray，累计可达边界的 contribution；标准 albedo RR 无偏截断 | [P Fig. 14, §5.2, §6 training times]
| Visibility target | appearance rejection 中命中/未命中的 `p_o` 都可转成 visibility label；任意边界 segment 用原始几何 ray test 得二值 GT | [P §5.2]
| Filtering/LOD/footprint | 七个 voxel resolutions 是唯一 LoD/filter axis；训练所有层联合，未使用相机 pixel footprint、cone、anisotropic filter 或跨层 stochastic/trilinear blend | [P §4.1–§4.2, Fig. 16; C release UI]
| Domain regularization | training voxel extent 向外均匀扩最多 0.5%，最小 dilation release 为 `1e-4`；inference 仍查询原始边界 | [P Fig. 13; C `arcade_world_final.ini`:201–212]
| Augmentation/distillation/teacher | 无 teacher/distillation；noise-to-noise 直接以 online unbiased MC estimate 训练 | [P §5]
| Online/offline generation | network training query online GPU path/ray tracing；voxelization 与 threshold optimization 是 offline precompute；不存训练 corpus | [P Fig. 4, §5]

release 的自定义 scene 路线有限：README 提供 Blender→`.vks` exporter，但 exporter 仅导出 default material color，任意 Blender node graph 未被承诺；voxel point cloud 依赖修改过的 cuda-voxelizer 与 trimesh。因此 repo 足以运行/训练 Arcade 和基本自定义资产，不等于公开了论文五场景的完整、原生材质转换管线。[C README]

## 7. Loss、optimizer 与训练 lifecycle

| 项 | 正式配置 | locator |
|---|---|---|
| Target transform/output transform | **论文**未报告 target transform。**Arcade release config** `learn log space throughput=1`：target 为 `log(1+RGB)`，推理 `exp(pred)-1` 后 clamp `>=0` | [C `arcade_world_final.ini`:141–150; `neural_lod_learning_kernels.cuh`:239–264; `wavefront_neural_pt.cuh`:256–273]
| Appearance loss | relative L2；noise-to-noise on unbiased MC target | [P §4.2, §5; C serialized throughput config]
| Visibility loss | binary cross entropy | [P §4.3; C `arcade_world_final.ini`:228–237]
| Optimizer | 正文两网 Adam，learning rate `0.005`；Arcade release补充 `β1=.9, β2=.99, ε=1e-15, l2_reg=1e-8` | [P p. 11; C `arcade_world_final.ini`:151–178,213–241]
| Optimizer wrappers | appearance: EMA decay `.95` 外包 exponential decay→Adam；visibility: exponential decay→Adam，无 EMA | [C `neural_lod_learning_init.cu`:580–610,800–840]
| LR schedule | 正文未报告；release decay start 8,000、interval 4,000、base `.33` | [C `arcade_world_final.ini`:151–178,213–241]
| Batch/query count | 正文未报告；Arcade release 两网均 `262,144` candidate queries/step；appearance 会剔除无有效几何命中的样本后按 tiny-cuda-nn granularity compact | [C `arcade_world_final.ini`:141–150,201–212; `neural_lod_learning.cu`:502–590]
| Steps | 正文 Table 1 称 8k–12k steps 收敛，并据各场景给出 total time；Arcade release 两网 `max training steps=30,000` | [P Table 1 caption; C `.ini`:147,206]
| Stages | voxelization → joint all-LoD appearance/visibility optimization（两网独立 objective）→ per-voxel threshold search → runtime；材质变化而几何不变时只需部分重训 appearance | [P Fig. 4, p. 11]
| Initialization | 网络/HashGrid初始化分布未报告；release tiny-cuda-nn default 构造。代码 `default_rng_t rng{1339}` 存在，但训练 query LCG 与模型初始化/训练 seed 的完整 correspondence 未报告 | [C `neural_lod_learning.h`:270–300]
| Model selection/repeats | checkpoint selection、early stop、validation protocol未报告；作者只说 8k–12k convergence，并明确 Pandanus 多次重训结果漂移 | [P Table 1, p. 13]
| Hardware | 单 NVIDIA RTX 3080；tiny-cuda-nn scalars half precision；release 构建测试 CUDA 11.3、driver 525.85/535.129、CC 7.5/8.6 | [P p. 11; C README]

正文 Table 1 的完整训练成本如下，所有数值均是单 RTX 3080；`100 steps / total`：

| Scene | Visibility | Appearance | Total |
|---|---:|---:|---:|
| Arcade | 2.52 s / 5.04 min | 1.08 s / 2.16 min | 7 min 12 s |
| Bay Cedar | 5.48 s / 10.96 min | 4.93 s / 9.86 min | 20 min 49 s |
| Rover | 3.12 s / 6.24 min | 1.66 s / 3.32 min | 9 min 33 s |
| Pandanus | 6.13 s / 12.26 min | 4.41 s / 8.82 min | 21 min 4 s |
| Fractal | 8.11 s / 16.22 min | 12.03 s / 24.06 min | 40 min 16 s |
| City Houses（supp） | 5.16 s / 8.60 min | 7.72 s / 12.86 min | 21 min 28 s |

[P Table 1; S Table 2]

这里有两个不能自行消解的 paper↔release gap。第一，`log1p` 是 release Arcade formal setting，但正文把目标直接表述为 average throughput 的 MC estimate + relative L2，没有披露变换；不能把 release 设置追溯成所有论文结果。第二，30k 是 release “max steps”，而正文说 8k–12k 收敛并以此计算 Table 1 totals；不能用 30k 重算论文训练时间，也不能认定论文跑满 30k。

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path | 每条 active ray反复：找下一个 occupied voxel → 1 次 visibility inference → hard threshold；visible则继续下一 voxel，occluded则采方向并做 1 次 appearance inference/一次 scatter；直到 escape/light/max depth | [P Fig. 4, Eq. 4, §4.5; C `wavefront_neural_throughput_visibility_lod.cu`:467–690]
| Call frequency | 每次 network invocation topology 固定，但每像素/每路径 visibility 次数与 occupied voxels、LoD、方向、threshold有关；release 给每 bounce 最大 visibility inference bound `3×resolution`，再由 max depth形成一个很松的静态上界 | [C `wavefront_neural_throughput_visibility_lod.cu`:489, `.ini`:100–110]
| Parameter count | 场景与 feat/level 决定：单网约 3.8M/7.56M/15.04M；Arcade exact appearance 3,814,400、visibility 3,800,064 | [P Table 2; C weights JSON]
| MAC/FLOP | 正文与代码未给端到端 MAC/FLOP；FullyFusedMLP 层固定，但 hash encoding、可变调用次数和 wavefront compaction使单像素成本非定值 | [P/C]
| Shared/per-asset bytes | 两网+七层 sparse grid 均 per-asset；无跨资产 shared decoder。总计 19.31–131.44 MB（五主场景） | [P Table 2]
| Texture/feature fetches | 正文未报告实际 GPU fetch count/cache traffic。算法上 appearance每次查 7-level 3D hash一次，visibility对两端共享参数但执行两次 7-level lookup；不能据此把整条 path写成固定读取 | [P Fig. 5,7; C config]
| Precision | tiny-cuda-nn network scalars half precision；voxel thresholds release 为 float32 binary；没有 INT8/FP8/quantization ablation | [P p. 11; C assets]
| Backend/coherence | CUDA + tiny-cuda-nn FullyFusedMLP + NanoVDB；要求 CC≥7.5；wavefront 每批 visibility 和每次 scatter 后 compact | [P §4.5, p. 11; C README]
| Runtime direction sampling | 主方法均匀球面，weight 乘 `4π`；release另有 Henyey–Greenstein可选开关但 Arcade formal为 off，论文不把它作为贡献 | [P Fig. 17; C `wavefront_neural_pt.cuh`:205–250, `.ini`:100–110]
| Visibility RR | 以 continuous non-collision probability为 survival lower bound；最高 `512^3` diagonal 只允许最多 0.001% rays被终止，对应单 voxel约 99.25% survival | [P §4.5]
| Latency/FPS | Fig. 15 只给曲线：1024²、128 spp、不同 max depth，声称跨分辨率 interactive→real-time及最高/最低 LoD最多约 5×；未给可逐项读取的精确 FPS表 | [P Fig. 15, p. 11]
| Isolated traversal latency | Arcade LoD 2 高像素分辨率、各渲染10秒：无 stochastic traversal `83.5 ms/frame`，有 `41.2 ms/frame` | [P Fig. 11]
| Converged-image time | Fig. 16 固定 512²、1024 spp，7层每场景总渲染 8–96 s，详见 §9；这些不是单次 interactive frame latency | [P Fig. 16 caption]
| Precompute included? | runtime time不含 voxelization、两网训练、20–120 s threshold search和 weights/grid load；memory table含两网+grid，不含训练态/原始 source常驻 | [P §4.4, Tables 1–2]

Table 2 的正式存储结果：

| Scene | Vis feat/params | App feat/params | Grid sparsity LoD0/6；bytes | Total | Original | Saving |
|---|---:|---:|---:|---:|---:|---:|
| Arcade | 2；3.81M / 7.63 MB | 2；3.80M / 7.60 MB | 99.43% / 71.48%；4.08 MB | 19.31 MB | 68.28 MB | 71.72% |
| Bay Cedar | 8；15.04M / 30.09 MB | 4；7.56M / 15.12 MB | 98.71% / 78.71%；8.93 MB | 54.14 MB | 752.92 MB | 92.81% |
| Rover | 4；7.54M / 15.10 MB | 4；7.56M / 15.12 MB | 98.62% / 72.07%；9.45 MB | 39.67 MB | 187.41 MB | 78.83% |
| Pandanus | 8；15.04M / 30.09 MB | 4；7.56M / 15.12 MB | 98.91% / 54.69%；8.60 MB | 53.81 MB | 1.40 GB | 96.16% |
| Fractal | 8；15.04M / 30.09 MB | 4；7.56M / 15.12 MB | 87.74% / 29.69%；86.23 MB | 131.44 MB | 1.92 GB | 93.15% |
| City Houses `S` | 8；15.04M / 30.09 MB | 4；7.56M / 15.12 MB | 98.89% / 67.58%；7.54 MB | 54.14 MB | 1026.26 MB | 94.72% |

[P Table 2; S Table 1]

## 9. 实验 protocol、baseline、指标与结果

主要质量指标为 FLIP perceptual difference；Fig. 16 对每个 512²、1024 spp converged image 排除空背景后取 mean FLIP。没有 source-state bootstrap、置信区间或多 seed aggregation；Pandanus 重训波动是定性披露，不是误差条。摘要的“around 25% quality improvements”没有披露聚合公式、baseline集合或置信区间，本报告不从各图反推一个新的 25% 口径。[P abstract, pp. 11–13]

### 9.1 五场景全部 LoD

下面每个 cell 为 `mean FLIP / render seconds`；秒数是固定 512²、1024 spp 完整图，不是帧延迟。

| Scene | L0 | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Arcade | .0687 / 71 | .0618 / 52 | .0559 / 40 | .0480 / 26 | .0387 / 17 | .0341 / 11 | .0407 / 8 |
| Bay Cedar | .1520 / 96 | .1022 / 84 | .0826 / 72 | .1152 / 52 | .0982 / 32 | .0815 / 18 | .0438 / 11 |
| Rover | .1894 / 50 | .1556 / 36 | .1517 / 26 | .1307 / 18 | .1071 / 15 | .0490 / 12 | .0426 / 9 |
| Pandanus | .1899 / 74 | .1843 / 71 | .2055 / 70 | .0987 / 65 | .1706 / 43 | .0908 / 28 | .0447 / 18 |
| Fractal | .0903 / 57 | .0857 / 49 | .1168 / 37 | .0973 / 24 | .0708 / 15 | .0557 / 12 | .0388 / 11 |

[P Fig. 16]

该表支持“更粗 LoD 通常更快”，不支持“质量随 LoD 单调更好/更差”：Bay Cedar、Pandanus、Fractal 均有层间反转，作者将一部分归因于训练难度、能量损失与场景频率结构。[P pp. 13–14]

### 9.2 复杂照明与大场景 supplementary

supplemental 的非均匀环境照明 mean FLIP（按其展示顺序 L6→L0）：

| Scene | L6 | L5 | L4 | L3 | L2 | L1 | L0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Arcade | .0369 | .0328 | .0302 | .0278 | .0243 | .0209 | .0227 |
| Bay Cedar | .0302 | .0210 | .0196 | .0260 | .0279 | .0284 | .0251 |
| Rover | .0475 | .0419 | .0431 | .0407 | .0432 | .0299 | .0413 |
| Pandanus | .0641 | .0689 | .0827 | .0619 | .0690 | .0501 | .0256 |
| Fractal | .0712 | .0656 | .0827 | .0879 | .0908 | .1019 | .1078 |
| City Houses, overcast | .0443 | .0369 | .0361 | .0384 | .0387 | .0367 | .0453 |
| City Houses, backlight | .0384 | .0329 | .0309 | .0308 | .0276 | .0222 | .0240 |

[S pp. 1–3]

这组结果说明 representation 并未 bake 固定光照，能在环境贴图变化下继续 path trace；但它不是训练/测试 lighting split 的统计泛化实验，也没有公布 exact envmap、exposure 与 camera locator。

### 9.3 与 Hybrid Mesh-Volume LoD 比较

Fig. 18 在两树资产、LoD 1/3/5 比较 Loubet et al. 2017 Hybrid LoD，指标为 FLIP：

| Asset | LoD | Ours | Hybrid LoD |
|---|---:|---:|---:|
| Oak Tree | 1 / 3 / 5 | .0944 / .0897 / .0604 | .1968 / .2315 / .2236 |
| Bare Tree | 1 / 3 / 5 | .0445 / .0593 / .0195 | .1352 / .1029 / .0924 |

Oak Tree 中 Ours 为 22.64 MB voxel grid + 37.72 MB networks = 60.36 MB；Hybrid mesh/textures 3.72 MB + volumetric data 79.22 MB = 82.94 MB，作者报告约 27% storage 改善。Ours 没有与 Hybrid 做 equal-time、equal-bytes和相同 estimator variance 的完整三重 matched control，因此这些数值只能支持论文给定 protocol 下的质量/存储比较。[P Fig. 18 and text]

### 9.4 与 Deep Appearance Prefiltering 比较

Fig. 19 在 glossy tree、LoD 1、方向光照下把 appearance training max depth限制为 2 来模拟 direct-light-only设定：Ours FLIP `.0645`，Bako et al. `.0836`。Ours 两网均 8 feat/level，作者称整个 pipeline 单 RTX 3080 少于一小时；对方实现由 Bako 团队提供 early access，论文引用其训练为 256×V100 cluster 上 0.5–2 days。由于实现、hardware、训练目标和资源不 matched，结论限于该图的质量与预处理量级，不能横向形成硬件效率排名。[P Fig. 19, acknowledgments]

### 9.5 Correlation-aware threshold 与 traversal

| Experiment | Protocol | Result | 结论边界 | locator |
|---|---|---|---|---|
| Fixed 0.5 vs optimal | Arcade LoD0，高像素分辨率，强 `[6,4,2]` RGB constant illumination | fixed 出现多处结构化 light leak；optimal 修正 | 定性图，无像素统计表 | [P Fig. 10]
| Stochastic traversal | Arcade LoD2，各10 s | 83.5→41.2 ms/frame，同时间 stochastic variance更低 | 这是 RR traversal，不是 stochastic visibility threshold | [P Fig. 11]
| Optimal vs probability threshold | 两种 LoD1 crop；`S=2^21/2^19` | surface-like: `.1072/.1246/.1111/.1337`；volume-like: `.2475/.2341/.2559/.2512`（optimal/stochastic依次） | stochastic 在 volume-like 可更好；optimal 对一般表面更稳健 | [P Fig. 20]
| Hash collision | visibility `S=2^19` vs `2^25` | 增大 table可减少 collision，但示例 reconstruction quality不再明显提升；per-voxel threshold可补偿一部分 collision | 不是“阈值提升 representation capacity”；它改变 decision boundary | [P Fig. 9]

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `ablation-inferior` | across-all-LoD uniform voxel sampling | 比“先采 LoD0 active voxel再映射 ancestor”收敛差 | 粗层 active voxel代表的细层 occupancy不同，均匀 coarse voxel权重不合适 | 这是 query measure 的差异，不是网络 topology 单独收益 | [P §5.1]
| `author-negative` | non-uniform LoD distributions | 不同场景收益显著变化 | 选定的 uniform LoD scheme跨场景更稳健 | release仍保留 `pdf_strength/shift`，说明它是实验轴而非被删除功能 | [P §5.1; C kernels]
| `ablation-inferior` | 两个边界点分别均匀采样（排除同 face） | voxel内部 point/segment density不均匀 | 应按 uniform direction 的 projected area采 boundary | 会静默改变 loss measure；复现不能只写“uniform boundary points” | [P Fig. 12]
| `ablation-inferior` | visibility 使用 voxel-local endpoint coordinates | 跨相邻体素/LoD 难共享相同端点几何信息 | object-space positions更能共享 exact/partial visibility | 对本地材质 latent不可直接照搬：材质 UV过滤与 scene position correlation不是同一量 | [P Fig. 7–8]
| `author-negative` | global visibility threshold 0.5 | structured surface出现 correlated light leaks | classifier imbalance与 hash collision使统一阈值不适合所有 voxel | 阈值优化降低决策错误，不提升 continuous probability表示质量 | [P Fig. 9–10]
| `ablation-inferior` | stochastic visibility probability替代 binary optimal threshold | volume-like crop有时更好，surface-like更差且可能漏光；增大 hash table才较可用 | 随机方法仍保留部分 learned correlation，但完全 correlated hard threshold对一般场景更 robust | 是 bias/variance与memory三方交换，不能一概记作失败 | [P Fig. 20, pp. 14–15]
| `author-negative` | 不做 stochastic traversal RR | Arcade LoD2 83.5 ms/frame，10 s同预算方差更高 | 长无遮挡路径需要过多 visibility calls | 这是 execution strategy，不是表示精度消融 | [P Fig. 11]
| `known-limitation` | runtime 无 appearance importance sampling | 高细节 LoD per-pixel variance高 | 均匀采 aggregate distribution，粗 LoD 时方差下降 | 缺 `sample/pdf` 是部署功能缺口 | [P Fig. 17]
| `author-negative` | Pandanus 多次重新训练 | 不同 run 的受影响 LoD改变；部分层 transported energy不守恒 | visibility/appearance联合误差与高频几何/材质难平衡 | 论文没有 seed/CI，不能把某次最好图当稳定结果 | [P p. 13]
| `known-limitation` | SH degree 8 对 glossy/high-frequency | Rover poster/快速 reflectance变化、锐利高光难重建 | 需 all-frequency direction encoding和定制 importance sampling | 失败源含 direction encoding与query coverage，不能只扩大 MLP | [P pp. 13–14]
| `paper-code-gap` | Arcade `log1p` throughput target | 正文依赖 unbiased MC target 的 noise-to-noise叙述；release先对单次/noisy MC estimate做 `log1p` | 作者未讨论 | 非线性变换后不再有 raw-throughput 空间的无偏性保证；这是不同 objective，不直接判为implementation bug | [P §5.2; C `neural_lod_learning_kernels.cuh`:239–264]
| `paper-code-gap` | visibility config export button | code 将“Write visibility config”仍传入 `throughput_network`; repo只含 throughput config JSON | 作者未讨论 | 明确 release defect；visibility topology仍可从 `.ini` 与构造代码重建 | [C `neural_lod_learning_ui.cu`:251–256]

已获得第一方材料没有披露逐次失败的网络宽度、层数、SH degree、hash level数量或 loss family search；不能从最终 `4×128` 反推出这些历史尝试。

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Architecture | 两个独立 `4×128` ReLU MLP；appearance HashGrid+两个“up to degree 8”SH，visibility shared/recurrent endpoint HashGrid | 无新增 topology | 构造代码和 Arcade config逐项对应；tiny-cuda-nn `degree=8` 为64 channels/方向 | topology `corroborated`；paper 的 degree 自然语言与 code 的 band-count 约定存在命名口径 gap，release identity固定64+64 |
| LoD | 7 层 `512^3→8^3` | City Houses同 protocol | constants明确 `NUM_LODS=7`、max power 9；`lod_levels` 静态字符串虽保留到 LoD8，UI循环只枚举7层，setter也 clamp | 正式运行只7层 |
| Data/query | online full domain；uniform LoD；projected-area boundary；0.5% outward | 只给结果 | kernels对应；另有 `pdf_strength/shift`，Arcade置0；min dilation `1e-4` | 正式 Arcade退化为 paper uniform；额外旋钮不证明论文使用 |
| Appearance target | unbiased MC throughput + relative L2；未写 transform | 未写 | Arcade `.ini`启用 `log1p` target，runtime `expm1`+nonnegative clamp | **未闭合冲突/遗漏**：release setting不能冒充所有 paper runs |
| Optimizer | Adam lr .005；8k–12k convergence | City Houses time | config补 `β=.9/.99, eps=1e-15,l2=1e-8`、EMA only appearance、decay schedule、30k max | 细节是 code evidence；30k max不等于论文实际 steps |
| Batch | 未报告 | 未报告 | Arcade两网 262,144；invalid appearance rows compact | code-only formal example |
| Threshold samples | 100 thresholds in `(0,1]`，1,000 samples/voxel；20–120 s | 未报告 | algorithm写死101 values `[0,1]`；Arcade config 10,000 samples/voxel，最终 clamp `[0,.9]` | **未闭合冲突**：paper experiment与release example不一致；不得任选其一“修正”另一方 |
| Runtime sampling | 无 importance sampling，均匀球面 | 视频/复杂照明结果 | uniform sphere + `4π`; HG可选但 formal off | paper一致；HG是release附加实验开关 |
| Stochastic mechanisms | RR traversal与 stochastic threshold是两个不同实验 | 无 | 两个独立 flags；Arcade formal RR on、stochastic threshold off | 对应清楚，报告不得混名 |
| Storage | Table 2 rounded counts/bytes | City Houses追加 | weights exact counts；threshold bins与4.08MB吻合 | bytes/grid闭合；Arcade两网 exact parameter counts与正文3.81M/3.80M顺序相反，未闭合 |
| Assets | 五主场景+比较资产 | City Houses | repo仅 Arcade 完整；Drive宣称其余scene+weights | partial release；外部 bundle未逐文件hash |
| Evaluation scripts | 512²/1024 spp quality；1024²/128 spp performance | 追加quality | README scripts默认 `ref_spp=512,spp=4096`，envmap example `1024/16384` | 脚本默认不是论文 Fig. 16 protocol，运行时需显式匹配 |
| Visibility config artifact | topology有正文 | 无 | UI export path写错 network，且正式 JSON缺失 | `release-code defect/paper-code-gap` |

repo HEAD `2035107a...` 相对 initial commit `8c109987...` 只有 README 一行 scene-download URL替换，因此上述 code gaps不是 2025 README更新带来的算法漂移。[C GitHub commit API]

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

1. **far-field appearance 的位置相关损失。** `φ_V` 删除 `p_i,p_o`，所以多 bounce indirect scattering 跨体素的位置相关性并未完全保留；separate visibility只保住直接 segment visibility，不能恢复被平均掉的 indirect transport。[P §3.3]
2. **LoD尺度之外的长程相关性。** 尚未在所选体素尺度预解的 long-range correlation 仍使 path length不均、需要原 runtime继续追踪。[P p. 11]
3. **能量不守恒与训练不稳定。** Pandanus 某些 LoD在不同 rerun中发生能量损失/精度波动；作者没有解决多层联合训练的energy balancing。[P p. 13]
4. **高频方向容量不足。** SH encoding可能无法表达锐利 glossy highlights；作者建议 hash/tri-planar/octahedral/all-frequency parameterization。[P pp. 13–14]
5. **训练和运行时 importance sampling缺失。** 训练难覆盖高频 angular configurations；runtime均匀采样导致高 detail方差。[P Fig. 17, §6]
6. **动态内容未支持。** geometry改变会使 visibility/grid失效；动态 neural representation只被列为未来方向。[P p. 14]
7. **prototype traversal低效。** 作者明确说尚不能与 raw real-time path tracing性能竞争，并建议融合 visibility inference和voxel stepping。[P p. 11, p. 14]
8. **预处理仍 per-asset。** 虽比hours/days方法快，但两网仍各需分钟，复杂 Fractal总计40min16s；threshold另需20–120s。[P Table 1, §4.4]
9. **存储不是恒小。** 高密度 Fractal grid达86.23MB、总131.44MB；固定 hash table只限制网络，不限制 active voxel threshold总量。[P Table 2]

### 12.2 未报告/材料不可得

- 所有资产的 exact source locator、版本、camera、environment maps、exposure/tone mapping与材质转换参数；
- 网络初始化、训练 seed、每实验 rerun数、checkpoint选择、validation split与统计置信区间；
- 论文全部场景的 batch size、optimizer β/ε、EMA、decay schedule、是否启用 `log1p`；release只闭合 Arcade example；
- exact voxelization occupancy criterion、几何边界膨胀前后的 source-scale单位和各资产AABB locator；
- Fig. 15 曲线的 machine-readable timing、kernel分解、network inference占比、内存带宽、能耗与CPU成本；
- 单次 appearance/visibility query的精确 MAC/FLOP、hash fetch/cache transaction、寄存器/occupancy；
- LoD自动选择、跨层过渡、temporal hysteresis、anisotropic pixel footprint与连续 filtering；
- 与 Bako/Hybrid 对照的完全 equal-time/equal-bytes/equal-quality configuration；
- ACM 161.93MB presentation内容：locator已确认，但直接下载403；
- Drive extra-scene bundle逐文件版本/hash。repo内 Arcade与其权重/threshold已锁定，但其余论文场景尚不能形成逐资产不可变 source ledger。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

容量不是主要放在 `4×128` MLP，而是在每资产的两张 multi-resolution hash tables 和 sparse thresholds：Table 2 中 network参数随 feat/level从3.8M到15.04M增长，Fractal又因grid不稀疏增加86.23MB。MLP更像共享解码器；hash table承担“哪个空间/LoD位置长什么样”，threshold grid承担离散 visibility校准。

对当前本地材质目标，真正可迁移的类比是“`prepare()` 根据 footprint从层级空间表示取少量状态，`evaluate()`只解码方向”；但本文 release没有把 hash-grid feature预取成可复用 `prepare` state，每次 appearance query仍执行完整HashGrid+MLP。若照搬整网，既没有本项目的固定纹理读取合同，也没有 view-conditioned reuse。

### 13.2 成功所依赖的假设

- asset静态且可在object AABB内体素化；
- screen-space远处允许把每体素内部 transport近似为far-field方向函数；
- coarse voxel的真实高维位置相关性可以牺牲，而最致命的single-segment light leak另由visibility classifier保留；
- source complexity主要来自大量细几何/多 bounce，而不是必须保留的可编辑native material graph；
- RTX上可把不小的hash tables与wavefront batches摊销到coherent inference；
- 每资产离线训练和数十MB runtime存储可接受。

这些假设与本项目“保留原生source参数/编辑能力、局部随机访问bare `f`、小型shader MLP”的主问题不同。尤其，scene voxel center不能替代material state/UV latent；`φ_V` 也不能作为local reference GT，因为它已经聚合geometry visibility和多次transport。

### 13.3 可迁移机制与不能迁移的部分

可迁移：

- **分层 spatial feature + 方向 decoder 的接口分工。** 可把空间层级选择、过滤、latent读取放进 `prepare()`，方向 MLP放进 `evaluate()`；
- **query measure必须按目标积分设计。** projected-area segment sampling表明“看似均匀的边界点”会改变训练measure；本项目同理需要冻结solid-angle、grazing与source state sampling；
- **把连续预测质量与离散决策校准分开。** threshold optimization降低visibility错误，不等于representation quality提升；这提醒 sampler accept/reject或classification proposal也应单独评估calibration；
- **全LoD联合训练的负迁移检查。** Pandanus表明共享多尺度容量可能造成某些层能量漂移，应按层/footprint报告而非只报总体均值；
- **wavefront compaction/coherence评测。** 如果未来将latent filtering用于批量路径，prepare/read成本必须按coherence分层测量。

不能直接迁移：

- object-space point-to-point visibility不是local scattering evaluator；
- `φ_V` 包含geometry/multiple scattering，不能写入本项目bare-`f` ABI；
- per-asset 3.8M–15M网络和最多131MB表示超出当前小shader预算；
- 均匀球面runtime sampling没有提供本项目所需matched `sample/pdf`；
- 七层体素LoD没有证明未见材质/状态G2泛化，也没有native parameter compiler证据。

### 13.4 与本项目 runtime contract 的关系

| 合同 | 本文状态 | 项目判定 |
|---|---|---|
| `prepare()` latent获取/footprint过滤 | 没有显式prepare；LoD全局离散选择，hash lookup在每次network query内 | 可作为层级feature预过滤研究线索，不是现成实现 |
| `evaluate(wo,wi)→linear f` | appearance输出aggregate voxel throughput，不是bare local `f` | ABI语义不兼容 |
| 随机访问 | 给定voxel和方向可随机查询appearance；visibility给端点可查 | 单次网络可随机访问；完整像素仍依赖可变遍历 |
| 固定读取/静态有界 | 单次MLP/HashGrid topology固定；完整path调用次数可变，release仅以 `max_depth × 3×resolution` 给出很松的worst-case上界 | 有理论上界但不是固定读取的local evaluator，也不满足当前shader预算；只可把单query当capacity diagnostic |
| `sample()/pdf()` | 无appearance matched sampler/PDF；只均匀球面或release HG开关 | 不满足需要材质驱动采样的完整产品合同 |
| 未见材质/状态 | 每资产两网，无holdout | 不提供G2/G2s/跨资产compiler证据 |

最合适的角色是 `asset-prefilter` 参考、`prepare`/LoD机制启发和 `capacity diagnostic`；不是当前 product evaluator、teacher GT或source compiler。若用于teacher，也只能教“某静态asset在某voxel尺度的aggregate transport”，不能教source-local scattering。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

当前仓库 NVIDIA identity 是 `nvidia-rta2024-functional-f@2`；它的 evaluator直接输出bare linear `f`，`prepare()`读取/过滤hierarchical z8 latent并复用view-conditioned state，训练以GPU online source `evaluate().f`为target；sampler走当前learned proposal的forward-KL route。[N `docs/learning.md`:3–9; `docs/realtime_material_compilation.md`:3–7]

| 本文机制 | 当前 NVIDIA 复现 | 分类 | 影响 |
|---|---|---|---|
| 7-level voxel HashGrid隐式LoD | NVIDIA是两张RGBA latent textures与footprint LOD/filter，不是scene voxel field | `not-applicable` to fidelity；可作 `budget-adaptation` candidate | 只能另建matched latent hierarchy候选，不能改写faithful z8 layout |
| far-field aggregate throughput `φ_V` | 当前输出source-local bare `f`，不含geometry visibility/transport | `not-applicable` | 把本文target接入NVIDIA evaluator会违反output measure与source semantics |
| endpoint visibility network/threshold | 当前local material program不预测scene occlusion | `not-applicable` | 应留在scene transport第二波，不应塞进material latent/evaluator |
| all-LoD shared decoder | NVIDIA hierarchical latent与decoder已支持footprint-conditioned读取 | `faithful` identity不由本文证明；可作跨层稳定性诊断 | 按mip/footprint分别报告energy与error，检查类似Pandanus层间漂移 |
| projected-area/query-measure设计 | 当前formal query已冻结half/difference、Gaussian footprint、online source route | `interface-adaptation` only if tested | 可新增matched sampling ablation，但必须新recipe identity，不能静默更换训练measure |
| `log1p` + relative L2 release setting | 当前NVIDIA为bare `f`上的`log1p` L1 | `not-applicable` + candidate ablation | 只支持“log domain对大动态范围有工程先例”，不支持把L2换入faithful recipe |
| per-voxel classifier calibration | 当前sampler是连续proposal/PDF，无hard threshold | `not-applicable` | calibration与representation要分报；本文不能证明当前sampler defect |

当前 NVIDIA formal config/实现 locator 为 `configs/learning/nvidia-rta2024-materialx-formal.json:6–82`、`src/ncls/learning/methods/nvidia.py:398–534`、`src/ncls/learning/models/nvidia_neural_appearance.py:252–287`。本报告未发现可由本文直接证明的当前 `suspected-defect`。最直接的行动是增加**隔离的 prepare/filtering diagnostic**：同一source/query/decoder/latent bytes下，比较现有hierarchical texture filtering与“共享multi-resolution feature grid”，且保持runtime输出仍是bare `f`；若改成aggregate voxel transport，必须是另一scene-level method identity。

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：把层级latent lookup显式移入`prepare()`，并在同一着色点复用，可改善多灯/多`wi`成本而不损质量 | 本文每appearance query重复空间HashGrid；项目合同允许prepare复用 | material latent对同一UV/footprint/wo可复用，方向decoder无需重复空间过滤 | current NVIDIA prepare/evaluate vs相同weights/latent、仅缓存空间encoding后的variant | source、checkpoint、latent bytes、precision、directions、lights、decoder | bare-f error/energy；prepare一次+N evaluate median/p90；bytes/fetch | 静态有界 local evaluator | N≥4时总时间无显著下降，或缓存状态超预算/造成质量差异 |
| H2：多尺度联合训练需per-level energy balancing，否则会出现层间不稳定 | Pandanus rerun中受影响LoD漂移、能量不守恒 | 本项目hierarchical latent在不同footprint也可能争共享decoder容量 | uniform level loss vs按level energy residual自适应reweight；同total queries | source split、query recipe、decoder、latent、optimizer、seeds≥3 | 每level normalized L1/energy median-p95、跨seed方差、runtime | compiler/training mechanism；runtime不变 | reweight不降低worst-level/跨seed误差，或改善仅来自额外queries |
| H3：multi-resolution feature grid在相同bytes/time下优于dense latent mip只对强spatial correlation source成立 | 本文hash fields在复杂asset达71.7–96.2% source compression | MaterialX纹理/空间变化可由共享hash利用，1×1 LayerStack无此优势 | hashed prepare candidate vs current hierarchical texture；分别在spatial MaterialX与1×1 LayerStack matched | decoder、total latent bytes、training work、precision、filter target、seeds | G1 error、filter/temporal error、fetch/time、collision artifacts、energy | static prepare+evaluate candidate | spatial source无Pareto改善，或1×1仍同等收益从而否定“spatial correlation”机制解释 |
| H4：连续跨层trilinear不一定优于单层随机选择，需同时测bias与variance | 本文只离散LoD，不提供答案；其Pandanus与Fig.16显示层间函数并不平滑 | 当前latent各mip可能学到不同语义 | adjacent-level stochastic one-fetch vs trilinear two-fetch；两者matched retraining和frozen diagnostic | footprint、latent hierarchy、decoder、queries、precision | filtered GT error、temporal spectrum、variance、fetch/time | 一读随机 vs两读确定 | trilinear在不更高成本下同时降低bias/variance，或stochastic噪声抵消读取收益 |
| H5：训练query measure比单纯增宽MLP更能改善glossy/grazing tail | 本文uniform direction+SH对glossy失败，作者要求importance sampling；projected-area ablation也证明measure关键 | 当前local material长尾同样由峰值coverage限制 | current recipe vs matched peak/grazing-aware query；另加等算力增宽control | total logical queries、optimizer、latent、decoder MAC或分别matched、source/seed | p95/p99、peak location/amplitude、energy、single-query runtime | training-only recipe change | query重分配不改善tail，或改善完全由改变evaluation measure产生 |

这些假设都不把本文标成compiler证据。H3/H4属于 `prepare/prefilter` 候选，H1属于runtime scheduling，H2/H5属于training diagnostic；任何scene aggregate target都必须另设scene-level identity。

## 16. 证据索引

### `P` main paper

- pp. 1–3，abstract/§1–§2：问题、贡献、70–95%摘要口径与相关工作；
- pp. 4–5，§3、Eq. 1–4、Fig. 2–3：generalized voxel path space、`T_V`、`φ_V`、far-field与visibility分解；
- pp. 5–6，§4.1–§4.2、Fig. 4–6：pipeline、七层voxel、appearance network/topology/HashGrid/SH；
- pp. 7–8，§4.3–§4.4、Fig. 7–10、Eq. 5–6：object-space recurrent visibility、weighted F threshold、collision/0.5 threshold；
- pp. 8–9，§4.5、Fig. 11：visibility RR、compaction与83.5/41.2ms；
- pp. 9–10，§5、Fig. 12–14：online LoD/voxel/projected-area/domain extrusion与appearance MC estimator；
- pp. 11–12，§6、Fig. 15–16、Tables 1–2：hardware、optimizer、training time、runtime、memory、五场景FLIP/timing；
- pp. 13–15，Fig. 17–20：variance、Pandanus instability、Hybrid/Bako/threshold comparisons、限制与未来工作；
- p. 15 acknowledgments：asset provenance和Bako early access。

### `S` supplemental

- pp. 1–2：五场景非均匀照明的L6→L0 FLIP；
- pp. 3–4：City Houses 16.8M triangles、两种lighting、memory/training tables。

### `C` official code/config at `2035107a...`

- `README.md`: build平台、CUDA/CC、Arcade与Drive assets、自定义scene/voxelize/render scripts；
- `rptr/cuda/neural-lod/neural_lod_learning.h`:21–37,120–225：7 LoD常量、两网defaults、threshold defaults；
- `neural_lod_learning_kernels.cuh`:25–225,239–264,330–360：LoD/voxel/boundary query、`log1p` target；
- `neural_lod_learning_init.cu`:580–660,800–910：appearance/visibility exact composite encodings、MLP、optimizer wrappers；
- `neural_lod_learning.cu`:502–590：online appearance step、invalid compaction与training；
- `neural_lod_learning.cu`:130–330：101 threshold candidates及per-voxel optimization；
- `wavefront_neural_pt.cuh`:125–273：endpoint inputs、uniform/HG direction proposal、appearance input与`expm1` output；
- `wavefront_neural_throughput_visibility_lod.cu`:295–330,467–690：threshold decision、variable voxel traversal与batched inference；
- `neural_lod_learning_ui.cu`:251–256：visibility config export误传throughput network；
- `neural_lod/scenes/arcade/arcade_world_final.ini`:100–251：release formal runtime、training、threshold与network config；
- `arcade_world_final_512_throughput_config.json` 与 weights JSON：serialized topology、optimizer和exact parameter counts；
- `rptr/ext/tiny-cuda-nn/.../spherical_harmonics.h`:393–397：degree 8输出64 channels；
- GitHub commit API for `2035107a...`：仅README scene link一增一删。

### `A` author/project material

- Saarland project record：论文身份、PDF/supp/code/video/DOI入口；
- Lingqi Yan research page与Intel media page：作者身份、project/media入口、70–95%与interactive→real-time作者摘要；
- ACM record：TOG 42(4) Article 78、2023-07-26、CC BY 4.0、161.93MB presentation locator；
- official 1:21 MP4 SHA `87F63F...`：结果/temporal演示；不作为未在P/S/C披露的数值来源。

### `N`

- `docs/realtime_material_compilation.md`:3–7：`prepare/evaluate/sample/pdf`、bare linear `f`、online source reference与matched sampler合同；
- `docs/material_scope.md`:3–7：native source语义与reference边界；
- `docs/research/experiment_framework.md`:8–10,35–39,61–71,90–100：query、G1/G2/G2s、matched control与bootstrap；
- `docs/learning.md`:3–9：当前 NVIDIA `functional-f@2`、hierarchical z8、bare-f evaluator与sampler recipe；
- `configs/learning/nvidia-rta2024-materialx-formal.json`:6–82、`src/ncls/learning/methods/nvidia.py`:398–534、`src/ncls/learning/models/nvidia_neural_appearance.py`:252–287：当前identity、formal配置与frame/evaluator实现。

### `I`

- §§13–15所有容量、迁移、contract与hypothesis判断均为本项目分析；它们不改写作者claim。

## Evidence review

```text
author_worker: /root/taming2026
reviewer: /root
reviewed_at: 2026-08-29
sources_rechecked:
  - main PDF 16/16 pages and SHA-256
  - supplemental PDF 4/4 pages and SHA-256
  - official code archive SHA-256 and GitHub main HEAD
  - Arcade ini, serialized throughput config, both weight headers and seven threshold assets
  - training/query, threshold metric, runtime sampling/traversal and config-export code paths
  - project runtime/source-contract evidence cited in N/I sections
findings_closed:
  - main and supplemental numerical tables were re-transcribed from rendered pages
  - 2023 training times, storage, FLIP, baseline and threshold/traversal results retain their original protocols
  - release log1p/expm1, 101 threshold candidates, 10000 samples per voxel, 30000-step cap and config-export defect have exact code/config locators
  - paper's up-to-degree-8 wording is separated from tiny-cuda-nn degree=8 as 64-channel implementation convention
  - local-material catalog classification corrected to scene-transport/asset-prefilter
remaining_evidence_gaps:
  - ACM 161.93 MB presentation download returned HTTP 403
  - official Google Drive extra-scene bundle was not downloaded and hashed file-by-file
  - paper-vs-release log1p, threshold candidate/sample-count, 8k-12k-vs-30k and 3.81M/3.80M-order discrepancies remain unresolved
  - exact assets/cameras/environments, seeds/repeats/checkpoint selection, all-scene release configs and machine-readable Fig. 15 timing remain unavailable
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
