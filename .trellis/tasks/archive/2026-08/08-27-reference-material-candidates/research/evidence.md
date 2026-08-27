# Reference material 候选：planning 证据

本文只记录进入方案选择所需的可核对事实，不是最终 shortlist。

## 1. NVIDIA neural appearance 的材质事实

### 1.1 2024 论文使用的五个 reference material

论文《Real-Time Neural Appearance Models》的 Figure 2 / Table 1 给出五个内部制作的 layered MaterialX 材质。它们最初在 Houdini 中制作，几乎所有参数由 4K–8K 纹理驱动，部分材质最多使用 14 个 4K tile。Supplemental 只公开了节点图截图，没有提供可直接恢复原材质的 `.mtlx`、完整纹理与资产包。

| 材质 | 视觉构成 | nodes | layers | parameters | RGB textures / channels | RGB MTexels |
|---|---|---:|---:|---:|---:|---:|
| Teapot ceramic | 陶瓷基体与吸收釉层，叠加 stain、dust | 37 | 5 | 121 | 5 / 11 | 1174 |
| Teapot metal handle | conductor 基体，叠加 dirt、grease | 41 | 2 | 91 | 11 / 19 | 152 |
| Cheese slicer plastic handle | 塑料基体，叠加 grease、dirt | 20 | 5 | 43 | 3 / 7 | 201 |
| Cheese slicer metal blade | conductor 刀片，叠加 grease、dirt 与划痕纹理 | 54 | 3 | 114 | 16 / 40 | 324 |
| Inkwell metal body | brass、oxide、verdigris 的多层混合 | 49 | 5 | 143 | 4 / 11 | 201 |

这些材质的强项不是单一 exotic closure，而是“高分辨率空间变化 + 多层多峰方向响应 + 污渍/灰尘/油脂/氧化的相关纹理”。

第一方来源：

- 论文项目页：https://research.nvidia.com/labs/rtr/neural_appearance_models/
- 正文：https://research.nvidia.com/labs/rtr/neural_appearance_models/assets/nvidia_neural_materials_author_paper.pdf
- Supplemental：https://research.nvidia.com/labs/rtr/neural_appearance_models/assets/nvidia_neural_materials_author_supplemental.pdf

### 1.2 当前 `NVlabs/neuralappearance` 仓库不是论文五材质资产发布

截至 2026-08-27，仓库 README 将其定位为 MaterialX/MDL reference 的训练 pipeline，并同时服务 2026 年《Taming optimization variance in compact neural shading networks》。仓库随附的 runnable examples 是：

- `Bark.mtlx` + basecolor/normal/roughness；
- `FauxLeather.mtlx` + basecolor/normal/roughness；
- `PatternedMetal.mtlx` + basecolor/metallic/normal/roughness。

默认配置另给出外部 NVIDIA vMaterials 的 `Wood_Tiles_Pine_Mosaic` MDL 示例。仓库支持多个 MaterialX/MDL source 直接在 GPU 上求 reference，但未包含 2024 论文的 Teapot、Cheese slicer、Inkwell 原始资产，也未包含 2026 论文的 6 个内部多层材质。后一结论来自公开仓库树与两篇论文资产清单的比对。

这三个示例也不能直接进入当前项目：它们都是 `<surface> → closure BSDF graph`，而当前 adapter 只接受 root-level `standard_surface` 子集。进一步用锁定的官方 MaterialX 1.39.4 validator 检查，三份原文档都因多重 binding 与类型不匹配而未通过；NVIDIA loader 明确使用 `mtlx_source=houdini` compatibility mode。因此它们是有价值的方言/closure fixture，但不是“有 MaterialX 1.39.4 就能无修改读取”的 portable assets。完整证据见 `research/nvidia-2026-materialx.md`。

第一方来源：

- 仓库：https://github.com/NVlabs/neuralappearance
- 随附材质目录：https://github.com/NVlabs/neuralappearance/tree/main/assets/materials
- 默认配置：https://raw.githubusercontent.com/NVlabs/neuralappearance/main/configs/default.json
- vMaterials：https://developer.nvidia.com/vmaterials

## 2. Ling-Qi Yan 及邻近微结构工作的语义

### 2.1 Geometric glints：高分辨率 normal/height field

2014 P-NDF 与 2016 position-normal distribution 工作把 source 定义为高分辨率 normal map 或 heightfield，加 footprint、intrinsic roughness 与 conductor/dielectric Fresnel。它们能表示 brushed/scratched metal、metallic paint、bumpy plastic 和 ocean glints。2016 方法把 position-normal 4D 分布近似为大量 Gaussian elements，允许标准 Monte Carlo renderer 进行局部 BRDF 求值。

适配 NeuralShading 时，原生查询应是：

```text
(height/normal resource, x, footprint covariance, wi, wo, material optical parameters) -> RGB f
```

这不是普通 point-sampled normal map。`footprint` 是 source semantics 的一部分，不能在采集前丢掉。

来源：

- 2014：https://rgl.epfl.ch/publications/Yan2014Rendering
- 2016：https://sites.cs.ucsb.edu/~lingqi/publications/paper_glints2.pdf
- 作者公开代码目录：https://sites.cs.ucsb.edu/~lingqi/publications/

代码压缩包公开，但页面没有清晰的可再分发许可证；正式登记前需要单独审计。

### 2.2 Wave-optics heightfield：结构色与彩色 glint

2018《Rendering Specular Microgeometry with Wave Optics》把 micron-resolution heightfield、物理 texel 尺度、空间位置与 coherence/footprint 尺度作为输入，对 phase-delay grating 进行 wave-optics 求值。它能在 scratched、brushed、isotropic heightfield 上产生随波长变化的彩色 glint。

作者仓库 `lingqi/WaveOpticsBrdf`：

- 输入 heightfield、query center/size、incident direction 与 diffraction model；
- 支持 geometric optics、单波长和多波长 wave optics；
- 单独提供 isotropic、scratched、brushed 三类 heightfield 下载；
- 代码为 GPL-3.0，heightfield 下载项仍需逐项确认许可；
- 当前工具生成固定 incident direction 下的 BRDF image，不直接提供适配本项目合同的随机访问 GPU evaluator 或 matched sampler。

来源：

- 论文：https://cseweb.ucsd.edu/~ravir/waveoptics.pdf
- 代码：https://github.com/lingqi/WaveOpticsBrdf

### 2.3 `Scratch Iridescence` 的作者归属与候选价值

“Scratch Iridescence: Wave-Optical Rendering of Diffractive Surface Structure”并非 Ling-Qi Yan 的论文；作者是 Sebastian Werner、Zdravko Velinov、Wenzel Jakob 与 Matthias Hullin。它把 surface roughness 表示为 scratch line segments，对各 segment 的 diffraction pattern 做解析求值和相干叠加，天然从局部彩色 glint 过渡到远场平滑 BRDF。项目页提供论文、supplemental、视频和代码。

这项工作的 source contract 比通用 heightfield 更专门但更可编辑：scratch segment field、宽度/深度/方向/密度、基底光学参数与 spectral query。它是“微观划痕结构色”最直接的候选，但旧实现基于 Mitsuba/GPL 生态，正式接入前要隔离上游并审计代码包与资产许可。

来源：

- 论文与代码：https://rgl.epfl.ch/publications/Werner2017Scratch
- 实时变体：https://light.informatik.uni-bonn.de/real-time-rendering-of-wave-optical-effects-on-scratched-surfaces/

### 2.4 Layered microflake：SpongeCake

SpongeCake 把每层定义为 SGGX microflake 或其他 phase-function volume，不含层间界面。论文用随机游走作为 multiple-scattering GT，并给出快速解析 single scattering 与近似 multiple scattering。原生参数可构造 fiber-like/surface-like layer、方向、roughness、thickness、reflectance，也可附 orientation/thickness maps，能覆盖 fabric、wood、leaf 与半透明 shade。

它适合作为无 texture 或少量 orientation texture 的参数化 source family，优点是原生可编辑、已有 matched evaluation/sampling 思路；缺点是公开页面未见完整 reference 实现，需从论文重建并做独立 oracle。

来源：

- 项目页：https://wangningbei.github.io/2022/SpongeCake.html
- 论文：https://sites.cs.ucsb.edu/~lingqi/publications/paper_spongecake.pdf

## 3. 邻近但更易落地的候选

### 3.1 RGL spectral measured BRDF

RGL material database 当前提供 62 个各向同性/各向异性测量材质，含 360–1000 nm、约 4 nm 间隔的 spectral 版本与 RGB 版本，并提供 reference `eval/sample/pdf` API。除特别标注外数据为 CC0。视觉辨识度高的具体条目包括：

- `irid_flake_paint1` / `irid_flake_paint1_fine`：iridescent flake car paint；
- `irid_flake_paint2`：带 flakes 的 iridescent car paint；
- `aniso_brushed_aluminium_1`：各向异性拉丝铝；
- `weta_brushed_steel_satin_pink`：Weta Digital 的 satin pink 拉丝钢；
- `aniso_morpho_melenaus`：Morpho melenaus 蝶翼的各向异性测量。

它们是空间均匀 measured BRDF，不能代替 scratch/heightfield source，但很适合先补足“结构色、强各向异性、实测 car paint”这一组无 texture reference。相比 MERL，新增价值是 spectral、anisotropic 与原生 importance sampling，而不是单纯多几个材质名。

来源：

- 数据库：https://rgl.epfl.ch/materials
- FAQ / 许可 / API：https://rgl.epfl.ch/pages/lab/material-database

### 3.2 2025 constant-time procedural glints

《Evaluating and Sampling Glinty NDFs in Constant Time》用 implicit multiscale 4D point process 表示 faceted geometry，支持 GGX/Beckmann、anisotropy、per-facet color、importance sampling，不需要 texture 或预计算，并公开 standalone GLSL/Shadertoy 实现。

它非常适合做：

- 无 texture、seed 可控的 fancy source；
- neural compiler 的 optimized-code control；
- evaluator 与 sampler 同时受压的近期开发表面。

但它更像一个已经运行成本很低的程序化 source，研究价值是“能否统一编译和保持 glint 分布”，不是加速该 source 本身。正式使用前还需确认 Shadertoy 代码的再分发许可。

来源：https://perso.telecom-paristech.fr/boubek/papers/Glinty/

## 4. 当前仓库已有的低成本展示增益

当前 OpenPBR 1.1.1 package 已锁定 83 个官方示例，其中以下材质无需新增 source family：

- `open_pbr_aluminum_brushed`：anisotropic metal；
- `open_pbr_carpaint`：coat；
- `open_pbr_pearl`：coat + thin film + subsurface；
- `open_pbr_soapbubble`：thin film + delta transmission；
- `open_pbr_velvet`：fuzz；
- `open_pbr_glass`：transmission/volume。

它们可以先组成 showcase pack，检验当前 viewer 是否只是默认材质选择不够有辨识度。需要注意：当前 OpenPBR reference 使用固定 RGB representative wavelengths；若要把 soap bubble / pearl 的 spectral aliasing 作为科学结论，必须先增加随机波长采集，而不能把 RGB 预览当完整 wave-optics GT。

仓库证据：`references/openpbr-1.1.1-v1/README.md` 与 `materials.json`。

## 5. 已收敛组合

shortlist 不选一个万能材质，而分三条轨道：

1. 立即展示：现有 OpenPBR fancy presets + LayerStack 中受 Layer Laboratory 启发但保持本族原生语义的无 texture layered recipes；
2. 下一阶段：full closure MaterialX graph + RGL spectral measured car paint/蝶翼 + geometric procedural glints；
3. 长期压力：wave-optics heightfield 或 scratch-segment source，保留 spectral 与 footprint 原生语义。

用户已确认近期路线可继续，同时要求把 2026 `.mtlx` 与历史可复现材质补齐。完整映射见 `research/material-lineage.md`。
