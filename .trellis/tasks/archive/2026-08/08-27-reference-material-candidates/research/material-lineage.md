# NVIDIA 外观类别的历史模型与可复现来源

## 1. 外观类别映射

| NVIDIA 外观类别 | 图形学模型脉络 | 现成代码或数据 | 对 NeuralShading 更合适的接入形态 |
|---|---|---|---|
| glaze、absorbing coat、dust、gold-and-ceramic | Jakob 等 2014 arbitrary layered BSDF；Zeltner/Jakob 2018 Layer Laboratory | Layer Laboratory BSD-3 源码、scene 与 measurement data；仓库自带 `aniso_gold_dust`、`aniso_gold_blue_dielectric`、gold + dielectric 示例 | 近期可在现有 LayerStack 原生语义中制作相似但独立的无 texture 配方；长期可把 Layer Laboratory/Fourier layer 作为独立 oracle，不把它反演成 LayerStack GT |
| stain、grease、dirt 的空间层与 mask | Shade-tree / closure graph 的 spatial mix 与 layer；MaterialX 是现代开放图表示 | NVIDIA `PatternedMetal` 给出 metal/dielectric closure mix 骨架；CC0 dirt/roughness/normal masks 可合法重建相同外观类别 | 扩展 MaterialX full closure graph；mask、图和颜色变换都属于 GT。普通 basecolor/roughness 重贴图不能替代 closure mix |
| oxidation、verdigris、metal patina | Dorsey/Hanrahan 1996：layered patina、`coat/erode/polish` 与 Kubelka–Munk；Merillou 等 2001：corrosion random walk 与 BRDF/texture | 论文和图版公开；未发现可直接登记的原作者代码包 | 用论文模型生成 thickness/mask，再由 conductor + oxide/patina layers 求值；可先做 MaterialX/LayerStack 中“相似外观”的受控配方，但必须标成重建而非 NVIDIA 原资产 |
| scratched steel、brushed brass、bumpy plastic | Yan 等 2014 high-resolution normal-map glints；2016 P-NDF；Raymond 等 2016 structured scratch SVBRDF | 2014 代码与资源、2016 code snippet；Raymond 论文模型可复现 | geometric-optics source 保留 normal/height resource、physical texel scale、footprint 与方向；不是 point-sampled normal map |
| 微观划痕的彩色衍射 | Werner 等 2017 Scratch Iridescence；Yan 等 2018 wave-optics heightfield | Scratch Iridescence 提供完整 code archive；`lingqi/WaveOpticsBrdf` 提供 GPL-3.0 code 与 isotropic/scratched/brushed heightfields 下载 | 独立 spectral + footprint source family；GPL 实现用进程外 oracle 或重新实现论文公式，不直接链接进项目 runtime |
| flake paint、sparkle、离散 glint | Jakob/Hašan/Yan 等 2014 discrete stochastic microfacets；Chermain 等 2020 procedural real-time glints；Kemppinen 等 2025 constant-time glinty NDF | 2014/2020 pbrt 代码与 scenes；2025 standalone GLSL/Shadertoy；RGL measured flake paint | 无 texture procedural source 可先验证 `evaluate/sample/pdf`；RGL measured source提供独立实测对照。二者不可互相冒充 |
| thin-film iridescence、油膜、soap bubble、pearl | Belcour/Barla 2017 varying iridescence；Guillén 等 2020 pearlescent materials | Belcour 提供 Mitsuba/GLSL code；当前 OpenPBR 已有 pearl/soapbubble；RGL 有 iridescent samples | OpenPBR 适合近期 RGB/固定代表波长展示；严谨 spectral 结论需随机波长或 spectral reference。薄膜干涉与 scratch diffraction 是两类 source |
| Morpho 等结构色实物 | measured spectral anisotropic BRDF | RGL `aniso_morpho_melenaus`，CC0 数据，BSD-3 `brdf-loader`，原生 `eval/sample/pdf` | 新增 RGL measured source family；无 texture、空间均匀，但能独立考核强光谱/各向异性方向响应 |
| 强 normal variation 与 LOD | Yan 2014/2016、real-time microstructure filtering、2025 implicit multiscale glinty NDF | normal/heightfield、GLSL 或论文实现 | source query 必须带 footprint/filter state；把 LOD 预烘为一张普通 normal map 会丢失原生语义 |

## 2. 可直接复用的关键上游

### 2.1 Layer Laboratory：最贴近 NVIDIA layer stress，且不需要 texture

Layer Laboratory 的公开仓库 commit `008cc94b76127e9eb74227fcd3d0145da8ddec30` 为 BSD-3-Clause。它不只提供通用算法，还包含可直接复现的材质组合：

- anisotropic gold；
- anisotropic gold + Henyey–Greenstein dust slab；
- gold + isotropic/anisotropic dielectric coating；
- anisotropic gold + blue scattering medium + dielectric coating。

这些组合已经覆盖 NVIDIA 的 brushed metal、dust、glaze、gold-and-ceramic 的主要方向响应压力，而且全部可做无 texture 参数化控制。它们不是 NVIDIA 同款资产，但非常符合用户要求的“相似效果/种类”。代价是 Fourier precomputation 很重，部分示例声明可需 64 GB 内存；因此更适合作为离线 oracle 或配方来源，而不是实时 target representation。

来源：

- https://rgl.epfl.ch/publications/Zeltner2018Layer
- https://github.com/tizian/layer-laboratory

### 2.2 RGL measured spectral BRDF：低风险增加真实 fancy appearance

RGL 数据库提供 spectral/RGB、isotropic/anisotropic 数据，默认 CC0；`rgl-epfl/brdf-loader` 为 BSD-3-Clause，并提供 evaluation、sampling、pdf。优先条目：

- `irid_flake_paint1` / `_fine`、`irid_flake_paint2`；
- `aniso_brushed_aluminium_1`、`weta_brushed_steel_satin_pink`；
- `aniso_morpho_melenaus`。

它们不带空间 texture，不能替代 scratch distribution 或 dirty layered graph，但能立即提供 MERL 没有的 spectral、anisotropic 与 measured flake/structural-color stress。

来源：

- https://rgl.epfl.ch/materials
- https://rgl.epfl.ch/pages/lab/material-database
- https://github.com/rgl-epfl/brdf-loader

### 2.3 Yan 2018 wave optics：最直接的 heightfield 结构色 oracle

`lingqi/WaveOpticsBrdf` 固定输入 micron-scale heightfield、位置与 footprint/coherence size、incident direction 和 diffraction model，能输出 scratched、brushed、isotropic 表面的单波长或多波长 BRDF image。优点是模型与三类 heightfield 都公开；缺点是 GPL-3.0、CPU/offline、当前接口不是随机访问双方向 evaluator，heightfield 的再分发许可还需逐项核对。

它应被视为长期 spectral oracle 或公式复现来源，不应把 GPL code 直接嵌入项目 runtime。

来源：https://github.com/lingqi/WaveOpticsBrdf

### 2.4 2025 constant-time glinty NDF：最佳无 texture procedural control

该模型是 memory/precomputation-free 的 implicit multiscale 4D point process，支持 GGX/Beckmann、anisotropy、per-facet color 与 importance sampling，并公开 standalone fragment shader。它能产生 metallic paint、sparkle、glitter 等外观，不依赖 NVIDIA 资产。

它最适合做 optimized-code control：既有 `evaluate()` 又有匹配 sampling 路径，运行成本本身已低，能测试 compiler 是否保留离散 glint 分布，而不是宣称 neural 方法必然加速它。正式登记前需确认 standalone shader 的再分发许可。

来源：https://perso.telecom-paristech.fr/boubek/papers/Glinty/

## 3. 分阶段 shortlist

### 近期：不等新 source family

1. 在现有 LayerStack 中增加两组原生参数化材质：`dusty-anisotropic-brass` 与 `absorbing-glazed-ceramic`。配方可以借鉴 Layer Laboratory，但 GT 仍由本项目 LayerStack reference 定义；不声称复刻其 Fourier output。
2. 把现有 OpenPBR `carpaint / pearl / soapbubble / aluminum_brushed` 组成 showcase pack。它们能马上改善视觉覆盖，但 carpaint 没有离散 flakes、aluminum_brushed 没有显式 scratch microgeometry，必须在文档中写清。

### 下一阶段：真正补 source diversity

1. `materialx.closure-graph@1`：先用 NVIDIA Bark 与 PatternedMetal 做 acceptance fixtures，再增加一个 CC0 mask 驱动的 glazed/dusty ceramic 和一个 oxidized/scratched brass。源图和纹理保持 GT。
2. `rgl.measured-spectral@1`：先接 flake paint、Morpho、brushed metal；它是无 texture 的 measured 路线，和 MaterialX spatial graph 互补。
3. procedural glinty NDF：作为无 texture、seed 可编辑、带 matched sampler 的 optimized-code control。

### 长期压力

1. Yan wave-optics heightfield；
2. Scratch Iridescence segment field；
3. 若需要 pearlescent platelets 的可编辑物理参数，再评估 Guillén 2020 或 SpongeCake，而不是把 OpenPBR thin film 当成等价 reference。

这组组合同时满足：近期能展示、历史模型可追溯、至少一个无 texture source、至少一个带空间资源 source，并且不会要求所有材质归约到 LayerStack。
