# Fancy Reference Material 候选报告

## 1. 结论

当前问题同时包含两层：已有 source family 的展示选择偏保守，以及缺少能表达复杂 closure、measured spectral response 和显式微结构的 source family。只更换几张 base-color/roughness 纹理可以改善画面，但不能验证 neural evaluator 是否保留 layering、方向高频、spectral、footprint/LOD 和 matched sampling。

推荐采用四条互补路线：

1. **立即展示**：使用当前 OpenPBR 的 car paint、pearl、soap bubble、brushed aluminum、velvet、glass，以及现有 LayerStack 的原生参数化层状配方；
2. **首个新增 source**：接入 NVIDIA vMaterials 2 的原生 MDL program，优先覆盖 color-shifting paint、patinated brushed copper、scratched aluminum、glazed ceramic 与 velvet；
3. **下一批科学压力**：full closure MaterialX、RGL measured spectral BRDF 和可匹配采样的 procedural glinty NDF；
4. **长期微结构压力**：Yan wave-optics heightfield 与 Scratch Iridescence segment field，保留 wavelength 与 footprint 原生语义。

这些路线不是相互替代：MDL 的 normal-map scratches 不等于显式 scratch diffraction，analytic thin film 不等于 measured structural color，OpenPBR anisotropy 不等于 footprint-filtered microgeometry。

## 2. NVIDIA 两篇论文真正使用的材质

### 2.1 2024《Real-Time Neural Appearance Models》

论文使用五个内部制作的 layered MaterialX/Houdini 材质：

| 材质 | 原生构成 | 主要压力 |
|---|---|---|
| Teapot ceramic | ceramic base、absorbing glaze、stain、dust | 多层、多 normal、空间 contamination |
| Teapot metal handle | conductor、dirt、grease | 金属高光与相关 mask |
| Cheese slicer plastic handle | plastic、grease、dirt | 多层空间粗糙度变化 |
| Cheese slicer metal blade | conductor、grease、dirt、scratch textures | 高光、划痕与大量纹理输入 |
| Inkwell metal body | brass、oxide、verdigris layers | conductor 与氧化/铜绿混合 |

它们约有 20–54 个 nodes、2–5 层、43–143 个参数，几乎所有参数由 4K–8K 纹理驱动，部分材质最多使用 14 个 4K tiles。Supplemental 给出节点图截图，但公开仓库没有发布这些材质的完整 `.mtlx` 和纹理。因此这五个原资产属于 **inspiration-only**；可复现的是外观类别和网络方法，不是原 GT。

第一方来源：[项目页与论文](https://research.nvidia.com/labs/rtr/neural_appearance_models/)、[supplemental](https://research.nvidia.com/labs/rtr/neural_appearance_models/assets/nvidia_neural_materials_author_supplemental.pdf)。

### 2.2 2026《Taming Optimization Variance in Compact Neural Shading Networks》

正文 Figure 4 使用两组共 15 个材质：

- 9 个 da Vinci Workshop 物体：Birdcage、Chandelier、Hammer、Lantern、Mirror、Palette、Chair、Scales、Table。论文把原 MDL 烘焙为单 UV tile 的 4K SVBRDF，再转换成 USDPreviewSurface；
- 6 个内部多层材质：Scratched steel、Bumpy plastic、Oxydized metal、Gold and ceramic、Brushed brass、Glazed ceramic。每个材质有 base 与 glazing、stain、dust 等 top layers，各层可有独立参数、纹理和 normal map。

公开仓库没有发布这六个内部多层材质。仓库随附的 Bark、FauxLeather、PatternedMetal 是 runnable examples，不是论文六材质的资产发布。论文的 9 个 da Vinci 目标经过烘焙和 USDPreviewSurface 转换，也不等于原始 MDL program 已作为训练 GT 发布。

第一方来源：[论文项目页](https://research.nvidia.com/labs/rtr/publication/bitterli2026taming/)、[公开仓库](https://github.com/NVlabs/neuralappearance)。

## 3. NVIDIA 公开 MaterialX 的可用性

仓库 commit `305b4b9c12e679398c487603dd8245c3f348526c` 随附三份 MaterialX 文档和 CC0 4K 纹理：

| 文档 | 原生 closure | 建议用途 | 证据等级 |
|---|---|---|---|
| `Bark.mtlx` | dielectric coat layer + Oren–Nayar diffuse，共用 normal | full closure MaterialX 的最小 layered fixture | direct，但需 NVIDIA/Houdini compatibility reference |
| `PatternedMetal.mtlx` | conductor 与 coated diffuse 的 spatial mix | closure mix、metal/dielectric 分支和 spatial mask fixture | direct，三者中优先级最高 |
| `FauxLeather.mtlx` | conductor，base color 被转成复 IOR | NVIDIA pipeline smoke test 与方言 fixture | direct，但不作为权威 leather model |

它们在 NVIDIA falcor2 中通过 `mtlx_source=houdini`、`mtlx_layering_mode=bsdf_mix` 运行；三份原文档都未通过项目锁定 MaterialX 1.39.4 validator，也不满足当前只接受 root-level `standard_surface` 的 adapter。正确边界是保留原 `.mtlx` 和纹理为 source identity，以 NVIDIA falcor2 为方言 reference，归一化文档或生成 shader 仅作为可重建 artifact。

完整图审计见 [nvidia-2026-materialx.md](nvidia-2026-materialx.md)。

## 4. Omniverse packs 与 MDL

Omniverse 下载页真正的材质包全部是 MDL：Base Materials、vMaterials 2 和 Automotive Materials；其余 OpenUSD packs 主要是 scene composition、geometry、environment 与 MDL binding。[官方下载页](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html)

`NVlabs/neuralappearance` 已经通过 falcor2 + MDL SDK 在 GPU 上直接求 MDL reference，README 和默认配置明确给出 `vMaterials_2/Wood/Wood_Tiles_Pine.mdl` / `Wood_Tiles_Pine_Mosaic`。这使 vMaterials 2 成为与公开 NVIDIA 2026 pipeline 对齐度最高的新增 source 候选，而不是只供 Omniverse viewer 使用的装饰资产。

当前项目锁定的 Falcor 8 只有部分 `UsdPreviewSurface → StandardMaterial` 转换，没有 MDL evaluator。OpenUSD 场景能加载 geometry，并不能证明 MDL 外观被忠实执行。UsdShade 可以解析 MDL `sourceAsset`、`subIdentifier` 和参数连接，但 scattering GT 仍必须由 MDL SDK/falcor2 evaluator 定义。[MDL in OpenUSD](https://docs.omniverse.nvidia.com/usd/latest/technical_reference/referencing_mdl.html)

### 4.1 首批 vMaterials 2 shortlist

| MDL | texture | 原生特征 | 选择理由 | 明确限制 |
|---|---|---|---|---|
| `Carpaint_Shifting_Flakes` | 无材质纹理 | thin film、measured curve、GGX、Fresnel/weighted layers，30+ presets | 无 texture、可编辑、强方向色变 | 不是显式随机 microflake glint 或 measured flake BRDF |
| `Copper_Antique_Brushed_Patinated` | 7 张 brushed/patina/smudge/normal/roughness maps | directional factor、GGX、custom curve/weighted layer，9 个 patina presets | 最接近 oxidized metal / brushed brass | 不是化学腐蚀模拟或衍射划痕 |
| `Aluminum_Scratched` | 3 张 packed scratch/normal maps | GGX、diffuse、custom curve layer，6 个 clean/grime presets | 空间划痕与相关 mask 的低风险 fixture | normal-map scratch 不带 footprint-filtered microgeometry |
| `Ceramic_Tiles_Glazed_Versailles` | noise/craquelure/thickness/drops、mortar maps | glaze、diffuse、GGX/Fresnel layers，27 个 presets | 最接近 glazed ceramic，空间与层状压力兼具 | 不是论文内部五层、多 normal GT |
| `Velvet` | diffuse/imperfection/normal maps | 原生 `sheen_bsdf`、custom curve、layered diffuse/GGX | 明显增加 closure diversity | 不是显式 fiber/microflake volume |

后备候选是 `Glass_Smudged` 和无 texture `Pearl`。前者引入 transmission + spatial contamination，后者提供小体量 clearcoat/color-falloff smoke test。

### 4.2 其他 pack 的角色

- **Automotive Materials**：适合第二阶段的 clearcoat car paint、anisotropic carbon fiber、dirty metal 与 hero scene；完整包超过 20 GB，且材质依赖 `OmniUber_Automotive` templates，不适合作为第一个最小 package；
- **Base Materials**：主要承担 Omniverse 场景依赖和基础覆盖，fancy 增量小；
- **Sample/Showcase/OpenUSD scenes**：用于 evaluator 稳定后的 viewer、composition 和工作流测试，不登记为独立 scattering source family。

三个 material ZIP 的逐包 license 均指向 NVIDIA Omniverse terms。官方下载页和 vMaterials 页面允许免费用于项目，但它们不是 CC0。原包应放 `assets/source-materials/`，根 Git 只保存 URL、版本、hash、license、module/export 与资源 manifest，不直接再分发大包。

完整远程 ZIP 与 MDL 抽查见 [omniverse-usd-packs.md](omniverse-usd-packs.md)。

## 5. 相似外观的历史模型与可复现 source

| 目标外观 | 历史模型/公开 source | 原生输入与查询 | 资源/许可 | 推荐身份 |
|---|---|---|---|---|
| glaze、dust、gold-and-ceramic | [Layer Laboratory](https://rgl.epfl.ch/publications/Zeltner2018Layer) | layer optical parameters；Fourier precompute 后求 BSDF | BSD-3 code；计算/内存较重 | direct offline oracle；或只借配方构造独立 LayerStack source |
| oxidation/verdigris | Dorsey/Hanrahan 1996 patina；Merillou 2001 corrosion | patina thickness/masks + conductor/layer optics | 论文可得，未发现完整原作者 package | reconstructable |
| brushed/scratched geometric glints | Yan 2014/2016 P-NDF | high-res normal/heightfield、physical texel size、position、footprint、directions | 作者代码/资源可得，许可需单审 | reconstructable footprint source |
| scratch diffraction | [Scratch Iridescence](https://rgl.epfl.ch/publications/Werner2017Scratch) | scratch segments、width/depth/orientation/density、wavelength、footprint、directions | code archive 可得；GPL/Mitsuba 隔离 | reconstructable spectral oracle |
| wave-optics heightfield | [Yan WaveOpticsBrdf](https://github.com/lingqi/WaveOpticsBrdf) | micron heightfield、position/query size、incident/outgoing、wavelength | GPL-3 code；heightfield 许可逐项核对 | reconstructable long-term oracle |
| flake paint/structural color | [RGL measured materials](https://rgl.epfl.ch/materials) | spectral/RGB measured table + `eval/sample/pdf` | 多数数据 CC0；loader BSD-3 | direct measured source |
| procedural sparkle | [Constant-time glinty NDF](https://perso.telecom-paristech.fr/boubek/papers/Glinty/) | seed、NDF、anisotropy、per-facet color、position/footprint、directions | shader 可得；再分发许可待确认 | reconstructable optimized-code control |
| thin-film/pearl/soap bubble | OpenPBR 1.1.1、Belcour/Barla、Guillén 2020 | layer/film optical parameters；严格 spectral 时需 wavelength | 当前 OpenPBR 已接入 | direct current RGB reference；spectral 结论需扩展 |

### 5.1 Ling-Qi Yan 相关路线的边界

Yan 2014/2016 的重点是 footprint 内的 position-normal distribution，2018 的重点是 micron-scale heightfield 的 wave-optical phase。两者都要求查询携带空间位置和 footprint/coherence scale；把资源点采样成一张普通 normal map 会改变 source semantics。

Scratch Iridescence 不是 Ling-Qi Yan 的论文，但它是“微观划痕结构色”最直接的相邻 source。应按独立作者、代码与许可登记，不能混写为 Yan 系列。

## 6. 候选总表

| 候选 | 等级 | 原生可编辑 | 空间资源 | spectral/footprint | `sample/pdf` | 当前接入状态 |
|---|---|---:|---:|---:|---:|---|
| 当前 OpenPBR fancy examples | direct | 是 | 可选 | 固定 RGB representative wavelengths | 是 | 已接入，可立即展示 |
| 当前 LayerStack fancy recipes | direct | 是 | 否 | 否 | 是 | reference 已有，只需新增本族配方 |
| vMaterials 2 MDL | direct upstream | 是 | 可选 | 依材质；纹理过滤语义需保留 | MDL SDK/falcor2 可提供，需项目 parity | 当前无 MDL family；首个新增目标 |
| NVIDIA Bark/PatternedMetal | direct upstream | 图/纹理可编辑 | 是 | 普通 texture footprint | falcor2 reference | 当前 MaterialX adapter 不覆盖 |
| Layer Laboratory | direct upstream | 是 | 可选 | Fourier layer，不是 spatial footprint | 有 | 未接入，适合 offline oracle |
| RGL measured spectral | direct upstream | 选择 measured entry，不可改变测量本体 | 否 | spectral；无 spatial footprint | 是 | 未接入，低风险下一批 |
| procedural glinty NDF | reconstructable | seed/NDF 可编辑 | 无预存 texture | 是 | 是 | 许可与实现待固定 |
| Yan geometric/wave optics | reconstructable | heightfield/尺度可编辑 | 是 | 是 | 当前公开工具不完全匹配 | 长期 |
| NVIDIA 2024/2026 内部多层资产 | inspiration-only | 原资产不可得 | 原纹理不可得 | 论文只给描述/图 | 不可得 | 不进入 reference registry |

“direct upstream”表示 source、权威 evaluator 与许可路径可得，不表示已经注册进 NeuralShading。只有完成 source family、reference package、typed editing、provider 与 parity 后才能称为当前 active reference。

## 7. 分阶段 shortlist

### 7.1 近期展示

1. OpenPBR：`carpaint`、`pearl`、`soapbubble`、`aluminum_brushed`、`velvet`、`glass`；
2. LayerStack：`dusty-anisotropic-brass` 与 `absorbing-glazed-ceramic` 一类的本族原生参数化配方；
3. 不对它们作超出 source 能力的宣称：car paint 没有离散 flakes，brushed aluminum 没有显式 scratch microgeometry，固定 RGB thin film 不是完整 spectral GT。

### 7.2 下一阶段 source diversity

优先顺序调整为：

1. `mdl.program@1`：vMaterials 2 精简集合；
2. `materialx.closure-graph@1`：NVIDIA Bark/PatternedMetal 作为方言与 closure fixtures；
3. `rgl.measured-spectral@1`：flake paint、Morpho、brushed metal；
4. procedural glinty NDF：无 texture、seed 可编辑、带 matched sampler 的 optimized-code control。

### 7.3 长期 spectral/footprint

1. Yan wave-optics heightfield；
2. Scratch Iridescence segment field；
3. 需要可编辑 pearlescent platelets 时再评估 Guillén 2020 或 SpongeCake。

## 8. `mdl.program@1` 的建议边界

后续实现必须把下列内容共同纳入 canonical source snapshot：

```text
pack id/version/content hash/license
MDL module path + exported material name
authored typed arguments
transitive imported modules
texture/resource URI + content hash + color/physical metadata
MDL language/SDK identity
reference backend identity and query convention
```

公共接口仍是项目的 `SourceSnapshot`、typed editor、reference program 和 `TrainingBatch@1`。falcor2/MDL 的内部 class、argument block、texture handler 和 scattering state 不进入公共 source contract。

最低完成条件：

1. `references/` 中有唯一、可追溯的 MDL package；
2. source family 能枚举/编辑 authored parameters，并用新 snapshot identity 拒绝 stale edit；
3. reference 能按项目方向、frame、UV/position 与 texture footprint 合同返回线性 `f`；
4. 若声明 sampling capability，必须同时提供匹配 `sample/pdf`；否则 capability 明确为 evaluate-only，不能静默用另一 BSDF sampler 冒充；
5. offline 与 live producer 都输出同一个 `TrainingBatch@1`，live 路径不做 host readback；
6. 以锁定的 NVIDIA falcor2 + MDL SDK 做独立逐方向或共同输入 parity，容差在正式结果前由数据类型和 oracle calibration 冻结；
7. distill/bake/USDPreviewSurface 结果只能作为显式 control 或派生 artifact，不能继承 MDL source identity。

## 9. 最终建议

如果只新增一个 material ecosystem，选择 **vMaterials 2 + 原生 MDL reference**。它同时满足：

- 与 NVIDIA 2026 公开 pipeline 直接对应；
- 有无 texture 与带 texture 材质；
- 有 layered、thin-film、sheen、transmission、patina、scratch masks 等明显超出当前展示集的语义；
- 参数和 presets 可编辑；
- MDL SDK 提供可锁定的编译/evaluation 路径；
- 不要求把 source 反演为 LayerStack 或简化为 `standard_surface`。

它仍不能替代 measured spectral 和 explicit microgeometry，所以后续仍应保留 RGL 与 wave-optics/glinty NDF 路线。这样得到的 reference 组合不是“更花哨的几张材质球”，而是能分别验证空间图、复杂 closure、实测方向响应、离散 glint、光谱与尺度过滤的 source family 集合。
