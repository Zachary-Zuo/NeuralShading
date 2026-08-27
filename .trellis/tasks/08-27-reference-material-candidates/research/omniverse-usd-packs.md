# Omniverse Downloadable Asset Packs 材质审计

## 1. 结论

这些资产能用，但用途必须分开：

- 三套 **MDL Material Asset Packs** 可以成为原生 reference source 候选，其中 vMaterials 2 与 NVIDIA 2026 开源 pipeline 的接口最匹配；
- 3D OpenUSD、Sample Scenes、Showcase Scenes 等 pack 适合作为 viewer、场景组合和工作流压力资产；若其 UsdShade 最终引用 MDL，权威 scattering 仍来自 MDL program，不来自 USD 容器本身；
- 当前 NeuralShading 只有 LayerStack、OpenPBR、MERL 与有限 MaterialX reference，没有 MDL source family，所以这些 MDL 不是“下载后直接加入现有 corpus”，而是下一阶段可实现的 source family；
- 项目锁定的 Falcor 8 虽有 USDImporter，但只把 `UsdPreviewSurface` 的一部分转换成 `StandardMaterial`，没有 MDL material evaluator；因此“USD 场景能打开”不代表 MDL 外观被忠实求值；
- 不应把 MDL distill/bake 成 USDPreviewSurface 或当前 `standard_surface` 后仍称为同一个 GT。派生的 PBR maps 可以成为 matched control 或独立 SVBRDF source。

官方页面：https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html

## 2. 官方 pack 结构

官方页面列出超过 250 GB 的 OpenUSD 样例内容，并分为 3D OpenUSD Assets、Materials、Environments 与 Workflow Samples。真正的材质类下载只有三套，且全部是 MDL：

| pack | 官方描述 | 本次远程 ZIP index 审计 | 判断 |
|---|---|---|---|
| Base Materials | 161 个材质，8.2 GB | 493 个 `.mdl`、4473 张 PNG、16 个 USD；包含多个版本/别名路径 | 基础覆盖与场景依赖价值高，fancy 增量较小 |
| vMaterials 2.4.0 | 页面称 1854 个 drag-and-drop appearances、225 个 source files；Developer 页面称当前约 2600 appearances | CloudFront 对象 2,220,534,625 bytes；315 个 `.mdl`、5688 PNG、568 JPG；大量 exports/presets | 最适合首个 MDL source package |
| Automotive Materials | 158 个 automotive surfaces，21 GB | CloudFront 对象 20,811,841,284 bytes；465 个 `.mdl`、3278 PNG、563 JPG、128 SBS、2 PSD；含多个版本与 pristine/imperfect 分支 | 适合 car paint、clearcoat、carbon fiber 与 hero scene 专项 |

三个包的 central directory 均未发现 `.mtlx`。因此“项目已有 MaterialX”不能直接读取这些材质；需要 MDL SDK reference，或明确改变 source identity 的蒸馏/烘焙路径。

vMaterials 官方页说明它以 MDL 1.7 为依赖，包含 MDL、对应纹理与 thumbnails，并把照片扫描、物理准确、可调参数、infinite tiling/tile randomization/atlas scattering 作为核心能力：https://developer.nvidia.com/vmaterials

## 3. 与 NVIDIA 2026 仓库的直接关系

`NVlabs/neuralappearance` README 明确支持两种 reference：MaterialX 经 falcor2 的 MaterialX backend，MDL 经 NVIDIA MDL SDK，均在 GPU 上在线求值而无需预生成数据。这里的 falcor2 是论文仓库锁定的独立上游，不是本项目当前的 Falcor 8。README 与 `configs/default.json` 给出的 MDL 示例是：

```json
{
  "type": "mdl",
  "path": "vMaterials_2/Wood/Wood_Tiles_Pine.mdl",
  "material": "Wood_Tiles_Pine_Mosaic"
}
```

这证明 vMaterials 2 是该公开训练 pipeline 的正式输入类型，不只是 Omniverse viewer 资产。近期最小风险路线是先把锁定的 NVIDIA falcor2 + MDL SDK 作为进程外/采集期 reference，而不是把 MDL 静默交给 Falcor 8 的 PreviewSurface converter。它不等于 2026 论文六个内部多层材质已经公开；论文中的 da Vinci Workshop 9 个对象也走的是烘焙 4K SVBRDF 后转 USDPreviewSurface 的另一条简化路线。

来源：

- https://github.com/NVlabs/neuralappearance
- https://github.com/NVlabs/neuralappearance/blob/305b4b9c12e679398c487603dd8245c3f348526c/configs/default.json

## 4. 代表性 fancy MDL 抽查

本次没有下载完整大包，而是利用 CloudFront byte-range 读取 ZIP central directory、逐包 license、`PACKAGE-INFO.yaml` 和以下 `.mdl`。表中 closure/resources 来自原始 MDL source 静态审计。

| 候选 | 原生结构与资源 | 对项目的价值 | 不能宣称的等价关系 |
|---|---|---|---|
| `Paint/Carpaint/Carpaint_Shifting_Flakes.mdl` | 约 70 KB；30+ exports；`thin_film`、`measured_curve_factor`、GGX、Fresnel/weighted layers；主材质不依赖外部纹理 | 无 texture、参数化、颜色随角度变化，适合先测复杂 closure 与编辑泛化 | 名称有 flakes，但不是显式随机微片 glint field，也不是 measured flake BRDF |
| `Metal/Copper_Antique_Brushed_Patinated.mdl` | brushed copper + patina 的 diffuse/normal/reflection/roughness/smudge 纹理；9 个 patina presets；directional factor 与 weighted layer | 最接近 NVIDIA oxidized metal / brushed brass 的公开可编辑替代类 | 贴图 patina 不是化学氧化仿真，也不是 scratch diffraction |
| `Metal/Aluminum_Scratched.mdl` | 三张 packed/normal scratch maps；6 个 clean/grime/rough presets；GGX + diffuse + custom-curve layer | 低风险空间划痕 fixture，能测多个相关 mask/normal 与高光 | normal-map scratches 不等于 Ling-Qi Yan heightfield P-NDF/wave optics |
| `Ceramic/Ceramic_Tiles_Glazed_Versailles.mdl` | 约 102 KB；27 个 exports；glaze + diffuse + GGX/Fresnel layers；noise/craquelure/thickness/drops 与 mortar maps | 最接近 glazed ceramic，兼具层状与空间 variation | tile procedural/preset 不等于论文内部多 normal-map 五层材质 |
| `Fabric/Velvet.mdl` | 原生 `sheen_bsdf`、custom curve 与 layered diffuse/GGX；diffuse/imperfection/normal maps；15 个 presets | 明显偏离普通 GGX，适合 closure diversity | 不是显式 fiber/microflake volume |
| `Glass/Glass_Smudged.mdl` | specular、diffuse transmission、fresnel/custom-curve layers；smudge/cloud/frit maps | transmission + spatial contamination 的组合压力 | 不能在仅 surface RGB evaluator 中忽略透射合同 |
| `Gems/Pearl.mdl` | 无 texture；core `flex_material` + clearcoat + color falloff | 小体量、可编辑、方向色变的 smoke test | 不是 spectral measured pearl，也不能替代 wave optics |

Automotive pack 的抽查显示其材质主要封装 `OmniUber_Automotive` template，并用 clearcoat、anisotropy、ORM/normal/roughness 与 dirt/scratch textures 组合。它适合后续 car-paint/carbon-fiber 专项，但首先需要一并锁定 template 与依赖资源，而且完整包实际超过 20 GB。

## 5. 原生 reference 边界

推荐的 source identity 是：

```text
(pack id/version/hash,
 MDL module path,
 exported material name,
 authored arguments,
 imported MDL modules,
 textures/resources,
 MDL language/SDK + falcor2 evaluator identity)
```

UsdShade 可以保存 MDL `info:sourceAsset`、`subIdentifier` 与参数连接；NVIDIA 官方文档同时说明 MDL material 可包含 surface、volume、emission、normal/displacement 等语义。USD adapter 因而可以作为 binding/import layer，但不能替代 MDL scattering evaluator。

来源：https://docs.omniverse.nvidia.com/usd/latest/technical_reference/referencing_mdl.html

本地 `external/Falcor` 的 USD 文档与实现显示，它目前只部分支持 `UsdPreviewSurface`，加载时映射为 `StandardMaterial`；clearcoat、occlusion、displacement 等输入也有已知缺口。故场景 pack 的可用性还要分成：geometry/composition 能否加载、是否提供 PreviewSurface fallback、MDL 原生外观能否求值。只有第三项能决定它是否可作本任务的 reference。

MDL SDK 本身是 BSD-3-Clause 开源 SDK，支持把 compiled material 生成 HLSL/GLSL/PTX/native code，也提供 distilling 和 baking。对本项目而言，应优先保持原生 compiled MDL 作为 GT；distilled PBR 仅作 optimized-code control 或能力降级对照。

来源：https://github.com/NVIDIA/MDL-SDK

## 6. 许可与仓库边界

三个 ZIP 都包含唯一逐包 license file，文本指向 NVIDIA Omniverse terms；官方下载页称这些 packs 可免费用于自己的项目，vMaterials Developer 页也写明 free use。它们不是 CC0，也不能把“可用于项目”自动解释成“可把 2–21 GB 原包提交到本仓库再分发”。

若后续接入：

- 原资产放 `assets/`，不进入根 Git；
- `references/` 只登记 pack URL、`PACKAGE-INFO.yaml` 的 package/version/commit、Content-Length、ETag/SHA-256、license path、module/export 与资源清单；
- 优先提供显式 fetch/install 说明，保留用户接受 NVIDIA terms 的动作；
- 不自动下载全部 250 GB 场景包，也不因 OpenUSD page 统一列出而假定所有第三方资产拥有相同再分发权。

三个 material ZIP 的 `PACKAGE-INFO.yaml` 均记录内部 package metadata；下载对象与文档标称体量并不完全一致，因此 manifest 必须锁定实际对象 hash/size，不能只记录网页展示数字。

## 7. 分阶段选择

1. **首个 MDL package**：vMaterials 2，只取可追溯的精简集合。第一批建议包含 `Carpaint_Shifting_Flakes`、`Copper_Antique_Brushed_Patinated`、`Aluminum_Scratched`、`Ceramic_Tiles_Glazed_Versailles`、`Velvet`，覆盖无 texture 与带 texture、层状/色变/划痕/釉/绒面。
2. **专项扩展**：Automotive Materials，等 MDL family 稳定后再接 clearcoat car paint、anisotropic carbon fiber 与 dirty metal；不先承担 20+ GB 全包。
3. **场景展示**：Sample Scenes/Showcase/OpenUSD assets 只在 source evaluator 稳定后用于 viewer 和工作流证据；需要的 Base Materials 作为场景依赖安装，不纳入首批 scientific reference shortlist。
4. **仍不可替代的长期 source**：显式 glinty NDF、measured spectral RGL、wave-optics heightfield 与 scratch segments；vMaterials 的 normal-map scratches、analytic color-shifting paint 不能取代这些微结构/光谱 GT。
