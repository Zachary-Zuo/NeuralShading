# Fancy Reference Material 候选研究

## 目标

为 NeuralShading 选择一组比当前 reference material 更有视觉辨识度、能有效暴露 neural evaluator 能力边界的源材质候选。候选可以是空间均匀材质或带 texture 的空间变化材质，但必须保留各自原生语义，并由对应的权威 reference 产生 GT。

## 背景与已确认事实

- 当前正式 source family 已覆盖 LayerStack、MERL、OpenPBR 与 MaterialX，但用户认为现有展示材质不够丰富，无法充分体现复杂外观。
- NVIDIA `NVlabs/neuralappearance` 与其论文已经进入项目 prior art；本任务需要进一步核对论文实际使用的 reference material、公开实现和资产可得性，而不是只借用其 neural representation。
- Lingqi Yan 团队关于结构色、微观划痕、glint 与微结构外观的工作可能提供更强的方向高频、空间变化和尺度依赖压力测试。
- 新 source 不得为了适配现有 LayerStackIR 或 neural backend 而预先反演成层模型；原生参数、图结构、纹理和测量资源都属于 GT。
- NVIDIA 2024 论文的五个复杂材质是内部制作的 layered MaterialX/Houdini 资产；公开仓库当前只随附 Bark、FauxLeather、PatternedMetal 三个较简单示例，并未发布论文五材质的完整图与纹理。
- 当前 OpenPBR package 已包含 brushed aluminum、car paint、pearl、soap bubble、velvet 与 glass 等高辨识度官方示例；应先区分“材质选择不够好”与“source family 能力不够”。
- RGL spectral material database 提供 CC0 measured car paint、各向异性金属和 Morpho 蝶翼，并有原生 `eval/sample/pdf` API，是比直接移植 wave-optics renderer 更低风险的 spectral/anisotropic 增量。
- Wave-optics heightfield 与 scratch-segment source 最能提供彩色 glint 和尺度压力，但会引入 spectral/footprint 查询、matched sampler、GPL 或非商业许可证隔离等额外边界。
- 2026 论文实际使用 9 个 da Vinci Workshop 烘焙 SVBRDF 与 6 个内部多层材质；公开仓库的 Bark、FauxLeather、PatternedMetal 是 runnable examples，不是这 6 个多层 GT 的资产发布。
- 三个公开 `.mtlx` 在 NVIDIA falcor2 的 Houdini compatibility path 中可用，但未通过项目锁定的官方 MaterialX 1.39.4 validator，也不满足当前 `standard_surface`-only adapter；它们适合作为未来 full closure / 方言归一化 acceptance fixtures，不能无修改登记为当前 MaterialX source。
- Layer Laboratory 已公开 BSD-3 代码和 gold + dust、gold + scattering medium + dielectric 等无 texture 示例，是复现 NVIDIA 相似层状外观类别的强来源；RGL measured 与 procedural glinty NDF 则分别提供 measured 和无 texture procedural 路线。
- Omniverse Downloadable Asset Packs 页面中的三套正式材质包全部是 MDL，不是 MaterialX：Base Materials、vMaterials 2 与 Automotive Materials。其余大量 OpenUSD pack 主要是几何、场景、环境与 MDL binding，不能仅因文件为 `.usd` 就当作新的材质 reference。
- `NVlabs/neuralappearance` 已原生支持 falcor2 + MDL SDK 在线求值，README 和默认配置直接以 vMaterials 2 的 `Wood_Tiles_Pine_Mosaic` 为示例；因此 vMaterials 2 与 2026 pipeline 的对齐度高于三个公开 `.mtlx`，但当前 NeuralShading 尚无 MDL source family/provider/reference program。
- 远程审计 Omniverse 三个材质包的 ZIP central directory 与 package license 后确认：包内有可读 `.mdl`、大量纹理、preset thumbnails 和逐包许可文件，没有 `.mtlx`；vMaterials 2 中还存在原生 layered/thin-film/sheen/transmission 等闭包与可编辑 presets，而非仅有 USDPreviewSurface 参数贴图。

## 需求

- R1：从论文、项目页、作者代码和公开数据等第一方来源，分别审计 NVIDIA 2024《Real-Time Neural Appearance Models》与 2026《Taming Optimization Variance in Compact Neural Shading Networks》实际展示或训练的 reference material。
- R1a：逐个检查 `NVlabs/neuralappearance` 当前公开的 `.mtlx`、纹理、图结构和所依赖的 MaterialX nodes，判断它们能否直接进入当前 `materialx.textured-surface@1`，需要扩展现有 MaterialX reference，还是只能借用外观思路。
- R1b：论文原资产是否公开只是可复现性证据之一；候选不要求是同一个物体或同一套纹理，只要能用权威模型与合法资产复现相同的外观类别和压力维度。
- R2：调查 Lingqi Yan 团队中与结构色、微观划痕、glint、高分辨率微结构及其预过滤相关的工作，明确每项工作的输入表示、reference 求值方式、空间/方向/尺度维度和资源可得性。
- R3：对 NVIDIA 材质中的每类外观——如 layered glaze、dust/stain/grease、scratched/brushed metal、oxidation/verdigris、flake paint、强 normal variation 与 LOD——追溯图形学中的代表性材质模型、论文 reference、公开代码和可用数据；不因作者范围限制而遗漏更适合 NeuralShading 目标合同的实现。
- R4：按视觉辨识度、科学压力测试价值、原生可编辑性、权威 reference 可获得性、贴图/数据体量、许可、GPU 采集可行性与实时 compiler 适配风险比较候选。
- R5：给出多候选组合，而不是强行选一个万能材质；明确哪些适合作为近期接入、研究压力测试和长期展示资产。
- R6：建议必须区分“可以直接登记为 source family”“需要先复现或移植 reference”“只能作为论文灵感或展示目标”。
- R7：审计 Omniverse 可下载资产包，区分 MDL material package、UsdShade/MDL binding、场景资产和环境资产；为有价值的 MDL 材质说明原生闭包、纹理、参数、依赖、许可与当前项目接入缺口。

## 验收标准

- [x] A1（需求交付，来源：用户要求）：准确列出 NVIDIA 论文中的 reference material 及其关键构成，并附第一方证据。
- [x] A1b（需求交付，来源：用户补充）：给出 2026 仓库每个公开 `.mtlx` 的结构审计、当前项目兼容性与推荐用途。
- [x] A1c（需求交付，来源：用户补充）：为 NVIDIA 展示的主要外观类别建立“论文效果 → 历史材质模型/论文 → 现成代码或数据 → 本项目接入形态”的映射，不要求原资产完全相同。
- [x] A2（需求交付，来源：用户要求）：形成 Lingqi Yan 相关工作的候选矩阵，至少覆盖结构色或波动光学外观、微观划痕/glint、尺度过滤中的两个不同方向。
- [x] A3（需求交付，来源：用户要求与仓库边界）：每个入围候选都说明是否需要 texture、height field 或 spectral data，能否编辑，reference 实现来源及许可或可再分发风险。
- [x] A4（需求交付，来源：用户允许多个且可有无 texture）：给出分阶段 shortlist，至少包含一个无 texture 的过程式或参数化候选和一个带空间资源的候选。
- [x] A5（语义正确性，来源：项目接口）：结论与 `docs/material_scope.md`、运行时 `prepare/evaluate/sample/pdf` 合同及当前 evaluator 研究顺序一致。
- [x] A6（需求交付，来源：Trellis 研究合同）：研究结果持久化到任务目录中的中文文档，并给出可直接进入后续接入任务的建议边界。
- [x] A7（需求交付，来源：用户补充）：回答 Omniverse packs 是否可用，并给出至少一组与 NVIDIA layered/flake/scratch/glaze 外观相符的具体 MDL shortlist；不得把 USD 容器、MDL 原生 GT 与蒸馏后的 USDPreviewSurface 混为一谈。

## 范围外

- 本任务不直接实现新的 source family、采集器、shader 或训练模型。
- 不把论文截图中无法获得权威实现或输入资产的外观伪装成可复现 GT。
- 不把结构色或衍射的物理 reference 简化为普通 RGB clearcoat 后仍声称保持原生语义。

## 已收敛的决策

- 采用“近期可落地与长期压力测试并列”的平衡组合；近期建议可继续，但必须补齐 NVIDIA 2026 `.mtlx` 审计和相似外观的历史可复现来源。
- 候选以“相同效果或材质类别可由权威 source 复现”为标准，不要求使用 NVIDIA 论文的同一物体、同一图或同一纹理资产。
- 对 NVIDIA 三个 `.mtlx`，采用“保留原文档为 source identity、兼容归一化或生成 shader 为可重建 artifact、以锁定上游实现做 parity”的边界；不把静默改写后的 `standard_surface` 当原 GT。
- shortlist 固定为三阶段：现有 LayerStack/OpenPBR 立即展示；full closure MaterialX + RGL measured + procedural glints 作为下一阶段；wave-optics heightfield / scratch segments 作为长期 spectral/footprint 压力。
- Omniverse vMaterials 2 提升为下一阶段的优先 source candidate：以原始 MDL module、export、参数、imports 和纹理为 GT，由 MDL SDK/falcor2 直接求值；任何 distill/bake/USDPreviewSurface 转换只能登记为对照或派生 artifact。
- Automotive Materials 作为 hero car-paint/carbon-fiber 场景与后续专项候选；Base Materials 主要满足 Omniverse 场景依赖和基础覆盖，不作为首批 fancy reference。OpenUSD scene packs 保留为最终 viewer/工作流展示资产，不冒充独立 source material family。
