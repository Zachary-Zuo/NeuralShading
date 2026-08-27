# Fancy Reference Material 候选研究设计

## 1. 交付物

本任务只交付研究结论，不实现 source family。最终文档写入 `research/report.md`，并以三份可追溯底稿支撑：

- `research/nvidia-2026-materialx.md`：论文资产、公开 `.mtlx`、兼容性、许可与推荐用途；
- `research/material-lineage.md`：外观类别到历史模型、代码/数据和接入形态的映射。
- `research/omniverse-usd-packs.md`：Omniverse packs 的格式、内容、许可、MDL 图审计与具体候选。

## 2. 证据等级

每个候选必须标为以下一种，避免把论文视觉目标误写成 ready-to-use GT：

1. **direct**：原始 source、权威 evaluator 与许可都可得，可直接规划 package；
2. **reconstructable**：论文模型和足够输入/代码可得，但需移植、重建或许可隔离；
3. **inspiration-only**：只有图像或不完整资产，不进入 reference shortlist。

`.mtlx` 再额外拆分四个问题：源文档是否 conformant、锁定 MaterialX 是否能加载、当前 adapter 是否覆盖、reference backend 的具体 layering/颜色语义是什么。

## 3. 比较轴

候选矩阵固定比较：

- 视觉辨识度与压力维度：layering、spatial variation、方向高频、spectral、footprint/LOD；
- 原生输入：参数、graph、texture、heightfield、segment field 或 measured table；
- query contract：`x/uv`、`footprint`、`wi/wo`、wavelength、`sample/pdf`；
- reference 权威性与可编辑性；
- 许可、资源体量、可固定 commit/hash 的程度；
- 与当前 LayerStack/OpenPBR/MERL/MaterialX 的增量，而非只比较预览“好不好看”；
- 是否只是 USD 中对 MDL 的 binding，还是包含可独立固定的 `.mdl` module、export、imports、参数与纹理；
- GPU 在线采集与最终固定成本 runtime 的风险。

## 4. 方案边界

```text
现有 source family
  ├─ LayerStack：新增本族原生 fancy recipes，不冒充外部 Layer Laboratory GT
  └─ OpenPBR：复用官方 thin-film/coat/anisotropy 示例，明确其不含 flakes/scratches

下一阶段 source diversity
  ├─ MaterialX full closure graph：graph + textures 是 GT
  ├─ MDL program：module + export + authored parameters + resources 是 GT
  ├─ RGL measured spectral：measurement table 是 GT
  └─ procedural glinty NDF：seed + analytic program 是 GT

长期 spectral/footprint
  ├─ wave-optics heightfield
  └─ scratch-segment diffraction
```

任何 MaterialX 方言归一化、Fourier precompute、shader generation 或 neural compilation 都是从 source 可重建的 artifact；原始 source identity 与 evaluator provenance 必须保留。

OpenUSD 在 Omniverse packs 中主要承担 composition 与 material binding。若 UsdShade 指向 MDL，USD adapter 只能解析绑定和参数，实际 scattering GT 仍由锁定的 MDL program 与 SDK reference 定义。MDL distilling 或把材质烘焙成 USDPreviewSurface 会改变 source representation，只能成为显式 control，不能继承原 MDL 的 GT identity。

## 5. 推荐结论的形式

最终报告不选“一个万能材质”，而给三阶段组合，并为每项明确：

- 代表外观与为什么能暴露模型边界；
- 是否需要 texture / heightfield / spectral data；
- 是否能编辑以及哪些参数是原生参数；
- direct / reconstructable / inspiration-only；
- 推荐 package / family 边界与第一项验收 fixture；
- 不能宣称的等价关系，例如 thin film ≠ scratch diffraction、anisotropic roughness ≠ explicit brushed microgeometry。

## 6. 验证与回滚

- 所有网络事实使用论文、作者项目页、官方仓库、官方数据库等第一方来源。
- MaterialX 兼容性区分静态图审计、官方 validator 与当前 adapter 源码，不以文件扩展名推断。
- 若后续 parity 证明方言归一化不能保留 NVIDIA falcor2 输出，该资产降级为 compatibility fixture，不进入 reference corpus。
- 若许可不允许再分发，保留 fetch script + hash 或进程外 oracle 方案；不能满足时降级为 inspiration-only。
