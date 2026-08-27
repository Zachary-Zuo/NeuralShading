# Fancy Reference Material 候选研究实施计划

## 0. Planning Gate

- [x] 用户确认采用“近期可落地与长期压力并列”的组合。
- [x] 用户明确相似效果/材质类别即可，不要求 NVIDIA 同一物体和纹理。
- [x] NVIDIA 2024/2026 论文、仓库 `.mtlx`、Ling-Qi Yan 微结构工作与邻近历史模型已经完成第一方证据收集。
- [x] 三个 NVIDIA `.mtlx` 已完成图结构、4K 纹理、许可、官方 MaterialX validator 与当前 adapter 的分层审计。
- [x] Omniverse 三套 MDL material pack 已完成官方目录、远程 ZIP index、代表 `.mdl`、依赖形态与逐包许可审计；未下载或 vendor 数 GB 资产。
- [x] 用户于 2026-08-27 确认纳入 Omniverse/MDL 后的更新计划；正式报告已经完成。

Final planning summary：报告将把“论文原资产是否开源”降为一个证据项，并新增原生 MDL 路线。近期继续用现有 OpenPBR/LayerStack；下一阶段优先评估 vMaterials 2 的 MDL program、full closure MaterialX、RGL measured 与 procedural glints；长期保留 wave-optics heightfield/scratch segments。NVIDIA 三个 `.mtlx` 保留为 source/compatibility fixtures；Omniverse USD 场景只作绑定/展示资源；MDL 或 MaterialX 都不静默蒸馏为当前简单 PBR 后继承原 GT identity。

## 1. 汇总正式报告

- [x] 写 `research/report.md`，合并 NVIDIA 2024/2026 实际材质、公开资产与历史模型映射。
- [x] 给每个候选标 direct / reconstructable / inspiration-only，并列出原生输入与 query contract。
- [x] 给出近期、下一阶段、长期 shortlist；至少一个无 texture 和一个带空间资源候选。
- [x] 纳入 Omniverse/MDL 审计，明确 vMaterials 2、Automotive、Base 与 USD scene packs 的不同角色。

## 2. 接入边界建议

- [x] 明确 NVIDIA Bark / PatternedMetal / FauxLeather 的不同用途和不能直接进入当前 MaterialX family 的原因。
- [x] 为 `materialx.closure-graph@1`、`rgl.measured-spectral@1`、procedural glinty NDF 与 wave-optics oracle 写最小 package/manifest 边界。
- [x] 为 `mdl.program@1` 写最小 package/manifest 边界：pack/version/hash、module、export、authored arguments、imports/resources、MDL SDK/falcor2 identity 与 `eval/sample/pdf` 能力。
- [x] 明确 OpenPBR thin film / anisotropy 与真实 flakes、scratch diffraction 的非等价关系。

## 3. 质量检查

- [x] 核对所有外部链接、论文作者归属、许可和本地 commit/hash。
- [x] 对照 `docs/material_scope.md`、`docs/realtime_material_compilation.md` 与 repository policy 检查 source provenance 和目录边界。
- [x] 做 PRD acceptance checklist 回填，确保没有把未运行的上游 renderer 写成“已复现”。

## 4. 后续实现任务的触发条件

- 本任务获用户批准后只完成研究报告，不自动 vendoring 资产、不改 reference code、不启动训练。
- 用户再选定首批路线后，为每个新 source family 单独 brainstorm；MaterialX closure、RGL measured 与 wave optics 不合并成一个实施任务。
