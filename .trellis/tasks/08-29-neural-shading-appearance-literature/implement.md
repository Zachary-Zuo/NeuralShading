# Neural Shading 与 Appearance 文献深度研究：执行计划

## 执行原则

- 本任务只写研究档案、综合、correspondence 和可证伪假设，不修改产品模型或训练/runtime 代码。
- 每次只把已经完整阅读第一方材料的论文推进到 `complete`，不批量生成摘要后再补证据。
- 波次表示执行顺序，不表示范围、深度或验收优先级差异。
- 个体报告完成前不写跨论文强结论；搜索摘要和二手资料只用于发现来源。
- 论文报告允许最多三个 `trellis-research` subagent 并行；主会话保留共享文件、合并、复核和最终综合所有权。
- 每个 worker 只编辑分配的报告与 paper-local scratch，不回退或覆盖其他 agent 的修改；报告经过独立 evidence review 后才能标记 `complete`。
- 任何验证前先按 `.trellis/spec/project/dev-environment.md` 判定本机状态并在进度回复中给出证据。

## Phase 0：规划收敛与环境门

- [x] 对 `prd.md` 做最终 convergence pass，确认没有开放问题、重复事实或无来源 hard gate。
- [x] 复核 `design.md` 的 evidence contract、报告 schema、promotion trigger 和仓库边界。
- [x] 复核本执行计划与 A1–A12 的映射。
- [x] 按 `dev-environment.md` 判定开发机状态；本机为完整 Windows，本任务不需要 GPU/构建，但所有后续验证仍遵守项目环境规则。
- [x] 运行 Trellis task validation 与 task-scoped `git diff --check`。
- [x] 向用户提交最终 planning summary；用户已明确批准执行，并授权多篇论文使用 subagent 并行研究。

回滚点：只含任务规划文件；尚未写研究报告。

## Phase 1：证据基础设施

- [x] 创建 `research/catalog.md`，登记 PRD 的全部初始必研论文、分类、wave、来源状态和报告状态。
- [x] 创建 `research/evidence-policy.md`，落实正文/supplemental/code/author/project/inference 标签、locator、冲突和缺失信息规则。
- [x] 创建 `research/report-template.md`，固定 front matter、16 个报告章节和完成检查清单。
- [x] 建立论文关系图和 load-bearing promotion 记录字段。
- [x] 建立 `scratch/sources/`、`scratch/extracted/`、`scratch/tools/`、`scratch/workers/` 使用约定与 task-local `.gitignore`，禁止把缓存作为根仓库第三方资产提交。
- [x] 把现有 `docs/research/prior_art.md`、归档 NVIDIA 复现任务和旧 reference-material 调研登记为 `N` 类项目证据，不复制成论文事实。
- [x] 建立并行 dispatch brief：固定 active task、文件所有权、必读上下文、禁止共享文件写入、handoff 字段和 evidence review 门。

验收：A1、A2、A11、A12 的结构已存在；所有初始论文都能从 catalog 定位。

## 每篇论文的固定研究循环

后续 Phase 2–4 对每篇被提升为完整报告的论文严格执行：

1. 锁定正式书目信息、main paper、supplemental、项目页、代码、配置、数据与 talk；记录 URL、访问日期、commit 和缓存 hash。
2. 完整阅读 main paper；提取后检查公式、图、表、图注、脚注和参考文献中承载的方法信息。
3. 完整阅读 supplemental/appendix/勘误；把新增或冲突配置单独登记。
4. 若有官方代码，固定 commit，定位论文正式配置并审计 architecture、data/query、loss、training lifecycle、runtime/export 与 benchmark。
5. 按 template 先写 `P/S/C/A` 事实章节；所有适用但缺失的字段明确写“未报告”。
6. 登记 success、author-negative、ablation-inferior、known-limitation 和 paper-code-gap；不得从最终设计推测失败历史。
7. 最后写 `N/I`：对当前项目的影响、迁移边界和可证伪假设。
8. 做 evidence review：逐项检查 locator、数值上下文、报告状态与 catalog 回链，再标记 `complete`。

若某篇来源缺失，完成可得证据审计并明确边界；不得停下来用猜测填空。

并行执行时，由主会话按每批最多三篇分配互不重叠的 `paper-id`。worker 完成 author pass 后先交还来源清单、未解析冲突和报告路径；主会话再安排未参与写作的 reviewer 或亲自完成 evidence review。共享 catalog 只由主会话更新。

## Phase 2：波次 1——局部 neural material / appearance

### 2.1 当前 NVIDIA 基线与优化稳定性

- [x] Real-Time Neural Appearance Models。
- [x] Taming Optimization Variance in Compact Neural Shading Networks。
- [x] Neural BRDF Representation and Importance Sampling。

阶段检查：区分 evaluator representation、training algorithm、sampler proposal 和部署系统；先建立 NVIDIA 2024/2026 的论文内部 correspondence。

并行建议：前三篇可作为首个三 worker 批次；它们共享 NVIDIA 语境但报告文件互不重叠，批次结束后统一做 correspondence review。

### 2.2 空间、BTF、层状与 compiler 表示

- [x] NeuMIP。
- [x] Neural Biplane Representation for BTF Rendering and Acquisition。
- [x] Neural Layered BRDFs。
- [x] MetaLayer。
- [x] Towards Comprehensive Neural Materials。

阶段检查：比较空间/方向 domain、feature planes、offset、latent algebra、hypernetwork、silhouette/displacement 与 runtime query 的不同边界。

### 2.3 LOD、sampling、hybrid prior 与移动部署

- [x] Neural Prefiltering for Correlation-Aware Levels of Detail。
- [x] BSDF Importance Baking。
- [x] A Hybrid Neural-Microfacet BRDF Model。
- [x] Neural Material Adapter。
- [x] Real-Time Neural Materials on Mobile VR。

阶段检查：分开记录 evaluator quality、matched sampling、filter semantics、analytic prior、distillation 和 texture-space/temporal amortization。

验收：A3、A6、A7；更新 catalog 和关系图，但尚不完成最终跨论文结论。

## Phase 3：波次 2——场景级与体积 neural light transport

### 3.1 对象化与动态 scene transport

- [x] NeLT: Object-Oriented Neural Light Transfer（用户提供16页正式正文；已完成逐页作者稿与独立 evidence review）。
- [x] Neural Global Illumination via Superposed Deformable Feature Fields（用户提供11页正式正文；已完成逐页作者稿与独立 evidence review）。
- [x] Dual-Band Feature Fusion for Neural Global Illumination with Multi-Frequency Reflections。
- [x] LightFormer: Light-Oriented Global Neural Rendering in Dynamic Scene。

### 3.2 Lighting field、probe 与体积推断

- [x] NeLiF: Neural Lighting Function Generation for Real-Time Indoor Rendering（用户提供11页正式正文；已完成逐页作者稿与独立 evidence review）。
- [x] Neural Light Probes for Real-Time Global Illumination。
- [x] `paper1469_1.pdf`：Real-Time Volumetric Light Transport Inference from Auxiliary Renderings。

阶段检查：明确每篇方法的 scene assumptions、可编辑轴、输入 buffer/feature field、目标 transport、跨场景泛化、history/reprojection、真实 inference cost 和与 local scattering 的语义差异。

验收：A4、A6、A7；场景级论文不得降格为摘要。

## Phase 4：load-bearing 对照追踪

- [x] 从波次 1/2 每篇报告的 related work、baseline、消融和代码依赖生成候选队列。
- [x] 按 design §8 的 promotion trigger 决定 `complete report` 或 `discovery-only`。
- [x] 对提升论文执行同一完整研究循环，不使用缩短模板。
- [x] 在 catalog 为未提升的一般 inverse rendering、NeRF/3DGS、生成式材质和 neural reconstruction 条目登记理由。
- [x] 若需要主动扩张到 promotion trigger 之外，停止并返回 planning；本轮未在执行中静默扩大范围。

验收：A5、A12。

## Phase 5：跨论文综合与项目启发

只有依赖的个体报告达到 `evidence-reviewed` 或 `complete` 后才开始对应综合。

- [x] `representation-and-coordinates.md`。
- [x] `optimization-and-loss.md`。
- [x] `filtering-and-lod.md`。
- [x] `sampling-and-integration.md`。
- [x] `deployment-and-amortization.md`。
- [x] 综合 local material 与 scene transport 的可迁移机制，同时明确不能迁移的 query semantics、visibility、scene dependence 和 cost domain。
- [x] 完成 `current-nvidia-correspondence.md`，逐项标记 faithful/underspecified/adaptation/deviation/defect。
- [x] 完成 `reproducible-hypotheses.md`，按证据强度和预期项目价值排序；每项写最小 matched 对照、冻结项、部署类别和证伪条件。

验收：A8、A9、A10。所有质量、时间和内存预期都是 report-only/排序信息，不构成本研究任务 hard gate。

## Phase 6：全量研究质量门

- [x] 逐项对照 A1–A12，确认每项可回链到 catalog、报告、综合或 implication 文档；本轮已用三份新解锁正文重新关闭 A4。
- [x] 检查初始必研集合全部存在，catalog 状态与文件 front matter 一致。
- [x] 抽查所有数值、architecture、training 和 runtime 结论都有 `P/S/C/A/N` locator。
- [x] 检查所有 `I` 分析位于事实章节之后，没有把本项目解释写成作者结论。
- [x] 检查每篇报告的失败分类，没有把“未采用/未报告”写成失败尝试。
- [x] 检查每篇 `evidence-reviewed` 完整报告都有独立 evidence review 记录，且 author/reviewer 所有权可追溯。
- [x] 检查跨论文表格没有直接排名不可比的硬件、数据、输入或指标。
- [x] 检查任务外没有新增第三方 PDF、代码 clone、数据、`reports/` 或 `artifacts/`；用户提供的四份根目录 PDF 均保持本地输入，不被任务复制或纳入暂存。
- [x] 运行 Markdown 内容检查与 Trellis artifact validation；任务文件当前为 untracked，另以内容级 whitespace/link/status 审计覆盖 `git diff --check` 不检查 untracked 文件的边界。
- [x] 使用 `trellis-check` 做 spec、需求、引用、文件组织和一致性审查并修复问题。

2026-08-29 来源恢复后的 Phase 6 已关闭：三份正文报告与七份共享综合均完成独立 evidence review；42 份持久化 Markdown、28 份完整个体报告、259 个相对链接、17 项可证伪假设及任务结构检查全部通过。论文未公开的 supplemental/code、state bytes、生成/更新时间和 runtime breakdown 继续作为显式证据缺口保留，不构成本轮验收失败。

建议的最终静态检查命令：

```powershell
conda run -n neural-shading python .\.trellis\scripts\task.py validate 08-29-neural-shading-appearance-literature
rg -n "TBD|TODO|待补|无来源" .trellis\tasks\08-29-neural-shading-appearance-literature\research
git diff --check
git status --short
```

`rg` 命中不是自动失败：合法的“来源未报告/材料不可得”必须保留，但需要逐项确认不是遗漏占位符。

回滚点：研究 Markdown 与 task artifact；不涉及产品代码、checkpoint 或 runtime 资产。

## 验收映射

| PRD | 主要执行阶段 |
|---|---|
| A1–A2 | Phase 1 |
| A3 | Phase 2 |
| A4 | Phase 3 |
| A5、A12 | Phase 4 |
| A6–A7 | 每篇固定研究循环 + Phase 2–4 |
| A8–A10 | Phase 5 |
| A11 | Phase 1、Phase 6 |
