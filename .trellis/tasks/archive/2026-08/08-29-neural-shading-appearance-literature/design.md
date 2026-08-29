# Neural Shading 与 Appearance 文献深度研究：技术设计

## 1. 设计目标

本任务建立一个以第一方证据为中心的文献研究 pipeline：先锁定来源并重建单篇论文，再做跨论文综合，最后形成当前 NVIDIA 复现 correspondence 和可证伪的方法假设。

```text
论文候选与关系发现
        ↓
catalog + source ledger + evidence policy
        ↓
逐篇 evidence packet
        ↓
逐篇完整研究报告
        ↓
跨论文机制综合
        ↓
当前 NVIDIA correspondence + 可复现实验假设
```

这不是产品代码设计。任务不修改 `src/`、训练配置、runtime 或 viewer；临时 PDF 抽取和代码审计工具只放任务 `scratch/`。

## 2. 单任务与并行研究模型

波次 1、波次 2 和 load-bearing 对照虽然可以分别验收，但它们共同修改同一个 catalog、证据规则、关系图和综合结论。拆成多个顶层或子任务会让来源状态、报告 schema 和交叉引用产生多份真相。

因此当前任务直接拥有全部研究产物，以 catalog 中的 wave/status 和 `implement.md` 检查点管理增量完成度。论文之间可以由多个 `trellis-research` subagent 并行研究，但它们写入同一任务下互不重叠的报告文件，不建立多份 catalog 或 evidence policy。若以后出现需要独立代码实现或正式实验的候选，再单独创建实现任务；不在本任务树内混入产品实现。

### 2.1 所有权

- 主会话拥有 `catalog.md`、`evidence-policy.md`、`report-template.md`、所有 `comparisons/`、所有 `implications/` 和最终状态变更。
- 每个 research subagent 每次只拥有显式分配的 `research/papers/<paper-id>.md` 与 `scratch/workers/<paper-id>/`；不同 worker 不编辑同一文件。
- subagent 可以读取其他已完成报告用于定位关系，但不得修改共享 catalog、模板、综合文档或其他 worker 的报告。
- 主会话负责把 worker handoff 中的来源、状态、关系和 promotion trigger 合并回 catalog。

### 2.2 并发与批次

- Codex 同时最多运行三个 research subagent，保留主会话负责协调、来源去重、共享文件更新和及时审查。
- 默认一名 worker 一次研究一篇论文；完成后可以复用同一 worker 继续同一论文谱系，以利用其已建立的上下文。
- 先完成 Phase 1 的 evidence policy 与 template，再启动论文 worker；worker 不得自行发明报告 schema。
- 每个 dispatch prompt 以 `Active task: .trellis/tasks/08-29-neural-shading-appearance-literature` 开头，声明文件所有权、共享工作区、不得回退他人修改，并注入 PRD/design/implement、evidence policy、template 和相关项目证据。

### 2.3 两阶段审查

1. **Author pass**：研究 worker 完整阅读来源并写出个体报告及 handoff。
2. **Evidence review**：主会话或未参与该报告写作的 subagent 检查来源完整性、locator、数值上下文、事实/分析分隔和失败分类；发现问题返回原 worker 修订，或由 reviewer 在明确所有权转移后修正。

未经 evidence review 的报告不能进入 `complete`，也不能作为跨论文综合的强结论来源。

## 3. 任务目录

```text
.trellis/tasks/08-29-neural-shading-appearance-literature/
  prd.md
  design.md
  implement.md
  research/
    catalog.md
    evidence-policy.md
    report-template.md
    papers/
      <paper-id>.md
    comparisons/
      representation-and-coordinates.md
      optimization-and-loss.md
      filtering-and-lod.md
      sampling-and-integration.md
      deployment-and-amortization.md
    implications/
      current-nvidia-correspondence.md
      reproducible-hypotheses.md
  scratch/
    sources/       # 不提交：PDF、supplemental、talk、代码快照
    extracted/     # 不提交：文本、页面渲染、检索中间物
    tools/         # 不提交：本任务临时诊断/审计脚本
    workers/       # 不提交：按 paper-id 隔离的 worker 中间物
```

`research/` 是可持久化交付；`scratch/` 只支持本地证据读取。根仓库不新增第三方论文、clone、数据或运行报告。

## 4. Catalog 与状态机

每篇候选在 `catalog.md` 中只有一个权威条目，至少记录：

- `paper_id`、正式标题、作者、venue、year、DOI；
- relevance class：`local-material`、`scene-transport`、`volume-transport`、`load-bearing-related` 或 `discovery-only`；
- wave 与纳入理由；
- main paper、supplemental、project page、official code/config/data/talk 的可用性；
- 固定 URL、访问日期、代码 commit 和本地缓存 SHA-256；
- report path、当前状态、阻塞证据和关联论文；
- 对一般 inverse rendering、NeRF/3DGS、生成式材质或 neural reconstruction 的 `promotion_trigger`。

报告状态按以下顺序推进：

```text
discovered
  → triaged
  → sources-locked
  → paper-read
  → supplemental-read
  → code-audited / code-unavailable
  → report-draft
  → evidence-reviewed
  → complete
```

来源不可得不会触发猜测。条目进入 `blocked-source` 或在报告中明确 `code-unavailable`，只要所有可得第一方材料已检查、缺口已登记，仍可形成带证据边界的完整报告。

## 5. 证据合同

### 5.1 来源优先级

1. 论文正式版或作者公开版本；
2. supplemental、appendix、勘误；
3. 官方项目页、作者代码、固定配置、数据说明；
4. 作者 talk、slides、公开 correspondence；
5. 出版社/会议的书目信息；
6. 二手综述、搜索摘要或第三方解读只用于发现来源，不用于补全方法事实。

较低层来源不能静默覆盖较高层来源。论文与代码不一致时，同时记录两个版本，并说明代码是默认示例、正式复现配置、后续修订还是未知差异。

### 5.2 证据标签

报告内的具体结论使用以下语义标签：

- `P`：paper 正文；
- `S`：supplemental/appendix；
- `C`：official code/config/data；
- `A`：author page/talk/correspondence；
- `N`：NeuralShading 本项目已有可追溯证据；
- `I`：本项目分析或迁移假设。

事实段落必须能定位到 page/section/figure/table/listing，或 commit/file/symbol/config key。`I` 不得混进 `P/S/C/A/N` 事实段落。

### 5.3 PDF 与代码读取

- PDF 先提取全文用于检索，再渲染并检查会承载技术信息的公式、图、表、图注、脚注和 appendix；不能只依赖抽取文本。
- supplemental 与正文分别登记，不能把 supplemental 内容无标记地写成正文声明。
- 代码审计固定 commit，先定位论文正式配置，再区分默认/example/smoke 配置；记录网络构造、数据路径、loss、训练 lifecycle、runtime/export 和 benchmark 实现。
- 大型仓库优先使用官方网页和浅层本地审计；若需要 clone，只放任务 `scratch/sources/`，不得修改项目锁定的 `external/` 上游。

## 6. 单篇报告 schema

每个 `research/papers/<paper-id>.md` 使用固定 front matter：

```yaml
paper_id: <stable-id>
title: <official title>
year: <year>
venue: <venue>
report_status: draft|evidence-reviewed|complete
main_source: <URL/DOI/local locator>
supplemental_status: available|unavailable|not-applicable
official_code_status: audited|available-not-audited|unavailable
official_code_commit: <hash or not-applicable>
last_verified: YYYY-MM-DD
```

正文保持以下顺序：

1. 研究对象与报告边界；
2. 来源与版本；
3. 原论文要解决的问题和假设；
4. 输入、输出、坐标与 query domain；
5. representation、网络逐层配置和数据流；
6. 数据、GT/reference、sampling/query recipe；
7. loss、optimizer、schedule、batch、steps 与 hardware；
8. inference、部署、参数量、bytes、MAC、time 和 memory；
9. 实验 protocol、baseline、指标与完整结果；
10. 消融、失败尝试与负结果；
11. 论文、supplemental、代码之间的 correspondence/冲突；
12. 作者声明的限制与未报告信息；
13. 本项目分析；
14. 对当前 NVIDIA 复现的影响；
15. 可证伪的迁移假设；
16. 证据索引。

若某字段不适用于论文，写清为什么不适用；若适用但未披露，明确写“未报告”。不得删除字段来制造完整感。

## 7. 成功、失败与限制的分类

报告不能把“作者没有采用”自动解释为“作者尝试并失败”。统一使用：

| 类别 | 含义 | 可用证据 |
|---|---|---|
| `author-positive` | 论文正式支持的成功结果 | P/S/A，必要时 C 对应 |
| `author-negative` | 作者明确报告的失败、退化或不稳定 | P/S/A/C 中的可定位记录 |
| `ablation-inferior` | 正式消融中较差但不一定完全失败 | P/S 表格、图和 protocol |
| `known-limitation` | 作者声明或由方法 domain 直接决定的限制 | P/S/A；推导需单列 |
| `paper-code-gap` | 论文与官方代码/配置不一致或缺失 | P/S 对 C 的 correspondence |
| `project-reproduction-outcome` | NeuralShading 实际复现结果 | N，必须链接任务/artifact |
| `project-hypothesis` | 我们对机制的解释或预测 | I，不得改写成论文结论 |

解释失败原因时，先给原始观察，再区分作者解释与本项目解释。没有证据时只保留候选原因，不作因果结论。

## 8. Load-bearing 相关论文提升规则

一般 inverse rendering、NeRF/3DGS、生成式材质和 neural reconstruction 只有满足以下至少一项才提升为完整报告：

1. 核心论文直接继承其 representation、loss、encoder、feature field 或 runtime；
2. 它是核心实验的关键 baseline，且对论文主要结论有实质影响；
3. 它解释某项失败、消融选择或适用边界；
4. 它提供可直接迁移到 `prepare/evaluate/sample/pdf`、LOD、compiler 或有界部署的机制；
5. 用户后续明确指定。

只满足关键词或大领域相似性时，保留为 `discovery-only`，并在 catalog 写明未提升理由。超出这些规则的主动扩张返回 planning。

## 9. 跨论文综合

综合文档只引用已经达到 `evidence-reviewed` 或 `complete` 的个体报告，不直接从搜索摘要写结论。

- `representation-and-coordinates`：query domain、方向/空间参数化、latent、analytic prior、factorization；
- `optimization-and-loss`：目标变换、loss、sampling、初始化、variance、distillation；
- `filtering-and-lod`：footprint、mip、scale supervision、spatial-angular correlation；
- `sampling-and-integration`：evaluator、proposal、pdf、environment/GI coupling；
- `deployment-and-amortization`：shader path、texture-space、temporal reuse、precision、hardware 和真实 cost domain。

比较矩阵必须区分论文实际比较与本项目重新归类，不能把不同场景、硬件、输入或 quality protocol 的数值横向拼成排名。

## 10. NVIDIA correspondence 与方法假设

`current-nvidia-correspondence.md` 以论文条目为轴，对齐：

- NVIDIA 2024 正文/supplemental；
- NVIDIA 2026 optimization 论文与官方仓库；
- 归档的忠实复现任务证据；
- 当前 `MethodDefinition`、训练配置、runtime/package 和已登记实验结论。

每项差异标记为：`faithful`、`author-underspecified`、`interface-adaptation`、`budget-adaptation`、`intentional-deviation`、`suspected-defect` 或 `not-applicable`。

`reproducible-hypotheses.md` 中每个候选包含：

- 来源报告和具体机制；
- 已知事实与迁移假设；
- 预期解决的 failure category；
- 最小 matched control；
- 需要冻结的 source/query/budget/seed；
- 质量、时间、内存观测项；
- 静态运行预算类别和停止/证伪条件。

这些是假设和后续任务输入，不是本任务内的实现承诺或质量 hard gate。

## 11. 一致性检查与异常处理

- catalog 的初始必研集合必须与 PRD 一致，报告文件与状态可双向追溯。
- 报告标为 `complete` 前，必须检查 main paper、supplemental 可用性、official code 可用性和所有固定章节。
- 找不到来源时记录检索路径和缺口，不用二手内容补全。
- 来源后续更新时保留原访问版本，新增版本记录；不能静默改写旧结论。
- 发现论文身份错误、重复版本或方法边界变化时先修正 catalog，再更新引用它的综合文档。
- 如果新证据实质改变已确认范围或要求主动扩张领域，返回 PRD planning；普通 load-bearing 提升按本设计执行。
- 并行 worker 的结论冲突时不按多数票合并；回到各自的一手 locator，保留来源版本差异，必要时由独立 reviewer 复核。
- worker 超时、来源不可得或上下文不足时，主会话重新分派同一文件所有权；不得让两个存活 worker 同时修订同一报告。

## 12. 风险与回滚边界

- **研究规模失控**：由初始必研集合、promotion trigger 和 catalog 状态控制；一般相关工作不自动升级。
- **报告看似详细但来源不实**：固定证据标签、locator 和“未报告”规则；无法定位的内容不能进入事实段。
- **先形成方法偏好再选择性阅读**：个体报告先于综合；本项目分析固定放在事实章节之后。
- **论文与代码版本漂移**：锁定 DOI/URL、访问日期和 commit；更新版本单独登记。
- **现有 NV 实现影响论文解读**：`N` 与 `I` 单列，不能回填为 `P/S/C`。
- **大文件污染仓库**：所有缓存进入 task `scratch/`，提交前检查根仓库边界。
