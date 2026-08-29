# 论文研究证据规则

本文把 PRD 的“先详细重建、最后分析”变成每个报告必须执行的检查合同。任何 worker、reviewer 和综合文档都遵守同一规则。

## 1. 核心原则

1. 先获取证据，再形成描述；不能先写常识版方法，再寻找看似支持它的引用。
2. 单篇报告先完成 `P/S/C/A` 事实层，再写 `N/I` 项目层。
3. 搜索摘要、二手博客、引用数量和论文榜单只用于发现入口，不用于补全网络、训练或结果。
4. 论文未披露的信息保持未知；“未报告”是有效研究结论，不是待模型猜测的空格。
5. 代码是独立证据，不自动代表论文正式配置；论文、supplemental 和代码之间的差异必须保留。

## 2. 证据标签与 locator

| 标签 | 来源 | 合法 locator 示例 |
|---|---|---|
| `P` | main paper | `[P §3.2, p.5, Fig.4]`、`[P Table 2]` |
| `S` | supplemental、appendix、勘误 | `[S §1.3, Listing 1]` |
| `C` | 官方代码、配置、数据说明 | `[C commit abc123, path/to/config.json:key]`、`[C symbol Model.forward]` |
| `A` | 作者项目页、talk、slides、公开 correspondence | `[A project page, accessed 2026-08-29]` |
| `N` | NeuralShading 项目现有证据 | `[N <repo-relative-path>#<heading>]`；run 结果还要给 artifact identity |
| `I` | 本项目分析、推导或迁移假设 | `[I]`，并回链支撑它的 P/S/C/A/N 条目 |

同一段落包含多个来源时，分别标记各自支持的范围。只在段末放一个模糊链接、无法判断它支持哪些数值或配置，不算合格 locator。

## 3. Source-lock 流程

每篇报告开始前建立 source ledger：

| 字段 | 要求 |
|---|---|
| Bibliography | 正式标题、作者、venue、year、DOI；以论文首页/出版社/作者项目页交叉核对 |
| Main paper | 固定 URL；本地缓存记录 SHA-256 与获取日期 |
| Supplemental | 逐项标记 available/unavailable/not-applicable，不能只检查项目页是否写了“supplemental” |
| Code | 官方性依据、repo URL、固定 commit、license、论文正式配置是否存在 |
| Config/data | 入口、版本、正式/example/smoke 身份 |
| Talk/correction | 可用性、发布日期、是否晚于论文正文 |
| Access gap | 检索过的第一方入口、失败原因和替代边界 |

本地 PDF、代码 clone、页面渲染和文本抽取只放 task `scratch/`。报告持久化 URL、hash、commit 和 locator，不提交第三方大文件。

## 4. PDF 阅读规则

1. 提取全文只用于检索和建立章节索引。
2. 必须视觉检查承载技术事实的页面：网络图、公式、表格、图注、脚注、appendix 和 supplemental；抽取乱码或版面丢失时回到原 PDF。
3. 阅读时记录原页码和论文印刷页码的差异；报告 locator 使用读者能复现的位置。
4. 不从 abstract 推导逐层配置，不从示意图推导未标注 tensor shape，不从结果图目测出未报告数值。
5. 引用作者原话只保留必要短语；主体用中文精确转述，避免长篇复制论文文本。

## 5. 代码与配置审计

1. 固定 commit 后定位 README 声明的 paper reproduction entry。
2. 区分 paper/formal、default、example、smoke、demo、later revision 和 third-party fork。
3. 从构造代码或正式 config 记录：输入顺序、shape、层数/宽度、activation、normalization、输出 transform、precision 和 parameter sharing。
4. 从训练入口记录：数据/reference、query/sampling、batch、loss 权重、optimizer、schedule、steps/epochs、stage boundary、seed 和 checkpoint selection。
5. 从 runtime/export 记录：资源布局、quantization、decoder path、sampling/pdf、调用频率和 benchmark scope。
6. 代码缺少论文声明的组件时登记 `paper-code-gap`；不能默认为作者用另一私有分支，除非有作者证据。

## 6. 数值与比较的上下文

任何质量、速度、内存或参数量必须同时记录足以解释它的上下文：

- 数据/材质/场景与 split；
- 输入分辨率、query 数或 sampling budget；
- 模型版本和 precision；
- 评测指标定义、聚合方式和单位；
- hardware、API/backend、batch/coherence；
- 是否包含预计算、prepare、texture-space pass、temporal reuse 或 I/O。

缺少上下文的数字可以作为“论文报告值”登记，但不得进入跨论文排名。不同论文的 FPS、MAC、PSNR 或 storage 默认不可比，除非综合文档明确证明 protocol matched。

## 7. 成功、失败和限制

| 分类 | 判定规则 |
|---|---|
| `author-positive` | 作者在明确 protocol 下支持的主要或次要成功结果 |
| `author-negative` | 作者明确说失败、不稳定、退化或无法处理，并提供文本/实验 locator |
| `ablation-inferior` | 消融比最终配置差；不能自动写成“失败” |
| `known-limitation` | 作者声明，或由输入/输出 domain 直接推出；项目推导必须标 `I` |
| `paper-code-gap` | P/S 与 C 不一致、正式配置缺失或公开资产不足 |
| `project-reproduction-outcome` | NeuralShading 真正执行过且可定位的结果；质量差不等于实现错误 |
| `project-hypothesis` | 对原因或迁移效果的候选解释；必须写证伪方式 |

禁止：

- 把“最终论文没有采用 X”写成“作者尝试 X 失败”；
- 把“未报告”写成“默认使用常见设置”；
- 把单个 seed、单个图或不同 budget 的结果写成普遍因果；
- 把我们的实现缺陷反推成论文限制。

## 8. 事实层与分析层的分界

报告第 1–12 节是来源事实与 correspondence。第 13 节开始才允许系统性 `I` 分析。

分析至少回答：

1. 方法真正把容量放在哪里；
2. 它成功依赖哪些 source、query、data、optimization 和 hardware 假设；
3. 哪些结论只适用于作者 protocol；
4. 对本项目是直接可迁移机制、需要改写的类比，还是不能迁移；
5. 最小 matched control 如何证伪该迁移假设；
6. 是否满足静态有界 runtime，或只能作为 teacher/capacity diagnostic。

## 9. Evidence review 门

独立 reviewer 必须检查：

- source ledger 是否覆盖 main/supplemental/code/config/data/talk 的可用性；
- 逐层 architecture、training、runtime 和实验数字是否有 locator；
- PDF 图表/公式是否经过视觉核对；
- `author-negative` 与 `ablation-inferior` 是否被正确区分；
- paper/code conflict 和未报告字段是否保留；
- `N/I` 是否晚于事实层且没有改写来源；
- 对当前 NVIDIA 复现的影响是否引用真实项目证据；
- 假设是否包含 matched control、适用范围、部署类别和证伪条件。

review 结果写入个体报告末尾：

```text
author_worker: <name>
reviewer: <name>
reviewed_at: YYYY-MM-DD
sources_rechecked: <list>
findings_closed: <count/list>
remaining_evidence_gaps: <list>
review_status: changes-requested|evidence-reviewed|complete
```

只有 `review_status` 至少为 `evidence-reviewed` 的报告才能进入跨论文综合。

## 10. Load-bearing 提升

一般 inverse rendering、NeRF/3DGS、生成式材质和 neural reconstruction 只有满足以下一项才升级：

- `direct-inheritance`：核心论文直接继承其 representation/loss/feature/runtime；
- `key-baseline`：它实质影响核心论文的主要实验结论；
- `failure-explanation`：它解释明确的失败、消融或边界；
- `runtime-transfer`：它提供可迁移到 `prepare/evaluate/sample/pdf`、LOD、compiler 或有界部署的机制；
- `user-specified`：用户明确要求。

否则只在 catalog 保留 `discovery-only` 和不提升理由。关键词相似不是提升证据。
