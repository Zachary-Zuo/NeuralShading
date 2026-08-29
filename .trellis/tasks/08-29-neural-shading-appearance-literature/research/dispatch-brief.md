# Research Subagent Dispatch Brief

每次 dispatch 必须把尖括号字段替换为具体值，并以 `Active task:` 行开头。主会话保留 catalog、模板、综合和最终状态所有权。

```text
Active task: .trellis/tasks/08-29-neural-shading-appearance-literature

你是本任务的 trellis-research subagent。你不是代码实现 agent；本轮只负责一篇论文的 author pass。

Assigned paper: <official title>
paper_id: <paper-id>
Owned output: .trellis/tasks/08-29-neural-shading-appearance-literature/research/papers/<paper-id>.md
Owned scratch: .trellis/tasks/08-29-neural-shading-appearance-literature/scratch/workers/<paper-id>/

你并非独自在代码库中工作。其他 agent 会并行写其他论文报告；不要回退、覆盖或格式化他人的修改。除 owned output 与 owned scratch 外，不得编辑 catalog、evidence-policy、report-template、comparisons、implications、prd/design/implement 或其他报告。如果共享文件需要更新，只在最终 handoff 中建议，由主会话合并。

开始前必须完整读取：
- 本任务 prd.md、design.md、implement.md；
- research/evidence-policy.md；
- research/report-template.md；
- docs/research/prior_art.md 中与本论文有关的现有项目判断；
- 对 NVIDIA 有影响时，再读取归档 faithful NVIDIA task 的相关 research/design evidence。

研究要求：
1. 优先正文、supplemental、作者项目页、官方代码/config/data、作者 talk/勘误；二手资料只用于发现来源。
2. 完整阅读 main paper 与可得 supplemental，不得只按摘要或 related-work 摘要写报告。
3. PDF 使用全文抽取帮助检索，但必须视觉检查关键公式、图、表、图注、脚注和 appendix。
4. 官方代码存在时固定 commit，区分 paper/formal、default、example、smoke 和后续 revision。
5. 严格按 report-template 的 16 节写作；先 P/S/C/A 事实，最后 N/I 分析。
6. 所有适用但未披露字段写“未报告”；禁止用常见设置补全。
7. author-negative、ablation-inferior、known-limitation、paper-code-gap 分开记录；禁止从最终设计虚构失败尝试。
8. 中文为主体，论文标题、标识符、公式和必要技术术语保留英文。
9. 不修改产品代码、不训练模型、不把 PDF/clone/数据加入根 Git。

完成 author pass 后：
- report_status 保持 report-draft；reviewer 保持 unassigned；
- 做模板中的自查，但不要自行声明 evidence-reviewed/complete；
- 最终 handoff 必须列出：output path、main/supp/code source locator、locked commit/hash、未解析冲突、remaining evidence gaps、建议提升的 load-bearing 论文及 trigger、对 catalog 的建议状态；
- 等待主会话安排独立 reviewer。
```

## Reviewer 补充指令

reviewer 不重写整篇报告，重点重新打开一手来源抽查 architecture、training、runtime、关键数值和负结果 locator。发现问题时列出精确章节与修订要求；只有全部 finding 关闭后才把 `review_status` 更新为 `evidence-reviewed` 或 `complete`。若 reviewer 需要直接修订，主会话必须先明确转移该报告文件所有权。
