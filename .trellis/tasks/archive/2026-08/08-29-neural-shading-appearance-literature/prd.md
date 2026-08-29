# Neural Shading 与 Appearance 文献深度研究

## 目标与价值

建立一套可追溯的 neural shading、neural appearance 与场景级 neural light transport 研究档案。每篇报告先按论文正文、补充材料、官方代码和配置详细重建原方法，再单独分析其成功条件、失败尝试、限制以及对 NeuralShading 当前 NVIDIA 方法复现和后续新方法设计的意义。

本任务解决的问题不是“快速总结论文”，而是避免先压缩论文语义、再凭经验补全实现所造成的幻觉和信息损失。最终价值是把文献结论转成可检查的 correspondence、候选机制与 matched 实验假设，而不是直接在本任务内修改产品模型。

## 已确认背景

- 仓库现有 `docs/research/prior_art.md` 和归档任务报告主要承担候选清单、source 选择与路线判断，不能替代逐论文的完整技术档案。
- 当前 NVIDIA 方法已有功能复现与论文 correspondence 证据；新任务必须复用这些证据并审计差距，不能把已有实现或旧摘要无条件当作论文事实。
- 场景级 transport 是本任务的正式范围。将它安排在第二研究波次只表示执行顺序，不表示降低报告深度或排除其跨线综合。
- `paper1469_1.pdf` 已确认是 Pacific Graphics 2026《Real-Time Volumetric Light Transport Inference from Auxiliary Renderings》，应归入体积/场景级 light transport inference，而不是局部 `evaluate(wo, wi)` neural material。
- 研究报告和综合分析统一保存在本任务目录下；大体积论文、代码 clone 和临时抽取物只进入任务 `scratch/` 或外部 locator，不作为根仓库持久化资产。

## 范围与执行波次

研究波次只决定执行先后。所有波次都属于本任务的验收范围。

### 波次 0：证据与目录基线

- 建立论文 catalog、关系图、证据等级、来源锁定和报告模板。
- 登记正文、supplemental、项目页、官方代码、固定 commit、配置、数据、talk、勘误和访问状态。
- 对无法获得的第一方材料记录缺口；不得使用二手摘要填补未知实现。

### 波次 1：局部 neural material / appearance evaluator

初始必研集合包括但不限于：

- Real-Time Neural Appearance Models；
- Taming Optimization Variance in Compact Neural Shading Networks；
- Real-Time Neural Materials on Mobile VR；
- NeuMIP；
- Neural Biplane Representation for BTF Rendering and Acquisition；
- Towards Comprehensive Neural Materials；
- Neural Layered BRDFs；
- MetaLayer；
- Neural BRDF Representation and Importance Sampling；
- BSDF Importance Baking；
- Neural Prefiltering for Correlation-Aware Levels of Detail；
- A Hybrid Neural-Microfacet BRDF Model；
- Neural Material Adapter。

### 波次 2：场景级与体积 neural light transport

初始必研集合包括但不限于：

- NeLiF: Neural Lighting Function Generation for Real-Time Indoor Rendering；
- NeLT: Object-Oriented Neural Light Transfer；
- Neural Global Illumination via Superposed Deformable Feature Fields；
- Dual-Band Feature Fusion for Neural Global Illumination with Multi-Frequency Reflections；
- LightFormer: Light-Oriented Global Neural Rendering in Dynamic Scene；
- Neural Light Probes for Real-Time Global Illumination；
- 本地 `paper1469_1.pdf` 对应的体积光传输推断方法。

### 波次 3：重要对照与机制追踪

- 阅读核心论文实际比较、继承或批评的重要方法，并维护引用/机制关系图。
- 当某个对照决定核心论文的表示、loss、sampling、LOD、部署或失败解释时，将其提升为完整逐论文报告。
- 只提供一般背景、且不影响本项目方法判断的论文可以保留为带来源的 catalog 条目，不强行扩写为完整报告。
- 一般 inverse rendering、NeRF/3DGS、生成式材质和 neural reconstruction 采用“机制触发纳入”：不主动做全领域普查；只有在它们构成核心论文的直接依赖、重要对照、失败解释或可迁移 shading/transport 机制时，才提升为完整报告。

## 需求

- R1：每篇完整报告先陈述原论文事实，最后才进入本项目分析；两部分必须有明确边界。
- R2：报告至少覆盖问题定义与适用范围、输入输出、坐标/方向参数化、representation、逐层网络与 tensor 形状、训练数据与 query recipe、loss、optimizer、schedule、batch、steps、hardware、runtime/deployment、参数量/bytes/MAC/时间/内存、实验配置、指标、结果、消融、失败尝试和论文限制。
- R3：每项具体事实标记证据来源：正文、supplemental、官方代码/配置、作者说明或本项目分析；数值和架构结论尽量带 page/section/figure/table/listing 或 commit/file locator。
- R4：论文未报告的信息写明“未报告”；第一方来源互相冲突时保留冲突，不自行选一个版本冒充原方法。
- R5：所谓失败尝试只包括论文/补充材料明确给出的负结果、代码中可验证的配置，或本项目后来实际执行并有 artifact 的实验；不得从最终方法反向虚构失败历史。
- R6：官方代码存在时，固定 commit 并建立 paper ↔ supplemental ↔ code/config correspondence；识别代码默认值、论文正式配置、示例配置和复现改动之间的差异。
- R7：完整个体报告形成后，再分别综合 representation/coordinates、optimization/loss、filtering/LOD、sampling/integration、deployment/system amortization，避免先写总论再把论文塞进既定结论。
- R8：面向当前 NVIDIA 复现建立专门 correspondence，区分已忠实实现、作者未披露、本项目有意偏离、预算适配和潜在缺陷。
- R9：提出的新方法或改进必须写成可证伪假设，指出直接证据、迁移假设、预计适用范围、静态运行预算类别和最小 matched 对照；不得把跨论文拼装自动称为 novelty。
- R10：本任务只交付研究档案、综合、correspondence 和实验假设，不直接修改产品模型、训练路径或 runtime；需要实现的候选另建任务。
- R11：所有维护性 Markdown 以中文为主体，技术标识、公式、论文标题和必要术语保留英文。
- R12：研究 corpus 可以在机制触发规则内增加 load-bearing 论文；一般相关工作不因关键词相近而自动扩大为完整报告。超出该规则的主动领域扩张必须返回 planning。

## 报告与索引产物

计划在本任务下维护：

```text
research/
  catalog.md
  evidence-policy.md
  papers/<paper-slug>.md
  comparisons/
    representation-and-coordinates.md
    optimization-and-loss.md
    filtering-and-lod.md
    sampling-and-integration.md
    deployment-and-amortization.md
  implications/
    current-nvidia-correspondence.md
    reproducible-hypotheses.md
```

目录可在 `design.md` 中细化，但不得把个体报告移出本任务的可追溯任务树。

## 范围外

- 在本任务内实现或训练新的产品候选。
- 把论文 PDF、第三方仓库或大型数据直接提交到根 Git。
- 将引用数量、榜单结果或二手综述当作方法正确性的证据。
- 为追求表面完整度而猜测未披露的实现细节。
- 与 shading、appearance、light transport 没有机制关联的一般视觉模型调研。

## 验收标准

- [x] A1（需求交付，来源：用户要求深入研究多篇相关论文）：存在带固定来源、状态、研究波次、关系和报告完成度的总 catalog，覆盖所有初始必研论文。
- [x] A2（理论/语义正确性，来源：用户要求避免压缩语义后猜实现）：存在统一 evidence policy 和报告模板，明确事实/分析分隔、locator、冲突、缺失信息和失败尝试的记录方式。
- [x] A3（需求交付，来源：用户指定 neural shading/appearance 主线）：波次 1 初始必研集合中的每篇论文都有通过证据检查的完整报告。
- [x] A4（需求交付，来源：用户确认场景级 transport 属于本任务范围）：波次 2 初始必研集合中的每篇论文都有同等证据标准和技术深度的完整报告；不得因执行顺序而降格为摘要。
- [x] A5（需求交付，来源：用户要求研究重要对比与相似方法）：所有被提升为 load-bearing 对照的方法都有完整报告；其余相关工作至少在 catalog 中说明为何未提升。
- [x] A6（理论/语义正确性，来源：论文方法 correspondence 与用户的防幻觉要求）：每篇完整报告覆盖 R2 字段；缺失字段明确标记“未报告/材料不可得”，没有无来源补全。
- [x] A7（需求交付，来源：用户要求分析成功和失败尝试）：每篇报告分别登记作者支持的成功证据、失败/负结果、限制和本项目分析，且后者不改写前者。
- [x] A8（需求交付，来源：用户要求为新方法和当前复现改进提供依据）：完成五个跨论文专题综合，并能回链到个体报告的具体证据位置。
- [x] A9（需求交付，来源：用户明确要求改进当前 NV 方法复现）：完成当前 NVIDIA 复现 correspondence，明确论文配置、仓库代码、当前实现和预算适配之间的关系。
- [x] A10（需求交付，来源：用户要求为后续提出新方法/改进提供依据）：形成一组按证据强度和预期价值排序的可证伪改进假设；每项都包含最小 matched 实验、评测轴和部署约束。候选质量、时间和内存预测仅用于排序，不构成本研究任务的 hard gate。
- [x] A11（理论/语义正确性，来源：`AGENTS.md` 根仓库边界与 Trellis 研究持久化合同）：研究产物全部位于本任务目录/任务树；根仓库未新增第三方 PDF、代码 clone、数据或运行 artifacts。
- [x] A12（需求交付，来源：用户确认“机制触发纳入”边界）：一般 inverse rendering、NeRF/3DGS、生成式材质和 neural reconstruction 只有满足 R12 时才提升为完整报告；catalog 对未提升条目保留可审计理由。
