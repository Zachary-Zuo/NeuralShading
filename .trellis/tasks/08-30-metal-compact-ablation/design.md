# 设计

## 实验矩阵

矩阵分三层执行，但不以早期层结果阻止后续已预注册变体：

1. **结构消融**：一次关闭或替换一个full机制，并为高耦合机制增加少量预注册交互项；
2. **容量/精度sweep**：在full结构与结构消融结论旁路上扫描固定的grid、MLP、rank、lobe/frame/field、read和precision profile；
3. **蒸馏与部署组合**：以full evaluator/sampler为teacher，把已观察到的非支配结构组合成有限个compact profiles并重新编译部署。

每个实验由versioned ablation manifest描述`base_identity`、唯一changed axes、budget、seed、selection、source/query和expected runtime layout。runner拒绝未登记的多轴变化；对确需联合变化的变体显式登记interaction identity。

## 比较与选择

先保留raw per-source-state rows，再对matched pair做bootstrap。质量、semantic correctness、steady-state time、lifecycle latency、delivery bytes与resident bytes不合成单分数。报告展示非支配集合、置信区间重叠与能力差异；“无贡献”要求在相关泛化/recipe strata没有可信收益且成本确有下降，边界不清时标为interaction/uncertain而非删除。

蒸馏teacher仅提供训练信号，student仍对authoritative reference做最终评测。量化和read reduction必须经过真实Slang/package/viewer测量，静态MAC/bytes只作解释变量。

## 产物

保留`metal_fused_full_v1`不变；compact产物使用独立profile identity并引用parent/full baseline。最终报告给出每个profile适用能力、quality/time/memory观测和未解决风险，产品默认选择留给后续用户对齐。
