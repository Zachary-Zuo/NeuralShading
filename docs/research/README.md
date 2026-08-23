# 研究过程文档

本目录保存项目早期问题推导、可行性分析和任务规划。带日期的三份文档从 Git 历史恢复，正文保留当时的 `teacher`、`oracle`、固定解析 closure、旧 claim 和旧 kill test，用于追溯研究方向如何形成；每份文件顶部已加历史快照说明，它们不覆盖当前稳定合同。

当前方向已经从“网络输出固定解析 closure”收敛为 neural material program：

- 小型 MLP 直接实现 `evaluate(wo, wi)`；
- `compile_material` 生成 material/spatial latent 与共享 decoder 所需资产；
- `prepare` 获取、过滤并编码 latent、footprint 和 `wo`，形成可复用 state；
- path-tracing profile 在 evaluator 成形后增加匹配且 PDF 可计算的 sampler；
- deferred 环境/面光积分是后续独立建模问题；
- 当前先做 evaluator 模型定义、单材质容量、共享 decoder + latent、compiler 泛化和 Slang 最小部署，不提前做多灯、PT 方差或 UE 系统 kill test。

当前问题定义与执行路线以以下文档为准：

- [实时材质编译：问题与解决路线](../realtime_material_compilation.md)
- [项目目标架构](../architecture.md)
- [散射后端合同](../contracts/scattering_backend.md)
- [源材质族、reference 与统一神经材质程序](../material_scope.md)

历史文档：

- [P0 任务清单](idea-neural-layered-materials-P0-任务清单-2026-08-21.md)
- [神经闭包代数方向可行性深评](idea-neural-layered-materials-analysis-2026-08-21.md)
- [SIGGRAPH 研究调研](idea-neural-layered-materials-siggraph-research-2026-08-21.md)
