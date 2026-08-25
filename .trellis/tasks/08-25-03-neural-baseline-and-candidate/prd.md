# 03 Neural Baseline 与候选选择

## Goal

在 `02` 冻结的数据协议上忠实建立 NVIDIA learned-frame evaluator + analytic two-lobe sampler baseline，实现 exact-core positive-residual 候选，并以 matched 2×2 实验选择实际部署方法。

## Scope And Dependencies

- 前置任务：`01` 与 `02` 已完成、提交并归档。
- 必须消费 `01` 的公共数学组件和 `02` 冻结的 corpus identity。
- 本任务是复杂任务；启动前根据前序真实产物补全并审阅三件套与 context manifests。

## Requirements

- 实现 NVIDIA two learned shading frames、direct positive BRDF MLP、9 参数 tilted-cosine/non-centered-GGX proposal、log-space L1、directional mollification、sampler KL 和 latent detach。
- 同时实现论文规模 `diagnostic` baseline 与 `≤2k MAC` deployment-matched `realtime` baseline；二者共享方法语义，不混淆性能声明。
- 实现 `core-frame-neural-v1`：exact top-interface core 加必选 positive neural residual，不允许 prediction clamp 或 lobe-only fallback。
- evaluator `{NVIDIA direct, exact-core residual}` 与 sampler `{NVIDIA GGX9, LTC-K2}` 形成 matched 2×2，对数据、训练预算、统计协议和部署成本做公平比较。
- 运行时 sampler 参数由 shared `ScatteringState` 的 neural head 产生；`sample/pdf` 调用同一解析 proposal。
- Torch 只承载 loss、optimizer 和 oracle；生产前向只有通用 Slang 源。
- 最终选择不预设自研候选胜出；没有可信 Pareto 改善时选择 deployment-matched NVIDIA baseline。

## Acceptance Criteria

- [ ] paper-scale diagnostic 与 deployment-matched baseline 都有真实训练、质量和成本结果。
- [ ] `prepare/evaluate/state/asset/weights` 通过父任务冻结的机械预算门。
- [ ] evaluator 通过 Q1、tail、energy、bootstrap CI 和 leave-one-state-out gate。
- [ ] 两种 sampler 都通过 PDF/null 归一化、histogram、re-evaluation 和同 evaluator MC 无偏性验证。
- [ ] 2×2 matched 结果冻结最终 evaluator/sampler identity，并产出可追溯 checkpoint、compiled materials 和 Slang parity 证据。
- [ ] 最终选择相对 deployment-matched NVIDIA baseline 非劣；否则 baseline 本身成为部署结果。
- [ ] 子任务完成质量检查、提交并归档后，父任务才允许进入 `04`。

## Out Of Scope

- 通用 MethodBundle loader、viewer renderer path 和旧方法删除。
