# 01 可复用散射数学原语

## Goal

建立唯一的 Falcor-free Slang 散射数学层，为 NVIDIA sampler baseline、LTC 候选、LayerStack interface、GPU oracle 和 viewer backend 提供同一份 `sample()/pdf()`、方向变换与微表面公式。

## Scope And Dependencies

- 本任务是父任务 `08-25-unified-scattering-method` 的第一个执行子任务，无前置子任务。
- 后续 `03`、`04`、`05` 只能依赖本任务导出的公共组件，不能复制公式。
- 本任务是复杂任务；启动前必须结合当时代码补全并审阅 `design.md`、`implement.md`、`implement.jsonl` 与 `check.jsonl`。

## Requirements

- 统一实现 cosine hemisphere、tilted cosine、LTC、GGX、GGX VNDF、non-centered anisotropic GGX NDF、frame/direction、finite mixture 及其参数约束。
- 每个分布明确方向约定、solid-angle measure、支持域、Jacobian、连续 PDF 和 reflection null-event mass。
- `sample()` 与 `pdf()` 必须共享相同参数解码和变换定义；不允许隐藏 rejection/resampling。
- 公共源码不得依赖 Falcor、MethodBundle、LayerStack IR 或具体 neural backend；方法私有 state/parameter head 不进入公共层。
- 迁移现有正确公式后，LayerStack reference 的数值语义与采集行为保持不变。
- 不修改锁定的 `external/` 上游源码。

## Acceptance Criteria

- [ ] 固定 quadrature 证明各连续 PDF 与显式 null mass 归一化。
- [ ] sample histogram 与 PDF 一致，且 `sample.pdf == pdf(sample.wi)`。
- [ ] grazing、极端各向异性和全部参数边界无 NaN/Inf。
- [ ] 同一 Slang 源通过锁定 Slang 2024.1.34、SlangPy/Falcor GPU oracle 编译。
- [ ] LayerStack reference GPU 回归通过，没有数值或数据采集语义漂移。
- [ ] 子任务完成质量检查、提交并归档后，父任务才允许进入 `02`。

## Out Of Scope

- neural evaluator/sampler 训练、数据重采决定、MethodBundle loader 和 viewer 集成。
