# 05 Viewer Method Deferred 与 PT

## Goal

让右侧 viewer 通过 `04` 的同一通用 MethodBundle specialization 正确显示 method deferred 与 method path tracing，并形成可重复 capture/replay 证据。

## Scope And Dependencies

- 前置任务：`01` 至 `04` 已完成、提交并归档。
- 本任务只能调用 `04` 的公共 loader/backend adapter，不得新增 method-specific viewer 分支。
- 本任务是复杂任务；启动前根据实际 loader 与 Falcor renderer lifecycle 补全并审阅三件套与 context manifests。

## Requirements

- 提供 `Reference PT | Method Deferred`：右侧对显式灯光方向调用通用 `evaluate()`。
- 提供 `Reference PT | Method PT`：右侧 ray hit 调 `prepare/sample/pdf/evaluate`，路径权重恰好使用一次 `f·|cos|/pdf`。
- method PT 拥有正确的 accumulation/reset 生命周期；两侧共享场景、相机、材质、灯光、曝光和 tone mapping。
- UI、CLI 与 capture manifest 显式记录两侧 integrator、bundle identity/runtime class、spp、bounce limit、seed 和 raw-authoritative 标志。
- paper-scale diagnostic 可以显示和测时，但不得被 UI/capture 标为 realtime。
- viewer 变更只位于根仓库自有源码和既定 Falcor overlay 边界。

## Acceptance Criteria

- [ ] method deferred pixel probe 与同方向 backend `evaluate()` 一致。
- [ ] method PT 与同 evaluator 的确定性积分在冻结 CI 内一致。
- [ ] source-reference PT 对照覆盖单层、四个既有尾部、多 lobe/各向异性和 grazing 场景。
- [ ] 两种比较模式均可 headless capture/replay，重复结果与 manifest identity 稳定。
- [ ] build overlay 结束后 `external/Falcor` 仍处于锁定提交和干净工作树。
- [ ] 子任务完成质量检查、提交并归档后，父任务才允许进入 `06`。

## Out Of Scope

- 重新设计/训练方法、升级锁定 Slang/Falcor 和保留旧 viewer fallback。
