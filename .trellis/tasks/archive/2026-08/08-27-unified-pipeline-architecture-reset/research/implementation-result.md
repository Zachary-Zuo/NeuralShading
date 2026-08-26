# 实施与验证结果

## 交付矩阵

| 范围 | 结果 |
|---|---|
| source/editor | 四个 family 使用 canonical `SourceSnapshot` 与 typed view/patch；MERL 只读 |
| reference/data | 四个 `ReferenceProgramDefinition`；offline/live 统一 `TrainingBatch@1` |
| live GPU | Falcor shared buffer → CUDA tensor → backward，禁止 host readback并验证 lease |
| method/training | 产品 registry 仅 NVIDIA；一个 runner；`TrainingCheckpoint@2` |
| deployment | reference/neural 共用 `ScatteringPackage@1`，三个独立 identity 与 tamper rejection |
| viewer | package loader、slot contract、固定等宽 panel、固定奇数 divider、Release build |
| 清理 | 旧方法、并行采集、旧 checkpoint/package/schema/reader/CLI/shader/test 已删除 |
| 边界 | pbrt tool/reference registry 未注册到新 pipeline；`external/Falcor` 干净 |

## 已执行门禁

- `conda run -n neural-shading python -m pytest -q`：64 passed，Falcor-only case 在普通解释器按 marker 跳过。
- `scripts/run_falcor_python.ps1 -m pytest tests/gpu -q`：6 passed。
- `scripts/build_viewer.ps1 -Configuration Release`：通过，输出 `NclsViewer.exe`。
- `git diff --check`：通过。
- 旧 identity reachability 搜索：稳定源码、shader、app、config、script、test、docs 与 spec 无命中；历史 research 文档保留原始叙述。

Release 链接仍输出 Falcor 上游已有的 `LNK4098` warning，但构建成功；本任务没有修改上游源码。
