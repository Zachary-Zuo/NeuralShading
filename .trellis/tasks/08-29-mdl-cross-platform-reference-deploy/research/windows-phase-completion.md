# Windows 前半段完成记录

## 结论

Windows/D3D12 前半段已经完成；同一任务保持 `in_progress`，等待实际 Ubuntu/A6000 上执行 Linux 后半段。当前结论不外推到任何尚未实测的 Linux 发行版、driver、glibc、compiler 或 Vulkan 组合。

## 环境

- 状态：完整 Windows。
- GPU：NVIDIA GeForce RTX 4090。
- Python：唯一 Conda 环境 `neural-shading`。
- Falcor：`8.0` / `9dc819c162b2070335c65060436041690b7937f8`，D3D12。
- Slang：`2024.1.34`。
- 公共 backend identity：`773855292cc1d17bcba8af009275659b5726a002d9bd7ddadffb6ff585ac9d25`。

## 交付与验证证据

| Gate | 结果 |
|---|---|
| `scripts/build_reference_backend.ps1 -Configuration Release` | 通过；五个runtime program被probe，LayerStack与仓库MDL fixture完成真实GPU query；`assets=not-managed` |
| `reference doctor --json` | `ready=true`；Falcor、MaterialX、OpenPBR、GLM、stb、MDL SDK/library/plugins/provider全部ready |
| unit | `122 passed` |
| 五族公共backend GPU集合 | `20 passed` |
| reference integration | `3 passed` |
| MDL SDK native fixture | constant diffuse与倾斜`geometry.normal`两项均通过 |
| falcor2冻结formal parity | carpaint/copper共264 query全部通过；报告见`artifacts/reference-parity/mdl/windows-unified-backend-formal-framecosfix/report.json` |
| fixed MDL online training | 2 steps通过；checkpoint `artifacts/training/mdl-windows-unified-backend-final/checkpoint.pt`，identity `13662dd8883ae41aeda14d498d9ef4bc26913407425d6d089914589c89fc0524`；1-batch load/evaluate通过 |
| Windows viewer Release | `NclsViewer.exe`构建通过 |
| static | `compileall`、Linux shell `bash -n`、`git diff --check`通过 |
| 上游洁净性 | Falcor、pbrt-v4、OpenPBR、openpbr-bsdf、glm、MaterialX、falcor2、stb均clean |

formal gate没有改动`tolerance`。carpaint response最大绝对误差为`5.960464477539063e-08`，copper为`7.450580596923828e-09`，两者PDF均通过。训练loss只作为smoke观测值，不是quality hard gate。

## Windows 阶段发现并修复的两类跨层缺陷

1. scalar 3D texture的`[depth,height,width]`被旧负索引当成含channel的shape，导致BSDF-data上传extent错误；已统一为spatial-first parser并覆盖scalar/RGBA 2D/3D。
2. MDL target response用material-local normal cosine，而公共renderer用输入frame cosine；旧迁移除错了cosine，导致normal-map铜材质response漂移而PDF不变。现已把normal cosine ratio保留在公共等价`f`中，并用倾斜`geometry.normal` fixture与formal packet回归。

## Linux 后半段待办

- 在实际Ubuntu/A6000记录OS、kernel、glibc、GPU/driver、Vulkan、compiler、Git、Conda与构建身份。
- 在无`assets/`前提下连续运行两次`deploy_reference_linux.sh`，证明首次构建与verified reuse。
- 用户复制资产后运行五族真实query、MDL native fixture与同一2-step training/checkpoint gate。
- Linux gate全部通过后才完成task、commit并archive。
