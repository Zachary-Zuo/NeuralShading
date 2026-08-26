# Viewer 层

正式入口是 `apps/viewer/`、`scripts/build_viewer.ps1` 与 `scripts/benchmark_viewer.ps1`。viewer 只依赖 Falcor 和公共 scattering/package 合同，不依赖训练代码。

开发前先读 `../project/unified-pipeline.md`。新增 program 不得修改 C++ 枚举；package module 从自身路径加载。两个 `ComparisonSlot` 必须对称，固定 50/50，失败不改变 peer extent 或 camera aspect。

质量门：slot/capture schema unit tests、package hash/ABI 失败矩阵、四 reference 与 NVIDIA 的 PT/deferred 组合、Release build、Falcor worktree clean。
