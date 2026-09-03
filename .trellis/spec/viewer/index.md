# Viewer 层

正式入口是 `apps/viewer/`、`scripts/build_viewer.ps1` 与 `scripts/benchmark_viewer.ps1`。viewer 只依赖 Falcor 和公共 scattering/package 合同，不依赖训练代码。

开发前先读 `../project/unified-pipeline.md`。新增 program 不得修改 C++ 枚举；package module 从自身路径加载。两个 `ComparisonSlot` 必须对称，固定 50/50，失败不改变 peer extent 或 camera aspect。

质量门：slot/capture schema unit tests、package hash/ABI 失败矩阵、program cache与editable instance candidate-compile-atomic-swap、五种 source reference 与已注册method的PT/deferred组合、Release build、Falcor worktree clean。MDL 的动态 artifact 路径还必须验证 compiler/file identity、V1 capability、formal boundary、matched sample/PDF 一致性，以及 car paint/ceramic 的真实 headless firefly 尾部。MDL package compatibility使用canonical source snapshot identity，不能退化为root `.mdl`文件hash。

路径追踪的命中解码、frame 与 UV footprint 统一遵守 [path-surface.md](path-surface.md)。

MDL source 的旧六项catalog与registry-derived `ViewerMaterialCatalog@1`、artifact/component identity、动态 string module、linked typed edit、资源绑定和原子 preset 切换遵守 [mdl-reference.md](mdl-reference.md)。

自动化 EXR/PNG 导出与 replay 的固定 spp、单 panel difference 尺寸统一遵守 [capture-harness.md](capture-harness.md)。
