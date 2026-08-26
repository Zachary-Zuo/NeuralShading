# NclsViewer 规格

viewer 是 Windows/D3D12 部署验证工具。它只读取 `ScatteringPackage@1`，并将 package program/material 变成 `ScatteringBinding`。

## 双 slot

`ComparisonSlot[2]` 完全对称；每侧独立保存 package、mode、capability、status、GPU resource、accumulation 与 timing。mode 为 `path-tracing` 或 `deferred`。加载、hash、ABI、module 或 capability 失败只在对应 slot 显示错误。

panel 宽度恒为 `floor(outputWidth / 2)`，高度相同；奇数总宽度的一个像素是固定 divider。camera aspect 取 panel extent，与 slot 是否 ready 无关。composite 按 1:1 texel 映射，不提供可拖动分割线。

## renderer 与编辑

场景 PT 与 deferred renderer 都只调用公共 scattering binding，不按 source family 或 method ID 分支。source editor 只渲染 `SourceParameterView@1` 并提交 `SourceEditPatch@1`。edit 成功产生新 snapshot 后，两个 slot 独立按 adaptation result rebind；新 asset 完整验证前不替换旧资源。

capture/replay 使用 `ncls.viewer-capture@4`，核心字段是 `slots[2]`，每项记录 package/runtime/material/source identity、mode 与 status。
