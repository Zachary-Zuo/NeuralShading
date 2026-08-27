# NclsViewer 规格

viewer 是 Windows/D3D12 部署验证工具。已编译方法从 `ScatteringPackage@1` 读取，并将 package program/material 变成 `ScatteringBinding`；另保留显式的 `source-reference` 请求来调用源材质族的权威 source transport。后者不是磁盘 package，不得填充虚假的 package/runtime/material identity。

## 双 slot

`ComparisonSlot[2]` 在选择、状态、输出与生命周期上完全对称；每侧独立保存 binding请求、mode、capability、status、GPU resource、accumulation 与 timing。binding请求可以是已验证 package或特殊值 `source-reference`，mode 为 `path-tracing` 或 `deferred`。加载、hash、ABI、module 或 capability 失败只在对应 slot 显示错误。

panel 宽度恒为 `floor(outputWidth / 2)`，高度相同；奇数总宽度的一个像素是固定 divider。camera aspect 取 panel extent，与 slot 是否 ready 无关。composite 按 1:1 texel 映射，不提供可拖动分割线。

## renderer 与编辑

package PT 与 deferred renderer 都只调用公共 scattering binding，不按 source family 或 method ID 分支。package PT在每个 surface hit构造 position、frame、UV/gradient与 material instance，续路径调用 binding 的 matched `sample/pdf`，直接光调用同一 state 的 `evaluate/pdf`；deferred从 G-buffer把同样的 footprint交给 `prepare`。`source-reference` 走独立的权威 source transport边界，不冒充 package binding。source editor只渲染 `SourceParameterView@1` 并提交 `SourceEditPatch@1`。edit成功产生新 snapshot后，两个 slot独立按 adaptation result rebind；新 asset完整验证前不替换旧资源。

reference PT 与 package PT 必须经 `PathSurface.slang` 共享命中解码、UV/V flip、geometric/shading frame、front-facing 和 ray-cone footprint。Falcor 的 `cameraU/cameraV/cameraW` 含共同 `focalDistance` 尺度，因此 primary spread 按 `2·|cameraV|/(|cameraW|·height)` 构造；不得只使用 `|cameraV|`。PT 输出与 deferred raster `ddx/ddy` 都是 normalized UV derivative，纹理或 latent 的实际尺寸由各自 sampler/backend 消费。非 triangle geometry 缺少可信 triangle Jacobian 时使用显式有限 fallback，不把未初始化 differential 传给 source/package。

capture/replay 使用 `ncls.viewer-capture@4`，核心字段是 `slots[2]`，每项记录 package/runtime/material/source identity、mode 与 status。
