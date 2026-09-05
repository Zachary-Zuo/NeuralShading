# NclsViewer 规格

viewer 是 Windows/D3D12 部署验证工具。已编译方法从 `ScatteringPackage@2` 读取，并把package拆成可按`program_id`复用的`ProgramRuntime`、独立`AssetBinding`和原子`InstanceBinding`；另保留显式的`source-reference`请求来调用源材质族的权威source transport。后者不是磁盘package，不得填充虚假的program/asset/instance identity。

## 双 slot

`ComparisonSlot[2]` 在选择、状态、输出与生命周期上完全对称；每侧独立保存 binding请求、mode、capability、status、GPU resource、accumulation 与 timing。binding请求可以是已验证 package或特殊值 `source-reference`，mode 为 `path-tracing` 或 `deferred`。加载、hash、ABI、module 或 capability 失败只在对应 slot 显示错误。

自动化 capture harness 对所有 ready 的 path-tracing slot 固定累计到 1024 spp 后才导出；最后一帧截断到剩余 sample，deferred slot 保持 0 spp。`slot-0.exr`、`slot-1.exr` 与 `difference.exr` 都使用单 panel 的 `view_resolution`，difference 由独立同尺寸纹理按 panel-local UV 计算；双 panel 的 `comparison.exr` 才使用总 `resolution`。正式基线使用 1024 spp，显式 headless target 可用于 smoke；交互持续 1 spp/dispatch。不得用全宽 composite 生成横向拉伸的 difference。

线性 capture 从 RGBA32F 资源显式导出 float32 EXR；不能使用 Falcor/FreeImage 默认的 half EXR 路径，否则合法的高动态范围样本超过 65504 后会在文件中变成 `Inf`。导出阶段不做 radiance clamp，raw EXR 保留真实尾部供收敛与 firefly 诊断。

panel 宽度恒为 `floor(outputWidth / 2)`，高度相同；奇数总宽度的一个像素是固定 divider。camera aspect 取 panel extent，与 slot 是否 ready 无关。composite 按 1:1 texel 映射，不提供可拖动分割线。

两侧常驻标题显示 Reference/Neural、family/profile、PT/Deferred 和实际状态；标题使用 GUI overlay，不改变 panel extent，不进入线性 EXR/difference。交换左右移动整个 slot，失败保留原 binding 时同时显示请求失败。

## renderer 与编辑

`PathTracer.cs.slang` 与 `DeferredRenderer.cs.slang` 分别是唯一的 PT/deferred 实现，通过 `SceneScattering` 调用公共材质接口，不按 source family 或 method ID 分支。只替换 active material ID，其他 primary/secondary/raster hit 保持 source。package PT 在每个 surface hit 构造 position、frame、UV/gradient 与 material instance，续路径调用 binding 的 matched `sample/pdf`，直接光调用同一 state 的 `evaluate/pdf`；deferred 从 G-buffer 把同样的 footprint 交给 `prepare`。`source-reference` 不冒充 package binding，但通过 `SceneReferenceProgram` 把各权威 source backend 的 canonical `prepare/evaluate/sample/pdf` 交给单一 reference integrator；integrator 不按 family 分支。source editor 只渲染 `SourceParameterView@1` 并提交 `SourceEditPatch@1`。edit 成功产生新 snapshot 后，两个 slot 独立按 adaptation result rebind；新 asset 完整验证前不替换旧资源。

共享 PT 每个 hit 固定取 4 个环境 NEE 样本，multiple-sample power MIS 的 light-sampled 与 BSDF-hit 两侧都使用 `4 * p_light`。续路径 origin 由实际 sampled direction 相对 geometric normal 的符号选侧。不能用 radiance、throughput 或 sample-weight clamp 掩盖不匹配的 sampler/PDF。

reference PT 与 package PT 必须经 `PathSurface.slang` 共享命中解码、UV/V flip、geometric/shading frame、front-facing 和 ray-cone footprint。Falcor 的 `cameraU/cameraV/cameraW` 含共同 `focalDistance` 尺度，因此 primary spread 按 `2·|cameraV|/(|cameraW|·height)` 构造；不得只使用 `|cameraV|`。PT 输出与 deferred raster `ddx/ddy` 都是 normalized UV derivative，纹理或 latent 的实际尺寸由各自 sampler/backend 消费。非 triangle geometry 缺少可信 triangle Jacobian 时使用显式有限 fallback，不把未初始化 differential 传给 source/package。

capture/replay 使用 `ncls.viewer-capture@4`，核心字段是 `slots[2]`，每项记录 package/program/asset/instance/source identity、mode 与 status。

PT 要求四入口 capability（mask 15），deferred 要求 prepare/evaluate（mask 3）。两侧都整 panel dispatch；deferred 使用相同局部查询预算且不包含 GI/场景阴影。当前 reader 只接受 catalog v2、handoff v2、scene v2 和 capture v4；旧输入通过当前 prepare/export 入口重建，无旧调度或 CLI 别名。
