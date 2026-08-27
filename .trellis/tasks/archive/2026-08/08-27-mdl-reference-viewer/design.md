# 设计：MDL Reference Viewer 集成

## 边界

正式 viewer 消费 `ncls.mdl-compiled-artifact@1`，不在 C++ 内重新实现 MDL 编译语义。启动前的准备脚本复用 `MdlSdkCompilerBridge` 生成六个 artifact 和 `build/mdl-reference/viewer/catalog.json`；viewer 只读、验证并执行这些产物。

## 数据流

```text
references/.../assets.json + vMaterials source
  -> MdlSdkCompilerBridge
  -> hashed compiled artifact + ignored viewer catalog
  -> NclsViewer MdlReferenceArtifact loader
  -> ProgramDesc named string module
       [MDL target-code types + project runtime + generated HLSL + viewer adapter]
  -> current Falcor 8 ReferencePathTracer
  -> shaderball display / capture manifest
```

## C++ source 与 artifact 合同

- 为 source family 增加 `ReferenceFamily::Mdl`；这不是新的 neural program 枚举。
- `ReferenceSource` 的 MDL 部分保存 catalog asset id、snapshot id、artifact root、artifact identity 和 material display name。
- 独立 loader 以 manifest root 为 containment 边界，验证 schema、SDK build、V1 capability audit、compiler identity、精确文件集合与 SHA-256，再加载 generated code、argument block、RO data 和 texture descriptors。
- UI 切换先在临时对象中完成全部验证、GPU resource 创建和 shader 编译；全部成功后再替换当前 source，满足原子失败隔离。

## Shader 组合

- 使用 Falcor 8 已有 `ProgramDesc::addShaderModule(name).addString(source, virtualPath)` 创建 material-specific 具名 module；不把生成 HLSL 写进根仓库。
- module 组合顺序与 formal provider 一致：固定宏、MDL target-code types、`mdl_runtime.slangh`、artifact generated HLSL、viewer adapter。
- adapter 从 `NclsViewerPathSurface` 构造 MDL `Shading_state_material`，调用 `init`、`ior` 与 `surface_scattering_evaluate/sample/pdf`。evaluate 返回 `bsdf_diffuse + bsdf_glossy`；sample 直接返回 SDK 定义的 `bsdf_over_pdf`、非投影半球 PDF、事件类型和 world direction。
- `ReferencePathTracer` 对 family 4 使用该 matched sample 延续路径，并用同一 MDL PDF 计算环境光 MIS。generic cosine/GGX proposal 只保留给没有 native sampler 的其他 source family。这个内部 transport 入口不提升为 provider/source 公共 capability。V1 纹理继续 `SampleLevel(..., 0)`。

## GPU 资源

- argument block 和 RO segment 按 16-byte row 上传为 `StructuredBuffer<float4>`。
- 2D texture 由 Falcor 读取原生资源；BSDF-data texture 根据 artifact pixel type/尺寸构造 3D texture。
- 资源数量遵守 artifact 已审计的 V1 静态上限，sampler 采用 formal provider 相同的 linear/wrap/Lod0 语义。
- MVP 同时只允许一个 material-specific MDL program；shaderball 的活动 source slot 使用它，其他 scene material 保持已有 family。发现第二个不同 MDL artifact 时明确拒绝。

## 启动与操作

- `tools/reference/prepare_mdl_viewer.py` 通过正式 Python source/bridge API 准备六个 artifact 并写 catalog。
- `scripts/launch_mdl_viewer.ps1` 依次调用准备器、唯一允许的 `scripts/build_viewer.ps1`，然后以可见进程启动 Release `NclsViewer`，传入 catalog/default asset。
- viewer UI 显示六项 preset、snapshot/artifact short id、SDK build 和 `ExplicitLod(0)` 状态。

## 验证与回滚

- unit 覆盖 catalog/artifact parsing、tamper、capability、路径 containment 与 boundary。
- GPU probe 复用 formal runtime query，比较 viewer adapter 同方向输出。
- GPU sample probe 对同一 artifact 验证 `sample.weight == evaluate(sample.direction) / pdf`、方向/event 有效性，并冻结 float32 容差。
- Release build + headless shaderball capture 验证真实 scene specialization、finite EXR、identity/spp/schema。
- overlay 脚本的 `finally` 负责回滚 Falcor CMake 临时补丁；任何失败保持根项目改动可诊断且 Falcor clean。
