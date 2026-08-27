# 统一 pipeline 架构

本项目只有一条从源材质到部署的正式路径：

```text
SourceFamilyDefinition → SourceSnapshot → ReferenceQueryStream
  → OfflineBatchSource | LiveReferenceBatchSource → TrainingRunner
  → TrainingCheckpoint@2 → MethodDefinition
  → ScatteringPackage@1 → ScatteringBinding → ComparisonSlot[2]
```

LayerStack、OpenPBR、MERL、MaterialX 与 MDL 保留各自原生语义和 reference；每个 runtime reference 都直接实现 canonical `prepare/evaluate/sample/pdf`，并保留自己的 state、closure 与 proposal。统一 scene composer 只转发 canonical state 调用，renderer 不识别 source family。pbrt 仅是仓库外部交叉验证边界。产品方法 registry 当前只包含 NVIDIA neural appearance。测试方法只在 `tests/fixtures/` 注入。

`SourceSnapshot` 是 source state 唯一真相。编辑器只消费 `SourceParameterView@1` 并提交 `SourceEditPatch@1`；方法通过 `SourceAdaptationContract` 返回 `unchanged/runtime-patch/recompile/unsupported`。

offline 与 live producer 都输出同一 `TrainingBatch@1`。一个 step 可包含多个独立命名 route；公共 runner不解释 evaluator/sampler语义。live target 在 Falcor shared buffer 与 CUDA tensor 间传递，不落 HDF5、不经过 CPU/NumPy response readback。

`ScatteringPackage@1` 可承载 reference 或 neural program，分别计算 `program_runtime_id`、`material_asset_id` 与 `package_id`。typed resource descriptor 可绑定 buffer、RGBA16F DDS mip texture 与 sampler；所有 resource bytes 参与 material/package identity。当前 viewer 对已编译方法只加载 package，不依赖 Python、PyTorch 或训练目录；`source-reference` 是显式保留的内建权威 source transport 请求，不是伪装成 package id 的磁盘包。

viewer 包含两个对称 `ComparisonSlot`。每侧独立选择已验证 package 或 `source-reference`，并选择其 capability 支持的 PT/deferred mode；固定宽度为 `floor(W/2)`，奇数像素留作 divider，失败只改变本 slot 状态。package PT 命中点与 deferred G-buffer 都构造同一 scattering context（含 UV footprint），neural transport 实际调用 package `prepare/sample/pdf/evaluate`。source reference 经 scene composer 调用对应 source family 的同名 canonical state 接口；capture 会把它记录为特殊请求而不虚构 runtime/material/package 身份。
