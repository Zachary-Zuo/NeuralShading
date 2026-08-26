# 统一 pipeline 架构

本项目只有一条从源材质到部署的正式路径：

```text
SourceFamilyDefinition → SourceSnapshot → ReferenceQueryStream
  → OfflineBatchSource | LiveReferenceBatchSource → TrainingRunner
  → TrainingCheckpoint@2 → MethodDefinition
  → ScatteringPackage@1 → ScatteringBinding → ComparisonSlot[2]
```

LayerStack、OpenPBR、MERL 与 MaterialX 保留各自原生语义和 reference；pbrt 仅是仓库外部交叉验证边界。产品方法 registry 当前只包含 NVIDIA neural appearance。测试方法只在 `tests/fixtures/` 注入。

`SourceSnapshot` 是 source state 唯一真相。编辑器只消费 `SourceParameterView@1` 并提交 `SourceEditPatch@1`；方法通过 `SourceAdaptationContract` 返回 `unchanged/runtime-patch/recompile/unsupported`。

offline 与 live producer 都输出同一 `TrainingBatch@1`。live target 在 Falcor shared buffer 与 CUDA tensor 间传递，不落 HDF5、不经过 CPU/NumPy。

`ScatteringPackage@1` 同时承载 reference 和 neural program，分别计算 `program_runtime_id`、`material_asset_id` 与 `package_id`。viewer 只加载 package，不依赖 Python、PyTorch 或训练目录。

viewer 包含两个对称 `ComparisonSlot`。每侧独立选择 package 与 PT/deferred mode；固定宽度为 `floor(W/2)`，奇数像素留作 divider，失败只改变本 slot 状态。
