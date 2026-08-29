# 统一 pipeline 架构

本项目只有一条从源材质到训练和部署的正式路径：

```text
source locator → SourceSnapshot
  → ReferenceExecutionPlan@1
  → ReferenceBackendSession[grouped prepare/evaluate/sample/pdf]
  → OnlineTrainingProducer
       ├─ AssetTileBatch@1
       ├─ EvaluatorBatch@3
       └─ MethodSamplerBatch@3
  → TrainingConfig@4 phase graph
  → TrainingRunner → TrainingCheckpoint@4
  → MethodDefinition.compile_program/asset/instance
  → ScatteringPackage@2
  → ProgramRuntime + AssetBinding + InstanceBinding
  → ComparisonSlot[2]
```

LayerStack、OpenPBR、MERL、MaterialX与MDL保留各自原生语义和reference。source family拥有locator、typed edit与snapshot identity；reference definition把snapshot编入execution group，公共backend/session不识别family。platform与Falcor device只由`ReferenceBackendCapability`拥有，upper layers不选择D3D12/Vulkan或构造底层session。

`NativeAssetCollection@1`统一表达多asset、domain、mip与tile+halo traversal；单texel source只是1×1 asset。训练数据由reference在GPU上按typed route即时产生，不存在离线batch/corpus产品。invalid reference行在GPU上压实补采，不变成黑色GT。

`MethodDescriptor@2`登记parameter groups、required components、active phases、batch dependencies、Python outputs、runtime artifacts和Slang entry points。runner按phase启用参数、optimizer/schedule/precision，使用低同步gradient audit与有界prefetch；generic conformance统一检查execution、gradient/update和artifact coverage。

`ScatteringPackage@2`独立计算program、asset、instance与package identity。viewer只在全部identity、typed resource与parity校验通过后原子替换slot binding；`source-reference`是显式权威transport请求，不是假package。
