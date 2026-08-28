# 统一 pipeline 架构

本项目只有一条从源材质到训练和部署的正式路径：

```text
source locator
  → SourceFamilyDefinition.load_snapshot()
  → SourceSnapshot
  → canonical ReferenceProgramDefinition
  → ReferenceQueryDispatcher (prepare/evaluate/sample/pdf)
  → OnlineTrainingProducer
       ├─ EvaluatorBatch(target_f)
       └─ MethodSamplerBatch(sample_u)
  → TrainingRunner → TrainingCheckpoint@3
  → MethodDefinition → ScatteringPackage@1
  → ScatteringBinding → ComparisonSlot[2]
```

LayerStack、OpenPBR、MERL、MaterialX 与 MDL 保留各自原生语义和reference。每个reference都实现同一个`prepare/evaluate/sample/pdf`合同，并保留自己的backend-specific `ScatteringState`、closure与proposal；统一dispatcher不识别source family，也不实现替代BRDF。pbrt只用于外部crosscheck。

`SourceSnapshot`是source state唯一真相。source family拥有locator解析、原生编辑与snapshot identity；reference program只负责把snapshot编译为canonical module和typed resources。method/source adapter只生成模型所需的native features、UV、LOD、footprint与materialization pyramid，不拥有reference math。

训练数据在GPU上按route即时产生，不存在离线batch/corpus产品。evaluator route调用source `evaluate()`并直接取得线性RGB `f`；method-sampler route只产生conditioning与`sample_u`，由当前learned sampler执行sample/PDF并用learned evaluator构造loss。source的`sample/pdf`仍是reference PT和数值验证的强制能力，但不是NVIDIA sampler的teacher。

Falcor shared output以CUDA tensor零拷贝消费，并由lease保护slot。material-local normal等因素可能让部分通用方向在局部domain中invalid；producer会在GPU上压实valid行并继续补采，记录拒绝统计，而不会把invalid行变成黑色GT、做response clamp或终止整个batch。

`ScatteringPackage@1`分别计算`program_runtime_id`、`material_asset_id`与`package_id`。typed descriptor保留buffer、texture、动态Slang module与sampler的绑定语义。viewer只加载package；`source-reference`是显式内建的权威source transport请求，不是伪装成package id的磁盘包。

viewer包含两个对称`ComparisonSlot`。每侧独立选择package或`source-reference`，并根据capability选择PT/deferred mode；package PT与deferred G-buffer构造同一个scattering context，neural transport调用package `prepare/sample/pdf/evaluate`。
