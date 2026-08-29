# 学习与部署

产品registry当前只发现`nvidia-neural-appearance`。它是NVIDIA《Real-Time Neural Appearance Models》的functional reproduction：训练期native-parameter encoder、hierarchical z8、learned-frame `3×64` evaluator、`3×32→9` matched sampler与bootstrap→latent finetune都在一个`MethodDefinition`内。MaterialX spatial与LayerStack 1×1是source-domain adaptation，不伪装成作者未公开资产。

公共`TrainingRunner`每step消费两条独立typed route。evaluator batch包含conditioning、`wi`与source `evaluate().f`生成的`target_f`；sampler batch只包含conditioning与`sample_u`。NVIDIA sampler从当前learned GGX9 proposal取样并计算自身PDF，用detached learned evaluator构造`luminance(f)·|cosθi|`的forward-KL score loss；source `sample/pdf`不参与这个loss。

`OnlineTrainingProducer`只通过公共`ReferenceBackendCapability.open()`取得session，因此LayerStack、OpenPBR、MERL、MaterialX和MDL共用同一训练查询路径。`nvidia.mdl-fixed-uniform@1`是当前固定、无空间纹理MDL状态的method adapter：它把受支持的原生typed参数编码到64个固定slot，保持1×1 materialization，并拒绝texture/resource、未知类型、非有限值、多snapshot或超限参数。它不改变MDL reference语义，也不是MDL专用producer。

NVIDIA evaluator直接输出`f_hat=exp(raw-3)`，用`log1p` L1与`target_f`比较。训练和FP16 runtime均不经过`f·cos` adapter，也不在掠射角除以cosine。

`TrainingConfig@3`记录source locator、online query recipe、typed routes、`run_class`、`correspondence_id`、`source_adaptation_id`和`recipe_id`。`TrainingCheckpoint@3`保存模型、optimizer、scheduler、RNG、source snapshot IDs、reference/query/adapter identity、lifecycle与validation state，并以原子写和SHA-256 sidecar落盘。旧config/checkpoint没有reader或converter。

统一命令为：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls learn list
.\scripts\run_falcor_python.ps1 -m ncls learn train <config.json> <checkpoint.pt>
.\scripts\run_falcor_python.ps1 -m ncls learn train <config.json> <checkpoint.pt> --resume <periodic.pt>
.\scripts\run_falcor_python.ps1 -m ncls learn evaluate <config.json> <checkpoint.pt> --batches 1
.\scripts\run_falcor_python.ps1 -m ncls learn export <checkpoint.pt> <package-dir> --material-index 0
```

smoke/profile只证明链路；只有冻结formal recipe完成全部global steps后才可标为functional-faithful。训练过程写`<checkpoint>.metrics.jsonl`，结束后写`<checkpoint>.summary.json`。

固定MDL Windows/Linux smoke使用`configs/learning/nvidia-rta2024-mdl-effect-pigment-smoke.json`完成两步online training和一次checkpoint reload/evaluate；target必须来自公共backend session，不能由预存batch或adapter计算。

部署输出只有`ScatteringPackage@1`：FP16 network weights与两张RGBA16F latent mip chain由同一checkpoint编译。FP32 training core用Torch↔Slang functional oracle验证；导出包另携带packed-FP16 `expected_f` oracle，由真实D3D12 package shader验证。
