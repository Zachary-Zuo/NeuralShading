# 学习与部署

产品 registry 当前只发现 `nvidia-neural-appearance`。它是 NVIDIA《Real-Time Neural Appearance Models》公开方法的 functional reproduction：训练期 native-parameter encoder、hierarchical z8、learned-frame `3×64` evaluator、`3×32→9` matched sampler 与 bootstrap→latent finetune 都在一个 `MethodDefinition` 内。MaterialX spatial 与 LayerStack 1×1 是 source-domain adaptation，不伪装成作者未公开资产。

公共 `TrainingRunner` 遍历方法声明的命名 route。NVIDIA 每步消费独立 evaluator/sampler batch，共用一个 Adam 与全局 cosine scheduler；sampler route 从当前 learned proposal 取样，score-function loss 只更新 sampler head。`TrainingCheckpoint@2` 保存模型、optimizer、scheduler、RNG、双 query stream、lifecycle 与 validation state，并以原子写和 SHA-256 sidecar落盘。

训练配置同时记录 `run_class`、`correspondence_id`、`source_adaptation_id` 与 `recipe_id`。smoke/profile 只证明链路；只有 frozen formal recipe 完成全部 global steps 后才可标为 functional-faithful。训练过程写 `<checkpoint>.metrics.jsonl`，结束后写 `<checkpoint>.summary.json`。

统一命令为：

```powershell
conda run -n neural-shading python -m ncls learn list
conda run -n neural-shading python -m ncls learn train <config.json> <checkpoint.pt>
conda run -n neural-shading python -m ncls learn train <config.json> <checkpoint.pt> --resume <periodic.pt>
conda run -n neural-shading python -m ncls learn evaluate <config.json> <checkpoint.pt> --batches 1
conda run -n neural-shading python -m ncls learn export <checkpoint.pt> <source-file> <package-dir>
```

部署输出只有 `ScatteringPackage@1`：FP16 network weights 与两张 RGBA16F latent mip chain由同一 checkpoint编译，viewer parity、deferred和 path tracing都消费同一 package math。FP32 training core用 Torch↔Slang functional oracle验证；导出包另携带 packed-FP16独立 oracle，由真实 D3D12 package shader验证，不能把 FP32 parity当作部署 FP16证据。
