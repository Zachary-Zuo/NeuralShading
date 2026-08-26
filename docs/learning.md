# 学习与部署

产品 registry 当前只发现 `nvidia-neural-appearance`。新方法实现一个 `MethodDefinition`：descriptor、trainable、objective、checkpoint state、runtime compiler、material compiler 和 source adaptation 均由定义拥有；公共 runner 不按具体模型分支。

`TrainingRunner` 只消费 `TrainingBatch@1`，`batch_source.kind` 选择 offline/live。evaluator、joint、sampler phase 共用同一生命周期。`TrainingCheckpoint@2` 严格校验方法 descriptor、训练配置、source contracts/state IDs、tensor schema，并以原子写和 SHA-256 sidecar 保存。

统一命令为：

```powershell
conda run -n neural-shading python -m ncls learn list
conda run -n neural-shading python -m ncls learn train --config <json> --output <dir>
conda run -n neural-shading python -m ncls learn evaluate --config <json> --checkpoint <pt> --output <json>
conda run -n neural-shading python -m ncls learn export --checkpoint <pt> --source <json> --output <dir>
```

部署输出只有 `ScatteringPackage@1`。当前 NVIDIA 身份仍是预算适配诊断，不改称论文忠实复现。
