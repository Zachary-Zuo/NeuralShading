# 学习与部署

产品registry当前包含`nvidia-neural-appearance`。公共训练接口已迁到`TrainingConfig@4`：config由有序phase组成，每个phase声明typed routes、parameter groups、loss terms、recipe、optimizer、cosine schedule区间、precision、checkpoint boundary、transition、metrics/gradient cadence与prefetch depth。phase名称和数量不是runner硬编码；NVIDIA reproduction通过`bootstrap → materialize-assets → finetune`实例化该合同。

三种batch各自只携带有语义的数据：`AssetTileBatch@1`携带多asset tile+halo；`EvaluatorBatch@3`携带conditioning、`wi`和online reference `target_f`；`MethodSamplerBatch@3`携带conditioning与`sample_u`。NVIDIA evaluator拟合线性`f`，matched sampler从当前learned GGX9 proposal采样并计算自身PDF，用detached learned evaluator构造forward-KL score；source sampler不是teacher。

runner按descriptor parameter registry启停梯度，phase-local Adam状态以parameter name保存；`carry-overlap`只迁移重叠状态。autocast/scaler由phase显式配置。每步在GPU聚合finite检查，cadence点才同步nonzero gradient和实际参数更新；metrics只按`log_interval`回读。prefetch queue有界预派发已脱离provider slot的batch，live lease则持有到forward/backward结束；它不跨validation、checkpoint或phase边界，因此checkpoint中的query stream游标始终与已消费batch一致。当前Falcor Python dispatch显式等待完成，尚不宣称与optimizer计算发生后台重叠。

`TrainingCheckpoint@4`保存plan、asset、query、component与phase身份，可在Windows和Linux单GPU上恢复同一online训练。统一命令为：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls learn train <config-v4.json> <checkpoint.pt>
.\scripts\run_falcor_python.ps1 -m ncls learn train <config-v4.json> <checkpoint.pt> --resume <periodic.pt>
.\scripts\run_falcor_python.ps1 -m ncls learn evaluate <config-v4.json> <checkpoint.pt> --batches 1
.\scripts\run_falcor_python.ps1 -m ncls learn export <checkpoint.pt> <package-dir> --material-index 0
```

smoke只证明真实full method shape下的数值、梯度、phase transition和短程收敛；不作为formal质量结论。部署由同一checkpoint编译`program/asset/instance`三段`ScatteringPackage@2`，并通过generic artifact conformance和packed-FP16 D3D12 parity。
