# 学习与部署

产品registry当前包含`nvidia-neural-appearance`与`metal-fused-neural-material`。公共训练接口使用`TrainingConfig@4`：config由有序phase组成，每个phase声明typed routes、parameter groups、loss terms、recipe、optimizer、cosine schedule区间、precision、checkpoint boundary、transition、metrics/gradient cadence与prefetch depth。phase名称和数量不是runner硬编码；NVIDIA reproduction通过`bootstrap → materialize-assets → finetune`实例化该合同，Metal完整方法通过`joint-coarse-to-fine → qat-refine`实例化同一合同。

三种batch各自只携带有语义的数据：`AssetTileBatch@1`携带多asset tile+halo；`EvaluatorBatch@3`携带conditioning、`wi`和online reference `target_f`；`MethodSamplerBatch@3`携带conditioning与`sample_u`。NVIDIA evaluator拟合线性`f`，matched sampler从当前learned GGX9 proposal采样并计算自身PDF，用detached learned evaluator构造forward-KL score；source sampler不是teacher。

Metal从第1步同时消费三种batch：codec reconstruction只是辅助项，online reference appearance loss贯穿codec、typed compiler、prepare和evaluator；proposal sampler也从第1步以非零冻结schedule训练，其target对evaluator与shared representation显式detach。末期QAT继续同一appearance/proposal目标，只模拟将部署的FP16 weight与INT8 grid精度，不再把独立codec或proposal阶段当成首次训练。

runner按descriptor parameter registry启停梯度，phase-local Adam状态以parameter name保存；`carry-overlap`只迁移重叠状态。autocast/scaler由phase显式配置。每步在GPU聚合finite检查，cadence点才同步nonzero gradient和实际参数更新；metrics只按`log_interval`回读。每个log window保存完整step wall与prepare wall的count/mean/p90/max、phase-local/rolling rate、group ID前缀与访问数、candidate/rejection，以及reference session hit/miss/create/evict、runtime/pass/resource/slot build、operation dispatch、resident group和allocated/reserved显存；中间冷构建尖峰不会再被“只记录窗口最后一步”掩盖。training与validation的reference profile独立reset并分别写入`profile/reference_*`和`profile/validation_reference_*`，验证冷启动不会污染下一段训练窗口。prefetch queue有界预派发已脱离provider slot的batch，live lease则持有到forward/backward结束；它不跨validation、checkpoint或phase边界，因此checkpoint中的query stream游标始终与已消费batch一致。当前Falcor Python dispatch显式等待完成，尚不宣称与optimizer计算发生后台重叠。

Metal online query使用`group-block-balanced@1`：同一global step的evaluator与sampler绑定同一execution group，每个group连续服务冻结的64步，完整cycle按group record数加权；validation使用冻结的104729-block offset，使其RNG与group都独立于training但仍可确定恢复。reference session只materialize训练实际请求的`evaluate` pass；public session默认仍保留`evaluate/sample/pdf`。这保持GPU online target、确定性resume与有界residency，同时消除“小LRU配逐步轮转”造成的稳态重建。

`TrainingCheckpoint@4`保存plan、asset、query、component与phase身份，可在Windows和Linux单GPU上恢复同一online训练。统一命令为：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls learn train <config-v4.json> <checkpoint.pt>
.\scripts\run_falcor_python.ps1 -m ncls learn train <config-v4.json> <checkpoint.pt> --stop-at-step <global-step>
.\scripts\run_falcor_python.ps1 -m ncls learn train <config-v4.json> <checkpoint.pt> --resume <periodic.pt>
.\scripts\run_falcor_python.ps1 -m ncls learn evaluate <config-v4.json> <checkpoint.pt> --batches 1
.\scripts\run_falcor_python.ps1 -m ncls learn export <checkpoint.pt> <package-dir> --material-index 0
```

每次训练同时生成`.metrics.jsonl`、`.summary.json`与`.review.json`；review的初尾window/bootstrap、耗时和内存只作观察，不自动触发后续实验。smoke/profile只证明真实full method shape下的数值、梯度、phase transition、短程学习与部署链路，不作为formal质量结论。

正式`learn export`与默认viewer catalog同时要求：checkpoint与当前method exact identity、`run_class=formal`、phase为`complete`，且全部required parameter group已有finite/nonzero-gradient/actual-update coverage。Metal中间结果只能通过显式`--diagnostic-preview`生成evaluate-only预览；它保持exact identity，但移除`sample/pdf` capability并在catalog、package与capture中标记`exact-diagnostic-evaluator-preview`。仅tensor shape兼容、执行finite或slot显示`ready`都不代表训练完成或材质已学会。

部署由同一checkpoint编译`program/asset/instance`三段`ScatteringPackage@2`，并通过generic artifact conformance和packed-FP16 D3D12 parity。Metal的Windows验证、692-export Linux smoke/long配置与恢复命令见[Metal Linux训练交接](metal_linux_training.md)。
