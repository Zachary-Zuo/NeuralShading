# Source、reference 与在线训练查询

五个正式source family通过`SourceFamilyDefinition.load_snapshot(locator)`产生canonical `SourceSnapshot`。LayerStack使用多层随机游走reference，OpenPBR、MERL、MaterialX与MDL使用各自权威实现；pbrt和falcor2只处在隔离验证边界。

每个source contract在registry中唯一映射到一个`ReferenceProgramDefinition`。reference将snapshot编译为`RuntimePayload`和`MaterialPayload`，所有buffer、texture、sampler与动态source module都带typed descriptor。`ReferenceBackendSession`只读这些descriptor，通过同一个family-agnostic shader执行：

```text
prepare(context, compiledMaterial)
evaluate(wi) → f, pdf, event, valid
sample(seed) → wi, weight, pdf, eta, event, valid
pdf(wi) → forward, reverse, valid
```

`evaluate().f`是线性RGB材质值，不含几何余弦。stochastic reference可用`evaluation_samples`重复求值并只平均`f`；非有限或合同不一致的结果标为invalid，不做亮度clamp或异常值过滤。

正式训练只有online路径。`OnlineTrainingProducer`持有公共`ReferenceBackendCapability`，组合route RNG、surface conditioning、method/source adapter与session结果：

- `reference-evaluator`生成`wo/wi`并返回`EvaluatorBatch(target_f)`；
- `method-sampler`生成独立conditioning与`sample_u`，不调用source scattering query，也不携带target。

带normal map的材质会改变局部shading frame，因此世界半球方向可能落到材质局部horizon以下。producer保留有效行并继续补采，在provenance中写`candidate_count`、`rejected_count`和`rejection_rounds`。这是proposal rejection，不是噪点清洗；达到轮次上限仍无法填满时明确失败。

训练response始终留在同一CUDA device，通过显式CUDA↔Falcor同步与双slot lease管理生命周期。项目不保存或读取HDF5、shard、corpus或recorded batch；磁盘只保存source资产、checkpoint、package与`artifacts/`中的诊断/验证结果。

公共backend manifest只管理项目源码、锁定的第三方源码/toolchain与编译布局，不管理source assets。Linux部署在`assets/`不存在时也必须完成compile deployment与仓库fixture probe；用户复制资产后，才运行五个真实source snapshot和training验收。
