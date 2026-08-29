# Online training 与 typed batch 合同

## 1. Scope / Trigger

修改 training config、route、batch、producer、runner、checkpoint或NVIDIA objective时适用。

## 2. Signatures

```text
TrainingConfig.from_dict(...) -> TrainingConfig@3
OnlineTrainingProducer.next_batch(request) -> EvaluatorBatch@2 | MethodSamplerBatch@2
MethodDefinition.training_objective(model, batches, lifecycle) -> (loss, metrics)
TrainingRunner.run(resume=None, stop_at_step=None) -> TrainingRunResult
TrainingCheckpoint.load(path) -> TrainingCheckpoint@3
```

## 3. Contracts

- config的`source.materials[*].locator`由source family解释；`online_query`只描述版本化query recipe。
- `EvaluatorBatch`字段是conditioning、`wi`、线性`target_f`；`MethodSamplerBatch`字段是conditioning、`sample_u`。二者schema不共享dummy tensor。
- method/source adapter只生成native features、UV/LOD/footprint与materialization pyramid，不拥有reference math。没有注册adaptation时fail closed。
- NVIDIA evaluator为`f_hat=exp(raw-3)`，直接以`log1p(f_hat)`对`log1p(target_f)`。
- NVIDIA sampler从当前learned GGX9 proposal取样并计算自身PDF；target density为`stopgrad(luminance(f_hat)*abs(wi.z))`。source `sample/pdf`不进入该loss。
- runner管理optimizer、scheduler、lifecycle、batch release与route RNG；checkpoint保存source snapshot/reference/query/adapter identity并严格恢复。
- `nvidia.mdl-fixed-uniform@1` 只接受一个`mdl.program@1` snapshot与1×1 materialization；64个固定parameter slots编码`bool/int/float/double/enum/color/float2/3/4`，拒绝texture/resource、未知类型、非有限值、多snapshot和超限参数。它是method adaptation，不改变MDL reference语义。
- online producer只持有`ReferenceBackendCapability`和其返回的session；checkpoint中的reference identity已包含backend semantic/build identity。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| route不是一条evaluator加一条method-sampler | config拒绝 |
| legacy offline/recorded/HDF5或`query_role/target_estimator`字段 | config拒绝 |
| method不支持source adaptation | producer构造失败 |
| MDL fixed adapter含空间纹理、未知类型、多snapshot、非有限值或超过64参数 | adapter构造失败 |
| sampler batch含reference target或缺`sample_u` | typed batch/descriptor拒绝 |
| loss、gradient或checkpoint tensor非有限 | runner失败，不写成功产物 |
| resume source/query/recipe/implementation identity漂移 | checkpoint恢复拒绝 |

## 5. Good / Base / Bad Cases

- Good：同一`learn train`命令只替换source locator和匹配的method config，即可在LayerStack、MaterialX和固定MDL间切换；target仍来自同一backend session。
- Base：sampler route不触发source dispatcher，独立seed stream仍随checkpoint恢复。
- Bad：source sampler方向作为NVIDIA sampler teacher，或运行时从`f·cos`除以cos恢复`f`。

## 6. Tests Required

- unit：batch exact fields、route independence、KL cosine、gradient ownership、config/checkpoint v3、resume bitwise state。
- GPU：Python/Slang `f` parity、GGX9 sample/pdf parity、packed-FP16 package expected_f。
- smoke：LayerStack、MaterialX与`effect-pigment-metallic`固定MDL均跨过materialization step并产出checkpoint；MDL额外验证checkpoint reload/evaluate。

## 7. Wrong vs Correct

```python
# 错
prediction_response = model(...)
prediction_f = prediction_response / clamp(abs(wi_z))

# 对
prediction_f = model.evaluate_f(latent, wo, wi)
```
