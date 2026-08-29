# Phase online training 与 typed batch 合同

## 1. Scope / Trigger

修改training config、phase、route、batch、asset collection、producer、runner、checkpoint或method objective时适用。

## 2. Signatures

```text
TrainingConfig.from_dict(...) -> TrainingConfig@4
OnlineTrainingProducer.next_batch(request) -> AssetTileBatch@1 | EvaluatorBatch@3 | MethodSamplerBatch@3
MethodDefinition.training_objective(model, batches, phase) -> (loss, metrics)
TrainingRunner.run(resume=None, stop_at_step=None) -> TrainingRunResult
TrainingCheckpoint.load(path) -> TrainingCheckpoint@4
```

## 3. Contracts

- config是有序phase graph；每phase显式声明routes、parameter groups、loss terms、recipes、optimizer、schedule、precision、transition、checkpoint/metrics/audit/prefetch cadence。runner不硬编码phase名称、数量或双route。
- `NativeAssetCollection@1`统一多asset/domain/mip tile+halo traversal；1×1 source不是另一套adapter。
- asset identity包含schema、role、domain、mip与payload；tile request显式给出asset/domain/mip/origin/extent/halo。collection使用有界working-set cache和lease；活跃tile不得被evict，batch构造失败也必须释放已经取得的lease。
- 三种batch不共享dummy字段。method descriptor必须严格登记batch dependencies与parameter ownership。
- runner只启用phase groups；optimizer状态以parameter name保存，`carry-overlap`仅传递同名交集。autocast/scaler由phase配置。
- 每步GPU聚合finite检查；audit cadence验证每个required group存在nonzero gradient和实际参数更新。metrics只在log cadence同步。
- prefetch有界且不跨validation、checkpoint、phase或stop边界；checkpoint query cursor不得领先已消费batch。
- objective每次返回active required component的全部Python outputs；完成checkpoint前required groups的gradient/update coverage必须齐全。
- checkpoint严格冻结method/components/config/phase graph/reference plan/asset collection/query/source identity与当前phase optimization状态。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| phase引用未知group/route或required component永不active | 构造拒绝 |
| asset/evaluator/sampler batch类型、shape、device错 | batch/runner拒绝 |
| loss/gradient非有限、required group零梯度或未更新 | runner失败 |
| objective漏required component output | generic conformance失败 |
| resume identity或phase cursor漂移 | checkpoint恢复拒绝 |
| tile越界、role/domain不匹配或mip chain不canonical | collection在materialize前拒绝 |
| completed checkpoint缺required group finite/nonzero/update coverage | checkpoint写出拒绝 |

## 5. Good / Base / Bad Cases

- Good：MaterialX的base-color/roughness/metalness/normal按各自role共享一个collection，encoder与decoder联合训练；phase checkpoint同时冻结asset collection identity与每个required group的梯度/更新覆盖。
- Base：常量1×1资产仍经同一个tile+halo请求与lease协议，只是其canonical mip chain长度为1。
- Bad：每个batch新建collection使cache永远失效；在tile lease存活时驱逐payload；或仅因loss finite就把从未更新decoder的checkpoint标成完成。

## 6. Tests Required

- unit：三种batch、multi-asset tile+halo、phase graph、carry-overlap resume、component正负例、checkpoint v4；
- GPU：NVIDIA Python/Slang evaluator与sampler梯度、真实online两phase smoke；
- static：无固定lifecycle、旧schema reader、offline batch或family-specific producer。

## 7. Wrong vs Correct

```python
# 错：构造batch中途失败时遗留tile lease，并用固定phase名称驱动runner。
tile = collection.acquire(request)
return EvaluatorBatch(tile=tile, target=bad_target)

# 对：typed batch拥有并最终释放lease；runner只解释配置中的phase graph。
with collection.acquire(request) as tile:
    batch = EvaluatorBatch.from_tile(tile, target)
runner.run_phase(config.phases[phase_cursor], batch)
```
