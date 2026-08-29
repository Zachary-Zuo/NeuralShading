# Phase online training 与 typed batch 合同

## 1. Scope / Trigger

修改training config、phase、route、batch、asset collection、producer、runner、checkpoint或method objective时适用。

## 2. Signatures

```text
TrainingConfig.from_dict(...) -> TrainingConfig@4
OnlineTrainingProducer.next_batch(request) -> AssetTileBatch@1 | EvaluatorBatch@3 | MethodSamplerBatch@3
expand_source_states(family, base_snapshots, typed_state_recipe) -> ExpandedSourceStates
MethodDefinition.training_objective(model, batches, phase) -> (loss, metrics)
TrainingRunner.run(resume=None, stop_at_step=None) -> TrainingRunResult
TrainingCheckpoint.load(path) -> TrainingCheckpoint@4
```

## 3. Contracts

- config是有序phase graph；每phase显式声明routes、parameter groups、loss terms、recipes、optimizer、schedule、precision、transition、checkpoint/metrics/audit/prefetch cadence。runner不硬编码phase名称、数量或双route。
- `NativeAssetCollection@1`统一多asset/domain/mip tile+halo traversal；1×1 source不是另一套adapter。
- producer在编plan前通过schema registry执行`expand_source_states()`；无recipe时也生成`ncls.identity-source-states@1` identity。Metal recipe只通过公共source editor产生Sobol/boundary/default typed states，责任组、bool/enum与hard/soft range均显式；无界continuous/int保持authored default。train/validation以recipe identity参与scramble，允许共享authored default但非默认采样state必须隔离。
- asset identity包含schema、role、domain、mip与payload；tile request显式给出asset/domain/mip/origin/extent/halo。collection使用有界working-set cache和lease；活跃tile不得被evict，batch构造失败也必须释放已经取得的lease。
- 三种batch不共享dummy字段。method descriptor必须严格登记batch dependencies与parameter ownership。
- runner只启用phase groups；optimizer状态以parameter name保存，`carry-overlap`仅传递同名交集。autocast/scaler由phase配置。
- 每步GPU聚合finite检查；audit cadence验证每个required group存在nonzero gradient和实际参数更新。metrics只在log cadence同步。
- prefetch有界且不跨validation、checkpoint、phase或stop边界；checkpoint query cursor不得领先已消费batch。
- objective每次返回active required component的全部Python outputs；完成checkpoint前required groups的gradient/update coverage必须齐全。
- checkpoint严格冻结method/components/config/phase graph/reference plan/asset collection/query/source identity与当前phase optimization状态。
- producer state必须额外保存`typed_state_pool_identity`；resume同时校验pool与`query_stream_identity`，generator/group/tile cursor只能在identity一致后恢复。不能只保存seed后在新registry上重生另一批states。

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
| typed-state recipe schema/字段、registry locator或range未知 | plan/session创建前拒绝 |
| resume的typed-state pool identity漂移 | producer恢复拒绝，不恢复任何cursor |

## 5. Good / Base / Bad Cases

- Good：MaterialX的base-color/roughness/metalness/normal按各自role共享一个collection，encoder与decoder联合训练；phase checkpoint同时冻结asset collection identity与每个required group的梯度/更新覆盖。
- Good：同一Metal graph的base export与多个typed states进入一个group；训练batch先选group再选group-local state，checkpoint冻结完整state pool identity。
- Base：常量1×1资产仍经同一个tile+halo请求与lease协议，只是其canonical mip chain长度为1。
- Bad：每个batch新建collection使cache永远失效；在tile lease存活时驱逐payload；或仅因loss finite就把从未更新decoder的checkpoint标成完成。

## 6. Tests Required

- unit：三种batch、multi-asset tile+halo、phase graph、carry-overlap resume、component正负例、checkpoint v4；
- integration：generic recipe registry扩展Metal train/validation states并验证非默认snapshot IDs不重叠、pool identity可恢复；
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

```python
# 错：producer按Metal分支即时随机改参数，resume只记seed。
if family_id == "mdl.program@1":
    snapshots = random_metal_edits(seed)

# 对：schema-selected recipe先形成有identity的canonical snapshot pool。
expanded = expand_source_states(family, base_snapshots, online_query["typed_state_recipe"])
plan = compile_single_program_plan(reference, expanded.snapshots, query_recipe={
    **online_query, "typed_state_pool_identity": expanded.identity,
})
```
