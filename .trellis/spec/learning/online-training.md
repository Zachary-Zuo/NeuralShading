# Phase online training 与 typed batch 合同

## 1. Scope / Trigger

修改training config、phase、route、batch、asset collection、producer、runner、checkpoint或method objective时适用。

## 2. Signatures

```text
TrainingConfig.from_dict(...) -> TrainingConfig@4
OnlineTrainingProducer.next_batch(request) -> AssetTileBatch@1 | EvaluatorBatch@3 | MethodSamplerBatch@3
MethodSourceAdapter.sample_tensors(source_index, generator, options) -> Mapping[str, Tensor]
expand_source_states(family, base_snapshots, typed_state_recipe) -> ExpandedSourceStates
MethodDefinition.training_objective(model, batches, phase) -> (loss, metrics)
TrainingRunner.run(resume=None, stop_at_step=None) -> TrainingRunResult
TrainingCheckpoint.load(path) -> TrainingCheckpoint@4
MetalAssetCooker.cook_asset(asset_index, mode, refinement_steps, refinement_bound) -> MetalCompiledAssetState
```

## 3. Contracts

- config是有序phase graph；每phase显式声明routes、parameter groups、loss terms、recipes、optimizer、schedule、precision、transition、checkpoint/metrics/audit/prefetch cadence。runner不硬编码phase名称、数量或双route。
- `NativeAssetCollection@1`统一多asset/domain/mip tile+halo traversal；1×1 source不是另一套adapter。
- producer在编plan前通过schema registry执行`expand_source_states()`；无recipe时也生成`ncls.identity-source-states@1` identity。Metal recipe只通过公共source editor产生Sobol/boundary/default typed states，责任组、bool/enum与hard/soft range均显式；无界continuous/int保持authored default。train/validation以recipe identity参与scramble，允许共享authored default但非默认采样state必须隔离。
- `MethodSourceAdapter.sample_tensors()`只有`(source_index, generator, options)`这一条正式签名；route options负责传递Metal的tile extent、patch size和direction proposal，不保留旧的二参数兼容入口。Metal adapter必须消费registry中exact locator、graph/schema/recipe/metal/finish/asset identity，并产生32个typed token、16维canonical optical、access/frame state以及9-slot相邻mip patch；缺失token由presence mask隔离，全部缺失时仍以global token得到有限pure compiler state。
- asset identity包含schema、role、domain、mip与payload；tile request显式给出asset/domain/mip/origin/extent/halo。collection使用有界working-set cache和lease；活跃tile不得被evict，batch构造失败也必须释放已经取得的lease。
- asset route按`asset/domain` round-robin推进，`asset_indices`只能选择已验证的子cohort；不得让第一个大纹理domain长期饿死其余role或asset。Metal source patch从canonical decoded mip chain随机访问相邻两级，保留transfer、normal renormalization和address mode，不持久化派生mip或训练batch。
- Metal新资产cook固定三条不同identity：`encoder-only`、`encoder-bounded-refinement`、`direct-optimized-control`。三者必须使用同一full-profile shared decoder和packing；bounded refinement从encoder state出发并投影到显式bound，direct control从自由state出发。优化两条路径冻结shared decoder，不能把decoder漂移混入asset state。
- `metal_fused_full_v1`训练形态固定为9 slots、32×64 typed set、64/128/192/256 shared U-Net、high/low各8通道、rank-8 adapter、6+4 evaluator lobes、11-component matched proposal和四级angular bank。smoke只能缩step、batch、tile/cohort，不能缩这些shape或关闭required branch。BF16 phase中，frame lerp与一维/二维random-access插值显式在FP32执行，其余网络继续autocast。
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
| Metal profile count、layout identity、slot/token上限或required route不精确匹配full profile | model/method config构造拒绝，不创建缩小版identity |
| Metal full method在runtime compiler完成前被要求`compile_program/compile_asset/compile_instance` | fail closed并指出runtime package边界，不生成伪package；`sample/pdf`数学实现不得随之降级 |
| optimized Metal asset path steps非正、bounded refinement超出`(0,0.5]`或encoder-only携带hidden steps | cook在编码/优化前拒绝 |
| BF16进入FP32 random-access table却使用不同dtype的lerp weight | 实现测试失败；坐标、表和weight统一提升FP32后再插值 |

## 5. Good / Base / Bad Cases

- Good：MaterialX的base-color/roughness/metalness/normal按各自role共享一个collection，encoder与decoder联合训练；phase checkpoint同时冻结asset collection identity与每个required group的梯度/更新覆盖。
- Good：同一Metal graph的base export与多个typed states进入一个group；训练batch先选group再选group-local state，checkpoint冻结完整state pool identity。
- Good：Metal typed edit只重编`MaterialProgramState`，替换texture bundle只改变`AssetState`；相邻mip分别经shared encoder/decoder后在structured state中连续插值。
- Good：Windows smoke先以FP32运行codec，再以BF16运行full joint appearance和proposal fit；所有13个required parameter groups记录finite/nonzero/update coverage。
- Base：常量1×1资产仍经同一个tile+halo请求与lease协议，只是其canonical mip chain长度为1。
- Bad：每个batch新建collection使cache永远失效；在tile lease存活时驱逐payload；或仅因loss finite就把从未更新decoder的checkpoint标成完成。
- Bad：预先离线编码52个纹理grid后只训练evaluator；把bounded refinement和direct control写入同一个asset identity；或为了mixed precision把angular lookup改成最近邻。

## 6. Tests Required

- unit：三种batch、multi-asset tile+halo、phase graph、carry-overlap resume、component正负例、checkpoint v4；
- integration：generic recipe registry扩展Metal train/validation states并验证非默认snapshot IDs不重叠、pool identity可恢复；
- GPU：NVIDIA Python/Slang evaluator与sampler梯度；Metal full-shape finite/nonnegative、typed missing/discrete与bundle分离、三路径asset cook、BF16敏感路径、Python/Slang matched proposal parity，以及真实online三phase smoke/resume/evaluate；
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

```python
# 错：旧adapter签名和预计算grid绕过了route options与joint encoder。
tensors = adapter.sample_tensors(source_index, generator)
spatial = frozen_asset_grids[asset_index]

# 对：单一正式签名按route请求真实相邻mip patch，codec端到端参与反向。
tensors = adapter.sample_tensors(source_index, generator, route.options)
spatial = model.spatial_state(tensors)
```

## Metal matched sampler 可执行合同

### 1. Scope / Trigger

修改`metal_fused_full_v1`的proposal state、component、sample/PDF、proposal loss、generated Slang layout或full-method capability时适用。该合同只拥有matched proposal；source reference sampler仍是独立control，不是训练teacher或产品fallback。

### 2. Signatures

```text
MetalPreparedState.proposal_state: float32[batch, 11, 8]
metal_sample_proposal(state, frames[batch,4,3,3], valid, wo, sample_u[batch,2])
  -> MetalProposalSample(wi, forward_pdf, reverse_pdf, valid, component, component_pdfs)
metal_proposal_pdf(state, frames, valid, wo, wi[batch,count,3])
  -> MetalProposalPdf(forward, reverse, valid, component_pdfs)
MetalFusedNeuralMaterialModel.sample_prepared(...) -> MetalScatteringSample
nclsSampleMetalFusedProposal(proposal, wo, float2 u) -> NclsMetalFusedProposalSample
nclsMetalFusedProposalPdf(proposal, wo, wi) -> float
```

### 3. Contracts

- component顺序固定为6个analytic core、4个positive residual和1个uniform full-hemisphere fallback；Python常量、`metal_fused_layout_v1.json`、generated Slang与package必须逐项相同。
- state字段固定为`normalized_weight, alpha_x, alpha_y, rotation_radians, active, frame_index, distribution_id, energy_clue`。`active`是二值topology mask；连续compiler activity只参与mixture mass，不能直接写入二值字段。
- `sample_u.x`同时执行CDF component选择并在命中区间内重映射为component radial变量，`sample_u.y`是azimuth；因此随机数上限是一个`float2`，不需要第三个随机数。
- 每个component先在其局部上半球采样，再将renderer-z为负的方向折回。PDF必须累加原方向与z镜像方向两个preimage，不能用rejection、null sample或只算一个preimage。
- fallback权重下限是0.02，保证合法非零能量state对整个renderer上半球有正密度。sample随后只调用一次公共directional evaluator，并形成`f * max(wi.z,0) / pdf.forward`；独立`pdf()`不执行texture decode、typed compiler或evaluator。
- zero/nonfinite energy clue、非法weight/active关系、错误enum/frame index、退化authored frame、grazing hemisphere、非法`float2`和prepared invalid都tensor化fail closed。反射轴偶然与旋转tangent共线不是坏state；使用确定性fallback tangent继续形成正交基。
- `proposal-fit`只启用`proposal_sampler`group，但同时消费`reference-evaluator`与`method-sampler`route；自采样forward-KL、evaluator-direction density fit、reflection/grazing/cosine mode coverage、weight-tail与sample/PDF identity必须同时执行。

### 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| component/field/order/layout identity漂移 | generator check或method preflight失败 |
| 正weight对应`active <= 0.5`、fallback无正weight或总energy clue为零 | Python/Slang proposal无效并返回零PDF |
| authored frame normal/tangent退化或含非有限值 | 整个proposal fail closed；不以fallback frame伪装有效state |
| 合法axis与tangent偶然共线 | 使用由axis确定的fallback tangent，proposal保持有效 |
| `sample_u`非有限或不在`[0,1)` | 返回invalid sample与零PDF，不触发GPU host同步 |
| sample方向重新调用独立PDF不一致，或weight identity不成立 | 数值测试失败，不用clamp或另一proposal修补 |
| proposal phase缺finite/nonzero gradient或parameter update | complete checkpoint构造失败 |

### 5. Good / Base / Bad Cases

- Good：continuous activity为0.2时mixture mass仍可微，ABI中的`active`写1；activity恰为0时权重为0且`active`写0。
- Good：tilted frame的specular axis与tangent在孤立方向共线时，sample/PDF两侧选择同一确定性fallback basis。
- Base：只有fallback component有权重时，PDF仍在整个renderer上半球积分为1。
- Bad：把continuous activity 0.2直接序列化为`active=0.2`，同时保留正weight；validator会把训练产生的合法state全部判无效。
- Bad：折回负z sample后只计算折回方向的局部密度；此时PDF不归一且sample与independent PDF不再代表同一分布。

### 6. Tests Required

- unit：11种单component+fallback mixture半球积分、sample→independent forward/reverse PDF、非法state/random/grazing/zero-energy/degenerate-frame fail closed、axis-tangent共线回归、layout/preflight闭包；
- GPU：full evaluator返回精确`[batch,direction,3]`、sample/PDF/weight identity、proposal group固定目标下降；
- Falcor/Slang：至少256个跨state/frame/wo/random tuple probes，比较sample方向、component、forward/reverse/direct PDF并覆盖11个component；
- online：full-shape三phase小步数训练，complete checkpoint断言13/13 groups finite、nonzero gradient和update，proposal identity error为零；
- static：generated layout `--check`、sample evaluator call上限1、随机数上限2、Falcor上游工作树干净。

### 7. Wrong vs Correct

```python
# 错：连续gate同时充当ABI topology flag。
proposal_state[..., ACTIVE] = activity
weight = activity * clue * exp(logit)

# 对：可微mass与二值topology分开。
proposal_state[..., ACTIVE] = (activity > 0).to(dtype)
weight = activity.float() * clue.float() * exp(logit.float())
```

```python
# 错：折回方向后只求一个局部preimage。
wi.z = abs(wi.z)
pdf = local_pdf(to_local(wi))

# 对：折叠映射的密度是两个preimage之和。
pdf = local_pdf(to_local(wi)) + local_pdf(to_local(mirror_z(wi)))
```
