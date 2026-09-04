# Phase online training 与 typed batch 合同

## 1. Scope / Trigger

修改 YAML training plan、phase、route、batch、asset collection、online data session、engine、checkpoint、hook 或 method facet 时适用。

## 2. Signatures

```text
TrainingPlanResolver.resolve(run_yaml, devices=None) -> ResolvedTrainingPlan@1
ResolvedTrainingPlan.to_runtime_config() -> internal TrainingConfig
DataExecutionPlan.build(...) -> DataExecutionPlan@1
OnlineTrainingProducer.next_batch(request) -> AssetTileBatch@1 | EvaluatorBatch@3 | MethodSamplerBatch@3
OnlineDataSession.next_batch/state_dict/load_state_dict/drain/close
MethodSourceAdapter.sample_tensors(source_index, generator, options) -> Mapping[str, Tensor]
expand_source_states(family, base_snapshots, typed_state_recipe) -> ExpandedSourceStates
MethodPlugin.objective.compute(model, batches, phase) -> (loss, metrics)
TrainingEngine.run(resume=None, stop_at_step=None) -> TrainingRunResult
save/load_training_checkpoint_v1(path) -> TrainingCheckpoint@1
load_evaluation_snapshot(path) -> EvaluationSnapshot
assess_checkpoint_readiness(checkpoint, descriptor, mode="formal" | "diagnostic-evaluator")
  -> CheckpointReadiness@1
MetalAssetCooker.cook_asset(asset_index, mode, refinement_steps, refinement_bound) -> MetalCompiledAssetState
```

## 3. Contracts

- 正式配置是 `base/method/data/recipe` 组合 YAML；resolver 严格展开并冻结 `ResolvedTrainingPlan@1`。公开 method key 不含 `@`。内部 runtime config 是 engine 的 phase graph 投影，不是用户输入格式。
- config是有序phase graph；每phase显式声明routes、parameter groups、loss terms、recipes、optimizer、schedule、precision、transition、checkpoint/metrics/audit/prefetch cadence。engine不硬编码phase名称、数量或双route。
- `NativeAssetCollection@1`统一多asset/domain/mip tile+halo traversal；1×1 source不是另一套adapter。
- producer在编plan前通过schema registry执行`expand_source_states()`；无recipe时也生成`ncls.identity-source-states@1` identity。Metal recipe只通过公共source editor产生Sobol/boundary/default typed states，责任组、bool/enum与hard/soft range均显式；无界continuous/int保持authored default。train/validation以recipe identity参与scramble，允许共享authored default但非默认采样state必须隔离。
- `MethodSourceAdapter.sample_tensors()`只有`(source_index, generator, options)`这一条正式签名；route options负责传递Metal的tile extent、patch size和direction proposal，不保留旧的二参数兼容入口。Metal adapter必须消费registry中exact locator、graph/schema/recipe/metal/finish/asset identity，并产生32个typed token、16维canonical optical、access/frame state以及9-slot相邻mip patch；缺失token由presence mask隔离，全部缺失时仍以global token得到有限pure compiler state。
- asset identity包含schema、role、domain、mip与payload；tile request显式给出asset/domain/mip/origin/extent/halo。collection使用有界working-set cache和lease；活跃tile不得被evict，batch构造失败也必须释放已经取得的lease。
- asset route按`asset/domain` round-robin推进，`asset_indices`只能选择已验证的子cohort；不得让第一个大纹理domain长期饿死其余role或asset。Metal source patch从canonical decoded mip chain随机访问相邻两级，保留transfer、normal renormalization和address mode，不持久化派生mip或训练batch。
- MDL decoded payload的物理通道数可能少于registry语义通道数（例如灰度`Sint8`对应单一`RGB` role）；asset collection只允许已定义的scalar→RGB广播和RGB→RGB+A默认alpha补全，其余不完整layout必须fail closed，不能靠reshape、截断或重复通道掩盖schema错误。
- Metal新资产cook固定三条不同identity：`encoder-only`、`encoder-bounded-refinement`、`direct-optimized-control`。三者必须使用同一full-profile shared decoder和packing；bounded refinement从encoder state出发并投影到显式bound，direct control从自由state出发。优化两条路径冻结shared decoder，不能把decoder漂移混入asset state。
- `metal_fused_full_v1`训练形态固定为9 slots、32×64 typed set、64/128/192/256 shared U-Net、high/low各8通道、rank-8 adapter、6+4 evaluator lobes、11-component matched proposal和四级angular bank。smoke只能缩step、batch、tile/cohort，不能缩这些shape或关闭required branch。BF16 phase中，frame lerp与一维/二维random-access插值显式在FP32执行，其余网络继续autocast。
- 三种batch不共享dummy字段。method descriptor必须严格登记batch dependencies与parameter ownership。
- engine只启用phase groups；optimizer状态以parameter name保存，`carry-overlap`仅传递同名交集。autocast/scaler由phase配置。方法只能通过 plugin lifecycle/objective facet 提供行为，不能拥有专用 runner。
- 每步GPU聚合finite检查；audit cadence验证每个required group存在nonzero gradient和实际参数更新。metrics只在log cadence同步。
- optional producer profile在training与validation之间严格分账。validation开始前先把尚未落盘的training counter暂存，validation结束后以`profile/validation_reference_*`写入首条validation row并reset；下一条training row只合并`profile/reference_*`训练计数。counter求和、`*_max`取最大、`resident_groups`取最后状态。
- prefetch有界且不跨validation、checkpoint、phase或stop边界；checkpoint query cursor不得领先已消费batch。
- objective每次返回active required component的全部Python outputs；完成checkpoint前required groups的gradient/update coverage必须齐全。
- checkpoint v1 严格冻结 resolved plan、method/facet、data execution/reference/asset/query/source identity、逐 rank RNG/data cursor、hook cursor 与当前 phase optimization 状态。发布前必须调用 data session `drain()`。
- 旧 checkpoint v4 只经 `LegacyCheckpointV4Importer` 产生 evaluation snapshot；不得 resume，也不得恢复旧 JSON config reader或 method alias。
- producer state必须额外保存`typed_state_pool_identity`；resume同时校验pool与`query_stream_identity`，generator/group/tile cursor只能在identity一致后恢复。不能只保存seed后在新registry上重生另一批states。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| phase引用未知group/route或required component永不active | 构造拒绝 |
| asset/evaluator/sampler batch类型、shape、device错 | batch/engine拒绝 |
| loss/gradient非有限、required group零梯度或未更新 | engine失败 |
| objective漏required component output | generic conformance失败 |
| resume identity或phase cursor漂移 | checkpoint恢复拒绝 |
| tile越界、role/domain不匹配或mip chain不canonical | collection在materialize前拒绝 |
| formal export不是`run_class=formal`、phase不是`complete`，或缺required group finite/nonzero/update coverage | readiness拒绝；不生成正式package/catalog |
| diagnostic evaluator preview没有exact descriptor、step-1端到端evaluator组覆盖或合法phase | readiness拒绝；不得用旧tensor shape兼容冒充可部署状态 |
| validation backend counter出现在下一条training profile | engine profile隔离测试失败；不得据此归因冷启动或长期吞吐 |
| v4 checkpoint 传给新 train resume | v1 loader明确拒绝；只读 evaluation importer 仍可加载 |
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
- Good：Windows smoke从第1步同时执行asset/evaluator/sampler三条route，13个required parameter groups都记录finite/nonzero/update coverage；第二phase再以FP32执行部署态QAT。
- Base：常量1×1资产仍经同一个tile+halo请求与lease协议，只是其canonical mip chain长度为1。
- Bad：每个batch新建collection使cache永远失效；在tile lease存活时驱逐payload；或仅因loss finite/shape compatible就把非formal、未完成checkpoint标成正式可部署。
- Bad：预先离线编码52个纹理grid后只训练evaluator；把bounded refinement和direct control写入同一个asset identity；或为了mixed precision把angular lookup改成最近邻。

## 6. Tests Required

- unit：YAML resolve/plugin、三种batch、multi-asset tile+halo、phase graph、carry-overlap resume、component正负例、checkpoint v1 与 legacy v4只读；
- integration：generic recipe registry扩展Metal train/validation states并验证非默认snapshot IDs不重叠、pool identity可恢复；
- GPU：NVIDIA Python/Slang evaluator与sampler梯度；Metal full-shape finite/nonnegative、typed missing/discrete与bundle分离、三路径asset cook、BF16/QAT敏感路径、Python/Slang matched proposal parity，以及真实online两phase smoke/resume/evaluate；
- static：无固定lifecycle、旧schema reader、offline batch或family-specific producer。

## 7. Wrong vs Correct

```python
# 错：构造batch中途失败时遗留tile lease，并用固定phase名称驱动engine。
tile = collection.acquire(request)
return EvaluatorBatch(tile=tile, target=bad_target)

# 对：typed batch拥有并最终释放lease；engine只解释resolved plan中的phase graph。
with collection.acquire(request) as tile:
    batch = EvaluatorBatch.from_tile(tile, target)
engine.run(resume=checkpoint, stop_at_step=target)
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
- proposal从`joint-coarse-to-fine`第1步开始与appearance共同训练。自采样forward-KL、evaluator-direction density fit、reflection/grazing/cosine mode coverage、weight-tail与sample/PDF identity必须同时执行；其evaluator response是detached target，functional call只允许`proposal_sampler`拥有proposal objective梯度，防止proposal loss反向拖动evaluator去迎合当前采样器。

### 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| component/field/order/layout identity漂移 | generator check或method preflight失败 |
| 正weight对应`active <= 0.5`、fallback无正weight或总energy clue为零 | Python/Slang proposal无效并返回零PDF |
| authored frame normal/tangent退化或含非有限值 | 整个proposal fail closed；不以fallback frame伪装有效state |
| 合法axis与tangent偶然共线 | 使用由axis确定的fallback tangent，proposal保持有效 |
| `sample_u`非有限或不在`[0,1)` | 返回invalid sample与零PDF，不触发GPU host同步 |
| sample方向重新调用独立PDF不一致，或weight identity不成立 | 数值测试失败，不用clamp或另一proposal修补 |
| proposal在首phase被禁用、权重从0开始，或缺finite/nonzero gradient/parameter update | config/complete checkpoint构造失败 |

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
- online：full-shape两phase小步数训练，首个audit即断言13/13 groups finite、nonzero gradient和update，complete checkpoint的proposal identity error为零；
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

## Metal step-1端到端、两phase QAT与Linux交接合同

### 1. Scope / Trigger

修改`metal_fused_full_v1`的phase graph、运行时量化训练、训练profile/review、Metal Windows smoke或Linux smoke/long config时适用。它防止把QAT写成前三phase的别名、把3-export正确性子集误作全族训练，或用Windows结果冒充Linux长训gate。

### 2. Signatures

```text
fake_quantize_fp16_ste(master: Tensor) -> deployed_value_with_master_gradient
MethodPlugin.objective.compute(..., phase.name="qat-refine")
  batches == {asset: AssetTileBatch, evaluator: EvaluatorBatch, sampler: MethodSamplerBatch}
TrainingEngine.run(resume=None, stop_at_step=N) -> checkpoint at exact global step N
TrainingPlanResolver.resolve("configs/training/runs/metal-linux-long.yaml")
build_training_review(config, descriptor, checkpoint, metric_rows, ...) -> ncls.training-review@1
python -m tools.learning.build_metal_linux_handoff --output <handoff.json>
```

### 3. Contracts

- phase顺序固定为`joint-coarse-to-fine → qat-refine`。首phase从第1步同时执行codec、pure typed compiler、optimized-state teacher、full evaluator与matched proposal，13个parameter groups全部active；粗到细只能通过冻结的loss-weight/schedule实现，不能用“先只训codec”的phase切断目标response梯度。QAT继续执行完整部署路径，但training-only optimized-state teacher不在QAT optimizer中。
- QAT保留FP32 master。`metal_runtime_parameter_names()`登记的最终部署weights只在functional forward中使用`x + (fp16(x)-x).detach()`；非runtime encoder/semantic head/optimized teacher不做FP16伪量化。codec high/low grid继续使用已有INT8 per-channel STE；phase precision为FP32，使FP16 storage模拟后的值按runtime语义进入FP32敏感累积。
- objective必须返回joint与proposal全部component trace，再附`runtime_fp16_quantization_trace`；fake quantization不能原地覆盖model state、改变parameter names、optimizer引用或checkpoint schema。
- 每个phase的`proposal_weight`固定为`linear-nonzero-ramp@1`且`start > 0`。proposal objective通过`torch.func.functional_call`冻结非proposal参数；QAT时proposal可读取FP16 STE runtime值，但梯度仍只归`proposal_sampler`。
- Windows smoke使用registry机械生成的3-export activation set和3个对应asset index，只缩source/batch/step；model shape、两phase route/loss/precision和所有required components不缩。fixed-stream micro-overfit每次重新执行authoritative reference，只恢复同一query cursor，不保存response batch。
- Linux smoke/long都显式包含registry全部692个opaque export和52个asset。两者source、typed-state recipe、model、route options、groups、loss、precision、optimizer与batch geometry完全相同；只允许run class/recipe identity、phase step budget、schedule total/offset和log/audit/validation cadence不同。evaluator/sampler的`direction_count`固定1；吞吐通过group-homogeneous `batch_size`扩展。
- online query必须声明`group_schedule={recipe:"group-block-balanced@1", weight:"record-count", block_steps:64, validation_offset_blocks:104729}`。cycle先让每个execution group各出现一次，再按record count确定性加权；一个64-step block与一个DDP rank只绑定一个group，避免每step切group导致反复materialize。validation在同一cycle中使用冻结的block offset，方向/source RNG和group选择都与training流独立但可恢复。reference session只请求训练实际使用的`evaluate` operation，不构建reference `sample/pdf` pass。
- `--stop-at-step N`只允许`resume.global_step <= N <= config.total_steps`，正常写出含optimizer/scheduler/precision/RNG/query cursor的checkpoint。跨config恢复仍拒绝；Linux long先用long config自身停点，再以同一config恢复。
- cadence记录step wall与batch prepare的window count/mean/median/p90/max、phase-local与rolling step rate/ETA、48-bit group ID前缀及window group count、candidate/rejected/round统计、forward/backward/optimizer GPU event、training/validation分账的session hit/miss/create/evict、runtime/pass/resource/slot build、operation dispatch/residency、validation/checkpoint write、allocated/reserved peak memory与显式sync；普通非log/audit step不新增同步。review用rolling而非run-global累计rate汇总；固定window/bootstrap delta、VRAM/time/bytes是report-only，不自动触发formal、追加seed、ablation或Pareto。
- 正式导出要求exact method identity、`run_class=formal`、`phase_name=complete`和全部required group覆盖。未完成checkpoint只能经显式`diagnostic-evaluator`模式导出evaluate-only preview，capability必须移除`sample/pdf`，UI/capture明确标注diagnostic；tensor shape compatible不是readiness。

### 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| phase graph不是两phase、首phase缺任一route/group/loss、proposal weight为0，或QAT不是FP32敏感累积recipe | method config构造失败 |
| runtime state未出现在functional parameter/buffer tree、或对非浮点runtime tensor做FP16 STE | QAT forward失败，不退回未量化forward |
| QAT只执行joint或proposal一侧、漏required output | generic objective conformance失败 |
| evaluator/sampler `direction_count != 1` | config验证失败；不把provider的单方向合同静默改义 |
| Linux config少于692 source、source/loss/precision/batch geometry漂移 | generated-form/config-pair验证失败 |
| group schedule字段漂移、block/validation offset非正或weight不是record-count | method/producer构造失败，不退回per-step group轮转或同group validation |
| training请求reference `sample/pdf`，或block内持续出现session miss/create | 性能结构门失败；先查operation plan/session lifecycle，不用增加超时掩盖 |
| 多值/UUID `CUDA_VISIBLE_DEVICES` 在单卡入口，或DDP rank/GPU列表不一致 | Linux launcher/交接合同拒绝 |
| review metric含NaN/Inf或属于另一config | review生成失败 |
| Windows package与MDL viewer catalog的source snapshot不同 | viewer slot为unsupported；生成同locator catalog后重跑，不放宽snapshot identity |

### 5. Good / Base / Bad Cases

- Good：第1个joint audit中codec/compiler/evaluator/proposal 13组都有真实更新；QAT functional call临时替换runtime weights为FP16-rounded STE tensor，optimizer仍持有原FP32 parameter，完成step后`state_dict()`名称和值域合同不变。
- Good：Linux smoke与long均加载692 locator；smoke使用long相同的asset 12/evaluator 64/sampler 64 geometry，因此目标机review可用于估算long的VRAM和ETA。
- Good：第9个group进入容量8的session时只有block首个request发生一次miss/create/evict，后续63步均hit且resident保持8；validation事件只出现在`profile/validation_reference_*`。
- Base：Windows固定query micro-overfit只使用3-export activation subset，但每次重新发出reference query且`target_f`始终在`cuda:0`，它只证明数值可优化，不宣称泛化质量。
- Bad：导出前调用`quantize_runtime_model()`原地覆盖master后继续训练；或QAT阶段只改INT8 grid而不让FP16 runtime weights进入forward。
- Bad：前20k只训练codec，随后把该checkpoint当作可部署evaluator；或每step跨group轮转并把懒materialize时间误判为模型反向变慢。
- Bad：Linux long沿用3-export Windows source list；或把`direction_count`改成8绕过producer的单方向合同。

### 6. Tests Required

- unit：FP16 STE前向等于round-trip FP16且gradient为1；proposal objective只给proposal group梯度；QAT groups覆盖全部runtime parameter groups但排除optimized teacher；两phase YAML resolve精确验证；formal/diagnostic readiness正负例；training/validation backend profile隔离；Linux smoke/long plan source与batch合同一致；handoff无自动后续。
- GPU：full model BF16/QAT finite、13组gradient/update、fixed proposal下降；真实Windows两phase在首个QAT step停点并恢复至complete。
- online：fixed query cursor下两phase分别重复authoritative reference，初尾window记录真实下降且无response持久化；完整formal checkpoint evaluate与package export通过。
- viewer：package、catalog与scene material的source snapshot一致；PT/deferred两个slot ready且linear EXR finite。
- static：YAML plan resolve、full-cohort preflight、Linux launcher shell syntax、DDP rank0/checkpoint语义、Falcor clean、`git diff --check`。

### 7. Wrong vs Correct

```python
# 错：原地把master变成FP16 round-trip，再继续optimizer step。
state[name] = state[name].half().float()
model.load_state_dict(state)

# 对：functional forward读取部署值，gradient回到原master。
runtime_value = master + (master.half().to(master.dtype) - master).detach()
loss = torch.func.functional_call(execution, functional_state, (batches,))[0]
```

```jsonc
// 错：Linux long只沿用Windows三个激活preset，或者用多方向改变provider语义。
{"source_count": 3, "evaluator": {"batch_size": 8, "direction_count": 8}}

// 对：全族source，单方向语义不变，通过batch size提高占用率。
{"source_count": 692, "evaluator": {"batch_size": 64, "direction_count": 1}}
```
