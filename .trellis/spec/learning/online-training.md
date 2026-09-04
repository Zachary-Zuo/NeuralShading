# Phase online training 与 typed batch 合同

## 1. Scope / Trigger

修改 YAML training plan、phase、route、batch、asset collection、online data session、engine、checkpoint、hook 或 method facet 时适用。

## 2. Signatures

```text
TrainingPlanResolver.resolve(run_yaml, devices=None) -> ResolvedTrainingPlan@1
ResolvedTrainingPlan.to_runtime_config() -> internal TrainingConfig
DataExecutionPlan.build(...) -> DataExecutionPlan@1
OnlineTrainingProducer.prefetch_steps(tuple[OnlineStepRequest, ...]) -> None
OnlineTrainingProducer.produce_steps(tuple[OnlineStepRequest, ...])
  -> Sequence[Mapping[str, AssetTileBatch@1 | EvaluatorBatch@3 | MethodSamplerBatch@3]]
PipelineOnlineDataSession.submit_step/acquire_step/state_dict/load_state_dict/drain/close
MethodSourceAdapter.sample_tensors(source_index, generator, options) -> Mapping[str, Tensor]
expand_source_states(family, base_snapshots, typed_state_recipe) -> ExpandedSourceStates
MethodPlugin.objective.compute(model, batches, phase) -> (loss, metrics)
TrainingEngine.run(resume=None, stop_at_step=None) -> TrainingRunResult
save/load_training_checkpoint_v1(path) -> TrainingCheckpoint@1
load_evaluation_snapshot(path) -> EvaluationSnapshot
assess_checkpoint_readiness(checkpoint, descriptor, mode="formal" | "diagnostic-evaluator")
  -> CheckpointReadiness@1
load_metric_rows(path, config_sha256, allow_empty=False) -> metric rows
MetalBudgetedAssetCooker.cook(encoded_values, mode, objective, refinement_steps,
                              refinement_bound) -> MetalBudgetedCookedAsset
```

## 3. Contracts

- 正式配置是 `base/method/data/recipe` 组合 YAML；resolver 严格展开并冻结 `ResolvedTrainingPlan@1`。公开 method key 不含 `@`。内部 runtime config 是 engine 的 phase graph 投影，不是用户输入格式。
- `execution.batch_size_multiplier` 是可选的正整数执行几何轴，默认1且不进入旧 plan 的序列化，因此旧 checkpoint 身份保持稳定。显式设置时，`to_runtime_config()`等比例放大所有 phase/route 的per-rank batch，放大后的batch进入runtime config SHA；它只用于有界吞吐探针或新冻结实验，不能在同一个matched run中途改变。
- config是有序phase graph；每phase显式声明routes、parameter groups、loss terms、recipes、optimizer、schedule、precision、transition、checkpoint/metrics/audit/prefetch cadence。engine不硬编码phase名称、数量或双route。
- `NativeAssetCollection@1`统一多asset/domain/mip tile+halo traversal；1×1 source不是另一套adapter。
- producer在编plan前通过schema registry执行`expand_source_states()`；无recipe时也生成`ncls.identity-source-states@1` identity。Metal recipe只通过公共source editor产生Sobol/boundary/default typed states，责任组、bool/enum与hard/soft range均显式；无界continuous/int保持authored default。train/validation以recipe identity参与scramble，允许共享authored default但非默认采样state必须隔离。
- `MethodSourceAdapter.sample_tensors()`只有`(source_index, generator, options)`这一条正式签名；route options负责传递Metal的tile extent、patch size和direction proposal，不保留旧的二参数兼容入口。Metal adapter必须消费registry中exact locator、graph/schema/recipe/metal/finish/asset identity，并产生32个typed token、16维canonical optical、access/frame state以及9-slot相邻mip patch；缺失token由presence mask隔离，全部缺失时仍以global token得到有限pure compiler state。
- asset identity包含schema、role、domain、mip与payload；tile request显式给出asset/domain/mip/origin/extent/halo。collection使用有界working-set cache和lease；活跃tile不得被evict，batch构造失败也必须释放已经取得的lease。
- asset route按`asset/domain` round-robin推进，`asset_indices`只能选择已验证的子cohort；不得让第一个大纹理domain长期饿死其余role或asset。Metal source patch从canonical decoded mip chain随机访问相邻两级，保留transfer、normal renormalization和address mode，不持久化派生mip或训练batch。
- MDL decoded payload的物理通道数可能少于registry语义通道数（例如灰度`Sint8`对应单一`RGB` role）；asset collection只允许已定义的scalar→RGB广播和RGB→RGB+A默认alpha补全，其余不完整layout必须fail closed，不能靠reshape、截断或重复通道掩盖schema错误。
- canonical Metal资产cook固定三条不同identity：`encoder-only@1`、`bounded-refinement@1`、`direct-control@1`。三者共用相同的两个RGBA8 SNORM部署asset shape；bounded refinement从encoder结果出发并投影到显式bound，direct control从独立可优化state出发。优化路径不得改变shared runtime decoder，也不得冒充pure compiler/editability证据。
- `metal_budgeted_hybrid_v3`和`metal_budgeted_direct_control_v3`共享同一静态layout：两次asset读取、`24→32→32→24` prepare decoder、Detail四通道到前四个frame semantic分量的无参数residual、完整24维semantic state输入的`44→64→64→64→6` evaluator、160 B PreparedState；hybrid/direct差异只能是已登记的final response解释。v1/v2 budgeted profile只解释历史诊断checkpoint；旧`metal_fused_full_v1`只在历史任务与显式control中保留，不是product registry或新训练配置的兼容目标。
- 三种batch不共享dummy字段。method descriptor必须严格登记batch dependencies与parameter ownership。
- engine只启用phase groups；optimizer状态以parameter name保存，`carry-overlap`仅传递同名交集。autocast/scaler由phase配置。方法只能通过 plugin lifecycle/objective facet 提供行为，不能拥有专用 runner。
- 每步GPU聚合finite检查；audit cadence验证每个required group存在nonzero gradient和实际参数更新。metrics只在log cadence同步。
- optional producer profile在training与validation之间严格分账。validation开始前先把尚未落盘的training counter暂存，validation结束后以`profile/validation_reference_*`写入首条validation row并reset；下一条training row只合并`profile/reference_*`训练计数。counter求和、`*_max`取最大、`resident_groups`取最后状态。
- prefetch有界且不跨validation、checkpoint、phase或stop边界；engine只依赖平台无关的step/session合同。host预取不推进cursor，checkpoint的logical/request/group/tile cursor不得领先已消费batch。
- objective每次返回active required component的全部Python outputs；完成checkpoint前required groups的gradient/update coverage必须齐全。
- checkpoint v1 严格冻结 resolved plan、method/facet、data execution/reference/asset/query/source identity、逐 rank RNG/data cursor、hook cursor 与当前 phase optimization 状态。发布前必须调用 data session `drain()`。
- `stop_at_step=0` 是合法的 train-only initialization/calibration checkpoint：rank 0仍须原子写 checkpoint、summary与review，但此时 metrics可以为空，review以0 record、0 rate和未完成coverage如实表示。`load_metric_rows(..., allow_empty=True)`只允许最终checkpoint的`global_step == 0`调用；任何正step训练仍要求至少一条metric，不能用该参数掩盖日志丢失。
- 旧 checkpoint v4 只经 `LegacyCheckpointV4Importer` 产生 evaluation snapshot；不得 resume，也不得恢复旧 JSON config reader或 method alias。
- producer state必须额外保存`typed_state_pool_identity`；resume同时校验pool与`query_stream_identity`，request count、reference logical ID、group/tile cursor只能在identity一致后恢复。每个request的generator由其不可变identity重建，不持久化执行计划相关generator state，也不能只保存seed后在新registry上重生另一批states。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| phase引用未知group/route或required component永不active | 构造拒绝 |
| asset/evaluator/sampler batch类型、shape、device错 | batch/engine拒绝 |
| loss/gradient非有限、required group零梯度或未更新 | engine失败 |
| objective漏required component output | generic conformance失败 |
| resume identity或phase cursor漂移 | checkpoint恢复拒绝 |
| step 0 calibration成功但metrics为空 | 生成0-record diagnostic review；checkpoint/summary/review仍完整提交 |
| 正step最终提交时metrics为空 | review失败；不得用`allow_empty`发布看似完整的训练产物 |
| tile越界、role/domain不匹配或mip chain不canonical | collection在materialize前拒绝 |
| formal export不是`run_class=formal`、phase不是`complete`，或缺required group finite/nonzero/update coverage | readiness拒绝；不生成正式package/catalog |
| diagnostic evaluator preview没有exact descriptor、step-1端到端evaluator组覆盖或合法phase | readiness拒绝；不得用旧tensor shape兼容冒充可部署状态 |
| validation backend counter出现在下一条training profile | engine profile隔离测试失败；不得据此归因冷启动或长期吞吐 |
| v4 checkpoint 传给新 train resume | v1 loader明确拒绝；只读 evaluation importer 仍可加载 |
| typed-state recipe schema/字段、registry locator或range未知 | plan/session创建前拒绝 |
| resume的typed-state pool identity漂移 | producer恢复拒绝，不恢复任何cursor |
| Metal profile identity、layout identity、slot/token上限、两次asset读取或required route漂移 | model/method config构造拒绝，不创建隐式兼容identity |
| budgeted pilot完成结构选择前请求`compile_program/compile_asset/compile_instance` | fail closed并指出pilot selection边界，不生成伪package；`sample/pdf`数学实现不得随之降级 |
| optimized Metal asset path steps非正、bounded refinement超出`(0,0.5]`或encoder-only携带hidden steps | cook在编码/优化前拒绝 |
| BF16进入FP32 random-access table却使用不同dtype的lerp weight | 实现测试失败；坐标、表和weight统一提升FP32后再插值 |

## 5. Good / Base / Bad Cases

- Good：MaterialX的base-color/roughness/metalness/normal按各自role共享一个collection，encoder与decoder联合训练；phase checkpoint同时冻结asset collection identity与每个required group的梯度/更新覆盖。
- Good：同一Metal graph的base export与多个typed states进入一个group；训练batch先选group再选group-local state，checkpoint冻结完整state pool identity。
- Good：Metal typed edit只重编`MaterialProgramState`，替换texture bundle只改变`AssetState`；相邻mip分别经shared encoder/decoder后在structured state中连续插值。
- Good：Linux single-material pilot从第1步同时执行asset encoder、typed compiler、semantic prepare、evaluator与proposal六组参数；第二phase再以FP32执行部署态QAT。
- Good：fresh run以`--stop-at-step 0`完成calibration，review明确记录0 metric与未完成coverage；随后exact resume从step 1开始训练。
- Base：常量1×1资产仍经同一个tile+halo请求与lease协议，只是其canonical mip chain长度为1。
- Bad：每个batch新建collection使cache永远失效；在tile lease存活时驱逐payload；或仅因loss finite/shape compatible就把非formal、未完成checkpoint标成正式可部署。
- Bad：预先离线编码52个纹理grid后只训练evaluator；把bounded refinement和direct control写入同一个asset identity；或为了mixed precision把angular lookup改成最近邻。

## 6. Tests Required

- unit：YAML resolve/plugin、三种batch、multi-asset tile+halo、phase graph、carry-overlap resume、component正负例、checkpoint v1 与 legacy v4只读；metric loader默认拒绝空文件，只有显式step-0路径接受空rows；
- integration：generic recipe registry扩展Metal train/validation states并验证非默认snapshot IDs不重叠、pool identity可恢复；
- GPU：NVIDIA Python/Slang evaluator与sampler梯度；Metal budgeted finite/nonnegative、typed missing/discrete与asset分离、三路径asset cook、BF16/QAT敏感路径、Python/Slang matched proposal parity，以及Linux真实online两phase smoke/resume/evaluate；
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
# 错：所有final review都要求非空metric，导致合法step-0 calibration已写checkpoint后仍失败。
rows = load_metric_rows(metrics_path, config_sha256=config.sha256)

# 对：只有最终checkpoint仍在step 0时允许空rows；正step保持严格检查。
rows = load_metric_rows(
    metrics_path,
    config_sha256=config.sha256,
    allow_empty=result.checkpoint.global_step == 0,
)
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

## Method-owned diagnostic readiness policy 合同

### 1. Scope / Trigger

修改 `MethodDescriptor`、checkpoint readiness，或让任一新方法支持未完成 checkpoint 的 evaluator-only diagnostic preview 时适用。它防止通用训练层按 `method_key` 猜测方法专属参数组与阶段，并确保 policy 变化进入 descriptor identity。

### 2. Signatures

```text
MethodReadinessPolicy(required_parameter_groups: tuple[str, ...],
                      allowed_phases: tuple[str, ...],
                      minimum_global_step: int = 1)
MethodDescriptor.readiness_policies: Mapping[str, MethodReadinessPolicy]
assess_checkpoint_readiness(checkpoint, descriptor,
                            mode="diagnostic-evaluator")
  -> CheckpointReadiness@1
```

### 3. Contracts

- `diagnostic-evaluator` 是当前唯一允许由方法声明的非正式消费 policy；formal export 与纯 visual diagnostic 的公共规则不由方法覆盖。
- policy 必须写在具体方法 descriptor 中；`required_parameter_groups` 必须是 descriptor 已登记 group 的非空无重复子集，`allowed_phases` 必须是 component active phase 或 `complete` 的非空无重复子集，`minimum_global_step >= 1`。
- readiness 只读取 policy，不比较任何已知 `method_key`、source family 或模型类。没有声明 policy 的方法一律拒绝 evaluator preview，即使 tensor shape、方法名称或 checkpoint phase 看似兼容。
- 非空 policy 进入 `MethodDescriptor.to_dict()` 与 `descriptor_sha256`；空 mapping 为兼容既有无 diagnostic 能力的方法而不序列化该可选字段。
- policy 只放宽显式 evaluator diagnostic 的生命周期入口，不代表 formal、sample/pdf、package 或产品部署 readiness。

### 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| policy key 不是 `diagnostic-evaluator` | descriptor 构造失败 |
| required group 未登记、phase 未被任何 component 声明，或 minimum step 小于1 | descriptor 构造失败 |
| 方法未声明 policy 却请求 evaluator diagnostic | readiness 拒绝并报告 method does not declare |
| checkpoint phase/step 不满足 policy | readiness 拒绝 end-to-end evidence |
| exact descriptor identity 或 required group finite/nonzero/update coverage 不完整 | readiness 拒绝；policy 不覆盖这些公共门 |

### 5. Good / Base / Bad Cases

- Good：budgeted Metal 自己声明 evaluator 所需五组参数以及两个训练 phase；proposal group 不是 evaluator preview 前置，但 formal export 仍要求全部 required component groups。
- Base：NVIDIA descriptor 没有 diagnostic policy，继续只能从 complete formal checkpoint 导出，不因已有 Metal 支持而获得隐式 preview。
- Bad：在 `readiness.py` 写 `if descriptor.method_key == "metal-..."`，或把相同 shape 的旧 checkpoint 当作 exact descriptor。

### 6. Tests Required

- unit：已声明 policy 的合法 phase/step/coverage 通过；缺 policy 即使 method key 相同也拒绝；未知 group、未知 phase、非法 policy key 与零 minimum step 均在 descriptor 构造时失败；空 policy 不改变既有 descriptor 序列化形态。
- static：training/readiness 上层不得出现具体 method key、source family 或模型类分支。

### 7. Wrong vs Correct

```python
# 错：每增加一个方法就修改通用 readiness。
if descriptor.method_key == "metal-budgeted-neural-material":
    required_groups = {"typed_compiler", "directional_evaluator"}

# 对：方法声明，通用层只验证和解释合同。
readiness_policies={
    "diagnostic-evaluator": MethodReadinessPolicy(
        required_parameter_groups=("typed_compiler", "directional_evaluator"),
        allowed_phases=("joint-response-fit", "complete"),
    )
}
```

## Linux DDP reducer、控制组与 checkpoint 提交合同

### 1. Scope / Trigger

修改多GPU launcher、objective forward、phase parameter ownership、metrics reduce、逐rank状态、checkpoint写入或process-group teardown时适用。它防止把“多进程+逐参数`all_reduce`”误称为DDP，也防止data/reference长尾或rank-0 I/O只表现为无来源的NCCL timeout。

### 2. Signatures

```text
DistributedContext.initialize(ExecutionContext) -> DistributedContext
DistributedContext.build_objective(objective, model, phase_name)
  -> (DistributedObjective, DistributedDataParallel)
DistributedContext.reduce_report(loss, metrics, scope) -> averaged scalars
DistributedContext.gather_rank_payload(local_rng_and_query_state) -> rank0 states
DistributedContext.run_rank_zero(label, action) -> rank0 result/status broadcast
```

### 3. Contracts

- Linux多卡仍是一进程一卡；Torch/SlangPy使用rank-local`cuda:0`，Falcor使用launcher冻结的物理adapter。NCCL初始化显式传`device_id=cuda:0`。
- lifecycle先配置phase的`requires_grad`与optimizer ownership，再按phase构造`DistributedObjective`和PyTorch`DistributedDataParallel`。wrapper拥有真实model，forward直接产生objective loss；phase内parameter graph稳定，phase boundary全rank同序重构。
- phase-local objective已由目标机DDP logging和required-group audit证明：phase声明的active参数均参与稳定反向图。因此固定使用`find_unused_parameters=False`、`gradient_as_bucket_view=True`、`static_graph=True`；若未来候选引入条件分支，必须先修正phase ownership或拆phase并新增图稳定性测试，不得静默放宽成兼容模式。禁止重新引入逐parameter gradient `all_reduce`、dummy trigger或每step reducer重建。
- NCCL data group只执行DDP reducer和GPU tensor collectives；辅助Gloo control group执行descriptor核对、小型rank state gather、rank-0 commit status与teardown readiness。model/resume/optimizer setup和phase transition等低频rank-local动作必须在进入下一次DDP collective前汇报任一rank异常；两组在所有rank同序创建和销毁，训练热循环不新增barrier。
- run开始时先核对config、resume cursor、stop target与checkpoint callback presence；scalar loss/metrics按冻结descriptor排序并pack成一次collective，字段或dtype跨rank不一致时在进入NCCL metric collective前由control group全部失败。
- throughput固定记录`steps_per_second`、`local_work_units_per_second`和`global_work_units_per_second`；后者为本次进程已完成的全局work units除以active elapsed，不能写成`steps/s × world_size`。
- checkpoint前所有rank drain并只gather RNG/query cursor；完整model/optimizer CPU snapshot和durable write只在rank 0执行。写入结束后广播success/failure，final artifact commit与hook close后全rank确认teardown readiness。
- `NCLS_DDP_TIMEOUT_SECONDS`只控制NCCL训练collective，默认300秒；`NCLS_DDP_CONTROL_TIMEOUT_SECONDS`只控制低频Gloo控制面，默认1800秒。不得通过提高前者掩盖data/reference长尾或collective desync。
- `NCLS_DDP_DEBUG=1`才启用`TORCH_DISTRIBUTED_DEBUG=DETAIL`与PyTorch 2.11已包含的NCCL trace/dump/desync/timing开关；性能正式run不默认开启。

### 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| phase parameter名称/shape/dtype/requires-grad跨rank不同 | DDP构造前control descriptor检查令所有rank失败 |
| phase声明active参数但某step不参与loss | DDP/static-graph与required-group gate失败；修正phase合同，不启用unused兼容扫描 |
| objective遗漏某rank的metric或scalar类型漂移 | packed metric reduce前失败，不进入次序不同的NCCL collectives |
| rank-0 checkpoint encode/write抛异常 | Gloo广播原error type/message；peer不等待NCCL watchdog |
| 非rank-0尝试构造完整model/optimizer checkpoint | unit/static gate失败 |
| data/reference rank长尾 | stage metrics保留每个rank原值并报告min/mean/max与straggler；reference group的48-bit identity同样逐rank汇总，不能仅归因为reducer |
| success路径有rank未完成hook/session close | teardown readiness报告失败rank，再同序销毁process group |

### 5. Good / Base / Bad Cases

- Good：一个phase只构造一个DDP reducer，25 MiB模型由少量gradient bucket处理；backward interval明确包含reducer，log row同时给出rank max和straggler。
- Good：peer在rank-0写periodic checkpoint时等待Gloo commit status；写入失败后所有rank收到同一失败语义。
- Base：单卡继续经同一个`DistributedObjective`执行，但不创建process group或DDP wrapper，checkpoint保存非envelope本地状态。
- Bad：backward后遍历328个parameter逐个clone/`all_reduce`；或非rank-0在rank-0写summary时提前`destroy_process_group()`。

### 6. Tests Required

- unit/Gloo：两rank objective gradient等于拼接global batch；packed metric均值、descriptor mismatch、rank state gather、rank-0 failure propagation与straggler字段。
- unit：单卡phase/resume与原结果一致；throughput按route work units计算；非rank-0不调用checkpoint codec。
- static：production training hot path无逐parameter gradient `all_reduce`，launcher仍一进程一卡且debug env为显式opt-in。
- Linux/NCCL：跨真实两phase的两卡smoke，检查DDP bucket/unused参数、rank0-only artifact、stop/resume与同序teardown；再做1/2/3/4卡matched scaling和故障注入。

### 7. Wrong vs Correct

```python
# 错：backward已结束，再按parameter串行同步，通信不能与backward重叠。
loss.backward()
for parameter in active:
    dist.all_reduce(parameter.grad)

# 对：phase先冻结parameter ownership，再让真实objective forward进入DDP reducer。
configure_phase(model, phase)
owner = DistributedObjective(plugin.objective, model)
execution = DistributedDataParallel(
    owner,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    static_graph=True,
)
loss = execution(batches, phase_context)
loss.backward()
```

## Canonical Metal budgeted evaluator 与 Linux pilot 合同

### 1. Scope / Trigger

修改当前`metal_budgeted_hybrid_v3`、`metal_budgeted_direct_control_v3`、对应source adapter、asset cook、proposal、训练phase、calibration或Linux pilot配置时适用。它确保质量结论直接来自目标预算内结构，也防止再次在Windows启动online训练、完整validation或runtime baseline。

### 2. Signatures

```text
MetalBudgetedModel.prepare/evaluate_prepared/sample_prepared/pdf_prepared(...)
  -> fixed-layout finite scattering result
MetalBudgetedAssetCooker.cook(tensors, mode, objective, ...)
  -> same-shape RGBA8-SNORM deployment asset
MethodPlugin.lifecycle.initialization_requests(config)
MethodPlugin.lifecycle.initialize_training_state(model, values, metadata)
TrainingPlanResolver.resolve("configs/training/runs/metal-budgeted-{hybrid,direct}-pilot.yaml")
tools/learning/build_metal_linux_handoff.py --output <artifact.json>
```

### 3. Contracts

- public method key固定为`metal`，implementation key为`metal-budgeted-neural-material`。hybrid与direct使用独立profile/correspondence和checkpoint identity；旧full checkpoint只能作为legacy只读evaluation/control，不能resume或经converter进入新方法。
- 主profile hard bound固定为`evaluate ≤20,000 dense MAC/direction`、PreparedState `≤192 B`。当前v2 layout分别为11,392 MAC和160 B；prepare decoder为2,560 MAC；runtime asset读取严格为两次。这些值由layout JSON、Python loader和generated Slang layout共同计算，不把analytic scalar/transcendental隐藏成dense MAC。
- asset输入是detail/context两条RGBA8 SNORM response mip；prepare固定`24→32→32→24`，并把Detail四通道无参数residual到semantic前四维，使高频frame语义具有短梯度路径；evaluator固定`44→64→64→64→6`并消费全部24维semantic state。direct最终线性`f`只消费前三个positive RGB，后三个通道只承担与detached analytic core匹配的训练辅助；hybrid gate为逐RGB`sigmoid`，值域固定`[0,1]`。
- source adapter保留exact MDL locator、typed state、access/frame/resource责任和最多9-slot native patch。确定性access/frame/resource/distribution字段绕过learned guess；`Aluminum_Anodized`所需Beckmann由registry BSDF-data slot确定，不能用材质名启发式。
- proposal固定primary analytic、secondary analytic、uniform full-hemisphere fallback三个mixture component。component位置与distribution enum独立：primary可为GGX或Beckmann，secondary可与primary同为GGX；重复distribution ID合法。每个component折回renderer上半球，PDF累加原方向与z镜像两个preimage，sample后独立PDF必须逐值一致。
- fresh run在第一次model forward之前，按`train-only-reference-rgb-percentiles@1`用固定seed `2026090401`取得16,384条`target_f`，冻结逐通道P50 scale、P95 peak和energy epsilon并写入checkpoint。resume只恢复这些buffer，不重新估计；validation数据不得进入calibration。
- phase固定为`joint-response-fit → deployment-qat-refine`，六个参数组从step 1共同参与appearance/proposal目标。QAT仅在functional forward中对runtime浮点weight做FP16 STE、对asset做RGBA8 SNORM STE；master、optimizer引用、state key与checkpoint schema不变。
- progress与metrics固定登记`loss/optimization_total`、`loss/appearance`、`loss/proposal`和`loss/proposal_weight`。连续密度NLL可为负，这是density超过1时合法的对数密度结果；进度显示不得把它误称complex loss，也不得用绝对值改变梯度。
- DDP validation按一个validation window保留原始batch顺序和每batch的全rank平均metric row，但所有scalar先在device上堆叠，再用一次packed collective聚合并一次性回读host；不得为每个validation batch重复descriptor/metric collective，也不得把整个window压成单一均值而丢失bootstrap单元。validation提交使用phase-local bounded lookahead和同一window boundary，队列不得跨validation/checkpoint/phase边界。
- single-material direct/hybrid是matched diagnostic pair：Tungsten exact locator、固定spatial anchor、paired one-texel UV、zero/one/four-texel footprint quota、uniform/cosine/near-reflection/grazing方向配额、validation seed`2026090402`、1792 joint + 256 QAT、per-rank batch 512、每16 step report和256-batch validation全部一致；唯一结构轴是profile/correspondence。旧`@1` recipe的per-rank batch 64只作为before-profile，高吞吐`@2` v1与full-semantic`@3` v2只作诊断；当前`@4` v3与前三代均不能按step合并。
- 两个pilot在原生Linux通过统一launcher串行执行；当前授权拓扑是物理GPU 5–9上的DDP5，global batch 2560，online session每次最多合并两个logical step生产。它是固定topology的matched pair，不是scaling研究。先step-0 calibration/checkpoint，再各自跑到recoverable step 128并按共同里程碑resume至冻结cap。Windows只执行unit、静态layout和必要的小型纯模型测试；不得执行online reference、pilot、完整validation、完整runtime baseline或旧long。
- pilot observed quality只用于预登记的hybrid/direct选择与failure classification。两者共同失败后也不得自动追加step/seed、扩大模型或启动teacher；任何扩张先回planning。完成结构选择前deployment facet必须fail closed。

### 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| profile MAC/state/read超界或layout identity漂移 | 构造/生成器失败；不放宽hard bound |
| source distribution enum未知，或component位置被当成distribution | compiler/sampler fail closed；合法重复GGX必须保持有效 |
| calibration缺失、样本数/seed/recipe漂移，或resume尝试重估 | training lifecycle拒绝 |
| direct/hybrid source、query、loss、optimizer、schedule、precision或asset mode不matched | handoff/config-pair测试失败 |
| total、appearance、proposal或任一required gradient/update非有限 | engine/readiness失败 |
| sample与independent forward/reverse PDF不一致 | sampler测试失败，不用clamp或fallback掩盖 |
| Windows请求pilot、完整validation或完整runtime workload | 停止执行并转交Linux；不得把Windows卡顿结果登记为实验 |
| pilot cap达到但quality低 | 登记empirical outcome与failure classification，不自动扩张 |

### 5. Good / Base / Bad Cases

- Good：GGX材质proposal distribution是`[0,0,2]`，Beckmann例外是`[1,0,2]`；两者sample/PDF均有限且normalized mixture有效。
- Good：fresh step 0只做一次train-only calibration并保存buffer；从该checkpoint resume至step 128时不再发出calibration request。
- Good：direct与hybrid都用`encoder-only@1`资产输入进行结构选择，避免把某一侧latent refinement收益误算成evaluator收益。
- Base：Windows CPU unit可验证shape、gradient、hash、配置和handoff，不能替代Linux online response、训练收敛或runtime结果。
- Bad：把旧full long当baseline重新跑、在Windows执行256-batch online validation，或因为hybrid质量较低就让它使用bounded refinement而direct仍用encoder-only。

### 6. Tests Required

- unit：layout budget/identity、两read、state packing、Beckmann/重复GGX、finite/nonnegative、half退化/grazing、sample↔PDF、RGB gate、direct auxiliary、asset cook三identity、all-parameter gradient、metric与progress、fresh/resume calibration、YAML pair和handoff无自动followup；method objective测试必须把真实返回mapping交给通用`validate_objective_outputs()`，不能只抽查标准loss而漏掉component contract；
- lightweight GPU：budgeted forward/backward、QAT值与梯度；不得隐式打开reference session；
- Linux online：step-0 calibration、step-128 stop/resume、完整2048 cap、独立256-batch validation和required-group audit；
- selection后：FP16/RGBA8 pack、Python quantized↔Slang exact/random parity、Package@2、typed edit/asset swap；
- static：layout generator `--check`、`compileall`、无upper-layer method/platform分支、`git diff --check`和Falcor clean。

### 7. Wrong vs Correct

```python
# 错：component index决定公式，因而拒绝[GGX, GGX, uniform]。
distribution = torch.tensor([GGX, BECKMANN, UNIFORM])[component]

# 对：从每个component自己的state读取并验证distribution enum。
distribution = round(proposal_state[..., DISTRIBUTION_ID])
selected = gather(distribution, component)
```

```text
# 错：在Windows上执行旧full baseline或完整pilot/validation。
ncls train <removed-legacy-metal-full-run.yaml> ...

# 对：Windows只生成带identity的交接；Linux按direct/hybrid冻结命令串行运行。
python tools/learning/build_metal_linux_handoff.py --output artifacts/.../linux-pilot-handoff.json
```

## 历史 `metal_fused_full_v1` matched sampler 只读合同

> 本节只解释旧package与回归测试的数学身份；对应训练run/recipe已从canonical配置目录移除，不能作为新任务入口。

### 1. Scope / Trigger

仅在维护显式历史control、legacy只读evaluation或复现旧package证据时适用；它不定义canonical `metal`，也不授权恢复旧训练。该合同只拥有旧full matched proposal；source reference sampler仍是独立control，不是训练teacher或产品fallback。

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

## 历史 `metal_fused_full_v1` training/package evidence合同

> 本节只解释既有artifact的字段，不构成恢复旧训练配置或长运行的授权。

### 1. Scope / Trigger

本节只用于解释既有旧full artifact、review与legacy package证据；旧Windows smoke和Linux smoke/long配置不再是product入口，不得据此启动新训练。任何canonical Metal训练均受上一节budgeted合同约束。

### 2. Signatures

```text
fake_quantize_fp16_ste(master: Tensor) -> deployed_value_with_master_gradient
MethodPlugin.objective.compute(..., phase.name="qat-refine")
  batches == {asset: AssetTileBatch, evaluator: EvaluatorBatch, sampler: MethodSamplerBatch}
TrainingEngine.run(resume=None, stop_at_step=N) -> checkpoint at exact global step N
load_evaluation_snapshot(<legacy-metal-full-checkpoint>) -> EvaluationSnapshot
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
