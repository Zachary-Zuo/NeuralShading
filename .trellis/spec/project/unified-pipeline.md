# 统一 pipeline 可执行合同

## 1. Scope / Trigger

修改 YAML training plan、source/reference、online data、method plugin、engine/checkpoint、package 或 viewer 时适用。新增 method/source 不得复制 CLI、训练循环、checkpoint、平台分支或磁盘 batch 路径。

## 2. Signatures

```text
TrainingPlanResolver(project_root).resolve(run_yaml, devices=None)
  -> ResolvedTrainingPlan@1
get_method_plugin("nvidia" | "metal") -> MethodPlugin
DataExecutionPlan.build(...) -> DataExecutionPlan@1
OnlineTrainingProducer(plugin, runtime_config, execution_context, data_execution_plan)
OnlineDataSession.submit_step(named_routes, boundary_id=...) -> logical_id
OnlineDataSession.acquire_step(logical_id) -> OnlineStepBatch
OnlineStepBatch.release()
TrainingEngine(plugin, data_session, runtime_config, ...).run(resume, stop_at_step)
  -> TrainingRunResult
save/load_training_checkpoint_v1(...) -> TrainingCheckpoint@1
load_evaluation_snapshot(v1_or_legacy_v4) -> EvaluationSnapshot
MethodPlugin.deployment.compile_program/asset/instance(...)
  -> deployment payloads
write_scattering_package(...) -> ScatteringPackageManifest@2
```

公开命令：

```text
ncls train <run.yaml> [--devices 0|0,1,...] [--output checkpoint.pt]
           [--resume checkpoint.pt] [--stop-at-step N]
ncls validate <checkpoint.pt> [--batches N] [--device N]
ncls export <checkpoint.pt> <package-dir> [--material-index N]
ncls eval worker|collect ...
```

## 3. Contracts

- run YAML 只组合 `base/method/data/recipe` 四类 fragment；严格 loader 拒绝重复/未知字段、循环依赖和 value category 漂移。resolved plan、展开文件 hash、method descriptor/facet identity 全部进入 plan identity。
- 公开 key 使用 lower-kebab 且不含 `@`；当前 method 为 `nvidia`、`metal`。版本只进入 descriptor/implementation/plan hash，不进入用户入口名称。
- `MethodPlugin` 必须同时提供 model/data/objective/lifecycle/checkpoint/deployment 六个 facet。公共 engine/data/CLI 只依赖 facet，不通过 implementation key 找 `MethodDefinition`，不按 method/source family 建专用分支。
- `SourceSnapshot` 是 source 唯一真相；各 family 保持原生 locator、资源、编辑与 reference 语义。`ReferenceExecutionPlan@1` 拥有 grouping/global-local index；backend 只执行 plan，platform/Falcor 只由 capability/launcher 拥有。
- 正式训练 batch 只在 GPU online 产生，不保存/读取 corpus。唯一`PipelineOnlineDataSession`拥有worker、bounded queue、residency、reference scheduling、rank partition、cursor、drain和lease边界；同步调试只改变capacity/pack配置，不存在旧session或`next_batch()`兼容入口。详见`../data/online-pipeline.md`。
- `TrainingEngine` 只解释通用 phase graph，固定 phase/step/validation/checkpoint/cleanup 生命周期；method 只能通过 facet 提供 objective 和 transition，不能注册另一条 runner。
- Linux多卡由phase-local `DistributedObjective`进入PyTorch DDP reducer；lifecycle先冻结active parameter，phase boundary再同序重构wrapper。禁止在backward后逐parameter手工`all_reduce`。NCCL data group与Gloo control group分责，后者只处理descriptor、小型rank state、checkpoint commit和teardown状态。
- checkpoint hook 在 data session drain 后把内部 engine 状态封装为 `TrainingCheckpoint@1`。新 resume 只接受 v1 且严格匹配 resolved plan/data/method facet identity。
- legacy v4 parser 隔离在只读 importer，只输出 `EvaluationSnapshot`。它可用于 validate、满足原 readiness 的 export/visual eval，但不能 resume；没有旧 JSON reader、converter、alias 或 fallback。
- TensorBoard 与 visual eval 通过 typed event/hook 接入且只由 rank 0 写。visual eval 独立 probe RNG/spool 可迟到，不阻断 optimizer；默认 reference 1024 spp path tracing、neural deterministic deferred。
- 单 device 在 Windows/Linux 直接执行；多个 device 只在 Linux/NCCL 自动 launch，Windows 在创建 Torch/Falcor 前拒绝。engine、producer、method 不读 OS 或物理 GPU 环境。
- `ScatteringPackage@2` 独立计算 program/asset/instance/package identity；viewer 验证 section、typed resource、source snapshot 和 parity 后原子绑定 slot。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| YAML unknown/duplicate/cycle/incompatible component | resolver 在构造 GPU runtime 前拒绝 |
| plugin 缺 facet、重复 key、data requirement 不闭合 | registry/plan 构造拒绝 |
| source/reference/data/query/plan identity 漂移 | session/resume/validate 拒绝 |
| loss/gradient/update 非有限或 required coverage 不完整 | engine/readiness 拒绝 |
| v4 用于 `ncls train --resume` | 明确拒绝，只允许 evaluation importer |
| Windows 请求多个 device | preflight fail closed，不回退单卡/Gloo |
| hook queue 满、worker crash 或迟到 | 状态可见；diagnostic hook 不改变 checkpoint 成功语义 |
| package URI/hash/section/instance binding 错 | loader/viewer 拒绝并保留旧 slot |
| editable compiler/candidate resource 创建失败 | 丢弃 candidate，active raw/compiled state 不变 |

## 5. Good / Base / Bad Cases

- Good：一个YAML把`method=metal`、`data=mdl-metal-budgeted-tungsten`与budgeted pilot recipe组合；同一plugin/data/engine合同可在支持的执行环境运行，当前single-material pair通过Linux统一launcher在固定GPU 5–9 DDP5拓扑串行执行。它不构成scaling研究，Windows仍不运行online pilot。
- Good：多个 native source state 编入 grouped plan，由同一 online session 产生 typed batch；method deployment facet独立输出三段 package，viewer 只解释公共 schema。
- Base：单 source、无纹理、单 phase、单 GPU 仍走同一 resolver/plugin/session/engine/checkpoint，不新增简化入口。
- Bad：runner 按 source family 选择 producer；exporter 识别 method 名补 artifact；新增 `configs/learning/*.json` 或 `ncls learn` fallback；用 `@版本` 作为公开 model 名。

## 6. Tests Required

- unit：YAML resolve/identity、plugin registry/facets、data plan/session、phase/resume/events/hooks、checkpoint v1 与 legacy v4 read-only；
- unit：package/editor/tamper，visual spool/worker/collector，launcher 单卡/Linux 多卡/Windows 拒绝；
- GPU：五 source 的统一 backend、NVIDIA/Metal objective、resident sampling、Slang/package parity；
- integration：新 checkpoint stop/resume/validate/export；legacy checkpoint validate/export；Windows 1024 spp visual diagnostic；
- Linux：NCCL 两卡smoke、rank device mapping、DDP bucket/gradient parity、rank0-only完整checkpoint、commit failure propagation和跨rank stage trace；
- static：无旧 JSON config/CLI/runner reader、无 upper-layer OS 分支、无磁盘 batch；Falcor overlay 构建后工作树干净。

## 7. Wrong vs Correct

```python
# 错：公开名携带实现版本，并由CLI选择具体definition。
definition = get_method("metal-fused-neural-material@3")
runner = MetalRunner(definition, config_json)

# 对：短key解析显式plugin，公共engine只依赖facet。
plan = TrainingPlanResolver(root).resolve("configs/training/runs/metal-budgeted-hybrid-pilot.yaml")
plugin = get_method_plugin(plan.selection.method)
TrainingEngine(plugin, data_session, plan.to_runtime_config()).run()
```

```python
# 错：用legacy checkpoint继续新训练。
engine.run(resume=load_checkpoint_v4(path))

# 对：v4只能变成只读evaluation snapshot；新resume严格读取v1。
evaluation = load_evaluation_snapshot(path)  # v1 or legacy v4
resume = load_training_checkpoint_v1(path)   # v1 only
```

```cpp
// 错：先修改active binding，compiler失败后raw/compiled已经分裂。
upload(active.buffers);
compile(active.buffers);

// 对：在candidate中完成上传与编译，成功后一次性替换。
auto candidate = active.buffers;
upload(candidate);
compile(candidate);
active.buffers = std::move(candidate);
```
