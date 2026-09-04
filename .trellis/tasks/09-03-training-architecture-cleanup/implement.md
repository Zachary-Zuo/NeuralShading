# 训练架构统一与跨平台清理：实施计划

## 当前执行状态（2026-09-04）

- [x] Phase 0：冻结旧调用链、Linux long 历史 profile 与 VRFrameGeneration 对照。
- [x] Phase 1：组合 YAML、短 key、严格 resolver 与 resolved plan identity。
- [x] Phase 2：显式 `MethodPlugin` 六 facet、`DataExecutionPlan` 与 `OnlineDataSession` 边界。
- [x] Phase 3：唯一 `TrainingEngine`、typed event、checkpoint v1 与 drain-safe stop/resume。
- [x] Phase 4：统一 `--devices`；单卡直跑、Linux/NCCL 多卡自动 launch、Windows 多卡 fail closed；旧 multi shell 已降为薄转发。
- [ ] Phase 5：host worker、Metal GPU residency/hot path、trace 与 reference scheduler primitive 已实现并通过 Windows unit/GPU；生产配置当前仍以 `reference_batch_steps=1` 为安全基线，GPU batch ring、packed reference 与 Linux before/after/overlap 证据必须在目标 Linux 完成后才能关闭本阶段。
- [x] Phase 6：rank-0 TensorBoard、异步 visual spool/worker/collector；真实 Windows reference 1024 spp + neural deferred capture 完成。
- [x] Phase 7：legacy v4 仅只读导入 evaluation snapshot；新 resume 拒绝 v4。
- [ ] Phase 8：旧 JSON、`ncls learn`、runner、全局 method lookup 和生成脚本已删除，viewer consumer 与稳定文档/spec 已迁移；待 Phase 5 Linux gate 后执行最终归档。

任务保持 `in_progress`。不能用 Windows RTX 4090 结果替代 AC11/AC11b 要求的原生 Linux/Vulkan/NCCL stage trace。

## 1. 执行原则

- 本任务以单个 umbrella task 分阶段执行。配置、data plane、engine、method 和 hook 共享同一 identity/lifecycle，强行拆成并行 child 会在迁移期制造多套临时合同；每个阶段仍必须拥有独立测试、rollback point 和可审查结果。
- inline 模式不维护 `implement.jsonl/check.jsonl`；进入 Phase 2 后先运行 `trellis-before-dev` 重新加载 project/core/data/learning 规范。
- 每个阶段只在前一阶段质量门通过后继续。旧路径仅作为临时 characterization oracle，不能成为最终 fallback。
- Windows 当前为完整开发环境，可执行 unit/GPU/Falcor/Slang 与 Windows viewer gate；Linux/NCCL、多卡与性能 profile 必须在真实 Linux 目标机补齐，不能用 Windows 结果代替。

## 2. Phase 0：冻结基线与 characterization

### 改动

- 在任务 research 中冻结当前 CLI/config/producer/runner/checkpoint 调用链与长训 profile 摘要。
- 为当前 NVIDIA/Metal typed routes、phase cursor、query RNG、lease、checkpoint v4 readiness 和 package export 增加最小 characterization fixtures。
- 记录当前 Metal `sample_local_patches()` 的 host readback/H2D、reference barrier、allocation 与 queue 行为，作为 before trace。

### 重点文件

- `tests/unit/test_online_training_producer.py`
- `tests/unit/test_training_runner_phase_graph.py`
- `tests/unit/test_training_checkpoint.py`
- `tests/unit/test_metal_training_handoff.py`
- `.trellis/tasks/09-03-training-architecture-cleanup/research/`

### Gate / rollback

- 不改产品行为；characterization 不稳定时先修正测试观察面。
- 保留现有 `artifacts/metal-linux-training/long/checkpoint.metrics.jsonl` 为历史 observed baseline，不把数值写成 hard gate。

## 3. Phase 1：短 key、YAML composition 与 resolved plan

### 改动

- 新增严格 YAML loader、component fragment registry、确定性 merge 与 `ResolvedTrainingPlan`。
- 添加 `PyYAML` 到 `environment.yml` 和 `pyproject.toml`。
- 建立公开 key `nvidia` / `metal` 与 data/recipe key；版本信息仅进入结构化 manifest。
- 将现有 Windows smoke、Linux smoke/long 和 NVIDIA 配置迁为组合 YAML；full source locator 从 versioned registry/source-set resolver 展开，不再内联 692 项。
- 暂时提供新 plan 到旧 runtime 的内部 migration adapter，用于下一阶段开发；它不是用户兼容入口。

### 新增/影响文件

- `src/ncls/learning/training/config.py` 或拆分后的 `config/{schema,loader,resolver}.py`
- `configs/training/{base,methods,data,recipes,runs}/**/*.yaml`
- `tools/learning/build_metal_training_configs.py`（转为 parity/check 或最终删除）
- `tests/unit/test_training_yaml.py`
- `tests/unit/test_training_plan.py`

### Gate / rollback

- 同一 YAML 两次 resolve 得到相同 canonical plan/hash；unknown/merge conflict/cycle/incompatible component fail closed。
- NVIDIA 与 Metal representative plan 的 source、phase、route、model shape 和 readiness 语义与旧 config characterization 对齐。
- rollback 为保留新 loader 但不切 CLI；不得让新旧 config 同时成为长期入口。

## 4. Phase 2：MethodPlugin facet 与 data contracts

### 改动

- 定义 `MethodPlugin`、model/data/objective/lifecycle/checkpoint/deployment facet 协议与显式 registry。
- 把 typed route/batch、`DataRequirement`、`TrainingDataDefinition`、`DataExecutionPlan` 移到 `ncls.data`，消除 data 对 concrete method 的 import。
- 用 adapter 包装现有 NVIDIA、Metal definition；逐步将 `source_adapters.py` 私有二元映射迁入各 method data facet。
- 建立 `OnlineDataSession` 同步实现，`num_workers=0`、`reference_batch_steps=1` 时与旧 producer 对齐。

### 新增/影响文件

- `src/ncls/data/{contracts,plan,session}.py`
- `src/ncls/learning/methods/contracts.py`
- `src/ncls/learning/methods/registry.py`
- `src/ncls/learning/{batches,producer,source_adapters,source_adaptation}.py`
- `tests/unit/test_method_plugin.py`
- `tests/unit/test_data_plan.py`
- `tests/unit/test_online_data_session.py`

### Gate / rollback

- registry 缺 facet、重复 key、unknown config、descriptor/data dependency 不闭合时构造失败。
- runner/engine-facing test 只依赖协议；static scan 不出现 method/source family 分支。
- 同步 data session 逐 step batch type/shape/device/provenance/cursor 与 characterization 一致。

## 5. Phase 3：固定 TrainingEngine、事件与新 checkpoint

### 改动

- 将唯一 lifecycle 从旧 runner 拆为 `TrainingEngine` + optimizer/precision/distribution/checkpoint services。
- 建立 typed event bus 和 progress、JSONL metrics、checkpoint hook；engine 不再直接写文件。
- 定义新 checkpoint schema，保存 resolved plan、逐 rank RNG/data cursor、phase optimization 与 hook cursor；实现 drain-safe resume。
- 数值命令改为 `ncls validate`；export 继续使用 method deployment facet。

### 新增/影响文件

- `src/ncls/learning/training/{engine,events,checkpoint,distribution}.py`
- `src/ncls/learning/training/hooks/{progress,metrics,checkpoint}.py`
- `src/ncls/learning/training/{runner,review,readiness}.py`
- `tests/unit/test_training_engine.py`
- `tests/unit/test_training_events.py`
- `tests/unit/test_training_checkpoint_new.py`
- `tests/unit/test_training_resume.py`

### Gate / rollback

- uninterrupted 与 stop/resume 的 model/optimizer/scheduler/RNG/data cursor 相同；phase/validation/checkpoint 不跨未释放 lease。
- hook 顺序、fatal/diagnostic 失败、interrupt cleanup 与 DDP reduce policy 有测试。
- rollback 为新 engine 使用同步 data session；旧 runner 尚未删除，但不能由新 CLI 调用。

## 6. Phase 4：统一 launcher 与单卡/多卡

### 改动

- 新增轻量 bootstrap：设备数决定 single/torchrun，且在导入/构造 GPU runtime 前完成 capability preflight。
- 实现 `ExecutionContext`；移除 producer/method/engine 对环境变量、OS 和物理 GPU 的直接读取。
- Linux 多 device 自动启动 NCCL DDP；Windows 多 device fail closed；单 device 两平台使用相同 plan/data/method/engine。
- shell/PowerShell launcher 只保留环境/Falcor build 注入。

### 新增/影响文件

- `src/ncls/cli.py`
- `src/ncls/learning/training/launch.py`
- `scripts/run_falcor_python.{ps1,sh}`
- `scripts/run_falcor_python_multi.sh`（最终删除或降为无产品逻辑的薄转发）
- `tests/unit/test_training_launcher.py`
- `tests/unit/test_execution_context.py`
- `tests/integration/test_distributed_training.py`

### Gate / rollback

- 一个 GPU 不启动 distributed；两个 GPU 的 Linux smoke 每 rank local device/seed/shard 正确且只 rank 0 写 checkpoint/TensorBoard。
- Windows 多 GPU 在 Falcor/Torch device 创建前报 capability error，不回退单卡。
- launcher 失败时所有 worker 返回非零且 rank 0 不发布成功 checkpoint。

## 7. Phase 5：通用 pipeline、GPU residency 与 Linux 性能

### 7.1 Host pipeline

- 实现 bounded host worker pool、persistent workers、start-method adapter、reorder buffer、backpressure、health/error propagation、drain/cancel。
- `num_workers=0/1/N`、worker crash、queue full、phase/checkpoint boundary 和 deterministic resume 全覆盖。

### 7.2 GPU residency 与 Metal hot path

- 实现按 bytes 计量的 `GpuResidencyManager`、lease、LRU 和 trace。
- 将 Metal typed metadata 常驻 GPU；把 `sample_local_patches()` 改为 GPU-native request/sampling，cache miss 才执行 host read/pinned H2D。
- 消除逐 step `asset_index/uv/mip` GPU→CPU readback；保持 transfer、address、canonical mip 和 normal renormalization。
- 预分配 route/batch arena，复用 reference input/output/compaction buffers，减少 allocator 和 D2D 临时复制。

### 7.3 Reference scheduling

- 为 reference backend 增加 concurrency capability，而不是在 pipeline 判断 Windows/Linux。
- global-sync 模式实现 logical-step packed dispatch、切分、invalid top-up 与 ready ring；stream-fence 模式实现受 event/lease 保护的双/三缓冲。
- stage trace 覆盖 cache、bytes、barrier、queue、reference 与 model timing。

### 新增/影响文件

- `src/ncls/data/{pipeline,residency,tracing}.py`
- `src/ncls/learning/mdl_metal_assets.py`（最终迁到 Metal data facet）
- `src/ncls/references/{backend,query}.py`
- `tests/unit/test_data_pipeline.py`
- `tests/unit/test_gpu_residency.py`
- `tests/unit/test_reference_scheduler.py`
- `tests/gpu/test_metal_gpu_asset_sampling.py`
- `tests/gpu/test_reference_packed_dispatch.py`

### Gate / rollback

- `num_workers=0`、`reference_batch_steps=1` 始终可作为显式同步基线。
- Metal hot path trace 中 request metadata host-readback bytes 为 0；cache/ready ring 峰值不超过配置预算，活跃 lease 不被驱逐。
- packed 与基线按 logical step 比较 query identity、valid/top-up、target 和 resume；不跨 group/phase/validation/checkpoint。
- Linux 同一 frozen smoke 做 before/after stage trace；证明至少 host/data stage 与 model 有实际 overlap，报告 GPU activity、吞吐、显存与 barrier/step。observed 数值不决定任务语义完成。

## 8. Phase 6：TensorBoard 与异步 visual eval

### 改动

- 实现 rank-0 TensorBoard hook、稳定 tags/global step、有界 writer/error channel 和 resume。
- 定义 visual request/result/status、独立 probe RNG、文件 spool、atomic claim/publish 和 collector。
- 实现 `ncls eval` 与 Windows worker；复用现有 viewer replay/headless 1024 spp capture，不新增 Linux renderer。
- 支持训练结束后补收迟到结果；保存 reference/neural/difference/PNG/EXR/provenance 并写 TensorBoard image/status。

### 新增/影响文件

- `src/ncls/learning/training/hooks/tensorboard.py`
- `src/ncls/learning/training/hooks/visual_eval.py`
- `src/ncls/visual_eval/{contracts,spool,worker}.py`
- `src/ncls/cli.py`
- `scripts/benchmark_viewer.ps1` 或新增薄 worker launcher
- `tests/unit/test_tensorboard_hook.py`
- `tests/unit/test_visual_eval_spool.py`
- `tests/integration/test_visual_eval_worker.py`

### Gate / rollback

- smoke/resume event file 可读取，关键 scalar tag/global step 正确，DDP 只有 rank 0 writer。
- request 重复领取、worker crash、超时、过期、queue capacity 和迟到 collect 都有确定状态。
- 一次真实 Windows worker capture 令 reference 达 1024 spp、neural 默认走 deterministic deferred，输出两 slot、difference 和 identity 完整 manifest；manifest 显式记录两侧 mode、target/actual spp 与 `training-diagnostic`，失败不改变训练 checkpoint。低 spp neural path tracing 和双 1024 spp 只作为手工深度检查，不是 cadence gate。

## 9. Phase 7：legacy v4 只读导入

### 改动

- 将当前 v4 parser/validation 隔离到 `LegacyCheckpointV4Importer`。
- importer 只输出 evaluation snapshot，接入 `validate`、readiness-preserving export 与 visual eval。
- 新 train resume 显式拒绝 v4；不注册旧 method key alias，不实现旧 JSON config reader/converter。

### 新增/影响文件

- `src/ncls/learning/training/legacy_checkpoint.py`
- `src/ncls/learning/training/readiness.py`
- `src/ncls/learning/export/`
- `tests/unit/test_legacy_checkpoint_v4.py`
- `tests/integration/test_legacy_checkpoint_evaluation.py`

### Gate / rollback

- 正确 v4 + sidecar 能验证并按原 readiness 导出；tamper、unknown descriptor、shape-only compatibility 全部拒绝。
- `ncls train --resume old-v4.pt` 明确拒绝；static dependency test 证明 engine/data pipeline 不 import legacy module。

## 10. Phase 8：迁移方法、删除旧架构与更新规范

### 改动

- 完成 NVIDIA/Metal 文件拆分和公共模块提取，删除临时 facet adapter。
- 删除旧 JSON configs、`ncls learn`、旧 producer/runner/config reader、method/source 私有映射、converter/alias/fallback 和无用生成脚本。
- 更新 `docs/architecture.md`、`docs/learning.md`、`docs/metal_linux_training.md`、repository policy 与 `.trellis/spec/{project,data,learning}/`。
- 递归检查 viewer/bundle 不依赖 PyTorch 或训练实现，source/reference/native edit 与 package contract 未退化。

### Gate / rollback

- 全仓 static scan 不存在旧正式入口、旧 config schema reader、method 专用 runner/CLI、散落 OS 分支或离线 batch/corpus。
- 所有新入口文档命令可执行；generated/config parity 和 package/viewer consumer 更新完成。
- 此阶段删除是最终门；若调用方未迁完，不执行对应删除。

## 11. 验证命令矩阵

所有 Python 命令使用唯一 Conda 环境。

### Windows/unit/static

```powershell
conda run -n neural-shading python -m pytest tests/unit/test_training_yaml.py tests/unit/test_training_plan.py
conda run -n neural-shading python -m pytest tests/unit/test_method_plugin.py tests/unit/test_data_plan.py tests/unit/test_online_data_session.py
conda run -n neural-shading python -m pytest tests/unit/test_training_engine.py tests/unit/test_training_events.py tests/unit/test_training_checkpoint_new.py tests/unit/test_training_resume.py
conda run -n neural-shading python -m pytest tests/unit/test_training_launcher.py tests/unit/test_execution_context.py
conda run -n neural-shading python -m pytest tests/unit/test_data_pipeline.py tests/unit/test_gpu_residency.py tests/unit/test_reference_scheduler.py
conda run -n neural-shading python -m pytest tests/unit/test_tensorboard_hook.py tests/unit/test_visual_eval_spool.py tests/unit/test_legacy_checkpoint_v4.py
conda run -n neural-shading python -m pytest tests/unit
conda run -n neural-shading python -m compileall -q src tests tools
git diff --check
```

### Windows GPU/Falcor/Slang/viewer

```powershell
.\scripts\run_falcor_python.ps1 -m pytest tests/gpu/test_metal_gpu_asset_sampling.py tests/gpu/test_reference_packed_dispatch.py
.\scripts\run_falcor_python.ps1 -m pytest tests/gpu
.\scripts\run_falcor_python.ps1 -m pytest tests/integration/test_legacy_checkpoint_evaluation.py tests/integration/test_visual_eval_worker.py
.\scripts\build_viewer.ps1 -Configuration Release
```

viewer 验证后确认 `external/Falcor` 回到锁定提交且工作树干净。

### Linux 单卡/多卡与性能

```bash
bash scripts/run_falcor_python.sh -m ncls train configs/training/runs/metal-linux-smoke.yaml --devices 0
bash scripts/run_falcor_python.sh -m ncls train configs/training/runs/metal-linux-smoke.yaml --devices 0,1
bash scripts/run_falcor_python.sh -m pytest tests/integration/test_distributed_training.py
```

随后以冻结的 representative config 分别运行同步基线与优化 pipeline，输出到不同 `artifacts/` run identity；采集 stage trace、GPU activity、吞吐、barrier、cache/queue 与峰值显存，写入任务 research 报告。不得覆盖历史 run，也不得根据结果事后改变 hard gate。

## 12. 启动前检查

- [ ] `prd.md`、`design.md`、`implement.md` 已经用户最终审阅并在后续消息明确批准。
- [ ] 运行 `trellis-before-dev`，读取 project/core/data/learning pre-development checklist。
- [ ] 检查 dirty worktree，登记与本任务重叠的用户改动；不回退无关文件。
- [ ] 确认 Windows 开发状态；Linux gate 的机器/命令/产物目录可用。
- [ ] 确认当前旧 long training 是否仍运行；删除旧 train 入口前先保存其最终 checkpoint/sidecar，不把进程中间状态当作迁移输入。
- [ ] Phase 0 characterization 全绿后才开始架构切换。
