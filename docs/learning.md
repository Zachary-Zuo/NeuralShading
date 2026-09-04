# 训练、验证与部署

## 配置方式

正式入口只接受 `configs/training/runs/*.yaml`。一个 run 组合四部分：默认执行策略、方法、数据定义和训练 recipe；resolver 会展开 source set、严格合并字段并生成不可变的 `ResolvedTrainingPlan@1`。公开方法名为 `nvidia` 或 `metal`，内部版本和实现 hash 自动进入 manifest。

方法通过统一 `MethodPlugin` 提供 model、data、objective、lifecycle、checkpoint 和 deployment 六个 facet。新增方法不应复制 CLI 或训练循环，只需实现自己的数据需求/source adapter、模型/目标及部署编译器，然后注册一个短 key。

## 命令

Windows 单卡：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls train `
  configs/training/runs/nvidia-layer-stack-smoke.yaml `
  --devices 0 `
  --output artifacts/training/nvidia-smoke/checkpoint.pt
```

Linux 单卡：

```bash
bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/metal-linux-smoke.yaml \
  --devices 0 \
  --output artifacts/training/metal-smoke/checkpoint.pt
```

当 `--devices` 只有一个序号时直接单卡执行；Linux 传多个序号时自动启动一个 torchrun/NCCL 作业，Windows 多卡明确拒绝：

```bash
bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/metal-linux-smoke.yaml \
  --devices 2,3,4 \
  --output artifacts/training/metal-ddp/checkpoint.pt
```

停止、恢复、数值验证和正式导出：

```bash
bash scripts/run_falcor_python.sh -m ncls train <run.yaml> \
  --devices 0 --output <checkpoint.pt> --stop-at-step <N>
bash scripts/run_falcor_python.sh -m ncls train <run.yaml> \
  --devices 0 --output <checkpoint.pt> --resume <checkpoint.stepXXXXXXXX.pt>
bash scripts/run_falcor_python.sh -m ncls validate <checkpoint.pt> --batches 8 --device 0
bash scripts/run_falcor_python.sh -m ncls export <checkpoint.pt> <package-dir> --material-index 0
```

新 checkpoint 内嵌 resolved plan，因此 `validate` 不再要求再次传 config。新训练 resume 只接受 `TrainingCheckpoint@1`；旧 v4 checkpoint 可只读地用于 `validate`、满足原 readiness 时的 export，以及 visual eval，不能继续训练。

## 数据调度与 GPU 利用率

`execution` 段统一配置：

- `num_workers`：只并行 CPU/host decode、读取和预处理；worker 不持有 CUDA/Falcor。
- `host_prefetch`、`ready_batches`：限制队列深度并提供 backpressure，避免无限占用内存。
- `reference_batch_steps`、`reference_inflight`：在 backend capability 允许时控制 reference packed dispatch/在途数。
- `transfer_streams`：控制 cache miss 的异步 H2D 通道；常驻或已命中的 tensor 不重复交换。
- `residency.budget_mib`：GPU 资源缓存的字节上限；lease 中对象不能被 LRU 驱逐。

同步基线始终是 `num_workers=0`、`reference_batch_steps=1`。调大这些值不是无条件更快：只有可复制 host 阶段才交给 worker；GPU/reference 并发必须由真实 capability、stream fence 和 lease 保证。Linux 的最终数值应通过 stage trace 分开报告 host、transfer、reference、model、barrier、cache 与峰值显存，不能仅用平均 GPU utilization 判断。

## 输出与 hooks

rank 0 写出：

- `checkpoint.pt` 与 `.sha256`：新 checkpoint；
- `checkpoint.stepXXXXXXXX.pt`：周期/phase boundary checkpoint；
- `checkpoint.metrics.jsonl`：逐 cadence 指标和 data/reference profile；
- `checkpoint.summary.json`、`checkpoint.review.json`：运行与首轮审阅；
- `checkpoint.tensorboard/`：TensorBoard event；
- `checkpoint.visual-eval/`：异步 request/status spool（启用时）。

TensorBoard hook 使用稳定 tag/global step，队列有界且只由 rank 0 写。visual eval 不属于验证集指标：它按固定 step 选择可恢复的随机 probe，比较 reference 和 neural 渲染。默认 reference 为 1024 spp path tracing，neural 为 deterministic deferred（`neural_mode: deferred`、`neural_spp: 0`）；一次当前 Windows 实测约 12.429 秒。把 neural 改成 path tracing 是显式手工诊断，不应加入常规 cadence。

Windows worker 与迟到结果收集：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls eval worker `
  <spool-dir> <artifact-root> --max-jobs 1
.\scripts\run_falcor_python.ps1 -m ncls eval collect `
  <spool-dir> <artifact-root> <tensorboard-dir>
```

visual eval 失败只记录 hook/status，不改变训练 checkpoint。正式 export 仍要求 exact method identity、formal run、complete phase 与所需 parameter group 的 finite/nonzero-gradient/actual-update coverage；短 smoke 只能作为诊断证据。
