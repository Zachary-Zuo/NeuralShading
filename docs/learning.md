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
  configs/training/runs/metal-budgeted-hybrid-pilot.yaml \
  --devices 0 \
  --output artifacts/metal-budgeted-pilot/hybrid/checkpoint.pt
```

当前 Metal direct/hybrid single-material pilot 只在原生 Linux 单 GPU串行执行；Windows 不运行online reference、完整validation、pilot或runtime baseline。精确的step-0 calibration、step-128恢复点和matched direct命令见[Metal Linux pilot](metal_linux_training.md)。

当 `--devices` 只有一个序号时直接单卡执行；Linux 传多个序号时自动启动一个 torchrun/NCCL 作业，Windows 多卡明确拒绝：

```bash
bash scripts/run_falcor_python.sh -m ncls train \
  <ddp-run.yaml> \
  --devices 2,3,4 --output <checkpoint.pt>
```

Linux多卡的objective按phase包装为真实PyTorch `DistributedDataParallel`：lifecycle先冻结本phase的active parameter，再构造一个`static_graph` reducer；phase合同保证声明为active的参数每步都参与loss，因此不启用unused-parameter兼容扫描。gradient bucket在backward中同步，不在backward后逐parameter执行`all_reduce`。phase切换时所有rank同序重构wrapper；model/resume/optimizer setup与phase transition的rank-local异常会先经control group汇总，不让正常rank独自进入下一次DDP collective。NCCL data group负责梯度和GPU scalar collective，辅助Gloo control group只负责descriptor核对、逐rank RNG/query cursor、checkpoint commit状态与有序退出，训练热循环不新增barrier。

诊断运行可显式设置：

```bash
NCLS_DDP_DEBUG=1 \
NCLS_DDP_TIMEOUT_SECONDS=300 \
NCLS_DDP_CONTROL_TIMEOUT_SECONDS=1800 \
bash scripts/run_falcor_python.sh -m ncls train <run.yaml> \
  --devices 0,1 --output <checkpoint.pt>
```

`NCLS_DDP_DEBUG=1`会启用PyTorch/NCCL detail、flight recorder、timeout dump、desync和timing信息，会影响性能，只用于诊断。两个timeout分属训练数据面和低频控制面；不得靠增大NCCL训练timeout掩盖reference/data长尾。

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

`execution` 段统一声明目标数据调度策略：

- `num_workers`：只并行 CPU/host decode、读取和预处理；worker 不持有 CUDA/Falcor。
- `host_prefetch`、`ready_batches`：限制队列深度并提供 backpressure，避免无限占用内存。
- `reference_batch_steps`、`reference_inflight`：在 backend capability 允许时控制 reference packed dispatch/在途数。
- `transfer_streams`：控制 cache miss 的异步 H2D 通道；常驻或已命中的 tensor 不重复交换。
- `residency.budget_mib`：GPU 资源缓存的字节上限；lease 中对象不能被 LRU 驱逐。

生产入口统一使用`PipelineOnlineDataSession`：engine按step提交named routes，session以有界pending/ready ring严格按logical ID交付；`ready_batches=1`和`reference_batch_steps=1`仍通过同一实现表达同步调试基线，不存在另一套兼容session。producer先为每个logical request冻结独立RNG，再把同group的连续evaluator请求合并成一次reference dispatch，按原ID切分target；改变pack或队列深度不会改变训练样本。只有可复制host阶段交给worker并可提前提交，CUDA residency、Falcor和lease仍由rank主进程拥有。`global-sync` Vulkan路径只获得host重叠与packed dispatch收益，`reference_inflight`不能视为同卡真实异步；只有backend明确提供stream fence时才允许增加inflight。Linux数值通过stage trace分开报告host、transfer、reference、model、barrier、cache与峰值显存，不能仅用平均GPU utilization判断。

## 输出与 hooks

rank 0 写出：

- `checkpoint.pt` 与 `.sha256`：新 checkpoint；
- `checkpoint.stepXXXXXXXX.pt`：周期/phase boundary checkpoint；
- `checkpoint.metrics.jsonl`：逐 cadence 指标和 data/reference profile；
- `checkpoint.summary.json`、`checkpoint.review.json`：运行与首轮审阅；
- `checkpoint.tensorboard/`：TensorBoard event；
- `checkpoint.visual-eval/`：异步 request/status spool（启用时）。

DDP checkpoint先在所有rank drain，只把小型RNG/data cursor收集到rank 0；完整model/optimizer CPU snapshot和文件写入只在rank 0执行。写入结果随后经control group广播，final artifact和hook关闭完成后所有rank再共同退出。metrics同时记录正确的`steps_per_second`、`local_work_units_per_second`、`global_work_units_per_second`，以及关键stage的逐rank原值、min/mean/max和straggler rank；reference group的48-bit identity也逐rank记录，`profile/backward_reducer_gpu_seconds`包含backward与reducer的合并区间。

TensorBoard hook 使用稳定 tag/global step，队列有界且只由 rank 0 写。visual eval 不属于验证集指标：它按固定 step 选择可恢复的随机 probe，比较 reference 和 neural 渲染。默认 reference 为 1024 spp path tracing，neural 为 deterministic deferred（`neural_mode: deferred`、`neural_spp: 0`）；一次当前 Windows 实测约 12.429 秒。把 neural 改成 path tracing 是显式手工诊断，不应加入常规 cadence。

Windows worker 与迟到结果收集：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls eval worker `
  <spool-dir> <artifact-root> --max-jobs 1
.\scripts\run_falcor_python.ps1 -m ncls eval collect `
  <spool-dir> <artifact-root> <tensorboard-dir>
```

visual eval 失败只记录 hook/status，不改变训练 checkpoint。正式 export 仍要求 exact method identity、formal run、complete phase 与所需 parameter group 的 finite/nonzero-gradient/actual-update coverage；短 smoke 只能作为诊断证据。
