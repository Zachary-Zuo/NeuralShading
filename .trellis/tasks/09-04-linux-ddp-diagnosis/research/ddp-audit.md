# Linux DDP 利用率与 timeout 审计

## 1. 结论摘要

当前训练路径并没有使用 `torch.nn.parallel.DistributedDataParallel`（DDP）的 reducer。它在每次 backward 完成后，按参数逐个执行同步 `dist.all_reduce()`；Metal 长训形态每 step 是 328 次 collective，总梯度约 25.09 MiB，其中 229 个 tensor 不超过 4 KiB。这个实现能得到平均梯度，却丢失了 DDP 的 bucket 合并、backward/通信重叠和 gradient view，额外产生逐 tensor clone、除法 kernel 与 Python 循环。它是多卡提效不足的首要已确认原因。

timeout 很可能不是单一的“网络太慢”。各 rank 会在同一个 64-step block 中选择不同 execution group，而 group session、材质资源与部分 reference 状态按需构建；reference rejection/top-up 的轮数也可能不同。较快 rank 先进入第一个梯度 `all_reduce`，较慢 rank 仍停留在 group 构建、asset decode 或 reference dispatch，300 秒后外观就会是 NCCL collective timeout。checkpoint 还存在另一条不对称路径：非 rank-0 先进入 NCCL 同步，rank-0 在同步前串行写文件，慢 I/O 同样会耗尽 process-group timeout。

数据面也尚未兑现重构设计：生产入口仍固定使用 `SynchronousOnlineDataSession`；`ready_batches`、`reference_batch_steps` 和 `transfer_streams` 没有接入实际 batch 生产，`ReferenceScheduler` 只有测试实例，`reference_inflight` 主要只扩大 slot 数量。因此 GPU model 与下一 batch 的准备没有形成流水。Linux/Vulkan reference 的全局同步合同决定了不能在同卡上盲目并发 Falcor 与 Torch，但仍可通过 group 内 packed dispatch、资源预热、host cache-miss decode 预取和 ready ring 减少 barrier 与 consumer starvation。

建议先实施“真实 DDP reducer + 可观测性/checkpoint 对称化”，再实施 data/reference scheduling。先做后者会继续被 328 次串行 collective 和不可诊断的 rank 长尾遮蔽。

## 2. 证据等级

| 等级 | 含义 |
|---|---|
| 已确认 | 可由当前 HEAD 的调用链、配置或可重复静态/本机检查直接证明 |
| 历史产物支持 | 仓库内已有 Linux 运行产物支持，但样本短或产生于本次重构前，不作为 hard gate |
| 待 Linux 验证 | 机制推导成立，但必须由目标 Linux/Vulkan/NCCL stage trace 确认是否是本次 timeout 的实际触发点 |

本次 Windows 环境具有 RTX 4090、`neural-shading` Conda 环境和 Windows Falcor 构建，可做源码、unit、单卡 GPU/Falcor 检查；它不能替代 Linux 多卡证据。历史对话中没有找到原始 `ProcessGroupNCCL` watchdog 日志，因此下文不把高概率因果链写成已复现的唯一根因。

## 3. 当前控制流

```text
ncls train --devices a,b,c
  → Linux launcher 校验物理 GPU，torchrun 启动一进程一卡
  → ddp_worker 把本 rank 的 CUDA_VISIBLE_DEVICES 缩成一张卡
  → 当前进程内 Torch/SlangPy 使用 cuda:0，Falcor 使用物理 adapter
  → init_process_group(backend="nccl", timeout=300s)
  → 每 rank 打开 SynchronousOnlineDataSession、普通 nn.Module、optimizer
  → 同步准备 batch → objective forward → backward
  → 对 active parameter 逐个 all_reduce → optimizer.step
  → logging 时逐 scalar all_reduce
  → checkpoint 时全 rank 组装状态、两次 all_gather_object，rank-0 写文件
  → rank-0 最终写 summary/review；其他 rank 可先 destroy_process_group
```

launcher 的一进程一卡映射总体正确，主要问题位于 reducer、data session、checkpoint/退出协议和观测面，而不是 `torchrun` 命令本身。

## 4. 风险清单

### P0-1：名义 DDP，实际为逐参数同步 SGD（已确认）

- `TrainingEngine._ddp_sync_gradients()` 遍历每个 active parameter，对有梯度参数先 clone，对无梯度参数构造零 tensor，再逐个调用 `dist.all_reduce()`，随后除以 world size 并重新赋回 `parameter.grad`：`src/ncls/learning/training/engine.py:325`。
- `TrainingEngine.run()` 创建的是普通 method model，没有 `DistributedDataParallel` wrapper：`src/ncls/learning/training/engine.py:754`。
- sync 发生在完整 backward 之后：`src/ncls/learning/training/engine.py:924` 至 `src/ncls/learning/training/engine.py:965`。因此通信无法与梯度生成重叠。
- 当前 Metal 长训模型共有 328 个 parameter tensor、6,578,157 个元素、约 25.09 MiB；joint phase 每 step 328 次 all-reduce，QAT phase 327 次。229 个 tensor 不超过 4 KiB，299 个不超过 64 KiB。可重复脚本：`scratch/analyze_gradient_collectives.py`。
- Git 历史表明早期实现曾用 `_DDPObjective` 包装 objective，随后围绕 unused parameter、phase reducer 重建和 dummy trigger 多次修补，最终改成手工 all-reduce。当前实现解决了部分动态图 correctness 问题，但回退掉了 DDP 的性能机制。

PyTorch DDP 的基本优化单元是 gradient bucket；reducer 可在 backward 中 bucket ready 时发起 reduction，`gradient_as_bucket_view=True` 还能避免梯度与 bucket 间复制。当前实现没有利用这些能力。官方合同见 [DistributedDataParallel 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。

### P0-2：数据长尾被呈现为 collective timeout（已确认机制，待 Linux 确认触发频率）

- group-block schedule 用 `sequence_index = block * world_size + rank`，所以同一 block 的 rank 会处理不同 execution group：`src/ncls/learning/producer.py:264`。
- reference group session 是按需 materialize；不同 group 的首次 Falcor/MDL 状态构建、纹理与 asset cache 成本不一致：`src/ncls/references/query.py:930` 附近。
- evaluator batch 对 invalid sample 最多执行 64 轮 rejection/top-up，每轮同步调用 reference：`src/ncls/learning/producer.py:478`。
- Vulkan dispatch 的 interop 边界含 `wait_for_cuda()`、blocking compute 与 `wait_for_falcor()`：`src/ncls/references/query.py:671`。backend capability 也把 Vulkan 标为 global synchronization、无 async submit：`src/ncls/references/backend.py`。
- rank-local stage timing 没有汇聚为 min/mean/max 和 culprit rank。较快 rank 在 `engine.py:960` 进入第一个 all-reduce 后，日志只能看到 NCCL 等待，无法说明另一 rank 实际仍在 reference/data stage。
- `_setup_ddp()` 把 timeout 固定为 `NCLS_DDP_TIMEOUT_SECONDS`，默认 300 秒：`src/ncls/cli.py:438`。PyTorch 当前 NCCL 默认 timeout 是 10 分钟；timeout 后 collective 会被异步 abort，因为继续执行 CUDA 不再安全。官方合同见 [Distributed communication package](https://docs.pytorch.org/docs/stable/distributed.html)。本项目较短的 300 秒不是根因，只是更早暴露任一跨 rank 长尾或 collective desync。

高概率时序如下：

```text
rank 0: cheap group → batch ready → backward → all_reduce(parameter 0) ─────────┐
rank 1: lazy group/material build → asset decode → rejection/top-up → backward ├→ 相遇或 300s timeout
rank 2: another group → reference global waits → backward ─────────────────────┘
```

### P0-3：checkpoint 与最终退出存在 rank 不对称（已确认）

- checkpoint 前的 `data_session.drain()` 当前同步实现是 no-op：`src/ncls/data/session.py:108`。
- 每个 rank 都把 optimizer state 搬到 CPU并编码完整 checkpoint：`src/ncls/learning/training/engine.py:557`、`src/ncls/learning/training/engine.py:708`。
- RNG/data state 分别执行一次 `all_gather_object`，而 envelope 又重复包含 RNG：`src/ncls/learning/training/engine.py:347`、`src/ncls/learning/training/engine.py:708`。完整 object 被发送到所有 rank，而 durable writer 只有 rank-0。
- periodic callback 在非 rank-0 直接返回；rank-0 继续磁盘写入，随后所有 rank 才进入 `_ddp_checkpoint_sync()`：`src/ncls/cli.py:259`、`src/ncls/learning/training/engine.py:1277`。若 rank-0 write/hook 超过 timeout，其他 rank 已在 NCCL 等待。
- final checkpoint/summary/review 也只有 rank-0 写；非 rank-0 可先进入 `destroy_process_group()`，没有最终成功状态广播或 ack。变量 `ddp_completed` 被赋值但未消费：`src/ncls/cli.py:295` 至 `src/ncls/cli.py:375`。

PyTorch 要求所有 rank 在一致顺序中显式销毁 process group，NCCL communicator 的 collective abort 顺序尤其需要一致。官方说明见 [Shutdown 文档](https://docs.pytorch.org/docs/stable/distributed.html#shutdown)。

### P1-1：数据执行计划与生产调用链脱节（已确认）

- CLI 固定实例化 `SynchronousOnlineDataSession`：`src/ncls/cli.py:97`。
- `next_batch()` 直接同步调用 producer，engine 的 queue fill 也在训练主线程逐个执行 `_prepare_step()`，然后才消费/计算：`src/ncls/data/session.py`、`src/ncls/learning/training/engine.py:845`。
- `ReferenceScheduler` 只有 unit test 构造，生产代码未实例化。
- `ready_batches`、`reference_batch_steps`、`transfer_streams` 只存在于 plan/config/test；`reference_inflight` 在生产中主要用于增大 reference slot 数量，没有形成 in-flight dispatch。
- `HostPipeline` 只覆盖 Metal asset cache-miss decode，并在当前 batch 内等待结果；没有在 model compute 时准备未来 batch：`src/ncls/learning/mdl_metal_assets.py:832`。

因此当前 `num_workers=2` 等配置不是完整的 online batch pipeline。Vulkan global synchronization 下不应开启同卡 Falcor/Torch 无约束并发；正确方向是减少 barrier、提前 materialize 下一 group、host-only 预取并建立有界 ready ring。

### P1-2：指标和日志会误导性能判断（已确认）

- logging 的 `global_work_units_per_second = steps_per_second * world_size`，没有乘每个 step 的实际 work units：`src/ncls/learning/training/engine.py:1043`。字段名与公式不一致。
- loss/metric 逐 scalar 执行 collective：`src/ncls/learning/training/engine.py:377`、`src/ncls/learning/training/engine.py:1065`。这在 log cadence 继续放大小 collective 开销，并隐式依赖各 rank 的 dict 顺序完全相同。
- prepare/forward/backward/optimizer timing 主要是 rank-0 local 观察，没有跨 rank 的 max 或 skew；`optimizer` interval 实际包含手工 gradient sync。
- 普通 rank 也创建进度条，增加重复输出风险。

### P1-3：缺少能保护 DDP correctness 的测试（已确认）

- `tests/unit/test_training_launcher.py` 主要验证 topology/env，不运行 NCCL。
- `tests/unit/test_training_engine.py` 不覆盖 DDP reducer、collective 次序或多 rank state。
- `tests/unit/test_multi_gpu_launcher.py` 主要是静态字符串断言。
- 先前实施计划提到的 `tests/integration/test_distributed_training.py` 当前不存在。
- 当前架构清理任务的 Phase 5 仍明确保留 Linux/NCCL stage trace 与 packed reference 为未完成项。

### P2：次级改进（已确认）

- `init_process_group()` 没有传 PyTorch 2.11 已支持的 `device_id=torch.device("cuda:0")`，错过 NCCL communicator eager init 和更早暴露初始化错误的机会。
- global batch 会随 world size 增长，当前报告没有严格区分固定 per-rank batch 的 weak scaling 与固定 global batch 的 strong scaling；吞吐与收敛比较容易混在一起。
- 没有 DDP reducer logging、per-rank stage event 或 production-safe flight recorder 配置，timeout 后缺少定位 collective/rank 的证据。

## 5. 历史 Linux 量化对照

两个 16-step run 使用相同 config hash `a136a7a9...`。计算使用累计 `work_units / training_elapsed_seconds`，不使用当前错误的 `global_work_units_per_second` 字段。

| run | GPU | elapsed (s) | 真实 work/s | prepare mean (s) | forward | backward | optimizer + sync | max step interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `single-gpu1-eta` | 1 | 22.924 | 62.12 | 0.641 | 0.378 | 0.312 | 0.013 | 3.915 |
| `ddp-234-final` | 3 | 45.048 | 94.83 | 0.701 | 0.391 | 0.328 | 1.286 | 8.692 |

- 3 卡相对单卡 speedup：`94.83 / 62.12 = 1.53×`。
- 3 卡并行效率：`1.53 / 3 = 50.9%`。
- `optimizer + sync` 均值从 0.013 秒上升到 1.286 秒，约为 99 倍；该 interval 包含逐参数 gradient sync，与当前源码机制一致。
- run 只有 16 step，startup/group warmup 占比高，且是本轮重构前的历史产物。它支持“同步设计严重限制 scaling”的判断，但不能充当最终性能门。

产物位置：`artifacts/metal-linux-training/ddp-234-final/` 与 `artifacts/metal-linux-training/single-gpu1-eta/`。

## 6. 推荐整改结构

### 阶段 A：恢复真正的 DDP reducer

1. 建立 engine-owned `DistributedObjective(nn.Module)`，持有真实 model；其 `forward()` 只调用 method objective，并把 loss 作为 DDP forward 的可追踪输出，detached metrics 存为旁路结果。
2. lifecycle 先配置当前 phase 的 `requires_grad`/parameter ownership，再在所有 rank 同步的 phase boundary 构造或重构 DDP wrapper。phase 内保持 parameter graph 稳定。
3. 首版使用 `find_unused_parameters=True`、`gradient_as_bucket_view=True`、`static_graph=False`。只有目标 trace 和 `_get_ddp_logging_data()` 证明 graph 可静态化后，才按 phase 启用 `static_graph=True` 或 `skip_all_reduce_unused_params`。
4. 删除手工逐参数 `_ddp_sync_gradients()`。optimizer 始终绑定 underlying model 的当前 phase parameters。
5. `init_process_group(..., device_id=torch.device("cuda:0"))`，保持一进程一卡和当前物理 adapter 映射。
6. `bucket_cap_mb` 只按真实 trace 调整。当前 25 MiB 模型应被少量 bucket 处理，而不是预设未经验证的极小 bucket 或继续 328 次 collective。

不能简单把 MetalModel 外面套 DDP：MetalModel 本身没有统一 `forward()`，objective 经 method facet 调用多个子模块，且 phase 会改变 active parameter group。wrapper 必须成为 objective 执行图的 owner，phase 切换必须是显式、全 rank 对称的 reducer lifecycle。

### 阶段 B：控制面、观测与 checkpoint 对称化

1. 将固定顺序的 loss/metrics 打包为一个 tensor，在 log cadence 做一次 collective；构造时跨 rank 校验 metric descriptor/hash。
2. 每个 stage 记录 rank-local duration；report 汇总 min/mean/max、straggler rank 与 step/group/request identity。修正 cumulative work/s 公式。
3. durable checkpoint 只在 rank-0 进行完整 model/optimizer CPU snapshot 与写入。其他 rank 只提交小型 RNG/data cursor；优先使用辅助 Gloo control group 的 gather/object/status broadcast，或者结构化 tensor collectives，避免把完整 object 广播给所有 rank。
4. rank-0 写入完成后广播 commit status；所有 rank 在同一 final ack 后按相同顺序 close/destroy。写入异常广播 failure，不能让 peer 等到 watchdog。
5. 辅助 Gloo group 可在 phase/group/checkpoint 边界使用 `monitored_barrier` 报出未到达 rank；不要每 step 加 barrier。PyTorch 的 `monitored_barrier` 只支持 Gloo，且用于 debug 会有同步开销。
6. Linux 诊断运行启用 `TORCH_DISTRIBUTED_DEBUG=DETAIL`。NCCL flight recorder/monitoring 环境变量要先在锁定的 PyTorch 2.11 目标环境验证可用，再冻结到脚本；不要把当前网上 2.14 文档中的全部变量未经版本核对直接写成 2.11 合同。

### 阶段 C：接通 online data/reference scheduler

1. 用真实异步/有界 `OnlineDataSession` 接管 logical request、host work、residency、reference dispatch、ready ring、drain 和 cursor；移除 engine 自身同步填 queue 的假预取。
2. Vulkan 保持 global-sync capability，不宣称同卡 reference/model overlap。先在同一 execution group 内把多个 logical step packed 成一次 reference dispatch，保持 RNG/top-up/输出次序，再切分到 ready ring。
3. 在进入 64-step group block 前显式 materialize/warmup 所需 group；预取下一 group 的 host-only asset/cache miss。增加 cost-aware/balanced schedule 或同 group matched baseline，用来隔离 rank 间 group 成本差异。
4. `reference_batch_steps>1` 只有通过逐 logical-step identity、invalid top-up、checkpoint drain/resume 一致性后才成为性能选项；同步 `1` 保留为 correctness baseline。

## 7. Linux 目标机验证矩阵

所有产物写入新目录，例如 `artifacts/09-04-linux-ddp-diagnosis/<run-id>/`，不得覆盖历史 run。

### 7.1 Preflight 与版本记录

```bash
nvidia-smi -L
conda run -n neural-shading python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.nccl.version())
print(torch.distributed.is_nccl_available())
PY
bash scripts/run_falcor_python.sh -m ncls train configs/training/runs/metal-linux-smoke.yaml --devices 0
```

记录 OS/kernel/driver、GPU topology、PyTorch/CUDA/NCCL/Falcor commit、完整 resolved plan/hash 和环境变量。确认每 rank Torch 是 local `cuda:0`、Falcor physical adapter 与 launcher manifest 一致。

### 7.2 Correctness

- 先运行synthetic NCCL reducer/control-plane gate：`bash scripts/run_falcor_python.sh --gpus 0,1 -- -m pytest tests/integration/test_distributed_training.py -q`。
- 1 step matched test：把多 rank 的 deterministic local batches 拼成 single-process global batch，对比 loss、每个 active gradient 和 optimizer 后参数；规定 dtype 对应的 atol/rtol。
- phase boundary：跨 `joint-coarse-to-fine → qat-refine`，验证 active/unused parameter 集、DDP wrapper 重建次序和 optimizer state。
- uninterrupted 与 checkpoint/resume：逐 rank RNG/data cursor、model/optimizer/scheduler 和下一个 logical request 一致。
- rank-0 only artifact：checkpoint、metrics/TensorBoard/summary 只生成一次，其他 rank 收到相同 commit status。

### 7.3 Scaling 与 stage trace

分别运行 1/2/3/4 卡：

- weak scaling：固定 per-rank batch/work，报告 global work/s、speedup、并行效率。
- strong scaling：固定 global work，报告 step time 与通信占比；不能与 weak scaling 混用结论。
- 每个 run 先 warmup，并跨至少一个 64-step group boundary；不能只用 16-step startup smoke。
- 对比 `manual-allreduce-before` 与 `ddp-reducer-after`，source/query/model/precision/batch/work-unit 完全 matched。

每 rank trace 至少包含：

```text
step, phase, block, execution_group, logical_request
prepare_host, asset_wait, reference_submit, reference_wait, rejection_rounds
forward, backward, reducer_wait, optimizer, metrics_reduce
checkpoint_snapshot, checkpoint_gather, rank0_write, commit_broadcast
queue_depth, consumer_starvation, cache_hit/miss, barrier_count
```

报告每项的 rank min/mean/max、p50/p90/p99 与 straggler rank；将 DDP reducer 的 bucket 数、bytes、overlap/unused/static-graph 信息一起保存。

### 7.4 Fault injection

| 注入 | 期望结果 |
|---|---|
| 指定 rank 在 group materialize 前 sleep | stage trace 指出该 rank/data stage；辅助 monitored barrier 报未到达 rank，不只留下 NCCL all-reduce timeout |
| 指定 rank 的 reference 抛异常 | 所有 worker 有界时间内非零退出，rank-0 不发布 success checkpoint |
| rank-0 checkpoint write 人为延迟/失败 | peer 等待显式 commit status；失败被广播，不耗尽 NCCL watchdog |
| phase boundary active parameter 不一致 | descriptor/hash 或 DDP consistency check 在首个训练 collective 前 fail fast |

### 7.5 判读规则

- reducer 后 `optimizer + sync` 显著下降且 bucket/overlap 正常：确认通信实现是主要瓶颈之一。
- rank max prepare/reference 远高于 mean，且 reducer 首 bucket wait 随之增长：确认数据长尾是 timeout/低效率主因。
- packed dispatch 降低 barrier/step，但 model/reference timeline 仍不重叠：这符合 Vulkan global-sync 合同，不应误判为 scheduler 失败。
- checkpoint delay 注入仍导致 NCCL watchdog：说明 rank-0 commit 协议仍不对称。
- 吞吐只作为 observed report；correctness、bounded failure、rank 对称退出和 trace 完整性才是硬验收。

## 8. 优先级结论

| 优先级 | 工作 | 原因 |
|---|---|---|
| P0 | 阶段 A：真实 DDP reducer | 直接消除每 step 328 次串行 collective，恢复 bucket 与 backward overlap |
| P0 | 阶段 B：per-rank stage、checkpoint/final ack | 先把数据长尾、通信 hang 与 rank-0 I/O 区分开，并消除已知不对称等待 |
| P1 | Linux matched correctness/scaling/fault injection | 当前没有原生 NCCL integration test 或足够 trace，不能仅靠 Windows/历史 smoke 宣称修复 |
| P1 | 阶段 C：接通 data/reference scheduler | 当前配置轴多数未生效；应在 reducer 与观测面可靠后优化 |
| P2 | bucket/cost schedule/timeout 调参 | 必须依据目标机 trace，不先验拍数值，不用增大 timeout 掩盖错误 |
