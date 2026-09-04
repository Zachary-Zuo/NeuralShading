# Linux DDP 整改技术设计

## 1. 设计目标

把当前“torchrun 多进程 + 手工逐参数 all-reduce”改为真正由 PyTorch DDP reducer 管理的同步训练，并建立能够区分 model compute、gradient communication、reference/data 长尾、checkpoint I/O 与进程退出的观测和控制面。随后再把已有 data/reference primitive 接入生产 pipeline。

本设计不改变 source/reference/material 语义，不修改 Falcor，不承诺 Vulkan 同卡 reference/model 异步执行，也不把提高 timeout 当作修复。

## 2. 目标结构

```text
torchrun / one process per GPU
        │
        ├─ NCCL data group ── DDP reducer + packed scalar metrics
        └─ Gloo control group ─ rank state gather / monitored boundary / commit status

phase lifecycle
  → configure requires_grad and optimizer ownership on underlying model
  → all-rank phase descriptor check
  → construct phase-local DistributedObjective + DDP
  → online session acquire batch
  → DDP forward objective → backward/reducer overlap → optimizer
  → rank-local stage trace + cadence reduction
  → drain → rank state gather → rank-0 checkpoint write → status broadcast
  → final all-rank ack → ordered teardown
```

## 3. DDP objective owner

MetalModel 没有覆盖全部训练目标的统一 `forward()`；method objective 会按 phase 调用多个子模块。因此 DDP owner 不是裸 model，而是公共 `DistributedObjective(nn.Module)`：

- 注册 underlying model 为 child module，使 reducer 能发现真实 parameters/buffers。
- `forward(typed_batches, step_context)` 调 method objective，返回 scalar loss；metrics 只以 detached tensor/值存入本 step 结果。
- lifecycle 在构造 wrapper 前设置 phase parameter ownership；phase 内不得随 step 改变 `requires_grad` 或参数集合。
- phase boundary 全 rank drain，比较 phase/parameter descriptor hash，销毁旧 wrapper，更新 optimizer ownership，再以一致参数顺序构造新 wrapper。
- 初始 DDP 配置使用 `find_unused_parameters=True`、`gradient_as_bucket_view=True`、`static_graph=False`。只有 Linux trace 证明某 phase graph 固定，才在该 phase 单独收紧。

这避免此前用 dummy reducer trigger、每 step rebuild 或手工 zero-gradient collective 绕过 DDP forward/backward 合同。

## 4. DistributedContext

把 process-group 与 rank collective 从 CLI/engine 分散逻辑收敛为公共 `DistributedContext`：

- `data_group`：NCCL，DDP reducer、GPU tensor metrics。
- `control_group`：Gloo，低频 rank metadata、故障定位和 checkpoint commit；所有 rank 同序创建/销毁。
- `init_process_group(..., device_id=cuda:0)` 在 PyTorch 2.11 下 eager 初始化 NCCL。
- 提供 `reduce_metrics()`, `gather_rank_state()`, `broadcast_status()`, `monitored_boundary()`, `close()`；业务代码不直接拼 collective 顺序。
- production hot loop 不插入 barrier。monitored boundary 只在 phase/group/checkpoint 或 debug cadence 使用。

任何 rank-local异常先转换为带 stage/request/rank 的 failure record；能进入 control collective 时广播失败，无法进入时交给 torchrun supervisor 与 flight recorder，不能尝试继续 optimizer/checkpoint。

## 5. 指标和 trace

- metric descriptor 在 run resolve 时冻结名称、dtype 与 reduce policy，所有 rank 校验 hash。
- 同 cadence scalars pack 成固定 tensor，一次 collective 后由 rank-0 解包；不按 Python dict 动态顺序逐项通信。
- throughput 使用累计 `global work units / active training elapsed`；同时报告 steps/s 与 per-rank/global work，不再混用。
- stage trace 每 rank 独立落盘或汇聚，必须保留 phase/step/block/group/request 和 straggler rank。
- reducer 记录 bucket 数/bytes、unused parameters、是否可 static graph；trace 功能不能在普通每 step 强制 `cuda.synchronize()`。

## 6. Checkpoint 协议

checkpoint 是显式的三阶段提交：

```text
all ranks drain and freeze consumed cursor
  → gather small rank-local RNG/data state to rank 0
  → rank 0 creates full model/optimizer snapshot and atomically writes
  → rank 0 broadcasts {success, checkpoint_id, error}
  → all ranks advance checkpoint cursor or fail together
```

- 非 rank-0 不构造完整 CPU optimizer/model payload。
- rank-local state 只含恢复所需的小对象；消除 envelope 中重复 RNG。
- periodic 与 final checkpoint 复用同一协议。
- summary/review 必须在 final commit 协议内，或者在全 rank 已确认训练完成后由 launcher 的父进程生成；不能让非 rank-0 在 rank-0 写入时提前 destroy NCCL group。
- 异常路径不发布成功 sidecar/summary；atomic writer 保留现有 durable 语义。

## 7. Data/reference pipeline 边界

第二批次才接通 `OnlineDataSession`：

- session 拥有 logical request、host pipeline、GPU residency、reference scheduler、ready ring、lease、drain 与 checkpoint cursor。
- engine 只 acquire/release typed batch，不自行同步 fill queue。
- Vulkan capability 仍是 `global`：不运行与 model stream 并发的同卡 Falcor dispatch；只做 group-homogeneous packed dispatch、预分配/复用、host-only 预取和 consumer ready ring。
- 进入 group block 前 materialize/warmup，并记录 warmup 成本；不同 rank 的 group mapping必须可做 same-group matched baseline 与 cost-balanced 实验。
- `reference_batch_steps=1` 是 correctness/rollback baseline；大于一的 pipeline identity 进入 resolved plan 与 checkpoint compatibility。

## 8. 失败语义

| 边界 | 失败处理 |
|---|---|
| DDP init / phase descriptor | 首步前 fail fast，所有 rank 非零退出 |
| data/reference stage | 携带 rank/stage/request provenance，取消 pending work，不写 success checkpoint |
| reducer/collective | 保存可用 flight-recorder 与 rank trace，由 torchrun 终止整个 worker group |
| checkpoint gather/write | 广播 failure；所有 rank 在同一协议点退出 |
| teardown | 所有 process group 同序 destroy；没有 rank-0 私有长操作夹在最后一个 collective 与 destroy 之间 |

## 9. 兼容与回滚

- 单卡路径继续使用同一 objective owner，但不构造 DDP；数值行为由 matched test保护。
- 第一批次保留 `SynchronousOnlineDataSession`，使 reducer/control-plane 修复不与 data pipeline 语义变化耦合。
- data pipeline 优化可显式退回 `reference_batch_steps=1` 与同步 session；不得静默 fallback 后仍报告优化模式。
- checkpoint schema 若增加 distributed metadata，提升版本并提供明确兼容检查；不按 shape 猜测恢复。

## 10. 设计风险

| 风险 | 处理 |
|---|---|
| objective 返回结构使 DDP 找不到 loss graph | wrapper 的公开 forward 直接返回 loss；metrics detached，增加两 rank gradient parity test |
| phase 间 parameter set 改变导致 reducer desync | 只在全 rank drain boundary 重构；descriptor hash 先校验 |
| `find_unused_parameters=True` 有额外 graph traversal | correctness-first；按 phase logging 证明静态后再收紧 |
| Gloo control group 与 NCCL group collective 次序交叉 | DistributedContext 固定协议；所有 rank 同序调用，fault injection覆盖 |
| rank-0 checkpoint memory 峰值 | 仅 rank-0 CPU snapshot，记录 peak；必要时采用 streaming writer，但不让各 rank复制完整 payload |
| packed reference 改变 RNG/top-up | logical request 先冻结；逐 step identity、数值和 resume test 作为启用门 |
