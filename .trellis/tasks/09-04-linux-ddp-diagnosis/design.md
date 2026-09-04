# Linux DDP 整改技术设计

## 1. 设计目标

把当前“torchrun 多进程 + 手工逐参数 all-reduce”改为真正由 PyTorch DDP reducer 管理的同步训练，建立能够区分 model compute、gradient communication、reference/data 长尾、checkpoint I/O 与进程退出的观测和控制面，并把已有 data/reference primitive 接入唯一生产 pipeline。模型、method、engine 与数据语义只依赖通用 capability 和 session 合同，Windows/Linux 设备与 API 差异停留在 launcher/backend 层。

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
  → submit bounded same-phase/same-group step requests
  → host-only prefetch + packed reference dispatch → ready ring
  → acquire/release typed step batches
  → DDP forward objective → backward/reducer overlap → optimizer
  → rank-local stage trace + cadence reduction
  → drain → rank state gather → rank-0 checkpoint write → status broadcast
  → final all-rank ack → ordered teardown

platform boundary
  Windows/Linux launcher → Device/ReferenceBackendCapability
  common plan/session/typed batch/checkpoint → engine/model/method
```

## 3. DDP objective owner

MetalModel 没有覆盖全部训练目标的统一 `forward()`；method objective 会按 phase 调用多个子模块。因此 DDP owner 不是裸 model，而是公共 `DistributedObjective(nn.Module)`：

- 注册 underlying model 为 child module，使 reducer 能发现真实 parameters/buffers。
- `forward(typed_batches, step_context)` 调 method objective，返回 scalar loss；metrics 只以 detached tensor/值存入本 step 结果。
- lifecycle 在构造 wrapper 前设置 phase parameter ownership；phase 内不得随 step 改变 `requires_grad` 或参数集合。
- phase boundary 全 rank drain，比较 phase/parameter descriptor hash，销毁旧 wrapper，更新 optimizer ownership，再以一致参数顺序构造新 wrapper。
- 初始审计阶段先允许unused探测；Linux目标机DDP logging与required-group audit已经证明两个phase的active graph固定且unused bytes为0，最终配置因此收紧为`find_unused_parameters=False`、`gradient_as_bucket_view=True`、`static_graph=True`。未来候选若引入条件分支，应修正phase ownership或拆phase并新增图稳定性测试，不恢复unused兼容扫描。

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

本任务接通唯一的 pipeline-backed `OnlineDataSession`：

- engine 把一个 global step 的有序 route request 组成通用 step request，并在 phase、validation、checkpoint、stop 与 execution-group 边界内提交有界 lookahead；session 按 logical step 顺序 acquire/release typed batch，engine 不再同步调用 producer 填自己的 batch queue。
- `OnlineProducer` 提供批量 step 生产入口；统一 producer 内部按 route 规划 host-only work，并把同 execution group 的连续 reference-evaluator request 交给 `ReferenceScheduler`。不增加 family-specific runner 或 Metal data loop。
- `submit` 只启动可序列化 host work；CUDA、Falcor、reference session、residency lease 与 packed dispatch 始终由 rank 主进程在 `acquire/pump` 边界拥有。ready ring 满、host worker失败或 consumer未释放时显式 backpressure/fail closed。
- Vulkan capability 仍是 `global-sync`：不运行与 model stream 并发的同卡 Falcor dispatch；session 以 `reference_batch_steps` 把同 group logical query 合并一次 dispatch，按冻结 ID/RNG/top-up 次序切分结果，再让模型连续消费 ready ring。stream-fence backend 才能把 `reference_inflight` 提升为真实并发。
- 进入 group block 前 materialize/warmup，并记录 rank-local build、reference wait、ready depth 与 consumer starvation。不同 rank 的 group mapping先由 trace 判断是否需要确定性的 cost-balanced 调度；任何调度变化都进入 query/plan identity，不使用运行时自适应结果偷偷改变 resume 序列。
- `reference_batch_steps=1`、`ready_batches=1`、`num_workers=0` 是同一实现的 correctness/rollback 配置；大于一的 pipeline identity进入resolved plan与checkpoint compatibility。生产CLI不再固定构造`SynchronousOnlineDataSession`，也不保留转发到新实现的compatibility alias。

## 8. 跨平台能力边界

- `TrainingEngine`、method model/objective/lifecycle、`DataExecutionPlan`、`OnlineDataSession` 与 source adapter 不读取 `platform.system()`、`CUDA_VISIBLE_DEVICES`、物理 GPU 序号、Falcor API 或 backend key。
- Linux launcher 负责一进程一卡、物理 adapter 映射与 NCCL；Windows launcher负责单卡与D3D12。两者都构造相同的 `ExecutionContext` 与 `ReferenceBackendCapability`，上层只按 capability 决定 global-sync packing 或 stream-fence inflight。
- backend 不支持请求能力时在 plan/session 构造阶段拒绝；不静默退回同步旧入口，也不通过 `if Windows/if Linux` 复制训练路径。
- 现有 legacy checkpoint v4 只读 importer 是独立数据格式边界，不作为平台 compatibility layer 扩张，本任务不删除也不新增其训练恢复能力。

## 9. 失败语义

| 边界 | 失败处理 |
|---|---|
| DDP init / phase descriptor | 首步前 fail fast，所有 rank 非零退出 |
| data/reference stage | 携带 rank/stage/request provenance，取消 pending work，不写 success checkpoint |
| reducer/collective | 保存可用 flight-recorder 与 rank trace，由 torchrun 终止整个 worker group |
| checkpoint gather/write | 广播 failure；所有 rank 在同一协议点退出 |
| teardown | 所有 process group 同序 destroy；没有 rank-0 私有长操作夹在最后一个 collective 与 destroy 之间 |

## 10. 迁移与回滚

- 单卡路径继续使用同一 objective owner，但不构造 DDP；数值行为由 matched test保护。
- session迁移一次完成：CLI、engine、tests与公共export改用pipeline-backed实现；旧`SynchronousOnlineDataSession`删除，不增加别名或双写状态。
- data pipeline 可通过同一实现的 `reference_batch_steps=1`、`ready_batches=1`、`num_workers=0` 配置回滚性能机制；不得静默 fallback 后仍报告优化模式。
- checkpoint schema 若增加 distributed metadata，提升版本并提供明确兼容检查；不按 shape 猜测恢复。

## 11. 设计风险

| 风险 | 处理 |
|---|---|
| objective 返回结构使 DDP 找不到 loss graph | wrapper 的公开 forward 直接返回 loss；metrics detached，增加两 rank gradient parity test |
| phase 间 parameter set 改变导致 reducer desync | 只在全 rank drain boundary 重构；descriptor hash 先校验 |
| static graph与候选条件分支不一致 | phase ownership和required-group测试先行；条件图拆phase，不用unused兼容扫描掩盖 |
| Gloo control group 与 NCCL group collective 次序交叉 | DistributedContext 固定协议；所有 rank 同序调用，fault injection覆盖 |
| rank-0 checkpoint memory 峰值 | 仅 rank-0 CPU snapshot，记录 peak；必要时采用 streaming writer，但不让各 rank复制完整 payload |
| packed reference 改变 RNG/top-up | logical request 先冻结；逐 step identity、数值和 resume test 作为启用门 |
| global-sync backend 被误当作可异步 | capability在plan构造时限制inflight；只允许host overlap和同group packing |
| 不等长local batch破坏DDP平均语义 | strong scaling只运行route batch可精确等分的1/2/4卡；3/5卡只做固定per-rank weak scaling |
| 为消除rank长尾改变数据分布 | 先用rank trace验证；若启用cost-balanced schedule，使用确定性静态代价并纳入query identity |
| compatibility入口长期共存 | 迁移后删除旧session与导出，baseline由同一实现的配置轴表达 |
