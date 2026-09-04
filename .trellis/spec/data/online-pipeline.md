# 通用 online data pipeline 合同

## 1. Scope / Trigger

修改训练 route、CPU/host 预处理、GPU 资源缓存、reference 调度、prefetch、lease、checkpoint drain 或 Linux 性能打点时适用。该合同确保方法只定义数据需求和获取语义，公共 data plane 负责并行、顺序、资源与恢复。

## 2. Signatures

```text
MethodDataFacet.requirements() -> tuple[DataRequirement, ...]
MethodDataFacet.create_source_adapter(snapshots, device) -> MethodSourceAdapter
DataExecutionPlan.build(data_key, source_family_id, routes, requirements,
                        execution, rank, world_size) -> DataExecutionPlan@1
OnlineStepRequest(logical_id, boundary_id, routes)
OnlineProducer.prefetch_steps(tuple[OnlineStepRequest, ...]) -> None
OnlineProducer.produce_steps(tuple[OnlineStepRequest, ...])
  -> Sequence[Mapping[route_name, OnlineBatch]]
OnlineDataSession.submit_step(routes, boundary_id=...) -> logical_id
OnlineDataSession.acquire_step(logical_id) -> OnlineStepBatch
OnlineStepBatch.release()
OnlineDataSession.state_dict()/load_state_dict(...)
OnlineDataSession.drain()/cancel_pending()/close()
HostPipeline(processor, num_workers, capacity, stage, rank)
GpuResidencyManager.acquire(key, size_bytes, loader) -> ResidencyLease
ReferenceScheduler(dispatcher, capability, batch_steps,
                   ready_capacity, maximum_inflight)
```

YAML `execution` 必须精确包含：

```yaml
devices: [0]
num_workers: 0
host_prefetch: 2
ready_batches: 2
reference_batch_steps: 1
reference_inflight: 1
transfer_streams: 0
residency: {budget_mib: 4096}
```

## 3. Contracts

- `num_workers` 只执行可复制、可序列化、确定性的 CPU/host processor。worker 使用 `spawn` 或 `forkserver`，不得拥有 CUDA context、Falcor device/session、GPU tensor 或 active lease。
- 正式训练只有一个`PipelineOnlineDataSession`。`ready_batches=1`、`reference_batch_steps=1`表达同步调试基线；不得恢复`next_batch()`、同步session别名或按平台分叉的兼容层。
- engine先按训练顺序提交完整step的named routes，再严格按返回的`logical_id`获取和释放`OnlineStepBatch`。lookahead不得跨phase、validation、checkpoint或stop边界；capacity包含pending、ready和acquired step。
- `HostPipeline` 队列有界，按 `logical_id` 恢复提交顺序；worker error、异常退出和 backpressure 必须传播到 rank owner。checkpoint/phase/validation 前 `drain()`，失败或 close 时取消未消费工作。
- `prefetch_steps()`只能安排host工作，不得推进producer cursor、创建CUDA/Falcor资源或改变logical sample。CUDA residency materialize和reference dispatch仍在rank主进程的`produce_steps()`中发生。
- CUDA、Falcor、reference session、GPU transfer 和 ready batch 由 rank 主进程拥有。`transfer_streams` 只能用于 cache miss 的 H2D，且发布 tensor 前必须用 event/fence 建立消费者依赖。
- `GpuResidencyManager` 按真实 bytes 计费并有硬预算；LRU 只能驱逐 refcount=0 的条目。loader 失败不得留下占位或已扣预算；oversize 单项 fail closed。
- source adapter 的 typed metadata 在初始化时常驻 GPU。Metal patch sampling 的 `asset_index/uv/mip` 选择保持设备端；命中路径的 request metadata host-readback bytes 必须为 0。只有资源 miss 可进入 host decode/pinned transfer。
- reference 并发来自 `ReferenceConcurrencyCapability`，不能由 `os.name` 或 backend 名称猜测。global-sync backend 只允许 group-homogeneous packed dispatch 降低 barrier/API transition；stream-fence backend 才能增加真实 inflight。
- packed request 保持 logical step、RNG 消费、group、invalid top-up、provenance 和输出切分。每个route request以不可变的`name/seed/request_index`创建独立generator；`ready_batches`、`reference_batch_steps`等执行计划参数不得进入样本RNG identity。`reference_batch_steps=1`是语义基线；大于一必须有逐tensor等价性与resume测试。
- 一次production中共享native/reference lease的多个step共用一个lifecycle；只有最后一个step释放后才调用producer `end_iteration()`。detached batch在发布ready step前即可结束iteration。
- `DataExecutionPlan.identity` 进入 checkpoint，rank partition 规则进入公共 identity，具体 rank 进入 session identity。checkpoint 前 session 必须没有 queued batch、in-flight dispatch 或 active consumer lease。
- 队列深度、host/transfer/reference/model wall、barrier、cache hit/miss/evict、readback/H2D bytes、resident/leased bytes 与峰值显存进入 `PipelineTrace`；观察值不自动成为 hard gate。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| method requirement 与 route kind/fields 不闭合 | `DataExecutionPlan.build()` 拒绝 |
| `num_workers < 0`、队列/预算非正、重复 device/rank 越界 | plan 构造拒绝 |
| processor 不可序列化或 start method 为 `fork` | host pipeline 在启动 worker 前拒绝 |
| worker 抛异常、非零退出或结果 logical ID 丢失 | session poison 并传播原 stage/rank/request，不静默同步重试 |
| ready ring已达到capacity | `submit_step()`显式报backpressure；不覆盖、重排或无限扩容 |
| acquire logical ID不是队首，或同时持有多个consumer step | session拒绝，防止训练顺序和lease边界漂移 |
| producer返回step数或route名称集合不匹配 | 释放已产出batch后失败，不发布部分step |
| residency 超预算且所有候选都有 active lease | acquire 拒绝；不驱逐活跃资源 |
| backend capability 不允许请求的 inflight | scheduler 构造拒绝，不按平台回退 |
| packed dispatch 跨 execution group 或结果数不符 | scheduler 释放已返回 lease 后失败 |
| checkpoint 时仍有 queue/inflight/lease | `drain/assert_idle` 失败，不发布 checkpoint |
| resume 的 data plan/query/source identity 漂移 | 恢复拒绝，不装载 cursor |

## 5. Good / Base / Bad Cases

- Good：Metal source adapter 声明三类 route，并在 8 GiB budget 内复用 GPU mip pyramid；两个 CPU worker只处理 miss 的 payload，命中 step 无 request metadata readback。
- Good：global-sync reference 把同一 group 的四个 logical step 合并一次 dispatch，再按原 ID 切分；RNG/cursor 与四次单步基线一致。
- Good：step 0/1共用一次packed reference dispatch；模型消费step 0时，step 2的Metal host decode可已提交，但GPU materialize仍由rank owner在后续production执行。
- Base：`num_workers=0`、`reference_batch_steps=1`、`reference_inflight=1` 使用同一 session/checkpoint 合同，是调试基线而不是另一套实现。
- Bad：用 DataLoader worker 构造 Falcor/CUDA session；每 step 把 GPU `asset_index` 转成 Python list；为提高利用率在 global-sync Vulkan 路径与 model 同卡并发。

## 6. Tests Required

- unit：plan closure/identity、`num_workers=0/1/N`顺序、step route set、backpressure、共享lifecycle、worker crash、drain/cancel、resume cursor；
- unit：residency bytes/LRU/active lease/loader rollback；reference pack/group boundary/result lease/inflight capability；
- unit：packed/unpacked逐tensor相同，execution plan改变不改变logical sample，rejection top-up仍按request切分；
- GPU：Metal resident sampling 与旧数值路径一致，命中 trace 的 metadata readback 为 0，normal/address/mip 语义不变；
- integration：stop/resume 的 model、optimizer、RNG/data cursor 相同，phase/validation/checkpoint 无未释放 batch；
- Linux：同一 frozen smoke 的 before/after stage trace，报告吞吐、GPU activity、显存、barrier 与 overlap，不用 Windows 结果替代。

## 7. Wrong vs Correct

```python
# 错：host worker读取GPU请求并创建reference session。
asset = int(asset_index.cuda().item())
session = create_reference_backend().open(plan, device="cuda:0")

# 对：rank owner保留设备所有权；worker只处理可序列化miss描述。
request = HostRequest(logical_id, FilePayload(path, expected_sha256), provenance)
host.submit(request)
lease = residency.acquire(asset_key, decoded.nbytes, upload_on_rank_stream)
```

```python
# 错：checkpoint时队列cursor已经前进但batch尚未训练。
save_checkpoint(data_session.state_dict())

# 对：先排空并确认无active lease，再冻结cursor。
data_session.drain()
save_checkpoint(data_session.state_dict())
```

```python
# 错：把执行优化旋钮混入随机样本identity，导致pack=1/2训练数据不同。
seed = hash((request.seed, request_index, reference_batch_steps))

# 对：执行计划只改变调度；logical request独立决定样本。
seed = hash((request.name, request.seed, request_index))
```
