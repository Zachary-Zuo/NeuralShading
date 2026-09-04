# 训练架构统一与跨平台清理：技术设计

## 1. 设计目标

本设计把当前散落在配置、producer、runner、method definition、launcher 和脚本中的训练职责整理为一条固定 pipeline：用户只选择 method、data、recipe 与 devices；resolver 生成可追溯的计划；每个 rank 打开独立的 online data session；公共 engine 执行固定 lifecycle；hook 负责 checkpoint、TensorBoard 和可视化 eval；method 只提供声明式数据需求、模型、objective、状态 codec 与部署 compiler。

重构保留既有 source/reference/scattering/package 语义，不建立 `FalcorMaterial / ReferenceMaterial / NeuralMaterial` 混合继承树，也不把 `LayerStackIR` 或任一 backend-specific `ScatteringState` 提升为公共 GT 表示。

## 2. 依赖方向与目录边界

```text
core contracts
   ↑
source_materials ──→ references
   │                    │
   └────────→ data.online ←──── method data facet
                         │
method plugin ───────────┼──→ learning.training engine
   │                     │              │
   └→ deployment compiler              ├→ checkpoint / TensorBoard
                                      └→ visual-eval job spool
                                                     │
                                      Windows capture worker → viewer
```

目标物理结构：

```text
src/ncls/
  core/                         # 保持公共 scattering/source identity 合同
  source_materials/             # 保持原生 source family
  references/                   # 保持 reference plan/backend/session
  data/
    contracts.py                # DataRequirement、typed routes/batches
    plan.py                     # TrainingDataDefinition/DataExecutionPlan
    session.py                  # OnlineDataSession 生命周期与 cursor
    pipeline.py                 # host/reference/ready queue 调度
    residency.py                # 按字节预算的 GPU residency 与 lease
    tracing.py                  # stage/queue/transfer/barrier trace
  learning/
    methods/
      contracts.py              # MethodPlugin 各 facet 协议
      registry.py               # 显式、fail-closed registry
      nvidia/                   # definition/model/data/objective/compiler
      metal/                    # definition/model/data/objective/compiler
      common/                   # 真正跨方法复用的网络/loss/runtime 工具
    training/
      config.py                 # YAML user config 与 resolved plan
      launch.py                 # launcher resolver、ExecutionContext
      engine.py                 # 唯一固定 lifecycle
      events.py                 # typed read-only events
      checkpoint.py             # 新 checkpoint 与 resume
      legacy_checkpoint.py      # v4 只读 importer
      hooks/                    # progress/metrics/TensorBoard/checkpoint/eval
    evaluation/                 # numerical validate，名称不再与 visual eval 混用
  visual_eval/
    contracts.py                # request/result/status schema
    spool.py                    # 文件队列、claim、幂等与原子发布
    worker.py                   # Windows viewer executor
```

迁移时可先在旧目录旁建立新模块，再逐调用方切换；最终删除 `learning/producer.py`、旧大一统 `method.py`、旧 runner/CLI reader 和已经迁移的重复实现。不能只增加 facade 后永久保留两套正式路径。

## 3. 配置、命名与 resolved plan

### 3.1 用户 YAML

手写配置只出现稳定、简短的选择 key：

```yaml
compose:
  method: metal
  data: mdl-metal-full
  recipe: formal

execution:
  devices: [0, 1]
  num_workers: 8
  host_prefetch: 4
  ready_batches: 4
  reference_batch_steps: 4
  residency:
    budget_mib: 8192

hooks:
  tensorboard:
    enabled: true
  visual_eval:
    interval_steps: 5000
    reference_spp: 1024
```

以上数值只是 schema 示例，不是跨机器默认值或性能门。正式 preset 的执行参数由目标机 preflight/profile 冻结；observed GPU utilization、吞吐和显存只进入报告。

组合顺序固定为 `base → method → data → recipe → 当前 YAML 显式字段 → 白名单 CLI override`。mapping 深合并，list 整体替换；未知 key、重复 component、循环 include、类型错误和不兼容 method/data 立即失败。YAML 使用 `PyYAML.safe_load`，依赖同步加入 `environment.yml` 与 `pyproject.toml`。

`ncls train CONFIG --devices 0` 直接单进程；`--devices 0,1` 覆盖 YAML 中唯一允许从 CLI 修改的设备选择并自动选择 launcher。CLI override 与最终来源链进入 run manifest。

### 3.2 ResolvedTrainingPlan

YAML 不直接成为 checkpoint identity。resolver 生成严格 typed、不可变的 `ResolvedTrainingPlan`，至少包含：

- method selector、descriptor identity 与各 facet implementation identity；
- 展开后的 source locators/snapshot IDs、reference plan/query recipe、data requirement/adapter identity；
- 完整 phase graph、objective/optimizer/precision/cadence；
- data execution policy、device topology、rank partition recipe；
- hook 与 visual-eval policy；
- 每个 YAML fragment 的 canonical path/hash 和白名单 override；
- schema 采用分离字段，例如 `{format_name: ncls.training-plan, format_version: 1}`。

新设计的用户 key 与 Python model 类名不含 `@`：公开 method key 为 `nvidia`、`metal`，类名收敛为 `NvidiaModel`、`MetalModel`。既有 source/reference/package ABI 中的版本 identity 不做无意义全仓改写；它们只存在于 resolved plan、manifest、checkpoint 和错误诊断，不要求用户手写。

## 4. Method plugin 合同

一种产品方法仍只注册一次，但注册对象是小型 facet 聚合，而不是新的 God object：

```python
@dataclass(frozen=True)
class MethodPlugin:
    key: str
    descriptor: MethodDescriptor
    model_factory: ModelFactory
    data: MethodDataFacet
    objective: ObjectiveFacet
    lifecycle: LifecycleFacet
    checkpoint: CheckpointCodec
    deployment: DeploymentCompiler
```

- `model_factory` 只创建 trainable model，不打开 source/reference/data 资源。
- `MethodDataFacet.requirements()` 声明所需 route、typed field、source/query/adaptation recipe；其 adapter/transform 只解释 method-local 输入，不创建 worker、queue 或 GPU scheduler。
- `objective` 消费 descriptor 声明的 typed batch，并返回 scalar loss、component outputs 和 metrics。
- `lifecycle` 只处理显式 phase transition/parameter ownership，不运行训练循环。
- `checkpoint` 定义 model state 的严格字段/shape/dtype；optimizer、RNG 与 data cursor 由公共 checkpoint envelope 保存。
- `deployment` 继续产出 program/asset/instance，公共 writer/viewer 不识别 method key。

NVIDIA 与 Metal 先通过兼容 facet 包装现有实现，再逐文件拆分；最终 registry 不允许 import-time 自动扫描、捕获异常后跳过或按函数签名过滤未知参数。

## 5. OnlineDataModule 与通用数据管道

### 5.1 三层对象

```text
TrainingDataDefinition  # 无设备资源的声明，可进入 resolved plan
        ↓ resolve(method requirement + source/reference capability)
DataExecutionPlan       # stage DAG、资源预算、rank 分片、batch schema
        ↓ open(ExecutionContext)
OnlineDataSession       # 每 rank 的实际资源、cursor、queue、lease
```

`OnlineDataSession` 提供 `next(step, phase)`、`drain(boundary)`、`state_dict()`、`load_state_dict()`、`profile_snapshot()` 和 `close()`。它不拥有 optimizer、model 或 phase loop；training engine 不识别 source family 或 method 名称。

### 5.2 Stage 与 worker 语义

```text
logical request/counter RNG
  → host asset locate/read/decode
  → pinned staging / residency materialize
  → GPU query + method adaptation
  → reference dispatch + invalid compaction/top-up
  → GPU-resident ready batch ring
  → model consumer
```

- `num_workers` 只控制可序列化、无 CUDA/Falcor owner 的 host process。`0` 是同步 characterization baseline；大于零启用 bounded worker pool、persistent worker、明确 start method、health/error channel 和可取消 task。
- `host_prefetch`、`ready_batches`、`reference_batch_steps`、reference slot/in-flight depth、transfer stream 与 residency budget 是不同资源轴，不复用一个含混的 `prefetch_depth`。
- producer completion 可以乱序，consumer 必须按 logical request ID 经过 reorder buffer 输出；seed/selection 在请求创建时冻结，不由完成时间决定。
- phase、validation、checkpoint、stop 和 close 都是 drain boundary。checkpoint 只保存已消费 cursor；未消费工作取消或排空，绝不让 cursor 领先 model step。
- worker 异常携带 stage/request/rank provenance 回传主进程；不能只打印、返回空 batch 或静默重启。

### 5.3 GPU residency

`GpuResidencyManager` 以 `(resource_identity, representation, device)` 为键，以实际 allocated bytes 为预算，使用 LRU + refcount/lease：

- 活跃 lease 不得驱逐；资源大于预算时在 dispatch 前失败并报告 largest resources，不静默切到逐 step CPU readback。
- source payload、canonical decoded mip、typed metadata table、reference execution group 与 ready batch 分别计量，避免一个条目数同时表示 KB tile 和 GB texture。
- immutable asset 在 host 只因 cache miss 读取；使用 pinned staging 和 non-blocking H2D。GPU 上进行 transfer、sRGB decode、normal renormalization、mip/gather 的实现必须保持现有 source semantics。
- Metal 的 `asset_index/uv/mip` 不再逐 step回读 CPU。method data facet 提供 GPU-native asset sampler，直接消费 device request；活跃 group 的资源按 64-step locality 预取并复用。
- 不强制 Falcor 与 Torch 零拷贝共享同一 texture。公共 capability 可声明 shared resource view；不支持时允许各 backend 拥有受预算约束的表示，但不得复制出无界 cache。

Metal registry 139 个去重 payload 的原生基准约 4.78 GiB，而全量 float32 mip 约 26.17 GiB，所以 cache 必须基于活跃 group 和字节预算，不能默认全量预载。

### 5.4 Reference scheduler 与 GPU 并发

backend 显式报告：

```text
synchronization = global | stream-fence
supports_async_submit
maximum_safe_slots
supports_shared_asset_view
```

- `global`：当前 Linux/Vulkan 路径。Falcor 8.0 的 interop 会 `cudaDeviceSynchronize()` / `submit(true)`，禁止宣称同设备 reference/model overlap。scheduler 在同一 execution group 内把多个 logical step 按原顺序 pack 成一次较大 dispatch，再切分到 GPU batch ring，以显存换取更少的 barrier、分配和 API transition。
- `stream-fence`：当前 D3D12 可通过 external semaphore 建立 device-side dependency。只有该 capability 下才允许 reference queue 与 model CUDA stream 使用双/三缓冲 overlap。
- 独立 reference GPU/P2P 是保留扩展点，不是本任务验收项；它将来只能作为另一种 execution capability，不改变 method/data contract 或默认 DDP world size。

packed dispatch 必须保持 logical step 的 RNG 消费、route identity、invalid top-up 和 provenance。`reference_batch_steps=1` 是语义基线；大于一的次序/数值/恢复一致性通过测试证明，并进入 resolved execution identity。

### 5.5 可观测性

每个 stage 发布统一 trace：host work、cache hit/miss/evict、H2D/P2P bytes、allocation bytes/count、reference submit/wait、barrier count、queue depth/wait、consumer starvation、forward/backward/optimizer、显存 allocated/reserved/外部占用。普通 step 不为 profile 增加同步；只在配置 cadence 聚合。

Linux before/after 使用相同 source/query/model/batch/work-unit 合同。吞吐、GPU activity、显存和时间是 observed report；结构验收是 hot path host readback 为零、资源有界、barrier 可解释、timeline 确有至少一类 host/data 与 model overlap。

## 6. 固定 TrainingEngine lifecycle

```text
resolve → preflight → launch
  → build ExecutionContext
  → open data session / model / optimizer / hooks
  → optional new-checkpoint resume
  → for phase in plan:
       phase transition
       for logical step:
         acquire ready typed batches
         forward → finite check → backward → DDP reduce → optimizer/scheduler
         publish StepCompleted
         execute cadence hooks
       drain phase boundary
  → final numerical validate
  → final checkpoint / summary
  → close hooks / data / distributed context
```

- engine 只解释 typed plan 与协议，不含 `if method == ...`、source family、OS 或固定 phase 名称。
- DDP 每 rank 拥有 model、data session、reference backend 和 deterministic rank shard；rank 0 负责 durable checkpoint/artifact/TensorBoard，metrics 使用声明的 reduce policy。
- phase transition、validation、checkpoint 和 visual-eval snapshot 都在已 drain 的一致 step 边界发生。
- interrupt/failure 逆序关闭，后台错误先传播再决定 checkpoint；不能把 partially advanced cursor 写成成功状态。

## 7. Launcher 与跨平台

`ncls train` 的 bootstrap 在导入 Torch/Falcor 和创建 GPU device 前只解析最小 YAML/execution 字段并执行 capability preflight：

- 一个 device：当前进程建立 `ExecutionContext`，method/data/engine 始终只看 local `cuda:0` 与 logical device identity。
- 多个 device + Linux：outer process 用同一 Python 环境 re-exec `torch.distributed.run`/NCCL；每个 rank 映射一个物理 GPU，Falcor Vulkan、Torch 与 SlangPy 一致绑定本 rank local device。
- 多个 device + Windows：在构造 GPU/runtime 前 fail closed，报告该配置只支持 Linux/NCCL；不自动回退单卡或 Gloo。
- shell/PowerShell launcher 只设置已锁定 Falcor build 与环境，不再拥有“单卡还是多卡”的产品决策。

method、data facet 与 engine 不读取 `sys.platform`、`CUDA_VISIBLE_DEVICES` 或物理 adapter index；这些只存在于 launcher/capability adapter 和 run manifest。

## 8. Hook、TensorBoard 与评测命名

### 8.1 Typed event

engine 发布不可变事件：`RunStarted`、`PhaseStarted`、`StepCompleted`、`ValidationCompleted`、`CheckpointCommitted`、`VisualEvalRequested/Completed/Failed`、`RunFailed`、`RunClosed`。hook 只消费事件/只读 snapshot，不反向驱动 optimizer 或向 method 请求私有字段。

每个 hook 注册失败策略：checkpoint 等一致性 hook 为 `fatal`；TensorBoard/visual-eval 等诊断 sink 的单次外部失败不回滚 optimizer，但必须生成 durable status、告警和最终未完成摘要。所有异步 sink 都有有界队列和错误回传；不允许后台线程只 `print` 后丢失数据。

### 8.2 TensorBoard

- 仅 rank 0 创建 writer；global step 来自 checkpoint 恢复后的 engine cursor。
- 稳定 tag 覆盖 loss/objective、learning rate、gradient/update、吞吐、stage timing、queue/cache、reference profile、显存和 visual-eval status/image。
- scalar/status 使用可靠 FIFO；图像来自 durable visual-eval result，不依赖训练 tensor 的悬空 CUDA 生命周期。
- flush/close 等待队列并传播错误；resume 不覆盖或倒退既有 step。

### 8.3 `validate` 与 `eval`

- `ncls validate`：数值 validation，复用冻结 source/reference/query recipe，输出指标；不称为用户所说的可视化 eval。
- `ncls eval`：手工或 cadence 触发 visual-eval job，比较同一 source/camera/lighting/seed 下的 reference 与 neural。
- `ncls export`：按 readiness 产生 formal 或显式 diagnostic package。

## 9. 异步 visual eval

Linux rank 0 在 cadence safe point 保存不可变 evaluation snapshot，并原子发布 `VisualEvalRequest`：

```text
run/step + method/checkpoint identity
source locator +完整 typed parameter state
camera + lighting + renderer identity
probe seed/selection identity
reference_spp = 1024
diagnostic/formal readiness label
```

probe RNG 与 training/validation stream 完全独立，`probe_id = hash(run identity, cadence index, visual seed)`；resume 对同一 probe 幂等，不重复生成不同随机视角。

文件 spool 位于 `artifacts/`，状态通过同 filesystem 内的原子 rename/manifest 转换：`pending → claimed → completed | failed`。Windows worker：

1. 校验 request/snapshot/code/source identity；
2. 用 method deployment facet 生成显式 diagnostic 或 formal package；
3. 调用现有 Windows/D3D12 viewer headless capture；
4. 等 reference PT slot 达到 1024 spp，并执行同相机/光照下的 neural deferred slot；manifest 必须标记 `training-diagnostic`、各自 mode 与 target，不能伪装成 matched spp；需要检查 neural `sample/pdf` 时可显式切换到有界低 spp path tracing；
5. 原子发布两个线性 EXR、difference、display PNG、capture manifest 与 result status。

rank 0 collector 轮询已完成结果并写 TensorBoard；训练结束后可用同一 collector 命令补写迟到结果。worker 不在线或 job 失败不改变 optimizer/checkpoint；queue 达到配置上限时生成 `skipped-capacity` 状态而不是无限占用磁盘。visual eval 始终标为训练诊断，不成为 formal deployment 或 checkpoint 选择证据。

训练 cadence 的 neural 默认使用 deterministic deferred evaluator，不执行 neural path tracing。它与 1024 spp reference 的 difference 是训练期外观诊断，不声称是 matched-integrator 误差；可选低 spp neural path tracing 与双 slot 1024 spp 都保留为显式、低频的深度检查入口，不能由普通 cadence 自动触发。

本任务不新增 Linux/Vulkan viewer。

## 10. Checkpoint 与兼容

新训练 checkpoint 使用新 schema，只允许相同 resolved plan/method/data execution identity 恢复。它保存：model state、phase-local optimizer/scheduler/scaler、engine cursor、每 rank RNG/data consumed cursor、coverage、hook cadence cursor 和已发布 visual-eval probe IDs。pending batch 不跨 checkpoint。

旧兼容仅存在于 `LegacyCheckpointV4Importer`：

- 要求原 `.pt` 与 SHA-256 sidecar，完整执行旧 v4/schema/method/component/readiness 校验；
- 输出不可变 `EvaluationSnapshot`，可供 `validate`、符合旧 readiness 的 diagnostic/formal `export` 与 `eval`；
- 丢弃 optimizer、scheduler、query cursor 和 training RNG，不暴露 `resume`；
- 旧 `method_key` 映射封闭在 importer，不注册全局 alias；未知 identity fail closed；
- 新 `train --resume` 收到 v4 时明确报错，不尝试 shape-based 或部分恢复。

旧 JSON config reader、`ncls learn train/evaluate`、converter、alias 和 fallback 在新 YAML parity 完成后删除。

## 11. 迁移与回滚

- 每个阶段先加 characterization/contract test，再切一个调用面，最后删除该阶段旧路径；不在一次提交中同时移动全部文件和改变语义。
- 同步 `num_workers=0`、`reference_batch_steps=1` 是 data plane rollback baseline；性能模式失败时显式切回基线做诊断，但正式配置不得静默降级。
- NVIDIA 先迁移以验证通用合同，Metal 再迁移并执行 GPU residency/packed dispatch；两者都通过后才能删除旧 producer/method/runner。
- visual-eval worker 与训练通过 versioned artifact 解耦；worker 部署失败可暂停 job 消费，不需要回退 engine。
- 不修改 `external/Falcor`。如未来需要 Vulkan shared fence，另建上游补丁任务并按 repository policy 保存显式 patch/application script。
- 现有工作区未提交改动视为用户所有；实现按文件检查冲突，不覆盖无关改动。

## 12. 主要风险与处理

| 风险 | 处理 |
|---|---|
| packed dispatch 改变 RNG/invalid 补采顺序 | logical request 先冻结 seed/cursor；基线与 packed 做逐 step identity/数值/resume 测试；改变 recipe 时拒绝跨 identity resume |
| GPU cache OOM 或驱逐活跃资源 | 字节预算 + lease；preflight largest active group；无可驱逐项时 fail closed，不静默回 CPU hot path |
| Linux 全设备同步误伤并发 | capability 为 `global` 时禁止 producer/model 同卡并发；只做合并 dispatch、驻留与 host overlap |
| DDP rank 数据重复或 checkpoint 游标不一致 | rank-strided deterministic request identity；checkpoint 前全 rank drain/barrier；逐 rank cursor 保存 |
| TensorBoard/worker 后台错误丢失 | durable status + error channel；flush/close 汇总；visual failure 非 fatal 但必须可见 |
| legacy importer 重新形成旧架构入口 | 只返回 evaluation snapshot，无 optimizer/cursor API；static test 禁止 train import legacy adapter |
| 大范围移动造成长期双路径 | implement 阶段设置删除门；最终 static scan 拒绝旧 CLI/config reader/producer 和 method 专用分支 |
