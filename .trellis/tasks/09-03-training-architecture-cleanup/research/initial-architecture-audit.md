# 训练架构初步审计

## 1. 结论摘要

当前工程不是“没有统一架构”，而是已经建立了正确的语义合同，但这些合同在代码组织和用户入口中没有形成同样清晰的边界：

1. source、reference、neural runtime 已经通过 `SourceSnapshot`、`ReferenceExecutionPlan@1`、typed batch 和 `ScatteringPackage@2` 保持了原生语义与公共执行合同；这部分应保留。
2. `learning` 同时容纳在线数据源、method/source adaptation、训练引擎、具体模型、目标函数、asset cook、Slang runtime 和部署 compiler，职责跨度过大。
3. `OnlineTrainingProducer` 与 `MethodDefinition` 都承担了多类生命周期；`TrainingRunner` 虽然已经是唯一 runner，却把 phase、DDP、prefetch、validation、metrics、checkpoint 和审计集中在一个 1249 行实现中。
4. 人工配置直接持久化完整 resolved plan。内部 schema/recipe/ABI identity 大量带 `@1/@2`，且 full-cohort Metal 配置将 692 个 locator 内联为 7342 行、约 348 KB；可复现性信息与用户选择面没有分层。
5. Linux 多卡已经有正确的 Falcor/Torch 物理卡映射雏形，但只能从 shell wrapper 进入；Windows launcher 没有同等入口，`ncls learn train` 也不会按 GPU 列表自动选择单卡或多卡。

因此理想重构不是推倒公共 scattering/source/reference 合同，而是把“声明式选择 → resolved plan → 资源会话 → 固定训练引擎 → checkpoint/compiler”拆成可组合层，并让内部版本 identity 留在 resolved manifest/checkpoint 中，不再要求用户手写。

## 2. NeuralShading 当前调用链

### 2.1 控制流

```text
ncls learn train <json> <output>
  -> TrainingConfig.load()
  -> get_method(method_key)
  -> OnlineTrainingProducer(definition, config)
  -> TrainingRunner(definition, producer, config).run()
  -> TrainingCheckpoint@4
  -> MethodDefinition.compile_program/asset/instance()
  -> ScatteringPackage@2
```

证据：

- CLI 同时负责 DDP setup、config、producer、runner、metrics、summary、review 和 checkpoint 写入：`src/ncls/cli.py:92`、`src/ncls/cli.py:244`。
- typed config 是单个严格 JSON 对象，直接含 source locator、online query、model context 和完整 phase graph：`src/ncls/learning/training/config.py:204`。
- runner 是唯一 phase graph orchestration，这一点正确；但其 `run()` 之外还内置 DDP gradient、prefetch、validation、metrics、audit、checkpoint 和 profile 归并：`src/ncls/learning/training/runner.py:186`、`:743`。

### 2.2 数据流

```text
native locator
  -> SourceFamilyDefinition.load_snapshot()
  -> SourceSnapshot
  -> canonical ReferenceProgramDefinition
  -> ReferenceExecutionPlan
  -> ReferenceBackendCapability.open()
  -> ReferenceBackendSession
  -> typed online batch
  -> method objective/model
```

这条链已经正确区分：

- source 是“材质原生语义”；
- reference 是“该 source family 的权威 GT 程序”；
- Falcor/D3D12/Vulkan 是 reference program 的执行 backend；
- neural method 是近似表示及其 compiler，不是另一种 source。

后续不应建立 `FalcorMaterial / ReferenceMaterial / NeuralMaterial` 这类混合继承树。更合适的是保留三个正交维度，再提供一个 facade 负责把 source snapshot、reference binding 和 compiled neural binding 组合成同一 `prepare/evaluate/sample/pdf` 消费面。

### 2.3 已确认的结构问题

| 区域 | 当前证据 | 问题本质 |
|---|---|---|
| 配置 | `TrainingConfig@4` 把全部 resolved identity 和 phase graph 放在手写 JSON 中 | 人工选择面与可复现 manifest 混在一起 |
| full cohort | 两个 Metal Linux config 各 7342 行、约 348 KB、内联 692 个 locator | source set 应由独立 registry/query 解析，不应复制进 run config |
| online data | `OnlineTrainingProducer` 直接解析 family、扩展 state、编 plan、开 backend、选 group、生成三种 batch、保存 cursor | dataset/data session、reference runtime 和 sampling policy 未分层 |
| adaptation | `create_method_source_adapter()` 通过私有 `(method_key,family,version)` 映射选择 | 扩展点没有进入 method plugin 的显式声明/验证 |
| method | `MethodDefinition` 同时拥有 model/objective/checkpoint state、phase transition、package compiler 和 edit classification | 统一发现是优点，但 definition 已成为多 facet 的 God object |
| runner | 唯一 runner 已成立，但文件 1249 行 | 应保留固定状态机，拆出 distribution、checkpoint、metric/event、validation policy 服务 |
| 平台 | `ReferenceBackendCapability` 已封装 manifest、Falcor import/device 和平台 toolchain | backend 内部边界较好，应继续保留 |
| launcher | Linux `--gpus` wrapper + `ddp_worker`，Windows PowerShell 只有单进程；CLI 固定 `backend="nccl"` | 用户入口和平台能力没有统一为 typed execution plan |
| 命名 | 产品 `method_key` 已是 `nvidia-neural-appearance`、`metal-fused-neural-material` | `@` 主要来自内部 schema/recipe/ABI/correspondence；真正要清理的是泄漏，不是删除版本身份 |
| 代码组织 | `learning` 中最大文件包含 method、runner、reference-facing producer、asset cook 和 runtime compiler | 物理目录没有表达既有规范中的依赖方向 |

### 2.4 Linux 长训的调度证据

仓库外运行产物 `artifacts/metal-linux-training/long/checkpoint.metrics.jsonl` 提供了 2101 个带 profile 的 training 记录。它不是完整 GPU timeline，不能直接换算为 GPU 利用率，但能确认当前共享调度的主要串行段：

| 阶段 | 全部记录 median | 全部记录 p90 |
|---|---:|---:|
| `batch_prepare_wall_seconds` | 1.324 s | 2.263 s |
| `forward_gpu_seconds` | 0.115 s | 0.153 s |
| `backward_gpu_seconds` | 0.136 s | 0.183 s |
| `optimizer_gpu_seconds` | 0.010 s | 0.011 s |

- phase 0 的 `prepare / (forward + backward + optimizer)` median 比值约为 `4.84`；phase 1 约为 `3.38`。
- `TrainingRunner` 的所谓 prefetch queue 由主线程连续调用 `_prepare_step()` 填充，然后才消费训练；没有 executor/worker 与 model compute 并发。
- `ReferenceBackendSession._dispatch()` 在返回 batch 前依次等待 CUDA、执行 Falcor、再等待 Falcor，当前 slot/lease 数量没有自动形成跨 step overlap。
- `batch_prepare` 包含 reference GPU 工作，因此不能把整段都称为“GPU idle”。下一阶段必须把 host prepare、reference submit/wait、batch adaptation、queue wait 与 model stream 分别打点，再用 Linux profiler 证明 overlap，而不是只比较总 step time。

据此，公共 data pipeline 不能只包一层 `torch.utils.data.DataLoader(num_workers=N)`。它需要把资源模型拆为：host worker pool、每 rank 唯一的 reference session owner、受 slot/lease 约束的 reference scheduler、ready-batch bounded queue，以及 model consumer；同步模式是可复现基线，异步模式再按 capability 开启。

### 2.5 GPU residency 与 Linux interop 边界

- `MdlMetalNativeAssetCollection.sample_local_patches()` 当前把 GPU `asset_index`、`uv`、`mip_level` 逐 batch 转为 CPU NumPy，再从 memmap 采样/解码，最后用 `torch.as_tensor(..., device=device)` 上传完整 patch。这是明确的 GPU→CPU→GPU hot-path 往返：`src/ncls/learning/mdl_metal_assets.py:403-466`。
- reference query 自身没有 host response readback：输入经 Falcor shared buffer `from_torch()`，输出用 `to_torch()` 映射为 CUDA tensor；但 `_rows()` 每次创建/拼接多个临时 tensor，随后 `_dispatch()` 每个操作执行一次 CUDA→Falcor 与 Falcor→CUDA 同步：`src/ncls/references/query.py:553-718`。
- Falcor 8.0 的 `CopyContext::waitForCuda/WaitForFalcor` 在 D3D12 使用 external semaphore，属于 device-side stream/queue 次序；在 Vulkan 分支则分别调用 `cuda_utils::deviceSynchronize()` 与 `submit(true)`，是全局/阻塞同步：`external/Falcor/Source/Falcor/Core/API/CopyContext.cpp:110-133`。因此 Linux 同设备 producer thread 或额外 CUDA stream 不会自然得到 reference/model overlap。
- 现有 reference group session 已有至少两个 shared-buffer slot，native asset 也有 lease/LRU 雏形；但前者在同步 dispatch 返回前已等待完成，后者主要按条目数而非字节控制，不能直接把“更多 slot/cache entries”等价为安全的显存预算。
- registry 静态估算显示，52 个 Metal texture set 的 246 个 slot 引用可去重为 139 个 payload；native payload 约 4.78 GiB，而统一解码为 float32 并计入 mip 后约 26.17 GiB。即使训练时 `nvidia-smi` 低于 5 GiB，也不适合全量 decoded preload；更合理的是 native-format/typed GPU cache、按字节预算和 64-step group locality 预取。

推荐的 Linux 优化顺序是：

1. 把 Metal patch request/decode/gather 改为 GPU-native 或“host worker + pinned staging + GPU residency cache”，禁止每 step 元数据回读；
2. 预分配 route-specific GPU batch arena/ring，缓存 group indices、静态 meta/template 与 source typed tables，减少 allocator 和 GPU-to-GPU 临时复制；
3. 在同一 execution group 的冻结 block 内，把多个 logical step 的 reference query 合成较大 dispatch，再按 step 切分 ready batch，以现有显存余量换取更少的 Vulkan global barrier；逻辑 request/RNG 顺序保持可恢复并进入新 recipe identity；
4. host decode/prefetch 与 model compute 并行；same-device reference/model overlap 仅在 backend 报告 stream-fence capability 时启用；
5. 可选 dedicated-reference-device/P2P 作为同一 scheduler 的未来 capability，不作为默认 DDP 语义或本任务硬性交付。

## 3. VRFrameGeneration 对照

| 设计 | 处理 | 原因 |
|---|---|---|
| 一个入口根据 GPU 列表长度选择 `SingleLauncher` / `TorchrunLauncher`（`src/main.py:121`） | 调整后采用 | 入口语义清晰；NeuralShading 需要先解析设备、再在导入 Torch/Falcor 前启动 worker，并保持物理 GPU 到 Falcor/Torch local device 的一致映射 |
| `_defaults -> method -> variant` 的 YAML 组合（`src/utils/config_loader.py:149`） | 调整后采用 | 适合减少重复；NeuralShading 需要 typed schema、确定性合并和 resolved manifest，不能依赖方法专用的 nested reset hack |
| 方法目录内并置 `dataset.py` 与 `model.py` | 采用概念 | 新方法的局部代码容易发现；公共 online reference/data session 与通用 asset/query recipe 仍须独立复用 |
| dataset registry 与 model registry 分离，通过 factory 创建 | 调整后采用 | 应使用显式、fail-closed 的 component spec/factory；不按任意 Python 构造签名过滤未知参数 |
| train 只装配 DataModule + TrainerCoordinator（`src/tasks/train.py:306-321`） | 采用概念 | 入口只负责解析、resolve、launch 和 run，固定 lifecycle 由 engine 管理 |
| 自动扫描所有 method 子包并捕获异常后跳过 | 不采用 | 产品方法缺依赖或注册失败必须在 discovery/preflight 阶段明确失败，不能静默消失 |
| 2282 行 DataModule、1360 行 base model、968 行 TrainerCoordinator | 不采用 | 这是另一种职责集中；只借鉴边界，不复制体量和领域特例 |
| 多卡 launcher 默认自动重试 | 不采用默认行为 | 失败重启会改变错误可见性与 resume 语义；只能作为显式作业策略，不能属于普通 train lifecycle |
| `method:variant` 字符串与任意 CLI kwargs | 不直接采用 | 简洁选择 key 可以保留，但 override 必须由 schema 限定并进入 resolved config identity |
| `num_workers`、`prefetch_factor`、`persistent_workers` 与 worker monitoring | 调整后采用 | `num_workers` 只表示可复制的 host stage；Falcor/GPU owner、reference in-flight depth 和 CUDA stream 使用独立配置与 capability |
| bounded queue、consumer wait 统计、后台异常检查与清理 | 采用概念 | NeuralShading 需要 fail-fast 异常传播、可审计 drain/cancel 和 deterministic resume，不能沿用后台线程只打印错误的弱语义 |
| 独立 data stream 与 GPU prefetch queue | 调整后采用 | 先证明 reference backend/session 是否允许与 model stream 并发，并把 slot/lease 和显存上限写入调度合同；不能仅凭 CUDA stream 名义宣称 overlap |
| `TrainingInfoHook` / `TrainingVisualHook` + 异步 TensorBoard writer | 调整后采用 | lifecycle event、rank-0 writer、stable global step、bounded/latest-image queue 值得复用；hook 不应反向调用 method 特有的 `export_tensorboard_images()`，而应消费 typed event/artifact |

## 4. 目标架构草案

目录名在 `design.md` 中冻结；这里先冻结职责方向。

```text
core contracts
  <- source families
  <- reference programs/backends

source set + online data plan
  -> TrainingDataSession (typed batches, leases, cursor, provenance)

method plugin
  -> ModelFactory
  -> DataAdapterFactory
  -> Objective/LifecycleDefinition
  -> CheckpointCodec
  -> DeploymentCompiler

resolved TrainingPlan
  -> ExecutionLauncher (single/distributed + platform capability)
  -> TrainingEngine (fixed lifecycle)
  -> events/checkpoint/evaluation
```

### 4.1 材质抽象

- `SourceFamilyDefinition` 继续拥有 native locator/snapshot/editor。
- `ReferenceProgramDefinition` 继续拥有 GT 的 runtime/material 编译；`ReferenceBackendCapability` 独占 Falcor 与 OS/toolchain。
- neural method 继续输出相同 scattering contract 的部署 program，但不冒充 source/reference。
- 新增面向上层的组合对象时，只组合 identity 与 binding，不统一 backend-specific `ScatteringState` 内存布局。

### 4.2 Online data

- 用户所说的 dataset 在本项目中应落为 `TrainingDataDefinition -> TrainingDataSession`，而不是传统磁盘 `torch.utils.data.Dataset`。
- definition 是可配置、无 GPU 资源的声明；session 在每 rank 打开 reference backend、native assets 和 RNG/cursor，产生 typed routes，并负责 lease/close/state_dict。
- source-state expansion、source set、query distribution、asset tile traversal、method/source adaptation 分成明确 component；公共 recipe 可跨 method 复用，专用 adapter 留在 method 目录。
- pipeline 配置至少区分 `host_workers`（可提供用户习惯的 `num_workers` alias）、`host_prefetch`、`reference_inflight`、`ready_batches` 与 `transfer_streams`；每项都有清晰 owner、内存上限和 `0/1/N` 语义。
- method 返回 typed `DataRequirement` 与可注册的 transform，不创建 thread/process/queue，也不决定 rank 分片；pipeline 根据 requirement、backend capability 与 execution context 生成 `DataExecutionPlan`。
- 每个 stage 发出统一 trace：enqueue/dequeue wait、queue depth、host prepare、reference submit/wait、adapt/transfer、consumer starvation、cancel/drain；性能优化建立在这些 trace 上。

### 4.3 Method plugin

- 继续维持“一种新方法只注册一个产品 definition”的用户心智，但 definition 改成显式聚合多个 facet，而不是由一个 800/1400 行类实现全部责任。
- runner 只看 trainable model、objective、phase lifecycle 和 typed data session；bundle/export 只看 checkpoint codec 与 deployment compiler。
- method 目录建议并置 `definition.py`、`model.py`、`data.py`、`objective.py`、`compiler.py`，公共网络、loss、asset codec 和 runtime 工具放独立共享模块。

### 4.4 配置与命名

手写 YAML 只引用简洁 key：

```yaml
method: metal
data: mdl-metal-full
recipe: formal
devices: [0, 1]
output: artifacts/metal/formal
```

loader 按 defaults/method/data/run overlay 解析为严格 typed `TrainingPlan`。resolved plan 才包含 `schema_version`、implementation hash、source snapshot IDs、reference/adapter/query identity、完整 phases 和 device topology，并随 checkpoint/artifact 保存。

- 人工选择 key 使用短的 lower-kebab 名称，不含 `@`。
- ABI/schema 版本不删除，改为结构化字段或只存在于 resolved manifest。
- 不允许同一字段同时存在显示名、选择 key 和带版本字符串三种相互替代的写法。

### 4.5 单卡、多卡与平台

- `ncls train <yaml> --gpus 0`：outer launcher 走单 worker。
- `ncls train <yaml> --gpus 0,1`：outer launcher 自动 re-exec/torchrun；每 rank 仍只看到 local `cuda:0`，Falcor 通过 execution context 得到对应物理 adapter。
- method、data adapter 和 engine 不读取 `sys.platform`、shell 变量或物理 GPU 序号；它们只消费 `ExecutionContext`。
- capability preflight 明确报告平台是否支持 requested topology。统一接口不等于伪造相同能力。

官方 PyTorch 文档仍将 Windows distributed 标为 prototype，并明确 Windows 不支持 NCCL；当前代码固定 NCCL。因此“Windows 多 GPU 正式支持”必须单独决定和验证，不能从 Linux launcher 推导。

### 4.6 Hook、TensorBoard 与可视化 eval

- `TrainingEngine` 发布 typed lifecycle event；checkpoint、JSONL、TensorBoard scalar、profile trace 和 visual-eval trigger 都是订阅者。hook 不能改变 optimizer/phase 控制流，失败策略在注册时明确为 fatal 或 diagnostic。
- TensorBoard writer 只在 rank 0 创建；scalar 进入有界 FIFO，图像进入按 tag 合并最新项的有界队列。后台写入错误必须回传 engine，不能仅 `print` 后继续造成“训练成功但日志缺失”。
- training validation 保留为数值合同，建议在命令/类型上改名为 `validate`；用户所说的 `eval` 是独立的 `VisualEvalRequest`，记录 checkpoint/global step、source state、camera、lighting、seed、renderer 和 1024 spp target。
- 当前权威 1024 spp comparison 只存在于 `NclsViewer` 的 Windows/D3D12 headless capture：两个 slot、相同 scene、各到 1024 spp，再导出 reference/neural/difference。Linux 部署只承载 Falcor/Vulkan online reference，并没有等价 viewer 可执行文件。
- 未完成 checkpoint 已有 `exact-diagnostic-evaluator-preview` 语义，但当前 `ncls learn export` 的正式路径和 viewer package readiness 仍有严格边界。训练期可视化若复用 viewer，必须把“只用于诊断的 snapshot/package”与 formal export 分开，并在图像、manifest 和 TensorBoard tag 中显式标记，不能提升为部署通过。
- 因此 visual eval 的执行位置不是实现细节：Windows worker 可直接复用权威 capture，但需要跨机 job/artifact transport；Linux 本地执行则意味着新增 portable/headless renderer 或定义较轻的 canonical probe，两者范围差异很大。
- 用户已接受异步 Windows worker：本任务冻结 `VisualEvalRequest -> diagnostic snapshot/package -> Windows capture -> VisualEvalResult -> TensorBoard collector`，不新增 Linux viewer。job 以内容 identity 幂等，训练只发布与采集，不因 worker 不在线而阻断 optimizer；队列上限、过期与失败必须可见。

## 5. 建议的任务拆分

该工作适合作为父任务，后续至少拆为四个可独立验收的实现子任务：

1. typed YAML composition、短 key 与 resolved manifest；
2. platform-neutral launcher、ExecutionContext 与单卡/多卡状态机；
3. TrainingDataDefinition/Session、source set/query recipe/adaptation 重组，以及 Linux stage trace/overlap 优化；
4. lifecycle event、TensorBoard sink 与 visual-eval job/artifact 合同；
5. method facet 拆分、NVIDIA/Metal 迁移、旧入口删除与全链路回归。

父任务保留整体依赖图、兼容决策、跨子任务验收和最终集成，不直接作为大爆炸式实现单元。

## 6. 已确认决策

- 已确认：多 GPU 正式路径为 Linux/NCCL；Windows 保持相同入口/config 语义，请求多卡时 capability fail-closed。
- 已确认：visual eval 使用异步 Windows/D3D12 capture worker；Linux 训练侧不新增 viewer。
- 已确认：旧 JSON/CLI 不兼容且不提供 converter；只保留 `TrainingCheckpoint@4` 的严格只读 importer，用于验证、符合原 readiness 的导出和 visual eval，不允许 resume。
