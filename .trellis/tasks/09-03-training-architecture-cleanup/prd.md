# 训练架构统一与跨平台清理

## Goal

全面审计并重构 NeuralShading 中以训练为中心的代码架构，使新增方法只需实现清晰、稳定、可复用的组件合同，再通过 YAML 完成拼装；训练入口、单卡/多卡调度、online reference query、跨平台执行和生命周期管理由公共框架统一承担。

这项工作的用户价值是降低当前架构的理解与维护成本，避免每增加一种方法就复制 runner、平台分支或方法专用胶水，同时保留源材质原生语义、在线 GT 和最终实时部署合同。

## 已确认的现状

- 项目既有边界要求继续成立：源材质族保留原生参数、图和资源语义；reference 是该 source family 的权威 GT；正式训练使用 GPU-resident online reference query，不引入持久化 batch/corpus；viewer 不依赖 PyTorch；实时 `prepare/evaluate/sample/pdf` 与 package 合同不被训练框架倒置。
- 代码审计确认当前产品 `method_key` 已经是不带版本后缀的 `nvidia-neural-appearance` 与 `metal-fused-neural-material`；可读性问题主要来自内部 schema/recipe/ABI/correspondence identity 直接泄漏到手写配置，而不是 method registry 本身必须用 `@`。
- 当前 Metal Linux smoke/long 配置各自内联 692 个 source locator，均为 7342 行、约 348 KB；现有生成脚本只能机械保持副本一致，尚未把 source set、method recipe、run budget 和 device topology 组合成简洁的用户配置。
- 当前多卡只由 Linux shell wrapper 启动，训练 CLI 本身没有 GPU 列表；Windows PowerShell launcher 只有单进程路径。训练进程固定初始化 NCCL，而官方 PyTorch 文档明确 Windows 不支持 NCCL，并仍将 Windows distributed 标为 prototype。
- 当前 Metal patch 路径每个 batch 都把 `asset_index/uv/mip` 从 GPU 读回 CPU，经 NumPy/memmap 随机访问与解码后再把 patch tensor 上传 GPU；这与“GPU-resident online training”的目标不一致，也是共享 pipeline 必须消除的主要往返。
- 当前 Falcor 8.0 在 Linux/Vulkan 下的 CUDA interop 使用全设备 `cudaDeviceSynchronize()` 与阻塞式 Falcor `submit(true)`；因此同一 GPU 上增加 CUDA stream 并不能直接让 reference dispatch 与 Torch model 安全 overlap。D3D12 可使用 external semaphore，目标调度必须由 backend concurrency capability 决定，不能假设平台等价。
- Metal registry 的 52 个 texture set 引用了 246 个 slot、139 个去重 payload；按 native payload 估算约 4.78 GiB，但全部解码为 float32 并包含 mip 后约 26.17 GiB。因此显存余量适合有界、按字节和访问局部性管理的 residency cache，不适合无条件预载全部资源。

## Requirements

### R1 现状与参考架构审计

- 以代码、测试、配置、文档和历史决策为证据，绘制 NeuralShading 当前训练数据流、控制流、注册/命名、平台适配和部署边界。
- 审计 `VRFrameGeneration` 中训练入口、GPU 选择、多卡启动、dataset/model/runner 分层、YAML 组装、registry 与生命周期设计，明确可复用的结构和不适用于材质编译的假设。
- 识别重复抽象、职责泄漏、方法专用分支、平台条件分支、模糊命名和迁移风险，不仅做目录移动或改名。

### R2 统一组件边界

- 定义 source material、reference execution、online query dataset/data source、method model、loss/optimization、training orchestration、checkpoint、evaluation、compiler/package 的职责和依赖方向。
- Falcor、其他 reference backend 与 neural method 通过保持各自原生语义的 typed contract 连接；公共抽象不得要求所有 source 归约为 LayerStack，也不得把 backend-specific `ScatteringState` 布局提升为公共接口。
- 公共结构不局限于 NVIDIA 或 Metal 等现有方法；方法专有逻辑必须有明确归属，并能通过注册的定义对象接入。

### R3 固定训练流程与 YAML 组装

- 形成唯一、固定的 train lifecycle；方法不得新增专用 runner、CLI、checkpoint、exporter 或磁盘 batch reader。
- dataset/data source 与训练 orchestration 分离，同时保留 GPU-resident online query、无 host readback 和 deterministic evaluation 等项目特性。
- YAML 只负责声明式选择和配置组件，不承载运行逻辑；配置解析后形成 typed、可验证的运行计划。
- 明确训练、验证、导出/编译各阶段的公共 hook，以及方法真正需要实现的最小接口集合。

### R4 单卡/多卡与跨平台

- 一个 GPU 的配置直接执行单进程单卡训练；多个 GPU 的配置自动使用统一的多卡路径，并明确进程模型、随机种子、采样划分、指标归并、checkpoint 写入和失败传播。
- 正式多 GPU 路径只支持 Linux/NCCL；Windows 保持同一入口与配置语义，但请求多个 GPU 时必须在创建 Torch/Falcor device 前由 capability fail closed，不在本任务中增加 Gloo fallback 或未经验证的 Windows collective backend。
- Windows/Linux 差异封装在平台或 backend adapter 内，训练 orchestration 与方法代码不得散落 OS 判断。
- 跨平台 online training 保持同一配置语义、batch schema 和方法接口；平台能力差异通过显式 capability/error 表达。

### R5 命名与发现

- 对用户可见的 model/method/config 名称定义简洁命名规则，将 ABI/schema/component 版本与显示名、选择 key 分离。
- 新主线的公开 method key 固定为 `nvidia`、`metal`，对应 concrete model 类名使用简洁的 `NvidiaModel`、`MetalModel`；新设计的 schema 使用独立 `format_name/format_version` 字段。既有 source/reference/package ABI identity 继续保留，但不进入用户手写选择面。
- 清点并迁移当前含 `@` 等不易读名称；兼容策略需在设计阶段基于实际调用面决定，避免静默接受拼写错误或永久保留重复别名。

### R6 渐进迁移与质量保护

- 制定可分阶段验证和回滚的迁移计划，优先建立合同与 characterization tests，再迁移现有方法，最后删除旧入口与重复实现。
- 迁移必须覆盖当前已注册方法、在线 reference query、checkpoint/resume、评测、MethodBundle/Slang 编译和 viewer 消费边界；具体部署验证强度按受影响范围决定。
- 新架构不读取旧 `TrainingConfig@4` JSON，也不保留旧 `ncls learn train/evaluate`、alias 或配置转换器；仓库内正式配置一次性迁移为 YAML/新 resolved plan 后删除旧入口。
- 只保留隔离的 `TrainingCheckpoint@4` 只读 importer：校验旧 hash/schema/component identity 后可构造验证/导出所需的 evaluation snapshot，用于数值 `validate`、diagnostic/formal package export 和 visual eval；它不得恢复 optimizer、query cursor 或继续训练，也不得把旧 method key 注册成新主线 alias。
- 不覆盖或回退当前工作区中与本任务无关的未提交改动。

### R7 通用 online data pipeline 与共享性能优化

- 设计符合 GPU-resident online reference query 的通用数据处理管道，吸收 `num_workers`、prefetch、bounded queue、backpressure、worker health、异常传播和可观测性等有用机制，但不把 Falcor/GPU session 错误地复制进传统磁盘 `torch.utils.data.Dataset` worker。
- 具体 method 只声明所需 typed data、source/query/adaptation recipe 与必要的 method-local transform；worker、队列、资源生命周期、CPU/GPU stage 调度、跨 rank 分片和消费顺序由公共 pipeline 负责。
- 区分可并行的 host asset I/O/decode/prepare、GPU reference dispatch、batch transform 与 model compute，并在不破坏 Falcor slot/lease、确定性 resume、phase/checkpoint/validation 边界的前提下形成实际 overlap。
- `num_workers=0` 提供同步基线；大于零时只并行允许离开 GPU/Falcor owner 的 stage。GPU in-flight depth/stream/slot 等能力使用独立配置，不把不同资源模型混为一个 `num_workers`。
- online hot path 禁止把 GPU 上的逐 step request metadata、UV、mip 或 reference result 回读 CPU 后再上传；只有 GPU residency cache miss 所需的磁盘/host decode 可以经过 pinned staging，并必须可与消费阶段独立观测。
- GPU 资源由公共 `ResidencyManager` 按字节预算、资源 identity、LRU/refcount 与 lease 管理，而不是只按条目数缓存。不可变 source payload、reference group runtime、typed metadata table、query/batch arena 尽量跨 step 复用；method 不自行维护另一套无界 cache。
- 公共 pipeline 使用预分配的 GPU-resident batch ring，并将 `ready_batches`、`reference_batch_steps`、`residency_budget`、transfer stream 与 reference slot/in-flight depth 分开配置；显存余量用于减少分配、复制、cache miss 和跨 API barrier，而不是自动扩大训练 batch 改变实验语义。
- reference scheduler 根据 backend concurrency capability 选择执行模式：支持 stream-level external fence 时允许同 GPU 双/三缓冲 overlap；Linux/Vulkan 当前只能采用安全的 global-sync 模式，通过多 logical step 合并 dispatch、group locality、驻留 cache 和 host pipeline overlap 摊薄 barrier。若以后提供独立 reference GPU 或 Vulkan shared fence，只通过新 capability 接入，不改 method/data contract。
- 在 Linux 代表性 online training profile 上记录优化前后的 stage wall time、queue wait、reference dispatch、model forward/backward、GPU activity 和吞吐，定位并消除共享 pipeline 造成的可避免空洞；观测数值用于报告和后续调优，不在缺少权威来源时编造固定利用率硬门。

### R8 通用训练 hook、TensorBoard 与可视化 eval

- 建立固定训练 lifecycle 的通用 hook/event 接口，hook 不得识别具体 method/source family，且 rank、同步、异常和 checkpoint 边界语义明确。
- 提供 TensorBoard hook，按配置记录训练/公共 validation loss、learning rate、分项 objective、gradient/update coverage、吞吐、stage timing、显存和 reference backend profile；只由 rank 0 写出，tag 命名稳定且 resume 不产生错误 step 回退。
- 保留并重新定义 `eval`：它不是验证数据集指标，而是按若干 training step 触发一次独立随机可视化 probe，同时渲染同一 source/camera/lighting 条件下的高样本 reference（固定 1024 spp）与当前 neural model，并保存可比较结果。cadence 默认用 deterministic deferred evaluator 呈现 neural 外观；需要检查 neural `sample/pdf` 与环境积分时，才显式选择有界低 spp path tracing。manifest 必须记录两侧 mode 与预算；双 1024 spp 只属于手工/低频深度检查，不能成为 cadence worker 的默认成本。
- 可视化 eval 使用独立、可恢复的 RNG/selection identity，不消费训练 query stream；每次保存 reference、neural、difference 与必要 provenance，并可写入 TensorBoard image panel。
- 已确认 visual eval 采用平台无关的异步 job：Linux 训练进程发布 immutable diagnostic snapshot/request，Windows/D3D12 capture worker 复用现有权威 1024 spp viewer harness；结果返回后由 rank 0 collector 写入 TensorBoard。worker 离线、积压或失败不得破坏 optimizer 状态，必须通过 job 状态与告警显式呈现。
- 数值 validation/evaluation 与可视化 eval 使用不同名称和合同，避免当前 `evaluate` 命令、训练期 validation loss 和用户所说的 render eval 继续混义。

## Acceptance Criteria

- [ ] AC1（需求交付；来源：用户要求“全面分析”）：任务研究材料给出 NeuralShading 当前架构图/调用链、责任矩阵和问题清单，并以文件与符号位置为证据。
- [ ] AC2（需求交付；来源：用户指定参考 VRFrameGeneration）：任务研究材料给出 VRFrameGeneration 对照表，逐项标明“直接采用 / 调整后采用 / 不采用”及原因。
- [ ] AC3（需求交付；来源：用户要求固定流程、YAML 拼装与通用封装）：`design.md` 定义稳定的目标分层、依赖方向、核心协议、typed configuration schema、单卡/多卡状态机、跨平台 adapter 边界及方法扩展点。
- [ ] AC4（需求交付；来源：用户要求 GPU 数量决定单卡/多卡且方法可拼装）：同一 train 入口和 YAML schema 能组装现有至少两种代表性 method；单 GPU 与多 GPU 仅改变设备/分布式执行配置，不改变 method 实现。
- [ ] AC5（理论/语义正确性；来源：用户要求 dataset/train 分离及项目 learning 合同）：dataset/data source 不拥有训练循环，runner 不识别具体 method/source family，Windows/Linux 分支不进入 method model 或通用训练循环。
- [ ] AC6（理论/语义正确性；来源：项目 online training 与 repository policy）：reference query 与 GT response 继续在 GPU 上产生和消费，不新增持久化训练 batch/corpus；source asset cache miss 可以进行有界 host I/O/decode，但不得演变为 reference response 的 host-staged 正式训练路径。
- [ ] AC7（需求交付；来源：用户要求简化 model 命名）：model/method 的公开选择 key 简洁且不含 `@`；版本信息保留在 descriptor/schema/ABI 元数据中，并有明确的旧配置迁移或拒绝策略。
- [ ] AC8（需求交付；来源：用户要求创建任务研究规划）：`implement.md` 将重构拆成可独立验证、可回滚的阶段，列出每阶段受影响文件、characterization/unit/integration 测试和最终清理门槛。
- [ ] AC9（理论/语义正确性；来源：项目统一 pipeline 合同）：重构完成后，现有 source/reference/native edit、checkpoint/resume、评测、package compile 与 viewer 消费的受影响合同均通过相应验证，且不存在方法专用 runner/CLI 或散落的平台判断。
- [ ] AC9a（需求交付；来源：用户确认只保留旧 checkpoint 读取）：代表性 `TrainingCheckpoint@4` 可经专用 importer 完成 hash/identity 校验并进入数值验证、符合原 readiness 的 package export 与 visual eval；将它传给新 `train --resume` 必须明确拒绝。旧 JSON config、旧 `ncls learn` 命令、全局 method alias 和 config converter 均不存在。
- [ ] AC10（需求交付；来源：用户要求通用 DataModule 式 pipeline 与共享 GPU 利用率优化）：至少 NVIDIA 与 Metal 通过同一 data pipeline 运行，method 仅提供声明式 data requirement/recipe/factory；`num_workers=0` 和大于零、bounded prefetch、backpressure、异常传播、清理与 deterministic resume 均有 characterization/integration 测试。
- [ ] AC11（需求交付；来源：用户要求提高 Linux 并行度/GPU 利用率）：冻结代表性 Linux online profile 并给出 before/after stage trace；公共管道能让至少一类 host/data preparation 与 GPU/model 工作实际 overlap，且不会跨越 lease、phase、validation 或 checkpoint 边界。GPU 利用率、吞吐和耗时作为 observed result 报告，不设置无来源的数值硬门。
- [ ] AC11a（理论/语义正确性；来源：用户要求避免反复 CPU/GPU 交换且合理利用显存）：代表性 Metal online hot path 不再逐 step 将 GPU request metadata/UV/mip 回读 CPU；GPU-resident source cache、batch ring 与 reference resource 的峰值均受可配置字节预算约束，cache miss、H2D/P2P bytes、分配次数和 barrier 次数进入 trace。
- [ ] AC11b（理论/语义正确性；来源：Falcor 8.0 Linux/Vulkan interop 约束）：global-sync backend 不宣称或测试伪造的 same-device reference/model overlap；它通过 logical-step packed dispatch 与 residency 降低 barrier/往返。只有 capability 声明 stream-fence 时才启用本任务内的真正同卡并发路径，并有次序、lease 与数据一致性测试。
- [ ] AC12（需求交付；来源：用户要求 TensorBoard 曲线）：通用 TensorBoard hook 能从 smoke/resume 运行写出可读取 event file，包含稳定 global-step 的 loss、learning rate、吞吐、显存和 pipeline/reference profile tag；DDP 下仅 rank 0 写出。
- [ ] AC13（需求交付；来源：用户重新定义 eval）：配置的 step cadence 能触发独立可视化 eval，对同一随机 probe 产出固定 1024 spp reference、默认 deterministic deferred 的当前 neural、difference 和 provenance，并写出 TensorBoard image panel；可选 neural path tracing 必须显式使用有界低 spp，两侧 mode/spp 不得混写为 matched，且它不消费或改变训练 RNG/query cursor，resume 后 probe identity 与 cadence 可验证。
- [ ] AC14（需求交付；来源：用户接受异步 Windows capture worker）：Linux train 可以发布版本化 visual-eval request 并继续训练；Windows worker 可幂等领取、用 diagnostic snapshot 完成现有 D3D12 1024 spp capture、原子发布结果，rank 0 collector 可补写 TensorBoard。重复、超时、失败和过期结果均有明确状态且不被当作 formal deployment evidence。

## Out of Scope

- 本任务不以发明新的 neural evaluator 候选或改变训练目标为主要目的；只有为验证框架扩展性所需的最小适配属于范围内。
- 不把 reference、source 原生语义或实时 runtime 合同改造成 VRFrameGeneration 的领域模型。
- 不引入离线训练 corpus。
- 不把可视化 eval 的输出当作 formal 泛化指标或 checkpoint 选择依据；它用于训练过程诊断和直观对比。
- 本任务不新增 Linux/Vulkan viewer 或另一套 1024 spp renderer；Linux 通过异步 job 接入现有 Windows/D3D12 capture worker。
- 不以“显存未满”为理由无条件预载全部 Metal decoded mip；residency 大小必须来自显式预算与 profile。
- 本任务不实现独立 reference GPU/P2P data service；只保留 capability 扩展点，不能改变默认 DDP world size。
