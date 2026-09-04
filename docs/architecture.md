# 统一训练与部署架构

项目只有一条正式训练路径。方法只声明“需要什么数据、怎样建模和怎样部署”，公共框架负责配置解析、数据调度、训练生命周期、checkpoint、日志与可视化：

```text
YAML run
  ├─ base：设备、data plane 与 hooks
  ├─ method：nvidia / metal
  ├─ data：source locator / source set
  └─ recipe：phase、route、loss 与优化器
        ↓ resolve + identity
ResolvedTrainingPlan@1
        ↓
MethodPlugin
  ├─ data requirements / source adapter
  ├─ model factory / objective / lifecycle
  ├─ checkpoint codec
  └─ deployment compiler
        ↓
DataExecutionPlan + OnlineDataSession
        ↓
TrainingEngine + typed events/hooks
        ↓
TrainingCheckpoint@1
        ├─ validate / visual eval
        └─ ScatteringPackage@2 → viewer
```

## 材质与 reference 边界

LayerStack、OpenPBR、MERL、MaterialX 与 MDL 保留各自原生 locator、参数、资源、编辑语义和 reference。`SourceSnapshot` 是统一追溯边界，不能把非层模型先反演成 `LayerStackIR` 再当作 GT。reference definition 把 snapshot 编入 `ReferenceExecutionPlan@1`；公共 backend/session 只执行 `prepare/evaluate/sample/pdf`，不按 source family 写分支。

底层平台差异由 launcher 与 `ReferenceBackendCapability` 隐藏。训练 engine、method 和 data contract 不选择 D3D12/Vulkan，也不读取物理 GPU 环境变量。Windows 单卡和 Linux 单卡进入同一个 engine；Linux 多卡由设备列表自动进入 torchrun/NCCL，Windows 多卡在构造 CUDA/Falcor runtime 前明确拒绝。

## 在线数据面

正式训练只使用 GPU-resident online reference query，不保存或读取 batch/corpus。`DataExecutionPlan` 固定 rank partition、CPU worker、队列、reference inflight、transfer stream 与显存 residency 预算；checkpoint 同时保存 plan identity 和 rank-local data cursor，恢复前后产生同一逻辑数据流。

`num_workers` 只执行可复制、可序列化的 CPU/host 阶段，不让子进程持有 CUDA 或 Falcor。GPU 侧由 rank 主进程拥有：typed metadata 常驻设备，资源按字节预算缓存并用 lease 防止活跃对象被驱逐；Metal texture miss 才经过 host decode/pinned transfer，命中路径不做逐 step GPU→CPU request readback。reference scheduler 的 packed dispatch、ready ring 与 inflight 数受 backend concurrency capability 约束，不能靠平台名猜测并发安全性。

三种 typed batch 分别承载明确语义：`AssetTileBatch` 提供 asset tile+halo，`EvaluatorBatch` 提供 conditioning、方向和 online reference `target_f`，`MethodSamplerBatch` 提供 conditioning 与 `sample_u`。具体方法通过 data facet 声明需要哪些 route/字段并创建自己的 source adapter；公共 data plane 决定调度、顺序、backpressure、错误传播、drain 和 resume。

## 固定训练生命周期

`TrainingEngine` 不知道 NVIDIA 或 Metal。它按 resolved phase graph 固定执行 phase setup、取 batch、forward/backward、optimizer/schedule、gradient audit、validation、checkpoint boundary 和 cleanup，并通过 typed event bus 发布状态。TensorBoard、JSONL、checkpoint 和 visual eval 都是 hook/外层服务，不在方法实现里重复一套训练循环。

公开 method key 只使用 `nvidia`、`metal` 这类短名称；版本、descriptor hash、implementation hash 和各 facet identity 存入 resolved plan/checkpoint，不塞进用户可见名称。新训练只写 `TrainingCheckpoint@1`。旧 `TrainingCheckpoint@4` 仅由隔离 importer 读取为 evaluation snapshot，不能恢复训练，也没有旧 JSON config 或旧 CLI fallback。

## 训练中可视化

visual eval 使用独立 probe RNG 和文件 spool，不阻塞训练；Windows worker 可晚到执行，collector 随后补写 TensorBoard。常规 cadence 保留真实 reference 1024 spp path tracing，neural 一侧使用同相机/灯光的 deterministic deferred 渲染并生成 difference。低 spp neural path tracing 与双侧 1024 spp 只用于手工深度检查，不能成为高频训练开销。

部署仍由同一 method plugin 编译 program/asset/instance 三段 `ScatteringPackage@2`。viewer 只在 identity、typed resource 与 parity 校验通过后原子替换 slot binding；`source-reference` 是显式权威 transport 请求，不是假 package。
