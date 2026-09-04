# Phase 0：训练配置与调度基线

## 1. 冻结目的

本文件记录重构开始前可由仓库源码重建的旧运行语义。它用于判断 YAML composition 和后续 engine/data session 迁移是否改变了 source、phase、route 或 checkpoint 身份；它不是要求新架构继续接受旧 JSON 的兼容承诺。

冻结命令：

```powershell
conda run -n neural-shading python -c "from pathlib import Path; from ncls.learning.training import TrainingConfig; [(lambda c,p: print(p.name,c.sha256,len(c.source['materials']),c.total_steps,','.join(x.name for x in c.phases)))(TrainingConfig.load(p),p) for p in sorted(Path('configs/learning').glob('*.json'))]"
```

## 2. 旧配置语义指纹

| 配置 | `TrainingConfig@4` SHA-256 | source 数 | 总 step | phase |
|---|---|---:|---:|---|
| `metal-fused-full-linux-long.json` | `2dd24f19c7f6f5bc53a2a1c6dbd0fa15ef54dec6ad96408d32dda359290ada3f` | 692 | 120000 | `joint-coarse-to-fine, qat-refine` |
| `metal-fused-full-linux-smoke.json` | `f74f5b9d447ddd8d7968e88e9476a74ea34931f038a1247cb24cdbf44a5871e1` | 692 | 16 | `joint-coarse-to-fine, qat-refine` |
| `metal-fused-full-windows-smoke.json` | `c120c2798667f391b5787e0be6a3ebf39800e49c6c3c2705e044f3122b876c53` | 3 | 16 | `joint-coarse-to-fine, qat-refine` |
| `nvidia-rta2024-layer-stack-smoke.json` | `915d676176093ed46d8993791040232c0f54975e173f1acd8745d5b9c3a01091` | 1 | 2 | `bootstrap, finetune` |
| `nvidia-rta2024-materialx-formal.json` | `a81943ea102379d568c354b1cfa272229c4419d21a4637adc4170aa641d961c9` | 1 | 300000 | `bootstrap, finetune` |
| `nvidia-rta2024-materialx-smoke.json` | `d13a6576bbe85e95945c4d7bc0dde756c3f5e4409a9dc719ec422922a363b3fe` | 1 | 2 | `bootstrap, finetune` |
| `nvidia-rta2024-mdl-effect-pigment-smoke.json` | `8efdc14942403f5de75ffad8979d9e41bb46d2acfd274393310173ac4b4e1aab` | 1 | 2 | `bootstrap, finetune` |

新 resolver 的迁移 adapter 必须逐份重建这些指纹；最终删除旧 JSON 后，测试改为冻结新 `ResolvedTrainingPlan` 的 canonical identity 与显式合同，不再把旧文件留作运行时 oracle。

## 3. 调度 before trace

历史 Linux long run 的 2101 条 timing 记录给出：

| stage | median | p90 |
|---|---:|---:|
| `batch_prepare_wall_seconds` | 1.324 s | 2.263 s |
| `forward_gpu_seconds` | 0.115 s | 0.153 s |
| `backward_gpu_seconds` | 0.136 s | 0.183 s |
| `optimizer_gpu_seconds` | 0.010 s | 0.011 s |

这组数据只能证明当前 prepare 串行段显著，不能把整段解释为 GPU idle，因为其中包含 reference GPU dispatch。后续 before/after trace 必须把 host I/O/decode、residency miss/H2D、reference submit/wait、adapt、ready queue wait 和 model compute 分开记录。

当前明确的结构性 before 行为：

- `TrainingRunner` 的 prefetch queue 仍由消费线程同步填充，没有 worker overlap；
- Metal patch 请求每 step 将 `asset_index/uv/mip` 从 GPU 读回 host，再上传 patch；
- Linux/Vulkan reference interop 使用全设备同步，因此不能用额外 CUDA stream 宣称同卡 reference/model overlap；
- full-cohort source locator 被两份 Linux JSON 各自内联 692 次。

这些结构将分别由 data pipeline trace、GPU residency test、backend concurrency capability test 与 source-set resolver parity test 关闭。

## 4. Visual eval 成本观测

2026-09-04 在 Windows / RTX 4090 上以 `640×360` 双 panel（每个 slot `320×360`）、每次 dispatch 16 samples 运行 NVIDIA diagnostic package。原始 harness 把同一个 1024 spp target 同时施加给 source reference 与 neural path-tracing slot；运行超过 20 分钟仍未完成，GPU 持续 100%，显存约 4.45 GiB，随后按用户要求终止。该结果分类为 `resource/throughput defect`，不作为固定性能 hard gate。

主要成本来自 neural MLP path tracer：每个 slot 约 1.18 亿 primary samples，且每条 path 包含多策略采样与有限多 bounce；reference 的降噪预算不应机械复制为训练期 neural 诊断预算。第一次修正保持 reference 1024 spp、把 neural 降为显式 16 spp，并在 capture manifest 分别记录 target/actual 与 `training-diagnostic`。该真实 worker 虽成功完成并修复了 source identity 问题，墙钟时间仍为 138.781 s，依然不适合作为训练 cadence。

因此 cadence 的第二次修正保留 1024 spp reference，但默认以同相机、同光照的 deterministic deferred evaluator 生成 neural 图。该 difference 用于快速观察 evaluator 外观漂移，明确不声称是 matched-integrator 误差。需要检查 neural `sample/pdf` 和实际环境积分时，可手工选择有界低 spp path tracing；双 1024 spp 仍只用于低频深度检查。

第二次修正后的真实 worker 在同一 Windows / RTX 4090、同一 `640×360` 输出规模下用 12.429 s 完成，包含进程启动、场景/着色器加载、diagnostic package 编译、1024 spp reference 和 neural deferred capture。capture manifest 记录 `comparison_purpose=training-diagnostic`、reference slot 为 `path-tracing/1024`、neural slot 为 `deferred/0`，线性 reference/neural/difference EXR 均为有限 float32；结果已由 collector 补写到 TensorBoard。相对 16 spp neural PT 的 138.781 s，本次 observed wall time 降低约 11.2 倍，但它仍只是本机观测，不升级为跨机器 hard gate。
