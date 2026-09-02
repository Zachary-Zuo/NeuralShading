# vMaterials 2 Metal：Windows正确性验证与Linux长训练

## 它是什么

`metal-fused-neural-material`把vMaterials 2 Metal的原生MDL参数、52组texture asset与online reference query编译成统一的`prepare/evaluate/sample/pdf` neural material。训练默认单进程单卡；Linux可通过`--gpus`启用torchrun/NCCL DDP，多rank共享梯度，只有rank0写出统一checkpoint/metrics，不保存response batch。

完整训练固定四个phase：

1. `codec-warmup`联合训练role-aware stems、shared encoder/decoder、semantic/structured heads、asset adapter与INT8 grid STE；
2. `joint-appearance`同时训练codec、pure typed compiler、analytic/angular/residual evaluator，并从MDL取得当前query的线性`f`；
3. `proposal-fit`保留完整evaluator路径，以detached `luminance(f)|cosθi|`训练10个有方向lobe加full-hemisphere fallback；
4. `qat-refine`在同一个objective中执行asset、evaluator和sampler三条route。FP32 master不被覆盖；实际会部署的weights在forward中做FP16 straight-through fake quantization，high/low grid继续做INT8 STE，敏感插值与累积保持FP32。

`target-visible-optimized-state-control`只在`joint-appearance`作为训练control更新，不进入QAT或runtime pack。QAT更新codec/shared asset state、typed compiler、prepare、direction/evaluator与proposal；这不是用前三阶段冒充的别名。

## 三份配置的边界

| 配置 | source范围 | batch geometry | 用途 |
|---|---:|---:|---|
| `metal-fused-full-windows-smoke.json` | registry机械选择的3个stratified export / 3个texture set | 小batch、16步 | RTX 4090上的四phase正确性、gradient/update、resume与部署验证 |
| `metal-fused-full-linux-smoke.json` | 全部692个opaque export / 52个texture set | 与long相同 | Linux目标机资源、四phase与完整source plan gate |
| `metal-fused-full-linux-long.json` | 全部692个opaque export / 52个texture set | asset 12、evaluator 64、sampler 64 | 120000步首轮质量训练 |

Linux smoke和long由`tools/learning/build_metal_training_configs.py`机械生成；两者的source、online state recipe、model profile、phase顺序、route options、parameter groups、loss、precision、optimizer和batch geometry逐字段相同，只允许run budget及log/audit/validation cadence不同。静态检查会拒绝第692个source遗漏、QAT precision漂移、loss变化或多方向route。Windows只缩小正确性验证的source与batch，不改变full model shape或required component。

本机 GPU5 gate 已实测通过：692 sources、四个 phase 各 4 步、1424 次 online query；`checkpoint.review.json` 报告 `complete=true`、所有 metric finite、gradient/update coverage complete、proposal identity error 为 0，峰值显存 `5,578,002,944` bytes（约 5.19 GiB），steady-state `median_steps_per_second=0.6459367207`。优化前首次启动的 692-source MDL eager materialization 约 12 分钟；优化后在已有 cache 的 GPU2 上，初始化加 16-step smoke 总 wall-clock 为 `7:14.75`，其中 runner 的 16 步训练约 21 秒。它是启动开销，不是离线训练 batch；完全没有编译 cache 时仍需执行一次 SDK 编译。

本次可复现产物位于 `artifacts/metal-linux-training/full-cohort-smoke-rerun/`，其中 `checkpoint.pt` 可由同一 smoke config resume；`artifacts/metal-linux-training/handoff.json` 保存了 GPU5 命令和 config/registry/toolchain hash。`long` 目录下此前失败的 checkpoint 不可 resume，应从新的 long 输出路径开始。

```bash
PYTHONPATH=src conda run --no-capture-output -n neural-shading python tools/learning/build_metal_training_configs.py
PYTHONPATH=src conda run --no-capture-output -n neural-shading python tools/learning/preflight_metal_fused.py \
  --output artifacts/metal-linux-training/full-cohort-preflight.json
```

preflight必须得到692 exports、178 execution groups、52 texture sets、64-entry parameter schema table、4种texture role class、6种typed parameter type、全部responsibility，以及20个required component和11-component proposal闭包。它与3-export Windows optimization subset相互独立。

训练样本与 MDL 编译缓存是两种不同的东西。`OnlineTrainingProducer` 每个 route 都通过 GPU-resident reference session 即时生成 target，不写入或读取磁盘 batch；训练 checkpoint、metrics、summary 和 review 写在下面命令指定的 `artifacts/metal-linux-training/...` 目录。`build/mdl-reference/cache` 只保存可重建的 MDL SDK 编译产物，用于跨 step 复用，不是训练数据集。

MDL artifact 的 decoded texture payload 使用 cache 根下的 `resource-payloads/<前两位>/<sha256>` 内容寻址存储，artifact 内的原始 `data` 路径通过 hardlink 指向共享内容，manifest 的逐文件哈希和 runtime 语义不变。这样不同 typed state 仍可有独立 argument/code artifact，但不会为同一纹理重复保存几十 MiB。首次构建仍会 eager materialize 全部 typed state，因此它会增加启动时间；这不改变 online query 或训练 step 的定义。

启动时对已有 artifact 的完整性校验会复用同一进程内、按文件 inode 与时间戳失效的有界 SHA-256 结果；共享 hardlink 的 decoded payload 不会被 692 个 typed state 重复读盘。对 decoded texture 的逐文件哈希现在延迟到该 artifact 第一次实际绑定 reference/native asset 时执行，启动阶段仍检查 manifest、文件存在性、尺寸、路径、HLSL/argument/RO 文件哈希；首次绑定通过 `FileResourcePayload.read_bytes()` 或 asset collection 的 `verify_texture_payloads()` 完整复核，删除或修改 payload 会 fail closed。这样不会把在线训练变成离线 batch，也不会降低完整性标准。

首次没有编译缓存时，692-state 的 SDK 编译仍需执行一次；已有 `build/mdl-reference/cache` 时，训练入口不会为了构造全量 plan 预先读取约 300 GB 的重复 hardlink payload，而是随实际 reference group/tile 使用按需读取。后续运行应直接复用 cache。

## Linux部署与smoke gate

先按[统一Reference Backend部署](reference_backend_deployment.md)部署锁定的Falcor/MDL toolchain，并由用户把`assets/source-materials/mdl-vmaterials2/2.4.0/Materials`复制到目标机。单卡 launcher 接受一个十进制`CUDA_VISIBLE_DEVICES`，并把Falcor映射到同一物理GPU、Torch映射到进程内`cuda:0`。若要在 GPU2、3、4 上运行一个同步 DDP 作业，使用：

```bash
bash scripts/run_falcor_python.sh --gpus 2,3,4 -- \
  -m ncls.cli learn train configs/learning/metal-fused-full-linux-smoke.json \
  artifacts/metal-linux-training/ddp/checkpoint.pt
```

```bash
CUDA_VISIBLE_DEVICES=5 bash scripts/deploy_reference_linux.sh
CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls.cli reference doctor

CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls.cli learn train \
  configs/learning/metal-fused-full-linux-smoke.json \
  artifacts/metal-linux-training/smoke/checkpoint.pt
```

只有目标Linux主机自己的smoke checkpoint、metrics、summary和review可以作为Linux gate；Windows证据不能替代它。检查：

- `checkpoint.review.json`中`complete=true`、metric全finite、gradient/update coverage完整；
- 四个phase都有真实step，`runtime_fp16_quantization_trace`存在且finite；
- `source_count=692`，peak VRAM不超过目标卡可用容量；
- `checkpoint.summary.json`与review的config/checkpoint hash一致；
- 没有host response readback或磁盘batch；DDP模式使用NCCL process group同步梯度。

## 启动、恢复与停止long run

先用long config自身运行16步并正常写出可恢复checkpoint；这是同一config identity内的启动点，不拿smoke config checkpoint跨config恢复：

```bash
CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls.cli learn train \
  configs/learning/metal-fused-full-linux-long.json \
  artifacts/metal-linux-training/long/checkpoint.pt \
  --stop-at-step 16

CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls.cli learn train \
  configs/learning/metal-fused-full-linux-long.json \
  artifacts/metal-linux-training/long/checkpoint.pt \
  --resume artifacts/metal-linux-training/long/checkpoint.pt
```

`--stop-at-step N`会在global step `N`写出包含optimizer、scheduler、precision、RNG、typed-state pool和query cursor的`TrainingCheckpoint@4`后正常退出。已经运行的进程可用一次`Ctrl+C`停止；这会放弃正在执行的step，随后从最近的`checkpoint.stepXXXXXXXX.pt`恢复，不修改config：

```bash
latest=$(ls -1 artifacts/metal-linux-training/long/checkpoint.step*.pt | sort | tail -n 1)
CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls.cli learn train \
  configs/learning/metal-fused-full-linux-long.json \
  artifacts/metal-linux-training/long/checkpoint.pt --resume "$latest"
```

监控只读取runner已有输出，不增加watcher或第二训练进程：

```bash
tail -f artifacts/metal-linux-training/long/checkpoint.metrics.jsonl
nvidia-smi dmon -s pucvmet -d 5
```

long config每5000步validation并写periodic checkpoint，总计24个cadence点；phase boundary都落在这些点上。Linux smoke与long使用同一batch geometry，因此以Linux smoke review中的`median_steps_per_second`和`peak_memory_bytes`估算long run：

```text
ETA_seconds ≈ 120000 / smoke_median_steps_per_second
checkpoint_disk ≈ 25 × smoke_checkpoint_bytes
```

这两个值是目标机的容量规划观察值，不是质量或完成门。若吞吐/显存异常，按`implementation defect / protocol defect / resource defect / normal empirical outcome`分类；前三类停止并修复或回planning，最后一类照实进入结果审阅，不自动加预算或换seed。

按本次 GPU5 smoke 的 steady-state 中位数，120,000 steps 约 `185,770` 秒，即 **51.6 小时**；GPU2/3/4 的 692-source smoke 中位数对应约 47.1--49.0 小时。单卡容量规划取保守约 **52 小时**，另加首次 MDL eager materialization、周期性 validation/checkpoint I/O 和机器负载波动。这个 ETA 只用于容量规划，不是质量或完成保证。

## 训练完成后的首轮审阅

每次`learn train`都会生成`checkpoint.review.json`。它记录四phase初尾窗口、固定2000次bootstrap的mean-loss delta区间、finite/gradient/update健康状态、peak VRAM、step rate、checkpoint/metrics bytes、reference提交wall time及forward/backward/optimizer GPU时间。loss delta只作report-only观察，不被事后改成质量hard gate。

训练完成后可以执行一次基础checkpoint evaluation与一个代表性package export：

```bash
CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls.cli learn evaluate \
  configs/learning/metal-fused-full-linux-long.json \
  artifacts/metal-linux-training/long/checkpoint.pt --batches 8

CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls.cli learn export \
  artifacts/metal-linux-training/long/checkpoint.pt \
  artifacts/metal-linux-training/long/package --material-index 0
```

review明确写入`automatic_followups=[]`与`next_action=user-review-required`。它不会启动formal matrix、更多seed、消融、compact、distillation或Pareto；这些都等待用户先看首轮效果后另行决定。

交接manifest由以下命令生成，绑定当前commit、toolchain/config/registry/method hashes和上述命令；`linux_execution_status`在目标机运行前保持`pending-on-target-host`：

```bash
conda run -n neural-shading python -m tools.learning.build_metal_linux_handoff \
  --output artifacts/metal-linux-training/handoff.json
```
