# vMaterials 2 Metal：Windows正确性验证与Linux长训练

## 它是什么

`metal-fused-neural-material`把vMaterials 2 Metal的原生MDL参数、52组texture asset与online reference query编译成统一的`prepare/evaluate/sample/pdf` neural material。训练始终由一个进程控制一张GPU，不保存response batch，也没有DDP或per-rank状态。

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

```bash
conda run -n neural-shading python tools/learning/build_metal_training_configs.py
conda run -n neural-shading python tools/learning/preflight_metal_fused.py \
  --output artifacts/metal-linux-training/full-cohort-preflight.json
```

preflight必须得到692 exports、178 execution groups、52 texture sets、64-entry parameter schema table、4种texture role class、6种typed parameter type、全部responsibility，以及20个required component和11-component proposal闭包。它与3-export Windows optimization subset相互独立。

## Linux部署与smoke gate

先按[统一Reference Backend部署](reference_backend_deployment.md)部署锁定的Falcor/MDL toolchain，并由用户把`assets/source-materials/mdl-vmaterials2/2.4.0/Materials`复制到目标机。launcher只接受一个十进制`CUDA_VISIBLE_DEVICES`，并把Falcor映射到同一物理GPU、Torch映射到进程内`cuda:0`。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/deploy_reference_linux.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls.cli reference doctor

CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls.cli learn train \
  configs/learning/metal-fused-full-linux-smoke.json \
  artifacts/metal-linux-training/smoke/checkpoint.pt
```

只有目标Linux主机自己的smoke checkpoint、metrics、summary和review可以作为Linux gate；Windows证据不能替代它。检查：

- `checkpoint.review.json`中`complete=true`、metric全finite、gradient/update coverage完整；
- 四个phase都有真实step，`runtime_fp16_quantization_trace`存在且finite；
- `source_count=692`，peak VRAM不超过目标卡可用容量；
- `checkpoint.summary.json`与review的config/checkpoint hash一致；
- 没有host response readback、磁盘batch或distributed process。

## 启动、恢复与停止long run

先用long config自身运行16步并正常写出可恢复checkpoint；这是同一config identity内的启动点，不拿smoke config checkpoint跨config恢复：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls.cli learn train \
  configs/learning/metal-fused-full-linux-long.json \
  artifacts/metal-linux-training/long/checkpoint.pt \
  --stop-at-step 16

CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls.cli learn train \
  configs/learning/metal-fused-full-linux-long.json \
  artifacts/metal-linux-training/long/checkpoint.pt \
  --resume artifacts/metal-linux-training/long/checkpoint.pt
```

`--stop-at-step N`会在global step `N`写出包含optimizer、scheduler、precision、RNG、typed-state pool和query cursor的`TrainingCheckpoint@4`后正常退出。已经运行的进程可用一次`Ctrl+C`停止；这会放弃正在执行的step，随后从最近的`checkpoint.stepXXXXXXXX.pt`恢复，不修改config：

```bash
latest=$(ls -1 artifacts/metal-linux-training/long/checkpoint.step*.pt | sort | tail -n 1)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls.cli learn train \
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

## 训练完成后的首轮审阅

每次`learn train`都会生成`checkpoint.review.json`。它记录四phase初尾窗口、固定2000次bootstrap的mean-loss delta区间、finite/gradient/update健康状态、peak VRAM、step rate、checkpoint/metrics bytes、reference提交wall time及forward/backward/optimizer GPU时间。loss delta只作report-only观察，不被事后改成质量hard gate。

训练完成后可以执行一次基础checkpoint evaluation与一个代表性package export：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls.cli learn evaluate \
  configs/learning/metal-fused-full-linux-long.json \
  artifacts/metal-linux-training/long/checkpoint.pt --batches 8

CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls.cli learn export \
  artifacts/metal-linux-training/long/checkpoint.pt \
  artifacts/metal-linux-training/long/package --material-index 0
```

review明确写入`automatic_followups=[]`与`next_action=user-review-required`。它不会启动formal matrix、更多seed、消融、compact、distillation或Pareto；这些都等待用户先看首轮效果后另行决定。

交接manifest由以下命令生成，绑定当前commit、toolchain/config/registry/method hashes和上述命令；`linux_execution_status`在目标机运行前保持`pending-on-target-host`：

```bash
conda run -n neural-shading python -m tools.learning.build_metal_linux_handoff \
  --output artifacts/metal-linux-training/handoff.json
```
