# vMaterials 2 Metal：Windows正确性验证与Linux长训练

## 它是什么

公开方法 `metal` 把 vMaterials 2 Metal 的原生 MDL 参数、52 组 texture asset 与 online reference query 编译成统一的 `prepare/evaluate/sample/pdf` neural material。单个 `--devices` 序号直接单卡训练；Linux 指定多个序号时由统一入口自动启用 torchrun/NCCL DDP，多 rank 共享梯度，只有 rank 0 写出 checkpoint/metrics，不保存 response batch。

完整训练固定两个端到端phase，总预算仍为120000步：

1. `joint-coarse-to-fine`运行105000步。每一步都执行asset、online reference evaluator和method sampler三条route；codec、asset adapter、typed compiler、prepare、direction/evaluator及proposal从第1步共同更新。codec reconstruction和teacher/compiler项是最终appearance目标的辅助项，不是独立预训练。
2. `qat-refine`运行15000步，继续同一appearance与proposal目标。FP32 master不被覆盖；实际会部署的weights在forward中做FP16 straight-through fake quantization，high/low grid继续做INT8 STE，敏感插值与累积保持FP32。

proposal从第1步使用非零、按phase step决定且可恢复的冻结线性ramp；`luminance(f)|cosθi|` target与shared evaluator/latent显式detach，proposal loss只拥有`proposal_sampler`梯度。`target-visible-optimized-state-control`只作为训练control，不进入runtime pack；QAT冻结该training-only teacher，但不改变最终evaluator输入语义。

## 三份配置的边界

| 配置 | source范围 | batch geometry | 用途 |
|---|---:|---:|---|
| `metal-windows-smoke.yaml` | registry机械选择的3个stratified export / 3个texture set | 小batch、16步 | RTX 4090上的两phase lifecycle、gradient/update、resume与部署链路验证 |
| `metal-linux-smoke.yaml` | 全部692个opaque export / 52个texture set | 与long相同 | Linux目标机资源、两phase与完整source plan gate |
| `metal-linux-long.yaml` | 全部692个opaque export / 52个texture set | asset 12、evaluator 64、sampler 64 | 120000步首轮质量训练 |

三份 run 位于 `configs/training/runs/`，共同组合 `methods/metal.yaml`、独立 data fragment 和 recipe；resolver 严格合并并输出 plan identity。Linux smoke 与 long 的 source、online state recipe、model profile、phase 顺序、route options、parameter groups、loss、precision、optimizer 和 batch geometry 对齐，只允许 run class、预算及 log/audit/validation cadence不同。Windows只缩小正确性验证的 source 与 batch，不改变 full model shape、route/loss 拓扑、proposal schedule 类型或 required component；训练、checkpoint、readiness、compiler 和 viewer 上层代码不按操作系统分支。

旧四phase GPU5结果与20k checkpoint只保留为根因证据，不能用于当前实现的速度、质量或readiness结论。旧20k只完成codec warmup，evaluator与proposal没有任何gradient/update coverage；viewer白模不是“训练较差”，而是部署了尚未开始训练的evaluator。当前Windows RTX 4090共享路径证据见任务artifact：同一当前descriptor下的16-step与544-step run都覆盖全部13组参数并完成joint/QAT；固定shaderball球面区域的reference-neural线性MAE从`0.4226`降至`0.1336`（约68.4%），neural平均亮度从`0.4314`降至`0.1250`，reference为`0.0478`。这证明最终evaluator不再保持初始化白模、训练信号能穿过checkpoint/package/viewer链路，但544步输出仍偏中性且不够接近reference，只是学习与部署正确性diagnostic，不是formal质量结论。此前不同implementation identity的544-step观察值`0.0621`只保留为历史证据，不与当前结果混用。

Linux目标机尚需在同一commit上重跑full-cohort smoke与120k；旧约52小时ETA随旧phase、逐step group thrash和旧metrics一并作废。新的绝对吞吐、显存与ETA只使用目标机修复后metrics报告，不设事后hard gate。

```bash
PYTHONPATH=src conda run --no-capture-output -n neural-shading python tools/learning/preflight_metal_fused.py \
  --output artifacts/metal-linux-training/full-cohort-preflight.json
```

preflight必须得到692 exports、178 execution groups、52 texture sets、64-entry parameter schema table、4种texture role class、6种typed parameter type、全部responsibility，以及20个required component和11-component proposal闭包。它与3-export Windows optimization subset相互独立。

训练样本与 MDL 编译缓存是两种不同的东西。`OnlineTrainingProducer` 每个 route 都通过 GPU-resident reference session即时生成target，不写入或读取磁盘batch；训练checkpoint、metrics、summary和review写在下面命令指定的`artifacts/metal-linux-training/...`目录。`build/mdl-reference/cache`只保存可重建的MDL SDK编译产物，用于跨step复用，不是训练数据集。content-addressed目录发布会在短暂文件句柄占用时有界重试；若另一进程先发布同identity，则加载并验证胜出artifact，绝不覆盖语义不同或损坏的目录。

MDL artifact 的 decoded texture payload 使用 cache 根下的 `resource-payloads/<前两位>/<sha256>` 内容寻址存储，artifact 内的原始 `data` 路径通过 hardlink 指向共享内容，manifest 的逐文件哈希和 runtime 语义不变。这样不同 typed state 仍可有独立 argument/code artifact，但不会为同一纹理重复保存几十 MiB。首次构建仍会 eager materialize 全部 typed state，因此它会增加启动时间；这不改变 online query 或训练 step 的定义。

启动时对已有 artifact 的完整性校验会复用同一进程内、按文件 inode 与时间戳失效的有界 SHA-256 结果；共享 hardlink 的 decoded payload 不会被 692 个 typed state 重复读盘。对 decoded texture 的逐文件哈希现在延迟到该 artifact 第一次实际绑定 reference/native asset 时执行，启动阶段仍检查 manifest、文件存在性、尺寸、路径、HLSL/argument/RO 文件哈希；首次绑定通过 `FileResourcePayload.read_bytes()` 或 asset collection 的 `verify_texture_payloads()` 完整复核，删除或修改 payload 会 fail closed。这样不会把在线训练变成离线 batch，也不会降低完整性标准。

首次没有编译缓存时，全量typed state的SDK编译仍需执行一次；已有`build/mdl-reference/cache`时，训练入口不会为了构造全量plan预先读取约300 GB的重复hardlink payload，而是随实际reference group/tile使用按需读取。后续运行应直接复用cache。

训练热循环使用`group-block-balanced@1`。evaluator与method sampler在同一global step选择同一个execution group，一个group连续服务64步；完整cycle按group record数加权，DDP rank按确定性stride分区，validation再使用冻结的104729-block offset形成独立holdout group流。backend residency仍有界，但只在block或validation边界发生必要的miss/create/evict，不再让178 groups配8-resident LRU形成每step稳态thrash。训练session只请求`evaluate`，不会再为未使用的reference `sample/pdf` pass付构建成本。

metrics每个log window记录完整step wall与prepare wall的count/mean/median/p90/max、phase-local/rolling rate、group ID前缀、candidate/rejection，以及session hit/miss、group create/evict、runtime/pass/resource/slot build、operation dispatch和resident数。training与validation分别使用`profile/reference_*`和`profile/validation_reference_*`，不会把validation提前materialize的group误记为下一段训练成本。第一个step或block转换的冷构建必须在window max中可见；普通step不为这些CPU counter增加GPU host sync。

## Linux部署与smoke gate

先按[统一Reference Backend部署](reference_backend_deployment.md)部署锁定的Falcor/MDL toolchain，并由用户把`assets/source-materials/mdl-vmaterials2/2.4.0/Materials`复制到目标机。单卡 launcher 接受一个十进制`CUDA_VISIBLE_DEVICES`，并把Falcor映射到同一物理GPU、Torch映射到进程内`cuda:0`。若要在 GPU2、3、4 上运行一个同步 DDP 作业，使用：

```bash
bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/metal-linux-smoke.yaml --devices 2,3,4 \
  --output artifacts/metal-linux-training/ddp/checkpoint.pt
```

```bash
CUDA_VISIBLE_DEVICES=5 bash scripts/deploy_reference_linux.sh
CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls.cli reference doctor

CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/metal-linux-smoke.yaml --devices 5 \
  --output artifacts/metal-linux-training/smoke/checkpoint.pt
```

只有目标Linux主机自己的smoke checkpoint、metrics、summary和review可以作为Linux gate；Windows证据不能替代它。检查：

- `checkpoint.review.json`中`complete=true`、metric全finite、gradient/update coverage完整；
- 两个phase都有真实step，`runtime_fp16_quantization_trace`存在且finite；
- `source_count=692`，peak VRAM不超过目标卡可用容量；
- `checkpoint.summary.json`与review的config/checkpoint hash一致；
- 没有host response readback或磁盘batch；DDP模式使用NCCL process group同步梯度。

## 启动、恢复与停止long run

先用long config自身运行16步并正常写出可恢复checkpoint；这是同一config identity内的启动点，不拿smoke config checkpoint跨config恢复：

```bash
CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/metal-linux-long.yaml --devices 5 \
  --output artifacts/metal-linux-training/long/checkpoint.pt \
  --stop-at-step 16

CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/metal-linux-long.yaml --devices 5 \
  --output artifacts/metal-linux-training/long/checkpoint.pt \
  --resume artifacts/metal-linux-training/long/checkpoint.pt
```

`--stop-at-step N` 会在 global step `N` 写出包含 resolved plan、optimizer、scheduler、precision、逐 rank RNG/data cursor 与 hook cursor 的 `TrainingCheckpoint@1` 后正常退出。已经运行的进程可用一次 `Ctrl+C` 停止；这会放弃正在执行的 step，随后从最近的 `checkpoint.stepXXXXXXXX.pt` 恢复，不修改 YAML：

```bash
latest=$(ls -1 artifacts/metal-linux-training/long/checkpoint.step*.pt | sort | tail -n 1)
CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/metal-linux-long.yaml --devices 5 \
  --output artifacts/metal-linux-training/long/checkpoint.pt --resume "$latest"
```

监控只读取 engine 已有输出，不增加 watcher 或第二训练进程：

```bash
tail -f artifacts/metal-linux-training/long/checkpoint.metrics.jsonl
nvidia-smi dmon -s pucvmet -d 5
```

long config每5000步validation并写periodic checkpoint，总计24个cadence点；phase boundary落在cadence点上。Linux smoke与long使用同一batch geometry，因此以修复后Linux smoke review中的phase-local step rate和`peak_memory_bytes`估算long run：

```text
ETA_seconds ≈ 120000 / smoke_median_steps_per_second
checkpoint_disk ≈ 25 × smoke_checkpoint_bytes
```

这两个值是目标机的容量规划观察值，不是质量或完成门。若吞吐/显存异常，按`implementation defect / protocol defect / resource defect / normal empirical outcome`分类；前三类停止并修复或回planning，最后一类照实进入结果审阅，不自动加预算或换seed。

不得继续使用旧四phase run推导约52小时的ETA。新的ETA只在目标Linux以当前semantic fingerprint完成full-cohort smoke后计算，并把一次性MDL materialization、block边界group build、validation/checkpoint I/O与steady-state step分别列出。它只用于容量规划，不是质量或完成保证。

## 训练完成后的首轮审阅

每次 `ncls train` 都会生成 `checkpoint.review.json`。它记录各 phase 初尾 window、固定 2000 次 bootstrap 的 mean-loss delta 区间、finite/gradient/update 健康状态、peak VRAM、step rate、checkpoint/metrics bytes、reference 提交 wall time及 forward/backward/optimizer GPU 时间。loss delta 只作 report-only 观察，不被事后改成质量 hard gate。

训练完成后可以执行一次基础checkpoint evaluation与一个代表性package export：

```bash
CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls validate \
  artifacts/metal-linux-training/long/checkpoint.pt --batches 8 --device 5

CUDA_VISIBLE_DEVICES=5 bash scripts/run_falcor_python.sh -m ncls export \
  artifacts/metal-linux-training/long/checkpoint.pt \
  artifacts/metal-linux-training/long/package --material-index 0
```

review明确写入`automatic_followups=[]`与`next_action=user-review-required`。它不会启动formal matrix、更多seed、消融、compact、distillation或Pareto；这些都等待用户先看首轮效果后另行决定。

`ncls export` 只接受 exact identity、`run_class=formal`、phase complete 且 required gradient/update coverage 完整的 checkpoint。Windows 短训即使跑完所有 phase 也只能显式生成 evaluate-only 诊断预览：

```powershell
.\scripts\prepare_metal_viewer.ps1 `
  -Checkpoint artifacts\metal-root-fix\windows-learning-probe-final\checkpoint.pt `
  -OutputRoot artifacts\viewer\metal-diagnostic `
  -DiagnosticPreview -DiagnosticLimit 1
```

diagnostic package移除`sample/pdf` capability，并在package/catalog/capture中标记`exact-diagnostic-evaluator-preview`。默认脚本指向120k formal checkpoint；旧20k和仅shape-compatible checkpoint不会再被接受。

交接manifest由以下命令生成，绑定当前commit、toolchain/config/registry/method hashes和上述命令；`linux_execution_status`在目标机运行前保持`pending-on-target-host`：

```bash
conda run -n neural-shading python -m tools.learning.build_metal_linux_handoff \
  --output artifacts/metal-linux-training/handoff.json
```
