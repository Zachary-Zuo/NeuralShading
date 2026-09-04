# vMaterials 2 Metal：预算内模型的 Linux pilot

## 它是什么

公开方法 `metal` 当前指向 `metal-budgeted-neural-material`。它保留 vMaterials 2 Metal 的原生 MDL locator、typed 参数、图和纹理资源，用对应 reference 在线产生 GT，再编译为固定两次 asset 读取、固定 PreparedState 和固定方向求值成本的 neural material。

当前阶段不是恢复旧 full long，而是在同一个 Tungsten 材质上比较两个完全 matched 的目标预算内结构：

- `metal_budgeted_hybrid_v2`：解析双 lobe 与神经输出共同形成最终线性 RGB `f`；
- `metal_budgeted_direct_control_v2`：相同 MLP、state、asset 和训练预算，最终 `f` 只消费 direct positive RGB，另外三个输出只训练为不进入结果的 core auxiliary。

两者的 `evaluate` 都是 11,392 dense MAC/direction，完整消费24维semantic state；prepare decoder 是 2,560 dense MAC，PreparedState 为 160 B，运行时 asset 读取数为 2。hard bound 是 `evaluate ≤20,000 MAC/direction`、PreparedState `≤192 B`；prepare、analytic 运算、weights、asset bytes 和实测 latency 另行完整报告。旧v1的10,368 MAC、8维condition checkpoint只作step512诊断对照，不能resume到v2。

旧 `metal_fused_full_v1`、`metal-windows-smoke.yaml`、`metal-linux-smoke.yaml` 和 `metal-linux-long.yaml` 只解释历史 checkpoint/package，不再是 canonical 训练入口，也不能 resume 到新方法。

## 为什么只在 Linux 运行

本轮 Windows 只做 unit、静态 layout 和必要的纯模型小测试。不要在 Windows 启动 online reference、256-batch validation、single-material pilot、完整 runtime baseline 或旧 long；这些操作曾造成整机失去响应，而且不能提供当前结构选择所需的 Linux matched 证据。

正式运行使用原生Linux和统一多GPU launcher。当前固定拓扑是物理GPU 5–9上的五卡DDP；它不是DDP scaling实验，不自动追加seed、formal cohort、teacher或旧long。

## 冻结的 matched pair

| 项 | hybrid | direct |
|---|---|---|
| run | `metal-budgeted-hybrid-pilot.yaml` | `metal-budgeted-direct-pilot.yaml` |
| source | Tungsten Brushed Medium Light Brushing exact locator | 相同 |
| train phase | 1792 step `joint-response-fit` | 相同 |
| QAT phase | 256 step `deployment-qat-refine` | 相同 |
| per-rank batch / global batch / direction | 512 / 2560 / 1 | 相同 |
| validation | 每128 step，256 batch，seed `2026090402` | 相同 |
| report / reference packing | 每16 step / 2 logical step | 相同 |
| asset mode | `encoder-only@1` | 相同 |
| 唯一结构轴 | hybrid correspondence/profile | direct correspondence/profile |

方向 recipe 按 batch 四等分为 uniform、cosine、距镜面反射方向 0–8° 和 grazing；空间 recipe 使用固定 anchor、zero/one/four-native-texel footprint 配额与沿 x/y 平衡的一 texel paired UV。该 paired UV 直接监督空间梯度，避免只有颜色均值而看不到微小划痕。

fresh run 在第一次 model forward 前，只用 training reference 和 seed `2026090401`生成16,384个 `target_f`，冻结逐通道 P50 scale、P95 peak 与 energy epsilon并写入 checkpoint。resume 必须复用 checkpoint calibration，不能重新估计或读取 validation 数据。

## 生成交接清单

在代码与配置已经同步到目标 Linux 主机的同一仓库快照后生成或复核清单：

```bash
conda run -n neural-shading python tools/learning/build_metal_linux_handoff.py \
  --output artifacts/09-04-metal-neural-budgeted-redesign/linux-pilot-handoff.json
```

清单中的单GPU命令只保留为可移植fallback，不是本轮DDP5证据入口；当前执行以本页下方GPU 5–9命令、checkpoint内resolved plan和任务research记录为准。

## Linux 执行顺序

先部署锁定的 reference backend，并确认 source assets 已由用户复制到 `assets/source-materials/mdl-vmaterials2/2.4.0/Materials`：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/deploy_reference_linux.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls reference doctor
```

然后严格串行执行两个候选。每个候选先停在step 0产生calibration checkpoint；以下示例由launcher把每个rank的Torch映射为`cuda:0`，同时让Falcor使用对应物理adapter：

```bash
bash scripts/run_falcor_python.sh --gpus 5,6,7,8,9 -- -m ncls train \
  configs/training/runs/metal-budgeted-hybrid-pilot.yaml \
  --devices 5,6,7,8,9 --stop-at-step 0 \
  --output artifacts/metal-budgeted-pilot/hybrid/checkpoint.pt
```

先运行到可恢复的 step 128，再从同一 checkpoint 继续到冻结的 2048-step cap：

```bash
bash scripts/run_falcor_python.sh --gpus 5,6,7,8,9 -- -m ncls train \
  configs/training/runs/metal-budgeted-hybrid-pilot.yaml \
  --devices 5,6,7,8,9 --resume artifacts/metal-budgeted-pilot/hybrid/checkpoint.pt \
  --stop-at-step 128 \
  --output artifacts/metal-budgeted-pilot/hybrid/checkpoint.pt

bash scripts/run_falcor_python.sh --gpus 5,6,7,8,9 -- -m ncls train \
  configs/training/runs/metal-budgeted-hybrid-pilot.yaml \
  --devices 5,6,7,8,9 --resume artifacts/metal-budgeted-pilot/hybrid/checkpoint.pt \
  --output artifacts/metal-budgeted-pilot/hybrid/checkpoint.pt
```

direct 使用 `metal-budgeted-direct-pilot.yaml` 和 `artifacts/metal-budgeted-pilot/direct/checkpoint.pt` 重复相同顺序。不要并发运行两者；这会改变 reference/显存竞争口径。

## 观察与停止条件

进度与 JSONL 分别显示：

- `loss/optimization_total`：实际反向目标；
- `loss/appearance`：逐通道 log/linear、chroma、peak、spatial-gradient 与 semantic runtime；
- `loss/proposal`：连续密度目标，可因合法 density 大于1而为负；
- `loss/proposal_weight`：实际组合权重。

因此负的 proposal NLL 不是复数 loss，也不能通过绝对值“修正”。任何 NaN/Inf、required group 零梯度/未更新、sample↔PDF 不一致或 source/query identity 漂移才是实现失败。

只读监控已有输出：

```bash
tail -f artifacts/metal-budgeted-pilot/hybrid/checkpoint.metrics.jsonl
nvidia-smi dmon -s pucvmet -d 5
```

达到cap后按预登记规则比较微小划痕的paired-UV/spatial-gradient、逐通道RGB/chroma、高光peak/energy与相同静态成本。`128/256 step`只是早期探针；实现正确时继续共同里程碑。若2048后选择仍不确定，只能按任务中预登记的单seed matched extension继续，不自动增加seed、asset refinement或模型宽度。

## 结构选择后的边界

只有 direct/hybrid 的 Linux eager 与 QAT 结果完成 failure classification 后，才冻结入选 profile并实现 Slang/package facet。随后再做 quantized Python↔Slang parity、Package@2、typed edit/asset swap与新 package 的 matched runtime。完整 runtime 仍只在 Linux/headless 一次性测量；Windows只允许有硬上限的接口 preflight，不形成 latency 结论。
