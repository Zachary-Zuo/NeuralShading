# 逐样本直接拟合、训练与评测

## 三条路径的边界

Python 侧现在明确分成三条互不混用的路径：

- `direct-fit`：每个 tile 单独优化表示参数，测量某种固定成本表示能达到的上界；它不产生通用预测网络。
- `train`：只读取 train tile 更新网络，只用 validation 选择 best checkpoint。
- `evaluate`：从一个不可变 checkpoint 显式评测 validation 或 held-out test；训练循环不会读取 test 指标。

当前唯一注册的网络是 `legacy-ltc-k2-p1@2`，对应历史的“精确顶层界面 + 两个 LTC 残差瓣”基线。名称中的 `legacy` 和 `P1` 是有意保留的限制说明；它不是 `default` 或 `final`，不会阻止后续换成小 neural decoder 或其他表示。

## TrainingConfig

所有超参数先解析为 `ncls.training-config@1`，并将完整 JSON 与 SHA-256 写入 run。最小示例：

```json
{
  "architecture_id": "legacy-ltc-k2-p1@2",
  "representation_id": "legacy-ltc-k2@1",
  "width": 64,
  "steps": 10000,
  "batch_size": 256,
  "learning_rate": 0.0003,
  "weight_decay": 0.00001,
  "gradient_clip": 5.0,
  "validation_interval": 250,
  "checkpoint_interval": 250,
  "max_validation_tiles": 4096,
  "seed": 20260822,
  "device": "cuda",
  "deterministic": true,
  "schema_name": "ncls.training-config",
  "schema_version": 1
}
```

运行：

```powershell
conda run -n neural-shading ncls learn train `
  --dataset data\reference-v2 `
  --run artifacts\runs\legacy-ltc-k2-p1-001 `
  --config configs\legacy-ltc-k2-p1.json
```

每个 run 包含：

```text
run/
  training_config.json
  run_manifest.json
  validation_history.json
  tensorboard/
  checkpoints/
    best.pt
    best.pt.sha256
    last.pt
    last.pt.sha256
```

checkpoint 保存 architecture、representation、feature contract、dataset ID、模型/optimizer/RNG 状态和 validation 证据。loader 总是先验证 sidecar hash。

## TensorBoard

训练记录 `train/loss`、gradient norm、learning rate，以及 validation loss、median/p90/p95 relative-L1。不会写入 `test/*` tag。

```powershell
conda run -n neural-shading tensorboard `
  --logdir artifacts\runs\legacy-ltc-k2-p1-001\tensorboard
```

逐样本直接拟合也在其输出目录下记录 `tensorboard/`，便于比较不同表示与优化设置。

## 显式 held-out test

best checkpoint 确定后，单独执行：

```powershell
conda run -n neural-shading ncls learn evaluate `
  --dataset data\reference-v2 `
  --checkpoint artifacts\runs\legacy-ltc-k2-p1-001\checkpoints\best.pt `
  --split test `
  --output artifacts\runs\legacy-ltc-k2-p1-001\test_metrics.json
```

评测拒绝 dataset ID 或 feature contract 不一致的 checkpoint。test 结果是独立文件，不回写训练选择记录。

## 表示上界

例如重新测量当前 K2 基线：

```powershell
conda run -n neural-shading ncls learn direct-fit `
  --dataset data\reference-v2 `
  --output artifacts\direct-fit\legacy-ltc-k2-test `
  --split test --family ltc --lobes 2 `
  --steps 800 --restarts 3
```

输出 `ncls.representation-ceiling@1` manifest、TensorBoard 和 `parameters.npz`。这里的参数是逐 tile 直接优化结果，只能用来判断表示上界，不能当成通用材质编译器。

## MethodBundle

训练 run 与部署 artifact 分离。当前 exporter 会把显式 checkpoint、feature contract、后端 shader、validation 指标、固定 parity probe 和全部内容哈希写成一个新 bundle：

```powershell
conda run -n neural-shading ncls bundle export-legacy-ltc-k2 `
  --checkpoint artifacts\runs\legacy-ltc-k2-p1-001\checkpoints\best.pt `
  --run-manifest artifacts\runs\legacy-ltc-k2-p1-001\run_manifest.json `
  --output artifacts\bundles\legacy-ltc-k2-p1-001

conda run -n neural-shading ncls bundle validate `
  artifacts\bundles\legacy-ltc-k2-p1-001
```

P1 compiler 已有无 Python 运行时依赖的 Slang 实现：它读取 bundle 中按小端 float32 规范化的 `weights/model.bin`，执行与 PyTorch 相同的 token MLP、GRUCell、view 编码和参数 head，再生成私有的 `legacy-ltc-k2` scattering state。逐层 PyTorch/Slang parity、固定 bundle probe 和 viewer GPU parity 均已通过，因此 exporter 输出 `runtime_class=realtime`。

这个可部署状态只证明完整的 compiler+backend 链路已经跑通，不表示 K2 误差已经达标。viewer 仍只通过公共 `prepare/evaluate/sample/pdf` 合同调用它；后续把 `evaluate()` 换成小 neural decoder 时不改变数据合同、viewer 的主可见性或 MethodBundle 生命周期。
