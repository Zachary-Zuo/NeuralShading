# NeuralShading

NeuralShading 研究如何把保持原生语义的源材质编译为随机访问、运行成本有界的 neural material program。源材质族用自己的 reference 产生 GT；目标运行时用小型 MLP 实现 `evaluate(wo, wi)`，通过 `prepare()` 复用同一着色点的编码，需要方向采样时提供匹配的 `sample()/pdf()`。

当前 reference 支持 LayerStack、MERL、OpenPBR、MaterialX 和 MDL。正式训练只在 GPU 上在线产生方向查询，不保存训练 batch。当前方法为 `nvidia` 和 `metal`，共用配置解析、数据调度、训练、checkpoint 与 TensorBoard。

## 开始训练

创建唯一环境，按[部署说明](docs/reference_backend_deployment.md)构建当前平台的 reference backend：

```powershell
conda env create -f environment.yml
conda run -n neural-shading python -m pip install -r requirements-torch-cu128.txt
conda run -n neural-shading python -m pip install -e .
```

激活 `neural-shading` 后，Windows/Linux 使用同一个 Python 入口：

```bash
python -m ncls train 0 --config configs/training/runs/nvidia-layer-stack-smoke.yaml
# Linux 多卡：自动启动一个 DDP 作业
python -m ncls train 0,1 --config configs/training/runs/nvidia-layer-stack-smoke.yaml
tensorboard --logdir outputs
```

GPU 编号只在命令中指定。每次新训练自动建立 `outputs/<config文件名>/<run-id>/`，checkpoint、TensorBoard、eval、导出和日志都在其中。明确 `--resume <checkpoint>` 才续写原 run。

Windows 训练中的图像对照直接进入 TensorBoard，reference 默认 128 spp，可在 YAML 调整。Linux 保留数值 validation 和同一个图像 eval 接口，当前图像实现为空操作。

## 目录

| 目录 | 内容 |
|---|---|
| `src/ncls` | Python 源码；方法位于 `learning/methods/nvidia/`、`learning/methods/metal/` |
| `apps/viewer`、`shaders/ncls` | Windows viewer 与共享着色器 |
| `configs`、`references` | 实验配置、源材质/reference 清单 |
| `assets`、`external` | 原始资产、锁定的第三方源码与 SDK |
| `build` | 可重建的构建和编译缓存 |
| `outputs` | 按 config/run 聚合的训练成果，删除前自行确认 |
| `artifacts` | 可清理的临时研究产物；旧 viewer PNG/EXR 原地保留，供后续模型分析 |
| `docs`、`.trellis` | 稳定文档、规范和任务记录 |

`artifacts` 不再承载新训练权重或默认部署依赖。本次架构重置不迁移旧成果，也不提供旧 checkpoint 兼容；后续实验从新架构重新训练。

## 文档

- [训练、续训、验证与导出](docs/learning.md)
- [架构](docs/architecture.md)、[仓库边界](docs/repository_policy.md)
- [源材质语义](docs/material_scope.md)、[在线数据](docs/data.md)
- [研究目标](docs/realtime_material_compilation.md)、[实验记录](docs/research/experiment_log.md)
- [Windows viewer](apps/viewer/README.md)、[测试与 Linux 验证交接](TESTING.md)
