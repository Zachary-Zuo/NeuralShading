# 统一训练与部署架构

项目由原生 source/reference、GPU 在线数据、方法、公共训练和部署组成。源材质保留自己的参数、图和资源，`SourceSnapshot` 记录权威语义；不能将非层模型先反演成 LayerStack 后当作 GT。

```text
python -m ncls train GPU_LIST --config YAML
  launcher + runtime + RunPaths
    → 配置解析（base / method / data / recipe）
    → PipelineOnlineDataSession + TrainingEngine
         Method：model / objective / source adapter / lifecycle / compiler
         数值 validation / checkpoint / TensorBoard
         统一图像 hook → Windows renderer 或 Linux 空实现
    → TrainingCheckpoint → validate / eval / export
    → ScatteringPackage → Windows viewer
```

## 源码责任

| 位置 | 责任 |
|---|---|
| `runtime.py`、`launcher.py`、`ddp_worker.py` | 先配置原生库和物理设备，再导入训练；自动 DDP |
| `runs.py` | 一次分配 run，所有输出消费同一组路径 |
| `core`、`source_materials`、`references` | 材质、原生 source、reference 语义与资源合同 |
| `data` | host 并行、GPU residency、reference 调度、队列、lease、cursor |
| `learning/methods/nvidia`、`learning/methods/metal` | 各方法的模型、数据适配、objective 和编译实现 |
| `learning/training` | 配置、phase engine、DDP、checkpoint、事件与观察 |
| `visual_eval` | 公共进程内图像接口、空实现和 Windows 实现 |
| `bundle`、`viewer`、`apps/viewer`、`shaders/ncls` | 通用部署格式、资源绑定和共享 renderer |

`Method` 直接提供 `create_trainable`、`training_objective`、`create_source_adapter`、生命周期和编译方法；注册表只绑定短 key。没有 definition/facet 转发链，也不为方法创建专用训练入口。

## 在线数据与恢复

正式训练不保存 batch/corpus。host worker 只执行可复制 CPU 工作，rank 主进程拥有 CUDA/Falcor。`PipelineOnlineDataSession` 按 logical ID 交付有界队列中的 batch，lease 保护活跃资源，checkpoint 前先 drain。

样本由 source、query recipe、route、seed、rank partition 和 cursor 决定；设备编号、预取或日志属于运行设置。checkpoint 记录运行来源，但不把完整执行计划或源码 hash 变成恢复门禁。

Linux 多卡使用 phase-local PyTorch DDP reducer，NCCL 汇总梯度与数值指标，低频 Gloo 控制组传递 rank state 和写出结果。完整 checkpoint 与所有运行文件由 rank 0 写入统一 run。

## 观察与部署

checkpoint、数值 validation、图像 eval 和日志独立调度。Windows 同步渲染当前模型，结果进入本 run 的 TensorBoard；Linux 在相同 hook 上调用空实现，保留数值 validation。空实现不触发模型复制、图像文件、渲染器、GPU 操作或额外 collective。

eval/export 直接读取同一当前 checkpoint。普通导出不要求 formal、complete 或梯度覆盖标签。`ScatteringPackage` 的资源大小、ABI、哈希和实际 shader parity 仍用于保证跨语言绑定正确；它们不宣称模型质量。

新成果都在 `outputs/<config>/<run>/`。`artifacts/` 中旧图像保留原位置，旧训练权重无需迁移；根仓库不提供历史权重兼容层。
