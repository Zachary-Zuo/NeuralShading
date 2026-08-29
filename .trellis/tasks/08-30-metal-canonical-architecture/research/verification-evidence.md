# 子任务1验证证据

## 环境

完整Windows开发机；GPU为RTX 4090；存在唯一`neural-shading` Conda环境、锁定Falcor Python构建与Release viewer构建。所有Python/pytest命令均经该环境执行，Falcor GPU测试经项目launcher执行。

## 静态与自动测试

- `python -m compileall -q src tests`通过；
- unit最终为`150 passed`；
- Falcor GPU为`33 passed`；
- integration为`3 passed`；
- legacy denylist仅命中明确拒绝`ScatteringPackage@1`的负向unit fixture；
- `git diff --check`通过；`external/Falcor`工作树为空状态。

## 完整方法的小预算训练验证

smoke只缩小step、batch或tile，不删除phase、encoder、decoder、evaluator、sampler、loss或artifact compiler：

- LayerStack两phase：bootstrap loss `0.232983`，finetune loss `0.198493`，reload evaluate `0.191624448`；
- 固定MDL effect pigment两phase：bootstrap loss `0.83024`，finetune loss `0.805769`，reload evaluate `0.763452172`；
- MaterialX使用完整4096²、13 mip资产模型并以batch 4运行两phase：bootstrap loss `0.203345`，finetune loss `0.131737`，reload evaluate `0.106681421`。

这些值只证明数值有限、梯度链完整、短程可下降，不作为质量门或formal结果。

## Package 与viewer

- LayerStack canonical package ID以`917b7f9e9913`开头，MDL canonical package ID以`3cf655`开头；二者均通过Python package v2严格验证；
- Release viewer由`scripts/build_viewer.ps1`完成构建，overlay在构建后清理；
- headless C++ smoke从canonical LayerStack package自身目录扫描，日志记录`Accepted ScatteringPackage 'nvidia-neural-appearance' (917b7f9e9913)`，说明module closure、typed blobs、samplers和GPU parity均通过；
- capture v4成功写入`artifacts/08-30-metal-canonical-architecture/viewer-loader-smoke/`，并确认`studio-v2.json`复制后的scene/environment/material/camera/lighting/display均生效。
