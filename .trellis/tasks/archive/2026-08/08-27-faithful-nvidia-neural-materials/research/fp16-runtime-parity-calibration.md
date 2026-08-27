# regular FP16 runtime parity 冻结记录

## 它是什么

`ScatteringPackage` 在导出时调用通用 `MethodDefinition.package_validation()`。NVIDIA 方法据 checkpoint 的 FP32 master 独立模拟以下部署路径，并把四个固定方向的 `response_cos` 写进 `validation/parity.json`：

1. mip 0 latent texel先按 RGBA16F 量化，再按 viewer parity 的 `uv=(0,0)` 做 wrap bilinear；
2. runtime weights、MLP input、bias、乘加 accumulator、ReLU activation与输出全部按 half 模拟；
3. learned frame normalize/cross/project保持 float；
4. evaluator输出按 `half(exp(raw-3))` 量化，再经过公共 `f ↔ response_cos` adapter。

viewer 不解释 NVIDIA 私有布局，只读取统一 parity 合同并在实际 D3D12 package shader上比较。缺少 expected值时的“仅有限性检查”不能作为本方法最终部署证据。

## 容差来源与冻结

- oracle：`nvidia-rta2024-packed-fp16-cpu-emulation@1`；
- `relative_tolerance = 2e-2`；
- `absolute_tolerance = 2e-4`；
- 来源：最长64项 half accumulator的保守 `O(10^-2)` 误差包络，同时覆盖 native half FMA与拆分 half multiply/add lowering；absolute项用于约束暗通道；
- 冻结时点：300k formal启动前；不得根据 formal package的实际误差放宽。

静态证据 `test_nvidia_deployment_shader_uses_regular_fp16_mlp_path` 同时要求部署 wrapper实际调用 `nclsNvidiaNeuralPrepareFp16` / `nclsNvidiaNeuralEvaluateFFp16`，避免一个数值接近但仍走 FP32 accumulator的实现蒙混过关。

## formal 前隔离 calibration

- 2-step diagnostic checkpoint SHA-256：`a45d6a34d9b9cd2f968e55da576fa9db61f22eb4c7f7e5866203e16bdcd36a83`；
- package id：`fcef8137e1a543ceb2b3e20275ff1e09db22bc96bd716b9fc9aaa92e549c8121`；
- program runtime id：`97ac0a766857868ec80ecce096efa60a17d1cd747cfac0d1ba64944aca904cf1`；
- material asset id：`32d955598eb9926a8a2fcb077dd01f137b2e1fdff194d9073b073429eb6ff769`；
- evidence：`artifacts/nvidia-faithful/materialx-fp16-parity-smoke/viewer-dual/capture.json`；
- 结果：同一 package/runtime/material在 slot 0 `path-tracing` 与 slot 1 `deferred` 均为 `ready`，GPU时间分别为 6.554 ms与0.910 ms；package加载过程中实际 D3D12 parity probe通过。

这是 pre-formal numerical calibration和transport vertical slice，不是质量结果，也不进入正式比较。
