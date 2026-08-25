# sampler head 无梯度：break-loop 分析

## 现象与最早失效点

`nvidia-diffuse-ggx9` 的 sampler-only 训练连续三次在 `loss.backward()` 前得到“不需要梯度”的 loss。逐层探针把最早失效点定位到 sampler head 的 64-wide Slang dot，而不是 sampler objective、state join 或 PDF：

- `nvidia_sampler_w`、`nvidia_sampler_b` 及其 broadcast view 均为 `requires_grad=True`；
- 已 detach 的 shared hidden 为 `requires_grad=False`，符合 sampler-only 冻结合同；
- `nclsUnifiedDot64` 先被冻结的 prepare/evaluator 路径调用后，再用可训练 sampler 权重调用，输出仍为 `requires_grad=False`；
- 将完全相同的算术改由此前未使用的 `nclsUnifiedPaperDot64Out` callable 身份调用，head、joined state、PDF 和 sampler 权重梯度全部恢复且有限。

调用顺序探针进一步得到：

| callable 的首次调用 | 后续用可训练权重调用 | 结果 |
|---|---|---|
| 冻结参数 + detach 输入 | 是 | 输出不进入 autograd graph |
| 可训练参数 + detach 输入 | 是 | 输出进入 autograd graph，梯度有限 |

因此问题不由 tensor 切片、detach 本身或 PDF 公式引起，而是 SlangPy `0.43.1` 的 Torch callable wrapper 按 callable 身份缓存首次观察到的可微参数掩码。相同 Slang 函数名跨“冻结 shared/evaluator”和“仅训练 sampler head”两种角色复用时，先发生的冻结调用污染后续训练调用。

## 根因分类

- **直接原因**：`nclsUnifiedDot64` 同时承载 prepare、冻结 evaluator state 和可训练 sampler head；sampler 阶段的首次调用来自冻结路径，缓存了无可微输入的 wrapper。
- **根因类别**：第三方 AD bridge 的 callable specialization / cache identity 与本项目训练阶段复用方式不匹配。
- **促成因素**：单一生产 Slang 源码的要求被误读为“所有同形状层也必须共用同一公开 callable 名称”；代码没有把训练角色的可微参数掩码当作 ABI 的一部分。
- **为什么之前的尝试无效**：调整 loss、detach 范围、join 或 PDF 都发生在最早失效点之后；只要 head 已经没有 `grad_fn`，下游公式无法恢复梯度。重复重跑同一路径也保持相同的首次调用顺序。

## 修复合同

1. 继续只保留一份 Slang 数学实现；不同训练角色用不同的公开 wrapper callable 身份，例如 prepare、evaluator-state、NVIDIA sampler head、LTC sampler head 分开命名，wrapper 内可转调同一基础算术。
2. 具有不同 active-gradient mask 的重复 diagnostic 层也保持独立 callable 身份；不能仅凭输入/输出宽度复用名称。
3. sampler-only 训练必须断言：shared、latent、evaluator 参数冻结；目标 sampler head 输出和 PDF 需要梯度；backward 后只有目标 head 获得有限非零梯度。
4. 回归测试同时覆盖 cold order 与真实 warm order：先执行冻结的 deployment/evaluator 路径，再执行 sampler-only 路径。只测“可训练调用恰好最先发生”会漏掉本问题。

## 系统性扩展检查

需检查所有 SlangPy Torch 可微入口，而不只修当前 GGX head：

- realtime prepare 与 evaluator 的同宽层；
- NVIDIA GGX9 与 LTC-K2 两个 sampler head；
- paper diagnostic 中连续三个 64-wide 层；
- evaluator 训练切换 sampler 训练时的同进程调用顺序；
- 新候选若在不同阶段改变同一 callable 的可训练参数集合，也必须采用角色级身份或显式证明 bridge 不受首次调用影响。

本规则只约束 SlangPy 训练桥接；Falcor/viewer 的生产推理仍 include 同一 core，不因此产生第二份数学前向。

