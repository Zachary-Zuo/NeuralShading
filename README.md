# NeuralShading

研究目标：把由有限种界面和介质组成的可编辑多层材质，编译成一个大小固定、能被实时灯光直接求值的解析参数包，并让它在未见材质组合上接近随机游走参考。

当前里程碑是最短研究闭环：

1. 定义 CPU/GPU 共用的层栈数据布局；
2. 在 Falcor 8.0 中实现并验证多层随机游走参考；
3. 生成方向响应数据；
4. 不引入预测网络，直接优化每个样本的 closure 参数，测量不同表示的上限；
5. 训练通用编译器，并把预测结果接入 Falcor 延迟着色。

随机游走参考、两套 v0 数据、CPU/PyTorch/Slang 共用的 176-byte 三槽参数包，以及第一个 38k 参数循环编译器都已跑通。

当前三槽表示由“精确顶层界面 + 两个 LTC 残差瓣”组成。它的方向域 relative-L1 为 median 6.73%、p90 31.20%，长尾仍然偏高；增加第三个残差瓣后只改善到 5.56%/25.24%。因此这套表示保留为基线，还没有最终定稿。编译器在高采样测试集上的 median 为 22.70%，说明表示和预测两边都还有工作，但预测误差更大。

## 环境

```powershell
conda env create -f environment.yml
conda run -n neural-shading python -m pip install -r requirements-torch-cu128.txt
conda run -n neural-shading python -m pytest
```

根仓库保存范围见 `docs/repository_policy.md`。Falcor 和 pbrt-v4 位于 `external/`，使用固定提交且当前没有本地修改；具体版本见 `AGENTS.md`。
