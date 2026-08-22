# 从层栈到解析着色参数包的编译器

`model/` 把可编辑 `LayerStack` 和观察方向转换为两个 LTC 残差瓣。精确顶层界面参数直接从层栈复制，不属于网络输出。

当前的 `RecurrentCompilerBaseline` 是“学习式顺序组合”对照，不是最终研究架构。它按从顶层到底层的顺序读取 layer token，输出 18 个无约束数值，再由 `closures.torch_eval.decode_ltc_residual()` 映射到非负幅度、有界尺度、剪切和角度。

```powershell
conda run -n neural-shading python -m model.train --dataset data/v0_train --output reports/compiler_v0
conda run -n neural-shading python -m model.evaluate
```

第一次 20,000-step 训练没有出现明显的 family split 差距，但在高采样 held-out oracle 上的 median relative-L1 仍为 22.70%。当前 K2 packet 的表示上界为 6.73%，两者之间主要是预测误差。不过 6.73% 本身也尚未达标，因此应先确认残差表示，再实现结构化反射/透射算子。完整结果见 `reports/compiler_v0.md`。
