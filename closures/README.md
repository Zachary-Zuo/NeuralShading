# 三槽解析着色参数包

当前基线把层栈响应拆成两部分：顶层界面的精确解析反射，以及穿过顶层后产生的其余多层传输响应。前者直接计算，后者用两个 LTC 残差瓣近似。固定 CPU/GPU 数据布局为 176 bytes：

| 字节数 | 内容 |
|---:|---|
| 16 | 魔数、版本、残差瓣数量和标志位 |
| 64 | 从可编辑层栈顶层直接复制的 `LayerInterface` |
| 48 × 2 | 每个 LTC 残差瓣的 RGB 幅度、二维逆尺度、三个剪切参数和角度 |

`packet.py` 定义二进制布局，`torch_eval.py` 定义可微训练求值，`packet.slang` 定义实时 Slang 求值。编译器只预测两个残差瓣的 18 个原始参数；顶层界面来自材质本身，不让网络重复学习已知量。

用 Falcor 验证全部 2048 个 oracle packet：

```powershell
./scripts/run_falcor_python.ps1 -m viewer.validate_oracle_lookup
```

相对归档的 FP16 响应，Falcor 求值的 tile relative-L1 为 median 0.017%、p99 0.024%、最大 0.099%。这个数字只证明 CPU/PyTorch/Slang 语义一致，不代表该 closure 表示已经足够准确；表示误差见 `reports/oracle_ceiling_v0.md`。
