# 方向响应 tile 格式 v1

一个 tile 对应一个程序生成的 `LayerStack` 和一个固定观察方向。tile 在入射半球的 128 个等立体角方向上保存随机游走参考响应 `f(view, light) × cos(light)`。这里已经乘过入射光方向的余弦，训练和积分时不能再次相乘。

## 每个分片包含的文件

| 文件 | 内容 |
|---|---|
| `metadata.json` | 格式、schema、先验版本、数据维度、采样数统计和层栈哈希 |
| `stacks.bin` | 连续排列的 752-byte `LayerStack` 数据 |
| `views.npy` | float32 `[view_count, 4]` 观察方向 |
| `light_directions.npy` | float32 `[bin_count, 4]` 等立体角入射方向 |
| `solid_angle_weights.npy` | float32 `[bin_count]` 积分权重，每项为 `2π/bin_count` |
| `index.npy` | uint32 `[tile_count, 2]`，把 tile 映射到 `(stack_index, view_index)` |
| `tiles.npy` | 可用内存映射读取的结构化记录 |

每条 `tiles.npy` 记录包含：

- `mean_a`：第一组独立随机样本的 RGB 均值，fp16 `[bin_count, 3]`；
- `mean_b`：第二组独立随机样本的 RGB 均值，fp16 `[bin_count, 3]`；
- `count`：A、B 每一组各自使用的样本数，uint16 `[bin_count]`。

A 和 B 使用彼此独立的随机数流。生成器先用 fp32 累积，只在写盘时把最终均值转为 fp16。自适应采样会分批增加样本，直到 A、B 两组的逐通道相对标准误差第 95 百分位都达到目标，或者样本数到达 `max_samples` 上限。

小规模试运行命令：

```powershell
./scripts/run_falcor_python.ps1 ./datagen/gen_tiles.py --stacks 4 --views 2 --bins 128 --max-samples 512
```
