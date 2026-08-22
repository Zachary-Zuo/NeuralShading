# 随机游走参考吞吐测试

日期：2026-08-22

设备与路径：RTX 4090、Falcor 8.0 D3D12、Release `FalcorPython`、Slang 2024.1.34。测试使用三层栈、4096 个方向查询、每个方向 A/B 各 64 个样本、`maxDepth=64`，预热后重复 7 次取中位数。

| 指标 | 测量值 |
|---|---:|
| 端到端时间中位数 | 4.285 ms |
| 每秒随机游走样本数 | `1.223e8` |
| 每秒方向查询数 | `9.558e5` |

计时包含 stack/view 上传、compute dispatch 和四个 fp32 统计量 readback。这里的一个 teacher sample 是固定 `(stack, view, light)` 下的一条随机游走样本。

v0 共 `5000 × 32 × 16 × 128 = 327,680,000` 个方向 bin。若 A/B 各 64 samples，工作量约 `4.19e10` 条 teacher samples，按本次吞吐的原始计算下界约 5.7 分钟。

多 tile writer 完成后又运行了 8192-tile 缩放标定：128 bins、A/B 各 64 samples、64 tiles/dispatch，含 Falcor 启动、family/local-state 生成、teacher、四统计量 readback、14.7 MB tile 与 manifest 写盘共 3.568 秒。

后续试验表明，固定 64 spp 只够低噪声地描述简单两层材质，不能作为深层材质的 oracle 真值。随机游走参考后来又加入“直接连接到最底层界面”的下一事件估计，以降低完全透射路径的方差。因此 3.568 秒只记录旧版 kernel 的批量调度能力，不能外推自适应 oracle 或正式 `v0-train` 的实际耗时。
