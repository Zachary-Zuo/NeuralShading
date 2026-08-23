# 历史实验报告

本目录保存迁移前 v0 实验的人工报告和轻量 JSON 指标，用于追溯“精确顶层界面 + LTC 残差瓣”基线的 6.73% median / 31.20% p90 结论以及后续诊断。

文件名和原始字段中可能出现迁移前的 `teacher`、`oracle`、通用 `ClosurePacket`、旧路径或旧命令。它们只表示当时的实验实现，不是当前接口。当前代码统一使用随机游走 `reference`、逐样本 `direct fit`、backend-specific `ScatteringState` 和 `MethodBundle`；可执行入口以根 `README.md`、`docs/` 和 `TESTING.md` 为准。

报告中的 `.npy`、`.npz`、`.pt` 和生成图像不进入 Git；需要复现实验时应使用当前数据合同重新运行，而不是让旧 writer 复活。
