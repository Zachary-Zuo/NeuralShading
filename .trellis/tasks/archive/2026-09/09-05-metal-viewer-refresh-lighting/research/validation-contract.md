# 实施验证合同

2026-09-05 用户批准实施；完整 Windows、RTX 4090，目标 hybrid checkpoint hash 已与 PRD 核对一致。

## Sampler 数值口径（先于正式 witness 冻结）

- 独立 oracle 是既有 Python proposal，GPU 输入为同一 FP32 state/frame/wo/u；CPU 使用 float64 计算独立期望。
- 方向使用 atol=2e-5、rtol=2e-4；PDF 使用 atol=2e-5、rtol=2e-3。来源是 FP32 归一化/三角函数与 log/exp 链的累计舍入，并允许窄 lobe PDF 的条件数放大；不是模型质量门。若超差先定位到分量、frame 或折回，不改容差包住结果。
- sample→独立 GPU pdf 与 weight 恒等式使用 FP32 相同数学，rtol=2e-5、atol=2e-6。
- hemisphere normalization 使用独立均匀方向估计，误差必须落在 5 个标准误加 0.002 数值积分余量内；拒绝以有限性替代归一化。
- cooked evaluator 延续已冻结的 FP16/SNORM8 package 容差（rtol=0.03、atol=0.0005），新增 proposal preparation 单独与未 pack 的 Python state 对照。

## 删除/迁移顺序

先迁移 active catalog/handoff/source preparation producer，再删除旧 reader；先完成 active-ID shader routing，再移除 host all-materials identity 拒绝；先让 source/neural 共用 renderer，再删除旧 pass/调度/输出路径。清理项按 design §7 表逐项登记最终证据，不删除用户 artifacts。
