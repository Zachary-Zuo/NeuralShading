# 设计

## Windows验证矩阵

验证分为全cohort preflight、component activation、四phase optimization、resume、package/viewer和profile六部分。activation batches由registry生成最小stratified set，确保覆盖每个texture role、schema token、recipe类别、direction chart、lobe/proposal component和parameter responsibility；四phase optimization复用这一覆盖集合以提高正确性验证效率，但仍运行full shape、真实online reference/loss/optimizer/data flow。它不是质量评测split，也不能支持最终泛化结论。

短程收敛使用固定identity的online query stream：每次仍重新运行authoritative reference，不保存response。比较预注册initial/final windows并以source-state matched bootstrap判断loss delta方向；pilot只用于冻结window长度和数值实现容差，不能在看过formal smoke后改判据。

## Config关系

`metal-full-smoke`和`metal-full-long`引用同一个method/profile/phase/loss/precision/source registry。允许差异只有每phase steps、batch sizes、validation/checkpoint/log cadence和stop cap。config diff test拒绝required component、shape、route/loss或precision漂移。

## Profile

runner输出GPU时间区间与显式sync记录。普通step的metrics在GPU累积；audit、validation和checkpoint按cadence同步。报告分解reference/group切换、asset tile encode、model forward/backward、optimizer和I/O，给Linux单GPU ETA/VRAM/磁盘估算但不设throughput hard gate。

## Linux handoff

handoff manifest绑定commit、environment/toolchain/source registry、config/checkpoint/package hashes和预期命令。Linux先运行backend doctor、registry/resource preflight和短smoke，成功后从Linux-native checkpoint开始long config。`CUDA_VISIBLE_DEVICES`必须解析为一个物理GPU；多值拒绝。

long run完成后runner生成review manifest，引用checkpoint、曲线、代表性render/sweeps、sample/PDF统计和基础bytes/time。它不触发新的训练或evaluation matrix。
