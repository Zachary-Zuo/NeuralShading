# step 200k 正式记录决定

## 决定

2026-08-27，用户确认训练已经进入缓慢收敛区间，要求不再等待原冻结 recipe 的 300k 结束，停止运行并按 step 200k 记录。本任务因此以 `checkpoint.step00200000.pt` 作为唯一正式经验结果；所有训练统计、验证、导出、方向评测和 viewer 证据都必须指向该 checkpoint，且不得把 200k 之后只有日志、没有 checkpoint 的尾段纳入结论。

## 身份与结论边界

- 原配置 `nvidia-rta2024-materialx-formal-300k-stage100k@1` 及其 SHA-256 保持不变，用于证明实现能够表达论文公开的 300k schedule；不创建 checkpoint 不兼容的伪 200k 配置。
- 实现层仍可称为 NVIDIA RTA 2024 公开方法的 `functional reproduction`；本次经验结果只能称为“用户在慢收敛区间冻结的 200k 观测”，不能称为“300k formal protocol 已完成”。
- 训练实际在收到决定时已越过 200k。报告必须登记实际日志尾点和排除数量，以证明正式数值确实按 200k 截断，而不是选择性隐藏轨迹。
- 该决定替代 `prd.md` 中 A8 对本次运行必须完成 300k 的旧验收要求；其余 correspondence、数学、数据、lifecycle、package、PT 和跨后端验收不变。

## 冻结证据

- checkpoint：`artifacts/nvidia-faithful/materialx-formal-300k/checkpoint.step00200000.pt`
- SHA-256：`ee3e6fb3bf105008247348989857f81801a9be992a59f90b07cb81eca4fe12fe`
- 配置 SHA-256：`7d18f631d709772953a42d9d5d2760333d183c60dcf21ff42b89019df2680edf`
- step 200k validation：evaluator log1p L1 `0.002758024726063013`；sampler forward KL `0.32063835859298706`；sampler valid fraction `0.9779692888259888`；总 loss `0.32339638471603394`。
- 截止 step 200k 的峰值显存：`4,795,763,712` bytes。
