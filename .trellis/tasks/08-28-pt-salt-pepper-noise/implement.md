# 实施计划：PT 细碎亮点

## Phase 1：冻结视觉与贡献归因

- [x] 把已生成的三材质高分辨率 capture、ceramic bounce 0/1 对照及 raw EXR 登记为 baseline identity。
- [x] 在 task-scoped 诊断 build 中输出 contribution/normal AOV 与固定 histogram；验证分类直接复用同一 radiance contribution。
- [x] 对 MDL ceramic、MDL car paint、OpenPBR car paint 运行同一诊断，判定 H2 成立、H1 不是主因，把证据写入 `research/root-cause.md`。
- [x] H1/H2 判别门已通过，进入 H2 正式实现。

## Phase 2：公共 transport 根修复

- [x] H1 未成立，不引入 speculative shading-normal adjustment 或 sample rejection。
- [x] source/package 的 PathSurface 与 native tuple 保持现状。
- [x] 增加 focused unit/GPU oracle，证明 direct pool 保留 accepted native tuple、continuous/delta MIS 正确。
- [x] 抽取共享 environment CDF/PDF/MIS helper，冻结 `n_light=4`、`n_bsdf=4`，并让 host CDF 匹配双线性 radiance reconstruction。
- [x] 把 primary BSDF pool 迁成 4 条完整 path samples；secondary 使用 4+4 direct pool 与单条 indirect continuation，并在 source/package 两条 PT 对称接线。
- [x] 使用 Falcor 官方 `UniformSampleGenerator` 生成每条 path 的 material sample stream，不引入自定义 lattice sampler。
- [x] 删除 task-scoped 产品 shader AOV；只保留任务脚本、证据和公共实现/测试。

## Phase 3：验证与视觉复核

- [x] 运行 focused unit 与 GPU oracle。
- [x] 运行完整 `tests/unit`、完整 `tests/gpu` 与 `tests/integration`。
- [x] 用 `scripts/build_viewer.ps1 -Configuration Release` 构建 viewer，并确认 overlay 回退后 Falcor clean。
- [x] 生成 task-scoped LayerStack ScatteringPackage，真实运行 source/package 双 PT slot 到 1024 spp；两个 slot 均 `ready`、raw EXR finite 且结果一致。
- [x] 重跑冻结 960×540、1024 spp 的三材质 before/after；生成视觉对照和 report-only tail/RSE/cost。
- [x] 审计 OpenPBR aluminum/glass、MERL chrome、MaterialX、LayerStack；未发现本次 primary-continuation 回归，ideal glass 的 delta/内部折射高方差作为另一类 PT 限制单独登记。
- [ ] 启动 viewer 让用户现场检查；用户视觉确认前不把主验收项标为完成。

## Phase 4：知识沉淀与收尾

- [ ] 使用 `trellis-break-loop` 记录“接口正确但公共几何 transport 域/直接光 estimator 仍可产生 firefly”的根因类别与预防门。
- [ ] 使用 `trellis-update-spec` 更新 viewer path-surface / capture / shared backend 合同及 required tests。
- [ ] 使用 `trellis-check` 完成质量门，提交实现；用户确认后归档任务并记录 journal。
