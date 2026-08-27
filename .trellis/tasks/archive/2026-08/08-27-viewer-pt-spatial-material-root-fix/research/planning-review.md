# 规划审阅记录

## 审阅结论

2026-08-27 已完成并逐项审阅 `prd.md`、`design.md` 与 `implement.md`。用户在看到最终规划摘要后明确回复“开始”，批准进入实现。

## 冻结摘要

- 目标是审计并根本修复 viewer PT 的空间 surface interaction 数据链，不接受曝光、固定 UV/LOD、禁用过滤或替换资产等表面修补。
- source reference PT 与 package PT 将共用 viewer-owned hit/UV/frame/footprint helper；deferred 保持独立 transport，但以相同公共 scattering context 约定做字段 parity。
- 实施先建立同像素 surface probe，具体修复由 UV、footprint/LOD、material identity 与采样结果证据决定。
- 高对比 UV/mip fixture 是 hard regression gate；walnut、denim 与 200k neural capture 是真实资产视觉证据。
- 不重训 200k checkpoint、不改变 package/math identity、不修改 `external/Falcor`，除非根因证据推翻当前 viewer-side 假设并触发重新规划。
- 完成时运行 scoped/full tests、Release viewer build、真实 capture、上游 clean 检查，重启 viewer，归档并创建 scoped 本地 commit，不 push。

## 连续执行授权

`task.json.meta` 已记录 task-scoped continuous execution、scoped local commit 与只在范围扩张/破坏性操作/外部阻塞时停止的策略。本次最终规划没有超出用户批准的“根本性检查和根本性修复”边界。
