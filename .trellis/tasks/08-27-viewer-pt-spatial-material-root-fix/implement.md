# Viewer PT 空间材质根因修复执行计划

## Phase 1：冻结根因证据

- [x] 将当前 screenshot、512 spp walnut capture、denim source capture 和 neural PT/deferred 对照登记到 `research/root-cause.md`。
- [x] 建立最小 surface probe，读取 primary hit 的 UV 与分项 footprint/LOD，绕过材质与 transport。
- [x] 通过 probe 明确直接根因和受影响的数据字段；记录为什么 texture loaded、direction parity、slot ready 仍未发现问题。

回滚点：只新增诊断/测试，不改变产品渲染。

## Phase 2：共享 surface interaction

- [x] 抽取 viewer-owned PT hit/surface helper，消除 reference/package 两份 triangle/UV/frame/footprint 逻辑。
- [x] 按 probe 证据修复 ray footprint 的 camera basis 共同尺度，并为非 triangle geometry写清 full-UV fallback。
- [x] 让 reference PT 与 package PT 都经 helper 构造公共 context，保持 source/backend 私有状态隔离。
- [x] 检查 deferred context 的 material ID、normalized UV gradient、frame 与 PT 约定；deferred 原合同无需产品改动。

回滚点：helper 可整体回退；source/package 数学与 artifact 不变。

## Phase 3：防复发验证

- [x] 增加 versioned analytic surface/UV/LOD GPU fixture；固定 UV、错误 V flip/frame 或强制最粗 mip都会失败。
- [x] 增加 camera basis scale、resolution/distance/angle footprint 单调性和有限性测试。
- [x] 生成 denim source reference PT local-light 高对比 capture；正确性 hard gate 使用独立 analytic oracle，真实资产指标只报告。
- [x] 生成 NVIDIA package PT/deferred 匹配局部照明的空间布局证据；相关性为 report-only。
- [x] 更新 shader ownership/unit 测试；capture schema 无需变化，确认交互累计语义不变。

建议验证命令在实施时按实际测试名收敛，至少包括：

```powershell
conda run -n neural-shading python -m pytest tests/unit -k "viewer or capture or slot" -q
conda run -n neural-shading python -m pytest tests/gpu -k "surface or uv or latent or footprint" -q
powershell -ExecutionPolicy Bypass -File scripts/build_viewer.ps1 -Configuration Release
```

## Phase 4：真实资产与全量质量门

- [x] 生成收敛的 walnut reference/neural PT、denim reference PT 和 neural PT/deferred capture；检查 raw 指标与 PNG。
- [x] 运行全量 source/reference/package 测试与 NVIDIA package parity/ABI 回归。
- [x] 运行全量 `pytest`、Release build、`git diff --check`，确认所有 `external/` 干净。
- [x] 使用 `trellis-check` 做 spec、数据流、复用和测试审查，并修正非 triangle fallback 与 UV/frame oracle 缺口。

## Phase 5：文档、viewer 与交付

- [x] 把长期 surface/UV/footprint/capture gate 更新到 `.trellis/spec/viewer/` 与稳定中文 viewer 文档。
- [x] 完成 `research/root-cause.md`，注明旧 4 spp/ready 证据失效边界。
- [x] 重启修复后的交互 viewer，让用户查看 walnut reference 与 200k neural PT；交互无 frame cap，持续累计。
- [x] 按 continuous 授权只 stage 本任务文件，创建 scoped work commit，不 push。
- [ ] 归档任务并记录 journal。
