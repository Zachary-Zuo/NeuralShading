# 实施计划：vMaterials preset catalog 与 capability audit

## Phase 0：环境与基线

- [x] 按 `.trellis/spec/project/dev-environment.md` 运行 G/E/FW/W 探针并报告本会话状态。
- [x] 读取 `trellis-before-dev` 注入的 project/core/data 规则，确认没有 source-specific producer 或 query 分支。
- [x] 保存现有 6 个 `assets.json` records 的 canonical 对照，确认 bridge、SDK、source root 和 11 个 module 可用。

## Phase 1：SDK discovery 与 capability contract

- [x] 在 C++ bridge 增加 SDK-authoritative `discover` 命令与 versioned discovery JSON。
- [x] 为 compiled material 与三个 sub-expression 输出稳定 hash。
- [x] 审计 `geometry.cutout_opacity`，让 catalog artifact 可表达 unsupported cutout，但保持 emission/volume/displacement 等现有 fail-closed。
- [x] 增加集中 pixel type mapping，保真 materialize `Rgb_16`/`Rgba_16`，并让 generic Falcor/viewer binder 支持 `RGBA16Unorm`；不得按 cutout 状态跳过纹理。
- [x] 区分 metadata-only inspection artifact 与 decoded runtime artifact，修正 viewer/parity 的 artifact 获取路径。
- [x] 在 Python bridge/artifact loader 中验证新字段，并提供正式 runtime support assertion。
- [x] 更新 `MdlReferenceProgram.compile_material()`，在 payload 构造前拒绝 punched cutout。
- [x] 增加 discovery、hash、opaque/cutout 与 emission 回归测试。
- [x] 增加 `Rgba_16` payload size、Uint16 typed binding 和无 8-bit 量化回归测试。

## Phase 2：family catalog 与生成器

- [x] 定义 11-family declarative specification、预期逐 family 数量和 primary exports。
- [x] 实现 `families.json` schema 与 catalog loader。
- [x] 改造生成器：SDK discovery、172 个独立 artifacts、有效 artifact 复用、`tqdm`、资源去重、signature/hash/capability 分组。
- [x] 实现 11/172/164/8、唯一性、资源 containment、primary entry 与旧 6 条稳定性检查。
- [x] 实现临时写入、重新加载验证、原子替换和 check-only 确定性模式。
- [x] 增加缺失/重复 preset、非法资源、错误 capability、unsupported locator 与 manifest tamper unit tests。

## Phase 3：全量本地 audit 与 manifests

- [x] 构建 Release MDL bridge。
- [x] 在新的 artifact root 上运行 172-entry audit；中断时只从已验证 entry 继续。
- [x] 生成并检查 `families.json` 和扩展后的 11-entry `assets.json`。
- [x] 核对逐 family 数量、总数、164/8 runtime 边界、资源闭包大小与 unsupported 原因。
- [x] 用 check-only 模式重建并确认输出确定性。

## Phase 4：验证与文档

- [x] 运行 targeted unit tests 和 manifest/schema tests。
- [x] 运行 bridge emission fail-closed、cutout catalog/runtime fail-closed 与 MDL native/current-Falcor 回归。
- [x] 运行与改动相关的 integration/reference smoke；不扩大为训练或 viewer capture。
- [x] 更新 data MDL spec、reference README 和首批 cohort 研究文档。
- [x] 运行 `git diff --check`，确认未修改 NVIDIA source tree、Base/Automotive assets 或无关用户文件。

## 验证命令

```powershell
nvidia-smi --query-gpu=name --format=csv,noheader
conda env list
Test-Path external\Falcor\build\windows-vs2022\bin\Release\python\falcor\falcor_ext.cp310-win_amd64.pyd

.\scripts\build_mdl_reference.ps1 -Configuration Release

conda run -n neural-shading python tools/reference/generate_mdl_vmaterials_manifest.py `
  --refresh-artifacts `
  --artifact-root build/mdl-reference/vmaterials-preset-audit-v1

conda run -n neural-shading python tools/reference/generate_mdl_vmaterials_manifest.py `
  --check `
  --artifact-root build/mdl-reference/vmaterials-preset-audit-v1

conda run -n neural-shading python -m pytest `
  tests/unit/test_mdl_source.py `
  tests/unit/test_mdl_vmaterials_catalog.py `
  tests/unit/test_mdl_reference_boundary.py -q

conda run -n neural-shading python -m pytest `
  tests/gpu/test_mdl_hlsl_feasibility.py `
  tests/gpu/test_mdl_native_crosscheck.py -q

git diff --check
```

命令行参数以实施后的最终 CLI 为准；若名称调整，必须同步脚本 help、README 与本清单。

## Rollback Points

- bridge 合同未通过测试：不运行全量 audit，不更新 manifests。
- 任一 preset artifact 无法分类：保留已完成 build artifacts，停止在 manifest 原子替换之前。
- 旧 6 条发生非预期 identity/audit 漂移：停止并定位 bridge/source 差异，不覆盖 `assets.json`。
- 164/8 capability 数量不符：按 implementation defect 或 source evidence 变化分类，回到规划，不修改门槛包住结果。
