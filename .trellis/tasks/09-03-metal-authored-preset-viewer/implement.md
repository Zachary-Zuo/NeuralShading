# Metal authored preset viewer 与自动匹配：实施计划

## 0. 开发前检查

- [x] 使用 `trellis-before-dev` 重读 project/core/viewer 与相关 data/learning deployment specs，并重新确认 Windows 完整环境。
- [x] 核对 dirty worktree与任务允许文件；不修改 `external/`、训练配置、checkpoint、模型或 shader 数学。
- [x] 用正式 loader 复核 registry 与 step 20000 checkpoint，只读取，不 resume/训练。

## 1. ViewerMaterialCatalog 合同

- [x] 新增 schema、Python model/reader/writer、canonical identity、path containment 与 exact hash validation。
- [x] 复用 package typed blob/resource/sampler 和 program/asset/instance validators，不改变 `ScatteringPackage@2` identity/behavior。
- [x] 定义 registry-derived taxonomy、reference/method bindings、typed edit writes 与 checkpoint preview metadata。
- [x] tests 覆盖 unknown fields、duplicate/hanging binding、tamper、unsafe path；导出使用无 `catalog.json` 的 staging 和最终原子发布。

**Review A**：catalog 无手写 Metal entry/参数清单，也不包含训练 lifecycle 或 UI widget state。

## 2. Catalog exporter

- [x] 新增显式 CLI/脚本，从 registry 机械恢复 692 authored snapshots/parameter views。
- [x] 正式 MDL provider 准备 runtime artifacts与 argument-write descriptors。
- [x] step 20000 `compile_program` 一次；asset 按 identity cache；为 692 authored states 生成 initial instances和 method-write descriptors。
- [x] 写 checkpoint step/phase、entry/component identities、files/hashes，自加载成功后原子发布到新 ignored output。
- [x] 输出 program/asset/instance/reference count、52个texture sets与hard-linked存储模式；不为日志递归扫描logical bytes，不执行任何训练。

**Review B**：692 entries 完整；shared program 只有一个 runtime identity；任一条失败不留下可加载 partial catalog。

## 3. C++ loader/resolver/cache

- [x] 在 Python `material_catalog.py` 与既有 C++ `MdlReference.*` 中严格加载 catalog，并组合现有 reference/program/asset/instance binding。
- [x] 新增 identity resolver，按 export/base snapshot/component binding匹配，不用 display/index。
- [x] program/asset/reference resource按 identity复用；切换 entry 不重复创建相同 program runtime。
- [x] unit/static tests覆盖 registry/checkpoint mismatch、tamper、unsupported类型与 cache collision。

## 4. Linked typed editor 与事务

- [x] 在既有 `ReferenceSource`/`NclsViewer` 生命周期中加入 linked committed/candidate transaction。
- [x] 通用 typed value normalizer/editor 覆盖 bool/int/enum/float/float2/color、hard/soft/unbounded input与 reset。
- [x] 同一 normalized value patch reference argument block和 neural raw buffer，并调用现有 instance compiler。
- [x] 计算/显示 `ViewerMaterialState@1`；linked mode 禁用 per-slot 私有编辑与 package 手选。
- [ ] 失败注入验证旧左右 binding、edit-state、capture identity与 accumulation不被 candidate 污染。

**Review C**：coordinates/frame 与其他 responsibility 使用相同 descriptor path；C++ 不按 Metal 参数名写业务 switch。

## 5. UI 与 replay

- [x] 实现 family/metal/finish/searchable preset、responsibility groups、typed controls、reset与诊断信息。
- [x] 默认 launcher 载入 catalog，linked mode 隐藏旧 package/manual slot controls。
- [x] step 20000 显示 `joint-appearance` 和 authored/edited-preview，默认右侧 evaluator/deferred。
- [x] capture/scene 保存 catalog/entry/edit/reference/checkpoint/component identities；capture v4 保留旧 reader/replay 路径。

## 6. 验证

- [x] unit/static：schema/export/692 coverage、typed bytes、edit hash、resolver、cache、transaction、capture/replay、无 renderer family branch。
- [ ] GPU：跨 metal/finish/texture-set切换，float/bool/enum/vector/color及 coordinates/frame 的 argument/raw/compiled probes，输出 finite。
- [x] viewer Release/headless：692-entry catalog默认启动，左右slot ready，1 spp capture/capture identity与三张EXR finite，Falcor clean。
- [ ] viewer interactive：搜索、跨texture-set快速切换、编辑/reset与failure rollback人工验收。
- [x] `compileall`、受影响 regressions、`git diff --check`、Falcor clean。

## 7. Scope guard 与 rollback

若实现需要修改 training recipe/checkpoint/model/source adapter、Metal Python/Slang 数学或 scattering ABI，立即停止并回 planning。viewer catalog失败时保留旧六项 catalog和独立 package入口，不使用重训或手写映射修复。

## 8. 2026-09-03 验证证据

- 正式`artifacts/viewer/metal-step00020000/catalog.json`：692 entries、145 rejected cutout、1 shared program、52 texture sets、692独立asset/instance identities。
- 全量materialize为11分36秒；旧实现现场速率8.44秒/entry且ETA 1小时32分。已有catalog命令含Conda启动为4.42秒。
- 原失败`medium_pitted_steel.pit_texture_selection`严格验证为`MediumPits`字符串；`Silver_Knurling.texture_scale`规范化为共享`0..2`range。
- `tests/unit`：214 passed；Release build成功；默认full-catalog headless capture两slot均ready，slot 0为1 spp reference、slot 1为deferred neural，slot/difference EXR全finite。
- 未运行训练、未修改checkpoint、训练config、Metal model/source adapter或evaluate/sample/pdf/prepare数学。
