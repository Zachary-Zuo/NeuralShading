# Metal authored preset viewer 与自动匹配

## 目标

基于现有 Metal registry、canonical MDL artifacts 与 step 20000 checkpoint，交付一个无需手选 JSON 的 viewer 材质入口：直接选择 692 个 authored preset、编辑当前已可表达的 typed 参数，并让 reference/neural 两侧自动同步。只做 UI 与必要的 viewer deployment/binding，不训练或修改材质算法。

## 需求

- R1：生成 `ViewerMaterialCatalog@1`，机械索引 registry taxonomy/parameter views、MDL reference artifacts 与 checkpoint 的 shared program/assets/instances。
- R2：UI 提供 family/metal/finish/searchable preset selector，并按六个 responsibility 分组生成 float/bool/int/enum/vector/color controls。
- R3：coordinates/frame 与其他参数一样由 descriptor 驱动，不新增专用 widget-to-model 逻辑；无 range 数值使用 typed input。
- R4：单一 `ViewerMaterialState` 同时 patch MDL argument block 与 neural raw instance，并调用现有 `nclsCompileMaterial` 更新右侧。
- R5：preset/package 自动匹配使用 source/program/asset/instance identity，不依赖 display name 或 UI index。
- R6：candidate 两侧全部通过后才 commit；失败保留旧画面与 state identity。
- R7：step 20000 的 method/step/phase 与 authored/edited-preview 状态可见，默认使用 evaluator/deferred preview。
- R8：旧 catalog/package/manual slot/CLI/capture/replay 入口保持可用，viewer 不依赖 Python/PyTorch/MDL runtime DLL。

## 验收标准

- [ ] **AC-A1｜需求交付｜来源：父任务 AC1**：默认 UI 不打开 JSON chooser即可搜索并选择 692 个 opaque presets，cutout rejected entries 不进入 selector。
- [ ] **AC-A2｜语义正确性｜来源：父任务 AC2/AC3**：catalog entry 与 component 全部从 registry/artifact/checkpoint 机械生成；只有一个 shared neural program，asset/instance binding 可验证并按 identity 复用。
- [ ] **AC-A3｜需求交付｜来源：父任务 AC4**：代表性 entry 的 coordinates、frame、metal-core、finish、aging、coating editable 参数均出现正确 typed 控件，值域和 choice 不手写。
- [ ] **AC-A4｜需求交付与语义正确性｜来源：父任务 AC5/AC6**：选择和编辑产生单一 edit-state，固定 bytes probes 证明 MDL argument 与 neural raw/compiled instance 接收相同 typed value，右侧无需手选 package。
- [ ] **AC-A5｜语义正确性｜来源：父任务 AC7**：artifact/hash/type/shader/resource/compiler 故障不会产生单边 commit，上一有效 comparison 保留。
- [ ] **AC-A6｜正确表述｜来源：父任务 AC8**：viewer/capture 显示 step 20000、`joint-appearance` 与 authored/edited-preview，默认 evaluator preview 不声称新训练或编辑泛化。
- [ ] **AC-A7｜兼容性｜来源：父任务 AC9**：现有六项 MDL catalog、独立 `ScatteringPackage@2`、manual slots、CLI 和旧 replay tests 继续通过。
- [ ] **AC-A8｜数值/实现正确性｜来源：父任务 AC10**：Windows Release/headless/interactive 覆盖多种 preset 与 typed edit，左右 finite、快速切换稳定、capture/replay identity 一致，Falcor clean。

## 范围外

- 不修改或运行 training config/recipe/runner/checkpoint，不重训。
- 不修改 Metal model/source adapter、Python/Slang evaluate/sample/pdf/prepare、coordinate/frame 数学或 scattering ABI。
- 不承诺 edited-preview 属于训练域或具有正式泛化质量。
- 不扩展 cutout/emission/volume/displacement，不新增其他 source family cohort。
