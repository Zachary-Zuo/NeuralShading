# Viewer capture harness 合同

## 1. Scope / Trigger

凡是修改 viewer 的 headless replay、EXR/PNG capture、PT 累计停止条件、training visual eval、difference/comparison 输出资源或 capture manifest 时适用。本合同同时区分 formal matched capture 与低开销 training diagnostic，避免把 neural 1024 spp 错误地加入高频训练 cadence。

## 2. Signatures

```text
NclsViewer --replay <capture.json> --headless --capture <output.json>

capture target:
  formal path-tracing slot spp = replay.reference_spp（正式基线为 1024）
  training-diagnostic slot 0 = reference path tracing 1024 spp
  training-diagnostic slot 1 = neural deferred 0 spp（默认）
    或显式低 spp neural path tracing（手工深度检查）
  samples per dispatch = replay.reference_samples_per_frame（仅 headless）
  deferred slot spp = 0（确定性单次求值）

output extents:
  slot-0.exr       = view_resolution
  slot-1.exr       = view_resolution
  difference.exr   = view_resolution
  comparison.exr   = resolution
```

`resolution = [outputWidth, outputHeight]` 是双 panel composite 尺寸；`view_resolution = [floor(outputWidth / 2), outputHeight]` 是单个 slot 的尺寸。manifest 同时记录 `comparison_purpose`、`difference_resolution = view_resolution`，以及每个 slot 的 `mode/target_spp/spp`。项目正式视觉基线和 training diagnostic 的 reference 均固定为 1024 spp；常规 training neural 不再被 reference target 隐式提升到 1024 spp。

## 3. Contracts

- `comparison_purpose=formal` 时，所有 ready path-tracing slot 必须恰好累计到共同 `reference_spp`。`comparison_purpose=training-diagnostic` 时，每个 PT slot 只累计到自己的 `target_spp`；deferred slot 的 target/actual 均为 0。最后一帧 sample 数必须截断到该 slot 剩余预算。
- `--frames` 只可延长 headless 运行；较小的值不得让 capture 提前于 target spp。达到 target 后 headless PT 停止累计，后续 benchmark frame 不改变导出图像。
- `reference_spp` 与 `reference_samples_per_frame` 是 headless execution contract；前者决定输出 sample count，后者只决定 dispatch batching。二者都不得进入交互 UI 或 viewer scene。
- 交互 PT 每次 dispatch 固定追加 1 spp，并可超过任何 capture target；交互手工 capture 记录 ready PT slot 当前的 matched spp，而不是强制或伪造 1024。
- quality-first deferred可在交互模式跨frame做coarse-to-fine tile预览，但headless capture必须强制`preview stride=1`并在导出前排空完整像素网格；capture manifest中的deferred ready不能来自未完成的交互coarse pass。
- `difference.exr` 必须由单 panel extent 的独立 linear 纹理导出，并以单 panel 局部 UV 对两个 slot 做逐像素差。不得从全宽 `comparison` 纹理截取或复用全宽 UV。
- 所有线性 EXR 都从 RGBA32F 资源显式写成 float32 channel。Falcor/FreeImage 的默认压缩 EXR 会落到 half；这会把合法的 `> 65504` HDR 样本写成 `Inf`，因此权威 capture 禁止使用默认 EXR export flags，也禁止用 clamp 掩盖导出溢出。
- 交互式 difference 显示仍遵守固定 50/50 panel：两个 panel 各自显示同一份正确比例的差分，divider 不参与差分采样。

## 4. Validation & Error Matrix

| 条件 | 必须行为 |
|---|---|
| formal 中任一 ready PT slot `< reference_spp` | headless 继续累计，不提前导出 |
| training diagnostic 中 PT slot `< target_spp` | 只继续该 slot，不把另一 slot 提升到共同 target |
| 最后一帧剩余 spp 小于 `reference_samples_per_frame` | 只发射剩余 sample，最终恰为 target |
| ready PT slot 已达 target | 仅 headless 停止该 slot；交互继续 1 spp/dispatch |
| 交互手工 capture 的 ready PT slot 为 0 spp 或 spp 不一致 | 拒绝 capture，要求至少一个 matched sample |
| slot 为 deferred | 保持 `slots[i].target_spp = slots[i].spp = 0`，不虚构 PT sample |
| training diagnostic 未显式记录 purpose 或每 slot target | 拒绝 capture；不能用旧 matched 语义猜测 |
| 总输出宽度为奇数 | difference 使用 `floor(outputWidth / 2) × outputHeight`，divider 不导致无符号坐标下溢 |
| 两个 slot 未同时 ready | 不导出 difference；保留 partial slot capture 语义 |

## 5. Good / Base / Bad Cases

- Good：training diagnostic 使用 reference PT 1024 spp 与 neural deferred 0 spp；两个 slot/difference 都是 `640×720`，manifest 明确记录 purpose/mode/target/actual。
- Good：formal 双 PT comparison 的两个 slot 都为 1024 spp；共同 target 语义保持不变。
- Base：replay 记录 target 1024、batch 16，headless 用 64 次 PT dispatch 到达 1024；同一 replay 在交互启动时仍为 1 spp/dispatch 且继续超过 1024。
- Bad：slot EXR 是 `640×720`，difference 却从 `1280×720` composite 导出；图像查看器会把单 panel 内容横向拉伸。
- Bad：把 `reference_samples_per_frame=16` 恢复为交互 `mSamplesPerFrame`，或用 capture target 截断交互 accumulation。

## 6. Tests Required

- `tests/unit/test_viewer_slots.py`：断言 formal matched target、training-diagnostic per-slot target、default reference 1024、neural deferred 0、headless 末帧截断、交互恒为 1、capture 门槛、difference 独立 view texture 与 panel-local UV。
- `tests/unit/test_visual_eval_worker.py`：worker 必须验证 purpose、两个 slot 的 mode/target/actual 以及输出 hash；默认请求不能把 neural 改成 path tracing。
- `scripts/build_viewer.ps1 -Configuration Release`：编译 C++ resource binding 与真实 `Composite.cs.slang`。
- headless capture：读取 EXR header，断言两个 slot 与 difference 的 width/height 完全相同；training diagnostic 中 reference PT 为 1024、neural deferred 为 0，formal PT slot 均为 1024。
- headless capture：同时断言线性 EXR 的 RGB channel 是 float32 且像素全 finite；高亮尾部保留原始值。
- 构建与 capture 后确认 `external/Falcor` 工作树干净。

## 7. Wrong vs Correct

```cpp
// Wrong：difference 复用双 panel composite，shape 与 slot 不同。
mpComparisonLinear->captureToFile(..., differencePath, ...);

// Correct：difference 使用单 panel 独立资源。
mpDifferenceLinear->captureToFile(..., differencePath, ...);
```

```cpp
// Wrong：所有模式都被 capture target 截断。
samples = min(batch, captureTargetSpp - slot.spp);

// Correct：交互连续累积；headless 才使用 target 与 batch。
if (!options.headless) return 1u;
samples = min(captureBatch, captureTargetSpp - slot.spp);
```

```slang
// Wrong：用全输出 UV 采样单 panel，并把结果铺满全宽。
gSlot0.SampleLevel(gLinearSampler, outputUv, 0.0f);

// Correct：先换算 panel-local UV，再逐像素比较。
gSlot0.SampleLevel(gLinearSampler, panelUv, 0.0f);
```
