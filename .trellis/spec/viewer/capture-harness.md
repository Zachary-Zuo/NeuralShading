# Viewer capture harness 合同

## 1. Scope / Trigger

凡是修改 viewer 的 headless replay、EXR/PNG capture、PT 累计停止条件、difference/comparison 输出资源或 capture manifest 时适用。本合同用于保证诊断产物可重复比较，不随临时 `--frames`、旧 replay 的 spp 或 composite 布局漂移。

## 2. Signatures

```text
NclsViewer --replay <capture.json> --headless --capture <output.json>

capture target:
  path-tracing slot spp = 1024
  deferred slot spp = 0（确定性单次求值）

output extents:
  slot-0.exr       = view_resolution
  slot-1.exr       = view_resolution
  difference.exr   = view_resolution
  comparison.exr   = resolution
```

`resolution = [outputWidth, outputHeight]` 是双 panel composite 尺寸；`view_resolution = [floor(outputWidth / 2), outputHeight]` 是单个 slot 的尺寸。manifest 同时记录 `difference_resolution = view_resolution` 与固定的 `reference_spp = 1024`。

## 3. Contracts

- 所有 ready 的 path-tracing slot 必须恰好累计到 1024 spp 才允许导出。最后一帧的 sample 数必须截断到剩余预算，不能越过 1024。
- `--frames` 只可延长 headless 运行；较小的值不得让 capture 提前于 1024 spp。达到 1024 后 PT 停止累计，后续 benchmark frame 不改变导出图像。
- replay 中历史 `reference_spp` 不决定新导出的 spp；稳定 harness 始终重新累计到 1024。
- `difference.exr` 必须由单 panel extent 的独立 linear 纹理导出，并以单 panel 局部 UV 对两个 slot 做逐像素差。不得从全宽 `comparison` 纹理截取或复用全宽 UV。
- 交互式 difference 显示仍遵守固定 50/50 panel：两个 panel 各自显示同一份正确比例的差分，divider 不参与差分采样。

## 4. Validation & Error Matrix

| 条件 | 必须行为 |
|---|---|
| 任一 ready PT slot `< 1024` spp | 阻止交互 capture；headless 继续累计 |
| 最后一帧剩余 spp 小于 `samples_per_frame` | 只发射剩余 sample，最终恰为 1024 |
| ready PT slot 已达 1024 spp | 停止该 slot 的继续累计 |
| slot 为 deferred | 保持 `slots[i].spp = 0`，不虚构 PT sample |
| 总输出宽度为奇数 | difference 使用 `floor(outputWidth / 2) × outputHeight`，divider 不导致无符号坐标下溢 |
| 两个 slot 未同时 ready | 不导出 difference；保留 partial slot capture 语义 |

## 5. Good / Base / Bad Cases

- Good：`resolution=[1280,720]` 时两个 slot 与 difference 都是 `640×720`；PT slot 均为 1024 spp，deferred slot 为 0 spp。
- Base：旧 replay 记录 16 spp，重新执行仍输出 1024 spp，新 manifest 写回固定目标。
- Bad：slot EXR 是 `640×720`，difference 却从 `1280×720` composite 导出；图像查看器会把单 panel 内容横向拉伸。

## 6. Tests Required

- `tests/unit/test_viewer_slots.py`：断言 1024 spp 常量、最后一帧截断、capture 门槛、difference 独立 view texture 与 panel-local UV。
- `scripts/build_viewer.ps1 -Configuration Release`：编译 C++ resource binding 与真实 `Composite.cs.slang`。
- headless capture：读取 EXR header，断言两个 slot 与 difference 的 width/height 完全相同；manifest 中 ready PT slot 的 `spp` 为 1024。
- 构建与 capture 后确认 `external/Falcor` 工作树干净。

## 7. Wrong vs Correct

```cpp
// Wrong：difference 复用双 panel composite，shape 与 slot 不同。
mpComparisonLinear->captureToFile(..., differencePath, ...);

// Correct：difference 使用单 panel 独立资源。
mpDifferenceLinear->captureToFile(..., differencePath, ...);
```

```slang
// Wrong：用全输出 UV 采样单 panel，并把结果铺满全宽。
gSlot0.SampleLevel(gLinearSampler, outputUv, 0.0f);

// Correct：先换算 panel-local UV，再逐像素比较。
gSlot0.SampleLevel(gLinearSampler, panelUv, 0.0f);
```
