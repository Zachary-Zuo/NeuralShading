# 根本迁移记录

## 当前状态

- 阶段 0–4 已完成：合同、正式 Python package、随机游走参考数据、训练/评测/导出和 Windows/Falcor viewer 均已落地。
- 阶段 5 已完成：旧源码删除、文档入口收敛、全量回归和仓库边界审计均已通过。
- 当前 `legacy-ltc-k2-p1@2` 已有 PyTorch/Slang compiler parity，可导出不依赖 Python 运行时的 realtime `MethodBundle`；它仍只是长尾未达标的历史基线。

迁移不维持旧 import、旧 CLI 或通用 K2 packet ABI。可信算法和结论已经迁入新位置；错误、重复和过时内容由 Git 历史追溯。

## 已冻结的边界

三个业务块只通过带版本的合同交换产物：

```text
MaterialProgram → 数据采集 → ReferenceDataset
ReferenceDataset → Python direct fit / train / evaluate → MethodBundle
MaterialProgram + MethodBundle → Windows viewer
```

- `MaterialProgram` 是公共作者格式，`LayerStackIR` 是当前规范化内部 IR。
- `prepare/evaluate/sample/pdf` 是 renderer 面向拟提方法的稳定散射合同。
- `CompiledMaterial` 和 `ScatteringState` 完全由 backend 私有。
- 以后把 `evaluate()` 换成小 neural decoder、latent 或新解析表示，不改变数据集、viewer 主可见性或顶层材质接口。

## 目录迁移结果

| 迁移前 | 最终处理 |
|---|---|
| 固定 `schema/` | MaterialProgram schema、LayerStackIR ABI 和一次性 v0 reader 进入 `src/ncls/core/material/` |
| 随机游走旧源码与采集脚本 | 进入 `shaders/ncls/reference/`、`shaders/ncls/data/` 和 `src/ncls/data/` |
| 通用 K2 packet | 删除；私有实现进入 `core/representations/legacy_ltc_k2` 与对应 shader backend |
| 表示拟合脚本 | 可信部分进入 `learning/direct_fit/`，重复/过时实验入口删除 |
| 泛化 `model/` | 重写为 `learning/models`、`training`、`evaluation` 和 `export` |
| Python lookup viewer | 删除；真正应用进入 `apps/viewer/` |
| 外部 pbrt probe | 进入 `tools/reference/pbrt_probe/` |
| 根级旧测试 | 重组为 `tests/unit`、`tests/gpu`、`tests/integration/reference` |
| 根级阶段计划和 v0 操作文档 | 删除；有效结论进入稳定文档，原始指标保留在明确标注的历史报告中 |

旧 `ncls-direction-tiles@1` 数据仍能通过 `ncls data convert-legacy-v0` 一次性转换，但项目不再保留第二套 writer 或旧包级 import shim。

## viewer 验证结果

第一版 viewer 已完成：

- 单一 orbit/pan/dolly 相机与共享主可见性；
- 左侧随机游走 reference 停止移动后累积，右侧 realtime MethodBundle deferred 结果；
- 球体、shader ball、hero 物体，HDRI、方向光、点光、矩形面光；
- MaterialProgram 常量层栈编辑、方法切换、共同曝光和差异显示；
- 全文件哈希、平台/合同检查和 GPU parity 后才接纳方法；
- EXR/PNG/材质/指标/manifest capture 与命令行 replay；
- 固定三段相机路径 JSON/CSV benchmark。

真实 P1 bundle 的 capture/replay 中，reference、approximation、comparison、difference、display 和 MaterialProgram 快照均逐字节一致。篡改 `weights/model.bin` 的期望哈希后，锁定方法 ID 的 replay 被明确拒绝并返回非零状态。

## 最终回归门槛

- CPU unit tests；
- Slang/GPU ABI、backend 和 P1 compiler parity；
- 随机游走参考的解析、互易性、深层栈、统计和生成测试；
- MethodBundle 内容哈希与 loader；
- viewer Release 构建、真实 bundle headless capture/replay、固定路径 benchmark；
- Python compileall、`git diff --check`、无旧 import/路径引用；
- `external/Falcor` 和 `external/pbrt-v4` 均保持锁定提交与干净工作树。

上述项目均已通过。本文件只作为已完成迁移的决策记录，不再承担待办计划。
