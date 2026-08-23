# 根本迁移记录

## 当前状态

- 阶段 0–4 的正式目录和基础闭环已经落地：合同、Python package、随机游走参考数据、训练/评测/导出和 Windows/Falcor viewer 均有正式入口。
- 旧源码删除、文档入口和仓库边界已经收敛；一次性正确性报告不再持久化，具体状态以当前测试与 `artifacts/` 中的运行输出为准。
- 当前已有 PyTorch/Slang compiler parity 和不依赖 Python 运行时的 realtime `MethodBundle`，证明了逐像素编译网络、backend 与 viewer 的部署闭环；它没有证明目标 neural `evaluate(wo, wi)` 已经建模完成。

迁移不维持旧 import、旧 CLI 或通用 backend packet ABI。可信算法和结论已经迁入新位置；错误、重复和过时内容由 Git 历史追溯。

## 已冻结的边界

三个业务块只通过带版本的合同交换产物：

```text
MaterialProgram → 数据采集 → ReferenceDataset
ReferenceDataset → Python direct fit / train / evaluate → MethodBundle
MaterialProgram + MethodBundle → Windows viewer
```

- `MaterialProgram` 是公共作者格式，`LayerStackIR` 是当前规范化内部 IR。
- `prepare/evaluate` 是所有实时表面方法的基础散射合同。目标 neural backend 在 `prepare` 中获取/编码共享 state，在 `evaluate` 中运行主要 MLP；`sample/pdf` 与专用积分器按 capability 提供。
- `CompiledMaterial` 和 `ScatteringState` 完全由 backend 私有。
- 加入小型 neural evaluator、material/spatial latent、matched sampler 或 integration head，不改变 viewer 主可见性或顶层材质接口；若训练需要 UV/footprint/LOD，必须版本化扩展数据合同。

## 迁移后的研究执行顺序

基础工程迁移完成后，当前方法研究按以下依赖推进：

1. 定义少量 latent、方向编码、`prepare` shared trunk 和 evaluator MLP 候选，并审计监督覆盖；
2. 验证单材质完整 `wo × wi` 容量；
3. 验证共享 decoder + 材质 latent；
4. 训练未见材质状态的 feed-forward compiler；
5. 导出 Slang 最小 neural evaluator 并测量单次查询；
6. 在 evaluator 固定后增加 matched sampler；
7. 研究多灯、环境/面光积分和 Falcor/UE 式工作流。

多灯 scaling、PT variance 和 UE 集成属于后半段系统验收，不能在表示与执行图未确定时作为当前 kill test。

## 目录迁移结果

| 迁移前 | 最终处理 |
|---|---|
| 固定 `schema/` | MaterialProgram schema、LayerStackIR ABI 和一次性 v0 reader 进入 `src/ncls/core/material/` |
| 随机游走旧源码与采集脚本 | 进入 `shaders/ncls/reference/`、`shaders/ncls/data/` 和 `src/ncls/data/` |
| 通用 legacy packet | 删除；具体布局进入对应 representation 与 shader backend |
| 表示拟合脚本 | 可信部分进入 `learning/direct_fit/`，重复/过时实验入口删除 |
| 泛化 `model/` | 重写为 `learning/models`、`training`、`evaluation` 和 `export` |
| Python lookup viewer | 删除；真正应用进入 `apps/viewer/` |
| 外部 pbrt probe | 进入 `tools/reference/pbrt_probe/` |
| 根级旧测试 | 重组为 `tests/unit`、`tests/gpu`、`tests/integration/reference` |
| 根级阶段计划和 v0 操作文档 | 删除；有效结论进入稳定文档，逐次运行指标只保存在被忽略的 `artifacts/` 中 |

旧 `ncls-direction-tiles@1` 数据仍能通过 `ncls data convert-legacy-v0` 一次性转换，但项目不再保留第二套 writer 或旧包级 import shim。

## viewer 当前实现

第一版 viewer 当前具备：

- Falcor `Scene` 多格式导入、单一 orbit/pan/dolly 相机、右侧 raster 主可见性与左侧独立 path-traced primary ray；
- instance/material ID 拾取，以及每个 Falcor material slot 独立的源材质绑定；
- LayerStack、MERL、OpenPBR 和 MaterialX 的 reference 与族专属编辑 UI；
- 方法为空时全宽 reference；显式选择方法后，右侧运行无跨帧累计、无随帧噪声的 deferred prepare/lighting；
- 固定 `studio-v1` MaterialX shaderball/Poly Haven HDRI/默认材质，以及方向光、点光、矩形面光；
- 方法切换、共同曝光和差异显示；
- 全文件哈希、平台/合同检查和 GPU parity 后才接纳方法；
- raw reference、显示专用去噪结果、EXR/PNG/材质/指标/capture v3 与命令行 replay；
- 固定相机路径 JSON/CSV benchmark 入口。

当前左侧已覆盖场景相交、阴影、直接光、环境 MIS 和跨物体间接反弹，是受 scene/layer 深度上限约束的完整场景 path tracer；raw Monte Carlo 均值保持权威，默认 a-trous 去噪仅用于显示。右侧仍未加入 path-traced 全局传输，因此图像差异是完整 reference 与实时系统的视觉差异，不能替代方向域 evaluator 指标。逐 material slot 的 authoring/capture 状态使用 `ncls.viewer-scene@1` 保存和验证。任何更强的正确性或逐字节一致性结论都必须由当前版本重新运行测试产生，不能引用已清理的历史报告替代。

## 最终回归门槛

- CPU unit tests；
- Slang/GPU ABI、backend 和 P1 compiler parity；
- 随机游走参考的解析、互易性、深层栈、统计和生成测试；
- MethodBundle 内容哈希与 loader；
- viewer Release 构建、真实 bundle headless capture/replay、固定路径 benchmark；
- Python compileall、`git diff --check`、无旧 import/路径引用；
- `external/Falcor` 和 `external/pbrt-v4` 均保持锁定提交与干净工作树。

这些项目是每次稳定发布前应重新执行的门槛。本文件记录迁移决策，不保存某一次运行的“已通过”报告。
