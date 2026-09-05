# 当前模型部署

## 1. 适用范围

训练状态的预览、eval/export 和当前 runtime 修改。与训练共用一个 checkpoint reader。

## 2. 签名

```text
load_checkpoint(path) -> TrainingCheckpoint
compile_evaluation_package(checkpoint, output, material_index=...) -> CompiledEvaluationPackage
Method.compile_program / compile_asset / compile_instance / package_validation
```

## 3. 合同

- 同一当前 checkpoint 直接供训练、validation、图像和导出消费；无 deployment/evaluation snapshot 转换或旧格式 importer。
- 模型结构与 source 必须实际匹配，不能靠 tensor copy 广播；代码提交与配置 hash 只记录来源。
- 初始化、短训和完成状态都能预览/导出。phase、step、梯度覆盖保留在诊断 metadata，不要求 formal/complete。
- package 记录当前 compiler 生成的 program/asset/instance 资源。shader module closure、资源大小/stride、sampler 和 ABI 对应真实 runtime；未知资源类型或坏绑定不能静默运行。
- 产出默认进入 run/exports。Windows 图像所需临时 package 留在同 run/eval。旧 artifacts 中的权重和视觉证据不移动、不修改。
- Python/Slang parity 检查真实部署算子与打包；不能仅以有文件或 ready slot 宣称数值正确。
- Metal spatial 的新 tensor/runtime ABI 为 `ncls.metal-spatial-method@1`，prepared layout 源为 `metal_budgeted_layout_v2.json`：80 half + 4 uint32，共 176 B；compiled instance 为 192 words（768 B）。prepare 137→32→32→24，proposal 80→16→13，共 7,664 dense MAC；evaluate 为 11,392 dense MAC。这些是静态算量，不能替代 GPU 时间/寄存器实测。
- 最多 9 个原生 UV 组，每组独立 Detail/Context SNORM8 plane 与 sampler；nonrepeat 每组最多三个 lookup，保守上限 54 reads/prepare。编译记录实际读取数与 latent bytes，evaluate/pdf 不重读纹理。空组有合法零纹理 binding，shader 跳过读取。上述成本只适用于当前 Metal backend，不是公共 source 或 state 合同。
- spatial cook 使用训练 encoder 的同一 hierarchy，按层暂存有预算的 host FP32 feature、GPU tile 推理，只持久化最终 latent。未见 snapshot 只要原生 schema/resource 兼容即可 cook，不要求训练 asset-ID，不创建 optimizer，不做隐藏 refinement。
- proposal 参数及其独立 frames 不依赖 wo；反向密度必须对照独立 `prepare(wi).pdf(wo)`，不能只验证冻结 prepared state 的 Python/Slang 互相一致。evaluator 用自身 view-conditioned frames；prepared FP16 后再次检查有限性，invalid 不得清成有效零。

## 4. 错误矩阵

| 条件 | 行为 |
|---|---|
| tensor 名称、shape、dtype 不符 | 模型加载边界失败 |
| source 或 package 资源缺失/绑定错误 | 对应边界报告 |
| 只有代码或日志设置改变 | 可读取当前模型，不比较完整训练 hash |
| step 0 或未完成 | 可导出，明确记录训练状态 |
| evaluator/sample/pdf 与独立 oracle 不符 | 报告 parity 失败，不改变权重/容差掩盖问题 |

## 5. 案例

正常：初始化模型生成 package，供结构和渲染诊断。基础：已训练模型用当前 compiler 导出新包。错误：增加专用 reader 或要求 complete 才能看图。

## 6. 验证

checkpoint 结构错误测试；当前方法的 packed Python/Slang parity、sample/pdf 一致性及边界 UV；Windows 真实新模型 eval/export。构建遵守 Falcor overlay 生命周期。

## 7. 错误与正确

```text
错误：运行设置变化 → full-plan hash mismatch → 不能预览。
正确：读取一份模型状态 → 当前 compiler → 新包与实际 renderer 检查。
```
