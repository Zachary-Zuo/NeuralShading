# 历史 viewer 图像观察

## 阅读范围

2026-09-05 直接查看以下五张原图，只读相邻 README/capture JSON；未启动 viewer，未重新渲染、裁剪或计算图像指标。它们都属于历史证据，不是新架构的运行依赖。

同时复用现已归档架构任务的[视觉盘点](../../09-05-architecture-reset-training-workflow/research/architecture-audit.md:49)。该盘点区分 viewer 输出与素材筛选缩略图，避免把所有 PNG 都当模型结果。

## 1. budgeted step2048 早期交互图

[查看原图](../../../../../../artifacts/viewer/metal-budgeted-ddp5-wrap-1d5f813-step2048/windows-review/hybrid-interactive.png)

- 身份：同目录 README 明确左侧为 Tungsten MDL reference PT，右侧为 `metal_budgeted_hybrid_v3` deferred，单方向光、零次次级反弹。
- 观察：左侧中心高光周围有密集细刷痕；右侧高光更平滑，轮廓和局部区域仍有明显方块。
- 限制：README 明确截图时右侧仍在逐块细化。不能把这些方块量化成 latent 的分辨率，也不能从此图确定全部纹理细节已经被模型抹掉；两侧 renderer 模式也不同。

## 2. 后续对称 deferred capture

[查看原图](../../../../../../artifacts/viewer/metal-viewer-refresh-lighting/current/smoke/deferred/capture-display.png) · [元数据](../../../../../../artifacts/viewer/metal-viewer-refresh-lighting/current/smoke/deferred/capture.json)

- 身份：Tungsten source；两侧均 `deferred`、`ready`；右侧 profile 为 `metal_budgeted_hybrid_v3`，package 为 `2054229aa9af1e162f6ec8c41a43e57e29ce230a0be18127b9640155f12eb6b2`。
- 分辨率：总图 640×360，单侧 320×360；共享 exposure `-0.5 EV`。元数据记录 source 为 `explicit-lod0`，不消费 UV derivatives。
- 观察：左侧亮部有较密的细碎变化；右侧内凹区域的亮带明显更平滑。两侧都能看见部分粗尺度竖向/弧形结构，因此“所有空间信息都为零”不准确。
- 限制：仍是低分辨率、tone-mapped 显示图；source/native lookup 与 neural mip 的过滤条件需要另行核实。差异同时可能包含 texture lookup、response 表示及方向高光误差，不能仅凭图像归因于 encoder。

## 3. 后续对称 PT capture

[查看原图](../../../../../../artifacts/viewer/metal-viewer-refresh-lighting/current/smoke/pt/capture-display.png) · [元数据](../../../../../../artifacts/viewer/metal-viewer-refresh-lighting/current/smoke/pt/capture.json)

- 身份：同一 Tungsten/profile/package；两侧均 PT，均 8 spp，单侧 320×360，scene bounce cap 4。
- 观察：两侧整体金属亮度结构接近，右侧中心亮带更连续；两侧都存在强烈颗粒。
- 限制：8 spp 的随机噪声足以掩盖微细节；metadata 的 `comparison_purpose: formal` 是旧 schema 字段，不自动赋予这张 smoke 图正式统计效力。GPU timing 只记录少量 dispatch 样本，不能直接转换成单次 evaluator 成本或实时帧率。

## 4. 同为 deferred 的 UI 截图

[查看原图](../../../../../../artifacts/viewer/metal-viewer-refresh-lighting/current/ui/both-deferred.png)

- 顶部标签显示两侧均 Deferred/Ready，右侧 `metal_budgeted_hybrid_v3`。
- 观察：右侧能看见长条纹，但表面亮部较平整；左侧相应区域受面板遮挡，且两物体显示位置不适合直接像素配对。
- 用途：确认空间变化并非完全不存在，辅助辨认材质和模式；不用于误差测量。

## 5. 更早的 full 20k 图

[查看原图](../../../../../../artifacts/viewer/metal-step00020000-tungsten/viewer-window.png)

- 依据[旧审计](../../../../09-04-metal-neural-budgeted-redesign/research/initial-evidence.md:16)，属于更早的 full profile。
- 观察：reference 侧刷痕明显；neural 高光带有绿色/黄色偏色，右侧仍有粗块。
- 限制：不同模型、训练和显示阶段。不能把这一偏色当作 budgeted v3 当前结论，也不能用它推断“增加 encoder 容量无效”。

## 下一次视觉证据需要回答什么

这些历史图支持“亮部细节明显不一致、粗尺度变化仍有部分保留”的观察。架构已完成；实施后的阶段收尾用当前 export/eval 入口，在同一 source/state、camera、照明、UV/footprint 合同与完整分辨率生成局部对照。优先固定方向和无路径噪声的 response crop，再补同模式 renderer 图；PT 只在已估计噪声水平后评价空间误差。本次更新没有重新运行 viewer。

当前没有在上述专项 probe 目录找到可明确归属 v4–v6 青铜/开裂钢的 PNG 对照；其结论仍来自旧数值报告。本轮没有用 Tungsten 图替代两种材质的视觉证据，也未转换 EXR 来制造新的实验结果。
