# Tungsten 单材质结构选择

## 它是什么

本轮在同一个原生 vMaterials 2 `Tungsten_Brushed_Medium_Light_Brushing` locator 上，对预算完全相同的 `metal_budgeted_hybrid_v3` 与 `metal_budgeted_direct_control_v3` 做五卡 matched pilot。训练使用GPU online reference、单卡 batch 512、全局 batch 2560；两侧都完成1792 step联合拟合与256 step部署量化感知微调，不保存训练batch。

artifact根为 `artifacts/metal-budgeted-pilot/ddp5-detail-frame-aac37e6/`。这里的结果只用于单材质结构选择和失败分类，不宣称代表692材质formal质量。

## 最终结果

step2048固定validation的均值如下；越低越好。每个配置包含256条同序validation row。

| profile | appearance | log RGB | linear RGB | chroma | peak RGB | spatial gradient |
|---|---:|---:|---:|---:|---:|---:|
| hybrid v3 | 1.14411 | 0.32031 | 1.37224 | 0.00159 | 1.06724 | 0.28294 |
| direct v3 | 1.83025 | 0.68838 | 1.59200 | 0.00752 | 1.74501 | 0.28120 |

逐row paired bootstrap的`direct-hybrid`差异为：

| metric | 均值差 | 95% CI | 解释 |
|---|---:|---:|---|
| appearance | +0.68615 | [0.68013, 0.69220] | hybrid明确更好 |
| log RGB | +0.36807 | [0.36633, 0.36985] | hybrid明确更好 |
| linear RGB | +0.21976 | [0.21307, 0.22634] | hybrid明确更好 |
| chroma | +0.005928 | [0.005911, 0.005946] | hybrid明确更好 |
| peak RGB | +0.67777 | [0.66522, 0.69035] | hybrid明确更好 |
| spatial gradient | -0.001745 | [-0.001930, -0.001563] | direct小幅更好，但两者都未恢复目标高频 |

hybrid与direct的runtime FP16 weight MAE分别约`5.04e-5`与`5.73e-5`，QAT没有造成整体崩溃。两侧形状相同，方向求值均为11,392 dense MAC、PreparedState为160 B、asset读取固定两次，因此上述质量差不是靠扩大hybrid预算得到的。

## 当前结论

按预登记规则选择hybrid作为canonical：它在peak、chroma和总体响应上都有稳定净收益，且没有越过20k MAC/192 B hard bound。direct继续保留为exact diagnostic对照，让Windows viewer能够直接观察“小MLP独自追移动高光”与“analytic core加learned gate”之间的差异。

这个选择不等于当前hybrid内部机制全部成功。hybrid的positive RGB trace到step2048仍约`2.4e-6`，主要输出来自gate调制analytic lobes；analytic trace又有明显长尾。因此当前证据支持“预算内analytic core对Tungsten主响应很重要”，不支持“神经residual已经学会有用修正”。下一轮应单独检验稳定core参数化、gate尺度与非退化positive residual，而不是继续给当前形态追加step。

共同的主要失败仍是空间细节。在线paired诊断中target one-texel log梯度约0.285，v2模型预测只有约0.001–0.004；v3 Detail短路径虽改善hybrid平均与peak，却未降低最终spatial error。这把下一结构方向收敛到role-separated Detail/Context、显式可量化局部高频feature以及对应matched消融，而不是增大batch、提高spatial loss权重或无条件延长训练。

## 下一步

先为hybrid与direct实现同形的FP16 weight pack、RGBA8 latent、ProgramState/PreparedState和evaluator-only Slang/package parity。deployment代码会改变method implementation identity，因此在该commit后两侧都从fresh identity复跑到2048，再生成两个exact Windows diagnostic package/catalog；当前`aac37e6`产物保留为结构选择证据，不伪装成新runtime的checkpoint。
