# Tungsten 单材质结构选择

## 它是什么

本轮在同一个原生 vMaterials 2 `Tungsten_Brushed_Medium_Light_Brushing` locator 上，对预算完全相同的 `metal_budgeted_hybrid_v3` 与 `metal_budgeted_direct_control_v3` 做五卡 matched pilot。训练使用GPU online reference、per-rank batch 512、global batch 2560；两侧都完成1792 step联合拟合与256 step部署量化感知微调，不保存训练batch。

最终有效artifact根为 `artifacts/metal-budgeted-pilot/ddp5-wrap-1d5f813/`，对应修复部署侧wrap bilinear oracle后的fresh pair。hybrid/direct checkpoint SHA-256分别为`8a15a5945085bddc781c1e60cd434ffa78b3a791ceed05dbbe007f8e7fb8971e`和`4848b783407eba3a0127910dca370ea97f95f904416e974ad61867a3bbff2042`。这里的结果只用于单材质结构选择和失败分类，不宣称代表692材质formal质量。

## 最终结果

step2048固定validation的均值如下；越低越好。每个配置包含256条同序validation row。

| profile | appearance | log RGB | linear RGB | chroma | peak RGB | spatial gradient |
|---|---:|---:|---:|---:|---:|---:|
| hybrid v3 | 1.14500 | 0.32048 | 1.37356 | 0.00159 | 1.06849 | 0.28299 |
| direct v3 | 1.83045 | 0.68995 | 1.59353 | 0.00752 | 1.74078 | 0.28120 |

逐row paired bootstrap的`direct-hybrid`差异为：

| metric | 均值差 | 95% CI | 解释 |
|---|---:|---:|---|
| appearance | +0.68544 | [0.67929, 0.69144] | hybrid明确更好 |
| log RGB | +0.36947 | [0.36770, 0.37123] | hybrid明确更好 |
| linear RGB | +0.21998 | [0.21327, 0.22659] | hybrid明确更好 |
| chroma | +0.005938 | [0.005920, 0.005955] | hybrid明确更好 |
| peak RGB | +0.67230 | [0.65965, 0.68481] | hybrid明确更好 |
| spatial gradient | -0.001790 | [-0.001967, -0.001617] | direct小幅更好，但两者都未恢复目标高频 |

hybrid与direct均完成QAT且没有数值崩溃。两侧形状相同，方向求值均为11,392 dense MAC、PreparedState为160 B、asset读取固定两次，因此上述质量差不是靠扩大hybrid预算得到的。

## 当前结论

按预登记规则选择hybrid作为canonical：它在peak、chroma和总体响应上都有稳定净收益，且没有越过20k MAC/192 B hard bound。direct继续保留为exact diagnostic对照，让Windows viewer能够直接观察“小MLP独自追移动高光”与“analytic core加learned gate”之间的差异。

这个选择不等于当前hybrid内部机制全部成功。hybrid的positive RGB trace到step2048仍约`2.4e-6`，主要输出来自gate调制analytic lobes；analytic trace又有明显长尾。因此当前证据支持“预算内analytic core对Tungsten主响应很重要”，不支持“神经residual已经学会有用修正”。下一轮应单独检验稳定core参数化、gate尺度与非退化positive residual，而不是继续给当前形态追加step。

共同的主要失败仍是空间细节。在线paired诊断中target one-texel log梯度约0.285，v2模型预测只有约0.001–0.004；v3 Detail短路径虽改善hybrid平均与peak，却未降低最终spatial error。后续v4角色分离、v5中心texel与v6双局部signed导数都没有形成跨材质spatial收益，fixed-batch overfit也远未接近target。因此下一结构方向应改变source语义到runtime latent的映射与asset cook监督，而不是增大batch、提高spatial loss权重或无条件延长训练。

## 下一步

两个exact Windows diagnostic package/catalog已经生成在`artifacts/viewer/metal-budgeted-ddp5-wrap-1d5f813-step2048/`。Linux侧真实Falcor/Vulkan边界witness中，hybrid/direct的Slang parity最大绝对误差分别为`1.87e-5`和`2.23e-4`，均通过冻结容差；Windows D3D12仍需在目标机按artifact README执行视觉检查，不能由Linux结果代替。可部署选择保持hybrid，direct仅作视觉/结构对照。
