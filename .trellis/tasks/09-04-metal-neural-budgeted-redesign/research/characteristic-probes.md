# Metal budgeted 特征材质专项 probe

## 它是什么

本协议落实用户授权的“主交付完成后，先对有特点的材质专项研究，再决定是否扩到普适性检查”。它只使用已经入选的`metal_budgeted_hybrid_v3`，不新增模型、loss、seed或部署预算，也不从Tungsten checkpoint恢复。

## 为什么选择这四项

四项均来自锁定的`metal-opaque-v1.json`，各取一个默认原生参数state；选择依据是机制互补，不是根据训练结果筛选：

| probe | 原生材质 | 要隔离的机制 |
|---|---|---|
| base | `Brass_Sheet_Yellow_Splotchy_Streaks` | 无涂层的有色金属基线，同时含sheet纹理和污渍 |
| high-frequency | `Bronze_Scratched_Yellow_Heavy_Worn_Scratches` | 深划痕、磨损及强空间高频 |
| chromatic-distribution | `Aluminum_Anodized_Pink` | 有色阳极氧化层、7个texture slot及Beckmann例外 |
| composite | `Steel_Painted_Light_Blue_Cracked_Matte` | 漆层、裂纹、粗糙金属基底的复合响应 |

四项分别使用versioned data fragment和run config；source locator、registry、seed、在线方向/footprint/paired-UV recipe与Tungsten主实验相同口径。

## 冻结执行

- GPU：物理GPU 5–9，DDP world size 5；每rank evaluator/sampler batch 512，全局batch 2560。
- 每项fresh初始化，单seed `2026090401`，在step 256停止；执行step 128和256两次冻结validation，不延长、不换seed。
- 每项累计`256×2560×2=1,310,720`个evaluator/sampler route work units。该数值是执行账本，不是质量门。
- 每项记录appearance、RGB、chroma、peak、spatial gradient、semantic-runtime、analytic/gate/positive分工、proposal和所有parameter group梯度/更新覆盖。
- 只有四项都通过实现正确性与有限性检查，才汇总共同结构趋势；256-step observed quality不称为泛化结论，也不与2048-step Tungsten绝对值直接排序。

## 预先解释规则

- high-frequency与composite的spatial仍接近target自身梯度尺度、而其他appearance下降：支持“Detail高频通路是跨材质瓶颈”，下一轮优先显式role-separated Detail/Context，而不是增加主干宽度或仅加spatial loss权重。
- anodized的chroma/peak显著落后base，同时analytic占主导：支持增加closure/distribution-aware analytic basis或让neural residual承担有色涂层校正；不据此把MDL改写为层GT。
- 四项行为差异明显：保留机制特定结论，下一轮做小型混合cohort，不宣称一个局部修复普适。
- 四项出现一致的finite/梯度/身份失败：先分类实现或protocol缺陷，不解释模型质量。

artifact固定写入`artifacts/metal-budgeted-probes/characteristic-v1/`；stdout、metrics、checkpoint、review与dmon不进入根仓库。本文件只追加实际结果和跨材质解释，不反写选择规则。
