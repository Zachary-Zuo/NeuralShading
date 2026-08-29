# 设计

## Evaluation manifest

新增通用versioned evaluation manifest，引用checkpoint/package、source registry、split/query recipe、controls、metrics、bootstrap unit/count和cost workload。CLI仍是generic `ncls learn evaluate`/evaluation runner，不按Metal分支。

## Split

texture-set identity是`G_asset`最小隔离单位；标准13×7提供metal/finish/pair split；parameter state按base family与state recipe隔离；special graph只在registered recipe域报告`G_recipe`。同一texture set复用的modules不得跨split。

## Metrics

每个query row保留source/graph/schema/asset/state/footprint/direction stratum。先输出raw/aggregated metric artifacts，再以source state为bootstrap unit计算median/mean/tail差异CI。semantic metrics与appearance metrics并列，不合成单分数。

## Cost

静态manifest读取真实weights/grids/records；GPU microbenchmark区分prepare/evaluate、coherent/divergent working set和interop；viewer固定scene/workload。reference current与optimized source、BC4/5/7 conventional control、full Python/Slang/quantized method分别标明capability/filtering。

## Interpretation

报告列出non-dominated observations但不预选成功threshold或aggregation。任何protocol/implementation defect使用新run identity重跑；正常empirical outcome不返回模型迭代。
