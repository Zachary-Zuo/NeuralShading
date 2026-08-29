# 设计

## Registry

新增 `ncls.mdl-metal-opaque-registry@1`：由锁定MDL SDK inspection生成，叶节点为exact export，去重表为graph/schema/texture-set/recipe。每次生成验证692/178/52/64和opaque capability，manifest保存生成器、SDK、bridge、source pack与资源闭包identity。

## Execution group

Metal registry为canonical `ReferenceExecutionPlan`提供group key：runtime/generated module、RO、texture binding set、resource layout和capability。backend session pool持有全局source index→group/local material映射、lazy group runtimes和确定性调度。训练batch选一个group，再在group内选base export/state，避免GPU program divergence；不暴露Metal leaf/routed两套public session。

MDL compiled-material record保存argument/RO byte offset。`nclsPrepareMdlTargetState()`从record初始化offset后只执行一次`init`；evaluate/sample/pdf复用prepared target state的optimized control另立entry/config，authoritative current path继续保留。

## Typed state recipe

`online_query.typed_state_recipe`定义参数责任过滤、continuous Sobol/stratified采样、bool/enum choices、边界/默认权重、pool capacity和独立seed。公共source editor应用patch；参数state artifact只保存source snapshot/argument runtime state，不保存response。query stream identity包含recipe、所有生成state IDs、group partition与cache policy。

## Filtered footprint

query kernel从`uvDx/uvDy`构造filter parallelogram，以固定低差异2D样本偏移surface UV，每个样本重新`prepare/evaluate`，再平均线性`f`。`footprint_samples`与`evaluation_samples`是独立配置/计数；zero footprint退化到单UV，invalid sample按冻结规则处理而不是写零。

## Asset collection

每个opaque texture-set生成role/schema/domain/mip descriptors。collection按tile+halo懒加载source pixels，在GPU working-set中提供codec target、UV/footprint采样与cook traversal；常量slot仍作为显式constant domain存在。asset identity包含source hashes、decode/transfer/normal/filter规则和tile policy。

canonical MDL record必须显式提供argument/RO offsets；缺offset/layout直接fail closed。不存在offset-zero默认、旧record兼容或single-session快路径。所有plan/state/asset identities进入v4 checkpoint/query resume检查。
