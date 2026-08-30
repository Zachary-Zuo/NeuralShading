# Metal runtime package 与交互 viewer

## 目标

把完整evaluator+sampler checkpoint编译为canonical `ScatteringPackage@2`与共享Slang backend，在不新增Metal renderer分支的前提下实现program/asset/instance三层binding、bundle replacement、typed edit、full path parity和viewer生命周期。

## 显式依赖

- 依赖`08-30-metal-canonical-architecture`已完成Package@2和viewer全仓迁移并删除v1 loader；
- 依赖`08-30-metal-fused-full-method`冻结的profile/layout、Python evaluator oracle和checkpoint；
- 依赖`08-30-metal-matched-sampler`的sample/PDF layout、Slang oracle和full capability；
- 依赖`08-30-metal-reference-foundation`的registry/schema/compatibility。

## 需求

- runtime module实现parent设计的`prepare/evaluate/sample/pdf`，structured decode只在per-hit prepare执行；
- shared program、finish asset和compiled instance分离记账/绑定，不同bundle共享`program_runtime_id`；
- package使用`program/asset/instance`三section与`B_shared/B_asset/B_instance`身份；
- typed resource loader支持full profile量化grid并fail closed；
- viewer按program runtime缓存shader/shared weights，asset切换只重绑asset并重编instance；
- package携带UI-safe native typed schema，通用material-compiler compute entry在编辑后一次性更新compiled state；
- bundle/schema/recipe不兼容、compile失败或resource失败时保留旧binding；
- Python FP32、BF16/QAT、Slang和viewer输出执行full matched parity。

## 不在范围

- 最终PT方差、full formal和Linux long quality结论；
- 组件消融/compact profile；
- 旧Package@1 reader/converter、Metal C++ enum或Metal专用渲染路径；
- 任意外部texture导入。

## 验收标准

- [x] [实现正确性｜Package@2 ABI] program/asset/instance hashes/layout/resources/capabilities验证完整，tamper/unknown/v1 format fail closed；
- [x] [数值正确性｜parent runtime] fixed probe和随机query上Python FP32→quantized→Slang→viewer差异落入由precision oracle预先推导的tolerance；
- [x] [需求交付｜parent R2/R3] bundle切换不重训、不重新编译shader；typed edit不重新编码bundle并只运行一次compiler；
- [x] [生命周期正确性｜viewer contract] 两个slot对称、三层atomic swap失败保留旧binding、deferred与PT都使用full capability；
- [x] [边界正确性｜project contract] viewer不依赖PyTorch、不读取source family/Metal identity选择renderer；
- [x] [实现正确性｜项目回归合同] 已迁移的NVIDIA Package@2、source viewer、Release build和Falcor clean全部成立，正式代码无v1 loader。

## 阻塞问题

无；typed editor与package映射已由parent设计冻结。
