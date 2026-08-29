# 设计

## 模块

模型按parent设计实现为五个可分账模块，但正式identity联合启用：

1. `MetalTextureCodec`：role stems、bundle set aggregator、shared U-Net encoder、flattened per-asset/mip/domain INT8-QAT grids、rank-8 adapter、shared structured/semantic decoder；
2. `MetalTypedCompiler`：32×64 typed token set encoder和4-block attention，输出fixed `MaterialProgramState`；
3. `MetalPreparedModel`：deterministic access、two-mip decode、frame/normal、lobe/spatial/view state；
4. `MetalDirectionalEvaluator`：raw/half-diff/learned frames/angular bank；
5. `MetalHybridHead`：analytic core、positive residual lobes、bounded log correction和softplus tail。

所有multi-asset tensors使用flat storage + versioned offsets/shapes；v4 checkpoint不保存Python对象图。semantic heads训练期保留但runtime export删除。

## Lifecycle

使用canonical phase graph的`codec-warmup`和`joint-appearance`，不在method内部根据`global_step`隐藏子lifecycle。每phase的routes、active groups、loss、precision与schedule由v4 config显式声明。proposal与QAT contracts在descriptor中预登记，分别由sampler/runtime及handoff完成并验证。

## Conditioning

Metal source adapter提供recipe/graph/schema/metal/finish/asset indices、typed token tensors/presence、UV domains/footprint和semantic source samples。`prepare()`输入compiler state+bundle grids+surface/wo；`evaluate()`只输入prepared state+wi。

## Controls

- encoder-only、bounded refinement和direct optimized使用同decoder/profile但不同run identity；
- target-visible optimized material state为teacher/control，不能进入pure compiler产品路径；
- core-only、direct-only、raw-only等只登记为后续ablation config，本child不以它们筛组件。

## Conformance

component manifest逐项登记五个模块内部的role stems、encoder trunk、high/low grids、dual heads、adapter、typed compiler、四路direction、analytic/multiplicative/residual/tail。由registry生成stratified activation set，累计execution trace、parameter-group gradient/update和export-state coverage；full identity出现未激活或无梯度required group立即失败。

## Export boundary

child完成时冻结layout JSON、tensor names、runtime profile、packing spec、proposal reservation和Python `prepare/evaluate` oracle；sampler child补齐proposal/sample/pdf，runtime child再生成Package@2/Slang。中间checkpoint明确标为evaluator-slice，不可导出或宣称full Metal package。
