# 设计

## Package映射

program section打包shared decoder/compiler/angular/evaluator/proposal weights和Metal Slang module；asset section保存bounded adapter、grid descriptors、量化high/low mip resources、samplers和role/schema metadata；instance section保存recipe/optical/raw typed buffer、compiled state和layout version。

不同packages若program identity相同则`program_id`相同。viewer以`programId`建立`ProgramGpuRuntime` cache，持有passes、shared buffers和material compiler；slot分别持有asset/instance resources。选择新bundle先完整验证asset、按同path同type迁移typed values并创建candidate instance，成功后原子交换完整slot。

## Typed resource

Python writer/loader与C++使用同一注册dtype表，full profile的INT8 grid、FP16 scale/adapter、typed raw/compiled buffer均使用structured buffer。descriptor验证dtype/shape/stride/alignment/bytes；未知dtype拒绝。shared data pack进统一`gNclsRuntimeWeights`，asset grid保持独立typed usages以支持asset替换。

## Editable material ABI

instance section提供host-only editor metadata、raw typed parameter buffer与compiled material buffer；full capability强制声明material-compiler compute entry。viewer按native path/type/range/enum递归生成通用UI，在candidate buffers写raw state并dispatch compiler。只有compiler成功后才替换slot buffers/editor state；失败时active binding保持不变。

## Slang

backend私有`PreparedState`保存two-mip structured state、frames、lobes、view token和proposal mixture。`evaluate()`只做direction charts/angular features、analytic bank和hybrid residual；`sample/pdf`复用同一mixture。所有循环上限来自生成layout；大矩阵循环使用保留静态`MaxIters`的`[loop]`，避免shader编译器完全展开，不能据此减少模型计算。invalid返回valid=0，不用clamp掩盖NaN。

Windows D3D12对quality-first full program采用viewer通用8×8 tiled dispatch并在workgroup之间同步提交，作为TDR scheduling boundary。offset属于package render shader公共常量，deferred与PT共用；它不改变per-pixel模型、采样器或输出语义，也不按source family选择路径。

Package@1 schema/reader不在本child出现；canonical architecture child已经迁移NVIDIA和viewer。本child只增加Metal typed usages与module implementation。
