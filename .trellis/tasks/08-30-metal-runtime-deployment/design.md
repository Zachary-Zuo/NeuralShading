# 设计

## Package映射

program section打包shared decoder/compiler/angular/evaluator/proposal weights和Metal Slang module；asset section保存bounded adapter、grid descriptors、量化high/low mip resources、samplers和role/schema metadata；instance section保存recipe/optical/raw typed buffer、compiled state和layout version。

不同packages若profile/checkpoint相同则`program_runtime_id`相同。viewer建立`ProgramRuntimeCache`持有passes/shared buffers；slot分别持有`AssetBinding`和`InstanceBinding`。选择新bundle先完整验证asset、迁移兼容typed values并创建candidate instance，成功后原子交换三层组合。

## Typed resource

Python writer/loader与C++使用同一注册dtype表，至少覆盖full profile的INT8 grid DDS/texture或typed buffer、FP16 fallback和sampler。descriptor验证format/shape/mips/stride/alignment/bytes；未知dtype拒绝。shared data优先pack进统一`gNclsRuntimeWeights`，避免新增硬编码usage。

## Editable material ABI

instance section提供host-only editor metadata和raw typed parameter buffer；full capability强制声明标准material-compiler compute entry。viewer按native path/type/range/enum生成UI，写typed buffer并dispatch compiler到SRV|UAV instance buffer。asset swap根据compatibility决定保留同名同类型值或恢复新schema default，再dispatch一次。

## Slang

backend私有`PreparedState`保存two-mip structured state、frames、lobes、view token和proposal mixture。`evaluate()`只做direction charts/angular features、analytic bank和hybrid residual；`sample/pdf`复用同一mixture。所有循环上限来自生成layout；invalid返回valid=0，不clamp NaN。

Package@1 schema/reader不在本child出现；canonical architecture child已经迁移NVIDIA和viewer。本child只增加Metal typed usages与module implementation。
