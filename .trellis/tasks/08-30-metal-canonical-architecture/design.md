# 设计

## 合同边界

source层拥有`SourceSnapshot`与native edit；reference definition把snapshot set编成`ReferenceExecutionPlan`；backend session执行plan但不识别family。learning层通过`NativeAssetCollection`与三类typed batches消费source资产和online response。method只通过`MethodDefinition`声明phase、components与compilers。deployment以`ScatteringPackage@2`连接viewer。

## Reference 与asset

`ReferenceExecutionPlan`包含ordered execution groups、global source-index map、material records、argument/RO offsets、resources、query recipe和semantic/build identity。backend可以为每group持有session/runtime；upper producer只请求group-homogeneous work item。

`NativeAssetCollection`以asset/domain/mip/role/schema索引tile+halo source tensors，提供training sample、cook traversal和GPU working-set leases。现有constant/MaterialX/LayerStack/MDL adapters统一实现collection；单texel source只是1×1 asset，不保留pyramid adapter。

## Training

`TrainingConfig@4`保存phase list；每phase引用route specs、loss IDs、parameter groups、optimizer/schedule、precision和step budget。batch union为`AssetTileBatch@1 | EvaluatorBatch@3 | MethodSamplerBatch@3`。runner按phase请求所需routes，parameter registry严格匹配descriptor；prefetcher通过backend slots/CUDA events持有lease。

`TrainingCheckpoint@4`严格保存config/descriptor/plan/asset/query identities、phase cursor、model、phase-local optimizer/scheduler/precision、RNG、coverage和validation。没有v3 loader。

## Method conformance

`MethodDescriptor@2`的component contracts驱动三项通用检查：execution coverage、gradient/update coverage、artifact coverage。fixture method同时提供正例和缺component/orphan parameter/missing artifact负例；NVIDIA descriptor完整声明自己的components并通过同一检查。

## Package 与viewer

`ScatteringPackage@2`manifest有`program/asset/instance`三个严格section及独立identity。Python/C++共享schema/typed resource vocabulary。viewer用`ProgramRuntimeCache`、`AssetBinding`、`InstanceBinding`原子组装slot；不存在v1 reader或旧material section探测。

## 迁移策略

迁移顺序遵守“先定义新合同并迁完一层调用方，最后删除旧层”，但任何阶段结束时不允许两套产品入口同时可用。开发中可在一个提交内临时并存未导出的内部symbol；child完成前必须删除。静态test维护旧symbol/format/field denylist。
