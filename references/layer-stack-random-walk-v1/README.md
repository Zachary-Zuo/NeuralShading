# ncls.layer-stack-random-walk@1

## 身份

- 源材质族：`ncls.layer-stack@1`；
- 角色：当前多层界面与均匀 slab 材质族的 GT reference；
- 原生表达：`MaterialProgram` 的 `ncls.composition.layer_stack@1` 子图及规范化 `LayerStackIR@1`；
- 求值：局部表面反射随机游走；
- 当前状态：active。

## 实现文件

项目维护的正式实现位于：

- `src/ncls/references/programs/layer_stack.py`；
- `src/ncls/references/backend.py`与`src/ncls/references/query.py`；
- `shaders/ncls/contracts/layer_stack_ir.slang`；
- `shaders/ncls/reference/sampling.slang`；
- `shaders/ncls/reference/interfaces.slang`；
- `shaders/ncls/reference/random_walk_reference.slang`；
- `shaders/ncls/reference_backends/layer_stack.slang`；
- `shaders/ncls/reference_query/reference_query.cs.slang`。

`LayerStackReferenceProgram`对program source和shader计算`implementation_sha256`；公共`ReferenceBackendDescriptor`另记录Falcor/Slang的semantic/build identity。路径变化本身不是语义正确性的证据；修改实现后必须产生新hash，并由checkpoint/报告记录实际使用的reference identity。

它与MERL、OpenPBR、MaterialX和MDL共用`ReferenceBackendCapability.open()`及同一种`ReferenceBackendSession`。Windows由backend选择D3D12，Linux选择Vulkan，上层不包含平台分支。`evaluate()`返回线性RGB `f`，不含cosine；需要`f·|cosθi|`的消费点显式计算。

## 验证边界

- 单界面原子使用解析性质、能量、互易性和采样/PDF 一致性验证；
- pbrt coated diffuse/conductor 只验证对应两界面 source slice，不声称 pbrt 是任意 N 层 GT；
- `N > 2` 验证组合算法、退化关系、互易性、有限值和独立实现一致性，不按层数枚举；
- pbrt coated diffuse 与 coated conductor 已有历史 smoke，覆盖 clear、吸收介质和散射介质，以及粗糙各向异性导体基底；当前 registry 仍把 `numerical_parity` 标为 `pending`，需要先修正并重跑锁定的 pbrt harness，不能仅凭历史数值改成 `ready`。不再以增加无物理目的的层数扩展 probe。
