# ncls.layer-stack-random-walk@1

## 身份

- 源材质族：`ncls.layer-stack@1`；
- 角色：当前多层界面与均匀 slab 材质族的 GT reference；
- 原生表达：`MaterialProgram` 的 `ncls.composition.layer_stack@1` 子图及规范化 `LayerStackIR@1`；
- 求值：局部表面反射随机游走；
- 当前状态：active。

## 实现文件

项目维护的正式实现暂时保存在：

- `shaders/ncls/reference/sampling.slang`；
- `shaders/ncls/reference/interfaces.slang`；
- `shaders/ncls/reference/random_walk_reference.slang`；
- `shaders/ncls/data/reference_tile.cs.slang`；
- `src/ncls/data/reference.py`；
- `src/ncls/data/generator.py`。

这些路径共同参与 `reference_source_sha256`。在 package manifest 和迁移回归测试实现前不移动，以免为了路径调整改变已验证的 reference 身份。

## 验证边界

- 单界面原子使用解析性质、能量、互易性和采样/PDF 一致性验证；
- pbrt coated diffuse/conductor 只验证对应两界面 source slice，不声称 pbrt 是任意 N 层 GT；
- `N > 2` 验证组合算法、退化关系、互易性、有限值和独立实现一致性，不按层数枚举；
- pbrt coated diffuse 与 coated conductor 交叉验证均已完成；验证覆盖 clear、吸收介质和散射介质，以及粗糙各向异性导体基底，不再以增加无物理目的的层数扩展 probe。
