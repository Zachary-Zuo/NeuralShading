# Metal-v1 texture encoder/decoder 共享与新资产路径

## 1. 数据事实

692 个 opaque exports 实际引用 52 个唯一 texture-set identities，而不是 692 套独立纹理。复用高度不均匀：

- 52 套中有 42 套只被一个 module 使用；
- 6 套分别被 11 个标准 metal modules 共享；
- 1 套被 12 个 modules 共享；
- 其余 3 套各被 2 个 modules 共享。

标准 13×7 矩阵中，每一种 finish 都只有 3 套独立 texture-set identities：Brass 一套、Bronze 一套、其余 11 种 metal 通常共享一套。对应统计为：

| finish | module | 独立 texture set | 唯一 source texture paths |
|---|---:|---:|---:|
| Base | 13 | 3 | 5 |
| Brushed | 13 | 3 | 5 |
| Foil | 13 | 3 | 5 |
| Hammered | 13 | 3 | 6 |
| Knurling | 13 | 3 | 7 |
| Scratched | 13 | 3 | 8 |
| Sheet | 13 | 3 | 5 |

因此 module/export 数量不能当作 texture encoder 的独立样本数。若给每个 finish 配一套完全隔离的 encoder，标准区每个 encoder 只有 3 个真实 texture sets；特殊 recipe 往往只有 1 套，更不足以支持可靠的未见资产泛化结论。

## 2. “新 decoder 怎么来”

如果要求同语义新资产可以跳过训练，那么新资产不能要求重新训练一套完整 decoder。推荐合同是：

```text
(Z_asset, A_asset) = E_shared(texture set, channel roles, schema)
decoded state      = D_shared(local Z_asset, role/schema token; A_asset)
```

- `E_shared` 是训练/资产编译期 encoder；
- `D_shared` 是随 method 发布的 runtime decoder，不随新资产重新产生；
- `Z_asset` 是新资产的 hierarchical local grids；
- `A_asset` 是 encoder 生成的、固定 shape 的 asset-local FiLM/low-rank modulation，存入 bundle；
- 新资产使用既有 `D_shared` 解释 `Z_asset/A_asset`，因此不存在“新的完整 decoder 从哪里来”的问题。

只有当新资产引入未注册的 channel role、bundle schema 或 graph recipe 时，才需要新增 schema adapter/方法版本；这不属于“同类型新 texture”的 zero-shot 路径。

## 3. 推荐的共享层级

用户已确认不采用“一种 finish 一整套互不共享的 encoder/decoder”，而采用分层语义共享：

1. **role-specific stems**：color、tangent-space normal、scalar roughness/mask/AO、packed correlated channels 分别使用适合其值域和损失的输入 stem；
2. **shared multiscale spatial trunk**：所有 texture sets 共享主要 encoder 容量，从 52 套资产共同学习局部、多尺度和 mip 结构；
3. **role/schema tokens + bundle set aggregator**：保留每个 channel 的语义、packed mapping、finish/recipe compatibility，并联合观察同一 bundle 的多张纹理，学习跨通道相关性；
4. **grid/adapter heads**：输出 hierarchical grids 与小型 asset-local modulation；
5. **shared decoder trunk + role/schema heads**：runtime structured feature trunk 共享，training-only semantic reconstruction head 按 role/schema 分流。

这种形态满足“同类型/语义使用公共 encoder”的目标，同时避免把只有 1–3 个资产的语义类完全隔离。finish/schema token 让模型知道当前语义，不要求所有 texture channel 使用相同统计或相同输出 head。

## 4. Encoder 与 decoder 如何联合训练

所有 source-train texture sets 同时训练同一 `E/D` 系统：

```text
texture/mip set
  → E_role + shared encoder + bundle aggregator
  → quantized local grids + generated adapter
  → shared decoder + schema head
  → semantic texture reconstruction + online appearance response
```

训练必须让 encoder-only 路径真实承担重建，不能让每资产自由 latent/adapter 把 encoder 架空。质量优先生命周期同时保留三种结果角色：

1. `encoder-only`：`E(T) → Z,A`，不做新资产优化；它是未见同语义 asset zero-shot claim 的唯一依据；
2. `encoder + bounded refinement`：以 `E(T)` 初始化，只优化新资产的 `Z/A`，冻结 shared decoder；它回答能否用短时资产 cook 接近最佳质量；
3. `direct optimized control`：同一 decoder 下从自由 grids/adapter 做完整匹配预算优化；它是 best observed codec control，不是产品 zero-shot 路径。

联合训练可使用 direct optimized state 作 stop-gradient teacher/canonicalization target，但 functional/semantic reconstruction loss 必须直接作用于 encoder-only 输出，防止 latent 坐标不定或 encoder 被绕过。

## 5. 如何测试未见 texture

split 单位必须是 texture-set identity，不是 export/module；共享同一 texture set 的 11 个 metal modules 不能跨 train/test 泄漏。只有 held-out asset 与训练资产具有已注册 channel roles 和 compatible schema 时，才称为同语义新资产。

三条路径使用同一 frozen decoder、量化 profile、mip targets 与评测协议，分别报告：

- semantic channel/normal/mask reconstruction；
- end-to-end filtered `f`、energy、peak 与连续 footprint；
- bundle bytes、encoder/cook wall time；
- encoder-only 到 bounded refinement/direct control 的 gap。

由于当前每个 exact finish 的独立资产很少，Metal-v1 可以把 encoder-only holdout 作为研究性泛化测试，但不能在有限 52 套 source-derived texture sets 上宣称已经解决任意外部 texture 的导入。更强结论需要后续扩展同 schema 的独立 texture corpus，并冻结新的 source/authoring contract。
