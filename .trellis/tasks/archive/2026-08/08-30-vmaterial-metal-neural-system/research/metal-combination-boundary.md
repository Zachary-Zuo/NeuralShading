# vMaterials 2 Metal 原生组合域与未来任意组合边界

## 1. 为什么 thumbnail 看起来高度可组合

vMaterials 2 Metal 有 837 个 authored exports，但 thumbnail 通常对应一个 preset/export，而不是一个可以独立拆装的组件。大量 thumbnail 来自三种复用：

1. 同一个 MDL 模板换 metal identity；
2. 同一个 module、同一 texture set 改 typed 参数默认值；
3. 少数复杂 module 预先写好若干 graph、mask 和 texture 的组合状态。

因此“视觉上像几层效果叠加”不等于 source 提供了独立的 `base × finish × patina × paint × dirt` 正交控制。判断组合是否具有 reference GT，必须看是否存在相应 native module/graph 和 typed schema，不能只看 thumbnail。

## 2. 当前 source 真正支持的组合

### 2.1 规则的 metal × finish 矩阵

最规则的区域是 13 种 metal identity × 7 种 primary finish：Base、Brushed、Foil、Hammered、Knurling、Scratched、Sheet，共 91 个 module、579 个 exports。扣除标准 Sheet 中 110 个 punched cutout presets 后，Metal-v1 的 opaque 部分有 469 个 exports。

这一矩阵为“换 metal identity”和“换 primary finish neural texture bundle”提供了最强的原生组合监督。个别 module 的参数数量、默认值或命名并不完全相同，因此它是 source-backed 的组合矩阵，不等于所有 graph tensor 都逐项相同。

### 2.2 稀疏的复合 recipe

规则矩阵之外有 36 个 module、258 个 exports，其中 223 个 opaque、35 个 cutout。opaque 特殊项包括：

- Copper 的 aging、antique brushed、brushed patina；
- Aluminum anodization；
- Brass/Bronze 的 antique 与 polished；
- blued steel、galvanized steel/zinc；
- painted steel 与 painted/cracked steel；
- pitted iron、carbon steel、cast metal；
- stainless brushed/milled；
- PCB copper/goldfinger、solder paste；
- diamond plate、metal mesh weave 等结构表面。

这些条目是 source 已定义的复合 recipe。Metal-v1 可以学习它们各自的 typed 参数域以及未见参数状态，但不能据此宣称其中的 patina、paint、rust、crack 或 pattern 已经成为可挂接到任意材质的独立组件。

## 3. 当前 family-local 合同下不支持的任意组合

只要 source 没有对应 module/graph/reference，以下组合都不属于 Metal-v1 的质量承诺：

| 超出范围的组合 | 示例 | 缺少的 reference 定义 |
|---|---|---|
| special overlay × 任意 metal | Gold + copper patina、Silver + anodization、Brass + galvanizing | 化学/光学参数、颜色与 roughness 响应、适用域 |
| special overlay × 任意 finish | hammered + patina、knurling + rust、foil + paint | mask 如何作用于已有 micro-normal/roughness，closure 如何混合 |
| 多个 primary finish 同时叠加 | brushed + scratched、milled + hammered | 空间频率、normal 合成、anisotropy 方向与层次顺序 |
| 跨 family 搬运 typed 参数 | 把 `cracks_darkness` 加到 Gold Foil，或把 `oxide_thickness` 加到 Stainless Milled | 参数对应的 graph node、texture channel、范围与耦合关系 |
| 任意 overlay 数量与次序 | dirt over paint over rust 与 rust under cracked paint | layer order、mask correlation、能量与覆盖规则 |
| 任意 structure × metal/finish | Copper diamond plate + scratched patina、Gold mesh weave | structure normal/AO/mask 与 metal/finish 的组合模板 |
| 枚举/资源域外插 | 新 pit pattern、枚举之间连续 morph | 新资源的语义、过滤、离散分支和 GT |
| 任意用户 texture set | 上传一组图片并当成新的 Scratched/Patina asset | channel role、transfer function、normal convention、LOD 与 authoring schema |
| 新 metal/alloy identity | catalog 外的合金或测量金属 | optical constants/reference 及其与 coating 的关系 |
| cutout/coverage 组合 | 任意 punched pattern 与上述材质叠加 | opacity、filtering、visibility 和 renderer composition 合同 |

这并不意味着模型将无法对其中某些组合产生“看起来合理”的结果；它意味着没有 source-authoritative GT 和 frozen semantics，不能把偶然好看的 extrapolation 作为受支持能力。

## 4. 后续支持任意组合需要新增什么

任意组合不是简单扩大训练集，而是新增一套 canonical authoring/reference contract。至少需要显式定义：

```text
MetalCore
× PrimaryFinish
× ordered Overlay[]
× ordered Coating[]
× optional Structure
→ authoritative composed reference
```

其中每个组件还要声明：适用 metal/finish 域、typed 参数、独立或共享 UV transform、输入/输出 channel、normal 合成、mask/coverage、layer order、closure mixture、过滤/LOD 与超域行为。只有这套 composition grammar 能被权威 reference 执行，才可以生成训练 GT 并评估新组合。

未来 neural 系统可以预留组件化 latent、typed token、graph token 和 ordered mixture 的容量，但 Metal-v1 不应把 source 中不存在的组合伪装成已验证的 native vMaterials 语义。

## 5. Metal-v1 与未来扩展的关系

Metal-v1 当前边界是：

- 对全部 692 个 opaque authored exports 保留 family-local typed editability；
- 对 13×7 标准矩阵学习 metal identity 与 finish asset 的组合结构；
- 对特殊复合 module 学习其 authored recipe，不拆成无依据的全局 overlay；
- 可以在架构中保留未来 component/ordered-overlay 插槽，但不把任意组合纳入当前验收；
- cutout 不进入 catalog。

这一边界既利用 thumbnail 背后的真实复用，也避免把视觉相似性误当作 source 已定义的正交材质空间。
