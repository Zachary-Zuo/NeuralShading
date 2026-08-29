# vMaterials 2 Metal opaque typed 参数审计

## 1. 它是什么

本文只统计用户已经确认进入 Metal-v1 capability 边界的 opaque source；145 个使用 `geometry.cutout_opacity` 的 exports 不进入 Metal-v1 catalog，因此其 cutout 专用参数也不进入下面的参数合同。

在 692 个 opaque authored exports 中，共出现 154 个唯一的 `参数名 × MDL 类型` 组合，分布在 64 套 family-local parameter schema 中。单个 export 实际暴露 9–31 个参数，不存在一个原生的“全材质统一 154 维参数向量”。同名参数在本次审计中没有发生类型冲突。

这里的 typed 参数是 source MDL export 的 authored arguments；它们与 module、图结构、固定 texture resources 一起决定 reference GT。全部 837 个 Metal exports 都没有暴露 editable `texture_2d` 参数，所以 finish neural texture bundle 的替换是新的 compiled-asset 能力，不是某个既有 typed 参数。

## 2. 类型构成

| MDL 类型 | 唯一参数名 | 原生含义 |
|---|---:|---|
| `float` | 130 | 连续标量；既有常见的 `[0,1]` 权重，也有角度、毫米、尺度和未声明范围的量 |
| `bool` | 10 | 离散开关，不能当作连续插值轴 |
| `color` | 8 | 线性颜色/反射颜色，应保留三通道语义 |
| `enum` | 3 | 离散模式或资源选择，应按枚举域编码 |
| `float2` | 2 | UV scale/translation 二维向量 |
| `int` | 1 | UV space index，是离散索引而不是连续数值 |
| **合计** | **154** | **64 套局部 schema 的并集** |

全目录共有 163 个不同参数名；相对 opaque 子集多出的 9 个名字全部是 cutout 专用参数：`cutout_bevel_width`、`cutout_bump_strength`、`cutout_roundness`、`cutout_size`、`hexagonal_grid`、`horizontal_offset`、`punching_grid_size`、`shape_select`、`square_grid_offset`。它们不进入 Metal-v1 catalog 或 neural 参数合同。

## 3. 参数在语义上控制什么

下面的分组用于澄清职责，不改变 source schema，也不表示现在已经选定模型结构。

| 语义组 | 主要控制内容 | 代表参数 |
|---|---|---|
| 空间坐标与纹理寻址 | UV 选择、投影、平铺、缩放、平移和旋转 | `texture_scale`、`texture_translate`、`texture_rotate`、`uv_space_index`、`infinite_tiling`、`coordinate_system`、`projection_type`、`no_uv`、`scale` |
| 几何 frame 与 rounded corner | 基于物体/相邻材质修正 shading frame，以及 bump 的对象空间尺度 | `enable_round_corners`、`roundcorners_enable`、`radius`、`radius_mm`、`roundcorner_radius`、`roundcorners_radius_mm`、`across_materials`、`roundcorners_across_materials`、`object_scaled_bump` |
| 金属光学身份与基础反射 | 金属颜色、法线/掠射反射率、roughness、anisotropy 和反射强度 | `metal_color`、`metal_tint`、`normal_reflectivity`、`grazing_reflectivity`、`metal_roughness`、`roughness`、`reflection_roughness`、`reflection_brightness`、`steel_anisotropy`、`copper_tint` |
| finish 与微结构 | brushed、scratch、knurling、milling、dent、normal/bump 和局部 roughness 变化 | `brush_width`、`brushing_anisotropy`、`brushing_bump_strength`、`scratch_bump_factor`、`scratches_variation`、`knurling_roughness`、`milling_bump_strength`、`uneven_normal_strength` |
| 老化、污染与腐蚀 | dirt、smudge、wear、rust、oxide、patina、stain、damage 与 cavity effect | `age`、`dirt_amount`、`smudge_amount`、`wear`、`bright_rust_amount`、`oxide_thickness`、`patina_amount`、`corrosion_offset`、`damage` |
| 涂层、油漆与复合表面 | anodization、paint、polish film、金属/污染层混合与裂纹 | `enable_coating`、`anodization_roughness`、`paint_color`、`paint_roughness`、`polish_film_strength`、`patina_metal_blend`、`cracks_bump_strength` |
| 离散模式与资源选择 | 投影/坐标模式、pit 纹理分支、UV set 和功能开关 | `coordinate_system`、`projection_type`、`pit_texture_selection`、`uv_space_index` 以及全部 `bool` |

一些参数跨组耦合。例如 `texture_scale` 既改变 texture sampling，也改变以纹理驱动的 normal/bump 空间频率；`patina_metal_blend` 同时改变颜色、roughness 与 closure mixture。后续训练不能用“每个参数只影响一个 PBR channel”的假设替代 native reference。

## 4. 对系统合同的直接含义

“保留全部原生 typed 参数”不等于“把 154 个槽全部送进同一个 MLP”。更忠于 source、也更适合随机访问运行时的合同是：

1. 每个 module/family 保留自己的 typed schema、存在性 mask、类型、枚举域和 source range；缺失参数不能与数值零混为一谈。
2. 连续外观参数由 source compiler 编成 bounded material condition；`bool`、`enum` 和索引采用离散编码，不承诺枚举值之间可插值。
3. UV/投影参数由 `prepare()` 和 neural texture accessor 执行，确保它们真实改变随机访问位置与过滤，而不是只作为网络提示。
4. rounded-corner 参数保留编辑入口与 reference 语义，但其几何邻域/frame 计算应由 renderer 或 `prepare()` 完成；evaluator 接收最终 shading frame，不要求小 MLP 从一个半径数值猜测几何法线。
5. finish-specific 参数仍然可以同时条件化 texture decoder 与 directional evaluator；分流是运行职责划分，不是删除自由度。

因此，未来公共表示应是“canonical common fields + family-local typed tokens/mask + graph/module identity”，而不是稠密 154D 零填充向量。最终字段、预算和编码方法要在需求确认后进入 `design.md`，本文只冻结 source 事实与语义边界。

从材质编辑角度，neural modeling 的主要对象是第 3–6 组：金属光学、finish 微结构、老化污染和涂层复合，因为它们直接改变局部散射、空间外观或 closure mixture。第 1–2 组不是次要功能，而是更适合确定性实现的编辑语义：坐标参数应真实改变 neural texture 的随机访问与过滤，几何 frame 参数应真实改变 evaluator 接收到的 shading frame。这样可以把模型容量用于需要学习的 source response，同时保持全部 typed editability。

## 5. 完整 opaque 参数名

### 5.1 `bool`（10）

`across_materials`、`enable_anodization_bump`、`enable_coating`、`enable_round_corners`、`enable_rust_damage`、`infinite_tiling`、`no_uv`、`object_scaled_bump`、`roundcorners_across_materials`、`roundcorners_enable`

### 5.2 `color`（8）

`color_1`、`grazing_reflectivity`、`metal_color`、`metal_tint`、`normal_reflectivity`、`paint_color`、`rod_color`、`weave_color`

### 5.3 `enum`（3）

`coordinate_system`、`pit_texture_selection`、`projection_type`

### 5.4 `float2`（2）

`texture_scale`、`texture_translate`

### 5.5 `int`（1）

`uv_space_index`

### 5.6 `float`（130）

`abrasion`、`age`、`age_variation`、`anodization_bump_size`、`anodization_roughness`、`AO_weight`、`balance`、`bare_metal_darkness`、`brass_type_select`、`bright_rust_amount`、`brightness`、`brightness_variation`、`brush_brightness_variation`、`brush_darkening`、`brush_gloss_variation`、`brush_height_blur`、`brush_width`、`brushing_anisotropy`、`brushing_bump_strength`、`brushing_roughness`、`bump_amount`、`bump_amount_1`、`bump_amount_2`、`bump_amount_3`、`bump_factor`、`bump_strength`、`cavities_darkening`、`cavities_dirt`、`cloudiness`、`copper_bump_amount`、`copper_tint`、`coppery`、`corrosion_offset`、`cracks_bump_strength`、`cracks_darkness`、`damage`、`damages_scale`、`dents_bump_strength`、`dents_scale`、`diffuse_variation`、`dirt_amount`、`dirt_brightness`、`dirt_dents`、`dirt_spots_weight`、`dirt_transition_softness`、`dirt_weight`、`drops_variation`、`factor`、`flow_stains_balance`、`grooves_dirt`、`heat_treatment_amount`、`imperfections_amount`、`impurities_weight_1`、`impurities_weight_2`、`impurities_weight_3`、`knurling_roughness`、`leak_dirt_weight`、`metal_roughness`、`metalness`、`metalweave_roughness`、`metalweave_roughness_variation`、`milling_bump_strength`、`oxidation_amount`、`oxide_contrast`、`oxide_roughness`、`oxide_thickness`、`paint_roughness`、`paint_roughness_variation`、`paint_stroke_normal_strength`、`patina_amount`、`patina_brightness`、`patina_bump_amount`、`patina_metal_blend`、`patina_metal_blend_softness`、`patina_spots_brightness`、`pattern_shininess`、`polish_film_roughness`、`polish_film_strength`、`radius`、`radius_mm`、`reflection_brightness`、`reflection_contrast`、`reflection_roughness`、`reflection_smooth`、`rod_roughness`、`rough_scratches_attenuation`、`rough_scratches_roughness`、`roughness`、`roughness_amount`、`roughness_metal_surface`、`roughness_scratches`、`roughness_variation`、`roundcorner_radius`、`roundcorners_radius_mm`、`rust_brightness`、`rust_saturation`、`scale`、`scratch_bump_factor`、`scratch_reflection_variation`、`scratch_variation_amount`、`scratches`、`scratches_abrasion`、`scratches_bump`、`scratches_bump_factor`、`scratches_bump_strength`、`scratches_variation`、`sheet_brightness`、`shiny_scratches_brightness`、`shiny_scratches_roughness`、`smudge_amount`、`smudges`、`smudges_1_weight`、`smudges_2_weight`、`smudges_falloff`、`smudges_roughness`、`smudges_scale`、`spotsdirt_amount`、`steel_anisotropy`、`steel_roughness`、`streaks`、`surface_dirt`、`surface_dirt_amount`、`surface_impurities`、`texture_rotate`、`uneven_normal_strength`、`wash_weight`、`wear`、`weave_color_variation`、`weave_roughness`、`zinc_roughness`

## 6. 证据边界

数量、名称和类型来自锁定 MDL SDK 对 692 个 opaque exact exports 的 metadata-only class compilation。参数的 authoritative minimum/maximum、soft range、默认值、枚举域和 module 归属仍保存在逐 export inspection 中；后续冻结 cohort 时应把它们编译成版本化 schema manifest。metadata-only 结果不代替 decoded texture、GPU query 或参数极值的 reference 验证。
