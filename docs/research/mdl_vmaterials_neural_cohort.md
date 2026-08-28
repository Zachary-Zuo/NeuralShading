# 首批 MDL neural 材质族

本文冻结首批需要由 neural material program 适配的 MDL 材质族，以及 Base Materials、vMaterials 2 和 Automotive Materials 三个 NVIDIA 材质包在当前研究中的分工。这里的“首批”指接下来的数据采集、训练与未见状态评测范围，不表示为每个 preset 分别训练模型。

## 1. 当前结论

- 首批 neural cohort 由 vMaterials 2.4.0 中的 11 个 module 构成，共 172 个 authored exports；它们全部位于 `Materials/vMaterials_2/` 下。
- 一个 module 一旦入选，就保留其全部 authored presets。筛选只发生在 module/材质族之间，不在入选 module 内按颜色或粗糙度删除 preset。
- 同一材质族训练一个支持原生 source 参数的 neural program；冻结权重后分别测试已见状态、未见插值、未见端点和离散资源切换。不得把每个颜色或粗糙度 preset 当成独立模型。
- Base Materials 与 Automotive Materials 已完成 archive 结构、依赖和缩略图层面的筛选，但不进入首批适配，也不需要为此完整解压。
- `references/mdl-vmaterials2-v1/families.json` 已由锁定 MDL SDK 枚举并逐 preset 审计：11 个 families、172 个 presets、164 个 opaque runtime-supported 状态和 8 个 punched cutout 状态；这份 catalog 是后续 neural source locator 与评测分组的入口。

## 2. 本地包状态

| pack | 本地状态 | 当前用途 |
|---|---|---|
| vMaterials 2.4.0 | ZIP 已下载，并完整展开到 `assets/source-materials/mdl-vmaterials2/2.4.0/` | 首批 MDL reference 与 neural 适配来源 |
| Base Materials | `Base_Materials_NVD@10013.zip` 已下载，未完整解压 | 已分析、暂缓；首批不读取 |
| Automotive Materials | `Automotive_Materials_NVD@10011.zip` 已下载，未完整解压；只把 302 张当前版本缩略图抽到 `artifacts/` 做筛选 | 已分析、暂缓；首批不读取 |

两个暂缓包当前不需要解压，也不进入训练、reference 采集或 viewer 的首批运行路径。ZIP 是否长期保留只是存储决策；删除不会改变首批 cohort，但以后复查相应材质时需要重新下载。

## 3. 首批 11 个材质族

| 分组 | 材质族 | 原始 module 相对路径 | authored exports | 主要覆盖 |
|---|---|---|---:|---|
| 参数化主族 | Versailles | `Ceramic/Ceramic_Tiles_Glazed_Versailles.mdl` | 27 | glaze、mortar、craquelure、污渍、粗糙度与 tile variation |
| 参数化主族 | Carpaint Metallic | `Paint/Carpaint/Carpaint_Metallic.mdl` | 31 | metallic base、clearcoat、flake 与 roughness 状态 |
| 参数化主族 | Carpaint Shifting Flakes | `Paint/Carpaint/Carpaint_Shifting_Flakes.mdl` | 31 | measured curve、方向性色变、flake layer 与 clearcoat |
| 参数化主族 | Effect Pigment Metallic | `Paint/Carpaint/Effect_Pigment_Metallic.mdl` | 11 | 随观察角变化的 effect pigment |
| 参数化主族 | Velvet | `Fabric/Velvet.mdl` | 15 | sheen closure、织物 normal 与 imperfections |
| 参数化主族 | Copper Antique Brushed Patinated | `Metal/Copper_Antique_Brushed_Patinated.mdl` | 9 | brushed direction、patina、smudge 与 metal/patina blend |
| 参数化主族 | Aluminum Scratched | `Metal/Aluminum_Scratched.mdl` | 6 | scratch normal、相关 mask、污渍与粗糙度 |
| 参数化主族 | Retroreflective Material | `Other/Retroreflective/Retroreflective_Material.mdl` | 7 | backscattering glossy response |
| 离散子族 | Carbon Fiber | `Composite/Carbon_Fiber.mdl` | 8 | 方向性编织、coat、worn/aluminized 状态 |
| 离散子族 | Suede Leather | `Leather/Suede_Leather.mdl` | 16 | suede 颜色连续编辑，以及 punched/cutout 资源和拓扑切换 |
| 空间资源对照 | Wood Tiles Pine | `Wood/Wood_Tiles_Pine.mdl` | 11 | tile layout、wood texture、rotation/scale variation；主要检验空间资源而非新增 closure |

前三类属于同一首批 cohort，但报告时必须分开：参数化主族用于连续编辑泛化；Carbon Fiber 与 Suede 的资源或 closure 切换作为离散子族；Wood Tiles Pine 作为空间资源和 NVIDIA neuralappearance 管线对应对照。

## 4. authored preset 与参数 probe

NVIDIA 缩略图适合筛掉只改变颜色或图案、没有新增研究价值的 module，但不能替代参数级审计。入选 module 应同时保存两类状态：

1. **authored presets**：保留原 module 导出的全部 preset，作为官方设计状态；
2. **parameter probes**：在同一原生材质上系统改变少数参数，形成未见插值、端点和组合 holdout。

Versailles 的 27 个 exports 中，有 12 个状态的 `glazing_craquelure_weight > 0`，15 个状态的 `drops_amount > 0`，但只有 `Ceramic_Tiles_Versailles_Antique_White_Dirty` 明确设置了非零 `dirt_weight`。裂纹是共享主材质的参数，不是单独命名为 `Cracked` 的一组 exports；在 256×256 固定光照缩略图中也很难看清。该族后续至少应构造：

- `craquelure_weight` 的关闭、插值和强端点；
- `dirt_weight` 的干净、插值和污渍端点；
- `tiles_roughness` 的插值与端点；
- 裂纹、污渍和粗糙度的组合 holdout。

这些 probe 仍由同一 MDL reference 正确呈现，不创建新的 source family，也不为每个 probe 重新训练 neural program。若 authored preset 实际改变纹理资源、cutout 或 closure 拓扑，则将其标成离散子族并单独报告。

## 5. 资产与材质族的组织

完整 vMaterials 解包目录在筛选阶段作为原生 source root，保持不动：

```text
assets/source-materials/mdl-vmaterials2/2.4.0/
  PACKAGE-INFO.yaml
  PACKAGE-LICENSES/
  Materials/vMaterials_2/...
```

不能把入选 `.mdl` 按“参数化”“离散子族”等研究分类直接移动到新目录。MDL module name、传递 import 和纹理 URI 都依赖原始相对结构。语义分组已经登记在 `references/mdl-vmaterials2-v1/families.json`，其中包含：

- family id、原始 module 和主 export；
- 全部 authored preset exports；
- 连续可编辑参数轴；
- 离散资源/closure 子族；
- reference、训练与评测角色；
- module、传递 import、target texture 与 SDK runtime data 的依赖闭包。

当前 catalog 去重后引用 81 个 pack-relative source resources；每个 preset 另记录 SDK BSDF-data content hash、compiled material hash 和三个 sub-expression hash。Wood 的 11 个 authored exports 被资源签名分成 10 个离散 resource-set；Suede 的 8 个 opaque 状态共享一个 resource-set，另外 8 个 punched 状态组成 `punched-cutout` 子族；Carbon Fiber 的 8 个状态保留为 authored opaque 离散对照。

资源格式 support 与 closure 输出 support 分开：punched atlas 的 SDK canvas type 是 `Rgba_16`，bridge 会完整保存 `1024×1024×4×Uint16` payload，generic Falcor binder 与 viewer 使用 `RGBA16Unorm`，不量化到 8-bit。8 个 punched preset 当前不可执行的唯一原因是公共 `MaterialProgram`、query、training 与 viewer 尚未定义 `geometry.cutout_opacity` 的输出和组合语义；它们仍完整留在 catalog，等待后续 opacity 专项任务。

最终需要释放空间时，生成独立的精简包，而不是从原包中零散搬文件：

```text
assets/source-materials/mdl-vmaterials2-curated/v1/
  Materials/vMaterials_2/<保持原始相对路径的入选 module 与依赖>
```

精简包独立通过 MDL 编译、资源闭包和 reference 查询后，再把正式 reference 切换到该 root；届时完整解包目录和 ZIP 才成为可选缓存。

## 6. Base 与 Automotive 的暂缓结论

### Base Materials

Base Materials 的主要价值是 Omniverse 场景依赖和基础材质覆盖。抽查的 leaf modules 多为零参数 wrapper，并依赖包内未携带、当前项目也没有正式登记的 `OmniPBR` / `OmniPBR_Opacity` template。它对首批 fancy closure、层结构和编辑泛化的增量有限，因此不进入当前 neural cohort。

### Automotive Materials

Automotive 当前版本有 302 个 leaf materials：270 个调用 `OmniUber_Automotive`，32 个调用 `OmniGlass`，leaf wrapper 本身不暴露可编辑参数。`OmniUber_Automotive` 虽有大量参数和纹理入口，但继续依赖包内未携带的 `OmniSurface::*` 与 `nvidia::core_definitions`。因此它不能只靠挑选几个 leaf `.mdl` 就成为当前可独立加载的 source family。

Automotive 仍有后续价值，尤其是 `Metal_Polished_01/02` 与 `Metal_Polished_Dirty_01..03` 的 clean/dirty 层结构、汽车玻璃和带 emission 的内饰材质。但 clean/dirty wrapper 会改变层拓扑，不应伪装成简单颜色或粗糙度编辑。等首批 vMaterials cohort 验证完 parameter-aware neural program 后，再以完整 template 依赖闭包作为专项接入。

本结论基于 2026-08-28 的本地 archive、MDL source 和官方 thumbnails 审计。
