# NVIDIA 2026 仓库 MaterialX 审计

## 1. 论文材质与仓库示例不是同一批资产

2026《Taming Optimization Variance in Compact Neural Shading Networks》正文 Figure 4 使用两组共 15 个材质：

- 9 个 NVIDIA da Vinci Workshop 物体：Birdcage、Chandelier、Hammer、Lantern、Mirror、Palette、Chair、Scales、Table。论文把原 MDL 烘焙成单 UV tile 的 4K SVBRDF，再转换成 USDPreviewSurface。
- 6 个内部多层材质：Scratched steel、Bumpy plastic、Oxydized metal、Gold and ceramic、Brushed brass、Glazed ceramic。每个材质由 base material 与若干 glazing、stain、dust 等 top layers 构成，各层可有独立参数、纹理和 normal map。

正文没有发布这 6 个多层材质的原始图与纹理。公开仓库的三个派生 Figure 1 配置都继承 `configs/default.json`；默认 reference 是 `FauxLeather.mtlx`。因此这些配置复现的是训练算法与网络规模对照，不是把论文 Figure 1 / Figure 4 的内部多层 GT 一并开源。

第一方来源：

- 论文与 supplemental：https://research.nvidia.com/labs/rtr/publication/bitterli2026taming/
- 仓库 README：https://github.com/NVlabs/neuralappearance
- 默认配置：https://github.com/NVlabs/neuralappearance/blob/305b4b9c12e679398c487603dd8245c3f348526c/configs/default.json

## 2. 三个公开 `.mtlx` 的原生图

审计固定公开仓库 commit `305b4b9c12e679398c487603dd8245c3f348526c`。全部纹理均为 4096×4096；文档均声明 MaterialX 1.38，使用 `uvtiling=(1,-1)` 翻转 V，以 `power(2.2)` 手工解码 base color，并对 roughness 做平方。

| 文档 | closure 结构 | 空间资源 | 语义评价 |
|---|---|---|---|
| `Bark.mtlx` | `surface → layer(dielectric_bsdf, oren_nayar_diffuse_bsdf)`；coat 与 diffuse 共用 normal | basecolor、roughness、normal | 三者中最像真正的 layered material；适合测试 closure layer、normal 与粗糙度纹理，但仍只有一层空间法线和简单 coat |
| `FauxLeather.mtlx` | `surface → conductor_bsdf`；base color 经 `artistic_ior` 变成 conductor 的复 IOR | basecolor、roughness、normal | 名称是皮革，原生 closure 却是 conductor；适合训练稳定性与高光压力 smoke test，不宜作为“皮革物理 reference” |
| `PatternedMetal.mtlx` | `surface → mix(conductor_bsdf, layer(dielectric_bsdf, oren_nayar_diffuse_bsdf))`；metallic texture 控制 closure mix | basecolor、metallic、roughness、normal | 三者中最适合做空间分支与 metal/dielectric 混合 fixture；视觉上是图案金属，不等同于划痕、氧化或 flake model |

材质源码：

- https://github.com/NVlabs/neuralappearance/blob/305b4b9c12e679398c487603dd8245c3f348526c/assets/materials/Bark.mtlx
- https://github.com/NVlabs/neuralappearance/blob/305b4b9c12e679398c487603dd8245c3f348526c/assets/materials/FauxLeather.mtlx
- https://github.com/NVlabs/neuralappearance/blob/305b4b9c12e679398c487603dd8245c3f348526c/assets/materials/PatternedMetal.mtlx

## 3. “可用”必须分成三个层次

### 3.1 在 NVIDIA 仓库内：可用

仓库把默认 `FauxLeather.mtlx` 作为可直接训练的 reference，并列出 Bark 与 PatternedMetal 作为可选项。其 loader 明确设置：

```text
mtlx_source = houdini
mtlx_layering_mode = bsdf_mix
```

所以它们的可运行合同是“falcor2 的 Houdini MaterialX 兼容输入 + 指定 layering mode”，不能直接外推为任意 MaterialX implementation 的便携文档。

第一方来源：https://github.com/NVlabs/neuralappearance/blob/305b4b9c12e679398c487603dd8245c3f348526c/neuralappearance/datagen/reference_materials.py

### 3.2 在锁定的官方 MaterialX 1.39.4 中：不能原样视为 conformant 文档

在完整 Windows 环境中，使用项目锁定 `external/MaterialX` 的官方 `mxvalidate.py --stdlib` 分别验证三份原文档，三者都报告 `is not a valid MaterialX document in v1.39.4`。错误分为：

- 同一个 input 同时存在 connection 与空 `value`，触发 `Node input has too many bindings`；
- `tiledimage` / `power` 链在 `color3` 与 `vector3`、scalar 与 `vector2` 之间存在类型不匹配；
- PatternedMetal 还叠加了两套 closure 的同类问题。

这不否定 NVIDIA falcor2 的兼容 loader 能运行它们，但证明“我们已经锁定 MaterialX 1.39.4”不足以让文件直接成为 portable reference。若归一化文档，必须把原文件保留为 source identity，并用 NVIDIA falcor2 输出与归一化图做 parity；不能把修复后的派生图冒充原始 GT。

### 3.3 在当前 NeuralShading `materialx.textured-surface@1` 中：三者都不能直接进入

当前 `_parse_surface()` 要求 `surfacematerial` 直接绑定 root-level `standard_surface`，并只解析 base color、roughness、metalness、normal 与有限常量。三份 NVIDIA 文档都先绑定 `<surface>`，再连接 closure BSDF graph；纹理也使用 root-level `tiledimage + power + constant`，不是当前 adapter 要求的 `nodegraph/image/explicit texcoord` 形态。因此三者会在解析 surface binding 时被拒绝，尚未到纹理加载阶段。

本地证据：`src/ncls/data/providers/materialx.py` 的 `_parse_surface()`。

## 4. 许可与 provenance

- 仓库代码与 `.mtlx` 位于 Apache-2.0 仓库。
- `assets/README.md` 把 Bark、FauxLeather、PatternedMetal 的纹理归因到 cgbookcase，并标为 CC0 1.0；其中 README 把 FauxLeather 拼成了 `FeauxLeather`，登记 manifest 时要保留原始链接并修正文档拼写映射。
- 每组纹理都是 4K：Bark 3 张、FauxLeather 3 张、PatternedMetal 4 张。正式接入应固定原文档、纹理 SHA-256、上游 commit、source dialect 与颜色变换语义。

第一方来源：https://github.com/NVlabs/neuralappearance/blob/305b4b9c12e679398c487603dd8245c3f348526c/assets/README.md

## 5. 推荐用途

1. `Bark`：作为完整 closure graph reference 的最小 layered acceptance material；不改写为当前 `standard_surface` 后仍叫同一个 GT。
2. `PatternedMetal`：作为 closure `mix`、metal/dielectric 分支与空间 mask 的 acceptance material；三者中研究价值最高。
3. `FauxLeather`：保留为 NVIDIA training pipeline 对照和方言兼容 fixture；不把它当作权威 leather model。

近期若不扩 reference，三者都不能无修改加入现有语料。后续实现应在以下两条边界中二选一：

- 新增 versioned `materialx.closure-graph@1`，用锁定的通用 closure 求值器保留完整图语义；
- 增加显式 `houdini-materialx@1` compatibility normalization，并以 NVIDIA falcor2 reference 做逐方向/图像 parity。

无论选择哪条，都应保留原始 `.mtlx` 和纹理为 source GT，把归一化图或生成 shader 当作可重建的 compiler artifact。
