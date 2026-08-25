---
name: core-material-interface
description: 任意材质族的通用接口：MaterialProgram 保存原生语义、族专属 IR、独立 reference、三类数据分开命名、接入一个源材质族的验收条件与 capability error 规则
paths:
  - src/ncls/core/material/**
  - src/ncls/source_materials/**
  - src/ncls/references/**
  - references/**
  - shaders/ncls/contracts/layer_stack_ir.slang
  - docs/contracts/material_program.md
  - docs/material_scope.md
---

# 任意材质族的通用接口

> 统一的是目标运行时的查询方式（`compile_material → prepare → evaluate`，按 capability 加 `sample/pdf`），**不是** reference 内部的材质表示。不同源材质族不需要共享 IR、参数布局、求值算法或资源类型。

## 数据流

```text
源材质资产（原生参数 + 图结构 + 纹理/测量/几何资源）
  → MaterialProgram（面向编辑/存储/交换的有类型 DAG；只对能规范化的族）或原生文档（.mtlx / MERL .binary）
  → 族专属内部 IR（LayerStackIR 只属于 LayerStack）
  → 该族自己的 reference（随机游走 / 解析 / 查表 / 原生图）
  → 统一查询语义下的方向响应或图像 GT
  → 训练 / direct fit / 评测 → neural evaluator + latent + MethodBundle
```

## GT 原则（`docs/material_scope.md`）

- 源材质由什么参数、图和资源定义，由它自己决定。解析模型的公式与参数、可编辑图的节点与连接、纹理及其颜色空间 / 通道语义 / 过滤、测量表及其参数化，都是 GT 的一部分。
- 除非源材质本身就是层模型，否则不得要求它提供层参数，也不得为迁就 `LayerStackIR` 或某个 backend 先反演 / 改写后再称为 GT。
- 源材质原生可编辑的参数必须保留编辑能力，且 reference 对编辑后的状态重新正确求值。测量材质没有连续参数就不伪造（MERL 只能换测量表）。
- reference 直接求值原生语义，不得经过项目要研究的 neural material program；同一族可有多个 reference 交叉验证（LayerStack：随机游走 + pbrt coated probe）。

## 代码位置与身份

- `MaterialProgram` 节点注册表：`src/ncls/core/material/registry.py`（`ncls.interface.*@1`、`ncls.medium.homogeneous@1`、`ncls.composition.layer_stack@1`）。操作身份是 `(namespace, name, version)`；加载器遇到未知操作 / 版本必须返回 capability error。
- 规范化：`core/material/canonical.py` 去掉不影响物理语义的 metadata，输出 IR + SHA-256；split 与去重只用规范化后的物理语义 hash，不用文件路径或 JSON 字节顺序。
- 非 LayerStack 族的原生 adapter 在 `src/ncls/source_materials/`（OpenPBR resolved inputs、MERL 表、MaterialX 文档 / 纹理 / 可编辑 constant input）；它们保留各自的 identity（`source_asset_sha256` 与 `state_sha256` 分开：前者是原始资源，后者是加上编辑后的状态）。
- 所有 reference 在 `references/registry.json` 登记 package、role（`ground-truth` / `independent-validation`）与六项 capability 状态；实现 / 资产路径只通过 `ncls.references.resolve_reference_path()` 的三种 `path_root`（`project` / `external` / `source-materials`）解析，拒绝绝对路径与越界。

## 三类数据分开命名

| 类 | 内容 | 位置 |
|---|---|---|
| 源材质资产 / 语料 | 原始定义、参数、图、纹理、测量表 | `assets/source-materials/`、`external/`，由 `references/` manifest 锁定 |
| reference 响应数据 | 查询后的方向响应、方差、图像 | `data/reference-responses/*.h5`（只允许 `reference-shard` v5） |
| 方法产物 | latent、decoder、compiler、sampler、`MethodBundle`、评测结果 | `artifacts/` |

## 接入一个源材质族的验收条件（`docs/material_scope.md`「接入一个源材质族的验收条件」）

1. 原生定义与资源版本化保存或可复现导入（先在 `references/<package-id>/` 记录 URL、commit、许可证、清单，再由脚本下载并校验 hash）；
2. 原生可编辑参数 round-trip，不丢失、不偷偷重参数化；
3. 一个不依赖统一近似方法的 reference；
4. 明确的查询域、颜色 / 光谱语义、方向约定和适用范围（`ReferenceDescriptor.incident_domain`、`capabilities`）；
5. viewer 或离线工具能正确呈现原始材质（当前四族都由 Falcor viewer 直接呈现）；
6. reference 身份、源资源和参数状态可被数据集逐项追溯；
7. 后续统一方法按输出行为评测，不把内部参数相似当正确性标准。

源材质族可以先于 neural backend 接入；backend 尚未覆盖时报告 capability 缺失，不影响该族作为 GT 存在。

## 反例

- 为了让 MERL 进 `LayerStackIR` 而拟合出一组"等效层参数"当 GT。
- 加载 `.mtlx` 时把纹理驱动的 roughness 用 scene override 静默替换。
- 新 reference 只把源码丢进 `tools/` 或 `external/` 而不在 `references/registry.json` 登记。
- 用 `metadata` 塞物理输出（`interior_medium`、`emission` 等有预留槽位）。
