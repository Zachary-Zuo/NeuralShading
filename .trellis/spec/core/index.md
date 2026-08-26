---
name: core-index
description: 公共核心层入口：MaterialProgram / 内部 IR / reference / 散射合同 / MethodBundle 五个公共合同的所有权、开发前检查与质量检查
paths:
  - src/ncls/core/**
  - src/ncls/references/**
  - src/ncls/bundle/**
  - shaders/ncls/contracts/**
  - shaders/ncls/scattering/**
  - shaders/ncls/backends/**
  - docs/contracts/**
---

# 公共核心

> 公共核心只保存三块共同遵守的合同和共用 shader，不形成第四条业务链路。这里的每个改动都会同时影响数据采集、训练评测和 viewer。

## 五个公共合同

| 合同 | 权威文档 | 代码 |
|---|---|---|
| `MaterialProgram`（唯一公共材质描述） | `docs/contracts/material_program.md` | `src/ncls/core/material/{program,registry,canonical,layer_stack}.py`、`schemas/material_program_v1.schema.json` |
| 内部 IR（族专属，如 `LayerStackIR`） | 同上「LayerStack 子图」 | `core/material/abi_layout.py` → `shaders/ncls/contracts/layer_stack_ir.slang` |
| reference 身份与 package | `references/README.md`、`docs/material_scope.md` | `src/ncls/references/`、`references/registry.json` |
| 散射合同（`prepare + evaluate`，capability 加 `sample/pdf/integrate_*`） | `docs/contracts/scattering_backend.md`、`docs/realtime_material_compilation.md` | `core/scattering/{contract,abi_layout}.py` → `shaders/ncls/contracts/scattering_{contract,backend}.slang` |
| `MethodBundle` | `docs/contracts/method_bundle.md` | `src/ncls/bundle/manifest.py`、`schemas/method_bundle_v1.schema.json` |

详细规则：

- `material-interface.md`：任意材质族的接入合同（GT 原则、IR、reference、验收条件）。
- `shared-slang-backend.md`：散射合同的实现规则与"每个方法一份 Slang"的架构。

`shaders/ncls/scattering/` 是上述散射合同共用的 Falcor-free 数学实现，不新增第六个公共 ABI：它统一拥有 frame、cosine/LTC/GGX、Fresnel、fixed-size mixture 与方向 `sample/pdf`，backend 和 reference 只做语义适配。

## 开发前检查清单

- [ ] 我改的是公共合同还是某个块的私有实现？私有实现（backend 的 `State` 字段、latent 维数、shader 布局）不得进入公共合同。
- [ ] 改 ABI / schema 时：JSON → 生成的 `.slang` → dataclass → 测试 → `docs/contracts/` 一起改，版本号按合同规则递增（节点语义变更升 `operation_version`，容器不兼容才升 `schema_version`）。
- [ ] 新能力用 capability 表达，不给结构体加"看起来通用"的字段；入口存在不等于 capability 成立。
- [ ] 已读 `project/method-constraints.md`（若涉及 backend）。

## 质量检查

- [ ] Python 与 Slang 的枚举 / 布局数值都来自同一份 `abi/*.json`，没有手写副本。
- [ ] 不支持的操作 / 版本 / capability 返回明确错误，没有"套用相近实现"的静默 fallback。
- [ ] 合同测试（`tests/unit/test_material_program.py`、`test_scattering_contract.py` 与各 backend/bundle 测试）和 GPU 编译冒烟（`tests/gpu/kernels/*.cs.slang`）覆盖了改动。
- [ ] 改公共方向分布时，GPU oracle 直接包含生产 Slang，并同时覆盖独立固定公式值、PDF quadrature、sample histogram、`sample.pdf == pdf(direction)` 与 null mass；不能只让 sample 与同源 PDF 互相自证。
- [ ] `docs/contracts/` 的描述与代码命名一致（`p1_v2_plan.md` D4.6 列出的已知不一致要一并修）。
