---
name: project-index
description: NeuralShading 项目级规则入口：三大功能块与公共核心、依赖方向、术语、文档语言、各层 spec 导航
---

# NeuralShading 项目级规则

> 本目录是所有层共享的规则。写任何代码或文档前先读这里，再读对应层的 `index.md`。
> 权威事实在 `AGENTS.md` 与 `docs/`；spec 只写"在这个仓库里怎么干活"，不复制定义，避免三处漂移。

## 项目一句话

把多种保持原生语义的源材质族编译成统一、随机访问、运行成本有界的 neural material program：`compile_material → prepare → evaluate(wo, wi)`，需要材质驱动方向采样时再加与 evaluator 匹配的 `sample/pdf`。完整定义见 `docs/realtime_material_compilation.md`。

## 三大功能块 + 公共核心（`docs/architecture.md`「三个业务块和公共核心」）

| 块 | 代码位置 | spec 层 |
|---|---|---|
| 源材质接入与数据采集 | `src/ncls/data/`、`src/ncls/source_materials/`、`src/ncls/references/`、`references/`、`shaders/ncls/reference/`、`shaders/ncls/data/`、`configs/corpus/`、`tools/reference/` | `data/` |
| 训练、评测与导出 | `src/ncls/learning/`、`src/ncls/bundle/`、`configs/learning/`、`configs/evaluation/`、`docs/research/` | `learning/` |
| Windows viewer | `apps/viewer/`、`patches/`、`scripts/build_viewer.ps1`、`scripts/benchmark_viewer.ps1`、`configs/viewer-*.json` | `viewer/` |
| 公共核心（不形成第四条工作流） | `src/ncls/core/`、`shaders/ncls/contracts/`、`shaders/ncls/scattering/`、`shaders/ncls/backends/`、`docs/contracts/` | `core/` |

依赖方向固定：三块都只依赖公共核心，块之间只通过 `MaterialProgram`、`reference-corpus`/HDF5 shard、`MethodBundle` 三种产物交换数据。viewer 不依赖训练代码或 PyTorch；训练侧只能通过 `MethodBundle` 向 viewer 交付方法。

## 本目录文件

| 文件 | 何时读 |
|---|---|
| `dev-environment.md` | 每个会话开始、任何"验证 / 测试 / 构建 / 训练"之前 |
| `method-constraints.md` | 提出、注册或实现任何新候选方法或 backend 之前 |
| `code-organization.md` | 新建文件、拆文件、写诊断脚本、改接口或字段名之前 |
| `coding-conventions.md` | 写 Python / Slang / C++ 代码时 |

## 统一术语（`docs/architecture.md`「统一术语」）

- `reference`：对某个源材质族具有权威语义的求值实现；新代码和文档不用 `teacher`。
- `direct fit`：不经通用 compiler，直接优化候选表示的 latent 或参数。
- 源材质资产 / reference 响应数据 / 方法产物是三类东西，分开命名不混用（`docs/material_scope.md`）。
- 有限数据和预算下的实验不称"上界/下界"；用 `optimized-code control`、`high-capacity teacher`、`best observed candidate` 等限定表述。
- `runtime_class=realtime` 只表示运行时完整、有界并满足声明的能力，不是质量宣称。

## 文档与语言

- 项目 Markdown、实验报告、`.trellis/spec/` 与 `.trellis/tasks/` 下的全部产物（`prd.md`、`design.md`、`implement.md`、`research/`）统一以中文为主体。文件名、标识符、命令、数学符号，以及 `tile`、`p95`、`closure`、`packet` 等便于准确交流的术语保留英文；不为追求"全中文"造生硬译名。
- 写作顺序：它是什么 → 为什么需要 → 当前结论 → 下一步做什么。
- `docs/*.md` 与 `docs/contracts/` 是稳定文档，只在实现或公共合同的客观事实改变时修改；研究候选、实验顺序和阶段性判断留在 `docs/research/`；单次运行的报告进 `artifacts/`，不进 Git。
- 研究方向变化时直接修改或删除旧内容，不叠加"旧说法不再适用"的注释层（`docs/research/README.md` 维护规则）。
- 物理或工程限制必须写明适用范围，不写成普遍定律。

## 根仓库边界（`docs/repository_policy.md`）

`external/`、`assets/`、`data/`、`build/`、`artifacts/`、`reports/` 全部被 Git 忽略且职责固定；`data/` 只放 `reference-responses/` 下由 reference 导出的 HDF5。上游克隆（Falcor 8.0、pbrt-v4、OpenPBR、openpbr-bsdf、GLM、MaterialX）必须保持锁定提交与干净工作树；需要改上游时先把补丁放进 `patches/` 并在 `AGENTS.md` 说明。

## 唯一环境

Python 只用 Conda 环境 `neural-shading`（`environment.yml` + `requirements-torch-cu128.txt`，PyTorch 2.11.0 / CUDA 12.8）；需要 Falcor 模块时经 `scripts/run_falcor_python.ps1`。新增长期依赖同步改 `environment.yml`。Windows PowerShell 读 UTF-8 文本必须显式 `-Encoding UTF8`。
