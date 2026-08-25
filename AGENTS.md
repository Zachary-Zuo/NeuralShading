# NeuralShading 项目约定

## 根本目标

接入多种保持原生语义的源材质族，用各材质族自己的 reference 产生 GT，再把它们编译成统一、随机访问、运行成本有界的 neural material program，并验证它在未见材质和参数状态上的质量—时间—内存 Pareto。目标运行时以小型 MLP 直接实现 `evaluate(wo, wi)`；`prepare()` 负责 latent 获取、过滤和对同一着色点可复用的 view-conditioned 编码；需要材质驱动方向采样的路径再提供与 evaluator 匹配的 `sample()/pdf()`。解析 closure 是对照、物理 core 或 sampling proposal，不是目标表示必须归约到的输出词汇。完整定义见 `docs/realtime_material_compilation.md`。

源材质的原生参数、图结构和资源是 GT 的一部分。除非源材质本身就是层模型，否则不得要求它提供层参数，也不得为了迁就当前 IR 或 neural material backend，先把它反演或改写成层模型后再称为 GT。如果源材质原生可调，项目必须保留这些参数的编辑能力并由对应 reference 正确呈现。完整边界见 `docs/material_scope.md`。

当前已经实现的第一种源材质族是多层界面与均匀 slab，近期工作仍围绕它的研究主线展开：

1. 可信的多层随机游走 reference；
2. 该 reference 生成的方向响应数据；
3. 按 `docs/research/experiment_framework.md` 的采样密度表生成 v1 语料，冻结按源表示类型定义的泛化考核（G1/G2/G2s 与工作流稳健性 W）和四层指标体系；
4. 在稳定评测框架内按容量档位（S/M/L）比较 `docs/research/model_candidates.md` 中的候选；每个结论要求 matched 对照与 bootstrap 置信区间，结果登记在 `docs/research/experiment_log.md`；
5. evaluator 与 compiler 主线稳定后，再扩展 matched sampler、环境积分和 UE 式实时工作流；MethodBundle/Slang/viewer 是每个研究阶段收尾执行一次的部署轨道。

当前已有一个端到端可部署方法用于验证 compiler、MethodBundle、Slang backend 和 viewer 生命周期。接下来的方法研究以 neural evaluator 的建模为中心：候选在统一评测框架内按容量档位比较，同时测量局部散射质量与实际单次查询成本；现有解析方法作为回归与成本对照，不决定目标 neural representation。

数据、学习、方法包和 Windows viewer 的基础闭环已经迁到正式架构。当前研究采用基准优先顺序：先冻结 v1 语料与评测协议，再在稳定框架内迭代候选；运行时合同（`evaluate()` 输出线性 `f`、固定读取数、`prepare` 复用）保留为候选注册时的静态约束，MethodBundle/Slang parity 与 viewer 证据由每个研究阶段收尾执行一次的部署轨道提供。不得在 evaluator 尚未成形时把多灯 scaling、PT 方差或 UE 集成写成当前可执行的 kill test。这个顺序不把 `LayerStackIR` 提升为所有源材质的 GT 表示，也不把某个 backend 的状态布局提升为公共接口。

## 文档与表述

- 本项目维护的 Markdown、实验报告和用户可见说明统一使用中文叙述。
- 统一中文指正文的叙述和逻辑，不要求逐字翻译技术术语。文件名、代码标识符、命令、数学符号，以及 `tile`、`median`、`p90`、`packet`、`closure` 等更便于准确交流的术语可以保留；首次出现不直观的术语时，用中文说明它是什么、为什么存在。
- 不为追求字面上的“全中文”制造生硬译名。判断标准是读者能否迅速明白“它是什么、为什么需要、结论是什么”。
- 写作顺序优先为“它是什么 → 为什么需要 → 当前结论 → 下一步做什么”。避免只堆缩写和指标。
- 叙述以目标方法、执行路线和可验证判据为中心。历史 backend 名称与旧指标只在复现实验、兼容说明或具体命令中出现，不用反复否定旧方法来定义新方向。
- 物理或工程限制必须说明适用范围，不能写得像普遍定律。例如 v0 为了让 RGB 三通道共用一次自由飞行采样，在有体散射时令三个通道的总消光系数相同；颜色差异仍可由各通道散射反照率表达。这是 v0 的实现约束，不是一般介质都满足的物理性质。
- 新代码、稳定文档和用户入口统一使用 `reference`（对某个源材质族具有权威语义的求值实现）、`direct fit`（逐样本直接拟合）和 backend-specific `ScatteringState`。随机游走是当前层栈材质族的 reference，不是该术语的唯一实现。迁移前报告若从 Git 历史或外部归档中恢复，其原始字段可以保留旧名称，但必须标明它不代表当前接口；这不构成把报告重新持久化到根仓库的理由。
- 有限数据、模型和训练预算下的实验不称为“上界”。新报告使用 `optimized-code control`、`high-capacity teacher`、`best observed candidate` 等限定表述；历史 schema/manifest 字段为复现可以保留。实时 backend 的单次执行、状态和访存仍必须静态有界，这是工程合同，不是模型质量宣称。

## 根 Git 仓库边界

- 根仓库包含项目自有源码、测试、环境声明、中文稳定文档、版本化验收门槛、资产清单，以及 `references/` 中的 reference registry/package 说明。
- 根仓库不包含 `external/`、`assets/`、`data/`、`build/`、`artifacts/`、`reports/` 和缓存。单次正确性验证、实验报告与运行摘要统一进入 `artifacts/`；第三方 reference 源码固定在 `external/`；原始源材质、纹理、测量表和 viewer 运行资产固定在 `assets/`；`data/` 只保存 `data/reference-responses/` 下由 reference 导出的 HDF5；它们都由 `references/` 中的 package/manifest 追溯。
- 完整规则见 `docs/repository_policy.md`。
- `external/Falcor`、`external/pbrt-v4`、`external/OpenPBR`、`external/openpbr-bsdf`、`external/glm` 和 `external/MaterialX` 是固定提交的独立克隆。当前均为干净工作树，本项目没有修改上游源码。MaterialX viewer 所需的 NanoGUI 及其依赖使用上游 gitlink 固定提交，由获取脚本初始化。
- 若以后确实需要修改上游，先把改动保存为根仓库中的显式补丁和应用脚本，并更新本文件；不得把未说明的修改留在 `external/`。

### Falcor viewer 构建 overlay

- `apps/viewer/` 是根仓库自有源码，通过 `patches/falcor-viewer-overlay.patch` 临时加入 Falcor 的 Samples CMake 树。
- 只使用 `scripts/build_viewer.ps1` 配置和构建 viewer。脚本在应用补丁前验证锁定提交与干净工作树，并在 `finally` 中反向应用补丁；构建结束后 `external/Falcor` 必须重新保持干净。
- overlay 只增加一个 `add_subdirectory()`，不改变 Falcor/Slang 运行逻辑。项目 shader、MethodBundle loader 和应用源码都保存在根仓库。

## 统一 Python 环境

- 项目唯一 Conda 环境名：`neural-shading`。
- 环境声明文件：根目录 `environment.yml`。
- PyTorch 固定为 `2.11.0` 的 CUDA 12.8 wheel，声明在 `requirements-torch-cu128.txt`；它与开发机 CUDA 12.8 toolkit 对齐。
- 所有 Python、pytest 和 pip 命令必须在该环境中执行。非交互命令优先使用：

  ```powershell
  conda run -n neural-shading python <args>
  conda run -n neural-shading python -m pytest <args>
  conda run -n neural-shading python -m pip <args>
  ```

- 不使用 `base`、其他已有 Conda 环境或系统 Python 执行本项目代码。
- 需要导入 Falcor Python 模块时，统一通过 `scripts/run_falcor_python.ps1` 启动；该脚本设置锁定构建的 `PATH`/`PYTHONPATH` 后仍使用 `neural-shading` 环境。
- 新增长期依赖时同步更新 `environment.yml`，确保环境可复现。
- 首次创建环境：

  ```powershell
  conda env create -f environment.yml
  conda run -n neural-shading python -m pip install -r requirements-torch-cu128.txt
  ```

## 锁定的上游源码

- Falcor：`external/Falcor`，tag `8.0`，commit `9dc819c162b2070335c65060436041690b7937f8`。
- Falcor 依赖清单锁定的 Slang：`2024.1.34`。
- pbrt-v4：`external/pbrt-v4`，commit `5f7a606806a4ac7b939131ded9d7a30ebd02416e`。
- ASWF OpenPBR：`external/OpenPBR`，tag `v1.1.1`，commit `f8d6d947dfae4c9b599965a86c22826ea7a8dbfb`。
- Adobe openpbr-bsdf：`external/openpbr-bsdf`，commit `9edf806740d2140846d9bef76e4342fc458e2ef5`。
- GLM：`external/glm`，tag `1.0.1`，commit `0af55ccecd98d4e5a8d1fad7de25ba429d60e863`。
- MaterialX：`external/MaterialX`，tag `v1.39.4`，commit `270b5cf2ae2be24a3b6ef4b0569f1c93038dda1d`；其 NanoGUI gitlink 为 `6452dd6944d2ba5c0c9bc0042a1894f703ce1ace`。

## PowerShell 中文文本编码

- 在 Windows PowerShell 中读取预期为 UTF-8 的文本文件，尤其是包含中文的 Markdown、JSON、配置和源码时，必须显式指定 UTF-8 解码：`Get-Content -LiteralPath <path> -Encoding UTF8`；一次性读取全文时再加 `-Raw`。
- 不因 `[Console]::OutputEncoding` 或 `$OutputEncoding` 已设为 UTF-8 而省略 `Get-Content` 的 `-Encoding UTF8`。
- 如果显式使用 UTF-8 后仍出现乱码，先确认文件实际编码；只有确认是旧式 GBK/ANSI 后才使用 `-Encoding Default`。不得把乱码当作原文继续分析或回写。

## Trellis 工作流

- 本项目由 Trellis 管理（见文末托管块）。项目级工程规则入口是 `.trellis/spec/project/index.md`；按功能块分为 `core/`、`data/`、`learning/`、`viewer/` 四层，各层 `index.md` 含开发前检查清单与质量检查。写代码前先读项目级规则，再读对应层。
- 每个会话在做任何验证、测试、构建或训练之前，先按 `.trellis/spec/project/dev-environment.md` 判定开发机状态（完整 / 仅 GPU / 静态），并在第一次涉及验证的回复里写明状态与证据；静态状态下把待运行命令写进 `TESTING.md`，不宣称"已验证"。
- 提出或实现任何新候选方法之前先过 `.trellis/spec/project/method-constraints.md`：可以超软线验证表达力，但不允许没有硬件部署可能的形态。
- 语言：`.trellis/spec/`、`.trellis/tasks/`（`prd.md`、`design.md`、`implement.md`、`research/`）与 `.trellis/workspace/` journal 统一以中文为主体；文件名、标识符、命令、数学符号与常用术语保留英文，规则与本文件「文档与表述」一致。Trellis 自带的 hook 脚本、`workflow.md` 状态机文本与 skill 正文保持原文，不翻译。
- 临时诊断脚本放当前任务目录 `.trellis/tasks/<task>/scratch/`，不进 `src/`、`scripts/`、`tools/`。

<!-- TRELLIS:START -->
# Trellis Instructions

These instructions are for AI assistants working in this project.

This project is managed by Trellis. The working knowledge you need lives under `.trellis/`:

- `.trellis/workflow.md` — development phases, when to create tasks, skill routing
- `.trellis/spec/` — package- and layer-scoped coding guidelines (read before writing code in a given layer)
- `.trellis/workspace/` — per-developer journals and session traces
- `.trellis/tasks/` — active and archived tasks (PRDs, research, jsonl context)

If a Trellis command is available on your platform (e.g. `/trellis:finish-work`, `/trellis:continue`), prefer it over manual steps. Not every platform exposes every command.

If you're using Codex or another agent-capable tool, additional project-scoped helpers may live in:
- `.agents/skills/` — reusable Trellis skills
- `.codex/agents/` — optional custom subagents

Managed by Trellis. Edits outside this block are preserved; edits inside may be overwritten by a future `trellis update`.

<!-- TRELLIS:END -->
