# 补齐 vMaterials preset catalog 与 capability audit

## Goal

把 vMaterials 2.4.0 的首批 neural cohort 从研究 shortlist 推进到可执行、可追溯的 MDL source catalog：正式登记 11 个原生 module 的全部 172 个 authored exports，计算每个入口的传递资源闭包与 runtime capability audit，使后续 reference 查询、参数化 neural 训练和未见 preset 评测能够消费同一份权威清单。

## Background

- 完整 source root 位于 `assets/source-materials/mdl-vmaterials2/2.4.0/Materials/`，11 个入选 module 都保持在 `vMaterials_2/` 的原始相对路径下。
- `references/mdl-vmaterials2-v1/assets.json` 当前只登记 6 个 executable snapshots；`docs/research/mdl_vmaterials_neural_cohort.md` 已经冻结 11 个 module、172 个 authored exports。
- 当前 bridge 已能从一次 class compilation 提取 exact export、typed 参数、传递 module imports、target-code 纹理、argument block、RO data 与 DF handle，但生成器只覆盖 6 个主入口。
- `Suede_Leather.mdl:1220,1254` 的 punched 分支输出 `geometry.cutout_opacity`；当前 MaterialProgram、reference query、训练 batch 与 viewer 没有 opacity 输出。

## Requirements

- R1：建立 machine-readable family/preset catalog，恰好覆盖首批 11 个 module 的 172 个 authored exports，并记录 family、原始 module、exact export signature 与评测角色。
- R2：通过锁定 MDL SDK 的 module API 枚举 exports，不用正则表达式充当 MDL parser；每个 authored export 独立 class-compile，不能用共享主材质的一次 audit 代替。
- R3：每个 preset 保存默认 typed 参数、MDL language、canonical source snapshot identity、传递 source modules、实际资源闭包和 runtime capability audit。
- R4：资源闭包覆盖 module/import、2D/3D texture、SDK BSDF data，以及 capability 审计所涉及的资源；路径保持 pack-relative且不得逃逸 source root。
- R5：`assets.json` 扩展为 11 个 family 主入口，并保持现有 6 个入口的 module、export、默认参数、source/resource hashes 与既有 audit 字段稳定；172 个 presets 放入独立 family catalog，不隐式扩大 viewer/parity 白名单。
- R6：catalog 显式记录 family role、resource signature、compiled-material/slot identity 与 runtime support，支持区分连续 authored 状态、离散资源状态和 capability 子族，而不是仅凭名称或 thumbnail 分类。
- R7：172 个 exports 全部登记；8 个 punched suede 标为离散 cutout 子族和 `runtime_supported=false`。正式 MDL runtime 遇到非恒等于 1 的 cutout 必须 fail closed，不得当作不透明 BSDF 执行。
- R7a：bridge 与通用 typed texture binder 必须保真支持 cohort 实际使用的 MDL decoded pixel types，包括 `Rgba_16`；不得因某材质同时含 unsupported closure 输出而跳过、量化或丢弃其 16-bit 输入资源。资源格式能力与 cutout 输出能力必须独立审计。
- R8：生成过程可恢复、显示真实 preset 进度，并在所有 entry 与完整性检查成功后原子更新 tracked manifests；失败必须区分 unsupported capability 与 implementation defect。
- R9：增加自动检查，覆盖 catalog 数量/唯一性、SDK discovery 对应、资源存在与 containment、snapshot 稳定性、capability 状态和 punched fail-closed；同步更新中文稳定文档。

## Acceptance Criteria

- [ ] AC1：正式 family catalog 恰好包含 11 个 families、172 个唯一 `module + export` entries，逐 family 数量为 `27/31/31/11/15/9/6/7/8/16/11`。（R1、R2）
- [ ] AC2：每个 entry 都有 SDK 产生的 exact signature、默认 typed 参数、snapshot id、source modules、resource closure、resource signature、compiled identity 和 capability audit。（R2、R3、R6）
- [ ] AC3：所有闭包文件存在且 hash 一致，所有路径为 source root 内的 pack-relative path；闭包包含 root module 和 target/capability 实际使用的资源。（R4）
- [ ] AC4：`assets.json` 包含 11 个主入口；原 6 个入口的 source identity 与既有 audit 内容无漂移，新增 5 个入口可由正式 provider 定位。（R5）
- [ ] AC5：164 个 opaque entries 为 `runtime_supported=true`；8 个 punched suede entries 为离散 cutout 子族、`runtime_supported=false`，且 source loader 可 catalog 它们、正式 runtime 编译明确拒绝它们。（R7）
- [ ] AC5a：`Rgba_16` atlas 以完整 4×Uint16 payload 进入 artifact，并可由通用 Python/Falcor 与 Windows viewer texture binder 创建 `RGBA16Unorm` 资源；punched 的 unsupported reason 只包含 `geometry.cutout_opacity`。（R7a）
- [ ] AC6：中断后可从已验证的逐 preset artifact 继续，进度以 `completed/total`、elapsed、rate、ETA 展示；tracked manifest 只在全量检查成功后更新。（R8）
- [ ] AC7：自动测试能检测 preset 缺失/重复、SDK discovery 不一致、非法/缺失资源、hash 漂移、错误 capability 状态和 opacity 静默降级，并在允许的开发机状态下通过。（R9）
- [ ] AC8：`references/mdl-vmaterials2-v1/README.md`、`docs/research/mdl_vmaterials_neural_cohort.md` 与数据层 MDL spec 准确反映正式登记结果、生成方式、164/8 runtime 边界和后续 neural 消费方式。（R9）

## Out of Scope

- 不扩展公共 MaterialProgram、reference query、训练 batch 或 viewer 来输出/消费 opacity；punched suede 的 runtime/neural 支持留给独立后续任务。
- 不解压或接入 Base Materials、Automotive Materials。
- 不启动大规模 reference 数据采集、neural 训练、正式质量比较或 viewer 逐材质 capture。
- 不把入选 module 改写成 LayerStack、OpenPBR、MaterialX 或 distilled PBR 后继续沿用 MDL GT identity。
- 不删除完整 vMaterials source root、三个 ZIP 或任何 authored preset。
