# 架构提交后的代码落点与设计决定

## 1. 基线与证据归属

2026-09-05，用户告知另一会话已全部提交，并要求据当前任务修改 PRD、design 等，规划具体代码修改。架构实现提交为 `ea2d743`，本次读取 HEAD 为 `e3f1c216401a1156c288a9e735a690bc446dca2b`。已跟踪工作树干净；工作区还有既有未跟踪资料，不属于本任务。

已读取归档架构任务的 [validation.md](../../09-05-architecture-reset-training-workflow/research/validation.md)、当前 project/learning/data/core spec 与根 [TESTING.md](../../../../../../TESTING.md)。新训练通过 `python -m ncls train`、`Method`、`PipelineOnlineDataSession` 和 `TrainingEngine`；新成果进入 `RunPaths` 管理的 `outputs/<config-stem>/<run-id>/`。不恢复旧 runner、checkpoint reader、跨机视觉队列或 full Metal 模型。

环境为完整 Windows：`nvidia-smi` 识别 RTX 4090，Conda 列表中存在 `neural-shading`，锁定 Falcor 的 `falcor_ext.cp310-win_amd64.pyd` 存在。本轮只做环境探针、源码检查及任务文档修改，没有执行项目 forward、测试、训练、构建或 viewer。归档任务的 314 unit / 19 GPU 等结果属于该任务，不能作为本次模型修改的验证。

旧实验与行号仍固定到 `cc4d76bf4df089b725ad91b2a2673ca177edff86`。以下表格使用当前 HEAD 的路径和行号；数学问题并未因目录迁移消失。

## 2. 当前接口与缺口

| 当前证据 | 确认的行为 | 代码计划 |
|---|---|---|
| [asset.py:188](../../../../../../src/ncls/learning/methods/metal/asset.py:188) | `_encode_source_patches` 先做手工 summary，再送小 MLP；并非 raw spatial encoder | 增加 `spatial_encoder.py`，以原始 mip0 tile 接入分组 stem、共享空间网络和 learned hierarchy |
| [native_assets.py:409](../../../../../../src/ncls/learning/methods/metal/native_assets.py:409)、[同文件:1013](../../../../../../src/ncls/learning/methods/metal/native_assets.py:1013) | 固定格式/transfer 解码；现有 tile 路径也会读 raw mip、单位化 normal；tile API 已存在 | 新 encoder 只请求 mip0；按角色做声明驱动的 decode。复用 collection/host/residency，不把 coarse raw mip 当唯一网络输入 |
| [data.py:413](../../../../../../src/ncls/learning/methods/metal/data.py:413) | 生成 query、旧 patch、粗略 LOD；paired 重新采另一份 patch | 只生成逐 query 定位与过滤字段，关联共享原始 tile；pair 复用同一 learned hierarchy 与 mip 随机数 |
| [source_adapters.py:45](../../../../../../src/ncls/learning/source_adapters.py:45)、[batches.py:61](../../../../../../src/ncls/learning/batches.py:61) | adapter 返回 tensor/provenance 二元组；conditioning 没有共享资源字段 | 增加明确的 adapter 返回类型与 conditioning 资源集合；Nvidia adapter 返回空集合 |
| [producer.py:815](../../../../../../src/ncls/learning/producer.py:815)、[同文件:885](../../../../../../src/ncls/learning/producer.py:885)、[同文件:934](../../../../../../src/ncls/learning/producer.py:934) | 所有 conditioning tensor 都按第 0 维拼接、筛选、截取 | 原始 tile 不能直接塞进现有 tensor 字典；增加通用资源关联与 select/concat，避免把共享 tile 当 batch row |
| [asset_cook.py:123](../../../../../../src/ncls/learning/methods/metal/asset_cook.py:123)、[runtime.py:180](../../../../../../src/ncls/learning/methods/metal/runtime.py:180) | cook 按每级 texel 中心重新取 source patch；部署在 SNORM texel 上做 bilinear | train/cook 共用 encoder hierarchy，train/cooked Python 共用 grid coordinate/read-plan；Slang 独立验证同一读法 |
| [metal_budgeted_asset.slang:62](../../../../../../shaders/ncls/backends/metal_budgeted/metal_budgeted_asset.slang:62) | shader 用完整变换后的 derivatives 和纹理尺寸算 rho；训练用 scale 近似 | 统一实际 Jacobian、rho、LOD；零 footprint 在任意 texture scale 下仍是 LOD0 |
| [evaluator.py:101](../../../../../../src/ncls/learning/methods/metal/evaluator.py:101) | 8D view 输入为 `wo, wo², wo.z, 1`，没有 footprint；0/1 texel 都在 mip0 | 在同一 8D 空间改为 `wo3 + bounded Jacobian4 + frac(LOD)`；所有新对照同步使用 |
| [reference_query.cs.slang:55](../../../../../../shaders/ncls/reference_query/reference_query.cs.slang:55)、[同文件:120](../../../../../../shaders/ncls/reference_query/reference_query.cs.slang:120) | `footprint_samples=1` 为中心；多样本先在空间求 reference，再平均线性 `f` | 复用已有空间积分，不另造 GT；D0 比较 1/16/64 空间样本，训练 point/filtered 配方显式分开 |
| [query.py:28](../../../../../../src/ncls/references/query.py:28)、[同文件:645](../../../../../../src/ncls/references/query.py:645) | `ScatteringQuery` 无 filter random 字段，上传 meta 固定为 0.5 | 增加可选 `filter_random[B]`，缺省维持 0.5；新 Metal 显式传入可复现的 query 随机数 |
| [method.py:78](../../../../../../src/ncls/learning/methods/metal/method.py:78)、[同文件:539](../../../../../../src/ncls/learning/methods/metal/method.py:539) | 参数组含 asset_variant；objective 分别走主/paired patch；QAT 主要包 runtime weight | 更新 descriptor、资源依赖、分组、checkpoint 和 objective；同一次 tile 编码供两次读取，加入 program/prepared pack 的 STE 路径 |
| [model.py:40](../../../../../../src/ncls/learning/methods/metal/model.py:40)、[method.py:847](../../../../../../src/ncls/learning/methods/metal/method.py:847) | 模型上下文固定资产数 52，部署要求 source 在 checkpoint 的 source 列表内 | C6：新 profile 删除资产数参数和学习表；编译可接受支持语义内的未训练 source snapshot |
| [method.py:338](../../../../../../src/ncls/learning/methods/metal/method.py:338)、[同文件:922](../../../../../../src/ncls/learning/methods/metal/method.py:922) | descriptor 声明 REVERSE_PDF，program payload 未声明该位 | 随 C5 对齐实际实现、descriptor 与导出 capabilities，不能只修数值函数 |

`ScatteringQuery` 的扩展是可选过滤条件，不改变 GT 家族接口，也不引入 method 分支。现有 footprint evaluator 只在**同一空间点的重复 evaluation samples**之间检查 PDF 一致；它并未要求不同 footprint 点的 PDF 相等。因此没有证据支持为开启空间平均而修改这项 shader 检查。

## 3. C6：encoder-only 的生命周期缺口

当前部署以“训练 source 是否出现在 checkpoint”为准入条件，与用户要求“固定共享 encoder/decoder，为新原始纹理直接编码”不相容。这是本任务 R8 下明确需要改的行为，不能只删除 `variant_scale_bias` 后宣称闭环完成。

新行为：`compile_asset(snapshot, checkpoint)` 先核对实际支持的 graph/schema/角色、通道、坐标域和资源，再用 snapshot 构造 adapter，前向编码；checkpoint 的训练 source 列表只用于训练追溯。资源 ID 继续定位资源，不索引 learned affine。`typed_compiler` 的 graph/schema/metal/finish 等原生类别条件不等于 asset ID 表，保留其已支持的原生语义范围；本任务不宣称任意未知 MDL 图无需新增 compiler 支持。

独立验收使用不在训练 source 列表内、但属于已支持 schema 的冻结 snapshot，以及同内容换资源定位的诊断资产。核对共享参数不变、没有 optimizer step、没有新增 learned state、cook/编译成功；另外报告 held-out response 质量。未知或不兼容语义仍按现有适配合同报错，不用放松检查来伪造泛化。

现有用户 `export --material-index` 仍按 checkpoint source 选择。未见资产的研究 witness 直接调用公开 Method compiler API，不为本轮新增另一套 export CLI；普适的新资产交互入口不列为本轮交付前置。

## 4. 固定的实现取舍

1. **共享原始 tile 进入通用 conditioning。** 原始 GPU 资源与 row tensor 分开；资源的 select/concat/release 通用实现，Metal 只解释自身 binding。避免 `B×9×2×4×P×P` 按 query 重复复制。该处是必要的公共修改，不是 Metal 专属训练调度器。
2. **先在原生分辨率学习，再对齐特征。** 五类输入 stem 分别处理 color、tangent normal、height/bump、scalar/data、packed；不在学习之前统一缩小图片。relative coordinate map 作用于 stem feature 的对齐；共享 UV 参数仍由 prepare 处理。
3. **首个形态固定。** stem 两层 3×3、宽 16；最多 9 slot 的角色感知拼接经 1×1 投影到 32，再两层 3×3；共享的 learned stride-2 block 生成 hierarchy。Detail 从 `H_l`、Context 从 `H_(l+2)` 生成四通道 latent。具体边界、tile halo 和 mip tail 见 design。
4. **过滤条件进入 decoder。** 仅修正 target 的 footprint average 不足以区分 mip0 下的 point/filtered query；在不扩 prepare 输入宽度的条件下传入完整 footprint Jacobian。它是 query 几何的固定映射，与禁止逐图数值归一化不冲突。
5. **C5 使用 view-independent proposal 参数。** 独立小 head 从 program8+latent8 生成 proposal frame/roughness/weights；反射轴仍按查询方向计算。增加 8 个 half 的 proposal frame 存储，计划 packed state 176 B；evaluate 保持 11,392 dense MAC、两次 texture read。新的 prepare dense MAC 为 3,024。额外 ALU、cook 时间和真实 GPU 成本另报。
6. **signed correction 先只训练 readout。** D1 冻结后，共享方向 trunk、core 和 gate 固定，单独训练最后 64→3 输出，避免“冻结 gate 参数”却通过共享 hidden layers 改变 gate。正值与 signed readout 做 matched 对照；不把受限读出的失败当整个 residual 族的表达力结论。

这些数值是本次待实施形态的结构选择和推算，不是实测结果。改变 source/query 或模型数学后必须 fresh run；旧 v3–v6 指标保留为历史证据，不拼入新 matched 表。

## 5. 风险与不扩张项

- 深层 latent 的 raw receptive field 随 mip 增长；不能以预平均 source mip 或跨 optimizer step 的 detached feature cache 偷换计算。首轮训练只覆盖冻结的 0/1/4 texel diagnostic footprint；全 mip cook 正确性与高 LOD 泛化分别报告。
- 原始 raw tile、learned activations 和 reference 的 GPU leases 共同占用显存。按 CPU 预先知道的 source/tile cohort 安排，不能每步把 CUDA asset ID 拷回 host；超配置资源预算报告 resource defect，不自动缩 raw 输入或继续加显存预算。
- 动态 slot presence 对 DDP 的未用参数处理须沿用共享 engine 的机制验证。Windows unit/mock 不证明 Linux/NCCL 实机正确。
- 当前 typed compiler 是已注册 Metal source 范围的适配；本轮不重建任意 MDL 图的普适 compiler。
- C1–C5 的历史触发条件、数学例子与影响范围保留于 [model-design-audit.md](model-design-audit.md)，实现选择以更新后的 [design.md](../design.md) 为准。
