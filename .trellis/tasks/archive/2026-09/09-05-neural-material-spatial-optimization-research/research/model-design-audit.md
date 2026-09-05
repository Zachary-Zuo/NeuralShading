# 模型设计审计：确定缺陷、语义缺口与研究取舍

## 1. 审计结论与证据边界

除了 [空间编码与采样问题](evidence-and-diagnosis.md)，还应处理优化对照的初始化、Slang 数值与 validity 传播，以及局部 frame 的人为不连续。reverse PDF 存在 view-conditioned state 的语义缺口；Beckmann core 的近似则必须先明确模型职责，不能一概当成实现错误。

用户追加要求是“顺便检查当前 neural 模型的设计有没有其它的问题，如果明确确实有错也应该一并处理”。以下保留最初静态审计的触发条件和证据；架构提交后，用户要求将任务细化为具体代码规划，修复选择已落实到 design §8。**以下没有任何一项被登记为代码已修复**。

代码证据固定在 `cc4d76bf4df089b725ad91b2a2673ca177edff86`，未特别注明时行号均指该快照。已静态检查 compiler、asset/prepare/evaluator、proposal、cooked runtime、Slang 与相关测试；未执行项目 forward、pytest 或 GPU shader。本文的 `tanh` 与 float32 例子是独立算术复核，不是当前编译器生成代码的运行结果。

最初核对时，架构会话已将 Python 方法移动到 `learning/methods/metal/`：asset/compiler/evaluator 的相关数学实现保留，sampler 内容相同。架构提交 `ea2d743` 后又在 HEAD `e3f1c21` 复核了这些路径，C1–C5 仍需处理；Slang budgeted 目录相对原快照没有差异。当前文件锚点、补充的 C6 未见资产编译缺口及具体 API 变化见 [提交后代码核对](post-architecture-code-plan.md)。

| 快照中的文件 | 交付时的新位置 |
|---|---|
| `models/metal_budgeted_asset.py` | [asset.py](../../../../../../src/ncls/learning/methods/metal/asset.py:188) |
| `models/metal_budgeted_compiler.py` | [compiler.py](../../../../../../src/ncls/learning/methods/metal/compiler.py:226) |
| `models/metal_budgeted_evaluator.py` | [evaluator.py](../../../../../../src/ncls/learning/methods/metal/evaluator.py:25) |
| `models/metal_budgeted_sampler.py` | [sampler.py](../../../../../../src/ncls/learning/methods/metal/sampler.py:232) |
| `mdl_metal_assets.py` | [native_assets.py](../../../../../../src/ncls/learning/methods/metal/native_assets.py:591) |

旧 `source_adapters.py` 中的 Metal 逻辑已移入 [data.py](../../../../../../src/ncls/learning/methods/metal/data.py:477)。读取时的 hybrid recipe 仍为 `footprint_samples: 1` 配合 0/1/4 texel 档；路径迁移本身没有解决输入/target 尺度差异。

| 项目 | 证据强度与判断 | 修复/决策依赖 | 对旧结论的影响范围 |
|---|---|---|---|
| C1 optimized program state 初始化改变值 | 确定的参数化错误 | 使用该 control 前修复 | 影响 compiler-gap 对照的起点；未找到证据证明旧 fixed-batch probe 用过它 |
| C2 Slang softplus 小值精度丢失 | 确定的不稳定算式；目标 GPU 的实际误差待测 | 部署 parity 前修复 | 不解释 Python 训练中的 residual trace，也未证明是视觉主因 |
| C3 invalid/非有限结果被包装成有效零值 | 确定的 Python/Slang 行为差异及错误传播缺陷 | 使用部署质量/采样结果前修复 | 可能掩盖数值失败；没有证据声称现存截图实际触发它 |
| C4 frame 在 `abs(n.z)=0.999` 处跳变 | 确定的坐标不连续；最终质量影响待测 | 新的 frame/方向拟合对照前修复 | 各向异性 core 与 half/difference 特征均受影响 |
| C5 reverse PDF 沿用正向 prepared state | 静态计算对象明确；与通常反向条件密度不一致，公共语义需落实 | 新 sampler 合同验证前解决 | 不据此声称只使用 forward PDF 的现有 PT 已有偏 |

## 2. C1：direct optimization control 应从相同状态起步

`src/ncls/learning/models/metal_budgeted_compiler.py:226` 的 `MetalBudgetedOptimizedProgramStateControl` 把已经解码的 `initial.compiler_condition`、`initial.spatial_scale_bias` 直接复制为参数，却在 `forward():268` 再做 `tanh`。因此零次优化后也不是原状态，例如 `0.8 → tanh(0.8) ≈ 0.66403677`。

这不是对照所需的 refinement：它在更新之前先改变初值和可达域。原 compiler 的 condition 在 `:192` 还叠加 `0.05 * proposal_condition`，某些维度可超过 1，所以直接 `atanh(clamp(initial))` 也会悄悄丢掉合法初始状态。

修复要求是明确 control 的可优化域，并令第 0 步与 initial 等价。可采用以 initial 为基点的零初始化、有限范围 delta，或与原 decoder 完全一致的 raw 参数化；域、边界及与 proposal 的耦合必须写清，不能为求逆而静默截断。确定性 access/frame/resource 仍不可优化。

后续 witness 检查全部可优化状态及同查询的输出在第 0 步一致，再验证允许字段能变化、确定性字段保持不变。已有 `tests/unit/test_metal_budgeted_model.py:254` 只覆盖后者，没有覆盖初始化等价。这个 control 与本研究 D1 的自由 asset latent control 是不同对象，不混用名称或实验结论。

## 3. C2：softplus 需保留 float32 可以表达的小响应

`shaders/ncls/backends/metal_budgeted/metal_budgeted_common.slang:41` 使用 `x > 20 ? x : log(1 + exp(x))`；Python 使用 `F.softplus`。按每一步 float32 舍入，`x=-20` 时 `exp(x)≈2.06115369e-9` 仍可表示，但加 1 后舍入为 1，最终得到零；稳定计算约为 `2.06115362e-9`。这里是加法消减精度，不是指数下溢。

该函数同时用于 positive RGB 和 prepare 的 lobe weight modulation。修复应使用适合目标 Slang 的稳定分支，例如在负尾部使用有误差依据的 `exp(x)` 近似，在其余区间使用稳定的 `log1p` 等价实现；不是将整个 softplus 换成 `exp`。分支阈值由 dtype 和误差推导确定。

后续验收以独立高精度 softplus 为 oracle，覆盖负尾部、零附近和正尾部，同时检查非负、单调及训练/部署误差。要报告小值的绝对误差和相对误差，不能依靠一个较大的全局 `atol` 掩盖全部尾部信号。本轮没有测量 shipped checkpoint 的 logit 分布，不推断它实际有多少查询受影响。

## 4. C3：零输出不能抹去 invalid 的来源

`metal_budgeted_evaluator.slang:139` 在 hybrid 结果非有限时返回 `float3(0)`。外层 `metal_budgeted.slang:31` 再以 `prepared.valid && isfinite(result.f)` 判定有效，于是非有限失败可能成为 `valid=1, f=0`；`sample():49` 也可能得到有效的零 weight。

同一 helper 在反射半球外或 half-vector 无效时直接返回零，外层同样没有接收失败标记。Python 的 `MetalBudgetedEvaluator.forward():455` 则保留 hemisphere、half-vector 与 finite 的联合 validity，清零数值后仍返回 `valid=false`。因此不能只比较两端的 RGB。

修复方案是让内部 evaluate 返回 `f` 与 validity/失败信息，由公共 `evaluate` 和 `sample` 传播；无效事件可以使用零数值占位，但不能重新标为有效。避免把 NaN/Inf 留给普通 renderer 累加，也避免 silent clamp。具体错误统计沿用新架构已有机制，不另造日常用户检查流程。依据是 [共享 Slang 合同](../../../../../spec/core/shared-slang-backend.md:44) 的非有限量错误规则。

后续 witness 包括正常的真实零响应、半球外查询、退化 half-vector 以及受控的非有限 evaluator 结果，分别核对 Python/Slang 的 `f`、validity 和 sample 行为。失败注入只能进入诊断，不通过修改正常 GT 或线上权重来完成。发现实际训练或部署数值溢出时还要定位其产生处；传播修复本身不会治好溢出。

## 5. C4：frame 的分支制造可避免的不连续

Python `_orthonormal_frame():25` 和 Slang `nclsMetalBudgetedOrthonormalFrame():84` 根据 `abs(n.z)<0.999` 在 Y/X helper 之间切换，再计算 `tangent=normalize(cross(helper,n))`。

取 `n=normalize((s,0,1))`、固定 rotation angle 为 0。阈值位于 `s≈0.04475493`，即法线倾角约 `2.56256°`：一侧 tangent 是 `(0,-1,0)`，另一侧是 `(n.z,0,-n.x)`，两者相差 90°。输入法线可以只变化极小量，坐标轴却交换了方向。正交性检查和 Python/Slang parity 都会通过，因两边实现了相同的不连续。

当 `alpha_x != alpha_y` 时，它会改变 analytic lobe 的方向；同一 helper 还在 evaluator 的 half/difference 特征中使用。MLP 可能学到部分补偿，但现有机制迫使它额外拟合一条人为接缝，不能指望 spatial loss 自动解决。

修复在当前允许的正 Z 域采用连续、数值稳定的 frame 构造，并明确 tangent 与 rotation angle 的约定。当前局部 normal 来自有界 slope 和固定 `z=1`，无需在这里承担覆盖整个球面的坐标图问题。half-vector 路径也须单独处理其有效域，不能假设两者的输入范围完全相同。

后续 witness 同时检查单位长度/正交/手性、跨旧阈值的小步连续性、固定角度下的各向异性主轴和各向同性旋转不变性。还需先通过 D0a 的原生 normal/frame 对齐，避免“消除接缝”却引入新的 authored angle 偏置。新表示 fresh 训练，不宣称旧 checkpoint 可无损套用。

## 6. C5：reverse PDF 应说明使用哪个 view 的条件状态

`MetalBudgetedPrepare.forward():139` 把 `wo` 送入 semantic decoder；最终 frame、roughness 和 proposal weights 都能依赖 `wo`。但 `metal_budgeted_sampler.py:243` 复制正向 state 和 frames，仅交换方向；Slang `metal_budgeted_sampler.slang:91` 做同一件事。

若写 `s(v)=prepare(asset,program,v)`，则目前返回的 reverse 是 `q(wo | wi; s(wo))`。从反向事件重新运行同一材质 sampler 的密度一般是 `q(wo | wi; s(wi))`，二者没有结构性相等保证。仓库中的 NVIDIA backend 在 `nvidia_neural_appearance.slang:108` 明确以 `wi` 重做 prepare；OpenPBR reference 也采用反向 prepare。

目前固定 proposal state 的 sample→pdf 测试只证明同一 frozen proposal 自洽，不能覆盖这项条件依赖。修复前需在新公共合同中确认 reverse 的定义，再用“交换方向并独立 prepare”的 witness 检查；不能把两条都沿用正向 state 的路径相等当作充分证据。

若公共 reverse 指反向事件的真实采样密度，有两条有界方案：让 sampler 的参数来自 view-independent 状态，或用反向 view 重新计算所需 proposal。前者不要求 evaluator 放弃 view conditioning；后者必须登记额外 prepare 成本，以及复用 latent 所需的状态 bytes。旧设计要求独立 `pdf` 不再运行 asset decoder，因此不能默默选择第二条并继续声称原预算。实施时单独审阅 sampler 方案，不借机扩大方向 evaluator。

这项缺口应在 sampler 的正式合同验证前关闭；它不阻塞只比较 forward `f` 的 D1/D2。还需检查真实消费者是否使用 reverse density，不能由此直接推断现存单向 PT 已产生偏差。

## 7. 应明确的模型取舍，不当作已定位根因

### 7.1 analytic core 是近似基函数

`metal_budgeted_evaluator.py:349` 的 Beckmann 分支返回 `D/(4 cos_o cos_i)`，没有 GGX 分支中的 masking-shadowing `G`。因此它不是 Smith-Beckmann BRDF。microfacet 中 `D` 与遮蔽项的职责及 Beckmann 的 `Lambda/G` 可由 [pbrt 官方教材](https://pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models) 核对。

旧 synthesis 使用了“Beckmann-compatible basis”措辞，而旧 design 称支持 Beckmann core。作为 learned basis，省略 `G` 可以是明确的近似；若作为物理 core 对照，则应选定具体模型并补齐、验证。先做有限方向网格的独立公式比对，确认语义与掠射误差，不直接改 GT。

当前两 lobe 都使用 Schlick RGB Fresnel；secondary 始终走 GGX，没有旧设计描述的 diffuse 类型分支。不能把它宣称为完整原生 conductor Fresnel 或已经实现的 diffuse contamination core。下一轮先如实报告近似，不自动增加 lobe 数或扩展模型族。

### 7.2 view-conditioned core 不自动具有物理参数含义

normal/frame、roughness 和 lobe weights 经过 view-conditioned decoder，不能直接当作重建出的原生属性。即便固定参数的 analytic 公式有互易性，也不能据此保证 `a(wo,wi; s(wo))` 与交换方向后的 `a(wi,wo; s(wi))` 相等。positive 输出、有界 correction 也不自动保证能量约束。

对具有相应原生语义的 source，可把交换方向与积分响应作为质量观测；auxiliary semantic target 若要求 view independence，必须约束真正对应的子表示。不能用假设的层参数监督任意 MDL 图，也不把这些物理质量观测临时升级为新 hard gate。

### 7.3 160 B 是 packed storage，不能当寄存器占用

`metal_budgeted_common.slang:129` 的展开 prepared state 包含 semantic24、view8、compactFrame8、两组三维 frame、lobes16、proposal12、access4 和 flags。按字段标量数量约 380 B，尚未计 alignment、外围 context/material 或 MLP 临时数组；`PackedState.words[40]` 才是 160 B。

这不意味着 GPU 必须常驻 380 B，也不是实际寄存器分配测量；编译器可消除或重用字段。后续分别报告 packed bytes、展开逻辑状态、编译资源与实测 query 时间。已有 11,392 dense MAC 计算仍成立，不包含全部解析和访存成本。

## 8. 已排除的误报与实施优先级

- cooked runtime 在 `metal_budgeted_runtime.py:251` 的入口确实执行 UV scale/rotation/translation；不能只读内层 sampler 就声称部署忘了坐标变换。其 LOD 选择与训练尺度近似仍属于 D0c 的核对范围。
- v3 已将完整 24D semantic state 送入方向网络；不能继续用旧 v1 的“只消费前八维”解释当前问题。
- `frame_state` 存的是 round-corner/object-scaled-bump 等控制，不是 anisotropy angle。字段未直接进入 evaluator 不足以证明可编辑参数失效，须追踪 typed token 与 renderer 提供的最终 frame。
- `evaluate()` 没有填充 PDF 不单独列为 bug：公共接口另外提供 `pdf()`，需由具体消费者合同判断是否要求 evaluate 同时返回它。

优先修复 D0 的 source/query/read 差异和 C4，随后才比较新的空间表示。C1 在使用对应 control 前完成；C2/C3 在部署结论前完成；C5 在 sampler 验证前完成。每一项修复都要有独立 witness，单纯 loss 降低或 Python/Slang 相互吻合不能替代数学与语义检查。

完整依赖与执行边界见 [design.md](../design.md) 和 [implement.md](../implement.md)。
