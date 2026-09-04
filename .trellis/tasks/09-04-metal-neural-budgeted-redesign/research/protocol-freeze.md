# 协议冻结：Metal budgeted redesign

## 1. 冻结范围与证据边界

本文件在新 product plugin 切换前冻结旧对照、单材质 pilot 和 matched runtime 的输入身份。冻结日期为 2026-09-04，仓库基线提交为 `802ff2fdbb640eaff0a8aa4d3eb760a7e738731a`。

冻结时，任务影响范围内的 tracked 文件没有既有修改；仅当前任务目录是新建未跟踪内容。根工作区另有用户既有的 ignored artifact 与无关未跟踪论文/图片，本任务不修改、移动或删除它们。

本协议只授权 diagnostic calibration、short pilot、代表性 cohort smoke 和部署验证；不授权恢复旧 v4 训练、自动执行 692-source formal long、追加 seed，或按结果循环扩大模型。

## 2. 旧 Metal full 历史对照

### 2.1 Checkpoint 与 package

| 项 | 冻结值 |
|---|---|
| checkpoint | `artifacts/metal-linux-training/long/checkpoint.step00020000.pt` |
| SHA-256 | `b00ca19208677a6e9a54210b7a7a3567647cf1eb618f7809085a40015422ca2a` |
| format | legacy `TrainingCheckpoint` v4，只读 |
| method key | `metal-fused-neural-material` |
| descriptor SHA-256 | `4e390a4eb489d10bea687a819306549d5be63fd7528e2046416c9261971293bb` |
| step / phase | `20000` / `joint-coarse-to-fine` |
| source states | 2070 |
| diagnostic package | `artifacts/viewer/metal-step00020000-tungsten/packages/003698fccac260627379da2383403f54f9b239b09f6489a4b0267e1ce483feb8/` |
| package manifest SHA-256 | `6f3f5eb4dfef58f11b5c50433b768ddecefd7d02210e1f8548fd2ad8cd9ac474` |
| package ID | `8d9cd538469dd2307490329a96df2c01ec88c49c46d097104645a7405301b109` |
| program / asset / instance ID | `2b4158ed78cfcb57c290270b9548f674db7115322ab4737557384ff1b975857e` / `0cdab49f5ece5eaf567d9a2b35c3cf0280a16c33c75a8b6058d3c1a4f0acab77` / `16f75985473ca60ef890a3a5a63e57695883c58ca72c55f9d8623cc725693e71` |

低层 loader 已确认该 checkpoint 的 descriptor 与当前 product descriptor 不相等；它不能通过当前 evaluation snapshot readiness，也不能 resume。task-local runtime harness 只读取已经冻结的 package ABI，不依赖把旧 checkpoint 伪装成当前方法。

### 2.2 静态成本账本

以下数字属于 `metal_fused_full_v1`，沿用其生成 layout 的 dense MAC 口径；它们不是新候选预算，也不是硬件 latency：

| 项 | 冻结值 |
|---|---:|
| `prepare` dense MAC | 2,416,000 |
| `evaluate` dense MAC / direction | 185,088 |
| 首次 `prepare+evaluate` dense MAC | 2,601,088 |
| prepared state | 2,816 B |
| 固定随机读取 | 106 |
| `B_shared` | 1,781,932 B |
| Tungsten `B_asset` | 95,073,072 B |
| `B_instance` | 2,880 B |

## 3. Tungsten 单材质 probe

### 3.1 Source 与 authored state

| 项 | 冻结值 |
|---|---|
| registry | `references/mdl-vmaterials2-v1/metal-opaque-v1.json` |
| registry file SHA-256 | `c5cea76caa4e13224a9be85fcaa19d1eeabb0e9e7ae48106f4898705fb04117e` |
| registry identity | `fa6642e60d469231839756d749283b3d7d93e7163284c4094837770379dec8cc` |
| export ID | `003698fccac260627379da2383403f54f9b239b09f6489a4b0267e1ce483feb8` |
| module | `::vMaterials_2::Metal::Tungsten_Brushed` |
| exact export | `::vMaterials_2::Metal::Tungsten_Brushed::Tungsten_Brushed_Medium_Light_Brushing(float,float,float,float,float,bool,float,bool,float2,float,float2,int)` |
| source snapshot ID | `d3eb28ced6de47017737e1e194fb607ecbacae53ad8d3dfbaeed186dc0d7b504` |
| texture-set ID | `cc3a9ea0212a9f37c85451d9352f70ce2e4795273907ebbf5d48f259238048b4` |
| compiled MDL artifact manifest SHA-256 | `bd8f2b6c904d9602dac661c63e53089f4005a715805e169d454ca7afa775cee7` |

authored 参数 state 固定为 registry default：`texture_scale=[1,1]`、`texture_translate=[0,0]`、`texture_rotate=0`、`uv_space_index=0`、`scratch_reflection_variation=0.5590000153`、`brush_width=0.3610000014`、`scratch_variation_amount=1`、`brush_height_blur=0.2039999962`、`scratches_bump=0.3499999940`、`enable_round_corners=false`、`roundcorner_radius=0.0099999998`、`across_materials=false`。

### 3.2 Query recipe

- recipe identity 文本：`metal-budgeted-tungsten-paired-uv-pilot@1`；实现后由 canonical JSON 计算 SHA-256，不手写假 hash。
- train seed：`2026090401`；validation seed：`2026090402`；runtime workload seed：`2026090403`。validation stream 不复用 train cursor。
- UV anchor：`[0.371, 0.619]`。纹理原生尺寸为 `4096×4096`；paired UV 使用同一方向/state，并分别加 `[1/4096, 0]` 与 `[0, 1/4096]`，address mode 仍由 source descriptor 决定。
- footprint 三档各占相同 quota：零导数、一个 texel（`uvDx=[1/4096,0]`、`uvDy=[0,1/4096]`）和四个 texel。不得把 paired UV 距离折入 LOD 后丢掉空间差分监督。
- 方向 quota 固定等分为 `uniform-hemisphere`、`cosine-hemisphere`、`near-reflection`、`grazing` 四类。每个 optimizer step 的四类计数必须相同；validation 每类 4096 条，顺序和 RNG identity 固定。
- `near-reflection` 围绕每个 frame 的 mirror direction 取 0°–8°；`grazing` 固定 `|z|∈[0.02,0.15]`。退化方向保留为 invalid 统计，不用别的 proposal 补齐。
- train 只通过 GPU-resident online reference 产生 response；不保存 response batch。允许保存 source/query identity、checkpoint、metric 与可视化输出。

### 3.3 Pilot cap 与选择

- direct 与 hybrid 各使用一个 seed、最多 2048 optimizer steps；batch geometry、online query count、optimizer schedule、asset shape和 quantization recipe必须 matched。
- validation 固定在 step `0/128/256/512/1024/2048`；达到 cap 即停止，不依据曲线临时加步数。
- eager pilot 完成后才对同一 checkpoint 做 quantized Python 与 Slang/package probe。部署层失败不得反向改写 eager 质量结论。
- 选择严格使用 `model-redesign-synthesis.md` §5.3：hybrid 只有在 peak/chroma/spatial-detail 的 paired bootstrap interval 至少一项显示稳定净收益、其他项没有明确退化且未越 hard budget时保留；统计不分或 direct 更好时选择 direct。
- 若两者共同失败，先按 asset/query/loss/quantization 分类；不得自动启动 teacher。只有共同失败且 failure classification 完成后，才允许一个不超过主 profile neural MAC 4 倍的 diagnostic teacher，同样不得扩大 seed/step cap。

## 4. Metric calibration 与容差

### 4.1 Train-only scale

在 step 0、模型前向之前，用 train seed 的 16384 条 online reference query 计算逐通道 `scale_rgb`：每通道取有限正值的 P50，并限制到 `[2^-12, 2^8]`。`energy_epsilon = max(64 * eps_float32 * max(scale_rgb), 1e-6)`。该 calibration payload 以 source/query/seed/count/结果计算 identity，并随 checkpoint 保存；validation 只读取冻结结果，不重新估计。

`appearance/chroma` 在 target RGB 和大于 `energy_epsilon` 时生效；`appearance/peak_rgb` 对每个通道独立取 train calibration 的 P95 为 support threshold。spatial-gradient 使用上节固定 paired UV，不使用图像后处理梯度代替训练信号。

### 4.2 数值容差来源

| 检查 | 冻结容差 | 来源 |
|---|---:|---|
| eager Python 同 dtype 重复/向量化 parity | `atol=2e-6, rtol=2e-5` | float32 三层 MLP 与 analytic helper 的固定执行次序 |
| sample→independent PDF | `atol=2e-5, rtol=2e-4` | float32 PDF 累加及 frame 重建误差；归一化另做统计积分 |
| quantized Python ↔ Slang | `atol=5e-4, rtol=3e-2` | FP16 weight/state、RGBA8 SNORM latent 的部署舍入与 FP32 sensitive accumulation |
| state/blob stride | exact | generated layout/host reflection，不允许近似 |

若后续证明容差的数学前提错误，只能更换 implementation/test identity并在正式结果前重新冻结；不得看过正式候选误差后扩大容差。

全部 observed quality、time、memory、吞吐、peak、chroma 与 spatial-gradient 数值均为 report-only。唯一硬门仍是用户确认的 `evaluate ≤ 20,000 dense MAC/direction` 和 `PreparedState ≤ 192 B`，以及数学/接口正确性。

## 5. Viewer 与图像 probe

旧证据图固定为：

| 项 | 冻结值 |
|---|---|
| path | `artifacts/viewer/metal-step00020000-tungsten/viewer-window.png` |
| SHA-256 | `df2ae8da61d74a63254ddb8b6d79aae305a21158510b00286b9fe34f54bcfdf9` |
| extent | `1294×758` |
| catalog ID | `ab804aa0c18e76732e8af41b028132d1d6266b57619a16f6f1fac5159bac2cfc` |

旧图只用于定位已观察缺陷，不参与新候选数值评测。新 headless capture 必须输出无 UI 的 matched linear EXR；每个单 panel 使用同一 scene/camera/light。微细节 crop 在单 panel normalized 坐标固定为 `[x0=0.20,y0=0.24,x1=0.80,y1=0.66]`，高光局部 crop 固定为 `[0.36,0.36,0.72,0.58]`；crop identity 是 scene identity、view extent 与上述坐标的 canonical hash。

## 6. Matched runtime 输入

四个 control 固定为：

1. optimized MDL：上节 Tungsten source snapshot 与 compiled artifact；
2. NVIDIA historical faithful：`artifacts/nvidia-faithful/materialx-recorded-200k/package/`，manifest SHA-256 `0f7ac33ef89af1d377aa7d5b554c14424cd692e952a2948b49991c928e6f8746`，package ID `6950aeb2a8225e01bf2e6acf2e32346ac0bec48bb1f608cab3f6caa322d02073`，checkpoint SHA-256 `ee3e6fb3bf105008247348989857f81801a9be992a59f90b07cb81eca4fe12fe`；它是旧 `nvidia-rta2024-functional@1` 历史结果，不冒充当前 `@2`；
3. 旧 Metal full：§2 的冻结 package；
4. 新 budgeted package：Phase 4 产生并回填 exact identity。

正式 workload 固定为 65,536 个 surface/query state；coherent 全部使用一个 state，divergent 只在同一 execution group 内按 seed `2026090403` 打乱合法 state。precision、输入 buffer、warm-up 32 次、measurement 100 次和 batch-boundary GPU 同步一致。分别测 `prepare-only`、prepared `evaluate-only`、`prepare+evaluate×{1,4,8}`、`sample-only`、`pdf-only`；GPU median/p90/bootstrap interval与 CPU load/compile wall 分开报告。

完整 workload 只在新 budgeted package 完成后于 Linux/headless 一次性运行四控制，不作为模型实现或 single-material pilot 的前置条件。Windows 仅执行 `count≤1024`、`warmup≤2`、`measurements≤3` 的 correctness/preflight smoke，用于确认 ABI、资源绑定、prepared-state adapter和 profiler lane；该结果受 dispatch floor支配，不形成相对 latency结论。2026-09-05 曾误启动的 Windows 完整 workload已应用户要求立即终止，未完成结果作废且不登记为实验结果。

当前 NVIDIA artifact 与旧 Metal/Tungsten artifact 的 source family 不同，因此 runtime 比较只表达公共 ABI 下的实现成本对照，不表达相同材质的质量对照。任何不能保证 workload/sync/precision 一致的路径只保留静态账本，不输出相对 latency 结论。

## 7. 2026-09-05 DDP5 执行修订

- `trigger`：用户明确要求在当前 Linux 主机使用 GPU 5–9 做五卡 DDP，交替训练 hybrid/direct，持续监督约 12 小时，并交付两个可在 Windows 检视的模型。
- `invalidated evidence`：§3.3 中“原生 Linux 单 GPU、每 step global batch 64”不再描述本轮执行 topology；它仍是历史冻结协议，不能用来解释本轮 DDP5 的样本预算或吞吐。本轮开始前尚无 budgeted pilot observed result，因此没有正式质量证据被覆盖。
- `scope impact`：model、source、query、loss、optimizer、schedule、precision、seed 与 2048-step cap保持不变；per-rank batch 仍为64，但 global batch变为320，source/query stream按 world size 5分区。新增 DDP step-0/8 gate、共同里程碑交替、事件驱动监督，以及 hybrid/direct 两个 evaluator-only diagnostic Windows package交接。
- `rerun required`：hybrid/direct 都从 fresh DDP5 step 0开始并使用独立 artifact root，不从单卡 checkpoint resume。任何改变 resolved plan、method/data/query identity或训练数值语义的修复都要求两侧从相同有效边界重跑；只改变监督器或不参与 identity 的错误报告可继续 exact checkpoint。

本修订由用户直接授权，不改变 `evaluate ≤ 20,000 dense MAC/direction`、PreparedState `≤192 B`、单 seed、2048-step cap和 failure-classification 规则。DDP5 的 direct/hybrid可以互相做 matched comparison，但不能与原单卡结果按 step合并；约12小时只是一轮执行 timebox，不是 observed quality hard gate。

## 8. 2026-09-05 剩余时间探索授权

- `trigger`：用户明确允许在原有实验与结论提前完成时，用剩余时间做少量neural material结构探索，并要求先对有特点的材质专项分析，再考虑普适性。
- `invalidated evidence`：无；该轨道不改写Tungsten direct/hybrid选择规则，也不把专项结果并入已冻结paired统计。
- `scope impact`：只在主训练、比较和两个Windows diagnostic交付完成后，按机制预选最多3个exact locator；每材质单seed、最多256-step diagnostic。只有至少两个专项指向同一failure mechanism时，才允许一个单seed、最多512-step mixed cohort。
- `rerun required`：每个材质/混合cohort使用独立source、query、config和checkpoint identity，不能resume Tungsten checkpoint或互相覆盖。实现bug改变identity时只重跑受影响的专项；主paired结果除非共享实现语义被改变，否则不重跑。

探索结果只登记design direction和候选优先级，不增加本任务hard gate，也不授权第二seed、teacher、旧full、formal 692-source long或无界模型变体循环。

## 9. 2026-09-05 吞吐与 batch 修订

- `trigger`：用户观察到GPU远未满载，明确要求先优化通用训练架构并允许增大batch，再推进更多direct/hybrid实验。
- `invalidated evidence`：§7的per-rank batch 64和原`@1`执行调度不再是后续主pair的冻结recipe；它们已执行到共同step128并作为before-profile保留，不能resume到新recipe。
- `scope impact`：source、query、model shape、loss、optimizer、schedule、precision、seed和2048-step基础cap不变。validation改为窗口级一次packed reduce但保留全部逐batch记录；新`@2` recipe使用每16 step report、prefetch/reference batch steps 2，并在预登记64/128/256/512中选择per-rank 512、global batch 2560。每step样本预算扩大，因此必须同时报告累计work units，不能与`@1`按step合并。
- `rerun required`：hybrid/direct从新config identity fresh step0开始；两侧使用相同batch、topology和cadence。`@1` checkpoint只作性能/早期结构诊断，不参与`@2`最终选择。

选择依据是同source/query/model的64-step DDP5 profile：batch 64/128/256/512的global work units/s中位数约为2,936/5,350/10,465/23,354，而Torch peak显存约705/707/709/748 MiB/rank。512支配本轮候选，达到预登记上限后不再因尚有显存余量事后扩大profile cap。

## 10. 2026-09-05 完整语义输入修订

- `trigger`：高吞吐`@2` v1 hybrid/direct到共同step512后，appearance虽分别改善到约`1.152/2.220`，但spatial-gradient仍约`0.282/0.283`；实现审计发现24维semantic state只有前8维进入neural evaluator。
- `invalidated evidence`：原先“semantic decoder的24维response-ready输出均被最终evaluator消费”以及“8维local condition足够承载neural方向响应”的假设失效。v1的质量结果仍是有效诊断对照，但不能代表完整语义输入候选。
- `scope impact`：只把directional condition从8维扩为24维；evaluator由`28→64→64→64→6`变为`44→64→64→64→6`，dense MAC从10,368变为11,392。asset/compiler/decoder/PreparedState 160 B/两次读取/output head/loss/source/query/batch512/optimizer/schedule/precision/seed保持不变。
- `rerun required`：使用新的hybrid/direct v2 profile、method schema与`@3` recipe identity；两侧从fresh step0重跑，v1 checkpoint不得resume。v1 step512 artifacts保留且不覆盖；先在共同128/256/512检查spatial、peak和parameter-group梯度，再按既有规则继续。

该修订直接处理runtime未消费已生成状态的实现/架构缺口，不扩展hard budget、seed、材质范围或teacher额度。

## 11. 2026-09-05 Detail→frame semantic 短路径修订

- `trigger`：v2共同step512的matched结果未改善spatial；exact checkpoint的online诊断显示target one-texel log梯度约`0.285`，预测梯度仅约`0.0011/0.0038`，且原始patch差异经过Detail与semantic decoder连续衰减。十倍学习率的fixed-batch spatial-only对照仍不能充分拟合，排除“只需多训主recipe”的解释。
- `invalidated evidence`：完整24维semantic进入evaluator只修复消费合同，不足以证明Detail的高频信号能穿过semantic decoder到达frame/response；v2作为信息连通消融保留，不再作为当前主候选。
- `scope impact`：semantic decoder输出后，将Detail四通道逐项residual到前四个frame semantic分量；不硬编码source channel含义，不新增参数、state、read或evaluate MAC。prepare增加四次逐项加法；其他model/loss/data/optimization轴完全不变。
- `rerun required`：分配hybrid/direct v3 profile、method schema`@3`与pilot recipe`@4`，两侧fresh step0重跑；v2 checkpoint不得resume。v3完成共同里程碑后无论observed quality高低都先做failure classification，不自动继续创建v4。

## 12. 2026-09-05 wrap bilinear oracle 修订

- `trigger`：v3 fresh pair完成并生成双package后，最终package的真实Falcor parity在冻结的`uv=(0,0)` witness显著失败；同一Slang路径在纹理内部坐标的合成fixture通过。逐层对照证明Python `_sample_level()`先对UV取模，却仍给`grid_sample`使用zero padding，导致bilinear footprint跨0/1边界时三个邻居被错误置零；GPU wrap sampler会从纹理另一侧读取这些邻居。
- `invalidated evidence`：`artifacts/viewer/metal-budgeted-ddp5-2dc0965-step2048/`的manifest parity expected错误，该handoff不得交付。它的program/asset/blob hash和两个训练run的observed quality仍可作诊断，但不能证明当前Python oracle、Slang和Windows viewer闭合。只在内部坐标运行的旧合成GPU parity也不足以覆盖address mode。
- `scope impact`：只把部署侧CPU/Python mip采样改为显式bilinear，并按descriptor对四个邻居分别wrap或clamp；训练online reference、模型、loss、source/query、optimizer、batch512、schedule、precision和2048 cap均不变。GPU fixture固定改用边界UV，最终package继续使用同一边界witness与原冻结`atol/rtol`。
- `rerun required`：`metal_budgeted_runtime.py`属于方法implementation identity，所以hybrid/direct必须在新hash下从fresh step0交替重跑；旧checkpoint不能resume或重新标记。新pair仍以共同2048为上限，质量结论若复现明确分离则不延长到4096；只有新package自身通过真实Falcor parity后才发布Windows handoff。

## 13. 2026-09-05 role-separated Detail pilot

- `trigger`：四个特征材质的256-step结果把空间失败定位到划痕/裂纹材质；随后不更新权重的high-pass/Detail×8机制probe虽然放大预测梯度，却在划痕青铜和开裂钢上同时增大spatial error，排除“现有共享聚合只缺增益”的解释。用户已授权在主pair完成且仍有时间时继续小型结构探索。
- `scope`：新增独立`metal_budgeted_hybrid_role_detail_v4` pilot profile。唯一模型差异是Detail聚合：四个RGBA通道分别只聚合`color/normal/scalar/packed`对应slot，各角色内部独立softmax；Context继续使用原共享slot softmax。网络层形状、参数数、两次纹理读取、RGBA8、PreparedState 160 B、evaluate 11,392 MAC、hybrid输出、source/query/loss/optimizer/seed保持不变。
- `controls`：在划痕青铜与开裂涂漆钢上各fresh运行v3 shared聚合control与v4 role-separated candidate；物理GPU 5–9、per-rank batch2048、global batch10240、step256 cap与validation recipe matched。已有旧identity v3结果只作假设来源，不代替新implementation identity下的fresh control。
- `interpretation`：用共同step256的256条同序validation row做candidate-minus-control paired bootstrap，报告appearance/log/linear/chroma/peak/spatial及内部信号。observed quality不设hard gate，不自动生成v5；若空间改善伴随主要appearance/peak退化，则登记trade-off并回到多通道角色分配设计，而不是继续调gain。
- `deployment class`：pilot仍满足两次固定读取、160 B PreparedState和11,392 evaluate dense MAC硬合同；它不是新的Windows交付前置，当前可交付v3 pair保持不变。

这属于“跨层合同 + 测试覆盖缺口”：纹理内部随机值无法区分zero padding与wrap，合成测试还绕过了最终DDS只读buffer。防复发由三层共同承担：CPU边界unit、真实GPU边界fixture、最终package自加载parity。
