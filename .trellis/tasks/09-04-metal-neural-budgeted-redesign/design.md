# 设计：Metal Budgeted Semantic Hybrid

## 1. 方法身份与迁移边界

新 canonical 方法使用：

- public key：`metal`（公共入口不增加版本化名称）；
- method key：`metal-budgeted-neural-material`；
- correspondence：`metal-budgeted-semantic-hybrid@1`；
- 主 profile：`metal_budgeted_hybrid_v1`；
- matched direct control：`metal_budgeted_direct_control_v1`；
- layout：`ncls.metal-budgeted-layout@1`。

`metal_fused_full_v1` 不改 shape、layout 或 checkpoint 语义。旧 v4 继续只读，当前旧实现只在 task-local historical benchmark 和已有 package/capture 证据中使用；新 `metal` plugin 不接受旧 checkpoint resume，也不提供 converter、shape fallback 或同名 profile 替换。

主 profile 的两个 hard budget 已由用户确认：

| 项 | hard bound | 验证 |
|---|---:|---|
| dense `evaluate` MAC / direction | ≤20,000 | layout 生成器按全部部署 linear 层求和，CI 静态检查 |
| prepared state | ≤192 B | Python/Slang generated layout、host stride 与 package manifest 四方一致 |

analytic scalar/transcendental、`prepare`、reads、weights、asset 与 latency 全部另报，不能因为不在 dense MAC 内而省略。

## 2. 组件与文件布局

### 2.1 Python model

新增而不在旧类上堆条件分支：

```text
src/ncls/learning/models/metal_budgeted_profile.py
src/ncls/learning/models/metal_budgeted_asset.py
src/ncls/learning/models/metal_budgeted_compiler.py
src/ncls/learning/models/metal_budgeted_evaluator.py
src/ncls/learning/models/metal_budgeted.py
src/ncls/learning/methods/metal_budgeted.py
src/ncls/learning/metal_budgeted_runtime.py
src/ncls/learning/metal_budgeted_asset_cook.py
src/ncls/learning/abi/metal_budgeted_layout_v1.json
```

`src/ncls/learning/methods/registry.py` 的 product module 从 `metal_fused` 切到 `metal_budgeted`，public key 仍为 `metal`。source/reference、公共 producer、engine、checkpoint、package writer 和 viewer 不增加 Metal 分支。

### 2.2 Slang/package

```text
shaders/ncls/backends/metal_budgeted/
  metal_budgeted.slang
  metal_budgeted_layout.generated.slangh
  metal_budgeted_asset.slangh
  metal_budgeted_compiler.slangh
  metal_budgeted_evaluator.slangh
  metal_budgeted_sampler.slangh
```

package 继续使用 `ScatteringPackage@2` 的 program/asset/instance 三段：

- program：shared compiler、semantic decoder、evaluator weights和module；
- asset：Detail/Context mip hierarchies、scale/bias、variant table和provenance；
- instance：typed raw state、compact ProgramState、access/frame/variant选择。

viewer 只按 package module/resource descriptor 绑定，不识别 `metal_budgeted` 字符串。

## 3. 静态 profile

### 3.1 Asset layout

每个 deployable asset variant 固定：

- `detail`: `RGBA8_SNORM`，finest level 对齐该 variant 的 canonical source extent；
- `context`: `RGBA8_SNORM`，每个 level 的线性分辨率为 detail 的四分之一、最低 1×1；
- 两者都保存独立 response mip，而不是从 finest latent 下采样；
- per-level/per-channel FP16 scale 与 bias；
- fractional LOD 使用一个由 scattering context 提供或确定性派生的随机值选择相邻整数 level；选中 level 后各执行一次 bilinear sample。

最多九个 source slot 只进入 asset compiler。部署时始终两次 texture sample，不按 slot 循环。若 `pit_texture_selection` 等离散资源不能合并，asset cook 为合法枚举值生成多个 variant；ProgramState 只选择一个，不能混合未定义资源语义。

### 3.2 ProgramState

ProgramState 是 instance-time compiler 输出，不直接等同 source closure：

```text
condition[8]              FP16
primary_lobe[8]           FP16
secondary_lobe[8]         FP16
spatial_scale_bias[8]     FP16
proposal_prior[3]         FP16
variant/type/flags        uint32
```

精确 stride 由 layout 生成器决定并计入 `B_instance`。UV/access 和 renderer final frame 仍使用 canonical typed/runtime 字段执行；learned compiler 不覆盖这些确定性结果。

### 3.3 PreparedState

目标 stride 为 160 B，绝不超过 192 B。建议字段：

```text
filtered_latent[8]        FP16
local_condition[8]        FP16
frame0_tangent_normal[6]  FP16
frame1_tangent_normal[6]  FP16
projected_wo[6]           FP16
primary_lobe[8]           FP16
secondary_lobe[8]         FP16
proposal_weight[3]        FP16
validity/type/flags       uint32
reserved/alignment
```

bitangent 在 evaluate/sample/pdf 中由相同 helper 重建。normalize、frame degeneracy、PDF normalization 与 analytic sensitive math 提升为 FP32。invalid state 返回 finite zero `f` / zero PDF / invalid sample，不能用 fallback material 冒充。

## 4. Asset compiler 与 semantic decoder

### 4.1 Training-only asset compiler

asset route 仍从 `NativeAssetCollection@1` 按 tile/halo 取得 source-native role 数据。encoder 采用浅层 role projection + masked sum + depthwise/separable spatial trunk，不保留 per-slot runtime decoder、bundle attention 或 64/128/192/256 U-Net 作为部署组件。

asset compile 有三条独立 identity：

1. `encoder-only`；
2. `encoder-bounded-refinement`；
3. `direct-optimized-control`。

三条共用同一 Detail/Context shape、quantization和runtime semantic decoder。refinement 只优化 asset latent，冻结 shared runtime；direct control 不能伪装成 pure compiler 工作流。

### 4.2 Runtime semantic decoder

输入为：

```text
detail4 + context4 + ProgramState.condition8 + footprint/level4 = 24D
```

网络固定 `24→32→32→24`，约 2,560 dense MAC。输出 24D response-ready state：

- primary normal/tangent residual；
- primary RGB modulation、`alpha_x/alpha_y`、weight；
- secondary RGB、roughness、mixture/type clue；
- 8D local correction condition。

source semantic auxiliary target只监督能够明确映射的 normal、roughness、mask、color/optical clue；未知 packed role 不强行解释成 PBR channel。所有受监督输出都必须进入最终 PreparedState/evaluator，不保留训练可见、runtime 不消费的旁路 head。

同一 asset latent 在同一 finish/recipe 的多个 typed states 间共享；训练 batch 中参数 state 变化只通过 ProgramState 进入 decoder。这是 texture/parameter 解耦的结构约束。

## 5. Responsibility-aware typed compiler

输入仍保留 32 个 typed token、presence、graph/schema/recipe/metal/finish 和 16D canonical optical。token width 固定 16：

```text
E_semantic + E_type + E_responsibility + E_discrete
  + Linear4x16(encoded_value)
→ masked mean
→ add graph/schema/recipe/metal/finish embeddings
→ concat canonical optical projection
→ compact MLP
→ ProgramState
```

不使用 self-attention。离散参数使用枚举/布尔 embedding，连续/color/float2 保持 source range normalization 和 presence；不存在的值不以零参与聚合。

responsibility routing：

- access 参数：canonical accessor 直接执行；
- frame/round-corner 参数：renderer/reference final frame 直接执行；
- resource selector：选择合法 asset variant；
- appearance 参数：进入 set compiler；
- capability 不支持或 variant 不兼容：compile fail closed。

compiler 是 pure feed-forward path。训练中的 optimized state control 直接优化相同 ProgramState 字段，但使用独立 manifest role；G2/G2s 和 editability 只引用 pure compiler 结果。

## 6. Evaluator

### 6.1 Direction features

固定输入 28D：

- 两个 frame 下的 prepared `wo` 与 query `wi`：12D；
- stable half/difference 与 cosine/validity：8D；
- local condition：8D。

不读取 angular texture，不保存 64D view token。half-vector 退化、grazing 和 hemisphere 规则由 Python/Slang 共用 exact-vector tests 锁定。

### 6.2 Shared trunk 与输出

主 body：`28→64→64→64`，activation 初始使用 shader-friendly HardGELU；输出 6D。总 dense MAC：

```text
28×64 + 64×64 + 64×64 + 64×6 = 10,368
```

hybrid profile：

```text
positive = softplus_or_exp(raw[0:3])
gate = sigmoid(raw[3:6])
f = positive + gate * (primary_core + secondary_core)
```

primary core 支持 anisotropic conductor GGX，以及用 type enum 选择的 Beckmann 例外；secondary core 是 optional dielectric specular 或 diffuse contamination。固定最多两 lobe，不按 graph 动态增加循环。

direct control 使用相同 6D head、state和body；最终 `f` 只由 direct positive RGB 产生，剩余通道承担预登记的 core-calibration auxiliary，保证训练图和MAC闭合。它具有独立 profile/config identity，不与 hybrid checkpoint 混用。

analytic scalar操作不计入上述 dense MAC，但静态 op 分类和GPU timing必须报告。若最终实现需要增加任何 linear、lookup 或 per-direction state，layout 生成器重新计算；不得靠注释维持预算。

## 7. Sampler

PreparedState 中固定三个 component：primary analytic、secondary analytic、uniform/cosine hemisphere fallback。fallback 有正权重下限；sample selector重用一个 `float2`，PDF累加实际折叠映射的所有 preimage。

proposal weights可由 semantic decoder 的三个输出训练，但 proposal objective 通过 functional detached state只拥有相应参数；appearance 不被 proposal NLL 拖动。若首版选择完全由 analytic state确定权重，proposal loss仍记录为 audit metric而不参与总梯度。

`sample()` 最多调用 evaluator 一次生成真实 `f` 和 `f·cos/pdf`；独立 `pdf()` 不运行 asset decoder、compiler或evaluator。

## 8. Loss、metric 与 tqdm

### 8.1 标准 metric keys

method objective 输出以下通用标量名：

```text
loss/optimization_total
loss/appearance
loss/proposal
loss/proposal_weight
appearance/log_rgb
appearance/linear_rgb
appearance/chroma
appearance/peak_rgb
appearance/spatial_gradient
appearance/core
appearance/semantic_runtime
```

`TrainingEngine` 不按 method 名分支，只在存在时选择 `loss/appearance`、`loss/proposal` 和 weight 加入 tqdm postfix；`loss` 字段继续保存反向使用的 optimization total。负 proposal NLL 原值保留，不做绝对值或假归一化。

### 8.2 Metric 定义

- chroma：对 `log1p(f / scale_rgb)` 减去通道均值，在 target energy 超过 train-only epsilon 的样本上计算；
- peak RGB：每个通道分别构造 top-energy support，不以单一 luminance 排序；
- spatial gradient：同一 source/state/direction 的 paired UV response 差；
- frequency score：只在固定空间 slice/图像 artifact 上做 band-energy/gradient spectrum 比较，不把单次 observed 值设 hard gate；
- validation 保留 experiment framework 的 normalized L1、energy、peak、reciprocity 与 source-state bootstrap。

所有 scale/epsilon/tolerance 在看 formal test 结果前由 source-train、dtype 或独立 calibration 冻结并带 identity。

## 9. 单材质 probe 与选择

固定 source 使用现有 Tungsten Brushed Medium Light Brushing exact locator。新增 versioned diagnostic recipe，在线生成：

- coherent spatial tile + matched neighbor UV；
- uniform、near-reflection、grazing、cosine direction quotas；
- fixed viewer camera/light crop与数值 direction slice；
- train与validation独立 seed/query identity。

先跑 eager short overfit，再对同 checkpoint做量化 Python和Slang/package probe。比较 old historical、direct control和hybrid；结果分类/选择规则完全沿用 `research/model-redesign-synthesis.md` §5.3。pilot只决定结构选择，不升级为 formal quality 声明。

## 10. Matched runtime harness

task-local `scratch/benchmark_scattering_runtime.py` 只编排公共 program/package ABI，不增加 product CLI。统一 compute workload预生成 surface、wo、wi和prepared buffer，测：

```text
prepare-only
evaluate-only
prepare+evaluate×{1,4,8}
sample-only
pdf-only
```

controls：optimized MDL source program、NVIDIA faithful package、旧 full package、新 budgeted package。coherent workload固定单 source/state；divergent workload只在同一 execution group 内切换合法 state。每项 warm-up 后使用 GPU timestamp/同步批次，报告 median/p90/throughput和 bootstrap interval；CPU compile/load时间另列。

结果写 `artifacts/09-04-metal-neural-budgeted-redesign/runtime/`，稳定结论写 task `research/`。Windows只运行最多1024 state、2次warm-up、3次measurement的接口 smoke；完整规模在新package完成后于Linux/headless一次性执行，不作为模型开发前置。viewer benchmark使用既有 `scripts/benchmark_viewer.ps1`，只作全链路补充。

## 11. 训练 lifecycle

正式 phase 名称避免假 coarse-to-fine：

1. `joint-response-fit`：从 step 1 激活 asset、pure compiler、evaluator和必要 proposal组；
2. `deployment-qat-refine`：INT8 latent + FP16 runtime weight STE，继续完整 appearance路径。

可选 curriculum 只能显式声明：direction mollification角度、spatial response mip范围、peak quota与过渡步数。若没有这些变化，不使用 coarse-to-fine 名称。phase-local DDP parameter graph必须稳定，`find_unused_parameters=False`。

旧 long checkpoint不resume。新方法先完成 single-material pilot，再做代表性标准 finish + complex recipe 的 bounded cohort smoke；692-source formal long不在本任务内自动启动。

## 12. 错误与兼容矩阵

| 条件 | 行为 |
|---|---|
| profile MAC或state超 hard bound | 构造/生成检查失败；只能改成 diagnostic identity并回planning |
| source slots无法合成且无合法variant contract | asset cook fail closed，不退回逐slot runtime |
| typed resource selector越域 | compiler拒绝，不取模或选默认纹理 |
| semantic supervised字段未被runtime消费 | component conformance失败 |
| direct/hybrid profile与checkpoint不匹配 | restore/export拒绝 |
| 旧 full checkpoint请求新train resume | v1 identity检查拒绝；v4仍只读evaluation |
| sample与independent PDF不一致 | sampler测试失败，不用clamp掩盖 |
| total loss负但appearance有限 |正常记录分项；不把符号当complex/nonfinite错误 |
| eager正确、quantized或Slang偏色 | 分类为quantization/parity defect，禁止改reference或扩大模型 |

## 13. 验证设计

- unit：layout identity/MAC/state、typed responsibility、asset variant、two-read accounting、metric公式、progress postfix、checkpoint/profile拒绝、sample/pdf数学；
- GPU：semantic decoder/evaluator梯度、single-material eager overfit、FP16/INT8、Python↔Slang exact/random probes、online two-phase stop/resume；
- integration：公共 `ncls train/validate/export`、Package@2、typed edit、asset swap；
- runtime：四种 control 的 matched harness；
- viewer：reference/neural linear EXR、microdetail crop、色度/峰值差、deferred/PT和Falcor clean；
- Linux：单/双GPU新 profile smoke与DDP checkpoint/resume；不自动执行formal long。

## 14. 稳定文档与研究登记

实现后更新：

- `.trellis/spec/learning/online-training.md`：新 Metal method/profile、route与QAT合同；
- `docs/research/model_candidates.md`：登记 budgeted semantic-hybrid 和 direct control；
- `docs/research/experiment_log.md`：只登记冻结、matched 的实际结果；
- `docs/learning.md` 与 `docs/metal_linux_training.md`：新 canonical入口和旧 checkpoint只读边界；
- task `research/`：probe、runtime、selection、deployment evidence。

## 15. 五卡交替训练与持续监督增补

### 15.1 执行身份

本轮 Linux 执行使用物理 GPU `5,6,7,8,9`，统一通过：

```text
bash scripts/run_falcor_python.sh --gpus 5,6,7,8,9 -- \
  -m ncls train <run.yaml> --devices 5,6,7,8,9 ...
```

每个 rank 内 Torch/SlangPy 只见 `cuda:0`，Falcor 使用对应物理 adapter。任何单独设置 `NCLS_FALCOR_GPU_INDEX`、绕过 launcher 的 `torchrun` 或多作业共享 GPU 5–9 都不是本轮证据。

冻结 recipe 的 `batch_size=64` 解释为 per-rank batch，DDP5 global batch 为 320。schedule 仍按 optimizer step 计数，source/query stream按 `(world_size, rank)` 分区。resolved plan、checkpoint 与报告必须记录五卡 topology；本轮 direct/hybrid 是新的 matched DDP5 pair，不与原单卡协议按 step 混为一组。

### 15.2 交替状态机

两个配置共享里程碑 `0 → 8 → 128 → 256 → 512 → 1024 → 2048`：

1. hybrid 到下一里程碑并写可恢复 checkpoint；
2. direct 到同一里程碑并写可恢复 checkpoint；
3. 仅在两侧都到达共同里程碑后生成 paired summary；
4. 数学/接口正确且没有训练阻断时继续下一段。

step 0 负责 calibration/checkpoint，step 8 是 DDP objective、rank-local reference、metric reduce、checkpoint commit 与 teardown 的短 smoke；step 128 以后沿冻结 validation cadence。phase boundary 1792 前后必须各有一次 stop/resume 证据。

### 15.3 低 token 监督器

监督器只负责 orchestration，不复制训练逻辑：

- child process 的完整 stdout/stderr 逐配置、逐 segment 写入 `artifacts/metal-budgeted-pilot/ddp5/logs/`；
- 主监督流只在 process exit、checkpoint commit、共同里程碑、错误签名、metric 非有限/突变、GPU 长时间空闲或每 15 分钟 heartbeat 时输出一条紧凑事件；
- 通过子进程 wait/exit 获取完成事件，metric 文件用增量 offset 读取，不反复把历史日志送入对话；
- heartbeat 只记录进度、最近 metric、GPU 5–9 利用率/显存与日志 offset，不打印逐 step tqdm；
- 监督器本身、状态和临时解析脚本位于当前 task `scratch/`；训练产物、监督 journal 与完整日志位于 ignored `artifacts/`。

stall 判定必须同时参考最近 metric/checkpoint 时间、进程存活、GPU 活动和日志 stage。reference 编译或 rank-0 checkpoint I/O 的长尾不能仅因 GPU 暂时空闲而判死；确认 collective desync、worker exit 或超过已观测 stage 上界后才终止整个 process group，并从最后共同有效 checkpoint 恢复。

### 15.4 修复与比较完整性

- DDP graph/reducer 问题优先修 phase ownership、条件分支或跨 rank descriptor；保持 `find_unused_parameters=False`、`static_graph=True`。
- rank-local reference、metric schema、checkpoint commit或teardown错误按公共合同修复；不得用增大 timeout 代替根因修复。
- 只改变 supervisor、错误报告或不参与 identity 的 instrumentation 时，可从 exact checkpoint 继续。
- 改变 model/data/query/loss/optimizer/schedule/precision、calibration或 resolved identity 时，两侧都从相同边界重跑；旧产物保留并标为 superseded，不覆盖。
- 效果差但实现正确时按冻结 failure classification 收尾，不自动扩宽网络、增加 seed或超过 2048-step cap。

### 15.5 两模型 Windows 交付

结构选择仍按 §9 和冻结 paired 指标决定 canonical profile，但用户需要直接查看两种结果。因此 hybrid/direct 各自保留 exact checkpoint，并允许在结构选择完成后编译独立的 evaluator-only diagnostic package/catalog：

- capability 只包含已通过 parity 的 `prepare/evaluate/anisotropic-frame`；未实现或未验证的 `sample/pdf` 不得声明；
- catalog/UI/capture 必须标记 `exact-diagnostic-evaluator-preview` 与 profile identity；
- 两个 package 不共享 checkpoint identity，不用 shape compatibility 互换；
- Linux 完成 checkpoint、量化 Python、Slang/package validation；Windows 只需按交接命令做 Release viewer 加载与视觉检查，Linux 不宣称完成 D3D12 证据。
