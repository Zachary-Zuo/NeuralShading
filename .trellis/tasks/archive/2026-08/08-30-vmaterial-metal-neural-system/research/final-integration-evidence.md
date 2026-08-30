# vMaterial Metal neural system 最终集成证据

## 完成边界

父任务的六个子任务均已完成、创建 scoped local commit 并归档。本次交付完成的是统一架构、原生 Metal reference、质量优先 full method、matched sampler、三部分 runtime/package/viewer，以及可在 Linux 单 GPU 上启动长训的同构配置与操作入口。Windows 已证明完整方法能正确进行 GPU-resident online gradient descent、resume 和部署；没有把 16-step correctness run 写成最终质量结论，也没有把 Linux static handoff 冒充目标机已执行结果。

## 六个子任务与提交

| 顺序 | 子任务 | 实现提交 | 归档提交 | 核心交付 |
| ---: | --- | --- | --- | --- |
| 1 | canonical architecture | `0cfeb88` | `4876fcb` | 唯一 canonical contracts、全仓递归迁移、旧 reader/alias/fallback 删除 |
| 2 | reference foundation | `6c20af9` | `8f3ab61` | 692 opaque registry、178 groups、52 assets、typed states 与 grouped online query |
| 3 | fused full evaluator | `867c67b` | `9a7e9cd` | shared codec、typed compiler、direction bank、6+4+tail hybrid evaluator |
| 4 | matched sampler | `e8deaaa` | `709a888` | 10 lobes + full-support fallback、matched `sample/pdf` 与 proposal phase |
| 5 | runtime deployment | `b1008b6` | `3edadb3` | Package@2 program/asset/instance、Slang、viewer bundle replacement/typed edit |
| 6 | Linux training handoff | `bccf2d8` | `174f1f6` | 四阶段 QAT correctness、profile、Linux smoke/long config 与 review handoff |

每个归档任务都保留各自的 `research/` 证据和可复跑命令；父任务不复制运行 artifact 到 Git。

## 需求闭环

### 原生语义与三部分组合

tracked registry identity 为 `fa6642e60d469231839756d749283b3d7d93e7163284c4094837770379dec8cc`：837 authored exports 被严格分成 692 opaque 与 145 cutout-rejected；opaque cohort 对应 178 个 execution groups、52 个 texture sets 和 64 个 schema table entries。GT 直接来自原生 MDL program、typed arguments、resources 与 graph identity，没有先反演成 LayerStack/Principled。

部署合同把 `MetalOpticalIdentity × FinishNeuralTextureBundle × NativeTypedParameterState` 映射为独立 `program/asset/instance`。bundle replacement 校验 recipe/schema/channel compatibility 并只更新 asset/instance；typed edit 通过 pure compiler 重建 bounded instance state，不重训 evaluator，也不重编码未变化 bundle。特殊 recipe 只在 registry 声明的 native family-local 域内成立。

### 完整方法，而非部分实现

最终 `metal_fused_full_v1` descriptor 登记并执行 20 个 required components 与 13 个 parameter groups。它包括 role-specific codec stems/heads、shared encoder/decoder、per-mip high/low grids、semantic/structured heads、bounded asset adapter、typed token compiler、spatial program、raw/half-difference/learned-frame angular features、6 个 analytic core、multiplicative correction、4 个 positive residual lobes、free positive RGB tail，以及 10-lobe + full-support fallback proposal。construction、phase graph、checkpoint 与 artifact compiler 对缺失/disabled/placeholder 分支 fail closed。

sampler 对每个 local distribution 显式计算 folded hemisphere 的两个 preimage，11 种 mixture 的确定性积分误差界为 0.3%；`sample()` 最多调用一次 directional evaluator，`pdf()` 不重复 texture decoder 或 typed compiler。Python/Slang 对 256 个 probes 的 direction/component/forward-reverse PDF parity 覆盖全部 11 个 components。

### Online optimization 与训练效率合同

Windows 最终 full-shape run 使用机械选择的 3 个 activation exports 和资产 `[6, 50, 22]` 覆盖全部 required components，同时独立对 692/178/52/64 全 cohort 做 preflight。训练运行真实 `codec-warmup → joint-appearance → proposal-fit → qat-refine` optimizer steps；13/13 groups 均有 finite、非零 gradient 和 update，并从 step 13 checkpoint 恢复到 step 16。QAT 使用 FP32 master weights、FP16 STE runtime rounding、INT8 grid STE 与 FP32 sensitive accumulation，不是 phase alias。

固定 query stream 每次重新执行 authoritative reference，不持久化 response；四阶段 initial/final loss delta 分别为 -0.12928905、-0.69964365、-0.42879959、-0.13873275。该结果只证明完整梯度路径可下降。最终 checkpoint SHA-256 为 `7d6d31a297787639987dec60220f582a0a69d0ae7717d5ad89284ef8bcb56f6d`。

runner 在普通 step 不读取全部 GPU scalars；audit/log/validation cadence 才同步。最终 profile 的 reference submit、forward、backward、optimizer、validation median 分别为 0.291815 s、0.133723 s、0.147494 s、0.015490 s、0.376084 s，peak allocation 为 403,271,168 bytes。它们是热点证据，不是验收 hard gate。

### Runtime 与真实成本基线

full profile 静态上限为 106 次 texture/angular random reads，PreparedState 为 2,816 bytes；`prepare()` 约 2,416,000 MAC，已 prepare 后每次 `evaluate()` 约 185,088 MAC。质量优先 package 的逻辑存储基线为 `B_shared=1,781,932` bytes、`B_asset=439,710,464` bytes、`B_instance=2,880` bytes。

Release viewer 的 full-shape PT/deferred capture 已产生 finite linear EXR，但当前 160×240 panel 的 GPU 时间约 64.8 s / 37.2 s。该观察明确说明当前实现不是最终实时 Pareto 点；任务价值在于先完整建立可训练、可部署和可消融的质量基线，后续是否优化 codec/grid/结构必须等 Linux 长训质量经用户审阅后再规划。

### Linux 单 GPU handoff

Linux smoke 与 long config 都显式包含 692 exports 和 52 assets，semantic fingerprint 为 `59c84c80ecf19a2ade3642b869e2c9767f925b1f95c37d88c819bc98afbbdf7b`。long budget 为 120,000 steps：20k codec、70k joint、15k proposal、15k QAT。最终 handoff manifest identity 为 `b880e18bf4252806edb80c96506fdeae00063d537f971fb6c59d560fa316f0c5`，绑定实现提交 `bccf2d8`、registry/config/toolchain 与 Windows diagnostic checkpoint。

handoff 状态保持 `pending-on-target-host`，只允许单进程、一个 `CUDA_VISIBLE_DEVICES` 条目。目标机必须先完成 source/hash/config preflight 与 Linux-native smoke，才开始 long run；结束后仅生成 `ncls.training-review@1`，其中 `automatic_followups=[]`、`next_action=user-review-required`。formal、额外 seed、消融、蒸馏、compact 与 Pareto 没有被自动排队。

## 最终质量证据

- 最新完整 unit：186 passed；
- 最新 Metal evaluator/runtime GPU subset：9 passed、35 deselected；各前置子任务还分别保存 full GPU/integration/parity 证据；
- `ScatteringPackage@2` validator、sample/PDF invariants、typed edit 与 bundle replacement：通过；
- Release viewer build 与 full-shape capture：通过；
- Linux shell syntax、Python compileall、config semantic regeneration：通过；
- `external/Falcor`：clean；
- 六个 child：全部 `completed` 并归档。

由此，父任务的完成边界已经达到。Linux 长训、最终材质质量、六类泛化 formal matrix、matched ablation 与产品 Pareto 仍是用户审阅长训效果后的新决策，不构成本父任务未完成项。
