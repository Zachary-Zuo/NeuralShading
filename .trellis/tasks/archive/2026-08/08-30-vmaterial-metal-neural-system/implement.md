# vMaterial Metal 神经材质系统执行计划

## 1. 执行方式与完成边界

parent拥有需求、canonical architecture、完整方法identity、child依赖和最终交付复核；代码由六个可独立验证的child完成。当前parent完成于“Windows已证明full method正确训练并可部署，Linux单GPU长训练版本已交付”，不等待Linux long run结束。

Linux long run完成后先由用户审阅效果。`08-30-metal-formal-evaluation`和`08-30-metal-compact-ablation`已从parent解除，不是自动后续，也不阻塞parent完成。只有新的用户批准才启动它们或新增训练预算。

所有迁移使用inline实现/检查。进入任一child前重新加载`trellis-before-dev`与目标层spec；不得因为parent批准而跳过child artifact、环境判定和质量门。

父任务与六个child使用同一continuous execution授权。每个child通过其完整质量门后，只提交该child明确拥有的源码、测试、spec和任务文档，立即归档，再启动下一个依赖child；不amend、不push、不纳入无法识别或无关dirty files。只有冻结范围发生实质变化、需要新外部权限或出现真实blocker时暂停。

## 2. 当前交付任务图

| 顺序 | Child task | 交付物 | 显式依赖 |
|---:|---|---|---|
| 1 | `08-30-metal-canonical-architecture` | grouped plan、multi-asset training、Method/Config/Checkpoint@4、Package@2及全仓递归迁移 | parent PRD/design |
| 2 | `08-30-metal-reference-foundation` | 692 opaque registry、MDL execution groups、typed-state pool、filtered footprint、52-asset collection | child 1 canonical contracts |
| 3 | `08-30-metal-fused-full-method` | shared codec + typed compiler + hybrid evaluator及component/gradient/artifact conformance | child 1；child 2 registry/query/assets |
| 4 | `08-30-metal-matched-sampler` | 10-lobe + fallback proposal、matched `sample/pdf`及proposal phase | child 3 evaluator correctness和proposal state；不依赖formal convergence |
| 5 | `08-30-metal-runtime-deployment` | Metal program/asset/instance compilers、Package@2、Slang/viewer bundle/edit/full path parity | child 3 + child 4；child 1 package/viewer contract |
| 6 | `08-30-metal-linux-training-handoff` | Windows full-profile四phase验证、短程收敛/profile、Linux单GPUsmoke/long config和handoff | child 1–5全部正确性证据 |

child 1是破坏式迁移：NVIDIA和全部现有调用方一起进入新合同并删除旧接口。child 3/4共同构成完整方法；任何required component未实现时不能跳到runtime或handoff。

## 3. 跨child质量门

### 3.1 Canonical architecture后

- [x] `ReferenceExecutionPlan@1`、`NativeAssetCollection@1`、typed batches、`MethodDescriptor@2`、`TrainingConfig/Checkpoint@4`、`ScatteringPackage@2`成为唯一正式接口；
- [x] 五个reference、NVIDIA、四个configs、CLI、tests/spec/docs/package/viewer递归迁移；
- [x] v3/v1 reader/schema/alias/converter/fallback和旧public symbols删除，旧format只在拒绝测试/历史说明出现；
- [x] Windows/Linux upper layer仍无platform/device/build-path分支；
- [x] 现有reference/NVIDIA/package/viewer判据不放宽。

### 3.2 Reference foundation后

- [x] registry与审计计数一致：692 opaque exports、178 groups、52 texture sets、64 schemas；145 cutout拒绝；
- [x] group-homogeneous session、argument/RO offsets、typed-state pool、global source index和lease/resume成立；
- [x] `NativeAssetCollection`覆盖role/schema/mip/tile+halo，cook与training使用同一identity；
- [x] authoritative footprint query和source-derived texture target没有host response batch；
- [x] 每个group/asset/schema具有representative Windows query/preflight证据。

### 3.3 Full evaluator后

- [x] full profile的所有R5 codec/compiler/direction/evaluator components存在、启用并进入phase graph；
- [x] parameter registry、component contracts和checkpoint tensors双向一致，无orphan/placeholder；
- [x] stratified activation中execution、gradient/update和Python artifact coverage闭合；
- [x] codec与joint-appearance短run finite，encoder/decoder确实联合反向；
- [x] typed edit不重训evaluator或重编码未变化asset。

### 3.4 Matched sampler后

- [x] proposal phase训练10 lobe slots + full-support fallback，不模仿source sampler；
- [x] sample→pdf、normalization、forward/reverse PDF、weight identity、support和finite成立；
- [x] sample至多一次directional evaluator，sample/pdf不重复texture decoder或typed compiler；
- [x] sampler component/parameter/artifact coverage闭合，不使用analytic-only fallback冒充full。

### 3.5 Runtime deployment后

- [x] `program/asset/instance` compilers与Package@2三层identity/resources一致；
- [x] Python FP32、BF16/QAT、Slang package和viewer full `prepare/evaluate/sample/pdf` parity；
- [x] bundle replacement/typed edit只更新asset/instance，program runtime缓存复用且失败原子；
- [x] viewer/package不识别Metal/module/preset，不存在v1 loader；
- [x] 两个slot对称，Release build后Falcor clean。

### 3.6 Windows/Linux handoff后

- [x] Windows用full shape完成四phase真实optimizer steps和phase resume；
- [x] 全cohort preflight及component execution/gradient/update/artifact coverage闭合；
- [x] optimization run使用机械生成的最小stratified子集提高验证效率，但保持full shape、真实online reference、loss/optimizer/phase data flow并覆盖全部required groups；
- [x] 无host target readback、磁盘batch、NaN/Inf、非法PDF、负f或silent clamp；
- [x] 预冻结统计方法证明短run末段loss低于初段；
- [x] profile记录reference/encode/forward/backward/optimizer/I/O、memory和sync；
- [x] Linux smoke/long config只改变budget/cadence，单GPU命令、resume/monitor/stop/recovery与review manifest齐全；
- [x] parent需求逐项有证据；不自动启动formal、追加训练、消融或Pareto。

## 4. 通用验证

所有Python/pytest使用`neural-shading`环境；Falcor Python通过平台launcher。第一次验证前按`dev-environment.md`重新报告机器状态。parent最终至少执行：

```powershell
conda run -n neural-shading python -m pytest tests/unit
scripts/run_falcor_python.ps1 -Command "python -m pytest tests/gpu"
scripts/run_falcor_python.ps1 -Command "python -m ncls.cli learn train <metal-full-smoke-config> <checkpoint>"
scripts/run_falcor_python.ps1 -Command "python -m ncls.cli learn export <checkpoint> <diagnostic-package>"
scripts/build_viewer.ps1 -Configuration Release
bash -n scripts/deploy_reference_linux.sh scripts/run_falcor_python.sh
git -C external/Falcor status --short
git diff --check
```

长于smoke的Windows run使用`tqdm`按真实phase/step/batch更新；结果写`artifacts/`。Linux long run只交付config/commands，不在Windows宣称已执行。

## 5. 风险与处理

| 风险 | 最早发现点 | 处理 |
|---|---|---|
| canonical migration范围大、调用方遗漏 | child 1 denylist/static tests | 递归迁移并删除旧接口；不加adapter过渡 |
| grouped MDL resources不能安全共享 | child 2 group/session tests | 收紧execution-group key，保持batch homogeneous；不退回family producer |
| 52 assets端到端encoder成本过高 | child 2/3 tile profile | tile+halo与GPU working set；不冻结encoder假装joint training |
| full profile在4090显存不足 | child 3 static/activation preflight | 优先microbatch/activation checkpoint；shape确实不可部署时回parent创建新profile identity，不静默tiny化 |
| required branch有参数但无梯度 | child 3/4 conformance | 补activation/loss/data flow；不能标记optional绕过 |
| sampler依赖尚未收敛evaluator | child 4 analytic/micro-overfit controls | 验证数学与优化路径，quality留Linux长训；不要求formal checkpoint |
| Package@2迁移破坏NVIDIA/viewer | child 1与5回归 | 修正唯一新合同；不恢复v1 reader |
| Windows短run loss不下降 | child 6 failure classification | 区分implementation/protocol/resource/normal empirical outcome；correctness问题修复，扩大预算需回planning |
| Linux新method首次smoke失败 | handoff preflight | 长训前停止，按报告返回；不自动启动formal/消融 |

## 6. 收尾

parent在六个当前child分别完成质量门、scoped local commit和归档，所有旧接口删除、Windows full-profile gate通过且Linux单GPUhandoff齐全后执行最终集成复核并归档。Linux long run及其checkpoint不属于parent完成证据；结果产生后先向用户提交审阅摘要，再决定新的任务范围。
