# Metal 方法训练交付与架构迁移审计

## 1. 本轮澄清后的任务定义

本任务的完成条件不是在 Windows 上把 692 个 opaque Metal exports 的质量优先模型训练到正式收敛。当前交付应是：

1. 把 source→reference→online training→checkpoint→package→viewer 递归迁移到能自然承载 multi-asset、typed compiler、完整 evaluator/sampler 的唯一最新接口；
2. 正确、完整地实现 `metal_fused_full_v1`，禁止只实现易做分支后用配置、默认值或空状态隐式跳过其余组件；
3. 在 Windows/RTX 4090 上证明完整方法每个训练阶段都能运行、梯度可达、更新有限且有明确收敛信号，并完成 checkpoint/export/Slang/viewer smoke；
4. 交付同一套 source、method、config 与 runner 可在已验证 Linux backend 上直接进行长训练的冻结版本；
5. Linux长训练checkpoint产生后先由用户审阅；只有新的批准才执行full formal、完整泛化、matched ablation或产品Pareto。

## 2. 已有 Linux 证据

`TESTING.md:104-158` 和 `.trellis/tasks/08-29-mdl-cross-platform-reference-deploy/implement.md:57-64,131-163` 已记录：

- 2026-08-29 在 Ubuntu 22.04.5、RTX A6000、driver 550.78、Vulkan 与锁定 Falcor 8 上完成实机验证；
- 五个 canonical reference program 通过同一 `ReferenceBackendCapability/Session` 完成 representative `evaluate/sample/pdf`、same-device CUDA tensor 与 lease 验证；
- 固定 MDL state 在 Windows 与 Linux 使用同一 config/CLI 完成两步 GPU-resident online training、有限 loss/gradient、materialization、checkpoint reload/evaluate；
- Linux 最终复核为 130 passed，platform/device选择只位于公共 backend/launcher。

因此本任务不重新证明 Linux backend 的初步可行性；它必须保证新 Metal 架构没有引入 Windows-only source/method/trainer分支，并交付可在该 backend上启动的长训练配置。新 Metal full method 的最终 Linux smoke仍需在交付机执行，不能用既有固定MDL smoke代替。

## 3. 当前接口为何不足

### 3.1 `TrainingConfig@3` / runner

- config强制恰好`reference-evaluator`与`method-sampler`两条route；
- lifecycle只有`bootstrap→materialization→finetune`一次转换；
- optimizer与cosine schedule全局唯一，不能表达codec warmup、joint appearance、proposal fit、QAT/refinement的参数组与精度策略；
- runner没有AMP policy、逐component/parameter-group gradient reachability或update audit；
- 每step把GPU loss/metrics转Python scalar并逐parameter执行同步finite检查，不适合作为长训练热路径；
- route顺序同步产生，没有利用现有双slot lease做reference/query prefetch。

Metal需要任意typed route/phase graph、显式trainable groups、混合精度、异步指标与可恢复phase-local optimizer state。因此应发布唯一最新config/checkpoint合同并递归迁移NVIDIA和全部测试/config；不在runner旁边增加Metal trainer。

### 3.2 `MethodDefinition`

当前descriptor只声明batch fields、flat tensor schema与有限执行上限，不能机械证明下列full components都已实现和实际执行：role stems、shared encoder、per-mip high/low grids、semantic/structured heads、asset adapter、typed compiler、四路方向编码、analytic core、multiplicative correction、positive residual lobes、free tail、proposal sample/pdf与QAT/runtime packing。

新合同必须登记required component/parameter group/phase/runtime presence，并输出component execution、gradient、optimizer update与compiled artifact coverage。full profile缺任何required component时构造、训练、checkpoint或export必须失败，不能退成默认值、zero branch、identity adapter或未验证placeholder。

### 3.3 Source adaptation / reference execution

- `NativeFeaturePyramid`只表示一个materialization pyramid，Metal需要52套role/schema-aware assets和tile/halo访问；
- 当前producer对snapshot均匀采样，direction proposal固定且`direction_count=1`；
- 当前MDL session不能自然承载多个material-specific generated modules/argument states；
- 随机跨692 exports会造成program/resource高度divergent，不能形成高效长训练。

需要以`ReferenceExecutionPlan`把snapshots按generated program/RO/resources分group，并让producer生成group-homogeneous、asset/tile-coherent batches；以`NativeAssetCollection`替换单pyramid并递归迁移已有adapters。

### 3.4 Package / viewer

`ScatteringPackage@1`只有`program + material`两层，`material`同时承担finish latent asset、typed instance state和source snapshot identity。Metal三部分组合、runtime cache、bundle replacement和typed edit会迫使viewer在这一个section上增加特殊判断。

应升级为唯一canonical package合同，显式分离shared program、neural texture asset与compiled instance state；viewer按typed usage/capability绑定并缓存program。旧package reader、schema probe、converter、alias与fallback全部删除，NVIDIA exporter、tests、fixtures、docs和viewer一起迁移。

## 4. 无兼容层迁移原则

- 迁移发生在公共接口，不增加Metal CLI/runner/session/exporter/viewer旁路；
- 根仓库全部正式调用方一次切换到新接口并删除旧symbol/schema/reader；
- 旧checkpoint/package只保留在历史artifact provenance中，不提供加载、转换或自动探测；
- 正常的source/method多态实现不是兼容层；以旧字段/旧version继续工作才是兼容层；
- 每次迁移设置静态负向测试，证明旧API、旧format名、legacy config字段和family/platform分支不再存在。

## 5. Windows correctness 与 optimization gate

Windows不要求full-quality convergence，但必须运行完整shape和全部训练阶段：

- full cohort registry/asset/schema/execution-group preflight；
- component activation与parameter-group gradient/update audit；
- forward/loss/gradient/optimizer/checkpoint全部finite；
- deterministic online micro-overfit或固定query-stream短训练给出统计可信的loss下降，而不是只跑2步；
- codec warmup、joint appearance、proposal、QAT/refinement各至少经过实际optimizer steps与resume boundary；
- GPU-resident target无host response readback，无持久化batch；
- 记录reference、asset encode、forward、backward、optimizer、validation/checkpoint的step breakdown、peak memory和同步热点；
- 导出短训diagnostic package并完成Python→quantized→Slang→viewer、`prepare/evaluate/sample/pdf`正确性smoke。

这些是实现/优化流程正确性证据，不是最终Metal质量或产品效率结论。

## 6. Linux handoff

交付同一个platform-neutral long-run config、source registry/locator、checkpoint lifecycle、resume命令、部署检查与failure recovery说明。Windows smoke config只能缩短step/query/validation数量，不能关闭full components或改变method profile。Linux最终从Linux-native run启动/恢复，产生自己的backend build identity和artifacts。

用户确认当前硬交付保持单进程、单GPU：Windows和Linux都通过一个可见CUDA设备执行同一训练合同。multi-GPU DDP、per-rank reference、distributed sharding/RNG/checkpoint不进入本任务，也不预埋未经验证的分布式旁路。

Linux long run结束后先交付checkpoint、训练曲线、代表性材质/参数/texture replacement效果与基础成本摘要，等待用户审阅；不自动进入full formal、matched ablation或Pareto。
