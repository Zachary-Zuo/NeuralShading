# Viewer 交互性能回归分析

## 1. 结论

当前 UI 卡顿不是 MDL 在每帧重新编译，也不是统一 `prepare/evaluate/sample/pdf`
接口出现了另一条材质专用路径。根因是两个成本相乘：

1. firefly 修复把一个 primary BSDF continuation 扩成了 4 条完整 path suffix，并在
   secondary vertex 上保留 `4 light + 4 BSDF direct + 1 continuation`；单个 global sample
   的材质调用和 ray 数显著增加；
2. viewer 仍把正式 capture 的 `samples per frame = 16` 当成交互预算。相机拖动时也执行
   完整 16-sample dispatch，只是关闭 accumulation，算完后把 `spp` 重新置零。

因此用户看到的是单次约 113 ms 的 PT dispatch 阻塞 UI；相机变化还会令 visibility pass
变 dirty。正式 reference 的 estimator 正确性与交互调度被混在了同一个
`mSamplesPerFrame` 开关中。

## 2. 开发机与诊断边界

本次在完整 Windows 开发机上执行，当前 Release viewer 日志确认：Falcor 8.0、
Direct3D 12、SM 6.7、NVIDIA GeForce RTX 4090。诊断固定使用同一 MDL car paint、
960×540 composite（单 slot 480×540）、scene bounce 2、1024 spp，只改变每次 dispatch
包含的 global sample 数。结果属于性能 diagnostic，不改变既有质量结论。

产物位于：

- `artifacts/diagnostics/pt-salt-pepper-noise/interactive-profile/spf1/`
- `artifacts/diagnostics/pt-salt-pepper-noise/interactive-profile/spf4/`
- `artifacts/diagnostics/pt-salt-pepper-noise/interactive-profile/spf16/`

## 3. 同场景 batching 对照

| samples per frame | PT slot GPU ms / dispatch | 1024 spp RSE |
|---:|---:|---:|
| 1 | 4.033536 | 0.0189165901 |
| 4 | 22.168576 | 0.0189165901 |
| 16 | 112.784384 | 0.0189167690 |

三组 1024 spp RSE 在记录精度内一致，说明这里改变的只是 batch 形状，不是 estimator 或
最终 reference identity。`SPF=16` 约等于 8.9 dispatch/s，还未计 UI、composite、
visibility 和 CPU/GPU 提交开销；这与“UI 几乎无法实时调整”的现象一致。`SPF=1`
把同一正式 estimator 的单次 PT latency 降到约 4 ms。

历史冻结 capture 也显示单样本工作已经显著变重：

| 材质 | 修改前 GPU ms | 修改后 GPU ms | observed 倍率 |
|---|---:|---:|---:|
| MDL car paint | 9.885696 | 103.712768 | 10.49× |
| MDL ceramic | 8.803328 | 44.376064 | 5.04× |
| OpenPBR car paint | 6.477824 | 47.052800 | 7.26× |

这些倍率受材质成本、路径命中率和 GPU divergence 影响，只作为 observed cost，不是
理论常数。

## 4. 单个 global sample 为什么变贵

当前 primary surface 先做 4 个 environment NEE，然后调用 4 次 native `sample()`；
每个成功 sample 都追踪自己的完整 suffix。每个 secondary/deeper hit 又执行 4 个 light
samples、4 个独立 BSDF direct samples，并在尚未达到 bounce cap 时再取一个 continuation。

以 scene bounce 2、所有 visibility 和 suffix 都命中的最坏路径形状估算：

| 工作 | 旧 estimator | 当前 estimator |
|---|---:|---:|
| native `evaluate()` | 12 | 36 |
| native `pdf()` | 12 | 36 |
| native `sample()` | 2 | 40 |
| primary/path/visibility rays 合计 | 约 15 | 约 73 |

这解释了为什么 MDL car paint 尤其慢：MDL 的 native `sample/evaluate/pdf` 本来就比简单
解析材质昂贵，新增调用又处于带分支的 path loop 中。实际倍率可能高于调用数倍率，原因
包括路径分歧、寄存器压力和较大的单线程顺序循环。

不能把这部分简单回退成“4 个 BSDF direct + 1 条完整 primary continuation”。任务内的
contribution AOV 已证明，那个版本仍由罕见的单条 high-throughput primary continuation
污染后续所有 secondary strategy；4 条完整 primary suffix 正是当前视觉改善的来源。

## 5. 交互调度为什么把成本放大成卡顿

`renderReference()` 和 `renderPackagePath()` 都直接用 `mSamplesPerFrame` 形成单个 dispatch。
shader 在每个像素内顺序循环 `gSamplesThisFrame` 次。当前 replay 恢复了正式 capture 使用的
值 16，因此一次交互帧也顺序完成 16 个昂贵 global samples。

相机拖动时，renderer 只把 `gAccumulate` 设为 false；`samplesThisFrame` 仍然是 16。
dispatch 完成后 slot `spp` 被置零，所做的大部分工作不会进入最终 accumulation。灯光和
其它连续 UI 参数每次变化也会重置 accumulation，下一帧仍按 16-sample batch 执行。

相机变化还调用 `resetReference(true)`，令 visibility pass 下一帧重新执行；灯光或材质
参数通常调用 `resetReference(false)`，不重建 visibility。这个额外 pass 会进一步影响相机
拖动，但不是材质/灯光滑条卡顿的首要根因。

普通 UI 修改只更新常量或 source buffer 并 reset accumulation。MDL artifact 的加载、GPU
resource 创建和 shader specialization 位于材质安装/切换路径，没有证据表明它们在普通
相机或灯光调整的每帧重复发生。

## 6. 正确的修复边界

下一步应先修 renderer scheduling，不修改材质接口，也不降低正式 reference 的总 1024 spp
或 MIS 数学：

1. 显式分离 `interactiveSamplesPerFrame` 与正式 refinement/capture batch；交互期间先固定为
   1 个 global sample/dispatch。
2. 相机拖动或连续参数编辑期间显示非权威 preview；结束交互时只 reset 一次，从 0 spp
   恢复同一正式 estimator 的 accumulation。
3. headless capture 和冻结 replay 继续使用记录的 batch 值与 1024 spp，保证产物 identity、
   可复现性和吞吐不变。
4. source reference PT 与 package PT 必须共享这套调度规则；不得按 MDL、ceramic 或 neural
   material 分支。

这一步预计把灯光/材质交互的 PT latency 从约 113 ms 降到约 4 ms。相机交互还需单独记录
warm visibility pass latency；若它仍主导，再优化 visibility invalidation 或 preview G-buffer，
不能用缩减正式 estimator 掩盖。

更深层的 estimator 性能优化应作为后续独立设计：例如 wavefront/branch compaction，或能保留
4-path primary averaging 统计语义的等价调度。未经新的 AOV 与 parity 验证，不应恢复单条
primary continuation，也不应加入 clamp、denoiser 或材质专用 fast path。
