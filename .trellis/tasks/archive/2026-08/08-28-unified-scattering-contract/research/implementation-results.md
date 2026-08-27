# 统一 scattering contract 实施结果

## 结论

本任务完成的是 root migration，不是 compatibility wrapper：LayerStack、MERL、MaterialX、OpenPBR 与 MDL 的 canonical shader backend 都直接实现 `prepare/evaluate/sample/pdf`；`ReferencePathTracer` 与 `PackagePathTracer` 只调用 prepared state。旧 `MdlViewerAdapter.slang`、renderer-side family estimator、generic proposal/PDF fallback 与 LayerStack 多层 `pdf=0` 路径已删除。

用户看到的白点包含两个不同问题：

1. 旧架构允许 evaluator、sampler 与 MIS PDF 分属不同 owner，这是必须消除的 estimator defect。
2. canonical sampler 生效后，car paint/ceramic 的极高值集中在 HDR environment 的真实镜面反射位置，并非随机散布；增加到 4 个 environment NEE 样本后，在不 clamp radiance/throughput 的情况下 mean RSE 明显下降。

质量检查还发现 OpenPBR official `sample()` 内部累计 PDF 与 independent `pdf()` 在极窄 continuous lobe 上存在不同 float32 路径。最初尝试用 independent `openpbr_eval()/pdf()` 重建公共 sample，在普通 aluminum/glass query 上能通过，却会让 car paint 的 `roughness=0.02` coat 在掠射方向发生 half-vector 重建相消：native sample weight 约 `0.007–0.34`，重建后达到 `10^14`，viewer 1024 spp raw max 达 `1.73×10^9`。最终 backend 恢复不可拆分的 Adobe native direction/event/PDF/weight tuple；independent eval/pdf 仍负责 NEE。logarithmic grazing probe 验证 native identity，最终 car paint capture max 回落到 `438.01` 且孤立 `>100` 像素为 0，没有 generic fallback 或 clamp。

capture 边界还发现 Falcor/FreeImage 在 `ExportFlags::None` 下把 RGBA32F 线性资源写成 half EXR，合法的 `>65504` 数值会在文件中变成 `Inf`。viewer 现已显式写 float32 EXR；这修的是权威诊断产物，不改变或裁剪 radiance。

## 架构结果

- Python `ReferenceProgramDescriptor` 缺 `PREPARE|EVALUATE|SAMPLE|PDF` 任一项即 fail closed。
- response measure 统一为 `f * abs(dot(Ns, wi))`，reflection/transmission 都只乘一次 cosine。
- project-owned reflection mixture 是 cosine support + 最多四个 rotated anisotropic GGX VNDF lobe；sample/PDF component normalization 与 null mass 配对。
- `SceneReferenceProgram` 只做 heterogeneous concrete-state dispatch。受 Slang/DXC resource handle lowering 约束，composer 保存 resource-free prepared value，并在调用点重建含 resource 的 concrete state；这不是第二套 ABI。
- MDL dynamic module 只含 SDK types、renderer runtime 与 generated target code；static `mdl.slang` 是 formal/viewer 共用的唯一 canonical backend。
- source/package PT 每 hit 做 4 个 environment NEE 样本，light-sampled 与 BSDF-hit MIS 都使用 `4*p_light`。续路径 origin 由 actual sampled direction 选 geometric-normal side。

## 验证矩阵

| 门 | 结果 |
|---|---|
| focused unit | 22 passed |
| full unit | 95 passed |
| full Falcor/D3D12 GPU | 33 passed |
| integration | 6 passed |
| Python bytecode | `compileall` passed |
| viewer | `scripts/build_viewer.ps1 -Configuration Release` passed |
| static/dead path | 旧符号只存在于 absence assertions；integrator 无 `surface.family`；无 generic fallback/clamp |
| upstream | Falcor、MaterialX、OpenPBR、openpbr-bsdf、pbrt-v4、glm 全部 clean |

GPU 数值门覆盖：reflection mixture `N=524,288` 的 sample→pdf 与 continuous/null mass；LayerStack 单层 weight 和多层 sample→pdf；MERL chrome；MaterialX low-roughness anisotropy；OpenPBR aluminum/car paint/glass transmission 各 `N=262,144`，其中前 `1/8` 是 logarithmic grazing `wo`；MDL canonical/formal target code；neural proposal/package parity。

## 1024 spp capture 结果

以下均来自最终 Release viewer 的 raw single-panel EXR；tail 数值是 report-only 诊断，不是事后新增 hard gate。

| source | mean RSE | max | p99.99 | `>100` components / isolated pixels | 解释 |
|---|---:|---:|---:|---:|---|
| MDL car paint | 0.030950 | 430.75 | 252.15 | 1 / 0 | 11 个高值像素组成一个 7×3 范围内的连续 HDR highlight |
| MDL ceramic | 0.034123 | 396.25 | 192.38 | 1 / 0 | 8 个高值像素组成一个连续 HDR highlight |
| OpenPBR brushed aluminum | 0.074891 | 875.0 | 614.15 | 6 / 1 | scratch/highlight 分布较宽；同一尾部在 one-bounce、修复前后 capture 稳定复现，不是逐帧新增的 estimator firefly |
| OpenPBR car paint | 0.026396 | 438.01 | 241.14 | 1 / 0 | native sample tuple 修复后，10 个高值像素组成一个连续 coat highlight；旧重建路径 max 为 `1.73×10^9` 且有 7 个孤立点 |
| OpenPBR glass | 0.072952 | 6542.16 | 5052.61 | 8 / 3 | transmission/多次折射形成宽 HDR tail；全 finite，未出现 car paint 的数量级爆炸 |
| MERL chrome | 0.014982 | 207.75 | 124.50 | 1 / 0 | 镜面环境 highlight 连续聚集 |
| MaterialX rusty metal | 0.020828 | 25.94 | 20.80 | 0 / 0 | 无 `>100` tail |
| LayerStack | 0.000919 | 269.25 | 58.21 | 1 / 0 | 5 个相邻高值像素，EXR finite |

相同 replay 下，4-sample environment NEE 相对原 1-sample estimator 的 observed mean RSE 变化为：car paint `0.045164 → 0.030950`（下降 31.47%）、ceramic `0.046799 → 0.034123`（下降 27.09%）、OpenPBR `0.082283 → 0.074891`（下降 8.98%）。这是质量—成本观测：slot GPU 时间相应上升，不把它写成新的产品 hard gate。

最终 capture 位于：

```text
artifacts/captures/unified-scattering-contract/mdl-carpaint-final/
artifacts/captures/unified-scattering-contract/mdl-ceramic-final/
artifacts/captures/unified-scattering-contract/openpbr-final-v2/
artifacts/captures/unified-scattering-contract/openpbr-carpaint-native-float32/
artifacts/captures/unified-scattering-contract/openpbr-glass-native-float32/
artifacts/captures/unified-scattering-contract/merl-final/
artifacts/captures/unified-scattering-contract/materialx-final/
artifacts/captures/unified-scattering-contract/layer-final/
```

## 已知但不阻塞的观察

- OpenPBR brushed aluminum 的正确 estimator 方差仍高于其他这组 capture；这属于当前材质/环境组合的 observed outcome，不是 sample/PDF mismatch。后续若要优化，只能在保持 official eval/sample/pdf 合同的前提下研究更好的 source-private proposal 或 environment strategy。
- Release build 仍显示 Falcor 既有 `LNK4098` 与 Slang implicit-conversion warnings；本任务没有修改上游，构建成功且 upstream clean。
