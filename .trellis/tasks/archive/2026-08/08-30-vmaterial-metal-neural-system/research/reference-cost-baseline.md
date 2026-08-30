# vMaterials Metal reference 成本基线

## 1. 它回答什么

本记录把 vMaterials Metal reference 从单纯的质量 oracle 扩展为成本基线。目标不是要求质量优先首版立即比 reference 更快，而是从任务开始就记录时间与存储，待完整模型稳定后持续做 matched 消融和部署优化。若最终方法在用户认可的质量与能力范围内，运行时间和存储/显存两项都不占优，它只能形成表达性研究结果，不能形成有产品价值的 neural material 方法。

本轮运行环境为完整 Windows 开发状态：锁定 Falcor 8、MDL SDK `2025.0.0-387700.1252`、D3D12 与 CUDA interop 均可用，GPU 为 RTX 4090。静态审计覆盖全部 692 个 opaque authored exports；动态计时先覆盖当前已有正式 decoded runtime 的 `Aluminum_Scratched` 与 `Copper_Antique_Brushed_Patinated`。所有本轮数字均为 diagnostic observed result，不是事后写入的 hard gate。

原始产物位于：

- `artifacts/reference-cost-baseline/metal/static-cost.json`
- `artifacts/reference-cost-baseline/metal/query-benchmark-repeat30.json`
- `artifacts/reference-cost-baseline/metal/hlsl-source-cost.json`
- `artifacts/reference-cost-baseline/metal/dxil-static-costEvaluate.json`
- `artifacts/reference-cost-baseline/metal/dxil-static-costPdf.json`
- `artifacts/reference-cost-baseline/metal/dxil-static-costSample.json`
- `artifacts/reference-cost-baseline/metal/viewer-*/capture.json`

## 2. 静态存储成本

### 2.1 全 opaque cohort 去重后

| 项目 | 规模 | 含义 |
|---|---:|---|
| opaque authored exports | 692 | Metal-v1 source coverage |
| opaque compiled graph identities | 178 | 不是每个 preset 都需要不同图 |
| opaque texture-set identities | 52 | neural bundle 与 split 的主要资产单位 |
| opaque source texture paths | 137 | 按源文件路径去重 |
| opaque runtime texture bindings | 138 | 同一源文件存在一次不同 gamma/runtime 解释 |
| source 压缩文件 | 323,338,273 bytes，约 308.36 MiB | JPG/PNG 交付体积；不能直接等同 GPU 随机访问驻留 |
| bridge decoded payload | 5,234,026,752 bytes，约 4.87 GiB | `Rgb` 仍为三通道的 bridge 中间文件 |
| 当前 viewer LOD0 GPU texture | 6,959,303,680 bytes，约 6.48 GiB | `Rgb` 展开到 RGBA8 等实际格式；不含 mip chain |

当前 formal MDL viewer 明确使用 `explicit-lod0`，不消费 UV derivatives，也不生成 mip。因此 6.48 GiB 是当前实装的 LOD0 总量，不是带完整过滤的未来 reference 总量。若仅作幂次纹理完整 mip chain 的解析估算，纹理驻留约再乘 `4/3`，即全局约 8.64 GiB；这是 derived estimate，不是已测实现。

### 2.2 单个 active texture set

52 个 opaque texture-set identities 的 active-set 分布如下：

| 成本 | minimum | median | p90 | maximum |
|---|---:|---:|---:|---:|
| source 压缩文件 | 1.26 MiB | 9.27 MiB | 15.35 MiB | 24.49 MiB |
| bridge decoded payload | 48 MiB | 144 MiB | 192 MiB | 444 MiB |
| viewer GPU LOD0 | 64 MiB | 192 MiB | 约 256 MiB | 592 MiB |

两个动态计时材质的具体资源为：

| 材质 | source 压缩 | decoded payload | GPU LOD0 | HLSL source | argument block |
|---|---:|---:|---:|---:|---:|
| Aluminum Scratched | 12.65 MiB | 144 MiB | 192 MiB | 60,824 B | 88 B |
| Copper Antique Brushed Patinated | 15.45 MiB | 224 MiB | 288 MiB | 49,662 B | 56 B |

这说明 reference 的驻留成本由 texture 主导。692 个 opaque export 的 argument block 中位数只有 72 B、最大 112 B；RO data 中位数为 0、最大 2 KiB。generated HLSL 的单 export 中位数约 59.18 KiB、最大约 145.38 KiB；按 178 个 graph representative 相加约 10.16 MiB。HLSL source 大小只是复杂度与交付代理，不是最终 GPU binary 或执行时间。

## 3. Online reference query 成本

诊断使用统一 `ReferenceBackendSession`，每次计时包含 Falcor/CUDA interop binding、operation dispatch、lease release、`end_frame` 和 CUDA synchronization。decoded artifact 已存在，因此 session open 是 warm-artifact 路径。每种 operation 先 warmup 10 次，再重复 30 次；65,536 查询使用 15 次重复。

| 材质 | operation | query 数 | median | p90 | median throughput |
|---|---|---:|---:|---:|---:|
| Aluminum Scratched | evaluate | 1 | 1.166 ms | 1.257 ms | 不解释为 shader 单 query |
| Aluminum Scratched | evaluate | 65,536 | 1.297 ms | 1.492 ms | 50.5 M query/s |
| Aluminum Scratched | sample | 65,536 | 1.320 ms | 1.535 ms | 49.7 M query/s |
| Aluminum Scratched | pdf | 65,536 | 1.308 ms | 1.404 ms | 50.1 M query/s |
| Copper Patinated | evaluate | 1 | 1.175 ms | 1.272 ms | 不解释为 shader 单 query |
| Copper Patinated | evaluate | 65,536 | 1.351 ms | 1.687 ms | 48.5 M query/s |
| Copper Patinated | sample | 65,536 | 1.307 ms | 1.475 ms | 50.1 M query/s |
| Copper Patinated | pdf | 65,536 | 1.260 ms | 1.897 ms | 52.0 M query/s |

1、256、4096 与 65,536 查询的 wall time 都接近 1.2–1.35 ms，说明当前计时主要受一次同步/interop 固定成本控制；不能把 1 query 的约 1.17 ms 当作 MDL shader 的单次执行时间。65,536 查询的 throughput 可用于估计当前 online producer 的大 batch 成本，但仍不是纯 GPU timestamp。

warm-artifact session open 为约 1.09–1.46 s；其中首个 session 还包含 Falcor device 初始化，不能把两者差异归因于材质。正式效率工具需要把 device startup、artifact load/texture upload、shader compile 和 steady-state query 分离计时。

当前 dispatcher 对 material-specific generated source module 一次只允许一个 snapshot，因此本轮没有伪造 Aluminum/Copper 混合 wave 的 divergence 数据。多材质 divergence 必须由扩展后的 benchmark 或 viewer 场景层实际测量。

## 4. Viewer 端到端成本

固定 shaderball 场景使用总输出 `640×360`，单 slot panel 为 `320×360`。source PT 的 primary surface 每 spp 固定执行 4 个 environment-light samples 与 4 个 BSDF path samples，scene bounce cap 为 2。下面的 slot GPU timer 包含 source material、lighting、ray tracing 和路径续传，不是单独的 `prepare/evaluate` 时间。

| 材质 | 每 dispatch spp | 三次 GPU slot 结果 | median |
|---|---:|---|---:|
| Aluminum Scratched | 1 | 1.076 / 1.111 / 1.127 ms | 1.111 ms |
| Copper Patinated | 1 | 0.632 / 0.638 / 0.653 ms | 0.638 ms |
| Aluminum Scratched | 16 | 24.554 / 32.227 / 32.356 ms | 32.227 ms |
| Copper Patinated | 16 | 12.472 / 12.599 / 19.059 ms | 12.599 ms |

1 spp 结果较稳定，可作为当前同场景 neural/source matched viewer comparison 的首个锚点。16 spp dispatch 存在更大波动，而且 shader 内循环、cache 与路径状态使其不能简单除以 16 充当单 spp 时间。

HLSL、优化 DXIL 与 viewer 调用图的进一步审计见 `research/hlsl-per-pixel-cost.md`。关键结果是：一个完全可见的 primary material hit 在当前 environment-only、1 spp 配置下不是一次 BRDF，而是 4 次公开 evaluate、4 次公开 pdf 和 4 次公开 sample。默认参数下约产生 Aluminum Scratched 120 次、Copper Patinated 68 次 material `SampleLevel`。当前 MDL backend 的公开 operation 会各自重复 generated `init()`，viewer 还丢弃 evaluate 已计算的 PDF 后重新调用公开 pdf；因此后续必须增加 prepare-hoisted、PDF-reuse 的 optimized-code control，不能只与当前组织开销比较。

## 5. 对模型与验收的影响

### 5.1 质量优先路线不变，但从第一天记账

完整融合候选仍先追求质量，可以暂时超过 reference 成本；不过每个训练 identity、MethodBundle 和 viewer 产物从一开始就必须记录：

- `B_shared`：shared texture decoder、evaluator、compiler 与 sampler 权重；
- `B_asset`：hierarchical grids、asset adapter、metadata 与 sampler hints；
- 单资产、典型 working set 与全部 52 bundle 的 disk bytes 和 runtime resident bytes；
- `prepare/evaluate/sample/pdf` 的 coherent 与 divergent query 成本；
- artifact compile、texture encode/refine、load/upload 和 typed edit state regeneration latency；
- matched viewer workload 的 slot GPU time。

完整候选稳定后，再以同一质量评测和运行合同做容量缩减、量化、共享/adapter 消融、read count 缩减与 lobe/frame/field 简化。不能等到最后才发现模型虽然好看却没有可压缩或可部署空间。

### 5.2 不能只与当前未压缩 viewer texture 比

成本比较需要三个角色：

1. **authoritative MDL reference**：提供准确 GT 和当前执行成本；
2. **conventional deployment control**：使用语义匹配的 BC4/BC5/BC7 等 GPU texture compression、完整 mip/filtering 与 optimized source code，提供非 neural 的部署对照；
3. **neural method**：统计 shared weights 与每资产 bundle 的完整成本。

只证明 neural bundle 小于当前 RGBA-expanded LOD0 reference 太容易，也不能证明 Random-Access Neural Compression 风格方法有独立价值。source JPG/PNG 虽然只有约 308 MiB，但它们不是直接 GPU random-access 表示；仍要同时报告 delivery disk、decoded upload 和 runtime resident 三层。

### 5.3 后续产品价值讨论（当前不冻结判据）

研究报告画出质量—steady-state time—resident memory/delivery storage Pareto，不把本轮 observed 数字写成 hard gate，也不在当前阶段决定按单材质、分层 cohort 还是 working set 汇总成功。先保留两个不同层次的结论语义：

- **研究交付成立**：完整方法语义正确、质量与成本均被可靠测量，低效率可以作为真实研究结论；
- **产品价值讨论**：总体上仍要求在用户认可的质量和能力范围内，相对 authoritative reference 与 conventional deployment control 展现时间或存储/显存优势；若两者最终都不占优，统一 shader 或研究新颖性不能单独替代产品价值。具体门槛、质量范围和聚合方式等有 full/compact/matched-control 实测后再共同确定。

shared decoder 的成本必须按 `N=1`、典型 working set 和全 cohort 三种规模报告，并给出 amortization break-even；不能只把 shared weights 除以 52 后报告一个好看的平均数。

## 6. 后续正式 baseline 补齐

当前动态结果只覆盖两个既有 Metal runtime，不足以代表 178 个 opaque graph identities。正式设计需要新增一次性成本轨道：

1. 静态层继续覆盖全部 692 exports、178 graphs 和 52 texture sets；
2. 按 standard finish、special recipe、texture working-set 大小和 HLSL complexity 分层选择代表材质，补齐 GPU query 与 viewer 分布；
3. 为 query dispatcher 加 GPU-only timestamp，拆开 interop/sync 固定成本；
4. 建立真实多材质 coherent/divergent workload；
5. 在 reference 与 neural 两侧采用相同 footprint/mip/filtering 语义后再比较；
6. 增加 conventional compressed-texture control，避免只与 uncompressed runtime 比。

这条成本轨道不改变“先做质量优先完整形态”的模型开发顺序；它从现在起只负责持续采集共同指标，并为后续 Pareto 判断提供事实，不在当前阶段触发效率 kill test 或冻结成功口径。
