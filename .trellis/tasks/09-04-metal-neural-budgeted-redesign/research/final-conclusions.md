# Metal budgeted neural material 本轮结论

## 交付结果

本轮已经在物理GPU 5–9上完成五卡DDP的hybrid/direct matched训练、共同step2048选择、部署量化阶段、checkpoint恢复、Slang/package parity与两个Windows diagnostic viewer包。最终可交付artifact为：

- 根目录：`artifacts/viewer/metal-budgeted-ddp5-wrap-1d5f813-step2048/`；
- hybrid checkpoint：`8a15a5945085bddc781c1e60cd434ffa78b3a791ceed05dbbe007f8e7fb8971e`；
- direct checkpoint：`4848b783407eba3a0127910dca370ea97f95f904416e974ad61867a3bbff2042`；
- hybrid package：`61447f0bb57979413cba4b72c3bf974cc18ae212b5a4bf22d519e5efcdc3d820`；
- direct package：`00747206e1a929eb2012aef594635a66775a1cf47e1d97a5a5a4461711b94182`。

两个package都是`exact-diagnostic-evaluator-preview`，只声明`prepare/evaluate/anisotropic-frame`，不把未交付的formal readiness、typed edit或`sample/pdf`视觉路径冒充完成。Linux Falcor/Vulkan从package自身manifest、FP16 blob与DDS重新加载后的最大绝对误差分别为`1.87e-5`和`2.23e-4`，均通过冻结容差。Windows D3D12尚未在本机运行，具体启动命令见artifact内README。

## 已确定的模型选择

hybrid与direct共享11,392 dense MAC/direction、160 B PreparedState、两次asset读取、相同source/query/loss/optimizer/QAT与训练量。step2048的`direct-hybrid`差异如下：

| metric | 差值 | 95% paired bootstrap CI | 结论 |
|---|---:|---:|---|
| appearance | `+0.68544` | `[+0.67929,+0.69144]` | hybrid明确更好 |
| log RGB | `+0.36947` | `[+0.36770,+0.37123]` | hybrid明确更好 |
| linear RGB | `+0.21998` | `[+0.21327,+0.22659]` | hybrid明确更好 |
| chroma | `+0.005938` | `[+0.005920,+0.005955]` | hybrid明确更好 |
| peak RGB | `+0.67230` | `[+0.65965,+0.68481]` | hybrid明确更好 |
| spatial gradient | `-0.001790` | `[-0.001967,-0.001617]` | direct仅有很小空间优势 |

因此当前best observed candidate是`metal_budgeted_hybrid_v3`；direct保留为同成本视觉对照，不再作为默认方向。这个选择的原因是analytic core对Tungsten的平均响应、高光和颜色有显著价值，不是hybrid用了更多网络或状态预算。

hybrid内部机制仍不理想。最终positive RGB trace约为`10^-6`量级，主体是learned gate调制analytic lobes；这说明“保留稳定物理core”有效，但“神经residual已经学会有用修正”不成立。下一轮应显式分离core、gate与residual的监督责任，而不是把当前hybrid胜出解释成完整neural representation已经成功。

## 训练效率与公共架构结论

通用validation packed reduce把相同规格验证从约`50.99 s`降到`41.25 s`；bounded lookahead、每16 step report和two-step reference packing进一步消除了训练侧的无谓同步。后续同一Tungsten/hybrid结构的batch profile为：

| per-rank/global batch | steady global work units/s | peak MiB/rank |
|---:|---:|---:|
| `512/2,560` | `21,135` | `747.65` |
| `1,024/5,120` | `42,364` | `965.80` |
| `2,048/10,240` | `87,677` | `1,399.98` |

本轮后置实验因此使用per-rank 2,048。更大batch确实是有效的效率轴，但它只提高每step工作量与GPU占用，不会修复空间表示；在2,048仍只有约1.37 GiB/rank时，显存不是当前模型实验的主要约束。是否继续到4,096应作为下一轮有界profile轴冻结，不能与本轮不同batch结果按step混比。

DDP和数据路径修复了四类通用缺陷：step-0空metric review、component output conformance、多source全局calibration聚合、MDL content cache per-key跨进程互斥与partial清理；另外把profile checkpoint tensor schema门禁前移到config resolve。所有修复都保留`static_graph=True`与`find_unused_parameters=False`，没有用延长timeout或unused扫描掩盖根因。

多source评测还新增了`group-block-balanced@2`。它让training继续按64 optimizer step绑定group，而DDP5的256-batch validation固定覆盖同一20-group cohort；旧`@1`语义保留供既有checkpoint复现。这避免了“一个window只看5个group、下个milestone又换材质”被误读成学习曲线。

## 特征材质与mixed cohort结论

平滑黄铜和阳极氧化铝从128到256 step的平均appearance、peak有明确改善；划痕青铜与开裂涂漆钢则始终暴露one-texel空间瓶颈。三种预算内或近似matched修订都没有形成跨两种高频材质的净收益：

- v4 role-separated Detail只带来极小、互相不一致的空间变化；
- v5使用请求中心texel，增加输入变化幅度后仍重新分配而不是消除误差；
- v6加入signed x/y局部导数并把两张RGBA plane都升到全分辨率，预计asset bytes为v3的`1.882×`，仍没有同时改善青铜和钢的spatial。

fixed-batch spatial-only优化也不能让预测梯度逼近target，因此继续训练、提高spatial loss权重、增加同类局部统计或扩大方向MLP都不是优先项。

最终mixed diagnostic使用692-source registry fragment、单seed、per-rank batch2,048、512-step cap。有效checkpoint为`artifacts/metal-budgeted-probes/mixed-cohort-v3/checkpoint.pt`，SHA-256 `61a2b4aae12fa3a1bdd222f51139b6f5092ec66568ef496fbde01b672692b656`。同一20-group分布上step128→512的appearance改善`-0.08675`，但peak退化`+0.13626`、spatial只改善`-0.00217`。四个完全相同query的group block中，log/chroma/peak改善，linear/spatial却分别退化`+0.02608/+0.000104`；预测gradient约`0.0030`，target为`0.1474`。

这给出两个限定结论：当前结构可以较快学习部分材质的平均散射与高光统计；当前source patch→RGBA latent→prepare链路没有学习到普适局部散射场。不同group的linear/peak方向不一致，且512-step training只覆盖有限group block，所以该实验不能宣称692-source泛化质量。

## 下一轮优先方向

第一优先级是在相同两读、160 B、11,392 MAC下重做asset表示，而不是先扩大网络：

1. family-specific compiler保留源材质原生语义，把高分辨率RGBA通道明确分配给局部color、normal/frame、roughness与mask/coverage证据；这是编译输出语义，不要求把MDL源材质改写成层模型。
2. 训练直接经过部署时真正使用的量化plane、mip选择和bilinear读取，并对被evaluator消费的局部语义加辅助监督；避免训练encoder看到patch统计，而runtime只看到已经丢失方向信息的四通道结果。
3. 保留analytic core作为窄峰与基础颜色结构，但给neural correction显式、可正可负且有界的reference-minus-core职责；分开报告core、gate、residual的目标与梯度，防止positive residual长期退化为零。
4. 首个matched消融只重新分配现有8个RGBA通道；若跨平滑、Beckmann、划痕和paint/crack都改善spatial，再把“增加高分辨率通道/asset bytes”作为独立Pareto轴。
5. 多source正式实验先冻结可在预算内覆盖完整cycle的代表性cohort或显式coverage统计；`group-block-balanced@2`作为最低评测协议，不再用随milestone变化的cohort判断趋势。

本轮不建议优先做：把当前v3继续到更多step、提高spatial loss、增加主MLP宽度、继续v4–v6同类patch summary，或把direct重新作为默认。batch大于2,048可以继续做吞吐profile，但它是执行效率问题，不是模型质量方向。

## 验证边界

本轮最终检查为：完整unit `360 passed`；当前budgeted GPU model/package `2 passed`；MDL asset integration `2 passed`；layout generator、两个package结构校验、两个package自加载GPU parity、`git diff --check`和Falcor上游clean均通过。额外执行的历史`metal_fused_full_v1`量化runtime测试在Linux Falcor compute dispatch处native fatal；当前budgeted测试隔离重跑通过，因此它被记录为尚未闭合的历史control，不用于否定本轮两个新package，但也意味着“四control matched runtime报告”仍未完成。Windows Release viewer/D3D12视觉capture同样需在目标Windows主机执行。
