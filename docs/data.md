# Reference 数据采集

## 它是什么

数据层是“给定研究问题后的确定查询采集接口”。它负责选择源材质状态、表面位置/footprint 和方向查询，在 Falcor 中调用各材质族自己的权威 reference，并统一写成一个 HDF5。collector 本身与材质无关；LayerStack、MERL、OpenPBR 和 MaterialX 都通过相同入口导出。

HDF5 保存的是可随机访问的监督快照，不是 neural 方法产物。模型结构、latent、loss、compiler 或 backend 改变不会改变数据布局。完整字段定义见 [ReferenceDataset HDF5 合同](contracts/reference_dataset.md)。

## 采集全部当前材质

Falcor Python 必须由项目脚本启动。下面命令把当前四个正式 provider 写入同一个文件：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli data collect-reference `
  --provider all `
  --output data\reference-responses\all-current.h5 `
  --families 128 --local-states 8 `
  --views 16 --validation-views 8 --test-views 8 --adversarial-views 8 `
  --lights 128 --spatial-samples 16 `
  --adaptive --batch-samples 256 `
  --min-samples 512 --max-samples 16384
```

`--families`、`--local-states` 和 Monte Carlo 参数只影响 LayerStack；`--spatial-samples` 与 `--footprint-width` 对当前 MaterialX provider 生效。MERL、OpenPBR 和 MaterialX 默认导出各自 manifest 中的全部已复现资产。

这条命令的方向数可以用于工程 smoke，但不是最终训练密度的默认结论。正式数据生成前应先完成 peak/掠射角监督审计，再冻结每个 provider 的 query proposal 和 state distribution。

E0 的定向覆盖诊断使用版本化的 `ncls.e0-peak-grazing-mixture@2`。它对每个 `wo` 分别生成 uniform、围绕真实镜面方向的多尺度球面 vMF peak，以及掠射角分量；完整球面的 OpenPBR 还包含透射侧 peak。vMF 是球面上的集中概率分布，半球版本把完整球 PDF 折叠后相加，因此近法线不会退化成环状采样，且仍有解析归一化 PDF。每个落盘 `wi` 都保存 mixture 的真实 PDF，并使用 `1 / (N p(wi))` 作为 Monte Carlo 积分权重：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli data collect-reference `
  --provider openpbr --material-id <transmission-asset-id> `
  --query-profile ncls.e0-peak-grazing-mixture@2 `
  --views 8 --lights 256 `
  --output artifacts\research\learning-goal\e0\openpbr-mixture-probe.h5
```

该 profile 先用于 E0 probe 和 proposal 对比，不自动成为后续正式训练分布。只有 audit 证明覆盖、估计方差与加权积分均满足冻结 gate 后，才会用明确配置生成正式 H5。

E1 的第一个单材质容量数据使用独立的 `ncls.e1-layer-stack-narrow-conductor@1` state profile 和 `ncls.e1-independent-peak-grazing-mixture@1` query profile。前者只包含一个固定极窄各向异性 conductor，不把 E0 六案例的集合外形提升为训练接口；后者让 train、validation、test 和 adversarial 四个 role 都使用不同 seed 的 uniform + 移动 peak + grazing mixture，并把 `wo` 扩展到 89°，使 held-out test 本身也能检验窄峰和掠射。对应监督入口由 `configs/research/e1-supervision-gates-v1.json` 冻结；它要求 64/16/16/16 个互不碰撞的 query group，并不代替模型质量 gate。

`ncls.e1-layer-stack-multi-interface@1` 是第二个一状态 E1 profile，只固定 E0 已验证的 `multi-interface-moving-peaks` 物理状态，用于区分 analytic core-only 与真正有贡献的 neural residual。它复用同一个独立 query profile 和 role 划分，但随机游走 reference 必须按 query group 自适应采样，并通过 `configs/research/e1-multi-interface-supervision-gates-v1.json` 后才能训练。新 ID 表明这是训练/残差压力用途；它不把整个 E0 boundary 集合变成训练 prior，也不改变公共 HDF5 或 reader 接口。

`ncls.e2-layer-stack-shared-decoder@1` 是 E2 的版本化共享表示 profile：固定 12 个材质族、每族 2 个保持相同图结构的局部参数状态。前 8 个族分别固定 1–8 个界面，另外 4 个族继续从研究 prior 采样；family 粒度划分产生 20/2/2 个 train/validation/test 状态，禁止同族局部状态跨 split。它只定义 LayerStack source distribution，不向公共 response reader 暴露 `LayerStackIR`。E2 使用 `ncls.e2-layer-stack-independent-peak-grazing-mixture@1`：每个 query role 明确锚定一个掠射 `wo`；LayerStack adapter 对单界面 Charlie sheen 从其原生 roughness 求解析峰中心，避免把镜面反射中心误当 sheen 峰。若 audit 证明某个 split group 已打满公共 adaptive 上限，CLI 可用 `--adaptive-max-samples-by-split-group GROUP=SAMPLES` 只提高该组上限；映射会写入 provider metadata，未知 group、重复 group、非 batch 整数倍或低于公共上限的值都拒绝。首份 v1 数据的失败仍由 `configs/research/e2-supervision-gates-v1.json` 复现；修正 proposal 后的数据必须通过 `configs/research/e2-supervision-gates-v2.json`，才能用于 optimized latent、target encoder 与共享 decoder 的比较。

## 单 provider 与指定资产

`--provider` 可以重复，`--material-id` 只允许在恰好一个 provider 时使用：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli data collect-reference `
  --provider merl --material-id alum-bronze `
  --views 16 --lights 256 `
  --output data\reference-responses\merl-alum-bronze.h5

.\scripts\run_falcor_python.ps1 -m ncls.cli data collect-reference `
  --provider materialx --material-id american_walnut_veneer `
  --views 8 --lights 128 --spatial-samples 64 `
  --surface-profile ncls.e0-footprint-scale-rotation-seam@1 `
  --footprint-width 0.000244140625 `
  --output data\reference-responses\materialx-walnut.h5
```

MaterialX 的 `ncls.e0-footprint-scale-rotation-seam@1` 是 E0/E5 之间的最小空间查询合同：在真实 UV 材质上组成 4 档 footprint 尺度 × 4 个方向，并额外固定 U、V 两轴 seam 的两侧坐标。它至少要求 20 个 spatial sample 和正 footprint 宽度。方向、尺度和 seam 坐标都实际写入 HDF5；audit 从 `uv/uv_dx/uv_dy` 重算覆盖，不以 profile 名称代替证据。默认 `ncls.constant-footprint@1` 只用于兼容普通 provider smoke，不能通过 MaterialX E0 gate。

快速 LayerStack smoke：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli data collect-reference `
  --provider layer-stack `
  --families 1 --local-states 1 --views 1 --lights 16 `
  --samples-per-replica 16 --max-depth 16 `
  --output data\reference-responses\layer-stack-smoke.h5
```

LayerStack 的默认 `ncls.layer-stack-research-prior@1` 仍用于随机结构/局部状态采样。E0 另提供固定的 `ncls.e0-layer-stack-boundary@1`，它不是训练 prior，而是六个可追溯 coverage probe：极窄 dielectric 高光、极窄各向异性 conductor、旋转各向异性 dielectric、色吸收 slab、满足当前 v0 同消光约束的色散射 slab，以及多界面移动峰。该 profile 的案例集合和数量都属于版本化语义，因此必须显式使用 6 个 family、每个 1 个状态：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli data collect-reference `
  --provider layer-stack `
  --layer-stack-state-profile ncls.e0-layer-stack-boundary@1 `
  --families 6 --local-states 1 `
  --query-profile ncls.e0-peak-grazing-mixture@2 `
  --views 2 --validation-views 1 --test-views 1 --adversarial-views 2 `
  --lights 128 --samples-per-replica 4096 `
  --output artifacts\research\learning-goal\e0\probes\layer-stack-boundary-v4.h5
```

案例 ID、profile ID 和采样配置同时进入原生 `MaterialProgram` metadata、provider metadata 与 HDF5 generation config。修改固定案例必须发布新 profile ID，不能在原 ID 下静默换状态。先用低预算 H5 按 state/query role 定位 reference noise，再只给最坏状态增加自适应预算；不得把百万样本上界无差别用于所有状态。

## 验证与读取

```powershell
conda run -n neural-shading python -m ncls.cli data validate `
  data\reference-responses\all-current.h5
```

验证会重新计算整个 HDF5 的语义内容哈希，并检查固定 group、shape、索引、payload offsets、split 和方向归一化。`--skip-hashes` 只用于局部诊断。

通用 Python 读取不解释源材质：

```python
from ncls.data import ReferenceDataset

with ReferenceDataset.open("data/reference-responses/all-current.h5") as dataset:
    train_groups = dataset.group_indices(source_split="train", query_role="train")
    batch = dataset.group_batch(train_groups[:32])
    print(batch["wo"].shape)
    print(batch["wi"].shape)
    print(batch["mean"].shape)
    print(dataset.state_payload(0))
```

response-only learning 使用 `ReferenceQueryStore`。当前 LayerStack baseline 需要原生 feature，因此显式使用 `LayerStackReferenceStore`；公共 store 不会出现 `interface_kinds`、层数或 OpenPBR 参数等 family-specific 字段。

## 采样职责

公共 collector 不替 provider 决定“恰当分布”。每个 provider 的 `source_states()` 决定原生参数/资产状态采样，`surface_samples()` 决定空间密度与 footprint，`query_plan()` 决定 `wo/wi` 及 proposal。所有实际 query、PDF、积分权重和 seed 都落盘，所以 learning 不需要猜测数据如何产生。

当前公共默认是确定性 stratified `wo` 与均匀立体角 `wi`：反射 family 使用上半球，含透射的 OpenPBR 使用完整球面。各 source split 与 query role 的方位角确定性扰动，不再复用完全相同的方向表。这是可复现基线，不足以自动覆盖极窄高光。

`QueryPlan` 允许共享的 `[light, 3]` 方向表、按 `wo` 的 `[view, light, 3]`，以及按 surface/footprint 与 `wo` 的 `[surface, view, light, 3]`。因此 peak 可以随 `wo` 和局部过滤后的 shading normal 移动，而不需要在公共 collector 中加入材质族分支。MaterialX provider 先用与 reference 相同的 normal texture、`SampleGrad`、trilinear 和 16× anisotropic sampler 求出每个 UV/footprint 的 shading normal，再围绕它的真实反射中心生成 `ncls.materialx-local-normal-peak@1` proposal；实际方向、PDF 与积分权重仍逐 query 落盘。v4 还为每个 `wo` 保存 query role；`--views` 是 train query 数，其他三类由 `--validation-views`、`--test-views`、`--adversarial-views` 明确给出。计数为零只适用于快速 provider/legacy smoke，不能通过正式 E0 gate。

选择 `ncls.e0-peak-grazing-mixture@2` 时，train 与 adversarial role 使用 mixture；validation/test 使用不同方位与种子的固定 uniform probe。这样模型不会在与训练完全相同的离散方向表上被选择或宣称 held-out。source state split 与 query role 仍是两个独立轴，公共 reader 可按任一轴或交集随机访问。`@1` 的分离 `z/方位角` peak 在极窄、旋转各向异性材质上出现高方差，已停止作为当前入口；历史 H5 自带其方向/PDF，并由对应生成提交复现，不能用 `@2` 冒充重生。

## 新增材质族

一个新材质族接入数据层只需要：

1. 锁定 source package/manifest 与 reference implementation；
2. 实现 `ReferenceProvider` 的六个方法；
3. 把 provider 注册到 `src/ncls/data/generator.py` 和 CLI choices；
4. 用最小真实资产导出、验证 HDF5，并增加 provider 集成测试。

不得为新材质修改 `/states`、`/queries`、`/responses`，也不得先把原生材质反演成 LayerStack。若公共查询语义确实不足，应提升整个合同版本，而不是新增隐含 family 字段。

## 重新生成策略

项目不读取旧数据格式，不提供转换、兼容 shim、旧 shard resume 或字段猜测。reference、source package、query proposal 或合同变化后，删除派生数据并从 manifest 锁定输入重新生成。正式响应位于 `data/reference-responses/`，一次性 smoke 与验收输出位于 `artifacts/`，两者都不进入根 Git。

当前 LayerStack 随机游走支持至多八个界面、各向异性粗糙度、层切线旋转、均匀吸收/散射介质和不透明基底。为了让 RGB 共用一次自由飞行采样，有体散射时要求三个通道总消光相同；这是该 v0 reference 的实现约束，不是一般介质规律。
