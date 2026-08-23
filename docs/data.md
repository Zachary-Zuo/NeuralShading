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
  --views 16 --lights 128 --spatial-samples 16 `
  --adaptive --batch-samples 256 `
  --min-samples 512 --max-samples 16384
```

`--families`、`--local-states` 和 Monte Carlo 参数只影响 LayerStack；`--spatial-samples` 与 `--footprint-width` 对当前 MaterialX provider 生效。MERL、OpenPBR 和 MaterialX 默认导出各自 manifest 中的全部已复现资产。

这条命令的方向数可以用于工程 smoke，但不是最终训练密度的默认结论。正式数据生成前应先完成 peak/掠射角监督审计，再冻结每个 provider 的 query proposal 和 state distribution。

E0 的定向覆盖诊断使用版本化的 `ncls.e0-peak-grazing-mixture@1`。它对每个 `wo` 分别生成 uniform、与镜面反射峰对齐的多尺度 peak，以及掠射角分量；完整球面的 OpenPBR 还包含透射侧 peak。每个落盘 `wi` 都保存 mixture 的真实 PDF，并使用 `1 / (N p(wi))` 作为 Monte Carlo 积分权重：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli data collect-reference `
  --provider openpbr --material-id <transmission-asset-id> `
  --query-profile ncls.e0-peak-grazing-mixture@1 `
  --views 8 --lights 256 `
  --output artifacts\research\learning-goal\e0\openpbr-mixture-probe.h5
```

该 profile 先用于 E0 probe 和 proposal 对比，不自动成为后续正式训练分布。只有 audit 证明覆盖、估计方差与加权积分均满足冻结 gate 后，才会用明确配置生成正式 H5。

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
  --footprint-width 0.000244140625 `
  --output data\reference-responses\materialx-walnut.h5
```

快速 LayerStack smoke：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli data collect-reference `
  --provider layer-stack `
  --families 1 --local-states 1 --views 1 --lights 16 `
  --samples-per-replica 16 --max-depth 16 `
  --output data\reference-responses\layer-stack-smoke.h5
```

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
    train_groups = dataset.group_indices("train")
    batch = dataset.group_batch(train_groups[:32])
    print(batch["wo"].shape)
    print(batch["wi"].shape)
    print(batch["mean"].shape)
    print(dataset.state_payload(0))
```

response-only learning 使用 `ReferenceQueryStore`。当前 LayerStack baseline 需要原生 feature，因此显式使用 `LayerStackReferenceStore`；公共 store 不会出现 `interface_kinds`、层数或 OpenPBR 参数等 family-specific 字段。

## 采样职责

公共 collector 不替 provider 决定“恰当分布”。每个 provider 的 `source_states()` 决定原生参数/资产状态采样，`surface_samples()` 决定空间密度与 footprint，`query_plan()` 决定 `wo/wi` 及 proposal。所有实际 query、PDF、积分权重和 seed 都落盘，所以 learning 不需要猜测数据如何产生。

当前公共默认是确定性 stratified `wo` 与均匀立体角 `wi`：反射 family 使用上半球，含透射的 OpenPBR 使用完整球面。train、validation、test 的方位角按 split 确定性扰动，不再复用完全相同的方向表。这是可复现基线，不足以自动覆盖极窄高光。

`QueryPlan` 允许共享的 `[light, 3]` 方向表，也允许按 `wo` 提供 `[view, light, 3]`，内部统一为后者。因此 peak 能随 `wo` 移动，而不需要在公共 collector 中加入材质族或网络分支。需要注意，当前 HDF5 的 split 仍属于 source state；“同一材质上的未见 query”角色不能只靠 state split 表达。E1 正式数据开始前必须把 train、validation、held-out test 和 adversarial probe 的 query 角色做成显式、可验证且不泄漏的合同，不能把 split 方位扰动冒充完整的 held-out query 设计。

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
