# 02 Directional Mollification 数据充分性实施计划

## 0. Planning Gate

- [x] `01` 已归档，公共方向与 measure 语义已读取。
- [x] NVIDIA 官方 paper/supplemental 已核对：mollify `ωo`，10°→0° cosine decay，前 20,000 iterations，每 target 256 cone samples。
- [x] 已只读检查 P1 v5 的实际 group：train 有 48/64/96 个 `wo`、每组 512/1024/2048 个独立 `wi`；dense 只有 4×8192，v5 无 cone/group 字段。
- [x] PRD 已做 requirement convergence 与 lossless convergence：代表覆盖、查询前冻结、二选一决定、失败路径完整性和 03 data entry 均有可执行 gate，无 blocking open question。
- [x] 本细化未改变父任务产品、范围、兼容性或风险边界；连续授权允许 `task.py start`。

Final planning summary：先锁定六个 state、四个现有 dense `wo`、四类 `wi`、论文四个正半径 level、256 jitter、双 replica reference budget和 support/noise/numeric/repeat 门，再允许 matched query。全过才复用 v5；任一失败就生成覆盖 30 state、引用 base v5 的 versioned mollification supplement，并发布 03 唯一 data entry。旧 v5 不改不删。

## 1. Protocol、Schema 与 Freeze

- [x] 新增 frozen protocol config/schema/parser，所有默认值写入配置而非 CLI 自由参数。
- [x] 实现 state/dense anchor selector 和 canonical anchor lock；绑定 protocol/base/shard hashes与冻结时间。
- [x] CLI `mollification-freeze` 只读旧 corpus，测试 exact vectors、tie-break、tamper 和缺失 role。
- [x] 在任何 fresh query 前先生成正式 anchor lock，记录其 SHA-256 到本计划和运行报告。

## 2. Matched Adequacy Audit

- [x] 实现 upper-cap deterministic 256-jitter generator、旧 v5 6D kNN reconstruction、support/noise/numeric/repeat metrics。
- [x] 复用 LayerStack provider/reference 执行 fixed matched blocks；raw output 与 summary 均绑定 lock/hash，不覆盖合法不同 identity。
- [x] 运行两次正式 audit，保存 report 与 binary decision；不得在结果后改 protocol/阈值。

## 3. Conditional Supplement

- [x] 若 decision 为 `reuse-v5`，生成只指向原 corpus 的 frozen training data entry，并证明 reconstruction repeat（本轮 binary decision 未选择此分支）。
- [x] 若 decision 为 supplement，新增 `mollified-reference-shard/corpus` schema、writer/reader/validator 和语义 hash。
- [x] 为 30 个 selection state 确定性锁定 8×64 train anchors、四个 curriculum level和 reference seed/budget。
- [x] 完整采集、逐 state 续采并验证全部 shard；发布唯一 supplement corpus identity。
- [x] 新增 learning curriculum reader 和 frozen training data entry；0° 明确回到 base v5。

## 4. Validation

```powershell
conda run -n neural-shading python -m pytest tests/unit -q -k "mollification or corpus or dataset or measure"
.\scripts\run_falcor_python.ps1 -m pytest tests/gpu tests/integration/reference -q -k "mollification or reference or dataset"
conda run -n neural-shading python -m ncls data validate-corpus artifacts/corpus/layer-stack-p1-v1.json
conda run -n neural-shading python -m ncls data validate-mollification <published-manifest>
```

- [x] 完整 unit/GPU gate 与 `git diff --check` 通过：unit `106 passed`；Falcor targeted `8 passed, 24 deselected`。
- [x] Falcor、pbrt-v4、OpenPBR、openpbr-bsdf、GLM、MaterialX 均保持锁定提交且干净。
- [x] artifacts/data/repository policy 审计通过，HDF5 与单次 report 均位于 ignored data/artifacts，未进入 Git 变更集。

## 5. Quality、Spec 与提交

- [x] 使用 `trellis-check` 审计协议先于结果、measure、hash、reader data flow、no fallback和测试独立性；reviewer 自修 schema 漂移、统计摘要重算、循环 re-export 与 tamper 覆盖。
- [x] 使用 `trellis-update-spec` 固化 directional-mollification supplement 与 data-entry 所有权；长期合同已写入 `.trellis/spec/data/reference-and-corpus.md`。
- [ ] 记录 dirty path 归属与逻辑提交计划，创建 scoped local commits；排除 `SmileySans-Oblique.otf`，不 amend、不 push。
- [ ] `trellis-finish-work` 归档并确认 archive/commit provenance 后才进入 `03`。

## 6. Rollback

- protocol lock 一旦 fresh query 开始即不可改；发现设计问题时版本号升级并保留旧失败证据。
- audit 失败不视为代码失败；只触发已冻结 supplement 路径。
- supplement 未完整验证前不发布 data entry，`03` 保持阻塞。

## 7. 运行记录

- 正式 protocol SHA-256：`7160bd6f210038a6d2f39cac3ec513287b0cacab3ffb94423737c8e4d0cbbec2`。
- 正式 matched anchor lock SHA-256：`d76e532c180e1988485a6bbb3809e4626bd693658918bbd0ce1a40ad9b2757c8`；base corpus ID：`0513d0c837b109f74cbf6fd4f811e05c6bc68c02226bd6d443f3225ef5dd64b7`。
- 首轮 audit `a/b` report SHA-256 均为 `22cee7e4c42b70a16e2d213dbad92719ed1413f86a8f96f7b93fb135438fc963`，决定为 `use-mollification-supplement-v1`；support 与 numeric 已明确失败。
- 首轮 supplement 在 512 paths/replica 时暴露 relative-SE denominator 实现偏离既有 reference 合同：实现只用 absolute floor，未复用 `max(abs(mean), 0.005*group_peak)`。这不是阈值变化；修复后由共享 helper 同时服务 audit 与 supplement。
- 失败采集目录 `D:\01_Workspace\NeuralShading\data\reference-responses\layer-stack-p1-mollification-v1` 含 26 个未发布 HDF5，共 `3,336,368` bytes；均可由 protocol 与 supplement lock 重建。本任务保留它们作为失败证据，不删除、不覆盖。修正语义的新采集使用隔离目录，最终 manifest 不得引用此失败目录。
- 修正 denominator 后的 `reference-se-v1` 仍在 512 cap 失败：`4ebd925…` view 2 / level 0，p95 `0.152108`、worst `0.365159`。按平方反比需要 `3291` paths，向上取二次幂得到 v2 cap `4096`；新增独立 budget plan 与 collection lock，threshold、jitter、anchor、level均不变。
- `reference-se-v2` 在 4096 cap 仍由 `4ebd925…` view 4 / level 0 以 p95 `0.0689937` 失败，worst `0.113843` 已过 hard gate。失败目录含 26 个未发布 HDF5、`3,336,478` bytes，保留不覆盖。相同推导得到 `5416→8192`，因此新增 v3 budget plan 与 collection lock。
- `reference-se-v3` 在 8192 cap 时 p95 已过门，但 `179606…` view 2 / level 0 的 worst `0.386128` 超过 hard `0.25`。失败目录含 27 个未发布 HDF5、`3,478,117` bytes，保留不覆盖。对实际失败 gate 应用平方反比得到 `19543→32768`，因此新增 v4 budget plan 与 collection lock。
- `reference-se-v4` 在 32768 cap 时同一 target 的 p95 为 `0.0307`、worst `0.319926`，仍只差 hard gate。失败目录含 27 个未发布 HDF5、`3,478,117` bytes，保留不覆盖。相同 hard-gate 推导得到 `53663→65536`，因此新增 v5 budget plan 与 collection lock。
- `reference-se-v5` 的 65536 单-dispatch结果与 v4 在报告精度上完全相同；单 target 32768/65536差分诊断触发 `DXGI_ERROR_DEVICE_REMOVED`，证明继续增加 full-dispatch cap不成立。v6 cap保持65536，改为 GPU 256-sample batch + 连续 sample offset + CPU float64 Welford；accumulator实现hash与 reference implementation hash必须在新 collection lock 中冻结。
- `reference-se-v6` 的 batched accumulator消除了TDR与float32长累加停滞，但 `179606…` view 4 / level 0 在65536 cap仍以 p95 `0.0844163` 失败，worst `0.151923` 已过。失败目录含27个未发布HDF5、`3,478,119` bytes，保留不覆盖。相同p95推导得到 `129727→131072`，因此新增v7 plan/lock。
- v7 plan SHA-256 为 `18652be10120ca550d951003247e98b9cdd49f609f1c15dabc17b32ae005ded4`，collection lock v2 SHA-256 为 `ad9dbcfb12fa6fb5bd06f6fe9724b75d1b9d309e8363757ff353202800a7f1fa`。它复用 v6 的27个 shard；`bd6de2…/6fff05…` 通过并各自写入 v7，`179606…` 在 view 6 / level 0 以 p95 `0.107237`、max `0.19514` 失败，正式 run 用时 `731.8s`，未发布 manifest。
- v8 plan SHA-256 为 `d430305f92dbf619d51ee3fa6634fc05f900437131ccf48efd2fd56b9e64abfd`，collection lock v3 SHA-256 为 `56d4434b3cb29c2e01756d77d9cde5b0601148156a4ee224527ec0e5ed59423a`。lock 精确复用29个 shard，仅采 `179606…`；新 shard dataset ID 为 `305e7d8cda35172fb241deb13aca7a2666b88ecda0c4602ed74d3cd0314faf02`，p95/max=`0.0588824/0.234875`，run用时 `1895.1s`。
- 正式 composite corpus ID 为 `f6931474890ab7642f244b84df2736e2a5fc1f9e169b5f7a620494184d99e4f3`；training data entry ID 为 `47ef20138007703f2d1b644bcb4ca4b084001da4ec975f1b712587d3e7e35a89`，variant=`base-v5-plus-mollification-v1`。
- composite manifest 覆盖 `30 states / 61,440 targets / 86,283,124,736 combined reference samples`，全体 shard 最大 p95=`0.0595588`、最大 relative SE=`0.249172`；预算 provenance 分布为 v6/v7/v8=`27/2/1`。manifest 文件 SHA-256=`0ceaa14457a345109dac58ba318841a9a7321b17744048348f6b7a490f808d4a`，data entry 文件 SHA-256=`0ec20f018fd8da194b4f3eac58a3c5f9a96a646035351ae90666f2bdafd7c878`。
