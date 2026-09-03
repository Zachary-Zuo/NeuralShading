# Bug Analysis：Metal catalog 枚举失败与物化过慢

## 1. Root Cause Category

- **Category**：B（跨层合同）+ E（隐式假设）+ D（测试覆盖缺口）。
- **具体原因**：MDL registry把enum authored default保存为`{"name","value"}`，source parameter view已经把它规范化为choice字符串，但method editor metadata又从registry取回对象，违反typed editor ABI。`float2` soft range同样从registry以数组进入只接受共享scalar range的runtime metadata。
- **性能原因**：692个entry被误当成692份重资产；实际只有52个texture sets。旧writer会为每个entry重复写出、哈希、再打开校验百MiB级grid和decoded texture，并在已有catalog命令末尾再次递归统计198.2 GB logical bytes。
- **平台原因**：4路native compile暴露Windows目录`os.replace()`的瞬时`WinError 5`，旧实现默认不同cache key即可无竞争，遗漏了并发进程关闭文件与目录原子发布的时间窗。
- **证据与置信度**：原失败条目`medium_pitted_steel`的`pit_texture_selection`对象值、`Silver_Knurling.texture_scale=[0,0]..[2,2]`、旧partial中单个grid 85–395 MiB、33条前后计时、NTFS `samefile`和692条真实成功输出均为直接运行证据；根因置信度高于99%。

## 2. Why Fixes Failed

1. **只修enum**：解决了第33条症状，但只跑33条仍遗漏第57条才出现的vector range，属于范围不完整；改为692条轻量预检后在13秒内发现。
2. **直接4路并发**：缩短native compile但没有覆盖Windows cache目录发布拒绝；增加窄范围`PermissionError`短退避，并保留provider的canonical cache key与partial清理。
3. **只做hardlink**：若仍让package writer先写大blob、随后deduplicate和严格回读，I/O已经发生；必须在writer落盘时按digest直接link，并把全量重复校验移到entry实际加载。
4. **保留logical_bytes日志**：已有catalog的核心解析只有0.69秒，但日志遍历占12.9秒；删除O(files)统计后整条Conda命令降到4.42秒。

## 3. Prevention Mechanisms

| Priority | Mechanism | Specific Action | Status |
|---|---|---|---|
| P0 | Runtime + test | 全692条typed editor预检先于重工作，错误带entry/name/export ID | DONE |
| P0 | Architecture | enum/vector range只从canonical typed view产生runtime-compatible metadata | DONE |
| P0 | Architecture | 按52个texture sets分组cook，package writer按verified digest硬链接 | DONE |
| P1 | Platform | native compile有界4路并发，Windows发布拒绝做有界短退避 | DONE |
| P1 | Test coverage | enum default与package hardlink strict roundtrip回归 | DONE |
| P1 | Documentation | 更新viewer、unified pipeline与cross-layer guide | DONE |
| P2 | Integration | 交互式快速切换、reset和故障注入仍需人工UI验收 | TODO |

## 4. Systematic Expansion

- **Similar Issues**：其他source family若authoring JSON与typed UI值形态不同，也必须在producer边界规范化；其他大catalog exporter也应区分entry identity与唯一asset identity。
- **Design Improvement**：preflight、content-addressed物化和entry加载时严格验证形成三段边界；catalog canonical identity不因物理存储策略变化。
- **Process Improvement**：新增全量schema扫描测试，性能报告同时给出唯一内容数和logical/physical语义，不再用早期线性ETA判断分组任务。

## 5. Knowledge Capture

- [x] 更新`.trellis/spec/viewer/mdl-reference.md`。
- [x] 更新`.trellis/spec/project/unified-pipeline.md`。
- [x] 更新`.trellis/spec/guides/cross-layer-thinking-guide.md`。
- [x] 增加enum和hardlink unit regressions。
- [x] 生成692条正式catalog并完成Release headless finite smoke。
- [ ] 交互窗口完成搜索、跨texture-set快速切换、typed edit/reset和失败回滚验收。
