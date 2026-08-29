# Metal formal evaluation 与成本报告

## 目标

在Linux long training checkpoint经用户首轮效果审阅并另行批准后，冻结并执行Metal-v1的质量、泛化、编辑、filtering和效率评测，产生可复现的matched报告与置信区间。该任务已从`08-30-vmaterial-metal-neural-system`父树解除，不会在long run后自动启动。

## 激活门

- 当前保持独立`planning`；
- 只有用户审阅Linux checkpoint、训练曲线和代表性效果后明确批准formal scope，才能更新本任务合同并start；
- 若用户选择追加训练或修改方法，先产生新checkpoint identity，再决定是否formal。

## 显式依赖

- 依赖已完成的`08-30-vmaterial-metal-neural-system`交付合同；
- 依赖Linux single-GPU long run产生且经用户选定的checkpoint、Package@2和review manifest；
- 具体formal matrix、预算与checkpoint selection必须在用户批准后重新冻结；
- 输出可以成为未来compact/ablation baseline，但不反向阻塞已完成parent。

## 需求

- formal前冻结source locator、split、query、method/profile/precision、seed strategy、budget、checkpoint selection和metric schema；
- 独立报告`G_asset/G_metal/G_finish/G_pair/G_param/G_recipe`，遵守适用域；
- 质量包含local transformed/linear、energy、peak/top support、reciprocity、semantic channel/normal/mask和continuous parameter/footprint sweeps；
- matched controls包含core/direct分账、encoder-only/refinement/direct codec roles、current/optimized source和conventional texture deployment；
- source-state层bootstrap不少于1,000次并给出CI；
- 成本拆分compile/cook/edit/load/upload、`C_prepare/C_eval`、viewer、`B_shared/B_asset`和delivery/resident；
- observed结果只报告与Pareto分析，不在执行中制造hard gate或自动聚合成功口径。

## 不在范围

- 根据结果自动修改模型、seed、预算或重复运行直到过门；
- 修改或补做matched sampler实现；
- compact/ablation训练本身；
- 任意域外组合质量承诺。

## 验收标准

- [ ] [实现正确性｜research protocol] formal manifest在结果产生前冻结且所有run identity可恢复；
- [ ] [需求交付｜parent R3/R4] 六类泛化、typed sweep、footprint boundary和semantic metrics完整；
- [ ] [统计正确性｜experiment framework] matched comparison使用source-state bootstrap CI，样本单位无texture/export泄漏；
- [ ] [需求交付｜parent R7] current/optimized/conventional/neural的time/storage/capability matched且层次标注清楚；
- [ ] [研究交付｜report-only] 低quality或低efficiency作为结果登记，不触发事后改门；
- [ ] [需求交付｜parent research contract] report、raw metric rows、configs、checkpoint/package hashes均写入`artifacts/`并登记experiment log。

## 阻塞问题

当前阻塞是尚无经用户审阅并批准进入formal的Linux long checkpoint。这是预期激活门，不允许自动越过。效率数值门槛与聚合方式仍等待formal事实后决定。
