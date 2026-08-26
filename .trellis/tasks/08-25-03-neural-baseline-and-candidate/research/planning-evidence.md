# 03 Planning Evidence

## 前序产物

- `01`把cosine、LTC、GGX/VNDF、tilted cosine、non-centered GGX、mixture和三态sample语义迁入`shaders/ncls/scattering/`；03只能复用，不复制公式。
- `02`的adequacy gate明确失败，正式训练入口只能使用`entry_id=47ef2013…5a89`。composite corpus覆盖30 states、61,440 targets，v6/v7/v8 provenance为27/2/1；validator从replica重算noise，reader在`.875`回0° base-v5。

## 当前代码事实

- `LearningPipeline`只有`predict_f/training_loss`单阶段接口；`runner.py`在取batch后直接调用两者。03需要最小扩展以传training progress和detached sampler stage，但默认路径必须保持旧pipeline行为。
- `lobe_residual.py`仍是注册占位，production方法未实现；它的lobe-only定义已被父任务淘汰，不能补完后冒充03候选。
- `p1_appearance_loss`已是独立模块，权重固定为`transformed + 0.25 linear + 0.10 energy + 0.15 peak`；tail guard和quality-v2已存在，可复用。
- `MollificationCurriculumStore`已实现strict entry/manifest校验和`.874/.875`边界，但尚未接训练runner。
- `environment.yml`和spec锁定`slangpy==0.43.1`；2026-08-26本机`conda run -n neural-shading python -c "import slangpy"`失败，说明环境声明与实际env尚未同步。

## 历史决策

`trellis mem`回查session`01a0392c-7014-7403-9eec-1b686ad234d8`：

- NVIDIA learned-frame evaluator + analytic two-lobe sampler是必做结构baseline，不是泛泛prior。
- 当前Slang 2024.1.34缺少已验证cooperative-vector加速，不阻止普通Slang运行论文规模网络；paper-scale只能标diagnostic。
- learned frames、sampler KL和解析sample/pdf本身不要求重采；唯一数据风险是directional mollification，已由02用versioned supplement关闭。
- matched 2×2和“自研无优势则部署baseline”属于父任务冻结选择规则。

## 旧资料取舍

- `p1_audit.md`关于旧M2 clamp死区、signed energy/core coverage、tail guard和小MLP预算的机制证据继续复用。
- `p1_v2_plan.md`只继续复用单一Slang/SlangPy与双编译机制；其中由旧run观察值提升的Q1绝对质量门已撤销。复现门改为method correspondence、稳定收敛、数学正确性和parity；03仍把通用viewer实现移交04/05，但必须交付原规模资产。
- `experiment_framework.md`的train-only统计、test只读、paired state bootstrap、成本字段和中文实验登记继续适用；30-state p95只作P1 selection，不能声明全项目长尾性能。
