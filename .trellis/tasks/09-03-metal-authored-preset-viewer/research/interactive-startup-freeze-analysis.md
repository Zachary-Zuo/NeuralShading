# Bug Analysis：Metal viewer 启动后持续无响应

## 1. Root Cause Category

- **Category**：B（跨层合同）+ D（测试覆盖缺口）+ E（隐式假设）。
- **Specific Cause**：renderer为了避免Windows TDR把quality-first neural evaluator拆成8×8 tile，却在同一个`onFrameRender`中循环全部tile，并在每个tile后同步`submit(true)`。默认1600×900的右panel形成11,250个串行workgroup；此前320×240的38,400像素已实测约29秒，因此默认分辨率会占住render/UI线程数分钟。
- **伴随原因**：默认deferred仍预编译未使用的package path tracer；linked初始事务重复安装同一reference并重复编译相同editor state；`ReferenceSource`候选按值复制含692条parameter view的17 MiB catalog。

## 2. Bayesian Evidence

| Hypothesis | Prior | Discriminating evidence | Posterior |
|---|---:|---|---:|
| 单frame同步排空neural tile | 50% | 代码有双层全图循环和每tile同步submit；旧进程第12秒后CPU停滞并持续`Responding=False` | 96% |
| catalog解析/文件哈希单独导致卡死 | 30% | 前10秒CPU仍推进且窗口响应；已有catalog Python结构读取约0.69秒 | 2% |
| MDL/reference shader首次编译导致永久卡死 | 20% | 新版仍做首次编译，只出现约2秒瞬时`Responding=False`后恢复 | 2% |

最终根因置信度高于99%：完整tile lattice的单frame同步所有权错误解释持续无响应；首次编译只解释短暂启动峰值。

## 3. Why Fixes Failed

1. **只做8×8 tile**：限制了单dispatch watchdog时长，却没有限制单UI frame总工作量，是把GPU安全边界误当成交互调度边界。
2. **headless finite smoke**：证明完整工作最终可结束及数值finite，但headless允许同步排空，无法覆盖Windows消息泵响应性。
3. **静态测试固化旧实现**：测试要求`executePackageTiles`包含全循环和`submit(true)`，却没有区分interactive与headless，反而把卡死路径当成正确合同。

## 4. Prevention Mechanisms

| Priority | Mechanism | Specific Action | Status |
|---|---|---|---|
| P0 | Architecture | interactive每frame至多一个workgroup；16×到1×coarse-to-fine，state变化可取消并重启 | DONE |
| P0 | Semantic boundary | headless/capture固定stride=1且完成全网格后导出 | DONE |
| P0 | Test coverage | 静态断言interactive单tile入口无循环/同步submit，headless保留精确全tile入口 | DONE |
| P1 | Runtime probe | Windows默认1600×900连续30秒采样`Responding`、CPU与私有内存 | DONE |
| P1 | Resource lifetime | catalog使用共享不可变所有权；program pass按slot mode懒创建；跳过重复reference/editor安装 | DONE |
| P2 | Interactive E2E | 人工继续覆盖搜索、跨texture-set切换、edit/reset和失败rollback | TODO |

## 5. Systematic Expansion

- **Similar Issues**：package path tracing若用于交互，也不能因单tile满足TDR就把完整像素×spp工作塞入一个frame；其他大catalog候选同样不能按值复制完整schema树。
- **Design Improvement**：明确区分workgroup安全上限、frame时间预算和capture完成目标；三者分别由tile尺寸、跨frame cursor和headless completion gate拥有。
- **Process Improvement**：任何“viewer headless通过”的结论都不能替代消息泵响应性探针。质量优先模型进入交互端时必须同时报告首次反馈、持续响应和exact completion语义。

## 6. Knowledge Capture

- [x] 更新`viewer/mdl-reference.md`的交互time-slicing、共享catalog与lazy pass合同。
- [x] 更新`project/unified-pipeline.md`和cross-layer thinking guide。
- [x] 更新`viewer/capture-harness.md`，冻结headless stride=1边界。
- [x] 更新unit静态回归并保存真实Windows进程证据。
- [ ] 完成交互控件与failure rollback人工验收。
