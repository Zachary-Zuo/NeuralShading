# 新会话连续执行 Prompt

你现在位于 `D:\01_Workspace\NeuralShading`，继续父任务 `.trellis/tasks/08-25-unified-scattering-method`。请严格遵守根 `AGENTS.md`、Trellis workflow 和项目 spec，把这个父任务真正一口气推进到全部子任务提交、归档，最后完成父任务跨层验收、提交和归档。该任务树的 `task.json.meta` 已记录用户对现有父计划边界内 planning refinement、implementation、scoped local commits 和 archive 的连续授权；不要再设置例行 planning approval 或 commit confirmation 停点。

先使用 `trellis-start` / `trellis-continue` 恢复现场，并在涉及历史决策时使用 `trellis-session-insight`。读取父任务的 `prd.md`、`design.md`、`implement.md`、`research/`、`task.json`，核对当前 Git 状态和 task/archive 状态，不要重新执行已经完成并归档的子任务。当前未提交的 `.trellis/tasks/08-25-unified-scattering-method` 和六个 `.trellis/tasks/08-25-0*` 目录是已确认的父/子规划资产，不是无关 dirty files；若它们在 `01` 完成时仍未提交，应在同一次提交确认中把“父子任务规划骨架”和“01 实现”列为两个独立逻辑 commit。不要触碰或提交无关文件 `SmileySans-Oblique.otf`，也不要覆盖用户已有改动。

严格按以下顺序推进，前一项没有通过 gate、提交并归档时不得启动后一项：

1. `.trellis/tasks/08-25-01-reusable-scattering-math`
2. `.trellis/tasks/08-25-02-mollification-data-adequacy`
3. `.trellis/tasks/08-25-03-neural-baseline-and-candidate`
4. `.trellis/tasks/08-25-04-generic-method-bundle-runtime`
5. `.trellis/tasks/08-25-05-viewer-method-deferred-pt`
6. `.trellis/tasks/08-25-06-legacy-method-reset`

对每个尚未完成的子任务执行完整 Trellis 生命周期：

1. 先核对所有前置子任务已经位于 archive，并读取它们的最终产物、spec 更新和验收证据。
2. 保持当前子任务为 `planning`。根据当时代码和前序真实产物做必要研究，更新其 `prd.md`，完整创建或更新 `design.md`、`implement.md`、`implement.jsonl`、`check.jsonl`。不得把父任务的概要清单当作已经完成的子任务详细设计。
3. 运行 requirement convergence、PRD convergence 和 `task.py validate`，把 final planning summary 记录到任务文件/进度报告。若细化没有改变父任务冻结的产品、范围、兼容性或风险决策，直接运行 `task.py start`，不要等待新回复；现有连续授权已经满足 start review。
4. 使用 `trellis-before-dev` 读取项目级和所属 core/data/learning/viewer spec；在第一次测试、构建或训练前按 `dev-environment.md` 判定开发机是完整、仅 GPU 还是静态状态，并报告证据。
5. 完整实现该子任务，不把失败 gate 留给后续任务，不引入兼容 fallback，不修改锁定的 `external/`。所有 Python 命令使用 `conda run -n neural-shading`，Falcor Python 使用 `scripts/run_falcor_python.ps1`，viewer 只使用 `scripts/build_viewer.ps1`。
6. 使用 `trellis-check` 做完整质量检查；同类 bug 若反复出现则使用 `trellis-break-loop`；完成后使用 `trellis-update-spec` 判断并写入应长期保留的架构知识。
7. 按 Trellis Phase 3.4 检查 dirty state、学习提交风格并形成逻辑提交计划；把计划记录到进度报告后直接创建仅包含已识别任务文件的本地 commits，不等待确认。不得 amend、不得 push、不得夹带 `SmileySans-Oblique.otf` 或其他用户文件；出现无法可靠归属的重叠修改时才暂停。
8. 提交完成后使用 `trellis-finish-work` 完成任务记录与归档。确认其确实进入 archive 后，再转入下一子任务 planning。

执行中遵守以下父任务不变量：

- 公共运行语义始终是同一 `prepare/evaluate/sample/pdf` 合同；deferred 和 PT 不得各自维护旁路。
- NVIDIA paper-scale baseline 可作为 `diagnostic` 通过通用 viewer 显示，但不能冒充当前 Slang 标量路径的 realtime；部署比较使用 matched `≤2k MAC` baseline。
- `02` 必须在结果产生前冻结 directional-mollification adequacy 协议；若现有 v5 不足，必须完成版本化新 corpus 后才能进入 `03`。
- `03` 必须忠实建立 NVIDIA baseline，并完成 evaluator × sampler matched 2×2；自研候选没有可信 Pareto 优势时选择 baseline。
- `04/05` 只能使用通用 MethodBundle Slang specialization，不得按 method ID 硬编码。
- `06` 只能在替代链路验收后删除旧路径；物理删除未版本化大数据前先解析并报告精确绝对目标、重建来源和可恢复性，必要时等我执行。

六个子任务全部归档后，回到父任务：

1. 核对六个 child 均已归档及其 commit/provenance 完整。
2. 重新读取父 `prd.md`、`design.md`、`implement.md` 与 core/data/learning/viewer 全部 Quality Check，执行父任务最终跨层验收：数学合同、数据 identity、method quality/sampler correctness、bundle genericity、viewer 两模式、capture/replay、旧路径 reachability、repository policy 和 external cleanliness。
3. 若验收发现需要代码修复，不要把实现塞进父任务；先建立下一个有序 corrective child（从 `07-` 开始）。只要修复不改变冻结范围，该 child 自动继承连续授权，完成 planning、实现、检查、scoped local commits 和归档后重新执行父验收。
4. 父验收全部通过后，更新最终文档/spec和父任务 acceptance checklist；直接创建仅包含已识别父任务文件的本地 commits，随后使用 `trellis-finish-work` 归档父任务。
5. 最终只在父任务确实归档、所有必须验证均有证据且无未处理 gate 时宣布完成，并汇总各子任务 commit、最终方法 identity、corpus identity、MethodBundle identity、viewer 证据、删除内容与仍由我处理的未版本化目录。

不要设置例行人工停点。仅在以下情况暂停：需要改变冻结的产品/范围/兼容性/风险决策；需要用户执行或授权不可恢复的数据删除；需要新的外部权限；存在无法通过安全范围内替代方案解决的真实 blocker；或 dirty changes 无法可靠归属。其余技术细节、planning refinement、任务启动、质量修复、scoped local commits、归档和进入下一子任务都应连续推进。不要为了“继续”而降低数学、架构、数据或验证标准。
