# 架构重置：源码审计与删除清单

## 1. 范围解释

本次按用户 2026-09-05 的修正执行：重新训练、没有旧成果迁移、没有旧格式兼容；Linux 保留数值 validation，图像 eval 保留统一接口但当前为空操作；viewer 视觉证据原地保留在 artifacts。这里只记录证据与改造依据，不把旧实现规定当成新需求。

## 2. 实施前确认的结构性问题（路径和行号为当时状态）

| 位置 | 当前问题 | 规划处理 |
|---|---|---|
| `src/ncls/__main__.py:3`、`src/ncls/cli.py:9`、`scripts/run_falcor_python.ps1`、`scripts/run_falcor_python.sh` | Python 命令先导入大块训练实现，平台脚本承担环境准备；shell 与 Python 各有 DDP 入口 | 轻量 Python bootstrap 在 native runtime 初始化前准备环境并启动当前解释器的 worker；统一设备参数 |
| `src/ncls/cli.py:158`、`:454` | 路径以 checkpoint 文件名拼接；默认进入 artifacts；新训练覆写 JSONL，同时复用 TensorBoard 目录 | 一个 run 路径对象，统一持久输出 |
| `src/ncls/learning/methods/contracts.py:106`、`:117`、`:278` | 六个 `_Definition*` 对象转发旧 MethodDefinition；facet hash 都由整份 definition hash 再加标签派生 | 方法直接实现新接口；移除 adapter 转发层和没有独立含义的 facet identity |
| `src/ncls/learning/training/checkpoint.py:18`、`checkpoint_v1.py:230` | 外部 v1、内部 v4、runner 和 snapshot 多次构造与校验同一内容 | 一份训练状态、一套保存读取代码；不再转成旧格式 |
| `src/ncls/learning/training/evaluation_snapshot.py:162`、`legacy_checkpoint.py:17` | 读取文件和 hash 后再调用 reader 重读；按 v4/v1 分流 | 删除旧分流、旧 importer、重复读取与重复校验 |
| `src/ncls/learning/deployment_snapshot.py:13` | 因 evaluation 全 implementation hash 锁定，再增加一套部署专用读取流程 | 从统一新 checkpoint 读取模型；训练/部署代码来源仅作记录，结构与实际 resource ABI 决定可用性 |
| `src/ncls/learning/training/plan.py:463`、`checkpoint_v1.py:233` | execution、hooks、所有 YAML 文件及实现身份一起进入严格 resume gate | 运行设置与训练状态分开；不对完整配置及源码做全等门禁 |
| `src/ncls/visual_eval/contracts.py:121`、`:135` | 写死 reference=1024；强制 neural spp 不得超过 reference，缺少算法上的必要性 | 删除跨机请求协议；本机 renderer 接收独立 YAML 参数，只检查执行需要的基本范围 |
| `src/ncls/learning/training/hooks/visual_eval.py:177`、`engine.py:1719` | 图像 cadence 依赖周期 checkpoint 提交，interval 配置不能独立决定执行时刻 | 独立观察 hook；不为出图序列化全 optimizer checkpoint |
| `src/ncls/visual_eval/spool.py`、`worker.py`、`collector.py` | 跨机工作队列和状态协议在 Linux 不渲染图像后失去产品用途 | 删除旧协议；提取 Windows renderer 代码，两个平台通过同一进程内接口调用，Linux 实现直接返回 |
| `src/ncls/learning/training/config.py:266`、`:268` | 数值 validation 必须正数配置，selection 强制 tail_guard | 支持明确配置的评估/保存策略；两个平台保留同一数值 validation 调度，Linux DDP 保留数值汇总 |
| `src/ncls/learning/training/readiness.py:86` | phase、run_class、完整梯度证据被升级成导出/预览准入，连初始化图也有 step 门槛 | 研究完成度与梯度诊断用于报告；可加载的当前模型可预览，compiler 只要求实际所需状态 |
| `tests/unit/test_training_plan.py:72` | 测试硬编码完整真实 YAML 的 hash；正常配置改变也要改测试数字 | 改为测试解析与组合行为、状态恢复、实际边界；删掉“当前配置必须不变”式测试 |
| `README.md:5`、`references/README.md:28` | 仍宣传已退出正式实现的 corpus/HDF5 与旧 CLI | 改为 online reference 和新训练入口；同步 AGENTS/CLAUDE、相关 spec 和 TESTING |

## 3. 实施前的历史方法与共享实现

产品 registry 当前只注册 `metal_budgeted` 与 `nvidia`，见 `learning/methods/registry.py:9`。但历史 full Metal 仍有 method、model、asset cook、runtime、layout、shader、工具及测试整条链。

不能靠批量匹配文件名删除，因为当前方法仍有真实复用：

- `learning/methods/metal_budgeted.py:39` 从旧 `metal_runtime.py` 导入 `fake_quantize_fp16_ste`。
- `learning/source_adapters.py:917` 的当前 adapter 继承 `MetalFusedMdlSourceAdapter`。
- proposal 数学及其 GPU 测试仍使用带历史名的公共 shader/layout。

实施时先把当前消费者必需的量化、MDL 数据适配和 proposal 数学移入明确的共享模块，再删除历史方法及其独占测试/配置/生成器。当前 Nvidia 与 Metal 源码只调整架构和接口，不趁机重新设计模型。

## 4. 过度防御的判定

删除的对象包括：固定实验数值门禁、重复字段重验、内部数据反复转字典/冻结/解冻、全源码 hash 相等才允许加载、旧格式转换、空转发适配器、无消费者协议和掩盖错误的兜底。

检查应放在实际责任边界：YAML 入口处理类型/拼写/基本范围，checkpoint restore 处理当前张量结构，GPU/跨语言资源绑定处理真实布局，DDP 处理设备映射与 collective 次序。输入已经解析后，下游不重做相同检查。

不是按关键词删除所有 `compatibility` 或 `fallback`：CUDA 12.8 driver compatibility library 服务于当前部署环境；DDS header 是实际文件格式；零长度法线等数学退化处理不属于旧权重兼容。项目自有旧 checkpoint/CLI/版本桥接则全部删除。

## 5. 视觉证据原地清单

2026-09-05 本机只读盘点：

- 152 份 `ncls.viewer-capture` JSON。
- 关联 563 个现存 PNG/EXR，文件长度合计 589,303,024 bytes。
- 另有 13 张 diagnostics 放大/overlay 图与 23 张 viewer 截图，合计 36 张。
- artifacts 其余 516 张 PNG 是素材筛选缩略图，不把它们误报为 viewer 模型对照。
- 58 条 manifest 文件引用当前不在对应位置；主要为 replay 元数据引用。实施不重渲染、不修复旧包、不以旧证据完整性阻塞架构清理。

原始盘点在 `../scratch/visual-evidence-inventory.json`，临时脚本也位于 scratch；这不是新产品的 schema 或维护接口。

后续分析优先查看：

| 现有目录 | 用途 |
|---|---|
| `artifacts/viewer/metal-viewer-refresh-lighting/smoke/` | 最近的双侧 PT 输出、差值、camera/lighting/slot 元数据；本组只有 1 spp，不能把噪声当细节误差 |
| `artifacts/viewer/metal-viewer-refresh-lighting/current/ui/` | 近期 viewer 显示与模式切换截图 |
| `artifacts/viewer/metal-budgeted-ddp5-wrap-1d5f813-step2048/` | 当前预算方法的历史 Windows 对照与交互截图 |
| `artifacts/metal-root-fix/` | Metal 修正前后的旧视觉证据 |
| `artifacts/nvidia-faithful/` | Nvidia 历史神经材质对照 |
| `artifacts/diagnostics/pt-salt-pepper-noise/` | 局部放大与火花/噪声诊断，避免与表示误差混淆 |
| `artifacts/training-architecture-cleanup/` | 旧 training diagnostic；既有记录指出部分 difference PNG 显示异常，EXR 另行判断 |

这些路径只是历史观察材料，不作为新训练输入或旧 runtime 加载入口。用户已明确不移动、不复制、不另建归档目录。

## 6. 实施状态

用户已确认全部范围：Linux 数值 validation 保留；图像 eval 通过同一接口调用空实现，后续接入无需修改训练路径。代码、配置和文档已切换到新架构，检查结果见 validation.md。当前方法的必要 helper 已并入自身目录；旧 proposal 链只有退役方法/测试消费者，已整体删除。旧成果没有移动、复制或删除。
