# 项目架构重置与统一训练工作流

## 目标与用户价值

全面整理项目架构、目录生命周期与日常训练流程，为后续重新设计模型、从头训练建立简洁基础。用户只选择训练动作、GPU 和 config；同一 config 的完整实验产出集中保存；任务临时目录可以删除，不影响正式入口。

用户于 2026-09-05 授权创建任务并规划。随后明确要求“开始执行”。代码和本机验证已完成，Linux 实机项待目标机执行；证据见 research/validation.md。

## 实施前盘点（以下位置与行号指改造前）

- 现有 engine 与 Linux 自动 DDP 已有统一实现，但环境准备仍通过平台脚本，文档存在重复 GPU 参数；见 `src/ncls/learning/training/launch.py:164`、`docs/learning.md:20`。
- 默认输出为 `artifacts/training/<config-stem>/checkpoint.pt`，其余文件按 checkpoint 名拼接；见 `src/ncls/cli.py:454`。模型和临时报告混放由 `docs/repository_policy.md:3` 明确规定。
- visual eval 默认关闭，通过文件 spool 和 Windows viewer worker 产生图像；`src/ncls/visual_eval/contracts.py:121` 硬编码恰好 1024 spp。
- 完整 plan 将 execution、hooks 和 YAML 输入文件身份一起用于 resume 校验；见 `src/ncls/learning/training/plan.py:463`、`src/ncls/learning/training/checkpoint_v1.py:230`。
- 旧 checkpoint importer、内部 v4 状态与新 v1 封装并存；README 仍介绍已经退出正式架构的 corpus/HDF5/旧 CLI。

## 需求

### R1：按生命周期整理目录

- 新产生的 `artifacts/` 内容仅承载任务临时产出，正式功能不依赖其中的文件。已有 viewer 视觉证据按用户要求原地保留；其余旧内容由用户自行处理。
- 新训练的权重、TensorBoard、eval 图像、metrics、摘要和部署导出按 config 聚合在固定的持久输出根目录。
- 原始资产、第三方依赖、构建缓存、正式配置、稳定文档和任务记录各有明确归属；清理历史残留与失效路径。

### R2：唯一 Python 训练入口

- Windows/Linux 使用同一 Python 命令语法，日常输入为 `train + GPU 列表 + config`。
- 单 GPU 直接训练，Linux 多 GPU 自动 DDP；平台环境准备与设备映射由内部启动层负责，用户不重复指定 GPU 或手拼 torchrun。
- 本次不扩展原生 Windows 多 GPU 训练。

### R3：清理过度防御与无效抽象

- spp、cadence 等可调值由 YAML 决定，修改正常取值无需改 Python、协议或同步修改测试常量。
- 全面审计重复内部校验、硬编码单一研究配置、重复身份/版本封装、无实际消费者的抽象与历史 workaround；删除没有真实执行需要的限制。
- 配置错误在输入边界给出一次清楚错误；内部使用已解析的数据，不逐层重新验证同一字段。
- 必须能捕获真实的模型张量不匹配、资源缺失和 DDP 次序错误；不得把这些问题静默变成成功。

### R4：训练与观测分离

- Windows/Linux 保留同一训练 engine、数值 validation 和图像 eval 调用接口；不得形成两条平台专用训练路径。
- Linux 数值 validation 保留，按同一 YAML 配置调度；多卡仍汇总数值指标并由 rank 0 写出。
- Linux 图像 eval 当前绑定空实现，接口被调用时直接返回，不渲染、不生成快照或文件、不消耗 RNG、不创建队列/后台进程、不参与额外 GPU/分布式操作。
- Windows 在同一个图像 eval 接口中执行本机渲染并自动写 TensorBoard；默认 reference 为 128 spp，实际值来自 YAML。后续 Linux 接入只替换实现，不修改 engine、方法或训练调用点。
- 删除旧跨机 visual-eval 队列、worker/collector 协议及多卡图像评估设施；保留的统一接口不是旧协议兼容层。
- 数值/图像评估、checkpoint 保存与日志 cadence 互相独立；可视化不消耗训练 RNG 或改变训练数据游标。

### R5：区分训练状态与运行设置

- 保留加载/恢复模型所必需的状态与结构信息；路径、GPU 物理编号、日志和图像设置不因整个配置文件 hash 不同而无条件阻止使用权重。
- 运行环境、代码版本和配置来源用于记录与解释，避免把溯源信息全部升级成运行门禁。
- 新架构下重复启动与显式续训有确定行为，权重不静默覆盖，TensorBoard 不混入另一场实验。

### R6：彻底移除历史兼容

- 不迁移旧权重、训练日志、optimizer、run 或部署包，不提供旧 checkpoint importer、旧 CLI alias、格式 converter 或兼容 fallback。
- 后续模型按新架构从头训练。当前方法源码按新接口整理；本次不设计新模型或以旧训练质量作为验收目标。
- 既有 source 原生语义、online reference 与公共 scattering 行为保留，不以清理为由引入磁盘训练 batch 或改变 GT。

### R7：只保留旧视觉证据

- 既有 viewer PNG/EXR、相关截图与已有解释元数据原地保留在 `artifacts/`，用于后续分析缺失细节。
- 任务研究笔记记录主要证据路径和已有材质/左右侧/模式信息即可；不移动、复制、打包或另建归档系统。
- 不要求保留或重新加载旧权重、部署包和 runtime 才能理解图像；不把历史视觉证据变成新训练的运行依赖。

## 验收标准

以下 AC1–AC8 均为需求交付，来源为用户 2026-09-05 的初始需求及本次范围修正；括号中标注对应需求。真实张量/资源/并发正确性同时来自现有公共运行合同，不引入 observed quality/time/memory 门槛。

- [x] AC1（R1/R7）：旧视觉证据保持原位置且未被本任务修改或删除；研究笔记可定位主要图片。新流程不依赖旧训练输出。
- [ ] AC2（R2）：同一 Python 命令完成 Windows 单卡、Linux 单卡/自动 DDP；只输入 GPU 与 config 即可获得固定输出目录。
- [x] AC3（R1/R5）：新 run 的权重、曲线、图像和导出集中保存；重复运行与续训不会混写或误覆盖。
- [x] AC4（R3/R4）：Windows 图像显示于 TensorBoard；仅改 YAML 即可使用 128 和另一个正整数 spp；不存在写死恰好某个 spp 的协议和测试。
- [ ] AC5（R4）：Windows/Linux 通过相同调用点执行数值 validation 和图像 eval hook；Linux 数值 validation 正常产生指标，图像接口为空操作且无上述副作用。替换图像实现不需要修改训练路径；旧跨机 visual job 机制退出正式代码、配置、测试和文档。
- [x] AC6（R3/R5）：修改目录、日志或图像设置不触发全 plan hash 门禁；模型结构不匹配仍有明确错误。检查按责任边界执行一次。
- [x] AC7（R6）：旧 checkpoint 格式、旧入口与转换链无正式消费者；旧格式不被自动兼容读取。
- [x] AC8（R1/R3/R6）：README、AGENTS/CLAUDE、相关 spec、配置示例和测试共同描述新架构；清理清单逐项落实，不以新增 facade 收尾。

AC2、AC5 的实现与公共行为测试已完成；保留未勾选状态以显式交接 Linux 实机验证，不能以 Windows 结果代替。

## 范围之外

- 旧成果迁移、旧模型复现和旧格式兼容。
- Linux 图像渲染实现、跨机 viewer worker、多 GPU 图像 eval、专用 eval GPU；本次仅保留统一图像接口及 Linux 空实现。
- 新模型设计、正式重训和模型质量研究结论。
- 移动、复制、归档旧视觉证据，或自动删除用户现有 `artifacts/`；旧内容由用户自行处理。
- 原生 Windows 多 GPU、第三方上游源码修改。
