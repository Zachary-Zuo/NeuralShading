# 项目架构重置：实施计划

> 状态：已交付，用户授权提交并归档，代码与本机验证完成；Linux 实机检查见 TESTING.md，验收证据见 research/validation.md。

## 1. 实施前收敛

- [x] 已确认 Linux 保留数值 validation，图像 eval 保留统一接口并绑定空实现；已同步全部规划文档。
- [x] PRD 收敛检查：需求来源、范围、验收映射完整，已删除解决的问题和临时重复表述。
- [x] 已呈现最终规划，用户随后回复“开始执行”；已运行 task.py start。
- [x] 进入实施时读 trellis-before-dev 和当前目标层规范；用户本任务明确修正的旧规则同步替换。
- [x] 本次 inline，不创建或使用 implement/check 子代理与 JSONL 调度清单。

## 2. 顺序执行

### A. 固定目录与路径责任

- [x] 实现统一 RunPaths；outputs 按 config/run 聚合，checkpoint、TensorBoard、eval、日志和导出消费同一对象。
- [x] 在 DDP 启动前分配一个 run；新训练与明确 resume 区分，避免 metrics 覆盖和 TensorBoard 混写。
- [x] 更新 .gitignore 和正式默认路径，清理对旧 artifacts 权重/部署包的运行依赖。
- [x] 旧 viewer 图像保持原位置，仅使用 research 笔记定位；不复制、不移动、不生成归档包。

### B. 统一启动与直接方法接口

- [x] 实现导入轻量的 Python bootstrap，集中解释 GPU 参数与 Falcor/CUDA 环境；调用当前 neural-shading 解释器。
- [x] Windows/Linux 单卡和 Linux 自动 DDP 接入同一 train 语法；保留真实物理设备映射与已有 NCCL reducer。
- [x] 替换所有旧启动调用方，删除多卡薄转发和重复启动策略；必要测试/工具共用启动模块。
- [x] 将当前方法的实现按职责整理；删除六个旧 definition 转发 adapter 与对应 hash 层。
- [x] 提取当前使用的 full Metal 共享 helper，再删除退役方法、独占 runtime/shader、工具、配置和测试。

### C. 简化配置与状态

- [x] 一次解析 YAML，默认值和用户覆盖直接传到消费者；移除写死实验标签/数值与反复内部字段重验。
- [x] 合并训练内存状态、checkpoint 保存读取、评估与导出读取；删除 v1/v4 转换、旧 importer 和格式 fallback。
- [x] 区分必要训练状态与来源记录/运行设置，删除完整 plan/源码 hash 相等门禁。
- [x] 删除把 formal/complete/梯度覆盖当作可视化和普通导出门禁的历史逻辑；实际缺失资源/张量错误直接报告。
- [x] 精确续训保存必要 optimizer、phase、RNG 和 query cursor；修正 TensorBoard 回退/续写语义。

### D. 收敛观察与评估

- [x] 两个平台保留同一数值 validation 与图像 hook 调用点；启动装配选择 Windows 本机图像实现或 Linux 空实现。
- [x] Linux 数值 validation 按 YAML 正常执行并保留 DDP 数值汇总；图像空实现直接返回，不提前创建 probe/快照/文件/队列/进程或发起 GPU/collective 操作。
- [x] 删除跨机 visual contracts/spool/claim/worker/collector，并替换其 CLI、测试和配置。
- [x] Windows 图像实现调用 renderer，公共 hook 将返回的当前模型对照图写入该 run TensorBoard；后续替换实现无需改动 engine/method/data。
- [x] YAML 默认 reference_spp=128；可调整取值，renderer 正确达到所配采样数；删除 1024 全等及无必要的双方 spp 大小限制。
- [x] 日志、checkpoint、数值 validation、图像 eval cadence 独立；不为出图保存完整 optimizer snapshot。

### E. 完整收口

- [x] 检查 scripts/tools/configs/tests 的消费者与入口，删除孤立、仅用于旧模型的分支和重复实现。
- [x] README、AGENTS/CLAUDE、repository_policy、architecture、learning、Linux 文档和相关 spec 描述同一架构。
- [x] 更新 TESTING 的真实命令；历史结论标明其范围，不为旧产物可用性保留代码。
- [x] 核对 R1–R7/AC1–AC8 的实现证据，不把模型质量或性能数字增加为任务 hard gate。

## 3. 验证方法与命令

当前会话已判定完整 Windows：RTX 4090、neural-shading 环境、Windows Falcor Python 构建存在。下列检查方案已按本机能力执行，具体通过结果和 Linux 待执行项见 research/validation.md。

- 配置/入口/路径/状态：有意义的 unit 测试覆盖任意合法 spp 透传、设备选择、单一 run、改变日志设置仍能加载、模型结构错误、同拓扑 resume 与 TensorBoard step。用同一调用点分别注入图像实现与空实现，验证前者产生结果、后者不调用 renderer/快照准备、不写图像文件、不改变 RNG；测试行为，不测试空函数具体写法。
- 删除测试：移除旧协议/旧格式/冻结配置 hash 的维护负担；确认旧入口不能被误用为新入口，不额外建立永久兼容拒绝框架。
- Windows 真实小规模检查：统一命令的新训练、短续训、一次本机图像 eval 和新 checkpoint 导出。用小分辨率确认两个不同 YAML spp 被执行；不要求完整正式训练或旧模型复现。
- Linux 目标机：同一 Python 命令的单卡和两卡短 smoke、实际卡号映射、rank0 写出；数值 validation 有指标，图像接口被共同流程调用但无渲染及文件/队列/GPU 副作用。本机没有 Linux 运行环境，命令和期望写入 TESTING 交接。
- 涉及 package/viewer 改动时，运行对应当前 ABI/资源测试及一次 Release overlay 构建；不是每个重构阶段都执行整套部署研究。
- 既有 reference/scattering 的相关回归测试保持数学合同，不以旧 checkpoint 作为 fixture。

当前入口命令：

```powershell
conda run -n neural-shading python -m ncls train 0 --config configs/training/runs/nvidia-layer-stack-smoke.yaml
conda run -n neural-shading python -m pytest tests/unit
.\scripts\build_viewer.ps1
git diff --check
```

Linux 的两卡命令仅改变设备列表为 0,1；若设备编号不同，直接替换列表，不叠加 CUDA_VISIBLE_DEVICES/--gpus/torchrun。验证 config 应使用新架构的短 smoke，不能误跑 Metal 正式 pilot。

## 4. 主要风险与回退点

- 旧 full Metal 与当前 budgeted 存在真实代码复用：先提取再删除，不能靠名字批量移除。
- Python 启动必须先配置原生动态库再导入 Torch/Falcor；同时更新调用方，防止两个入口继续漂移。
- 旧严格 hash gate 的移除不能掩盖真实张量/资产结构不匹配；只在加载边界检查实际结构。
- 输出管理和 TensorBoard resume 必须一起切换，避免只修 JSONL。
- 本任务不操作用户旧成果；回退只涉及本任务代码。不得清空 artifacts、修改 external 或顺带处理用户未跟踪的论文/字体。
