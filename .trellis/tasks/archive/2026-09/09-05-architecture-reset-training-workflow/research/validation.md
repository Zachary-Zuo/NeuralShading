# 架构重置实施与验收记录

2026-09-05：代码、配置、当前文档和测试已切换到新架构，经用户确认提交并归档。没有迁移、复制、删除或重新加载旧训练成果；没有进行正式模型重训。Linux 实机检查待在目标机器执行，命令见根目录 TESTING.md。

## 已交付的结构

- `python -m ncls train GPU_LIST --config YAML` 是统一入口。Python bootstrap 先配置当前环境与物理 GPU，再启动单卡或 Linux torchrun；DDP 启动前只创建一个 run。
- `RunPaths` 将新 checkpoint、TensorBoard、eval、导出和日志放入 `outputs/<config-stem>/<run-id>/`。新启动隔离，显式 resume 回到原 run；JSONL 回退和 TensorBoard purge 同步处理。
- 当前方法直接实现 `Method`，分别位于 `learning/methods/nvidia/` 与 `metal/`。一份 TrainingCheckpoint 承载当前状态，训练、验证、预览和导出共用读取路径。
- 删除旧格式转换/reader、六个转发 adapter、跨机 visual spool/worker/collector、退役 full Metal 的独占代码与测试。当前方法实际依赖的 helper 并入所属方法；旧 proposal 链经消费者审计后整体删除。
- YAML 直接控制 spp/cadence；reference 默认 128。删除固定 1024、双方 spp 大小关系、formal/complete/coverage 准入和全 plan/源码 hash 加载门禁。实际 tensor、资源 ABI 和 DDP 次序检查保留。
- Windows 图像实现使用当前模型同步编译、渲染，并写 TensorBoard。Linux 通过共同 hook 调用空实现；数值 validation 保留。独立 Linux eval 也不探测 runtime 或读取 checkpoint。
- 原生 MDL source 准备集中在 `viewer/export.py`，供图像和 export 共用；通用 `scripts/launch_viewer.ps1` 消费显式 package/source。viewer 默认 package 根目录为 outputs。

## 本机证据

环境为完整 Windows：RTX 4090、neural-shading、锁定 Windows Falcor Python 与 Release viewer。没有把 Windows 结果用作 Linux/NCCL 证明。

| 检查 | 实际结果与记录 |
|---|---|
| 全量 unit | 314 passed，28.24 s；`../scratch/unit-results-final-green.txt` |
| 当前方法与公共 GPU 回归 | 19 passed，14.83 s；`../scratch/gpu-results.txt`，覆盖 Nvidia training/latent/proposal、Metal budgeted model/sampler/package/residency、公共 package/reference 与 viewer path-surface |
| Release viewer overlay 构建 | 成功；`../scratch/viewer-build-final.txt`；构建后 external/Falcor 干净 |
| Python 语法 | src/tests/tools 共 228 个文件通过 AST 解析；`../scratch/final-smoke-result.txt` |
| 平台脚本语法 | scripts 下 PowerShell 文件均通过 Parser 检查；Linux 部署脚本通过 Git Bash 的 `bash -n`，此项仅证明 shell 语法 |
| 新状态初始化、续训与数值 validation | 0→2 step 完成；独立 validation 产生有限 loss；`../scratch/final-smoke-init.txt`、`final-smoke-resume-2.txt`、`final-validate.txt` |
| 修改运行设置后恢复 | reference_spp 改为 77、host_prefetch/ready_batches 改为 3，原 run 的 checkpoint 仍可加载并保存；`../scratch/resume-settings-result.txt`。本项检查恢复加载，不声称额外训练步数 |
| YAML spp 与 TensorBoard | 128 和 33 的实际 capture 均为 ready，actual spp 等于配置；comparison/difference 图像事件存在，三个线性 EXR 均 finite；`../scratch/final-check-128.txt`、`final-check-33.txt` |
| 初始化导出与 MDL source | LayerStack 与 MDL 的 step 0 checkpoint 均可导出；MDL viewer 加载新 catalog，在 33 spp 下成功出图；`../scratch/final-export-init.txt`、`final-mdl-export.txt`、`final-check-mdl.txt` |
| 仓库检查 | `git diff --check` 无问题；稳定代码/文档无旧启动脚本、旧 reader 或 artifacts 权重默认路径消费者 |
| 实际目录残留 | 逐项确认后删除 9 个仅含旧 `.pyc` 或为空的退役源码目录；当前有源码的目录保留。初次批量清理被自动审批拒绝，缩小到固定目录后成功 |

本任务 smoke 输出：

- `outputs/nvidia-layer-stack-smoke/260905-172009-9d5b6c/`：128 spp 观察证据。
- `outputs/visual-33/260905-181003-ed3da4/`：新训练、续训、设置变更与 33 spp。
- `outputs/nvidia-mdl-effect-pigment-smoke/260905-183524-047c85/`：初始化状态的原生 MDL 导出/图像链路。

这些都是接口和执行正确性检查。没有用初始化/两步模型的图像质量得出候选质量结论。构建/运行仍有锁定上游的 LNK4098、Slang 隐式转换和 D3D12 Agility 提示，本次各进程成功退出，capture 与 EXR 单独检查通过。

## 验收映射与待实机项

| 验收 | 状态 |
|---|---|
| AC1 | 已完成。盘点的 152 份 capture 及 563 张关联 PNG/EXR 仍在原路径，关联图像总计 589,303,024 bytes，与规划盘点相同；未关联图像也仍存在。没有声称执行过原文件逐字节 hash 对照 |
| AC2 | 入口实现及 Windows 单卡已验证；设备列表、rank 映射、torchrun 构造通过 unit。Linux 单卡/两卡仍待实机 |
| AC3 | 已完成，run 隔离、checkpoint/resume、TensorBoard purge 与实际固定输出均有证据 |
| AC4 | 已完成，128/33 只改 YAML；实际 slot、PNG/EXR 与 TensorBoard 已检查 |
| AC5 | 公共调用点、数值 validation、Linux 空实现均已实现。unit 证明空实现不改变 RNG、reference dispatch、模型结果或产生图像目录，并保留数值窗口；Linux NCCL 数值汇总仍待实机 |
| AC6 | 已完成，运行设置变更可恢复；实际模型 shape/dtype 错误仍报错 |
| AC7 | 已完成，无旧 checkpoint reader、converter、旧训练入口或跨机 visual worker |
| AC8 | 已完成，目录、代码、配置、测试、README、AGENTS/CLAUDE、references、当前研究记录规则与相关 spec 同步 |

Linux 待验证的边界是 NCCL 通信、Falcor/Vulkan 互操作、真实物理卡映射、两 rank stop/resume 和 teardown。TESTING.md 提供单卡、两卡、续训命令及期望；不需要增加新平台训练路径。

## 后续使用

后续模型从新架构创建新 run。旧视觉证据按 `architecture-audit.md` 第 5 节定位，直接阅读原 PNG/EXR 与已有元数据。旧 artifacts 的其他内容由用户自行处理。

用户自行清理时：artifacts 中旧权重、optimizer、旧 TensorBoard 日志、过期部署包和临时报告可以删除；PNG/EXR、截图及解释对应图像的 capture/replay JSON、指标 CSV 原地保留。混有图像的目录按文件清理。本次上面列出的三个 outputs smoke run 也只是运行验证结果，后续不需要时可整组删除；正式训练仍使用相同 outputs 布局。

用户随后明确要求提交并归档，代码提交为 `ea2d743`；本任务按 Trellis 工作流归档。Linux 实机检查仍以待执行项保留。
