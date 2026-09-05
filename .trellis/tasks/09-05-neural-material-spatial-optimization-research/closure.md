# 实现阶段收尾与服务器研究移交

用户于 2026-09-05 明确要求「提交并归档」，并将接下来的实验单独建为服务器约 24 小时自主研究任务；随后确认独占 GPU 5、6、7、8、9。本次按这一最新交付边界收尾，而不把尚未执行的原诊断计划勾选为完成。

已交付：原生多 UV raw encoder、共享 conditioning 资源和移交所有权、训练/cook/量化读取、C1–C5 修复、新 runtime ABI、C6 受控 fixture、划痕青铜 train 0→2/validate/export/viewer smoke 与相应 unit/GPU 证据。详细命令、产物、局限见 [实施验证](research/implementation-validation.md)。

原验收项的收尾口径：

| 原项 | 本阶段结果 | 转交内容 |
|---|---|---|
| AC1–AC8 | 研究、设计与代码规划已交付 | 作为后继任务依据保留 |
| AC9 | raw/tile/cook/梯度与固定解码测试通过 | 三个实际 source 的完整分层 D0 |
| AC10 | 多 UV、FP16/SNORM、Jacobian、独立 reference footprint witness 通过 | 真实 normal/参数顺序的剩余 D0 证据 |
| AC11 | 未在训练列表的 snapshot、冻结 E/D、无 optimizer 的 fixture 通过 | 真实新 texture-set/locator 与 held-out response 验证 |
| AC12 | C1–C5 的 unit/GPU 正确性 witness 已通过 | 后续新形态保持回归 |
| AC13 | 共享资源和完整公开入口 smoke 已通过 | Linux/NCCL 实机与服务器监督生命周期 |
| AC14 | step 2 capture 与静态成本已交付 | 真正 matched summary、D1/条件分支、CI 和单次查询成本 |
| AC15 | 新 ABI、实际 bytes/reads/MAC 与 shader 检查已交付 | 后续候选重新登记真实成本 |

后继任务：[服务器 24 小时自主研究](../09-05-neural-material-24h-server-research/prd.md)。它独立计时、独立 fresh run；本任务归档不代表 raw 方法已提高质量，也不代表六个 D1 run 已执行。旧检查点及图像继续保留在原 outputs 位置。
