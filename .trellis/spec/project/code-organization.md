# 代码组织

文件按 source、data、method、package、viewer 的实际职责划分。方法模型、objective、source adapter 和 compiler 并置于 `learning/methods/<method>/`；共享 helper 仅在存在当前真实消费者时抽出，不为退役模型保留“共享”空壳。

日常训练只有 `python -m ncls train GPU_LIST --config YAML`。新训练产出集中于 `outputs/<config>/<run>/`；`artifacts/` 用于可清理研究产物和原地保留的旧视觉证据。一次性诊断脚本放活动任务 `scratch/`。

架构变更递归切换消费者并删除旧 reader、alias、converter、schema probe、薄转发器和历史专用工具；不建立迁移层，不搬运旧成果。公共 `Method` 直接提供实现，公共 engine/data/图像接口不按平台或具体方法复制。
