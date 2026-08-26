# 代码组织

文件按 source、data、method、package、viewer 语义单元划分。一次性诊断放活动任务 `scratch/`，产物放 `artifacts/`。接口迁移必须递归切换调用方并删除旧 reader、alias、converter、schema probe 和 fallback。长期入口只有通用 CLI/build/benchmark。
