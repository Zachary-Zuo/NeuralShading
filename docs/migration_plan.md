# 架构重置状态

2026-09-05 起采用统一 Python 入口、按 config/run 聚合的 outputs、直接 Method 接口和单一 checkpoint。旧 checkpoint importer、旧训练脚本、full Metal 及跨机视觉队列已删除。旧成果不迁移，旧 viewer 图像原地保留。当前使用说明见 [architecture.md](architecture.md) 和 [learning.md](learning.md)，执行合同见 `.trellis/spec/project/unified-pipeline.md`。
