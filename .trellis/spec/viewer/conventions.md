# Viewer 约定

viewer 使用严格 package loader、`ComparisonSlot[2]`、一个 scene PT renderer 与一个 deferred renderer。每个 slot 独立 package/mode/status/resource/timing；panel 为相同 `floor(W/2)×H`，奇数像素是 divider。source editor 只解释 typed parameter tree。capture v4 保存 `slots[2]`，不含左右角色或可变分割位置。错误只污染当前 slot。详见 `../project/unified-pipeline.md`。
