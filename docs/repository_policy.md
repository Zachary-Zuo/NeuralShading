# 仓库边界

根仓库保存项目源码、测试、环境声明、中文稳定文档、版本化门槛、资产清单与 `references/` package 说明。`external/`、`assets/`、`data/`、`build/`、`artifacts/`、`reports/` 和缓存不进入根 Git。训练 run、checkpoint、ScatteringPackage、capture、benchmark 与验证报告进入 `artifacts/`。第三方上游保持固定提交和干净工作树。
