# 仓库边界

根仓库保存项目源码、测试、环境声明、中文稳定文档、版本化门槛、资产清单与 `references/` package 说明。`external/`、`assets/`、`data/`、`build/`、`artifacts/`、`reports/` 和缓存不进入根 Git。训练 run、checkpoint、ScatteringPackage、capture、benchmark 与验证报告进入 `artifacts/`。第三方上游保持固定提交和干净工作树。

MDL 接入遵循同一边界：MDL SDK package、falcor2 oracle clone 与独立 stb clone 位于 ignored `external/`；vMaterials archive 和展开内容位于 `assets/source-materials/mdl-vmaterials2/2.4.0/`；SDK 生成的 HLSL、argument block、RO data 与解码纹理是可重建 cache，位于 `build/mdl-reference/`；formal/oracle/native parity 报告只进入 `artifacts/reference-parity/mdl/`。根仓库只登记 fetch/build/run 脚本、固定 revision/hash、schema、资产 manifest 和项目自有 bridge/runtime/provider。不得从 oracle 目录复制未登记二进制或源码进入正式路径。
