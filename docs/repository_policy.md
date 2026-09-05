# 仓库边界

根仓库保存项目源码、测试、环境声明、中文稳定文档、版本化门槛、资产清单与 `references/` package 说明。`external/`、`assets/`、`data/`、`build/`、`outputs/`、`artifacts/`、`reports/` 和缓存不进入根 Git。正式训练数据由reference在GPU上online产生，不存在可持久化的batch/corpus产品；训练 run、checkpoint、TensorBoard、eval、导出和运行日志集中进入 `outputs/<config-stem>/<run-id>/`。`artifacts/` 保存可清理的临时研究、benchmark 与独立验证报告；旧 viewer PNG/EXR 按用户要求原地保留。本次不迁移或删除旧成果，新训练和默认部署不依赖 artifacts；用户以后可以自行清理。第三方上游保持固定提交和干净工作树。

MDL接入遵循同一边界：MDL SDK package、falcor2 oracle clone与独立stb clone位于ignored `external/`；vMaterials archive和展开内容位于`assets/source-materials/mdl-vmaterials2/2.4.0/`；SDK生成的HLSL、argument block、RO data与解码纹理是可重建cache，位于`build/mdl-reference/`；formal/oracle/native parity报告只进入`artifacts/reference-parity/mdl/`。根仓库只登记fetch/build/run脚本、固定revision/hash、schema、资产manifest和项目自有bridge/canonical runtime。不得从oracle目录复制未登记二进制或源码进入正式路径。
