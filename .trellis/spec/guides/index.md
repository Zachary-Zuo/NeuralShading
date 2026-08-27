# 开发思考清单

- 新 source 是否保留原生语义、typed edit 与权威 reference？
- 新方法是否只增加一个 `MethodDefinition`？
- offline/live 是否输出同一 batch schema 且 live 无 host readback？
- package 的 runtime/material/package identity 是否独立且可篡改拒绝？
- viewer 两个 slot 是否对称、固定 50/50、失败隔离？
- source/neural PT 是否都只经 canonical prepared state？方向 sampler、throughput weight 与 MIS PDF 是否属于同一 estimator，renderer 是否完全不识别 source family？`finite` 不能替代尖锐 closure 的权重尾部和空间聚集检查；见 `../core/shared-slang-backend.md`、`../viewer/conventions.md` 与 `../viewer/mdl-reference.md`。
- 是否同步 schema、测试、稳定文档并删除旧入口？
