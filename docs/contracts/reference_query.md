# ReferenceQuery 合同

`ReferenceQueryDispatcher`把一个canonical `ReferenceProgramDefinition`及其`SourceSnapshot`编译为GPU query实例，提供`evaluate/sample/pdf`三个typed operation。每次operation内部都先调用同一个backend的`prepare()`，不存在family-specific query shader或替代BRDF。

`evaluate`返回线性RGB `f`、forward/reverse PDF、event与valid；`sample`原样返回direction、weight、双向PDF、eta、event与valid tuple；`pdf`是独立方向查询。训练evaluator只消费`f`，source `sample/pdf`保留给transport与合同验证。

所有输入/输出tensor位于同一CUDA device。Falcor shared output由lease保护；active lease存在时不能结束frame、复用slot或close。局部domain invalid行由online producer拒绝并补采，不能写成零target或在host读回response后筛选。

该合同没有磁盘schema、reader或writer。source/query/recipe identity随`TrainingCheckpoint@3`保存；单次parity与诊断报告进入`artifacts/`。
