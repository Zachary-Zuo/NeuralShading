# Reference backend query 合同

`create_reference_backend()`返回唯一公共`ReferenceBackendCapability`。它负责平台识别、Falcor环境/device、build layout和program provider preflight；`open()`把canonical `ReferenceProgramDefinition`及其`SourceSnapshot`编译为`ReferenceBackendSession`，提供`evaluate/sample/pdf`三个typed operation。每次operation内部都先调用同一个program的`prepare()`，不存在family-specific query shader或替代BRDF。

upper modules不得直接导入Falcor、选择D3D12/Vulkan、构造session或解析平台路径。unknown OS/arch、缺失Falcor build或program provider时fail closed；MDL SDK compiler是`mdl.program@1`内部provider，不是第二套公共backend。

`evaluate`返回线性RGB `f`、forward/reverse PDF、event与valid；`sample`原样返回direction、weight、双向PDF、eta、event与valid tuple；`pdf`是独立方向查询。训练evaluator只消费`f`，source `sample/pdf`保留给transport与合同验证。这里的`f`以query输入的`NclsShadingFrame`为transport measure：source若内部应用normal map或`geometry.normal`，必须把source-native response中的material-normal cosine比值编码进`f`，保证renderer执行`f × |N_input·wi|`时恢复原生response。

所有输入/输出tensor位于同一CUDA device。Falcor shared output由lease保护；active lease存在时不能结束frame、复用slot或close。局部domain invalid行由online producer拒绝并补采，不能写成零target或在host读回response后筛选。

typed texture统一使用spatial-first shape：2D为`[height,width,(channels)]`，3D为`[depth,height,width,(channels)]`。binder从前置axes解析extent，并在创建GPU resource前验证rank和payload元素数；scalar与RGBA只在末尾channel axis上有差异。

该合同没有训练batch的磁盘schema、reader或writer。source/query/recipe identity连同backend semantic/build identity随`TrainingCheckpoint@3`保存；单次parity与诊断报告进入`artifacts/`。
