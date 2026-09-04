# Reference execution 与 online query 合同

`ReferenceExecutionPlan@1`是source snapshot集合进入GPU reference backend前的唯一编译结果。plan包含有序execution groups、稠密global source index、group-local material index、argument/RO offset、query recipe，以及reference descriptor和snapshot identity。一个work item只能引用一个`execution_group_id`；upper producer按group生成同质batch，backend只做global→local index路由，不识别source family。

`create_reference_backend()`返回唯一公共`ReferenceBackendCapability`。`open(plan, query_capacity, device, slot_count)`为plan的每个group创建底层session，共同提供`evaluate/sample/pdf`。每次operation都执行该group程序的`prepare()`；unknown platform、缺失provider、未知group、跨group source index或设备/shape错误均fail closed。

`evaluate`返回线性RGB `f`、双向PDF、event和valid；`sample`返回direction、weight、双向PDF、eta、event和valid；`pdf`是独立方向查询。公共`f`相对于query输入的`NclsShadingFrame`定义，source-owned normal或geometry normal必须把measure转换编码进`f`，使renderer的`f × |N_input·wi|`恢复原生transport response。

Falcor shared output 由 lease 保护；active lease 存在时不能复用 slot、结束 frame 或 close。online producer 在 GPU 上压实局部 domain 有效行并继续补采，不能把 invalid 行写成零 target，也不能读回 host 后筛选。训练不持久化 batch；data execution plan、reference、query stream、asset collection 与 source identity 进入 `TrainingCheckpoint@1`。

typed texture使用spatial-first shape：2D为`[height,width,(channels)]`，3D为`[depth,height,width,(channels)]`。binder在创建资源前验证rank、extent、dtype和payload元素数。
