# Engineering Change Intelligence Agent：系统架构

## 业务目标

系统面向软件仓库的工程变更分析和故障诊断。输入可以是架构文档、PR 说明、错误日志、运行手册和评测结果；输出必须区分直接证据、推断与待验证项，并给出可执行的下一步。系统不把低相关片段包装成确定结论。

## 状态图

主链路固定为 Route → Retrieve/Rerank → Verify Evidence → Execute Tools → Synthesize。证据不足时，Verify Evidence 转入 Rewrite Query，携带原问题和工程任务词进行第二次检索；达到检索预算后必须停止重试并暴露 `retrieval_retry_limit`。

共享 AgentState 保存原始查询、规范化查询、意图、候选工具、命中证据、检索次数、证据质量和停止原因。Graph Path 与 Tool Trace 记录每个节点及工具调用，因此一次失败不是黑盒字符串，而是可定位到具体阶段的事件序列。

## 可视化调试

前端执行图使用工作流接口返回的真实节点和条件边。用户可在节点执行前设置断点；暂停时服务端保留内存 Checkpoint。用户可以继续当前状态、编辑查询后从 Route 重启，或终止并清理 Checkpoint。验证失败、查询改写、回退和停止原因会以不同状态显示在轨迹中。

## 检索与模型

默认 Embedding 为 `BAAI/bge-small-zh-v1.5`，输出 512 维归一化稠密向量。内存余弦索引和 Chroma 共享同一 Embedding Provider。Reranker 默认使用结构与关键词启发式，也可以延迟加载 Cross-Encoder；可选依赖或模型加载失败时回退到启发式策略，并暴露原始错误。

## 工程工具

Router 识别仓库问答、证据摘要、变更影响、故障诊断、PR 审查、验证计划、Runbook、架构说明、对比和评估。所有分析工具只消费当前命中的 SearchHit，并把来源与章节保留在结果中。外部 LLM 仅在证据门控通过后调用；低证据问题不会进入模型。
