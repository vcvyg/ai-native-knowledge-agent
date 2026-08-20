# RepoPilot：系统架构

## 业务目标

系统面向软件仓库的工程变更分析和故障诊断。输入可以是架构文档、PR 说明、错误日志、运行手册和评测结果；输出必须区分直接证据、推断与待验证项，并给出可执行的下一步。系统不把低相关片段包装成确定结论。

## 状态图

外层主链路为 Route → Recall Memory → Plan/Select Skill → Retrieve/Context Pack → Verify Evidence → ReAct → Reflection → Synthesize。证据不足时，Verify Evidence 转入 Rewrite Query；达到检索预算后停止并暴露 `retrieval_retry_limit`。ReAct 内层按 Plan → Action → Tool Call → Observation → Reflection 执行，存在后续动作才继续，且受独立 step budget 和全局 transition limit 约束。

共享 AgentState 保存原始查询、意图、Skill、Plan、Memory hits、Packed Context、Observation、Reflection、检索次数、证据质量和停止原因。Graph Path 与 Tool Trace 记录每个节点及工具调用，因此一次失败不是黑盒字符串，而是可定位到具体阶段的事件序列。

## 可视化调试

前端执行图使用工作流接口返回的真实节点和条件边。用户可在节点执行前设置断点；暂停时服务端保留内存 Checkpoint。用户可以继续当前状态、编辑查询后从 Route 重启，或终止并清理 Checkpoint。验证失败、查询改写、回退和停止原因会以不同状态显示在轨迹中。

## 检索与模型

默认 Embedding 为 `BAAI/bge-small-zh-v1.5`，输出 512 维归一化稠密向量。内存余弦索引和 Chroma 共享同一 Embedding Provider。稳定 Chunk ID 由文档身份、章节和规范化内容生成；文档局部修改只编码 delta。删除进入可恢复 tombstone，恢复后复用稳定身份。

Context Engine 使用 Dense + exact terms 混合召回、Cross-Encoder/heuristic 重排、`[[WikiLink]]` Evidence Graph 有界扩展，再按 token budget、去重、来源优先级和单来源上限进行 Context Packing。响应暴露实际 token 估算、丢弃 chunk、来源数和关系边。

## Memory

Working Memory 保存当前 Graph State 与 Tool Observation；Episodic Memory 保存成功/失败任务轨迹；Semantic Memory 保存仓库知识摘要；Procedural Memory 保存 Skill 和工程步骤。长期记忆支持稳定来源、metadata 和幂等导入，按 relevance、recency、task type 与 importance 排序。历史记忆只能辅助规划，不能替代本轮直接证据。

## 工程工具

Router 识别仓库问答、证据摘要、变更影响、故障诊断、PR 审查、验证计划、Runbook、架构说明、对比和评估。Skill 显式声明 instructions、allowed tools、input/output schema 和 evaluation。Tool Registry 统一 Native Tool 与 MCP Streamable HTTP Tool；写/破坏性工具必须经过 Human Approval Gate，并受 idempotency key 和 rollback 补偿边界保护。

外部 LLM 仅在证据门控通过后调用；低证据问题不会进入模型。每次运行把 Plan、Action、Observation、Reflection、Citation 和 Stop Reason 写入可评分 trajectory，可导出 SFT tool-policy；只有同任务多轨迹存在真实奖励差时才生成 preference pair。
