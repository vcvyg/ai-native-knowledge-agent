# LangGraph Integration

RepoPilot 的主执行链已经由 LangGraph 驱动。`KnowledgeAgent.ask()`、SSE 流式问答和调试接口都进入同一个 compiled graph，不存在旁路 demo 或只供导入的 adapter。

## Runtime boundary

`app/workflow.py` 是唯一编排入口：

- LangGraph `StateGraph` 执行 Route、Memory、Plan、Retrieve、Verify、Rewrite、ReAct、Reflect 和 Synthesize。
- 条件边根据 `AgentState` 选择证据重试、下一项工具或结束。
- `recursion_limit` 是 retrieval budget 和 ReAct budget 之外的全局循环保护。
- 每个业务节点写回独立 state copy，因此 checkpoint 不会被后续原位修改污染。
- 图的 `updates` stream 被转换为已有 `GraphEvent`，继续服务 SSE 和运行指标。

## Checkpoint and interrupt

每个运行分配独立 LangGraph thread ID，并使用 `InMemorySaver` 保存节点边界状态。调试断点在节点入口调用 `interrupt()`：

1. `/api/debug/run` 执行到断点后返回 paused 和 next_node。
2. `/api/debug/resume` 用同一 thread ID 与 `Command(resume=...)` 继续。
3. 编辑问题并 restart 时删除旧 checkpoint，从 Route 创建全新 thread。
4. 完成或 cancel 时删除 checkpoint，避免服务进程持续积累状态。

当前 checkpointer 是单进程内存实现，适合本地演示和测试。生产部署可将同一编排层替换为数据库 checkpointer，从而支持进程重启后的恢复；业务节点和 API 协议无需重写。

## Evidence and tool safety

LangGraph 只负责运行时，不绕过 RepoPilot 的工程约束。Evidence Gate 仍会阻止低证据请求进入外部 LLM；Tool Registry 仍负责 Skill 白名单、MCP/Native 统一执行、human approval、idempotency 和 rollback。这样既获得标准持久化与恢复语义，也保留现有可验证的工程行为。
