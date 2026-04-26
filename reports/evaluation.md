# Evaluation Plan

| Query Type | Example | Expected Tool Chain | Metric |
| --- | --- | --- | --- |
| Concept QA | RAG 是什么，为什么能降低幻觉 | intent_router -> hybrid_retrieval -> rerank -> answer_synthesizer | Citation coverage |
| Compare | RAG 和 Agent 有什么区别 | intent_router -> hybrid_retrieval -> compare_tool -> answer_synthesizer | Answer completeness |
| Resume | 这个项目怎么写进简历 | intent_router -> hybrid_retrieval -> resume_tool -> answer_synthesizer | JD keyword coverage |
| Evaluation | 怎么评估这个系统 | intent_router -> hybrid_retrieval -> evaluation_tool -> answer_synthesizer | Metric correctness |
| Follow-up | 刚才那个再讲讲难点 | memory -> intent_router -> hybrid_retrieval -> answer_synthesizer | Context consistency |

Recommended metrics:

- Retrieval: Top-K hit rate, MRR, average rerank score.
- Generation: groundedness, citation coverage, hallucination rate.
- Agent: tool selection accuracy, average tool calls, recovery rate.
- Engineering: p95 latency, API success rate, indexing time.

