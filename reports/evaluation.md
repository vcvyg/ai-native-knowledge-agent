# RAG + Agent 评估方案

## 评估目标

项目评估不只看“回答像不像”，而是拆成检索、生成、Agent 路由和工程性能四层，分别判断系统是否真的基于资料回答、是否选对工具、是否能稳定运行。

## 评估用例

| 场景 | 示例问题 | 期望工具链 | 主要指标 |
| --- | --- | --- | --- |
| 课程问答 | 什么是 RAG，为什么能降低幻觉？ | intent_router -> hybrid_retrieval -> rerank -> answer_synthesizer | 引用覆盖率、答案忠实度 |
| 资料总结 | 总结这份实验报告的原理和步骤 | intent_router -> hybrid_retrieval -> summarize_tool -> answer_synthesizer | 重点覆盖率、来源准确性 |
| 自动出题 | 根据数据库事务出 5 道选择题 | intent_router -> hybrid_retrieval -> generate_quiz -> answer_synthesizer | 题目有效率、答案一致性 |
| 概念解释 | 什么是 Embedding？ | intent_router -> hybrid_retrieval -> explain_concept -> answer_synthesizer | 定义准确率、例子相关性 |
| 项目包装 | 这个项目怎么写进简历？ | intent_router -> hybrid_retrieval -> resume_tool -> answer_synthesizer | JD 关键词覆盖率 |
| 追问记忆 | 刚才那个再讲讲难点 | memory -> intent_router -> hybrid_retrieval -> answer_synthesizer | 上下文一致性 |

## 指标设计

- 检索层：Top-K Hit Rate、MRR、平均 Rerank Score、标题/章节命中率。
- 生成层：Groundedness、Citation Coverage、Hallucination Rate、Low-evidence Block Rate。
- Agent 层：Intent Accuracy、Tool Selection Accuracy、Average Tool Calls、Follow-up Recovery Rate。
- 工程层：p50/p95 Latency、API Success Rate、Index Build Time、Upload Parse Success Rate。

## 当前已暴露的运行指标

前端 Runtime 面板已经展示：

- Latency：单次请求端到端耗时。
- Retrieved：召回 chunk 数量。
- Router：意图路由置信度。
- Vector DB：实际使用的向量检索后端，例如 `chroma_hashing` 或 `tfidf_local`。
- LLM Mode：当前是本地 synthesizer 还是 OpenAI-compatible 模型。

## 后续实验计划

1. 构造 50-100 条课程资料问答集，覆盖定义、步骤、公式、对比、实验总结、简历包装等问题。
2. 对比不同检索配置：TF-IDF、Chroma Hashing、bge embedding、Hybrid + Rerank。
3. 统计 Top-K 命中率、引用覆盖率和低证据拦截率。
4. 分析 Bad Case：误召回、跨文档串扰、低相关片段被强行总结、DOCX 表格解析缺失。
5. 将评估结果写入报告，用作面试时解释系统迭代依据。
