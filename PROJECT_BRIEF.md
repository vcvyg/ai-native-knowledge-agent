# 项目简历说明

## 项目名称

AI 课程资料知识库 + 学习 Agent 系统

## 项目定位

面向大学课程复习场景的 AI 学习助手，支持上传课件、笔记、实验报告等课程资料，并围绕资料完成 RAG 问答、重点总结、自动出题、概念解释、复习计划和错题回顾。系统融合文档切分、语义召回、混合检索、Rerank、工具调用型 Agent、多轮记忆和前端可视化，能够展示从用户问题到检索依据、工具选择和答案生成的完整链路。

## 技术栈

Python、FastAPI、Scikit-learn、TF-IDF Vectorizer、PDF/DOCX 解析、RAG、Hybrid Retrieval、Rerank、Agent Tool Use、Session Memory、HTML/CSS/JavaScript、OpenAI-compatible LLM Adapter。

## 核心功能

- 课程资料知识库：支持 Markdown/TXT/PDF/DOCX 上传、标题解析、章节切分、Chunk 构建和 metadata 管理。
- 混合检索：融合字符级语义相似度、词级相似度、关键词重合度和标题/章节加权。
- Rerank：对召回片段进行二次排序，提升高相关依据的展示优先级。
- Agent 路由：根据问题自动识别课程问答、重点总结、自动出题、概念解释、复习计划、错题回顾等意图。
- Tool Use：显式调用 hybrid_retrieval、rerank、summarize_tool、generate_quiz、explain_concept、make_review_plan、mistake_review、answer_synthesizer 等工具。
- Memory：支持同一会话中的追问补全。
- 可视化：展示答案、引用片段、检索分数、调用链路、响应延迟、召回数量和 LLM 模式。

## 简历写法

面向大学课程复习场景，构建 AI 课程资料知识库 + 学习 Agent 系统，支持 PDF/DOCX/TXT/Markdown 资料上传、Chunk 切分、混合检索、Rerank、RAG 问答和引用追溯；设计 Router + Tool 的轻量级 Agent，根据用户意图自动选择重点总结、自动出题、概念解释、复习计划和错题回顾工具，并通过前端展示 Agent Trace、召回分数和响应延迟，提升学习问答的可信度和可解释性。

## 个人负责

负责系统整体架构设计与核心链路实现，完成课程资料解析、知识库构建、检索召回、Rerank、Agent 工具路由、多轮 Memory 和 FastAPI 接口开发；负责前端演示页面，实现资料上传、答案、引用来源、工具调用链路和运行指标的可视化；预留 OpenAI-compatible LLM 接口，支持后续接入 DeepSeek、Qwen、OpenAI 或混元等大模型服务。

## 可继续增强

- 将 TF-IDF 原型替换为 Sentence-Transformers / OpenAI Embedding。
- 将本地内存索引替换为 FAISS、Chroma、Milvus 或 pgvector。
- 引入 Cross-Encoder 或 LLM Rerank。
- 增加更完整的 PDF 版面解析、OCR 和公式识别能力。
- 构建问答评估集，统计 Top-K 命中率、引用覆盖率、幻觉率和工具选择准确率。
- 增加真实错题本表结构，记录错题、答案、知识点和推荐复习 chunk。
- 使用 Docker Compose 部署前后端和向量数据库。

