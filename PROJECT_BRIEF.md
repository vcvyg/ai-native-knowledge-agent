# 项目简历说明

## 项目名称

AI 课程资料知识库 + 学习 Agent 系统

## 项目定位

面向大学课程复习场景的 AI 原生学习助手。用户可以上传课件、笔记、实验报告等资料，系统自动完成文档解析、Chunk 切分、向量化建库、混合检索、Rerank、RAG 问答、知识点总结、自动出题、复习计划和错题回顾。

这个项目不是普通聊天机器人，而是一个可演示的 RAG + Agent 产品原型：它能展示从用户问题、意图识别、工具选择、知识召回、证据重排到最终答案生成的完整链路，贴合 AI 原生工程师对 LLM 应用落地、知识库系统、Agent 架构和工程交付能力的要求。

## 技术栈

- 后端：Python、FastAPI、Pydantic、Uvicorn
- 文档处理：Markdown/TXT/PDF/DOCX 解析、文本清洗、章节识别、Chunk 切分
- 检索系统：Chroma 可选持久化向量库、TF-IDF 本地向量检索 fallback、关键词召回、标题/章节加权
- RAG：Top-K 检索、Hybrid Retrieval、Rerank、Evidence Guard、引用来源追踪
- Agent：Intent Router、Tool Use、轻量 Planning、Session Memory
- LLM：OpenAI-compatible Adapter，可接入 OpenAI、DeepSeek、Qwen、混元等模型服务
- 前端：HTML/CSS/JavaScript，展示问答、引用、工具链路、检索分数和运行指标

## 核心功能

- 课程资料知识库：支持上传 Markdown、TXT、PDF、DOCX，并将资料转成统一 Markdown 文档入库。
- 向量数据库：支持 Chroma 持久化向量库；本地未安装 Chroma 时自动 fallback 到 TF-IDF 向量检索，保证 demo 可运行。
- 混合检索：融合向量相似度、关键词覆盖、标题/章节命中，避免只靠关键词或只靠向量导致召回偏差。
- Rerank：对初次召回片段进行二次排序，优先展示高相关证据。
- Evidence Guard：当证据分数过低时，不强行编答案，而是提示资料缺口或建议补充资料。
- Agent Router：根据用户意图选择问答、总结、出题、概念解释、复习计划、错题回顾、项目包装和评估说明等工具。
- Tool Trace：前端展示每次回答调用了哪些工具、耗时、输入输出摘要，方便面试讲解 Tool Use。
- Session Memory：保留最近对话上下文，支持“刚才那个再讲讲”这类追问。

## 简历写法

面向大学课程复习场景，设计并实现 AI 课程资料知识库 + 学习 Agent 系统，支持 PDF/DOCX/TXT/Markdown 资料上传、文档切分、Chroma/TF-IDF 向量检索、混合召回、Rerank、RAG 问答、引用追踪和低证据保护；采用 Router + Tool 的轻量 Agent 架构，根据用户意图自动调用总结、出题、概念解释、复习计划、错题回顾等工具，并在前端展示 Agent Trace、召回分数、响应延迟和 LLM 模式，提升学习问答的可信度与可解释性。

## 个人负责

负责系统整体架构设计与核心链路实现，完成文档解析、知识库构建、向量检索后端抽象、混合召回、Rerank、Agent 工具路由、多轮 Memory、FastAPI 接口和前端演示页面；预留 OpenAI-compatible LLM 接口，支持后续接入 DeepSeek、Qwen、OpenAI 或混元等大模型服务。

## 面试可讲亮点

- 为什么是轻量 Agent：学习场景强调稳定和可解释，不做复杂多 Agent，而是用 Router + Tool 降低不可控性。
- 为什么要混合检索：课程资料里专业名词、公式名、实验名很多，关键词召回更准；开放问法又需要向量召回，所以采用融合排序。
- 为什么加 Evidence Guard：RAG 系统最怕低相关片段被模型硬解释成答案，低证据保护能降低幻觉。
- 为什么保留 fallback：真实工程里演示环境、依赖安装、模型服务都可能不稳定，fallback 能保证核心流程可用。

## 后续优化

- 将 Hashing/TF-IDF embedding 替换为 bge-small-zh、bge-base-zh 或 OpenAI text-embedding 系列。
- 引入 Cross-Encoder 或 LLM Rerank，提高精排质量。
- 增加 OCR、公式解析和版面结构恢复，提升课件/实验报告解析效果。
- 构建问答评估集，统计 Top-K 命中率、MRR、引用覆盖率、低证据拦截率和工具选择准确率。
- 将错题本持久化为结构化表，记录错题、正确答案、知识点、来源 chunk 和复习计划。
- 使用 Docker Compose 部署前后端、向量数据库和模型网关。
