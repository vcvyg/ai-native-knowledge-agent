# AI Course Study Agent

面向大学课程复习场景的 RAG + 学习 Agent 项目。它可以把课件、笔记、实验报告等课程资料构建成知识库，支持资料问答、重点总结、自动出题、复习计划和错题回顾。项目独立于原有课程设计和远程服务器，可以在本机直接运行。

## Features

- Course-material knowledge base with Markdown/TXT/PDF/DOCX ingestion.
- Document chunking, metadata extraction, TF-IDF semantic retrieval.
- Hybrid retrieval: vector similarity + keyword overlap + title boost.
- Rerank stage with section and query-term boosts.
- Evidence-quality guard: low-confidence retrieval will not be forced into an answer.
- Built-in AI-native concept notes for RAG, Agent, Embedding, Rerank, OpenAI and Prompt Engineering.
- Learning Agent router with Tool Use style trace.
- Study tools: course QA, summary, quiz generation, concept explanation, review plan, mistake review.
- Session memory for follow-up questions.
- Optional OpenAI-compatible LLM adapter.
- FastAPI backend and a browser-based demo console.

## Run

```powershell
cd D:\ai-native-knowledge-agent
python -m uvicorn app.main:app --host 127.0.0.1 --port 8015 --reload
```

Open:

```text
http://127.0.0.1:8015
```

## Optional LLM Adapter

The app runs without an API key. To connect a real OpenAI-compatible model:

```powershell
$env:AI_AGENT_LLM_BASE_URL="https://api.openai.com/v1"
$env:AI_AGENT_LLM_API_KEY="your_key"
$env:AI_AGENT_LLM_MODEL="gpt-4o-mini"
```

## Resume Positioning

Project name: AI 课程资料知识库 + 学习 Agent 系统

Suggested summary:

面向大学课程复习场景，构建支持课程资料上传、RAG 问答、知识点总结、自动出题和错题回顾的 AI 学习助手；设计 Router + Tool 的轻量级 Agent 架构，根据用户意图自动选择问答、总结、出题、概念解释、复习计划和错题回顾工具，并通过前端展示引用来源、召回分数、工具调用链路和响应延迟。
