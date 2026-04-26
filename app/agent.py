from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .rag_engine import KnowledgeBase, SearchHit, compact_text
from .study_tools import (
    explain_tool,
    mistake_review_tool,
    quiz_tool,
    review_plan_tool,
    summarize_tool,
    synthesize_explain,
    synthesize_mistake_review,
    synthesize_quiz,
    synthesize_review_plan,
    synthesize_summary,
)


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    output: dict[str, Any]
    latency_ms: int


@dataclass
class AgentResponse:
    session_id: str
    answer: str
    intent: str
    citations: list[dict[str, Any]]
    trace: list[ToolCall]
    suggestions: list[str]
    metrics: dict[str, Any]


class SessionMemory:
    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}

    def ensure(self, session_id: str | None) -> str:
        sid = session_id or str(uuid.uuid4())
        self._sessions.setdefault(sid, [])
        return sid

    def add(self, session_id: str, role: str, content: str) -> None:
        history = self._sessions.setdefault(session_id, [])
        history.append({"role": role, "content": content})
        del history[:-8]

    def recent(self, session_id: str, limit: int = 4) -> list[dict[str, str]]:
        return self._sessions.get(session_id, [])[-limit:]

    def last_user_topic(self, session_id: str) -> str:
        for item in reversed(self._sessions.get(session_id, [])):
            if item["role"] == "user":
                return item["content"]
        return ""


class CapabilityRouter:
    def __init__(self) -> None:
        self.cards = [
            {
                "intent": "rag_answer",
                "tools": ["hybrid_retrieval", "rerank", "answer_synthesizer"],
                "description": "课程问答 文档问答 知识库 资料查询 RAG 引用来源 根据上下文回答",
            },
            {
                "intent": "summarize",
                "tools": ["hybrid_retrieval", "rerank", "summarize_tool", "answer_synthesizer"],
                "description": "总结 重点 复习 提纲 章节归纳 课程资料 知识点整理",
            },
            {
                "intent": "quiz_generation",
                "tools": ["hybrid_retrieval", "rerank", "generate_quiz", "answer_synthesizer"],
                "description": "出题 选择题 判断题 练习题 测验 自测 根据资料生成题目",
            },
            {
                "intent": "concept_explain",
                "tools": ["hybrid_retrieval", "rerank", "explain_concept", "answer_synthesizer"],
                "description": "解释概念 是什么 怎么理解 定义 原理 例子 易错点",
            },
            {
                "intent": "review_plan",
                "tools": ["hybrid_retrieval", "rerank", "make_review_plan", "answer_synthesizer"],
                "description": "复习计划 学习计划 备考安排 时间表 学习路径",
            },
            {
                "intent": "mistake_review",
                "tools": ["hybrid_retrieval", "rerank", "mistake_review", "answer_synthesizer"],
                "description": "错题 错题本 薄弱点 回顾 继续追问 相似题 巩固",
            },
            {
                "intent": "agent_design",
                "tools": ["agent_planner", "hybrid_retrieval", "rerank", "answer_synthesizer"],
                "description": "Agent 智能体 Tool Use Planning Memory 工具调用 任务规划 多轮记忆 架构设计",
            },
            {
                "intent": "compare",
                "tools": ["hybrid_retrieval", "rerank", "compare_tool", "answer_synthesizer"],
                "description": "对比 区别 vs 优缺点 RAG Agent Embedding 向量数据库 Rerank",
            },
            {
                "intent": "resume_packaging",
                "tools": ["hybrid_retrieval", "rerank", "resume_tool", "answer_synthesizer"],
                "description": "简历 岗位匹配 项目亮点 负责内容 技术栈 腾讯校招 AI原生工程师",
            },
            {
                "intent": "evaluation",
                "tools": ["hybrid_retrieval", "rerank", "evaluation_tool", "answer_synthesizer"],
                "description": "评估 指标 命中率 准确率 召回率 延迟 实验结果 RAGAS 测试集",
            },
        ]
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        self.matrix = self.vectorizer.fit_transform([c["description"] for c in self.cards])

    def route(self, query: str) -> tuple[str, list[str], float]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).ravel()
        best_idx = int(scores.argmax())
        card = self.cards[best_idx]

        lowered = query.lower()
        if any(token in lowered for token in ["vs", "区别", "对比", "比较"]):
            card = next(c for c in self.cards if c["intent"] == "compare")
            best_idx = self.cards.index(card)
        elif any(token in query for token in ["总结", "重点", "归纳", "提纲", "复习重点"]):
            card = next(c for c in self.cards if c["intent"] == "summarize")
            best_idx = self.cards.index(card)
        elif any(token in query for token in ["出题", "选择题", "判断题", "练习题", "测验", "自测"]):
            card = next(c for c in self.cards if c["intent"] == "quiz_generation")
            best_idx = self.cards.index(card)
        elif any(token in query for token in ["复习计划", "学习计划", "备考", "时间表", "学习路径"]):
            card = next(c for c in self.cards if c["intent"] == "review_plan")
            best_idx = self.cards.index(card)
        elif any(token in query for token in ["错题", "错了", "薄弱", "巩固", "回顾"]):
            card = next(c for c in self.cards if c["intent"] == "mistake_review")
            best_idx = self.cards.index(card)
        elif any(token in query for token in ["解释", "是什么", "怎么理解", "定义"]):
            card = next(c for c in self.cards if c["intent"] == "concept_explain")
            best_idx = self.cards.index(card)
        elif any(token in query for token in ["简历", "岗位", "面试", "负责"]):
            card = next(c for c in self.cards if c["intent"] == "resume_packaging")
            best_idx = self.cards.index(card)
        elif any(token in query for token in ["评估", "指标", "效果", "准确", "命中"]):
            card = next(c for c in self.cards if c["intent"] == "evaluation")
            best_idx = self.cards.index(card)
        elif any(token in query for token in ["链路", "架构", "工具调用", "执行流程", "怎么运行"]):
            card = next(c for c in self.cards if c["intent"] == "agent_design")
            best_idx = self.cards.index(card)

        return card["intent"], list(card["tools"]), float(scores[best_idx])


class OptionalLLMClient:
    """OpenAI-compatible adapter.

    The demo runs without an API key. If environment variables are present, the
    same agent can call a real LLM without changing endpoint code:
    AI_AGENT_LLM_BASE_URL, AI_AGENT_LLM_API_KEY, AI_AGENT_LLM_MODEL.
    """

    def __init__(self) -> None:
        self.base_url = os.getenv("AI_AGENT_LLM_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("AI_AGENT_LLM_API_KEY", "")
        self.model = os.getenv("AI_AGENT_LLM_MODEL", "gpt-4o-mini")

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key)

    def chat(self, query: str, context: str) -> str | None:
        if not self.enabled:
            return None
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是企业知识库 Agent，回答必须基于给定资料，无法确认时说明缺口。",
                },
                {
                    "role": "user",
                    "content": f"问题：{query}\n\n资料：\n{context}\n\n请给出结构化中文回答，并保留来源线索。",
                },
            ],
            "temperature": 0.2,
        }
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=20,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            return None


class KnowledgeAgent:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.memory = SessionMemory()
        self.router = CapabilityRouter()
        self.llm = OptionalLLMClient()

    def ask(self, query: str, session_id: str | None = None, top_k: int = 6) -> AgentResponse:
        started = time.perf_counter()
        sid = self.memory.ensure(session_id)
        normalized_query = self._resolve_follow_up(query, sid)
        trace: list[ToolCall] = []

        intent, tools, confidence = timed_call(
            trace,
            "intent_router",
            {"query": query},
            lambda: self._route(normalized_query),
        )

        hits: list[SearchHit] = []
        if "hybrid_retrieval" in tools:
            hits = timed_call(
                trace,
                "hybrid_retrieval",
                {"query": normalized_query, "top_k": top_k},
                lambda: self.kb.search(normalized_query, top_k=top_k),
            )
            if "rerank" in tools:
                trace.append(
                    ToolCall(
                        name="rerank",
                        input={"candidates": len(hits)},
                        output={"strategy": "vector + keyword + title + section boost"},
                        latency_ms=0,
                    )
                )

        if "agent_planner" in tools:
            timed_call(
                trace,
                "agent_planner",
                {"intent": intent, "query": normalized_query},
                lambda: agent_plan(intent, tools),
            )
        if "compare_tool" in tools:
            timed_call(
                trace,
                "compare_tool",
                {"query": normalized_query, "evidence": len(hits)},
                lambda: compare_tool(normalized_query, hits),
            )
        if "resume_tool" in tools:
            timed_call(
                trace,
                "resume_tool",
                {"query": normalized_query, "evidence": len(hits)},
                lambda: resume_tool(hits),
            )
        if "evaluation_tool" in tools:
            timed_call(
                trace,
                "evaluation_tool",
                {"query": normalized_query, "evidence": len(hits)},
                lambda: evaluation_tool(),
            )
        if "summarize_tool" in tools:
            timed_call(
                trace,
                "summarize_tool",
                {"query": normalized_query, "evidence": len(hits)},
                lambda: summarize_tool(normalized_query, hits),
            )
        if "generate_quiz" in tools:
            timed_call(
                trace,
                "generate_quiz",
                {"query": normalized_query, "evidence": len(hits)},
                lambda: quiz_tool(normalized_query, hits),
            )
        if "explain_concept" in tools:
            timed_call(
                trace,
                "explain_concept",
                {"query": normalized_query, "evidence": len(hits)},
                lambda: explain_tool(normalized_query, hits),
            )
        if "make_review_plan" in tools:
            timed_call(
                trace,
                "make_review_plan",
                {"query": normalized_query, "evidence": len(hits)},
                lambda: review_plan_tool(normalized_query, hits),
            )
        if "mistake_review" in tools:
            timed_call(
                trace,
                "mistake_review",
                {"query": normalized_query, "evidence": len(hits)},
                lambda: mistake_review_tool(normalized_query, hits),
            )

        answer = timed_call(
            trace,
            "answer_synthesizer",
            {"intent": intent, "hits": len(hits), "llm_enabled": self.llm.enabled},
            lambda: self._synthesize(normalized_query, intent, hits),
        )

        citations = [hit.to_dict() for hit in hits[:4]]
        suggestions = suggest_followups(intent)
        self.memory.add(sid, "user", query)
        self.memory.add(sid, "assistant", answer)
        metrics = {
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "router_confidence": round(confidence, 4),
            "retrieved_chunks": len(hits),
            "llm_mode": "openai_compatible" if self.llm.enabled else "local_synthesizer",
        }
        return AgentResponse(
            session_id=sid,
            answer=answer,
            intent=intent,
            citations=citations,
            trace=trace,
            suggestions=suggestions,
            metrics=metrics,
        )

    def _route(self, query: str) -> tuple[str, list[str], float]:
        return self.router.route(query)

    def _resolve_follow_up(self, query: str, session_id: str) -> str:
        if re.search(r"(刚才|上面|这个|它|该项目|再|继续)", query):
            topic = self.memory.last_user_topic(session_id)
            if topic:
                return f"{topic}\n追问：{query}"
        return query

    def _synthesize(self, query: str, intent: str, hits: list[SearchHit]) -> str:
        context = "\n\n".join(
            f"[{idx + 1}] {hit.chunk.title} / {hit.chunk.section}: {hit.chunk.text}"
            for idx, hit in enumerate(hits[:5])
        )
        llm_answer = self.llm.chat(query, context)
        if llm_answer:
            return llm_answer

        if not hits:
            return "知识库里暂时没有足够依据回答这个问题。可以先上传相关文档，或把问题拆成概念、流程、评估指标三个部分再问。"

        if intent == "compare":
            return synthesize_compare(query, hits)
        if intent == "resume_packaging":
            return synthesize_resume(query, hits)
        if intent == "agent_design":
            return synthesize_agent_design(query, hits)
        if intent == "evaluation":
            return synthesize_evaluation(query, hits)
        if intent == "summarize":
            return synthesize_summary(query, hits)
        if intent == "quiz_generation":
            return synthesize_quiz(query, hits)
        if intent == "concept_explain":
            return synthesize_explain(query, hits)
        if intent == "review_plan":
            return synthesize_review_plan(query, hits)
        if intent == "mistake_review":
            return synthesize_mistake_review(query, hits)
        return synthesize_rag_answer(query, hits)


def timed_call(trace: list[ToolCall], name: str, input_payload: dict[str, Any], fn):
    started = time.perf_counter()
    result = fn()
    elapsed = int((time.perf_counter() - started) * 1000)
    trace.append(
        ToolCall(
            name=name,
            input=input_payload,
            output=summarize_tool_output(result),
            latency_ms=elapsed,
        )
    )
    return result


def summarize_tool_output(result: Any) -> dict[str, Any]:
    if isinstance(result, tuple):
        return {"result": list(result)}
    if isinstance(result, list):
        if result and isinstance(result[0], SearchHit):
            return {
                "hits": len(result),
                "top": [
                    {
                        "source": hit.chunk.source,
                        "section": hit.chunk.section,
                        "score": round(hit.score, 4),
                    }
                    for hit in result[:3]
                ],
            }
        return {"items": len(result)}
    if isinstance(result, str):
        return {"text": compact_text(result, 180)}
    return {"value": result}


def synthesize_rag_answer(query: str, hits: list[SearchHit]) -> str:
    bullets = []
    for hit in hits[:3]:
        bullets.append(f"- {compact_text(hit.chunk.text, 170)}（来源：{hit.chunk.source}）")
    return "基于知识库检索，可以这样回答：\n" + "\n".join(bullets)


def synthesize_compare(query: str, hits: list[SearchHit]) -> str:
    context = " ".join(hit.chunk.text for hit in hits[:4])
    if "rag" in query.lower() and "agent" in query.lower():
        return (
            "RAG 和 Agent 的关系可以这样理解：\n"
            "- RAG 解决“从哪里找依据”的问题，核心是文档切分、Embedding、召回、Rerank 和带引用生成。\n"
            "- Agent 解决“下一步做什么”的问题，核心是意图判断、工具选择、Planning、Memory 和执行链路追踪。\n"
            "- 在本项目里，Agent 会先判断问题类型，再调用 RAG 检索工具、对比工具或简历包装工具，所以 RAG 是 Agent 的一个关键工具。\n"
            f"- 知识库依据：{compact_text(context, 160)}"
        )
    return synthesize_rag_answer(query, hits)


def synthesize_resume(query: str, hits: list[SearchHit]) -> str:
    return (
        "简历上建议把这个项目写成“基于 RAG 与 Agent 的课程知识库智能问答系统”。可突出三点：\n"
        "- 工程闭环：完成文档解析、Chunk 切分、向量召回、混合检索、Rerank、答案生成与前端演示。\n"
        "- Agent 能力：设计意图路由器，根据问题自动选择知识库检索、对比分析、评估说明、简历包装等工具。\n"
        "- 学习场景：支持课程问答、重点总结、自动出题、复习计划和错题回顾，产品形态更贴近学生真实使用。\n"
        "- 岗位适配：覆盖 LLM 调用、Prompt 工程、Embedding、向量数据库替换接口、RAG、Tool Use、Planning、Memory 等 AI 原生工程关键词。\n"
        f"- 可引用依据：{hits[0].chunk.source} / {hits[0].chunk.section}"
    )


def synthesize_agent_design(query: str, hits: list[SearchHit]) -> str:
    return (
        "这个 Agent 的执行链路是：用户问题 -> 意图路由 -> 工具选择 -> 混合检索 -> Rerank -> 答案生成 -> 引用追溯 -> 多轮记忆更新。\n"
        "其中 Tool Use 体现在可调用 retrieval、compare、evaluation、resume 等工具；Planning 体现在按问题类型组合工具链；Memory 体现在 session 内追问补全。"
    )


def synthesize_evaluation(query: str, hits: list[SearchHit]) -> str:
    return (
        "评估可以从四层做：\n"
        "- 检索层：Top-K 命中率、MRR、召回片段相关性。\n"
        "- 生成层：答案忠实度、引用覆盖率、幻觉率。\n"
        "- Agent 层：工具选择准确率、平均调用步数、失败恢复能力。\n"
        "- 工程层：响应延迟、并发稳定性、知识库增量更新耗时。\n"
        f"当前演示接口已经返回 latency、retrieved_chunks 和 router_confidence，便于后续扩展实验面板。"
    )


def agent_plan(intent: str, tools: list[str]) -> dict[str, Any]:
    return {
        "intent": intent,
        "steps": [
            "normalize_query",
            "route_intent",
            *[tool for tool in tools if tool != "answer_synthesizer"],
            "grounded_answer",
            "update_session_memory",
        ],
    }


def compare_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    return {
        "axes": ["目标", "输入输出", "关键组件", "工程风险"],
        "detected_topics": detect_topics(query),
        "evidence_sections": [hit.chunk.section for hit in hits[:3]],
    }


def resume_tool(hits: list[SearchHit]) -> dict[str, Any]:
    return {
        "matched_jd_keywords": [
            "RAG",
            "Embedding",
            "Agent",
            "Tool Use",
            "Planning",
            "Memory",
            "Prompt Engineering",
        ],
        "evidence_sources": [hit.chunk.source for hit in hits[:3]],
        "suggested_project_name": "基于 RAG 与 Agent 的课程知识库智能问答系统",
    }


def evaluation_tool() -> dict[str, Any]:
    return {
        "retrieval_metrics": ["Top-K hit rate", "MRR", "rerank score"],
        "generation_metrics": ["groundedness", "citation coverage", "hallucination rate"],
        "agent_metrics": ["tool selection accuracy", "average tool calls", "recovery rate"],
        "engineering_metrics": ["p95 latency", "API success rate", "indexing time"],
    }


def detect_topics(query: str) -> list[str]:
    topics = []
    lowered = query.lower()
    for label in ["rag", "agent", "embedding", "rerank", "memory", "tool use"]:
        if label in lowered:
            topics.append(label)
    if "简历" in query or "岗位" in query:
        topics.append("resume")
    return topics or ["knowledge_base"]


def suggest_followups(intent: str) -> list[str]:
    common = [
        "把这个项目整理成简历项目经历",
        "RAG 和 Agent 在这个系统里分别负责什么",
        "如何把本地 TF-IDF 替换成真实 Embedding 和向量数据库",
    ]
    if intent == "evaluation":
        return ["设计一组问答评估集", "怎么降低幻觉率", "如何记录召回命中率"]
    if intent == "resume_packaging":
        return ["帮我写 STAR 面试讲法", "压缩成简历两行版", "补一个项目难点和解决方案"]
    return common
