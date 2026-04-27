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


LOW_EVIDENCE_SCORE = 0.055

CONCEPT_CARDS: dict[str, dict[str, Any]] = {
    "rag": {
        "title": "RAG（检索增强生成）",
        "aliases": ["rag", "检索增强生成", "retrieval augmented generation"],
        "definition": "RAG 是把外部知识库检索结果放进大模型上下文，再让模型基于资料生成回答的方法。",
        "system_role": "在本项目里，RAG 负责资料解析、chunk 切分、召回、重排、引用来源和答案生成，是课程资料问答的主链路。",
        "interview": "面试时可以强调它降低幻觉、支持私有资料问答，并且可以通过 Top-K 命中率、引用覆盖率和答案忠实度评估效果。",
    },
    "agent": {
        "title": "Agent（智能体）",
        "aliases": ["agent", "智能体", "ai agent"],
        "definition": "Agent 是能根据目标判断下一步动作，并调用工具完成任务的应用架构。",
        "system_role": "在本项目里，Agent 不是复杂多智能体，而是 Router + Tool 的轻量架构：先识别问答、总结、出题、复习计划等意图，再调用对应工具。",
        "interview": "这种设计可控、可解释、容易落地，适合课程学习场景，也贴合 AI 原生工程师岗位对 Tool Use、Planning、Memory 的要求。",
    },
    "embedding": {
        "title": "Embedding（向量表示）",
        "aliases": ["embedding", "embeddings", "向量", "词向量", "语义向量", "嵌入"],
        "definition": "Embedding 是把文本映射成向量，使语义相近的内容在向量空间中距离更近。",
        "system_role": "在本项目里，它对应知识库检索层：上传资料被切成 chunk 后向量化，用户问题也向量化，然后做相似度召回。",
        "interview": "可以说当前 demo 用 TF-IDF 模拟本地向量检索，工程接口保留为可替换形态，后续能换成 bge、OpenAI embedding、Chroma、FAISS 或 Milvus。",
    },
    "rerank": {
        "title": "Rerank（重排）",
        "aliases": ["rerank", "重排", "二次排序", "重排序"],
        "definition": "Rerank 是对初次召回的候选片段重新排序，把更能回答问题的上下文排到前面。",
        "system_role": "在本项目里，Rerank 融合向量分、关键词覆盖、标题/章节命中和结构加权，减少只靠相似度带来的跑偏。",
        "interview": "可以强调向量召回负责“广撒网”，Rerank 负责“精排序”，两者配合提升答案相关性。",
    },
    "llm": {
        "title": "LLM（大语言模型）",
        "aliases": ["llm", "大模型", "大语言模型", "deepseek", "qwen", "通义", "混元"],
        "definition": "LLM 负责理解用户问题、组织语言和生成自然语言回答。",
        "system_role": "在本项目里，LLM 是可选适配层：没有 API Key 时走本地 synthesizer，有 OpenAI-compatible API 时可切换真实模型生成。",
        "interview": "这能体现工程解耦：检索、路由和工具链不依赖某一家模型服务，部署时可按成本和效果切换供应商。",
    },
    "openai": {
        "title": "OpenAI",
        "aliases": ["openai", "gpt", "chatgpt"],
        "definition": "OpenAI 是提供 GPT 系列大模型、Embedding、语音、多模态等 AI API 的公司和平台。",
        "system_role": "在本项目里，OpenAI 可以作为可选 LLM/Embedding 提供方，用于答案生成、语义向量化或后续评估。",
        "interview": "如果面试官问到 OpenAI，可以把它放在“可插拔模型供应商”角度讲，而不是把项目绑定到某一个平台。",
    },
    "prompt": {
        "title": "Prompt 工程",
        "aliases": ["prompt", "提示词", "prompt engineering", "提示词工程"],
        "definition": "Prompt 工程是设计输入格式、约束、示例和输出结构，让模型更稳定完成任务的方法。",
        "system_role": "在本项目里，不同工具会使用不同回答策略，例如问答强调依据和引用，总结强调复习重点，出题强调题干、选项、答案和解析。",
        "interview": "可以强调你不是只写一句 prompt，而是把任务拆成路由、检索、工具调用和结构化生成多个环节。",
    },
}


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

        def select(intent: str) -> None:
            nonlocal card, best_idx
            card = next(c for c in self.cards if c["intent"] == intent)
            best_idx = self.cards.index(card)

        if any(token in lowered for token in ["vs", "区别", "对比", "比较", "异同", "差别"]):
            select("compare")
        elif re.search(r"(出|生成|来)\s*\d*\s*(道|个)?\s*(选择题|判断题|练习题|题目|quiz)", lowered) or any(
            token in query for token in ["出题", "选择题", "判断题", "练习题", "测验", "自测"]
        ):
            select("quiz_generation")
        elif any(token in query for token in ["总结", "重点", "归纳", "提纲", "复习重点", "梳理"]):
            select("summarize")
        elif any(token in query for token in ["复习计划", "学习计划", "备考", "时间表", "学习路径"]):
            select("review_plan")
        elif any(token in query for token in ["错题", "错了", "薄弱", "巩固", "回顾"]):
            select("mistake_review")
        elif any(token in query for token in ["简历", "岗位", "面试", "负责", "腾讯", "校招", "jd"]):
            select("resume_packaging")
        elif any(token in query for token in ["评估", "指标", "效果", "准确", "命中", "召回率", "幻觉"]):
            select("evaluation")
        elif any(token in query for token in ["链路", "架构", "工具调用", "执行流程", "怎么运行", "模块"]):
            select("agent_design")
        elif any(token in query for token in ["解释", "是什么", "什么是", "怎么理解", "定义", "原理", "作用"]) or detect_concept_card(query):
            select("concept_explain")

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
                    "content": "你是课程资料知识库 Agent。优先基于给定资料回答；如果资料不相关或证据不足，要明确说明缺口。对 RAG、Agent、Embedding、OpenAI 等通用项目概念，可以补充通用解释，但必须标明这部分不是上传资料直接证据。",
                },
                {
                    "role": "user",
                    "content": f"问题：{query}\n\n资料：\n{context}\n\n请给出结构化中文回答：先给结论，再列关键依据和来源；不要把低相关资料硬解释成答案。",
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
            if evidence_is_weak(hits) and intent in {"summarize", "quiz_generation", "review_plan"}:
                fallback_query = fallback_query_for_intent(intent, normalized_query)
                fallback_hits = timed_call(
                    trace,
                    "fallback_retrieval",
                    {
                        "reason": "low_evidence",
                        "original_top_score": round(hits[0].score, 4) if hits else 0.0,
                        "query": fallback_query,
                    },
                    lambda: self.kb.search(fallback_query, top_k=top_k),
                )
                if fallback_hits and (not hits or fallback_hits[0].score > hits[0].score):
                    hits = fallback_hits

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
            {
                "intent": intent,
                "hits": len(hits),
                "top_score": round(hits[0].score, 4) if hits else 0.0,
                "evidence_quality": "low" if evidence_is_weak(hits) else "ok",
                "llm_enabled": self.llm.enabled,
            },
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
            "top_score": round(hits[0].score, 4) if hits else 0.0,
            "evidence_quality": "low" if evidence_is_weak(hits) else "ok",
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

        concept_card = detect_concept_card(query)
        if concept_card and intent != "compare":
            return synthesize_concept_card(query, concept_card, hits)

        if not hits:
            return "知识库里暂时没有足够依据回答这个问题。可以先上传相关文档，或把问题拆成概念、流程、评估指标三个部分再问。"

        if evidence_is_weak(hits) and intent not in {"resume_packaging", "agent_design", "evaluation"}:
            return synthesize_low_evidence_answer(query, hits)

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


def evidence_is_weak(hits: list[SearchHit]) -> bool:
    return not hits or hits[0].score < LOW_EVIDENCE_SCORE


def fallback_query_for_intent(intent: str, query: str) -> str:
    if intent == "quiz_generation":
        return f"{query} 自然语言处理 核心概念 模型 方法 评估 定义 原理"
    if intent == "summarize":
        return f"{query} 章节重点 核心概念 方法流程 应用场景"
    if intent == "review_plan":
        return f"{query} 复习重点 核心概念 典型题型 易错点"
    return query


def detect_concept_card(query: str) -> dict[str, Any] | None:
    lowered = query.lower()
    for card in CONCEPT_CARDS.values():
        for alias in card["aliases"]:
            alias_lower = alias.lower()
            if alias_lower in lowered:
                return card
    return None


def source_summary(hits: list[SearchHit], limit: int = 3) -> str:
    useful = [hit for hit in hits[:limit] if hit.score >= LOW_EVIDENCE_SCORE]
    if not useful:
        return "资料命中：当前知识库没有检索到足够直接的片段，以下回答主要来自项目内置知识卡片。"
    parts = [f"{hit.chunk.source} / {hit.chunk.section}" for hit in useful]
    return "参考来源：" + "；".join(parts)


def evidence_bullets(hits: list[SearchHit], limit: int = 3) -> list[str]:
    bullets = []
    seen: set[str] = set()
    for hit in hits:
        snippet = compact_text(hit.chunk.text, 150)
        if snippet in seen:
            continue
        seen.add(snippet)
        bullets.append(f"- {snippet}（来源：{hit.chunk.source} / {hit.chunk.section}）")
        if len(bullets) >= limit:
            break
    return bullets


def synthesize_concept_card(query: str, card: dict[str, Any], hits: list[SearchHit]) -> str:
    lines = [
        f"{card['title']}可以这样理解：",
        f"- 是什么：{card['definition']}",
        f"- 在本系统里的作用：{card['system_role']}",
        f"- 面试表达：{card['interview']}",
    ]
    if hits and not evidence_is_weak(hits):
        lines.append(f"- {source_summary(hits, limit=2)}")
    else:
        lines.append("- 说明：当前上传课程资料里没有足够直接的对应片段，所以这里使用项目内置概念卡片兜底，避免强行引用无关 chunk。")
    return "\n".join(lines)


def synthesize_low_evidence_answer(query: str, hits: list[SearchHit]) -> str:
    lines = [
        "这个问题在当前知识库里的直接证据不足，我不建议硬从低相关片段里拼答案。",
        "可以这样处理：",
        "- 如果这是课程资料问题，建议补充更具体的章节名、概念名，或上传对应课件。",
        "- 如果这是刚上传的 Word 实验报告，请看资料列表的 chunk 数；只有 1 个 chunk 或正文很短时，通常说明旧索引只读到了封面，需要删除旧资料后重新上传。",
        "- 如果这是项目/岗位问题，可以问 RAG、Agent、Embedding、Rerank、OpenAI、Prompt 等关键词，我会走项目内置概念卡片兜底。",
    ]
    if hits:
        lines.append("低相关候选片段仅供定位，不作为强依据：")
        lines.extend(evidence_bullets(hits, limit=2))
    return "\n".join(lines)


def synthesize_rag_answer(query: str, hits: list[SearchHit]) -> str:
    if evidence_is_weak(hits):
        return synthesize_low_evidence_answer(query, hits)
    lines = [
        "根据知识库里命中的资料，可以归纳为：",
        f"- 直接回答：{compact_text(hits[0].chunk.text, 220)}",
        "- 关键依据：",
    ]
    lines.extend(evidence_bullets(hits, limit=3))
    lines.append(f"- {source_summary(hits, limit=3)}")
    return "\n".join(lines)


def synthesize_compare(query: str, hits: list[SearchHit]) -> str:
    if "rag" in query.lower() and "agent" in query.lower():
        return (
            "RAG 和 Agent 的关系可以这样理解：\n"
            "- RAG 解决“从哪里找依据”的问题，核心是文档切分、Embedding、召回、Rerank 和带引用生成。\n"
            "- Agent 解决“下一步做什么”的问题，核心是意图判断、工具选择、Planning、Memory 和执行链路追踪。\n"
            "- 在本项目里，Agent 会先判断问题类型，再调用 RAG 检索工具、对比工具或简历包装工具，所以 RAG 是 Agent 的一个关键工具。\n"
            f"- {source_summary(hits, limit=2)}"
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
