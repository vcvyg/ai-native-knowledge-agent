from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests

from .context_engine import ContextBundle
from .embeddings import EmbeddingProvider
from .engineering_tools import (
    evidence_summary_tool,
    explain_component_tool,
    impact_analysis_tool,
    incident_triage_tool,
    pr_review_tool,
    runbook_tool,
    synthesize_evidence_summary,
    synthesize_explain_component,
    synthesize_impact_analysis,
    synthesize_incident_triage,
    synthesize_pr_review,
    synthesize_runbook,
    synthesize_validation_plan,
    validation_plan_tool,
)
from .memory import AgentMemory
from .rag_engine import KnowledgeBase, SearchHit, compact_text
from .skills import SkillRegistry, ToolDefinition, ToolRegistry
from .workflow import END, AgentState, EventSink, GraphEvent, GraphRun, StateGraph


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


@dataclass
class DebugResponse:
    run_id: str
    status: str
    next_node: str | None
    breakpoints: list[str]
    query: str
    response: AgentResponse


@dataclass
class DebugCheckpoint:
    run_id: str
    session_id: str
    query: str
    top_k: int
    breakpoints: set[str]
    state: AgentState
    run: GraphRun


def bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read an integer setting without letting a malformed env break startup."""

    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def bounded_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


LOW_EVIDENCE_SCORE = bounded_float_env(
    "AI_AGENT_LOW_EVIDENCE_SCORE", default=0.20, minimum=0.0, maximum=1.0
)

CONCEPT_CARDS: dict[str, dict[str, Any]] = {
    "rag": {
        "title": "RAG（检索增强生成）",
        "aliases": ["rag", "检索增强生成", "retrieval augmented generation"],
        "definition": "RAG 是把外部知识库检索结果放进大模型上下文，再让模型基于资料生成回答的方法。",
        "system_role": "在本系统里，RAG 为变更分析、故障诊断和 PR 审查提供仓库文档、日志与运行手册证据。",
        "engineering": "质量不能只看回答是否流畅，还要验证 Top-K 命中、引用覆盖、答案忠实度与证据缺口。",
    },
    "agent": {
        "title": "Agent（智能体）",
        "aliases": ["agent", "智能体", "ai agent"],
        "definition": "Agent 是能根据目标判断下一步动作，并调用工具完成任务的应用架构。",
        "system_role": "在本项目里，Agent 由显式 State Graph 编排路由、检索、证据验证、查询改写、工具执行和答案生成；低证据时进入有界 Verification Loop。",
        "engineering": "显式图把 Planning、Tool Use、Memory、失败恢复和停止条件落实为可观察、可测试的控制流。",
    },
    "embedding": {
        "title": "Embedding（向量表示）",
        "aliases": ["embedding", "embeddings", "向量", "词向量", "语义向量", "嵌入"],
        "definition": "Embedding 是把文本映射成向量，使语义相近的内容在向量空间中距离更近。",
        "system_role": "在本项目里，文档和查询由 BAAI/bge-small-zh-v1.5 编码为真实稠密向量，再通过内存余弦检索或 Chroma 持久化索引召回。",
        "engineering": "Embedding Provider 与向量存储解耦，并暴露模型名、向量维度和实际后端，便于诊断索引不兼容。",
    },
    "rerank": {
        "title": "Rerank（重排）",
        "aliases": ["rerank", "重排", "二次排序", "重排序"],
        "definition": "Rerank 是对初次召回的候选片段重新排序，把更能回答问题的上下文排到前面。",
        "system_role": "在本项目里，Rerank 融合向量分、关键词覆盖、标题/章节命中和结构加权，减少只靠相似度带来的跑偏。",
        "engineering": "向量召回负责扩大候选集合，Rerank 负责精排；可选模型失败时必须保留错误并安全回退。",
    },
    "llm": {
        "title": "LLM（大语言模型）",
        "aliases": ["llm", "大模型", "大语言模型", "deepseek", "qwen", "通义", "混元"],
        "definition": "LLM 负责理解用户问题、组织语言和生成自然语言回答。",
        "system_role": "在本项目里，LLM 是可选适配层：没有 API Key 时走本地 synthesizer，有 OpenAI-compatible API 时可切换真实模型生成。",
        "engineering": "检索、路由和工具链不绑定单一模型供应商，部署时可按质量、延迟、成本和合规要求切换。",
    },
    "openai": {
        "title": "OpenAI",
        "aliases": ["openai", "gpt", "chatgpt"],
        "definition": "OpenAI 是提供 GPT 系列大模型、Embedding、语音、多模态等 AI API 的公司和平台。",
        "system_role": "在本项目里，OpenAI 可以作为可选 LLM/Embedding 提供方，用于答案生成、语义向量化或后续评估。",
        "engineering": "OpenAI 在这里是可插拔模型供应商之一，不是工作流、证据层或调试能力的前提。",
    },
    "prompt": {
        "title": "Prompt 工程",
        "aliases": ["prompt", "提示词", "prompt engineering", "提示词工程"],
        "definition": "Prompt 工程是设计输入格式、约束、示例和输出结构，让模型更稳定完成任务的方法。",
        "system_role": "不同工程工具使用不同输出约束：影响分析强调依赖与兼容性，故障诊断强调假设与证据，验证计划强调 oracle 与回滚。",
        "engineering": "Prompt 只负责局部任务约束，稳定性还依赖路由、检索、状态图、工具协议和停止条件。",
    },
}


class CapabilityRouter:
    def __init__(self, embeddings: EmbeddingProvider) -> None:
        self.embeddings = embeddings
        self.cards: list[dict[str, Any]] = [
            {
                "intent": "repository_qa",
                "tools": ["hybrid_retrieval", "rerank", "explain_component", "answer_synthesizer"],
                "description": "仓库问答 架构文档 组件职责 代码证据 配置 依赖 调用关系 根据上下文回答",
            },
            {
                "intent": "evidence_summary",
                "tools": ["hybrid_retrieval", "rerank", "evidence_summary", "answer_synthesizer"],
                "description": "总结证据 变更摘要 归纳工程资料 提炼事实 证据地图 范围边界",
            },
            {
                "intent": "change_impact",
                "tools": ["hybrid_retrieval", "rerank", "impact_analysis", "answer_synthesizer"],
                "description": "变更影响 影响面 下游调用 兼容性 breaking change 配置 数据 状态 依赖 风险",
            },
            {
                "intent": "incident_diagnosis",
                "tools": ["hybrid_retrieval", "rerank", "incident_triage", "answer_synthesizer"],
                "description": "线上故障 报错 日志 根因 排查 timeout OOM 初始化失败 重试 异常 恢复",
            },
            {
                "intent": "pr_review",
                "tools": ["hybrid_retrieval", "rerank", "pr_review", "answer_synthesizer"],
                "description": "PR 审查 code review 风险 兼容性 测试缺口 回归 错误处理 安全 代码变更",
            },
            {
                "intent": "validation_plan",
                "tools": ["hybrid_retrieval", "rerank", "validation_plan", "answer_synthesizer"],
                "description": "验证计划 测试方案 复现步骤 验收标准 oracle 故障注入 回归测试 如何证明",
            },
            {
                "intent": "runbook_generation",
                "tools": ["hybrid_retrieval", "rerank", "runbook", "answer_synthesizer"],
                "description": "运行手册 runbook 发布 上线 回滚 值班 SOP 扩量 检查清单 应急处置",
            },
            {
                "intent": "agent_design",
                "tools": ["agent_planner", "hybrid_retrieval", "rerank", "answer_synthesizer"],
                "description": "Agent 智能体 State Graph Tool Use Planning Memory 工具调用 任务规划 执行链路 架构设计",
            },
            {
                "intent": "compare",
                "tools": ["hybrid_retrieval", "rerank", "compare_tool", "answer_synthesizer"],
                "description": "对比 区别 vs 优缺点 RAG Agent Embedding 向量数据库 Rerank",
            },
            {
                "intent": "evaluation",
                "tools": ["hybrid_retrieval", "rerank", "evaluation_tool", "answer_synthesizer"],
                "description": "评估 指标 命中率 准确率 召回率 延迟 实验结果 RAGAS 测试集",
            },
        ]
        self.matrix = self.embeddings.encode_documents([c["description"] for c in self.cards])

    def route(self, query: str) -> tuple[str, list[str], float]:
        q_vec = self.embeddings.encode_queries([query])[0]
        scores = self.matrix @ q_vec
        best_idx = int(scores.argmax())
        card = self.cards[best_idx]

        lowered = query.lower()

        def select(intent: str) -> None:
            nonlocal card, best_idx
            card = next(c for c in self.cards if c["intent"] == intent)
            best_idx = self.cards.index(card)

        if any(token in lowered for token in ["故障", "报错", "异常", "根因", "排查", "诊断", "超时", "incident", "timeout", "oom", "不可用"]):
            select("incident_diagnosis")
        elif any(token in lowered for token in ["pr", "pull request", "代码审查", "code review", "审查这次", "review 这次"]):
            select("pr_review")
        elif any(token in lowered for token in ["变更影响", "影响哪些", "影响面", "受影响", "下游", "breaking", "兼容性影响"]):
            select("change_impact")
        elif any(token in lowered for token in ["验证计划", "测试方案", "如何验证", "怎么验证", "验收标准", "故障注入", "复现步骤"]):
            select("validation_plan")
        elif any(token in lowered for token in ["runbook", "运行手册", "上线步骤", "发布计划", "回滚步骤", "值班", "sop"]):
            select("runbook_generation")
        elif any(token in lowered for token in ["vs", "区别", "对比", "比较", "异同", "差别"]):
            select("compare")
        elif any(token in query for token in ["总结", "归纳", "摘要", "证据地图", "梳理"]):
            select("evidence_summary")
        elif any(token in query for token in ["评估", "指标", "效果", "准确", "命中", "召回率"]):
            select("evaluation")
        elif any(token in query for token in ["链路", "架构", "工具调用", "执行流程", "怎么运行", "模块"]):
            select("agent_design")
        elif any(token in query for token in ["解释", "是什么", "什么是", "怎么理解", "定义", "原理", "作用"]) or detect_concept_card(query):
            select("repository_qa")

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
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key)

    def chat(self, query: str, context: str) -> str | None:
        if not self.enabled:
            return None
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是工程变更与故障诊断 Agent。只基于给定的仓库文档、变更说明、日志和运行手册形成结论。区分已证实事实、合理假设和待验证项；证据不足时必须明确缺口，不得虚构代码、日志或根因。",
                },
                {
                    "role": "user",
                    "content": f"工程问题：{query}\n\n证据：\n{context}\n\n请给出结构化中文回答：先给判断，再列直接依据、风险和下一步验证；不要把低相关资料硬解释成结论。",
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
            self.last_error = None
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            # Surface the failure instead of silently degrading to the local
            # synthesizer so operators can tell "LLM unavailable" apart from
            # "evidence insufficient".
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None


class KnowledgeAgent:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.memory = AgentMemory()
        self.router = CapabilityRouter(kb.embedding_provider)
        self.llm = OptionalLLMClient()
        self.skills = SkillRegistry()
        self.tool_registry = self._build_tool_registry()
        self.mcp_tools_by_intent: dict[str, list[str]] = {}
        self.mcp_config_error: str | None = None
        self._register_configured_mcp_tools()
        self.max_retrieval_attempts = bounded_int_env(
            "AI_AGENT_MAX_RETRIEVAL_ATTEMPTS", default=2, minimum=1, maximum=3
        )
        self.max_react_steps = bounded_int_env(
            "AI_AGENT_MAX_REACT_STEPS", default=3, minimum=1, maximum=5
        )
        self.context_token_budget = bounded_int_env(
            "AI_AGENT_CONTEXT_TOKEN_BUDGET", default=1800, minimum=256, maximum=8000
        )
        self._seed_long_term_memory()
        self.workflow = self._build_workflow()
        self._debug_runs: dict[str, DebugCheckpoint] = {}
        self._debug_lock = threading.Lock()

    def workflow_spec(self) -> dict[str, Any]:
        return {
            "framework": self.workflow.framework,
            "checkpointing": "langgraph_in_memory",
            "interrupts": "langgraph_dynamic",
            "entrypoint": "route",
            "nodes": [
                "route",
                "recall_memory",
                "plan",
                "retrieve",
                "verify_evidence",
                "rewrite_query",
                "react",
                "reflect",
                "synthesize",
            ],
            "edges": [
                {"from": "route", "to": "recall_memory", "when": "always"},
                {"from": "recall_memory", "to": "plan", "when": "always"},
                {"from": "plan", "to": "retrieve", "when": "retrieval tool selected"},
                {"from": "retrieve", "to": "verify_evidence", "when": "always"},
                {
                    "from": "verify_evidence",
                    "to": "rewrite_query",
                    "when": "low evidence and retry budget remains",
                },
                {"from": "rewrite_query", "to": "retrieve", "when": "retry"},
                {
                    "from": "verify_evidence",
                    "to": "react",
                    "when": "evidence sufficient or retry limit reached",
                },
                {"from": "react", "to": "reflect", "when": "tool observation available"},
                {"from": "reflect", "to": "react", "when": "plan has another action"},
                {"from": "reflect", "to": "synthesize", "when": "accepted or bounded stop"},
            ],
            "max_retrieval_attempts": self.max_retrieval_attempts,
            "max_react_steps": self.max_react_steps,
            "context_token_budget": self.context_token_budget,
        }

    def ask(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 6,
        *,
        event_sink: EventSink | None = None,
    ) -> AgentResponse:
        started = time.perf_counter()
        sid = self.memory.ensure(session_id)
        state = self._new_state(query, sid, top_k)
        graph_run = self.workflow.run(state, event_sink=event_sink)
        response = self._build_response(sid, state, graph_run, started, record_memory=True)
        self.workflow.cancel(graph_run.thread_id)
        return response

    def debug_start(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 6,
        breakpoints: list[str] | None = None,
    ) -> DebugResponse:
        started = time.perf_counter()
        sid = self.memory.ensure(session_id)
        state = self._new_state(query, sid, top_k)
        active_breakpoints = self._validate_breakpoints(breakpoints or [])
        run_id = str(uuid.uuid4())
        graph_run = self.workflow.run(state, breakpoints=active_breakpoints)
        return self._finish_debug_step(
            run_id,
            sid,
            query,
            top_k,
            active_breakpoints,
            state,
            graph_run,
            started,
        )

    def debug_resume(
        self,
        run_id: str,
        *,
        query: str | None = None,
        breakpoints: list[str] | None = None,
        restart: bool = False,
    ) -> DebugResponse:
        with self._debug_lock:
            checkpoint = self._debug_runs.get(run_id)
        if checkpoint is None:
            raise KeyError(run_id)
        started = time.perf_counter()
        active_breakpoints = self._validate_breakpoints(
            breakpoints if breakpoints is not None else sorted(checkpoint.breakpoints)
        )
        next_query = (query or checkpoint.query).strip()

        if restart:
            self.workflow.cancel(checkpoint.run.thread_id)
            state = self._new_state(next_query, checkpoint.session_id, checkpoint.top_k)
            state.trace.append(
                ToolCall(
                    name="debug_query_edit",
                    input={"previous_query": checkpoint.query},
                    output={"query": next_query, "restart_from": "route"},
                    latency_ms=0,
                )
            )
            graph_run = self.workflow.run(state, breakpoints=active_breakpoints)
        else:
            if next_query != checkpoint.query:
                raise ValueError("editing a paused query requires restart=true")
            state = checkpoint.state
            graph_run = self.workflow.run(
                state,
                start_at=checkpoint.run.next_node,
                prior_path=checkpoint.run.path,
                prior_events=checkpoint.run.events,
                breakpoints=active_breakpoints,
                skip_breakpoint_once=checkpoint.run.next_node,
                thread_id=checkpoint.run.thread_id,
            )

        return self._finish_debug_step(
            run_id,
            checkpoint.session_id,
            next_query,
            checkpoint.top_k,
            active_breakpoints,
            state,
            graph_run,
            started,
        )

    def cancel_debug(self, run_id: str) -> bool:
        with self._debug_lock:
            checkpoint = self._debug_runs.pop(run_id, None)
        if checkpoint is None:
            return False
        self.workflow.cancel(checkpoint.run.thread_id)
        return True

    def _new_state(self, query: str, session_id: str, top_k: int) -> AgentState:
        normalized_query = self._resolve_follow_up(query, session_id)
        return AgentState(
            query=query,
            normalized_query=normalized_query,
            retrieval_query=normalized_query,
            top_k=top_k,
            trace=[],
            session_id=session_id,
        )

    def _validate_breakpoints(self, breakpoints: list[str]) -> set[str]:
        valid = set(self.workflow_spec()["nodes"])
        unknown = sorted(set(breakpoints) - valid)
        if unknown:
            raise ValueError(f"unknown breakpoint nodes: {', '.join(unknown)}")
        return set(breakpoints)

    def _finish_debug_step(
        self,
        run_id: str,
        session_id: str,
        query: str,
        top_k: int,
        breakpoints: set[str],
        state: AgentState,
        graph_run: GraphRun,
        started: float,
    ) -> DebugResponse:
        completed = graph_run.status == "completed"
        response = self._build_response(
            session_id,
            state,
            graph_run,
            started,
            record_memory=completed,
        )
        if completed:
            self.workflow.cancel(graph_run.thread_id)
            with self._debug_lock:
                self._debug_runs.pop(run_id, None)
        else:
            checkpoint = DebugCheckpoint(
                run_id=run_id,
                session_id=session_id,
                query=query,
                top_k=top_k,
                breakpoints=breakpoints,
                state=state,
                run=graph_run,
            )
            with self._debug_lock:
                self._debug_runs[run_id] = checkpoint
        return DebugResponse(
            run_id=run_id,
            status=graph_run.status,
            next_node=graph_run.next_node,
            breakpoints=sorted(breakpoints),
            query=query,
            response=response,
        )

    def _build_response(
        self,
        sid: str,
        state: AgentState,
        graph_run: GraphRun,
        started: float,
        *,
        record_memory: bool,
    ) -> AgentResponse:

        citations = [hit.to_dict() for hit in state.hits[:4]]
        suggestions = suggest_followups(state.intent)
        if record_memory:
            self.memory.add(sid, "user", state.query)
            self.memory.add(sid, "assistant", state.answer)
            self.memory.remember_episode(
                query=state.query,
                answer=state.answer,
                intent=state.intent,
                stop_reason=state.stop_reason,
                sources=[str(item["source"]) for item in citations],
                success=state.evidence_quality in {"ok", "not_required"},
            )
            self.memory.update_working(
                sid,
                current_task=state.query,
                intent=state.intent,
                graph_path=graph_run.path,
                last_observation=state.observations[-1] if state.observations else None,
                stop_reason=state.stop_reason,
            )
        kb_stats = self.kb.stats()
        metrics = {
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "router_confidence": round(state.router_confidence, 4),
            "retrieved_chunks": len(state.hits),
            "top_score": round(state.hits[0].score, 4) if state.hits else 0.0,
            "evidence_quality": state.evidence_quality,
            "llm_mode": "openai_compatible" if self.llm.enabled else "local_synthesizer",
            "llm_error": getattr(self.llm, "last_error", None),
            "vector_backend": kb_stats["vector_backend"],
            "vector_backend_error": kb_stats["vector_backend_error"],
            "embedding_model": kb_stats["embedding_model"],
            "embedding_dimensions": kb_stats["embedding_dimensions"],
            "reranker": kb_stats["reranker"],
            "reranker_requested": kb_stats["reranker_requested"],
            "reranker_error": kb_stats["reranker_error"],
            "reranker_degraded": kb_stats["reranker_degraded"],
            "reranker_fallback_count": kb_stats["reranker_fallback_count"],
            "retrieval_attempts": state.retrieval_attempts,
            "selected_skill": state.selected_skill,
            "plan": state.plan,
            "react_steps": state.react_steps,
            "reflection": state.reflection,
            "memory_recall": state.memory_hits,
            "memory_stats": self.memory.stats(),
            "context": state.context_metrics,
            "mcp_config_error": self.mcp_config_error,
            "stop_reason": "breakpoint" if graph_run.status == "paused" else state.stop_reason,
            "graph_path": graph_run.path,
            "workflow_steps": graph_run.transitions,
            "run_status": graph_run.status,
            "next_node": graph_run.next_node,
            "graph_events": [serialize_graph_event(event) for event in graph_run.events],
        }
        return AgentResponse(
            session_id=sid,
            answer=state.answer,
            intent=state.intent,
            citations=citations,
            trace=state.trace,
            suggestions=suggestions,
            metrics=metrics,
        )

    def _build_workflow(self) -> StateGraph:
        graph = StateGraph()
        graph.add_node("route", self._route_node)
        graph.add_node("recall_memory", self._recall_memory_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("verify_evidence", self._verify_evidence_node)
        graph.add_node("rewrite_query", self._rewrite_query_node)
        graph.add_node("react", self._react_node)
        graph.add_node("reflect", self._reflect_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.set_entrypoint("route")
        graph.add_edge("route", "recall_memory")
        graph.add_edge("recall_memory", "plan")
        graph.add_edge(
            "plan",
            lambda state: "retrieve" if "hybrid_retrieval" in state.tools else "react",
        )
        graph.add_edge("retrieve", "verify_evidence")
        graph.add_edge(
            "verify_evidence",
            lambda state: "rewrite_query"
            if state.evidence_quality == "low"
            and state.retrieval_attempts < self.max_retrieval_attempts
            else "react",
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("react", "reflect")
        graph.add_edge(
            "reflect",
            lambda state: "react"
            if state.reflection.get("decision") == "continue"
            and state.react_steps < self.max_react_steps
            else "synthesize",
        )
        graph.add_edge("synthesize", END)
        return graph

    def _route_node(self, state: AgentState) -> None:
        state.intent, state.tools, state.router_confidence = timed_call(
            state.trace,
            "intent_router",
            {"query": state.query},
            lambda: self._route(state.normalized_query),
        )
        state.tools.extend(self.mcp_tools_by_intent.get(state.intent, []))

    def _recall_memory_node(self, state: AgentState) -> None:
        recalled = self.memory.recall(
            state.normalized_query,
            task_type=state.intent,
            limit=4,
        )
        state.memory_hits = [hit.to_dict() for hit in recalled]
        state.trace.append(
            ToolCall(
                name="memory_recall",
                input={"query": state.normalized_query, "task_type": state.intent},
                output={
                    "hits": len(recalled),
                    "partitions": [hit.record.partition for hit in recalled],
                    "scores": [round(hit.score, 4) for hit in recalled],
                },
                latency_ms=0,
            )
        )

    def _plan_node(self, state: AgentState) -> None:
        skill = self.skills.select(state.intent)
        state.selected_skill = skill.name
        configured_mcp = self.mcp_tools_by_intent.get(state.intent, [])
        actions = [tool for tool in skill.allowed_tools if tool in state.tools]
        actions.extend(tool for tool in configured_mcp if tool not in actions)
        if not actions:
            actions = [
                tool
                for tool in state.tools
                if tool not in {"hybrid_retrieval", "rerank", "answer_synthesizer"}
            ]
        state.plan = {
            "objective": state.normalized_query,
            "skill": skill.name,
            "actions": actions[: self.max_react_steps],
            "allowed_tools": list(skill.allowed_tools),
            "retrieval_budget": self.max_retrieval_attempts,
            "react_budget": self.max_react_steps,
        }
        state.trace.append(
            ToolCall(
                name="planner",
                input={"intent": state.intent, "memory_hits": len(state.memory_hits)},
                output=state.plan,
                latency_ms=0,
            )
        )

    def _retrieve_node(self, state: AgentState) -> None:
        state.retrieval_attempts += 1
        bundle = timed_call(
            state.trace,
            "hybrid_retrieval",
            {
                "query": state.retrieval_query,
                "top_k": state.top_k,
                "attempt": state.retrieval_attempts,
                "token_budget": self.context_token_budget,
            },
            lambda: self.kb.build_context(
                state.retrieval_query,
                top_k=state.top_k,
                token_budget=self.context_token_budget,
            ),
        )
        current_hits = bundle.hits
        if current_hits and (
            not state.best_hits or current_hits[0].score > state.best_hits[0].score
        ):
            state.best_hits = current_hits
            state.context_metrics = bundle.to_dict()
        elif not state.context_metrics:
            state.context_metrics = bundle.to_dict()
        state.hits = state.best_hits or current_hits

        if "rerank" in state.tools:
            kb_stats = self.kb.stats()
            state.trace.append(
                ToolCall(
                    name="rerank",
                    input={"candidates": len(current_hits), "attempt": state.retrieval_attempts},
                    output={
                        "requested_strategy": kb_stats["reranker_requested"],
                        "strategy": kb_stats["reranker"],
                        "fallback_error": kb_stats["reranker_error"],
                        "degraded": kb_stats["reranker_degraded"],
                        "fallback_count": kb_stats["reranker_fallback_count"],
                    },
                    latency_ms=0,
                )
            )

    def _verify_evidence_node(self, state: AgentState) -> None:
        weak = evidence_is_weak(state.hits, state.normalized_query)
        state.evidence_quality = "low" if weak else "ok"
        if not weak:
            state.stop_reason = "evidence_sufficient"
        elif state.retrieval_attempts >= self.max_retrieval_attempts:
            state.stop_reason = "retrieval_retry_limit"
        state.trace.append(
            ToolCall(
                name="evidence_verifier",
                input={
                    "attempt": state.retrieval_attempts,
                    "threshold": LOW_EVIDENCE_SCORE,
                },
                output={
                    "quality": state.evidence_quality,
                    "top_score": round(state.hits[0].score, 4) if state.hits else 0.0,
                    "next": "rewrite_query"
                    if weak and state.retrieval_attempts < self.max_retrieval_attempts
                    else "react",
                },
                latency_ms=0,
            )
        )

    def _rewrite_query_node(self, state: AgentState) -> None:
        previous_query = state.retrieval_query
        state.retrieval_query = rewrite_query_for_retry(
            state.intent,
            state.normalized_query,
            state.retrieval_attempts,
        )
        state.trace.append(
            ToolCall(
                name="query_rewrite",
                input={"query": previous_query, "reason": "low_evidence"},
                output={"query": state.retrieval_query},
                latency_ms=0,
            )
        )

    def _react_node(self, state: AgentState) -> None:
        if "hybrid_retrieval" not in state.tools:
            state.evidence_quality = "not_required"
            state.stop_reason = "tool_only_workflow"
        actions = list(state.plan.get("actions", []))
        if state.react_steps >= len(actions) or state.react_steps >= self.max_react_steps:
            return
        tool_name = actions[state.react_steps]
        state.react_steps += 1
        execution = self.tool_registry.execute(
            tool_name,
            {
                "query": state.normalized_query,
                "hits": state.hits,
                "intent": state.intent,
                "tools": state.tools,
            },
            idempotency_key=(
                f"{state.session_id}:{state.query}:{state.retrieval_query}:"
                f"{tool_name}:{','.join(hit.chunk.id for hit in state.hits)}"
            ),
        )
        observation = {
            "tool": tool_name,
            "source": execution.source,
            "status": execution.status,
            "output": summarize_tool_output(execution.output),
            "idempotent_replay": execution.idempotent_replay,
            "approval_token": execution.approval_token,
        }
        state.observations.append(observation)
        state.trace.append(
            ToolCall(
                name=tool_name,
                input={
                    "intent": state.intent,
                    "query": state.normalized_query,
                    "evidence": len(state.hits),
                    "skill": state.selected_skill,
                },
                output=observation,
                latency_ms=execution.latency_ms,
            )
        )

    def _reflect_node(self, state: AgentState) -> None:
        actions = list(state.plan.get("actions", []))
        failed = any(item["status"] == "failed" for item in state.observations)
        waiting = any(item["status"] == "waiting_approval" for item in state.observations)
        if failed:
            decision, reason = "stop", "tool_failure"
            state.stop_reason = "tool_failure"
        elif waiting:
            decision, reason = "stop", "human_approval_required"
            state.stop_reason = "human_approval_required"
        elif state.react_steps < len(actions) and state.react_steps < self.max_react_steps:
            decision, reason = "continue", "planned_action_remaining"
        else:
            decision, reason = "accept", "evidence_and_tool_observation_ready"
        state.reflection = {
            "decision": decision,
            "reason": reason,
            "observations": len(state.observations),
            "remaining_actions": max(0, len(actions) - state.react_steps),
        }
        state.trace.append(
            ToolCall(
                name="reflection",
                input={"plan": state.plan, "evidence_quality": state.evidence_quality},
                output=state.reflection,
                latency_ms=0,
            )
        )

    def _synthesize_node(self, state: AgentState) -> None:
        state.answer = timed_call(
            state.trace,
            "answer_synthesizer",
            {
                "intent": state.intent,
                "hits": len(state.hits),
                "top_score": round(state.hits[0].score, 4) if state.hits else 0.0,
                "evidence_quality": state.evidence_quality,
                "llm_enabled": self.llm.enabled,
                "vector_backend": self.kb.stats()["vector_backend"],
            },
            lambda: self._synthesize(
                state.normalized_query,
                state.intent,
                state.hits,
                state.memory_hits,
            ),
        )

    def _route(self, query: str) -> tuple[str, list[str], float]:
        return self.router.route(query)

    def _resolve_follow_up(self, query: str, session_id: str) -> str:
        # Deliberately narrow: "这个/它/再" appear in ordinary standalone
        # questions ("这个系统的架构是什么") and would pollute retrieval with
        # the previous topic. Only explicit anaphora triggers a merge.
        if re.search(r"(刚才|上面|该项目|继续|再讲|再详细|那个|追问)", query):
            topic = self.memory.last_user_topic(session_id)
            if topic:
                return f"{topic}\n追问：{query}"
        return query

    def _synthesize(
        self,
        query: str,
        intent: str,
        hits: list[SearchHit],
        memory_hits: list[dict[str, Any]] | None = None,
    ) -> str:
        concept_card = detect_concept_card(query)
        if concept_card and intent == "repository_qa":
            return synthesize_concept_card(query, concept_card, hits)

        if not hits:
            return "知识库里暂时没有足够依据回答这个问题。可以先上传相关文档，或把问题拆成概念、流程、评估指标三个部分再问。"

        weak_evidence = evidence_is_weak(hits, query)
        if weak_evidence and intent not in {"agent_design", "evaluation"}:
            return synthesize_low_evidence_answer(query, hits)

        # Only grounded questions reach the external model. Project meta intents
        # use deterministic local templates when retrieval evidence is weak.
        if not weak_evidence:
            context = "\n\n".join(
                f"[{idx + 1}] {hit.chunk.title} / {hit.chunk.section}: {hit.chunk.text}"
                for idx, hit in enumerate(hits[:5])
            )
            if memory_hits:
                context += "\n\n历史经验（仅作辅助，不替代直接证据）：\n" + "\n".join(
                    f"- {item['partition']}: {compact_text(str(item['content']), 180)}"
                    for item in memory_hits[:3]
                )
            llm_answer = self.llm.chat(query, context)
            if llm_answer:
                return llm_answer

        if intent == "compare":
            return synthesize_compare(query, hits)
        if intent == "agent_design":
            return synthesize_agent_design(query, hits)
        if intent == "evaluation":
            return synthesize_evaluation(query, hits)
        if intent == "evidence_summary":
            return synthesize_evidence_summary(query, hits)
        if intent == "change_impact":
            return synthesize_impact_analysis(query, hits)
        if intent == "incident_diagnosis":
            return synthesize_incident_triage(query, hits)
        if intent == "pr_review":
            return synthesize_pr_review(query, hits)
        if intent == "validation_plan":
            return synthesize_validation_plan(query, hits)
        if intent == "runbook_generation":
            return synthesize_runbook(query, hits)
        if intent == "repository_qa":
            return synthesize_explain_component(query, hits)
        return synthesize_rag_answer(query, hits)

    def _build_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        handlers = {
            "evidence_summary": lambda payload: evidence_summary_tool(
                str(payload["query"]), list(payload["hits"])
            ),
            "explain_component": lambda payload: explain_component_tool(
                str(payload["query"]), list(payload["hits"])
            ),
            "impact_analysis": lambda payload: impact_analysis_tool(
                str(payload["query"]), list(payload["hits"])
            ),
            "incident_triage": lambda payload: incident_triage_tool(
                str(payload["query"]), list(payload["hits"])
            ),
            "pr_review": lambda payload: pr_review_tool(
                str(payload["query"]), list(payload["hits"])
            ),
            "validation_plan": lambda payload: validation_plan_tool(
                str(payload["query"]), list(payload["hits"])
            ),
            "runbook": lambda payload: runbook_tool(
                str(payload["query"]), list(payload["hits"])
            ),
            "compare_tool": lambda payload: compare_tool(
                str(payload["query"]), list(payload["hits"])
            ),
            "evaluation_tool": lambda _payload: evaluation_tool(),
            "agent_planner": lambda payload: agent_plan(
                str(payload.get("intent", "agent_design")),
                list(payload.get("tools", [])),
            ),
        }
        for name, handler in handlers.items():
            registry.register_native(
                ToolDefinition(
                    name=name,
                    source="native",
                    description=f"Native engineering tool: {name}",
                    input_schema={"type": "object"},
                    risk="read",
                ),
                handler,
            )
        return registry

    def _seed_long_term_memory(self) -> None:
        for skill in self.skills.list():
            self.memory.remember(
                "procedural",
                f"{skill['name']}: {skill['instructions']}",
                source="skill_registry",
                task_type=str(skill["intent"]),
                importance=0.75,
                external_id=str(skill["name"]),
                metadata={"allowed_tools": skill["allowed_tools"]},
            )
        seen_docs: set[str] = set()
        for chunk in self.kb.chunks:
            if chunk.doc_id in seen_docs:
                continue
            seen_docs.add(chunk.doc_id)
            self.memory.remember(
                "semantic",
                f"{chunk.title} / {chunk.section}: {compact_text(chunk.text, 260)}",
                source=chunk.source,
                importance=0.55,
                external_id=chunk.doc_id,
            )

    def _register_configured_mcp_tools(self) -> None:
        """Register MCP tools from JSON without storing credentials in the project."""

        raw = os.getenv("AI_AGENT_MCP_TOOLS_JSON", "").strip()
        if not raw:
            return
        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                raise ValueError("configuration must be a JSON array")
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("each MCP tool configuration must be an object")
                name = str(item["name"])
                endpoint = str(item["endpoint"])
                if not endpoint.startswith(("http://", "https://")):
                    raise ValueError(f"MCP endpoint must use http(s): {name}")
                risk = str(item.get("risk", "read"))
                if risk not in {"read", "write", "destructive"}:
                    raise ValueError(f"invalid MCP risk for {name}: {risk}")
                self.tool_registry.register_mcp(
                    ToolDefinition(
                        name=name,
                        source="mcp",
                        description=str(item.get("description", f"MCP tool: {name}")),
                        input_schema=dict(item.get("input_schema", {"type": "object"})),
                        risk=risk,  # type: ignore[arg-type]
                        remote_name=str(item.get("remote_name", name)),
                    ),
                    endpoint=endpoint,
                )
                for intent in item.get("intents", []):
                    self.mcp_tools_by_intent.setdefault(str(intent), []).append(name)
        except Exception as exc:
            self.mcp_config_error = f"{type(exc).__name__}: {exc}"


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


def serialize_graph_event(event: GraphEvent) -> dict[str, Any]:
    return {
        "sequence": event.sequence,
        "node": event.node,
        "status": event.status,
        "latency_ms": event.latency_ms,
        "next_node": event.next_node,
        "detail": event.detail,
    }


def summarize_tool_output(result: Any) -> dict[str, Any]:
    if isinstance(result, ContextBundle):
        return {
            "hits": len(result.hits),
            **result.to_dict(),
        }
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


def evidence_is_weak(hits: list[SearchHit], query: str | None = None) -> bool:
    if not hits or hits[0].score < LOW_EVIDENCE_SCORE:
        return True
    if not query:
        return False
    return not has_direct_query_signal(query, hits)


def has_direct_query_signal(query: str, hits: list[SearchHit]) -> bool:
    """Require at least one domain-bearing token from the original question.

    Rewritten queries intentionally add generic retrieval vocabulary. Without
    this check those generic words can make an unrelated chunk look strong and
    cause the verification loop to accept fabricated evidence.
    """

    english = {
        item.lower()
        for item in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}", query)
        if item.lower() not in {"is", "the", "and", "for", "with", "what", "how"}
    }
    chinese_sequences = re.findall(r"[\u4e00-\u9fff]{4,}", query)
    chinese: set[str] = set()
    for sequence in chinese_sequences:
        upper = min(8, len(sequence))
        for width in range(4, upper + 1):
            chinese.update(
                sequence[index : index + width]
                for index in range(len(sequence) - width + 1)
            )
    generic = {
        "这个项目",
        "什么是",
        "是什么",
        "怎么写",
        "解释这个",
        "根据资料",
        "资料的",
        "工程资料",
    }
    chinese.difference_update(generic)
    signals = english | chinese
    if not signals:
        return True
    return any(
        any(
            signal
            in f"{hit.chunk.title} {hit.chunk.section} {hit.chunk.text}".lower()
            for signal in signals
        )
        for hit in hits[:4]
    )


def fallback_query_for_intent(intent: str, query: str) -> str:
    if intent == "change_impact":
        return f"{query} 调用方 接口 状态 数据 依赖 兼容性 回滚"
    if intent == "incident_diagnosis":
        return f"{query} 原始错误 失败阶段 配置 依赖 超时 资源 回退"
    if intent == "pr_review":
        return f"{query} 变更范围 兼容性 错误处理 重试 测试 回归"
    if intent == "validation_plan":
        return f"{query} baseline oracle 故障注入 停止条件 回滚验证"
    return query


def rewrite_query_for_retry(intent: str, query: str, attempt: int) -> str:
    """Create a bounded retrieval retry query without asking the LLM.

    Intent-aware expansion keeps the retry deterministic and testable. The
    workflow graph owns the retry count, so this function cannot loop itself.
    """

    intent_query = fallback_query_for_intent(intent, query)
    expansions = {
        "repository_qa": "组件 职责 输入 输出 依赖 调用关系 失败模式",
        "evidence_summary": "事实 证据 来源 范围 假设 缺口",
        "change_impact": "接口 状态 数据 依赖 下游 兼容性 回滚",
        "incident_diagnosis": "异常 日志 失败阶段 根因假设 诊断 恢复",
        "pr_review": "变更 风险 兼容 错误处理 重试 测试 回归",
        "validation_plan": "基线 测试 oracle 故障注入 回退 回滚",
        "runbook_generation": "前置检查 发布 观测 扩量 回退 恢复",
        "compare": "定义 区别 联系 优点 缺点 应用场景",
        "agent_design": "架构 状态 节点 工具 调用流程 验证",
        "evaluation": "评估集 指标 召回率 MRR 忠实度 延迟",
    }
    suffix = expansions.get(intent, "组件 证据 风险 验证 回滚")
    return f"{intent_query} {suffix} retrieval-retry-{attempt + 1}"


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
        f"- 工程约束：{card['engineering']}",
    ]
    if hits and not evidence_is_weak(hits):
        lines.append(f"- {source_summary(hits, limit=2)}")
    else:
        lines.append("- 说明：当前工程资料里没有足够直接的对应片段，这里只使用内置概念卡片说明系统机制，不把无关 chunk 当作证据。")
    return "\n".join(lines)


def synthesize_low_evidence_answer(query: str, hits: list[SearchHit]) -> str:
    lines = [
        "这个问题在当前工程知识库里的直接证据不足，不能从低相关片段拼出确定结论。",
        "建议补充：",
        "- 相关代码路径、PR diff 或架构决策记录；",
        "- 原始错误、时间窗口、run id、关键配置与最近变更；",
        "- 期望行为、实际行为、复现步骤和可接受的回退目标。",
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
            "- 在本系统里，Agent 会先判断工程任务，再调用 RAG 检索、影响分析、故障诊断或验证工具，所以 RAG 是证据获取能力之一。\n"
            f"- {source_summary(hits, limit=2)}"
        )
    return synthesize_rag_answer(query, hits)


def synthesize_agent_design(query: str, hits: list[SearchHit]) -> str:
    return (
        "RepoPilot 的外层链路是：Route -> Recall Memory -> Plan/Select Skill -> "
        "Hybrid Retrieve/Rerank -> Evidence Graph/Context Pack -> Verify Evidence -> "
        "ReAct -> Reflection -> Synthesize。证据不足时进入有预算的 Query Rewrite。\n"
        "内层 ReAct 按 Action -> Native/MCP Tool -> Observation -> Reflection 决定继续或停止；"
        "Working/Episodic/Semantic/Procedural Memory 分别保存当前状态、历史轨迹、仓库知识与工程步骤。"
        "State Graph 同时约束检索、工具步数和全局转换，Graph Path、Context Budget、Memory Recall、"
        "Tool Trace 与 Stop Reason 都能用于调试、评测和 post-training。"
    )


def synthesize_evaluation(query: str, hits: list[SearchHit]) -> str:
    return (
        "评估可以从四层做：\n"
        "- 检索层：Top-K 命中率、MRR、召回片段相关性。\n"
        "- 生成层：答案忠实度、引用覆盖率、幻觉率。\n"
        "- Agent 层：意图准确率、工具选择准确率、证据门控准确率、平均调用步数和停止原因。\n"
        "- 工程层：响应延迟、并发稳定性、知识库增量更新耗时。\n"
        "当前演示接口返回 latency、retrieved_chunks、reranker、retrieval_attempts、graph_path 和 stop_reason，并配有 JSONL 基线评估。"
    )


def agent_plan(intent: str, tools: list[str]) -> dict[str, Any]:
    return {
        "intent": intent,
        "steps": [
            "normalize_query",
            "route_intent",
            "recall_partitioned_memory",
            "select_skill",
            *[tool for tool in tools if tool != "answer_synthesizer"],
            "reflect_on_observation",
            "grounded_answer",
            "write_episodic_memory",
        ],
    }


def compare_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    return {
        "axes": ["目标", "输入输出", "关键组件", "工程风险"],
        "detected_topics": detect_topics(query),
        "evidence_sections": [hit.chunk.section for hit in hits[:3]],
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
    return topics or ["knowledge_base"]


def suggest_followups(intent: str) -> list[str]:
    common = [
        "分析这项变更的调用方、状态与回退风险",
        "为失败路径设计可执行的验证计划",
        "如何在轨迹图中定位一次低证据重试",
    ]
    if intent == "incident_diagnosis":
        return ["还需要补充哪些日志才能确认根因", "给出最小复现与回退步骤", "把诊断过程整理成 Runbook"]
    if intent == "pr_review":
        return ["列出这个 PR 的测试缺口", "检查重试与停止条件", "生成合并前验证计划"]
    if intent == "evaluation":
        return ["设计一组变更分析评估集", "怎么验证证据门控", "如何记录工具选择准确率"]
    return common
