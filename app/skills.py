from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Callable, Literal

import requests

RiskLevel = Literal["read", "write", "destructive"]
ToolHandler = Callable[[dict[str, Any]], Any]
RollbackHandler = Callable[[], None]


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    instructions: str
    allowed_tools: tuple[str, ...]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    evaluation: tuple[str, ...]
    intent: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    source: Literal["native", "mcp"]
    description: str
    input_schema: dict[str, Any]
    risk: RiskLevel = "read"
    remote_name: str | None = None

    @property
    def requires_approval(self) -> bool:
        return self.risk in {"write", "destructive"}


@dataclass
class ToolOutcome:
    output: Any
    rollback: RollbackHandler | None = None


@dataclass(frozen=True)
class ToolExecution:
    execution_id: str
    tool: str
    source: str
    status: Literal["completed", "waiting_approval", "rolled_back", "failed"]
    output: Any
    idempotent_replay: bool
    latency_ms: int
    approval_token: str | None = None


class MCPClient:
    """MCP Streamable HTTP client with initialize and ``tools/call`` support."""

    def __init__(self, endpoint: str, timeout: float = 20.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.session_id: str | None = None
        self._initialized = False

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._initialized:
            self._initialize()
        return self._rpc("tools/call", {"name": name, "arguments": arguments})

    def _initialize(self) -> None:
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "RepoPilot", "version": "0.6.0"},
            },
        )
        self._rpc("notifications/initialized", {}, notification=True)
        self._initialized = True

    def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        *,
        notification: bool = False,
    ) -> Any:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        if not notification:
            payload["id"] = str(uuid.uuid4())
        headers = {"Accept": "application/json, text/event-stream"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        if response.headers.get("Mcp-Session-Id"):
            self.session_id = response.headers["Mcp-Session-Id"]
        if notification or not response.content:
            return None
        body = _decode_mcp_response(response)
        if body.get("error"):
            raise RuntimeError(f"MCP tool failed: {body['error']}")
        return body.get("result")


class ToolRegistry:
    """Native/MCP tool registry with approval, idempotency and rollback boundaries."""

    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}
        self._completed: dict[str, ToolExecution] = {}
        self._pending: dict[str, tuple[str, str]] = {}
        self._rollbacks: dict[str, RollbackHandler] = {}
        self._lock = threading.RLock()

    def register_native(
        self,
        definition: ToolDefinition,
        handler: ToolHandler,
    ) -> None:
        if definition.source != "native":
            raise ValueError("native tools must declare source=native")
        self._definitions[definition.name] = definition
        self._handlers[definition.name] = handler

    def register_mcp(
        self,
        definition: ToolDefinition,
        *,
        endpoint: str,
        client: MCPClient | None = None,
    ) -> None:
        if definition.source != "mcp":
            raise ValueError("MCP tools must declare source=mcp")
        mcp = client or MCPClient(endpoint)
        remote_name = definition.remote_name or definition.name
        self._definitions[definition.name] = definition
        self._handlers[definition.name] = lambda payload: mcp.call_tool(remote_name, payload)

    def definition(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def definitions(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self._definitions.values()]

    def request_approval(self, name: str, payload: dict[str, Any]) -> str:
        definition = self.definition(name)
        if not definition.requires_approval:
            raise ValueError(f"tool does not require approval: {name}")
        token = str(uuid.uuid4())
        with self._lock:
            self._pending[token] = (name, _payload_digest(payload))
        return token

    def execute(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        approval_token: str | None = None,
    ) -> ToolExecution:
        definition = self.definition(name)
        key = idempotency_key or _payload_digest({"tool": name, "payload": payload})
        with self._lock:
            cached = self._completed.get(key)
            if cached is not None:
                return ToolExecution(
                    **{
                        **asdict(cached),
                        "idempotent_replay": True,
                    }
                )

        if definition.requires_approval:
            expected = self._pending.get(approval_token or "")
            if expected != (name, _payload_digest(payload)):
                token = self.request_approval(name, payload)
                return ToolExecution(
                    execution_id="",
                    tool=name,
                    source=definition.source,
                    status="waiting_approval",
                    output={"reason": "human_approval_required", "risk": definition.risk},
                    idempotent_replay=False,
                    latency_ms=0,
                    approval_token=token,
                )

        started = time.perf_counter()
        execution_id = str(uuid.uuid4())
        try:
            raw = self._handlers[name](payload)
            outcome = raw if isinstance(raw, ToolOutcome) else ToolOutcome(output=raw)
            result = ToolExecution(
                execution_id=execution_id,
                tool=name,
                source=definition.source,
                status="completed",
                output=outcome.output,
                idempotent_replay=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                approval_token=None,
            )
            with self._lock:
                self._completed[key] = result
                if outcome.rollback is not None:
                    self._rollbacks[execution_id] = outcome.rollback
                if approval_token:
                    self._pending.pop(approval_token, None)
            return result
        except Exception as exc:
            return ToolExecution(
                execution_id=execution_id,
                tool=name,
                source=definition.source,
                status="failed",
                output={"error": f"{type(exc).__name__}: {exc}"},
                idempotent_replay=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                approval_token=None,
            )

    def rollback(self, execution_id: str) -> ToolExecution:
        with self._lock:
            handler = self._rollbacks.pop(execution_id, None)
        if handler is None:
            raise KeyError(f"no rollback registered for execution: {execution_id}")
        started = time.perf_counter()
        handler()
        return ToolExecution(
            execution_id=execution_id,
            tool="rollback",
            source="native",
            status="rolled_back",
            output={"rolled_back": True},
            idempotent_replay=False,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class SkillRegistry:
    def __init__(self, skills: list[SkillDefinition] | None = None) -> None:
        selected = skills or default_skills()
        self._by_intent = {skill.intent: skill for skill in selected}
        self._by_name = {skill.name: skill for skill in selected}

    def select(self, intent: str) -> SkillDefinition:
        return self._by_intent.get(intent, self._by_name["repository_explainer"])

    def list(self) -> list[dict[str, Any]]:
        return [skill.to_dict() for skill in self._by_name.values()]


def default_skills() -> list[SkillDefinition]:
    common_input = {"type": "object", "required": ["query", "evidence"]}
    common_output = {"type": "object", "required": ["result", "evidence"]}
    definitions = [
        (
            "change_impact",
            "change_impact_analysis",
            "Map affected contracts, state, dependencies and verification boundaries.",
            ("impact_analysis",),
        ),
        (
            "incident_diagnosis",
            "incident_triage",
            "Build evidence-backed hypotheses, checks, recovery and rollback steps.",
            ("incident_triage",),
        ),
        (
            "pr_review",
            "pull_request_review",
            "Review compatibility, failure handling, security and test gaps.",
            ("pr_review",),
        ),
        (
            "validation_plan",
            "validation_planning",
            "Define baseline, functional, failure, recovery and regression oracles.",
            ("validation_plan",),
        ),
        (
            "runbook_generation",
            "rollback_planning",
            "Produce approval-aware operational and rollback procedures.",
            ("runbook",),
        ),
        (
            "evidence_summary",
            "evidence_mapping",
            "Summarize direct evidence, source boundaries and missing information.",
            ("evidence_summary",),
        ),
        (
            "repository_qa",
            "repository_explainer",
            "Explain repository components from packed, cited context.",
            ("explain_component",),
        ),
        (
            "agent_design",
            "agent_architecture_planning",
            "Plan a bounded State Graph, ReAct loop, tools, memory and verification path.",
            ("agent_planner",),
        ),
        (
            "compare",
            "evidence_comparison",
            "Compare alternatives using the same evidence and explicit trade-offs.",
            ("compare_tool",),
        ),
        (
            "evaluation",
            "agent_evaluation",
            "Evaluate intent, source, tool, evidence-gate, reward and latency signals.",
            ("evaluation_tool",),
        ),
    ]
    return [
        SkillDefinition(
            intent=intent,
            name=name,
            description=instructions,
            instructions=instructions,
            allowed_tools=tools,
            input_schema=common_input,
            output_schema=common_output,
            evaluation=("direct_evidence", "bounded_execution", "explicit_stop_reason"),
        )
        for intent, name, instructions, tools in definitions
    ]


def _payload_digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return sha256(encoded.encode()).hexdigest()[:24]


def _decode_mcp_response(response: requests.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError("MCP response must be a JSON object")
        return body
    for line in response.text.splitlines():
        if line.startswith("data:"):
            body = json.loads(line.removeprefix("data:").strip())
            if isinstance(body, dict):
                return body
    raise RuntimeError("MCP event stream did not contain a JSON-RPC response")
