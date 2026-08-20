from __future__ import annotations

from typing import Any

from app.skills import MCPClient, ToolDefinition, ToolOutcome, ToolRegistry


def test_write_tool_requires_approval_is_idempotent_and_rolls_back() -> None:
    registry = ToolRegistry()
    state: list[str] = []

    def apply_change(payload: dict[str, Any]) -> ToolOutcome:
        state.append(str(payload["value"]))
        return ToolOutcome(output={"applied": payload["value"]}, rollback=lambda: state.pop())

    registry.register_native(
        ToolDefinition(
            name="apply_change",
            source="native",
            description="test mutation",
            input_schema={"type": "object"},
            risk="write",
        ),
        apply_change,
    )

    waiting = registry.execute("apply_change", {"value": "v1"}, idempotency_key="change-1")
    assert waiting.status == "waiting_approval"
    assert state == []

    completed = registry.execute(
        "apply_change",
        {"value": "v1"},
        idempotency_key="change-1",
        approval_token=waiting.approval_token,
    )
    replay = registry.execute("apply_change", {"value": "v1"}, idempotency_key="change-1")

    assert completed.status == "completed"
    assert replay.idempotent_replay is True
    assert state == ["v1"]

    rolled_back = registry.rollback(completed.execution_id)
    assert rolled_back.status == "rolled_back"
    assert state == []


def test_mcp_client_initializes_session_before_tool_call(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class Response:
        def __init__(self, body: dict[str, Any] | None, session: str | None = None) -> None:
            self._body = body
            self.content = b"" if body is None else b"{}"
            self.text = ""
            self.headers = {"content-type": "application/json"}
            if session:
                self.headers["Mcp-Session-Id"] = session

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._body or {}

    def post(_url, *, json, headers, timeout):
        calls.append({"payload": json, "headers": headers, "timeout": timeout})
        method = json["method"]
        if method == "initialize":
            return Response({"jsonrpc": "2.0", "id": json["id"], "result": {}}, "session-1")
        if method == "notifications/initialized":
            return Response(None)
        return Response(
            {"jsonrpc": "2.0", "id": json["id"], "result": {"content": [{"text": "ok"}]}}
        )

    monkeypatch.setattr("app.skills.requests.post", post)

    result = MCPClient("http://localhost:3000/mcp").call_tool("repo_status", {"repo": "demo"})

    assert [item["payload"]["method"] for item in calls] == [
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]
    assert calls[-1]["headers"]["Mcp-Session-Id"] == "session-1"
    assert result["content"][0]["text"] == "ok"
