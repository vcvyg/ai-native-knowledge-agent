from __future__ import annotations

from app.memory import AgentMemory


def test_memory_import_is_idempotent_and_partitioned() -> None:
    memory = AgentMemory()
    records = [
        {
            "id": "incident-1",
            "partition": "episodic",
            "content": "Embedding 下载超时后检查代理并复用本地缓存。",
            "task_type": "incident_diagnosis",
            "importance": 0.9,
        },
        {
            "id": "skill-1",
            "partition": "procedural",
            "content": "PR review 必须检查兼容性、失败路径与回滚验证。",
            "task_type": "pr_review",
        },
    ]

    first = memory.import_records(records, source="hebb-mind")
    second = memory.import_records(records, source="hebb-mind")

    assert first == {"created": 2, "duplicates": 0, "total": 2}
    assert second == {"created": 0, "duplicates": 2, "total": 2}
    assert memory.stats()["episodic"] == 1
    assert memory.stats()["procedural"] == 1


def test_memory_recall_exposes_ranking_factors() -> None:
    memory = AgentMemory()
    memory.remember(
        "episodic",
        "Embedding 模型下载超时，代理配置错误导致初始化失败。",
        source="trajectory",
        task_type="incident_diagnosis",
        importance=0.8,
    )

    hits = memory.recall("Embedding 下载超时怎么排查", task_type="incident_diagnosis")

    assert hits
    assert hits[0].relevance > 0
    assert hits[0].task_match == 1.0
    assert hits[0].to_dict()["partition"] == "episodic"
