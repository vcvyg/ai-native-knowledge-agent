from __future__ import annotations

import re
from typing import Any

from .rag_engine import SearchHit, compact_text

RISK_RULES = {
    "API / contract": ["api", "contract", "schema", "接口", "协议", "字段", "兼容"],
    "state / data": ["state", "checkpoint", "database", "index", "状态", "数据", "索引", "幂等"],
    "dependency / runtime": ["dependency", "model", "timeout", "依赖", "模型", "超时", "资源"],
    "recovery / operations": ["retry", "fallback", "rollback", "重试", "回退", "回滚", "告警"],
    "verification": ["test", "metric", "evaluation", "测试", "指标", "验证", "评估"],
}


def evidence_summary_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    return {
        "scope": infer_scope(query, hits),
        "key_evidence": evidence_records(hits, limit=5),
        "source_count": len({hit.chunk.source for hit in hits}),
    }


def explain_component_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    return {
        "component": infer_scope(query, hits),
        "evidence": evidence_records(hits, limit=4),
        "analysis_axes": ["responsibility", "inputs_outputs", "dependencies", "failure_modes"],
    }


def impact_analysis_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    text = corpus_text(query, hits)
    return {
        "change_scope": infer_scope(query, hits),
        "affected_areas": matched_risk_areas(text),
        "evidence": evidence_records(hits, limit=5),
        "required_checks": [
            "确认调用方与配置的向后兼容性",
            "验证状态与数据迁移是否可重放、可回滚",
            "覆盖成功、失败、重试与回退路径",
            "比较变更前后的质量、延迟和资源指标",
        ],
    }


def incident_triage_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    text = corpus_text(query, hits).lower()
    hypotheses: list[str] = []
    if any(token in text for token in ["timeout", "超时", "download", "下载"]):
        hypotheses.append("外部依赖或模型下载不可达，初始化阶段耗尽超时预算")
    if any(token in text for token in ["oom", "memory", "内存", "资源"]):
        hypotheses.append("模型或索引的峰值内存超过运行环境预算")
    if any(token in text for token in ["dimension", "维度", "index", "索引"]):
        hypotheses.append("Embedding 维度、模型版本与既有索引不一致")
    if any(token in text for token in ["retry", "重试", "loop", "循环"]):
        hypotheses.append("重试缺少边界或复用了无效输入，导致故障被放大")
    if not hypotheses:
        hypotheses.append("现有证据只能定位故障阶段，尚不足以确认单一根因")
    return {
        "incident": compact_text(query, 180),
        "hypotheses": hypotheses[:4],
        "evidence": evidence_records(hits, limit=5),
        "next_checks": [
            "保留原始异常、失败阶段和关联 run id",
            "检查最近变更、配置差异和依赖可用性",
            "用最小输入复现，并区分初始化、检索、重排和生成阶段",
            "执行一次受控回退，对比错误率与延迟是否恢复",
        ],
    }


def pr_review_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    return {
        "review_scope": infer_scope(query, hits),
        "risk_areas": matched_risk_areas(corpus_text(query, hits)),
        "evidence": evidence_records(hits, limit=5),
        "review_questions": [
            "新旧配置、API 与持久化状态是否兼容？",
            "失败是否保留原始错误，并暴露可诊断的阶段信息？",
            "重试是否有预算、停止条件和幂等边界？",
            "可选依赖不可用时是否能安全降级？",
            "测试是否覆盖主路径、失败路径、回退路径和回归指标？",
        ],
    }


def validation_plan_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    return {
        "target": infer_scope(query, hits),
        "evidence": evidence_records(hits, limit=4),
        "plan": [
            {"stage": "baseline", "check": "固定输入、版本、配置与现有质量/延迟指标"},
            {"stage": "functional", "check": "验证成功路径的输出、引用和状态转换"},
            {"stage": "failure", "check": "注入超时、空结果和依赖不可用，检查重试与停止条件"},
            {"stage": "recovery", "check": "触发回退或回滚，确认服务恢复且无脏状态"},
            {"stage": "regression", "check": "运行单元测试、JSONL 评测并人工复核证据边界"},
        ],
        "oracle": ["结果有直接证据", "状态路径符合预期", "失败可诊断", "回退可恢复"],
    }


def runbook_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    return {
        "operation": infer_scope(query, hits),
        "evidence": evidence_records(hits, limit=4),
        "steps": [
            "记录版本、配置、变更单、负责人和回滚目标",
            "检查依赖、容量、索引与健康探针",
            "小流量执行并观察错误率、p95 延迟、资源和证据质量",
            "异常时停止扩量，保留诊断上下文并执行受控回退",
            "恢复后重新验证主路径，记录结果与遗留风险",
        ],
    }


def synthesize_evidence_summary(query: str, hits: list[SearchHit]) -> str:
    data = evidence_summary_tool(query, hits)
    lines = [f"结论：当前证据覆盖的是「{data['scope']}」，可用于形成工程判断，但结论范围不应超出这些材料。", "关键证据："]
    lines.extend(format_evidence(data["key_evidence"]))
    lines.append(f"证据边界：共命中 {data['source_count']} 个来源；未出现于材料中的实现细节仍需代码或运行结果确认。")
    return "\n".join(lines)


def synthesize_explain_component(query: str, hits: list[SearchHit]) -> str:
    data = explain_component_tool(query, hits)
    lines = [f"组件定位：{data['component']}", "从工程视角应同时确认职责、输入输出、依赖和失败模式。", "直接依据："]
    lines.extend(format_evidence(data["evidence"]))
    lines.append("建议下一步：沿调用方、状态写入点和回退分支继续追踪，避免只看单个函数。")
    return "\n".join(lines)


def synthesize_impact_analysis(query: str, hits: list[SearchHit]) -> str:
    data = impact_analysis_tool(query, hits)
    lines = [f"变更判断：这不是单点替换，影响范围为「{data['change_scope']}」。", "优先检查的影响面："]
    lines.extend(f"- {area}" for area in data["affected_areas"])
    lines.append("验证与控制：")
    lines.extend(f"- {check}" for check in data["required_checks"])
    lines.append("证据：")
    lines.extend(format_evidence(data["evidence"], limit=3))
    return "\n".join(lines)


def synthesize_incident_triage(query: str, hits: list[SearchHit]) -> str:
    data = incident_triage_tool(query, hits)
    lines = ["诊断结论：当前可以形成按证据排序的假设，但不能在没有运行日志的情况下直接宣称根因。", "候选假设："]
    lines.extend(f"- {item}" for item in data["hypotheses"])
    lines.append("下一步排查：")
    lines.extend(f"- {item}" for item in data["next_checks"])
    lines.append("关联证据：")
    lines.extend(format_evidence(data["evidence"], limit=3))
    return "\n".join(lines)


def synthesize_pr_review(query: str, hits: list[SearchHit]) -> str:
    data = pr_review_tool(query, hits)
    lines = [f"PR 风险结论：需要围绕「{data['review_scope']}」做系统级审查，而不只检查代码能否运行。", "审查清单："]
    lines.extend(f"- {item}" for item in data["review_questions"])
    lines.append("当前材料暴露的风险面：" + "、".join(data["risk_areas"]))
    lines.append("关联证据：")
    lines.extend(format_evidence(data["evidence"], limit=3))
    return "\n".join(lines)


def synthesize_validation_plan(query: str, hits: list[SearchHit]) -> str:
    data = validation_plan_tool(query, hits)
    lines = [f"验证目标：{data['target']}", "执行顺序："]
    lines.extend(f"- {idx}. {item['stage']}：{item['check']}" for idx, item in enumerate(data["plan"], 1))
    lines.append("通过标准：" + "；".join(data["oracle"]))
    lines.append("证据来源：")
    lines.extend(format_evidence(data["evidence"], limit=2))
    return "\n".join(lines)


def synthesize_runbook(query: str, hits: list[SearchHit]) -> str:
    data = runbook_tool(query, hits)
    lines = [f"运行手册范围：{data['operation']}", "执行与回退步骤："]
    lines.extend(f"- {idx}. {step}" for idx, step in enumerate(data["steps"], 1))
    lines.append("约束：任何扩量都必须以健康指标和证据质量均通过为前提。")
    return "\n".join(lines)


def corpus_text(query: str, hits: list[SearchHit]) -> str:
    return " ".join([query, *[hit.chunk.text for hit in hits[:6]]])


def infer_scope(query: str, hits: list[SearchHit]) -> str:
    cleaned = re.sub(r"[？?。.!]", "", query).strip()
    if cleaned:
        return compact_text(cleaned, 90)
    if hits:
        return f"{hits[0].chunk.title} / {hits[0].chunk.section}"
    return "待确认的工程变更"


def matched_risk_areas(text: str) -> list[str]:
    lowered = text.lower()
    areas = [area for area, tokens in RISK_RULES.items() if any(token in lowered for token in tokens)]
    return areas or ["component boundary", "error handling", "verification"]


def evidence_records(hits: list[SearchHit], limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        key = (hit.chunk.source, hit.chunk.section)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "source": hit.chunk.source,
                "section": hit.chunk.section,
                "score": round(hit.score, 4),
                "snippet": compact_text(hit.chunk.text, 170),
            }
        )
        if len(records) >= limit:
            break
    return records


def format_evidence(records: list[dict[str, Any]], limit: int | None = None) -> list[str]:
    selected = records if limit is None else records[:limit]
    return [
        f"- {item['snippet']}（{item['source']} / {item['section']}）"
        for item in selected
    ]
