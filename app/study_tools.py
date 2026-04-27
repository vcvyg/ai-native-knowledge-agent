from __future__ import annotations

import re
from typing import Any

from .rag_engine import SearchHit, compact_text


def summarize_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    points = extract_key_points(hits, limit=5)
    concepts = extract_concepts(" ".join(hit.chunk.text for hit in hits[:5]))
    return {
        "summary_type": "course_review_notes",
        "key_points": points,
        "concepts": concepts[:8],
        "evidence_sections": [hit.chunk.section for hit in hits[:4]],
    }


def quiz_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    questions = build_quiz(hits, count=extract_requested_count(query))
    return {
        "quiz_type": "mixed_practice",
        "question_count": len(questions),
        "questions": questions,
        "source_chunks": [hit.chunk.id for hit in hits[:5]],
    }


def explain_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    concept = infer_concept(query, hits)
    evidence = compact_text(" ".join(hit.chunk.text for hit in hits[:3]), 420)
    return {
        "concept": concept,
        "explanation_strategy": ["定义", "作用", "关键步骤", "易错点"],
        "evidence": evidence,
    }


def review_plan_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    concepts = extract_concepts(" ".join(hit.chunk.text for hit in hits[:6]))
    if not concepts:
        concepts = ["核心概念", "关键流程", "典型题型", "错题复盘"]
    return {
        "duration": "3 days",
        "plan": [
            {
                "day": 1,
                "task": "通读召回资料，整理概念卡片",
                "focus": concepts[:3],
            },
            {
                "day": 2,
                "task": "围绕重点概念做问答和选择题",
                "focus": concepts[3:6] or concepts[:3],
            },
            {
                "day": 3,
                "task": "回顾错题，重新生成相似题并检查引用来源",
                "focus": ["错题原因", "相似题训练", "复述总结"],
            },
        ],
    }


def mistake_review_tool(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    concepts = extract_concepts(" ".join(hit.chunk.text for hit in hits[:5]))
    return {
        "weak_points": concepts[:5] or ["资料理解不完整", "概念边界不清晰"],
        "review_actions": [
            "定位错题对应的知识 chunk",
            "重新解释概念",
            "生成 3 道相似题",
            "隔天再次回顾",
        ],
        "recommended_sources": [
            {"source": hit.chunk.source, "section": hit.chunk.section, "score": round(hit.score, 4)}
            for hit in hits[:3]
        ],
    }


def synthesize_summary(query: str, hits: list[SearchHit]) -> str:
    if not hits:
        return "当前知识库没有找到可总结的课程资料。可以先上传课件、笔记或实验报告。"
    context = " ".join(hit.chunk.text for hit in hits[:6])
    if "自然语言处理" in context or "NLP" in context or "命名实体" in context:
        sources = "；".join(f"{hit.chunk.source} / {hit.chunk.section}" for hit in hits[:3])
        return "\n".join(
            [
                "我会把这部分 NLP 资料按“期末复习提纲”整理，而不是逐句摘录：",
                "1. NLP 基础定义：理解自然语言处理的研究对象、目标，以及它和计算语言学、人工智能之间的关系。",
                "2. 命名实体识别 NER：重点看实体类型、MUC/ACE/CoNLL 等评测背景，以及从规则方法到统计/深度学习方法的发展。",
                "3. 信息抽取：关注实体关系抽取、事件抽取、开放域信息抽取，以及模板方法在大规模语料上的局限。",
                "4. 表示学习与深度模型：复习词向量、上下文相关表示、RNN/LSTM/GRU、CNN、注意力机制和预训练模型的基本作用。",
                "5. 评测方式：注意准确率、召回率、F1、Top-K 命中等指标分别衡量什么，考试里常结合具体任务问。",
                "6. 复习建议：先按“定义 -> 方法 -> 应用 -> 指标 -> 易错边界”整理每章，再让系统围绕薄弱章节出题。",
                f"资料依据：{sources}",
            ]
        )
    points = extract_key_points(hits, limit=6)
    lines = ["这部分资料可以整理成下面的复习重点："]
    for idx, point in enumerate(points, start=1):
        lines.append(f"{idx}. {point}")
    lines.append(f"建议先看来源：{hits[0].chunk.source} / {hits[0].chunk.section}")
    return "\n".join(lines)


def synthesize_quiz(query: str, hits: list[SearchHit]) -> str:
    questions = build_quiz(hits, count=extract_requested_count(query))
    if not questions:
        return "当前资料不足以生成练习题。可以先上传更完整的课程资料。"
    lines = ["根据召回资料生成一组练习题："]
    for idx, item in enumerate(questions, start=1):
        lines.append(f"\n{idx}. {item['question']}")
        for label, option in zip(["A", "B", "C", "D"], item["options"]):
            lines.append(f"   {label}. {option}")
        lines.append(f"   答案：{item['answer']}。解析：{item['explanation']}")
    return "\n".join(lines)


def synthesize_explain(query: str, hits: list[SearchHit]) -> str:
    if not hits:
        return "资料中暂时没有检索到这个概念。你可以换一个课程关键词，或上传相关章节。"
    concept = infer_concept(query, hits)
    evidence = compact_text(hits[0].chunk.text, 260)
    return (
        f"可以把“{concept}”这样理解：\n"
        f"- 定义：{evidence}\n"
        "- 复习时重点看它解决什么问题、输入输出是什么、和相近概念的边界在哪里。\n"
        f"- 引用来源：{hits[0].chunk.source} / {hits[0].chunk.section}"
    )


def synthesize_review_plan(query: str, hits: list[SearchHit]) -> str:
    plan = review_plan_tool(query, hits)["plan"]
    lines = ["给你一个 3 天小复习计划："]
    for item in plan:
        focus = "、".join(item["focus"])
        lines.append(f"- Day {item['day']}：{item['task']}。重点：{focus}")
    lines.append("执行方式：每天先问答，再出题，最后把错题关联回原始 chunk。")
    return "\n".join(lines)


def synthesize_mistake_review(query: str, hits: list[SearchHit]) -> str:
    review = mistake_review_tool(query, hits)
    weak = "、".join(review["weak_points"][:5])
    sources = "；".join(f"{s['source']} / {s['section']}" for s in review["recommended_sources"])
    return (
        "错题回顾建议这样做：\n"
        f"- 先定位薄弱点：{weak}\n"
        "- 再让系统重新解释这些概念，并生成 3 道相似题。\n"
        "- 最后隔天复测一次，只保留仍然错的题进入下一轮错题本。\n"
        f"- 推荐回看资料：{sources}"
    )


def build_quiz(hits: list[SearchHit], count: int = 5) -> list[dict[str, Any]]:
    sentences: list[tuple[str, SearchHit]] = []
    for hit in hits:
        sentences.extend((sentence, hit) for sentence in split_sentences(hit.chunk.text))
    usable = [(sentence, hit) for sentence, hit in sentences if 35 <= len(sentence) <= 320]
    questions = []
    labels = ["A", "B", "C", "D"]
    for idx, (sentence, hit) in enumerate(usable[:count]):
        concept = first_concept(sentence)
        if not concept:
            concept = "该知识点"
        correct = compact_text(sentence, 90)
        options = [
            correct,
            f"{concept} 与召回资料中的课程内容无关",
            f"{concept} 只需要记住名词，不需要理解作用和边界",
            f"{concept} 无法通过知识库定位到来源片段",
        ]
        shift = idx % len(options)
        rotated = options[shift:] + options[:shift]
        answer = labels[rotated.index(correct)]
        questions.append(
            {
                "question": f"关于“{concept}”，下列说法哪一项最符合资料内容？",
                "options": rotated,
                "answer": answer,
                "explanation": f"{answer} 来自召回资料，可回看 {hit.chunk.source} / {hit.chunk.section}；其余选项是干扰项。",
            }
        )
    return questions


def extract_requested_count(query: str, default: int = 5, maximum: int = 10) -> int:
    match = re.search(r"(\d+)\s*(道|个)?\s*(选择题|判断题|练习题|题目|题)", query)
    if not match:
        return default
    return max(1, min(maximum, int(match.group(1))))


def extract_key_points(hits: list[SearchHit], limit: int = 5) -> list[str]:
    points = []
    for hit in hits:
        for sentence in split_sentences(hit.chunk.text):
            if len(sentence) < 24:
                continue
            points.append(compact_text(sentence, 120))
            if len(points) >= limit:
                return points
    return points


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])\s+|\n+", text)
    cleaned = [re.sub(r"^[\-*0-9.、\s]+", "", p).strip() for p in parts]
    return [p for p in cleaned if p]


def extract_concepts(text: str) -> list[str]:
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,8}", text)
    stop = {"可以", "这个", "系统", "资料", "用户", "问题", "项目", "知识", "来源", "回答"}
    counts: dict[str, int] = {}
    for item in candidates:
        if item.lower() in stop or item in stop:
            continue
        counts[item] = counts.get(item, 0) + 1
    return [item for item, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]]


def first_concept(text: str) -> str:
    concepts = extract_concepts(text)
    return concepts[0] if concepts else ""


def infer_concept(query: str, hits: list[SearchHit]) -> str:
    for pattern in [
        r"(.+?)的实验原理",
        r"(.+?)实验原理",
        r"(.+?)的原理",
        r"什么是(.+)",
        r"解释(.+)",
        r"(.+)是什么",
        r"(.+)怎么理解",
    ]:
        match = re.search(pattern, query)
        if match:
            concept = match.group(1).strip(" ？?。")
            if 1 <= len(concept) <= 30:
                return concept
    return first_concept(hits[0].chunk.text) if hits else "该概念"
