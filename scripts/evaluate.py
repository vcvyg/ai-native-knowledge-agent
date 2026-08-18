from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.agent import KnowledgeAgent
from app.evaluation import evaluate_agent, load_cases
from app.rag_engine import KnowledgeBase

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline RAG + Agent evaluation set")
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "evaluation" / "cases.jsonl",
        help="JSONL evaluation set",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "evaluation-results.json",
        help="Result JSON path",
    )
    args = parser.parse_args()

    os.environ.setdefault("AI_AGENT_VECTOR_BACKEND", "memory")
    os.environ.setdefault("AI_AGENT_RERANKER", "heuristic")
    kb = KnowledgeBase(ROOT / "data" / "knowledge_base")
    agent = KnowledgeAgent(kb)
    result = evaluate_agent(agent, load_cases(args.cases))
    stats = kb.stats()
    result["configuration"] = {
        "vector_backend": stats["vector_backend"],
        "embedding_model": stats["embedding_model"],
        "embedding_dimensions": stats["embedding_dimensions"],
        "reranker": stats["reranker"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
