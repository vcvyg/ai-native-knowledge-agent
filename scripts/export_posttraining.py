from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.agent import KnowledgeAgent
from app.evaluation import load_cases
from app.rag_engine import KnowledgeBase
from app.trajectory_learning import (
    TrajectoryOracle,
    build_preference_pairs,
    build_sft_example,
    build_training_record,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run labeled cases and export explicit Agent trajectories for post-training."
    )
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation" / "cases.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "posttraining-trajectories.jsonl",
    )
    parser.add_argument(
        "--sft-output",
        type=Path,
        default=ROOT / "reports" / "posttraining-sft.jsonl",
    )
    parser.add_argument(
        "--preference-output",
        type=Path,
        default=ROOT / "reports" / "posttraining-preferences.jsonl",
    )
    args = parser.parse_args()

    os.environ.setdefault("AI_AGENT_VECTOR_BACKEND", "memory")
    os.environ.setdefault("AI_AGENT_RERANKER", "heuristic")
    agent = KnowledgeAgent(KnowledgeBase(ROOT / "data" / "knowledge_base"))
    records = []
    for case in load_cases(args.cases):
        response = agent.ask(case.query)
        oracle = TrajectoryOracle(
            expected_intent=case.expected_intent,
            expected_sources=case.expected_sources,
            expected_tools=case.expected_tools,
            expect_low_evidence=case.expect_low_evidence,
        )
        records.append(build_training_record(case.query, response, oracle))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.sft_output.parent.mkdir(parents=True, exist_ok=True)
    args.preference_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    sft_examples = [build_sft_example(record) for record in records]
    args.sft_output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in sft_examples),
        encoding="utf-8",
    )
    preference_pairs = build_preference_pairs(records)
    args.preference_output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in preference_pairs),
        encoding="utf-8",
    )
    mean_reward = sum(record["reward"]["total"] for record in records) / max(len(records), 1)
    print(
        json.dumps(
            {
                "records": len(records),
                "sft_examples": len(sft_examples),
                "preference_pairs": len(preference_pairs),
                "mean_reward": round(mean_reward, 4),
                "output": str(args.output),
                "sft_output": str(args.sft_output),
                "preference_output": str(args.preference_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
