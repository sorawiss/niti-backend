import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from app.llm.config import (
    QueryRewriteWorkflow,
    get_query_rewrite_settings,
)
from app.llm.factory import create_llm_gateway
from app.rag.query import retrieve_laws
from app.rag.rewrite import QueryRewriteService


DEFAULT_CASES = Path("tests/evaluation/query_rewrite_cases.json")


def _recall(results: list[dict[str, Any]], expected: set[str]) -> float:
    returned = {
        str((result.get("metadata") or {}).get("section"))
        for result in results
    }
    return len(returned & expected) / len(expected)


async def evaluate(cases_path: Path, top_k: int) -> int:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    gateway = create_llm_gateway()
    settings = get_query_rewrite_settings().model_copy(
        update={"rag_enabled": True}
    )
    rewriter = QueryRewriteService(gateway, settings)
    rows: list[dict[str, Any]] = []

    try:
        for case in cases:
            original_results = await asyncio.to_thread(
                retrieve_laws,
                query=case["query"],
                top_k=top_k,
            )
            rewrite = await rewriter.rewrite(
                case["query"],
                workflow=QueryRewriteWorkflow.RAG,
            )
            rewritten_results = await asyncio.to_thread(
                retrieve_laws,
                query=rewrite.effective_query,
                top_k=top_k,
            )
            expected = set(case["expected_sections"])
            rows.append(
                {
                    "id": case["id"],
                    "ambiguous": bool(case["ambiguous"]),
                    "rewrite_status": rewrite.status.value,
                    "baseline_recall": _recall(original_results, expected),
                    "rewritten_recall": _recall(rewritten_results, expected),
                }
            )
    finally:
        await gateway.close()

    baseline = sum(row["baseline_recall"] for row in rows) / len(rows)
    rewritten = sum(row["rewritten_recall"] for row in rows) / len(rows)
    ambiguous_rows = [row for row in rows if row["ambiguous"]]
    ambiguous_baseline = sum(
        row["baseline_recall"] for row in ambiguous_rows
    ) / len(ambiguous_rows)
    ambiguous_rewritten = sum(
        row["rewritten_recall"] for row in ambiguous_rows
    ) / len(ambiguous_rows)
    relative_gain = (
        (ambiguous_rewritten - ambiguous_baseline) / ambiguous_baseline
        if ambiguous_baseline
        else (1.0 if ambiguous_rewritten > 0 else 0.0)
    )

    summary = {
        "top_k": top_k,
        "baseline_recall": baseline,
        "rewritten_recall": rewritten,
        "ambiguous_relative_gain": relative_gain,
        "passes_no_regression": rewritten >= baseline,
        "passes_ambiguous_gain": relative_gain >= 0.05,
        "cases": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(
        (
            summary["passes_no_regression"],
            summary["passes_ambiguous_gain"],
        )
    ) else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(evaluate(args.cases, args.top_k)))


if __name__ == "__main__":
    main()
