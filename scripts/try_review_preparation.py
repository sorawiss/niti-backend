"""Inspect PR1 using a local DOCX; live services run only when invoked."""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from zipfile import BadZipFile

from docx.opc.exceptions import PackageNotFoundError
from dotenv import load_dotenv
from lxml.etree import XMLSyntaxError

from app.documents.models import ParsedDocument
from app.documents.parser import parse_docx
from app.llm.base import LLMGateway
from app.llm.errors import LLMGatewayError
from app.llm.factory import create_llm_gateway
from app.reviews.models import ReviewPreparation
from app.reviews.planner import ReviewInputError, ReviewPlanner, ReviewPlanningError
from app.reviews.retrieval import ReviewRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


async def run(
    document_path: Path,
    output_dir: Path,
    *,
    top_k: int = 5,
    gateway: LLMGateway | None = None,
    retriever: ReviewRetriever | None = None,
) -> int:
    """Write one isolated run. Supplied gateway resources remain caller-owned."""
    document_path = document_path.resolve()
    output_dir = output_dir.resolve()
    if not document_path.is_file():
        raise FileNotFoundError(f"DOCX not found: {document_path}")
    if document_path.suffix.lower() != ".docx":
        raise ValueError("Input must be a .docx file")
    if document_path.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("DOCX exceeds the upload API's 5 MiB limit")
    # Refuse to overwrite a previous run, including its failed-planner evidence.
    output_dir.mkdir(parents=True, exist_ok=False)
    stage = "parsing"
    owned_gateway = False
    print(f"Artifacts: {output_dir}", flush=True)
    try:
        document = ParsedDocument(
            document_id=str(uuid4()),
            filename=document_path.name,
            blocks=parse_docx(document_path.read_bytes()),
        )
        write_json(output_dir / "document.json", document.model_dump(mode="json"))
        if not document.blocks:
            raise ReviewInputError("DOCX contains no supported content blocks")
        print(f"Parsed {len(document.blocks)} blocks", flush=True)

        stage = "configuration"
        if gateway is None:
            load_dotenv(PROJECT_ROOT / ".env")
            gateway = create_llm_gateway()
            owned_gateway = True

        stage = "planning"
        print("Planning with the current prompt (one LLM call, no retries)...", flush=True)
        plan, response = await ReviewPlanner(gateway).plan(document)
        write_json(output_dir / "plan.json", plan.model_dump(mode="json"))
        write_json(output_dir / "planner-response.json", response.model_dump(mode="json"))
        query_count = sum(len(topic.queries) for topic in plan.topics)
        print(f"Plan accepted: {len(plan.topics)} topics, {query_count} queries", flush=True)

        stage = "retrieval"
        print("Searching the existing RAG...", flush=True)
        # Sequential retrieval avoids concurrent cold loading of the embedding model.
        retrieval = await (retriever or ReviewRetriever(top_k=top_k, concurrency=1)).retrieve(plan)
        result = ReviewPreparation(
            document_id=document.document_id,
            filename=document.filename,
            status="ready" if all(q.status == "retrieved" for q in retrieval.queries) else "partial",
            plan=plan,
            retrieval=retrieval,
            model=response.model,
            provider=response.provider,
            usage=response.usage,
        )
        write_json(output_dir / "result.json", result.model_dump(mode="json"))
        write_json(output_dir / "run.json", {
            "stage": "completed", "status": result.status,
            "topic_count": len(plan.topics), "query_count": query_count,
            "source_count": len(retrieval.sources),
        })
        print(f"Status: {result.status}; unique legal sources: {len(retrieval.sources)}")
        for query in retrieval.queries:
            print(f"  {query.query_id}: {query.status} ({len(query.source_ids)} sources)")
        print(f"Result: {output_dir / 'result.json'}")
        return 0 if result.status == "ready" else 2
    except ReviewPlanningError as error:
        if error.response is not None:
            write_json(output_dir / "planner-response.json", error.response.model_dump(mode="json"))
        if error.plan is not None:
            write_json(output_dir / "rejected-plan.json", error.plan.model_dump(mode="json"))
        write_json(output_dir / "run.json", {
            "stage": stage, "status": "failed", "code": "invalid_review_plan",
            "reason": str(error), "query_count": error.query_count,
            "max_queries": error.max_queries,
        })
        print(f"Planning rejected: {error}", file=sys.stderr)
        if error.query_count is not None:
            print(f"Queries: {error.query_count}; limit: {error.max_queries}", file=sys.stderr)
        print(f"Diagnostics: {output_dir / 'run.json'}", file=sys.stderr)
        return 1
    except LLMGatewayError as error:
        write_json(output_dir / "run.json", {
            "stage": stage, "status": "failed", "code": error.code,
            "retryable": error.retryable,
        })
        print(f"LLM error: {error.code}; diagnostics: {output_dir / 'run.json'}", file=sys.stderr)
        return 1
    except (ValueError, OSError, BadZipFile, PackageNotFoundError, XMLSyntaxError, KeyError) as error:
        write_json(output_dir / "run.json", {
            "stage": stage, "status": "failed", "code": type(error).__name__,
        })
        print(f"Preparation failed at {stage}: {type(error).__name__}", file=sys.stderr)
        return 1
    finally:
        if owned_gateway and gateway is not None:
            await gateway.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PR1 with a DOCX and live services; save plans and failure diagnostics."
    )
    parser.add_argument(
        "document", nargs="?", type=Path,
        default=PROJECT_ROOT / "docs" / "Thai_Employment_Contract_Mock.docx",
    )
    parser.add_argument("--output-dir", type=Path, help="A new directory; existing paths are refused.")
    parser.add_argument("--top-k", type=int, choices=range(1, 21), default=5)
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output_dir = args.output_dir or PROJECT_ROOT / "tmp" / "review-preparation" / run_id
    try:
        return asyncio.run(run(args.document, output_dir, top_k=args.top_k))
    except (ValueError, OSError) as error:
        print(f"Cannot start: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
