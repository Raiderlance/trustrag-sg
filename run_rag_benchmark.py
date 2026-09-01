"""Run development questions through retrieval, evidence assessment, and Gemini."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from rag import BaselineRAGPipeline, DenseMMRRetriever, GeminiGenerator, RAGPipeline


DEFAULT_QUESTIONS = (
    "evaluation_questions/benchmark_60_evidence_sufficiency_annotations.xlsx"
)
DEFAULT_OUTPUT = "data/experiments/rag_gemini_benchmark/results.jsonl"


def load_completed(path: Path) -> set[str]:
    """Return question IDs with successful records in a resumable output file."""
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if not record.get("error"):
                completed.add(str(record["question_id"]))
    return completed


def main() -> None:
    """Run gated or baseline RAG over the benchmark and save JSONL and CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--sheet", default="evidence_sufficiency_annotation")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=("gated", "baseline"), default="gated")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--request-interval", type=float, default=13.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--mmr-lambda", type=float, default=0.9)
    parser.add_argument("--model", default=None)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    default_output = (
        DEFAULT_OUTPUT if args.mode == "gated"
        else "data/experiments/rag_gemini_benchmark/baseline_results.jsonl"
    )
    output = Path(args.output or default_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.restart and output.exists():
        output.unlink()
    completed = load_completed(output)

    questions = pd.read_excel(args.questions, sheet_name=args.sheet)
    required = {"question_id", "question"}
    missing = required - set(questions.columns)
    if missing:
        raise ValueError(f"Question workbook is missing columns: {sorted(missing)}")
    questions = questions.loc[questions["question"].notna()].copy()
    if args.limit is not None:
        questions = questions.head(args.limit)

    retriever = DenseMMRRetriever.from_jsonl(
        args.chunks, candidate_k=args.candidate_k, lambda_mult=args.mmr_lambda
    )
    generator = GeminiGenerator(model=args.model, min_request_interval=args.request_interval)
    pipeline = (
        RAGPipeline(retriever, generator)
        if args.mode == "gated"
        else BaselineRAGPipeline(retriever, generator)
    )

    for position, (_, row) in enumerate(questions.iterrows(), start=1):
        question_id = str(row["question_id"])
        if question_id in completed:
            print(f"[{position}/{len(questions)}] {question_id}: already complete")
            continue
        # Only these two explicit columns enter the runtime. Gold answers, annotations,
        # and retrieved text in the workbook remain evaluation labels, not instructions.
        question = str(row["question"])
        record = {
            "question_id": question_id,
            "category": None if pd.isna(row.get("category")) else str(row.get("category")),
            "question": question,
            "model": pipeline.generator.model,
        }
        try:
            if args.mode == "gated":
                answer, chunks, assessment = pipeline.ask_with_assessment(question, args.top_k)
            else:
                answer, chunks = pipeline.ask(question, args.top_k)
                assessment = None
            record.update({
                "mode": args.mode,
                "assessment": assessment.model_dump() if assessment else None,
                "answer": answer,
                "retrieved": [
                    {
                        "rank": rank,
                        "chunk_id": item.chunk.get("chunk_id"),
                        "source_id": item.chunk.get("metadata", {}).get("source_id"),
                        "title": item.chunk.get("metadata", {}).get("title"),
                        "score": item.score,
                    }
                    for rank, item in enumerate(chunks, start=1)
                ],
                "error": None,
            })
            status = assessment.sufficiency if assessment else "ANSWERED"
        except Exception as error:  # Preserve progress and expose per-question API failures.
            record["error"] = f"{type(error).__name__}: {error}"
            status = "ERROR"
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{position}/{len(questions)}] {question_id}: {status}", flush=True)
        if args.delay:
            time.sleep(args.delay)

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    latest_by_question = {str(record["question_id"]): record for record in records}
    flat_rows = []
    for record in latest_by_question.values():
        assessment = record.get("assessment") or {}
        flat_rows.append({
            "question_id": record["question_id"],
            "category": record.get("category"),
            "question": record["question"],
            "model": record.get("model"),
            "mode": record.get("mode"),
            "sufficiency": assessment.get("sufficiency"),
            "support": assessment.get("support"),
            "completeness": assessment.get("completeness"),
            "conditions": assessment.get("conditions"),
            "temporal_validity": assessment.get("temporal_validity"),
            "reason": assessment.get("reason"),
            "supporting_chunk_ids": ", ".join(assessment.get("supporting_chunk_ids", [])),
            "retrieved_chunk_ids": ", ".join(
                str(item.get("chunk_id")) for item in record.get("retrieved", [])
            ),
            "answer": record.get("answer"),
            "error": record.get("error"),
        })
    csv_path = output.with_suffix(".csv")
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    print(f"Saved JSONL: {output}")
    print(f"Saved CSV:   {csv_path}")


if __name__ == "__main__":
    main()
