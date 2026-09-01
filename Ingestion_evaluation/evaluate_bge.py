"""Semantic retrieval evaluation for the chunking experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EXCLUDED_NUMBERS = {46, 47, 48, 49, 50, 51, 52, 58, 60}
TOP_KS = (1, 3, 5)
CONFIG_NAMES = ("350w_50o", "250w_40o", "150w_30o")


def question_number(value: object) -> int | None:
    """Extract the first integer from a benchmark question identifier."""
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def parse_gold_sources(value: object) -> set[str]:
    """Parse a delimited spreadsheet cell into normalized gold source IDs."""
    if pd.isna(value) or not str(value).strip():
        return set()
    return {part.strip() for part in re.split(r"[,;|+]", str(value)) if part.strip()}


def load_questions(project_root: Path) -> tuple[pd.DataFrame, dict]:
    """Load scorable benchmark questions and return dataset audit counts."""
    path = project_root / "evaluation_questions" / "benchmark_60_with_draft_gold_answers.xlsx"
    all_questions = pd.read_excel(path)
    all_questions["question_number"] = all_questions["question_id"].map(question_number)
    all_questions["gold_sources"] = all_questions["gold_source_ids"].map(parse_gold_sources)
    included = all_questions.loc[~all_questions["question_number"].isin(EXCLUDED_NUMBERS)].copy()
    scored = included.loc[included["gold_sources"].map(bool)].copy()
    audit = {
        "workbook_rows": len(all_questions),
        "excluded_rows": len(all_questions) - len(included),
        "remaining_rows": len(included),
        "scored_rows": len(scored),
        "unscored_rows": len(included) - len(scored),
    }
    return scored, audit


def load_chunks(path: Path) -> pd.DataFrame:
    """Load a JSONL chunk corpus into the evaluation dataframe schema."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame([
        {
            "chunk_id": record["chunk_id"],
            "text": record["text"],
            "source_id": record["metadata"]["source_id"],
            "title": record["metadata"]["title"],
            "heading_path": " > ".join(record["metadata"].get("heading_path", [])),
        }
        for record in records
    ])


def evaluate_configuration(
    name: str,
    chunks: pd.DataFrame,
    questions: pd.DataFrame,
    model: SentenceTransformer,
    query_embeddings: np.ndarray,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Score one chunking configuration using source-level relevance labels."""
    document_embeddings = model.encode(
        chunks["text"].tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    scores = query_embeddings @ document_embeddings.T
    rows = []
    for query_pos, (_, question) in enumerate(questions.iterrows()):
        ranking = np.argsort(-scores[query_pos], kind="stable")
        ranked = chunks.iloc[ranking].reset_index(drop=True)
        gold = question["gold_sources"]
        relevant_positions = [
            rank
            for rank, source_id in enumerate(ranked["source_id"], start=1)
            if source_id in gold
        ]
        row = {
            "configuration": name,
            "question_id": question["question_id"],
            "category": question["category"],
            "question": question["question"],
            "gold_sources": ", ".join(sorted(gold)),
            "mrr": 1 / relevant_positions[0] if relevant_positions else 0.0,
            "first_relevant_rank": relevant_positions[0] if relevant_positions else np.nan,
            "top5_sources": ", ".join(ranked.head(5)["source_id"]),
            "top5_chunk_ids": ", ".join(ranked.head(5)["chunk_id"]),
        }
        for k in TOP_KS:
            retrieved_sources = set(ranked.head(k)["source_id"])
            row[f"recall@{k}"] = len(gold & retrieved_sources) / len(gold)
        rows.append(row)
    return pd.DataFrame(rows)


def run_evaluation(project_root: Path, batch_size: int = 32) -> dict:
    """Evaluate all chunking configurations and persist detailed summaries."""
    questions, audit = load_questions(project_root)
    corpora = {
        name: load_chunks(project_root / "data" / "experiments" / name / "chunks.jsonl")
        for name in CONFIG_NAMES
    }
    model = SentenceTransformer(MODEL_NAME)
    query_embeddings = model.encode(
        [QUERY_PREFIX + question for question in questions["question"].tolist()],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    details = pd.concat(
        [evaluate_configuration(name, chunks, questions, model, query_embeddings, batch_size)
         for name, chunks in corpora.items()],
        ignore_index=True,
    )
    metric_columns = [f"recall@{k}" for k in TOP_KS] + ["mrr"]
    summary = details.groupby("configuration")[metric_columns].mean().sort_values("mrr", ascending=False)
    by_category = details.groupby(["configuration", "category"])[metric_columns].mean()
    output_dir = project_root / "data" / "experiments" / "evaluation_bge"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.reset_index().to_csv(output_dir / "retrieval_metrics.csv", index=False)
    by_category.reset_index().to_csv(output_dir / "retrieval_metrics_by_category.csv", index=False)
    details.to_csv(output_dir / "retrieval_details.csv", index=False)
    run_info = {
        **audit,
        "model": MODEL_NAME,
        "query_prefix": QUERY_PREFIX,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "max_sequence_length": model.max_seq_length,
        "batch_size": batch_size,
        "normalized_embeddings": True,
        "similarity": "dot product (equivalent to cosine for normalized embeddings)",
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    return {
        "questions": questions,
        "corpora": corpora,
        "details": details,
        "summary": summary,
        "by_category": by_category,
        "run_info": run_info,
        "output_dir": output_dir,
    }

