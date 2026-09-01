"""Build an all-question BGE+MMR retrieval bundle for sufficiency annotation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from evaluate_bge import QUERY_PREFIX, load_chunks, parse_gold_sources, question_number
from evaluate_diversification import CANDIDATE_K, FINAL_K, MODEL_NAME, mmr_select

MMR_LAMBDA = 0.9


def build_bundle(project_root: Path, batch_size: int = 32) -> Path:
    """Retrieve top-five chunks for every question and write an annotation bundle."""
    workbook = project_root / "evaluation_questions" / "benchmark_60_with_draft_gold_answers.xlsx"
    questions = pd.read_excel(workbook)
    questions["question_number"] = questions["question_id"].map(question_number)
    questions["gold_sources"] = questions["gold_source_ids"].map(parse_gold_sources)
    questions = questions.sort_values("question_number").reset_index(drop=True)
    chunks = load_chunks(project_root / "data/experiments/350w_50o/chunks.jsonl")

    model = SentenceTransformer(MODEL_NAME)
    query_embeddings = model.encode(
        [QUERY_PREFIX + text for text in questions["question"].tolist()],
        batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=True,
    )
    document_embeddings = model.encode(
        chunks["text"].tolist(), batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )
    scores = query_embeddings @ document_embeddings.T
    similarity_rankings = np.argsort(-scores, axis=1, kind="stable")

    records = []
    for position, question in questions.iterrows():
        candidates = similarity_rankings[position, :CANDIDATE_K]
        selected = mmr_select(
            candidates, scores[position], document_embeddings, FINAL_K, MMR_LAMBDA
        )
        retrieved = []
        for rank, chunk_index in enumerate(selected, start=1):
            chunk = chunks.iloc[chunk_index]
            retrieved.append({
                "rank": rank,
                "score": round(float(scores[position, chunk_index]), 6),
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "title": chunk["title"],
                "heading_path": chunk["heading_path"],
                "text": chunk["text"],
            })
        records.append({
            "question_id": question["question_id"],
            "category": question["category"],
            "difficulty": question["difficulty"],
            "expected_behavior": question["expected_behavior"],
            "question": question["question"],
            "gold_source_ids": sorted(question["gold_sources"]),
            "gold_answer": "" if pd.isna(question.get("gold_answer")) else str(question.get("gold_answer")),
            "gold_evidence_summary": "" if pd.isna(question.get("gold_evidence_summary")) else str(question.get("gold_evidence_summary")),
            "retrieval_method": f"{MODEL_NAME} top-{CANDIDATE_K} + MMR lambda={MMR_LAMBDA} top-{FINAL_K}",
            "retrieved_chunks": retrieved,
            "evidence_sufficiency": "",
            "correct_system_behavior": "",
            "annotation_rationale": "",
            "annotation_status": "pending_review",
        })

    output_dir = project_root / "data" / "experiments" / "evidence_sufficiency"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "retrieval_bundle_all_60.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(build_bundle(root))
