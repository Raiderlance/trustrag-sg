"""Compare similarity-only and MMR selection from the same BGE top-20 pool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from evaluate_bge import QUERY_PREFIX, load_chunks, load_questions
from evaluate_retrievers import score_rankings

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CANDIDATE_K = 20
FINAL_K = 5
PRIMARY_LAMBDA = 0.9
LAMBDA_SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9)


def mmr_select(
    candidate_indices: np.ndarray,
    query_scores: np.ndarray,
    document_embeddings: np.ndarray,
    final_k: int,
    lambda_mult: float,
) -> np.ndarray:
    """Select candidates in MMR order using normalized embedding dot products."""
    if not 0 <= lambda_mult <= 1:
        raise ValueError("lambda_mult must be between 0 and 1")
    remaining = list(candidate_indices)
    selected: list[int] = []
    while remaining and len(selected) < final_k:
        if not selected:
            best = max(remaining, key=lambda index: query_scores[index])
        else:
            selected_embeddings = document_embeddings[selected]

            def mmr_score(index: int) -> float:
                redundancy = float(np.max(document_embeddings[index] @ selected_embeddings.T))
                return lambda_mult * float(query_scores[index]) - (1 - lambda_mult) * redundancy

            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return np.asarray(selected, dtype=int)


def build_mmr_rankings(
    similarity_rankings: np.ndarray,
    scores: np.ndarray,
    document_embeddings: np.ndarray,
    lambda_mult: float,
) -> np.ndarray:
    """Place MMR-selected candidates first while retaining a full ranking."""
    rows = []
    for query_pos in range(len(similarity_rankings)):
        candidates = similarity_rankings[query_pos, :CANDIDATE_K]
        selected = mmr_select(
            candidates, scores[query_pos], document_embeddings, FINAL_K, lambda_mult
        )
        selected_set = set(selected.tolist())
        remainder = np.asarray(
            [index for index in similarity_rankings[query_pos] if index not in selected_set],
            dtype=int,
        )
        rows.append(np.concatenate([selected, remainder]))
    return np.vstack(rows)


def run_diversification_evaluation(project_root: Path, batch_size: int = 32) -> dict:
    """Compare similarity ranking with an MMR lambda sweep and save results."""
    questions, audit = load_questions(project_root)
    chunks = load_chunks(project_root / "data/experiments/350w_50o/chunks.jsonl")
    model = SentenceTransformer(MODEL_NAME)
    query_embeddings = model.encode(
        [QUERY_PREFIX + question for question in questions["question"].tolist()],
        batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=True,
    )
    document_embeddings = model.encode(
        chunks["text"].tolist(), batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )
    scores = query_embeddings @ document_embeddings.T
    similarity_rankings = np.argsort(-scores, axis=1, kind="stable")

    detail_frames = [
        score_rankings("Similarity top-5", chunks, questions, similarity_rankings, CANDIDATE_K)
    ]
    sweep_rows = []
    for lambda_mult in LAMBDA_SWEEP:
        rankings = build_mmr_rankings(similarity_rankings, scores, document_embeddings, lambda_mult)
        label = f"MMR λ={lambda_mult:.1f}"
        frame = score_rankings(label, chunks, questions, rankings, CANDIDATE_K)
        detail_frames.append(frame)
        sweep_rows.append(frame)
    details = pd.concat(detail_frames, ignore_index=True)
    metric_columns = ["recall@1", "recall@3", "recall@5", "mrr"]
    summary = details.groupby("method", sort=False)[metric_columns].mean()
    diversity_columns = [
        "unique_sources@5", "duplicate_fraction@5", "max_source_share@5",
        "candidate_recall@20", "diversification_candidate",
    ]
    diversity = details.groupby("method", sort=False)[diversity_columns].mean()
    by_category = details.groupby(["method", "category"])[metric_columns].mean()

    baseline = details.loc[details["method"] == "Similarity top-5"].set_index("question_id")
    primary_label = f"MMR λ={PRIMARY_LAMBDA:.1f}"
    primary = details.loc[details["method"] == primary_label].set_index("question_id")
    paired = baseline[["question", "category", "gold_sources", "recall@1", "recall@3", "recall@5", "mrr",
                       "top5_sources", "top5_chunk_ids", "unique_sources@5"]].copy()
    paired.columns = [f"baseline_{column}" for column in paired.columns]
    for column in ["recall@1", "recall@3", "recall@5", "mrr", "top5_sources", "top5_chunk_ids", "unique_sources@5"]:
        paired[f"mmr_{column}"] = primary[column]
    for metric in ["recall@1", "recall@3", "recall@5", "mrr", "unique_sources@5"]:
        paired[f"delta_{metric}"] = paired[f"mmr_{metric}"] - paired[f"baseline_{metric}"]
    paired = paired.reset_index()

    output_dir = project_root / "data/experiments/evaluation_diversification_350w_50o"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.reset_index().to_csv(output_dir / "retrieval_metrics.csv", index=False)
    diversity.reset_index().to_csv(output_dir / "diversification_metrics.csv", index=False)
    by_category.reset_index().to_csv(output_dir / "retrieval_metrics_by_category.csv", index=False)
    details.to_csv(output_dir / "retrieval_details_all_lambdas.csv", index=False)
    paired.to_csv(output_dir / "paired_similarity_vs_mmr_0.9.csv", index=False)
    run_info = {
        **audit,
        "corpus": "350 words / 50 overlap",
        "chunks": len(chunks),
        "model": MODEL_NAME,
        "candidate_k": CANDIDATE_K,
        "final_k": FINAL_K,
        "primary_lambda": PRIMARY_LAMBDA,
        "lambda_sweep": list(LAMBDA_SWEEP),
        "mmr_formula": "lambda * query_similarity - (1-lambda) * max_similarity_to_selected",
        "normalized_embeddings": True,
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    return {
        "summary": summary,
        "diversity": diversity,
        "by_category": by_category,
        "details": details,
        "paired": paired,
        "run_info": run_info,
        "output_dir": output_dir,
    }
