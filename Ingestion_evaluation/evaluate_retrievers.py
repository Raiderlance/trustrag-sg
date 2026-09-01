"""Compare lexical, dense, and reranked retrieval on the fixed 350/50 corpus."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from evaluate_bge import QUERY_PREFIX, load_chunks, load_questions

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
CANDIDATE_K = 20
TOP_KS = (1, 3, 5)


def score_rankings(
    method: str,
    chunks: pd.DataFrame,
    questions: pd.DataFrame,
    rankings: np.ndarray,
    candidate_k: int = CANDIDATE_K,
) -> pd.DataFrame:
    """Compute retrieval, diversity, and error metrics for ranked chunks."""
    rows = []
    for query_pos, (_, question) in enumerate(questions.iterrows()):
        ranked = chunks.iloc[rankings[query_pos]].reset_index(drop=True)
        gold = question["gold_sources"]
        relevant_positions = [
            rank for rank, source_id in enumerate(ranked["source_id"], start=1)
            if source_id in gold
        ]
        top5 = ranked.head(5)
        top20 = ranked.head(candidate_k)
        top5_sources = top5["source_id"].tolist()
        unique_top5 = set(top5_sources)
        source_counts = pd.Series(top5_sources).value_counts()
        row = {
            "method": method,
            "question_id": question["question_id"],
            "category": question["category"],
            "question": question["question"],
            "gold_sources": ", ".join(sorted(gold)),
            "mrr": 1 / relevant_positions[0] if relevant_positions else 0.0,
            "first_relevant_rank": relevant_positions[0] if relevant_positions else np.nan,
            "candidate_recall@20": len(gold & set(top20["source_id"])) / len(gold),
            "unique_sources@5": len(unique_top5),
            "duplicate_fraction@5": 1 - len(unique_top5) / 5,
            "max_source_share@5": source_counts.iloc[0] / 5,
            "top5_sources": ", ".join(top5_sources),
            "top5_chunk_ids": ", ".join(top5["chunk_id"]),
            "top20_sources": ", ".join(top20["source_id"]),
        }
        for k in TOP_KS:
            row[f"recall@{k}"] = len(gold & set(ranked.head(k)["source_id"])) / len(gold)
        missing_gold = gold - unique_top5
        available_missing = missing_gold & set(top20["source_id"])
        row["missing_gold_at_5"] = ", ".join(sorted(missing_gold))
        row["missing_gold_available_at_20"] = ", ".join(sorted(available_missing))
        row["diversification_candidate"] = bool(
            available_missing and row["duplicate_fraction@5"] >= 0.4
        )
        if row["recall@5"] == 1:
            row["error_type"] = "success"
        elif row["candidate_recall@20"] == 0:
            row["error_type"] = "candidate_retrieval_miss"
        elif row["diversification_candidate"]:
            row["error_type"] = "duplicate_crowding"
        elif row["candidate_recall@20"] > row["recall@5"]:
            row["error_type"] = "ranking_miss"
        else:
            row["error_type"] = "partial_or_missing_gold"
        rows.append(row)
    return pd.DataFrame(rows)


def run_comparison(project_root: Path, batch_size: int = 32) -> dict:
    """Compare TF-IDF, dense BGE, and cross-encoder-reranked retrieval."""
    questions, audit = load_questions(project_root)
    chunks = load_chunks(project_root / "data/experiments/350w_50o/chunks.jsonl")
    query_texts = questions["question"].tolist()
    document_texts = chunks["text"].tolist()

    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    document_tfidf = tfidf.fit_transform(document_texts)
    tfidf_scores = linear_kernel(tfidf.transform(query_texts), document_tfidf)
    tfidf_rankings = np.argsort(-tfidf_scores, axis=1, kind="stable")

    dense_model = SentenceTransformer(DENSE_MODEL)
    query_embeddings = dense_model.encode(
        [QUERY_PREFIX + query for query in query_texts], batch_size=batch_size,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
    )
    document_embeddings = dense_model.encode(
        document_texts, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )
    dense_scores = query_embeddings @ document_embeddings.T
    dense_rankings = np.argsort(-dense_scores, axis=1, kind="stable")

    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    reranked_rows = []
    for query_pos, query in enumerate(query_texts):
        candidates = dense_rankings[query_pos, :CANDIDATE_K]
        pairs = [(query, document_texts[index]) for index in candidates]
        cross_scores = np.asarray(reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False))
        reranked_candidates = candidates[np.argsort(-cross_scores, kind="stable")]
        # Preserve a complete ranking for MRR after reranking the candidate pool.
        reranked_rows.append(np.concatenate([reranked_candidates, dense_rankings[query_pos, CANDIDATE_K:]]))
    reranked_rankings = np.vstack(reranked_rows)

    details = pd.concat([
        score_rankings("TF-IDF", chunks, questions, tfidf_rankings),
        score_rankings("BGE", chunks, questions, dense_rankings),
        score_rankings("BGE + reranker", chunks, questions, reranked_rankings),
    ], ignore_index=True)
    metric_columns = ["recall@1", "recall@3", "recall@5", "mrr"]
    summary = details.groupby("method")[metric_columns].mean().sort_values("mrr", ascending=False)
    diversity_columns = [
        "unique_sources@5", "duplicate_fraction@5", "max_source_share@5",
        "candidate_recall@20", "diversification_candidate",
    ]
    diversity = details.groupby("method")[diversity_columns].mean()
    by_category = details.groupby(["method", "category"])[metric_columns].mean()
    errors = pd.crosstab(details["method"], details["error_type"])

    output_dir = project_root / "data/experiments/evaluation_retrievers_350w_50o"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.reset_index().to_csv(output_dir / "retrieval_metrics.csv", index=False)
    diversity.reset_index().to_csv(output_dir / "diversification_metrics.csv", index=False)
    by_category.reset_index().to_csv(output_dir / "retrieval_metrics_by_category.csv", index=False)
    errors.reset_index().to_csv(output_dir / "error_type_counts.csv", index=False)
    details.to_csv(output_dir / "retrieval_details.csv", index=False)
    run_info = {
        **audit,
        "corpus": "350 words / 50 overlap",
        "chunks": len(chunks),
        "dense_model": DENSE_MODEL,
        "reranker_model": RERANKER_MODEL,
        "query_prefix": QUERY_PREFIX,
        "candidate_k": CANDIDATE_K,
        "batch_size": batch_size,
        "dense_embedding_dimension": dense_model.get_sentence_embedding_dimension(),
        "dense_max_sequence_length": dense_model.max_seq_length,
        "normalized_dense_embeddings": True,
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    return {
        "summary": summary,
        "diversity": diversity,
        "by_category": by_category,
        "errors": errors,
        "details": details,
        "run_info": run_info,
        "output_dir": output_dir,
    }

