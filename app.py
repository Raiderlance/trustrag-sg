"""Streamlit demonstration interface for TrustRAG SG."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from rag import (
    BaselineRAGPipeline, DenseMMRRetriever, GeminiGenerator, RAGPipeline,
    _dotenv_value,
)


CHUNKS_PATH = Path(os.getenv("RAG_CHUNKS_PATH", "data/processed/chunks.jsonl"))
SAMPLE_QUESTIONS = (
    "Who can buy an HDB flat under the couples and families route?",
    "How long is a resale Option to Purchase valid after it is granted?",
    "I am 30 and single. Can I buy any HDB resale flat on my own?",
    "What will the CPF Ordinary Account interest rate be next year?",
    "I am 55. Tell me exactly how much my CPF LIFE monthly payout will be."
)


@st.cache_resource(show_spinner="Loading BGE model and persisted document embeddings…")
def load_retriever(chunks_path: str) -> DenseMMRRetriever:
    return DenseMMRRetriever.from_jsonl(chunks_path, candidate_k=20, lambda_mult=0.9)


def render_sources(chunks) -> None:
    st.subheader("Retrieved evidence")
    st.caption("BGE dense top-20 → MMR λ=0.9 → final top 5")
    for rank, item in enumerate(chunks, start=1):
        metadata = item.chunk.get("metadata", {})
        title = metadata.get("title", "Untitled")
        source_id = metadata.get("source_id", "Unknown")
        chunk_id = item.chunk.get("chunk_id", "Unknown")
        with st.expander(f"[{rank}] {title} · {chunk_id} · score {item.score:.3f}"):
            st.write(item.chunk.get("text", ""))
            columns = st.columns(2)
            columns[0].caption(f"Agency: {metadata.get('agency', 'Unknown')} · Source: {source_id}")
            url = metadata.get("url")
            if url:
                columns[1].link_button("Open authoritative source", url, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="TrustRAG SG", page_icon="🏠", layout="wide")
    st.title("TrustRAG SG")
    st.write(
        "Evidence-grounded answers about HDB housing and CPF, with visible retrieval "
        "and an optional evidence-sufficiency gate."
    )

    with st.sidebar:
        st.header("Demo controls")
        mode = st.radio(
            "Generation mode",
            ("Evidence-gated", "Baseline"),
            help="Both modes use exactly the same BGE/MMR top five. The gated mode assesses evidence first.",
        )
        model = st.text_input(
            "Gemini model",
            value=(
                os.getenv("GEMINI_MODEL")
                or _dotenv_value("GEMINI_MODEL")
                or "gemini-3.5-flash-lite"
            ),
        )
        st.markdown("**Fixed retrieval configuration**")
        st.caption("BAAI/bge-small-en-v1.5 · candidates 20 · top 5 · MMR λ=0.9")
        st.divider()
        st.caption(
            "Prototype only. Verify consequential housing or CPF decisions on the linked agency page."
        )

    selected = st.selectbox("Try an example", ("Write my own question",) + SAMPLE_QUESTIONS)
    initial = "" if selected == "Write my own question" else selected
    question = st.text_area(
        "Question",
        value=initial,
        height=100,
        placeholder="Ask a question about HDB eligibility, housing grants, loans, resale, or CPF…",
    )

    if st.button("Retrieve and answer", type="primary", disabled=not question.strip()):
        try:
            retriever = load_retriever(str(CHUNKS_PATH))
            generator = GeminiGenerator(
                model=model.strip() or None,
                min_request_interval=13 if mode == "Evidence-gated" else 0,
            )
            with st.spinner("Retrieving evidence and generating a grounded response…"):
                if mode == "Evidence-gated":
                    pipeline = RAGPipeline(retriever, generator)
                    answer, chunks, assessment = pipeline.ask_with_assessment(question.strip(), 5)
                else:
                    pipeline = BaselineRAGPipeline(retriever, generator)
                    answer, chunks = pipeline.ask(question.strip(), 5)
                    assessment = None

            st.subheader("Answer")
            st.markdown(answer)
            if assessment:
                st.subheader("Evidence assessment")
                label_col, support_col = st.columns([1, 3])
                label_col.metric("Decision", assessment.sufficiency)
                support_col.write(assessment.reason)
                with st.expander("Assessment details"):
                    st.markdown(f"**Support:** {assessment.support}")
                    st.markdown(f"**Completeness:** {assessment.completeness}")
                    st.markdown(f"**Missing conditions:** {assessment.conditions}")
                    st.markdown(f"**Temporal validity:** {assessment.temporal_validity}")
                    ids = ", ".join(assessment.supporting_chunk_ids) or "None"
                    st.markdown(f"**Supporting chunk IDs:** {ids}")
            render_sources(chunks)
        except Exception as error:
            st.error(f"The request could not be completed: {error}")
            st.info("Check GEMINI_API_KEY, model availability, quota, and the chunk file path.")

    with st.expander("How this prototype uses AI"):
        st.markdown(
            "1. BGE embeds the question locally.\n"
            "2. Dense similarity finds 20 candidates.\n"
            "3. MMR selects five relevant, less-redundant chunks.\n"
            "4. Gemini either answers directly (baseline) or first produces a typed "
            "support/completeness/conditions/time assessment.\n"
            "5. The answer is constrained to retrieved evidence and numbered citations."
        )


if __name__ == "__main__":
    main()
