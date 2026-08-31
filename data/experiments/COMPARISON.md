# Chunking experiment comparison

All configurations use the same 22 cached source documents and the same section-aware
`lxml` extraction. Only maximum words and overlap differ.

| Configuration | Chunks | Mean words | Median | Min–max | Chunks under 50 words | Stored words including overlap |
|---|---:|---:|---:|---:|---:|---:|
| 350 words / 50 overlap | 280 | 234.3 | 313 | 10–350 | 42 (15.0%) | 65,606 |
| 250 words / 40 overlap | 364 | 186.0 | 250 | 10–250 | 45 (12.4%) | 67,716 |
| 150 words / 30 overlap | 567 | 126.5 | 150 | 10–150 | 67 (11.8%) | 71,716 |

Smaller chunks provide finer retrieval granularity but increase embedding count and can
remove useful context. Choose the configuration using the same embedding model, vector
index settings, retrieval `k`, questions, and answer-generation prompt for every run.
Compare retrieval hit/recall at `k`, reciprocal rank, citation correctness, answer
correctness, latency, and index size. Do not choose solely from these size statistics.
