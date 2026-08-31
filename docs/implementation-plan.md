# Implementation Plan

## Delivery strategy

The project is built as a local, evidence-first RAG system. Each milestone produces a
committed artefact that can be inspected independently of a live Gemini key. The corpus,
golden set, configuration, reports, and tests therefore remain useful evidence even when
model-backed evaluation is unavailable.

| Order | Deliverable | Rubric evidence |
|---|---|---|
| 1 | Corpus manifest, clause convention, and enriched retrieval-confuser cases | AC-01, AC-02, AC-08 |
| 2 | Configuration, guardrail policy, acceptance traceability, and quick-start | NFR-01, NFR-04, NFR-07 |
| 3 | Markdown loader, section-aware chunker, hash-deduplicated persisted local index | AC-01 |
| 4 | Independent BM25 and vector retrieval, RRF, cross-encoder reranking, query decomposition | AC-03, AC-04, AC-07 |
| 5 | Gemini generation with citation validation, structured Pydantic output, retries, and abstention | AC-02, AC-05, AC-06, NFR-05, NFR-08 |
| 6 | AC-referenced unit tests, golden-set runner, metrics reports, and failure taxonomy | AC-08, AC-09, NFR-06 |
| 7 | Two-Gemini-model comparison, selection note, end-to-end verification, and README polish | AC-10, NFR-02, NFR-07 |

## Milestone gates

1. Do not call Gemini until offline ingestion and hybrid retrieval tests pass.
2. Every generated answer must be validated against a retrieved clause ID before it is
   returned; invalid citations become an abstention.
3. Preserve the 30 source documents as synthetic source-of-truth. Any corpus enrichment
   updates both its linked golden-set entries and `docs/corpus_design.md`.
4. Reports record their model, prompt/config version, corpus fingerprint, and run time so
   their claims can be reproduced.

## Document timing

`business_case.md`, the guardrail policy, and acceptance criteria are living documents:
their problem statement, scope, criteria, and intended targets belong in the repository
now. Only measured values, final operational cost/latency, and the chosen model rationale
are updated after evaluation has run.
