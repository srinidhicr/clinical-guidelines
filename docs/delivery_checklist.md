# Delivery Checklist

Status is maintained as implementation evidence is added. A checked item means it exists
in the repository; it is not considered verified until its associated test has passed.

## Corpus and requirements

- [x] 30 hand-authored synthetic guideline documents across six specialties
- [x] Stable source, specialty, document-type, and clause-ID metadata
- [x] Enriched contrastive cases, exceptions, and negative rules
- [x] Golden set with at least 20 questions, including multi-clause and abstention cases
- [x] Business case and synthetic-data guardrail policy
- [x] Testable AC-01 through AC-10 specification

## Ingestion and indexing — AC-01

- [x] Explicit markdown loader scoped to document headers
- [x] Section-aware, clause-preserving chunking with applied overlap configuration
- [x] Embedding model and chunk parameters externalised in `config/settings.yaml`
- [x] Persisted FAISS index, JSON chunk metadata, and index manifest
- [x] Corpus/config fingerprint plus content-hash deduplication for idempotency
- [x] AC-01 automated test — passed locally

## Retrieval quality — AC-03, AC-04, AC-07

- [x] Independent BM25 lexical retriever
- [x] Independent FAISS semantic retriever
- [x] Reciprocal Rank Fusion of both result sets
- [x] Cross-encoder reranking before generation
- [x] Multi-part query decomposition / rewrite
- [x] Retrieval settings read from configuration
- [x] AC-03, AC-04, and AC-07 tests — passed locally

## Generation and grounding — AC-02, AC-05, AC-06

- [x] Pydantic answer/citation schema
- [x] Gemini client with retries and safe failure
- [x] Grounded-only prompt plus citation/provenance and claim-support validation
- [x] Deterministic scope/evidence gates, confidence threshold, and abstention path
- [x] Provenance-only logging (request ID, confidence, citations, and error type only)
- [x] AC-02, AC-05, AC-06 and NFR-05/NFR-08 tests — passed locally

## Evaluation — AC-08, AC-09, AC-10

- [x] Golden-set structure and traceability test
- [x] Re-runnable deterministic pipeline evaluation runner
- [ ] Run and commit RAGAS report with all four required metrics (implementation complete; live run pending)
- [x] Deterministic failure-taxonomy runner for retrieval, grounding, synthesis, and abstention risks
- [ ] Commit the taxonomy report generated from the completed full-golden-set RAGAS report
- [x] Two-Gemini-model comparison runner with a documented, grounding-first selection rule
- [ ] Commit completed full-golden-set per-model reports and comparison/selection report
- [x] Evaluation-harness tests — passed locally

## Engineering and handoff — NFR-01 to NFR-07

- [x] `.env.example`, no hard-coded secret policy, and `.gitignore`
- [x] Externalized runtime/retrieval configuration
- [x] `python -m src.pipeline --run-all` builds the index, produces a cited sample answer, and runs deterministic evaluation
- [x] Minimal CLI interface
- [x] README setup, single-command run, architecture, and limitations
- [x] Cost/latency note based on representative query (Q01)
- [ ] Full test run, generated index, reports, and final acceptance audit
- [ ] At least three PR-driven merge commits (complete in GitLab workflow)
