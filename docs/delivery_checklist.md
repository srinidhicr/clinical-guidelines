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

- [ ] Independent BM25 lexical retriever
- [ ] Independent FAISS semantic retriever
- [ ] Reciprocal Rank Fusion of both result sets
- [ ] Cross-encoder reranking before generation
- [ ] Multi-part query decomposition / rewrite
- [ ] Retrieval settings read from configuration
- [ ] AC-03, AC-04, and AC-07 tests

## Generation and grounding — AC-02, AC-05, AC-06

- [ ] Pydantic answer/citation schema
- [ ] Gemini client with retries and safe failure
- [ ] Grounded-only prompt and citation validation
- [ ] Confidence threshold and abstention path
- [ ] Provenance-only logging
- [ ] AC-02, AC-05, AC-06 and NFR-05/NFR-08 tests

## Evaluation — AC-08, AC-09, AC-10

- [ ] Golden-set structure and traceability test
- [ ] Re-runnable pipeline evaluation runner
- [ ] Committed RAGAS report with all four required metrics
- [ ] Failure taxonomy: retrieval, grounding, and synthesis
- [ ] Two-Gemini-model comparison report and selection rationale
- [ ] Evaluation-harness tests

## Engineering and handoff — NFR-01 to NFR-07

- [x] `.env.example`, no hard-coded secret policy, and `.gitignore`
- [x] Externalized runtime/retrieval configuration
- [ ] `pipeline.py` single callable and `python -m src.pipeline` entry point
- [ ] Minimal CLI interface
- [ ] README setup, single-command run, architecture, and limitations
- [ ] Cost/latency note based on representative query
- [ ] Full test run, generated index, reports, and final acceptance audit
- [ ] At least three PR-driven merge commits (complete in GitLab workflow)
