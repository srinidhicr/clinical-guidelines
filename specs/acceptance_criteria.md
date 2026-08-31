# Acceptance Criteria

Each criterion below is written in testable form and must be referenced by at least one
test or golden-eval-set entry carrying its `AC-NN` id (AC-Traceability Rule). The `Test
reference` column is filled in as the corresponding test is written — see `tests/`.

---

### AC-01 — Idempotent ingestion into a persisted index

**Given** a synthetic corpus of 30 guideline documents
**When** the ingestion pipeline is run twice in succession without clearing the index
**Then** the resulting FAISS index contains exactly one vector per chunk (no duplicates),
and the index persists to disk between runs.

*Test reference:* `tests/test_ingestion.py::test_ingestion_is_idempotent`

---

### AC-02 — Grounded answers with clause-level citations

**Given** a natural-language question answerable from the corpus
**When** the pipeline generates a response
**Then** the response includes at least one citation identifying the source document and
clause id, and the cited clause actually supports the claim made.

*Test reference:* `eval/golden_eval_set.json` (all `single_clause` / `multi_clause`
entries) + `tests/test_generation.py::test_answer_has_valid_citation`

---

### AC-03 — Hybrid retrieval with fusion

**Given** a query
**When** retrieval runs
**Then** both a BM25 (lexical) result set and a vector (semantic) result set are produced
independently, and combined via Reciprocal Rank Fusion into a single ranked list before
reranking.

*Test reference:* `tests/test_retrieval.py::test_hybrid_fusion_combines_both_sources`

---

### AC-04 — Reranking before generation

**Given** a fused candidate list from AC-03
**When** the candidates are passed to the reranker
**Then** a cross-encoder reranker re-scores the candidates and only the top-K reranked
results are forwarded to the generation layer (verified by asserting the reranker's
output order differs from the fusion-only order on at least one test query).

*Test reference:* `tests/test_retrieval.py::test_reranker_reorders_candidates`

---

### AC-05 — Abstention when the corpus is insufficient

**Given** a question the corpus does not cover (e.g., golden set `abstain`-type entries)
**When** the pipeline processes it
**Then** the response's structured `grounding_confidence` field falls below the
configured abstention threshold and the returned answer text explicitly states the
corpus does not support an answer, rather than fabricating one.

*Test reference:* `eval/golden_eval_set.json` (`abstain`-type entries, Q20–Q21) +
`tests/test_generation.py::test_abstains_on_unsupported_question`

---

### AC-06 — Validated structured output

**Given** any query processed by the pipeline
**When** the generation layer returns a result
**Then** the result validates against the Pydantic schema (`answer`, `citations`,
`guideline_source`, `section`, `strength_of_recommendation` [nullable], and
`grounding_confidence`), and a schema violation raises rather than being silently passed
through.

*Test reference:* `tests/test_generation.py::test_response_matches_pydantic_schema`

---

### AC-07 — Query transformation for multi-part questions

**Given** a multi-part or ambiguous query (e.g., golden set `multi_clause` entries such
as Q22, a comparison across two conditions)
**When** the query reaches the query-transformation stage
**Then** it is rewritten, expanded, or decomposed into sub-queries before retrieval, and
retrieval is run against the transformed form rather than the raw input.

*Test reference:* `eval/golden_eval_set.json` (Q22) +
`tests/test_retrieval.py::test_multi_part_query_is_decomposed`

---

### AC-08 — Committed golden evaluation set

**Given** the repository
**When** `data/golden_eval_set.json` is inspected
**Then** it contains at least 20 questions, each with a reference answer and expected
context (clause ids), spanning single-clause, multi-clause, and abstention cases.

*Test reference:* `tests/test_eval_set.py::test_golden_set_has_min_20_entries_with_expected_context`

---

### AC-09 — RAGAS metrics computed and committed

**Given** the golden evaluation set and a working pipeline
**When** `eval/ragas_eval.py` is run
**Then** context precision, context recall, faithfulness, and answer relevancy are
computed for every golden question and the aggregate results are written to
`eval/reports/ragas_report.json` (or `.md`).

*Test reference:* `tests/test_eval_harness.py::test_ragas_report_is_generated`

---

### AC-10 — Two-model comparison

**Given** the golden evaluation set
**When** `eval/llm_comparison.py` is run against at least two Gemini model variants
(e.g., `gemini-1.5-flash` vs `gemini-1.5-pro` — confirm exact available model names at
implementation time)
**Then** a comparison report is committed showing metrics per model and a written
selection rationale for the model chosen in the final pipeline.

*Test reference:* `tests/test_eval_harness.py::test_model_comparison_requires_two_completed_full_runs_for_selection`

---

## Non-Functional Requirements (testable form)

| ID | Test reference | Verifies |
|---|---|---|
| NFR-01 | `tests/test_config.py::test_no_secret_environment_file_is_tracked` | No API keys in tracked files; `.env.example` present |
| NFR-02 | `tests/test_pipeline.py::test_single_command_run_executes_index_answer_and_evaluation` | `python -m src.pipeline` runs end-to-end on sample data |
| NFR-03 | manual review + `docs/guardrail_policy.md` | Synthetic data only; no plaintext PII in logs |
| NFR-04 | `tests/test_config.py::test_retrieval_parameters_are_externalized` | chunk size, top-K, thresholds read from `config/settings.yaml` |
| NFR-05 | `tests/test_generation.py::test_retries_on_transient_failure` | Gemini call retries then fails gracefully |
| NFR-06 | `tests/test_eval_harness.py::test_subset_evaluation_writes_separate_report` | Re-running eval script reproduces metrics from committed data |
| NFR-07 | `docs/business_case.md` (cost/latency note) | Concept-level cost/latency addressed |
| NFR-08 | `tests/test_logging.py::test_query_logged_with_provenance` | Every non-abstained answer logs its citations |