# Evaluation Failure Taxonomy

This project separates failures by the pipeline layer that can be changed. The report is
generated from committed evaluation evidence, not from a model-written diagnosis:

```bash
python -m eval.failure_taxonomy
```

By default it reads `eval/reports/ragas_report.json` and writes
`eval/reports/failure_taxonomy_report.json`. Run it only after a complete full-golden-set
evaluation has been written. A smoke report can be analysed explicitly for diagnostics,
but it must not be presented as the full evaluation:

```bash
python -m eval.failure_taxonomy --input eval/reports/ragas_smoke_report.json --output eval/reports/failure_taxonomy_smoke_report.json
```

When the instructor-authorized Groq judge is used, retain its provider-labelled RAGAS
report and generate the taxonomy from that report explicitly:

```bash
python -m eval.failure_taxonomy --input eval/reports/ragas_groq_report.json --output eval/reports/failure_taxonomy_report.json
```

| Category | Evidence rule | Primary remediation |
|---|---|---|
| `retrieval_miss` | Expected clause is missing (`context_recall < 1.0`) | Inspect transformed queries, BM25/vector candidates, RRF ranks, reranker, and final top-K. |
| `retrieval_noise` | Supporting clause is present but deterministic precision is below 0.50 | Reduce distractors with retrieval/reranker/top-K tuning without losing the supporting clause. |
| `abstention_retrieval_risk` | An abstention case has retrieved contexts even though it has no expected clause | Strengthen confidence and abstention gating; retrieval alone must not cause an answer. |
| `grounding_failure` | Completed RAGAS faithfulness is below 0.85 | Compare answer claims to exact retrieved/cited clauses; restrict generation or abstain. |
| `synthesis_failure` | Completed RAGAS answer relevancy is below 0.80 | Inspect decomposition and answer focus; remove irrelevant synthesis. |
| `ragas_unassessed` | Live RAGAS did not complete | Do not claim grounding/synthesis quality. Resolve provider/quota and rerun. |

One question may have more than one category. For example, a retrieval miss plus low
faithfulness is reported as both retrieval and grounding evidence, rather than incorrectly
attributing the whole failure to generation. The taxonomy intentionally does not infer
grounding or synthesis failures if the input report has no completed RAGAS metrics.
