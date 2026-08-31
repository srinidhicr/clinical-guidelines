# Business Case: Clinical Guidelines Assistant

**Business Case ID:** AAIE_008_HLC
**Domain:** Healthcare — Clinical Practice Guidelines
**Track:** Gen AI Core + RAG Engineering

## 1. Problem

Clinicians managing patients across multiple specialties need to quickly confirm what a
guideline actually recommends — first-line therapy, dosing limits, contraindications, or
referral thresholds — without manually searching through lengthy source documents. A
generic chatbot is unsuitable here: an unsupported or fabricated answer in a clinical
context is a patient-safety risk, not just an inconvenience.

This project builds a retrieval-augmented assistant that answers guideline questions
**strictly grounded in a defined guideline corpus**, cites the exact source and clause for
every claim, and refuses to answer when the corpus does not support a response.

## 2. Target Users

- **Primary:** Clinicians (physicians, nurse practitioners) who need a fast, cited
  reference lookup during or between patient encounters.
- **Secondary:** Clinical educators and trainees cross-checking guideline recommendations
  while reviewing cases.

The assistant is a **reference lookup tool**, not a diagnostic or treatment-decision system.
It does not see patient records and does not recommend a course of action for a specific
patient — see Section 4, Guardrail 2.

## 3. Corpus Description

The corpus consists of **30 fully synthetic guideline documents** spanning six specialties
(Cardiology, Endocrinology, Respiratory Medicine, Infectious Disease, Nephrology,
Rheumatology), five documents per specialty:

| Doc type | Count per specialty | Total |
|---|---|---|
| Care pathway (condition-specific) | 2 | 12 |
| Dosing & contraindication reference | 1 | 6 |
| Referral criteria | 1 | 6 |
| Comorbidity management pathway | 1 | 6 |

Every document carries a `Source` identifier, `Specialty`, `Version`, and issuing-body
metadata in its header, and every section is tagged with an explicit `[clause id: ...]`
marker. This gives the ingestion pipeline a stable, human-readable citation anchor at the
clause level rather than an opaque chunk index — every generated answer can point to
exactly the source and section it came from.

**All conditions, drug names, dosing figures, and criteria in the corpus are fabricated**
for this project and issued under a fictional "Synthetic Clinical Standards Board," per
the Synthetic-Data Rule. None of it should be interpreted as real clinical guidance.

## 4. Domain Guardrails

1. **Grounding is mandatory.** Every substantive answer must carry at least one
   clause-level citation (document + section id). No claim may be generated without a
   supporting retrieved passage.
2. **Reference only, not medical advice.** The assistant states what a guideline
   recommends; it does not issue a decision for an individual patient, and every response
   involving clinical action includes this scope framing.
3. **Abstention over hallucination.** If retrieval does not return a passage that
   supports the question (topic outside the corpus, missing detail, or below a
   confidence threshold), the assistant explicitly declines rather than guessing.
4. **Out-of-scope handling.** Questions asking for a diagnosis, a decision for a named
   patient, or information outside the guideline domain (e.g., billing, staffing) are
   declined with a short explanation of what the assistant can help with instead.
5. **No real patient data.** The system is built and tested exclusively against the
   synthetic corpus; it is not connected to any EHR or real patient data source.

## 5. Success Metrics

The project's success is measured primarily through the RAG evaluation harness (see
`eval/reports/`) rather than subjective review:

| Metric | Target | Measured Result | Status | Source |
|---|---|---|---|---|
| Context precision (RAGAS) | ≥ 0.75 | **0.902 (90.2%)** | Exceeded | `eval/reports/ragas_groq_report.json` |
| Context recall (RAGAS) | ≥ 0.75 | **0.931 (93.1%)** | Exceeded | `eval/reports/ragas_groq_report.json` |
| Faithfulness (RAGAS) | ≥ 0.85 | **0.937 (93.7%)** | Exceeded | `eval/reports/ragas_groq_report.json` |
| Answer relevancy (RAGAS) | ≥ 0.80 | **0.878 (87.8%)** | Exceeded | `eval/reports/ragas_groq_report.json` |
| Citation presence | 100% | **100% (27/27)** | Achieved | Golden set validation |
| Abstention correctness | 100% | **100% (2/2)** | Achieved | Golden set Q20–Q21 |
| Ingestion idempotency | 100% | **100%** | Achieved | `tests/test_ingestion.py` |

All four measured RAGAS metrics exceeded target thresholds across the 29-question committed golden evaluation set.

An instructor-authorized Groq judge (`openai/gpt-oss-20b`) was utilized for RAGAS evaluation to prevent provider quota exhaustion, while preserving Google Gemini as the application generation provider. Detailed evidence is committed under `eval/reports/`.

## 6. Cost and Latency Considerations (NFR-07)

This is a local, small-scale reference implementation, not a production performance
benchmark. The representative query is golden-set **Q01**: *“What is the first-line
management for Stage 2 Arterial Pressure Elevation Syndrome (APES)?”* It exercises query
transformation, BM25 and vector retrieval, RRF, cross-encoder reranking, one Gemini
structured-generation call, citation validation, and provenance logging.

- **Normal request latency:** after the local embedding and reranker models have been
  downloaded and the FAISS index is present, latency is primarily the local retrieval and
  reranking work plus one Gemini generation request. Provider response time is variable,
  so no fixed latency promise is made.
- **First-run latency:** the first command on a new machine can be noticeably slower
  because Sentence-Transformers weights are downloaded and the index may be built. These
  are local setup costs, not per-question Gemini latency.
- **External cost:** normal answer requests use the configured Gemini generation model.
  The local embeddings, FAISS index, BM25 retrieval, and cross-encoder reranker do not
  make paid provider calls. Live RAGAS is more expensive than one answer because it makes
  multiple Gemini judge/embedding requests per golden question; it is therefore explicit
  and opt-in rather than part of the default `--run-all` command.
- **Operational control:** use a one-question RAGAS smoke run before a full evaluation,
  keep `evaluation.max_workers` at one, and inspect Gemini project quota/model
  availability before rerunning failed provider calls.
