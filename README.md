# Clinical Guidelines Assistant

A local retrieval-augmented assistant for a **synthetic** clinical-guidelines corpus.
It helps users locate what the committed synthetic guidance says, returns clause-level
citations, and abstains when the corpus cannot support an answer. It is a reference
lookup tool only: it does not use real patient data and does not provide diagnosis,
prescribing, or patient-specific treatment advice.

**Business case:** `AAIE_008_HLC` · **Track:** Gen AI Core + RAG Engineering

## What the project demonstrates

- A 30-document, six-specialty synthetic corpus with stable document, source, section,
  and clause identifiers.
- Section-aware, clause-preserving chunking; local BGE embeddings; and a persisted FAISS
  index with content-hash deduplication and a manifest for idempotent rebuilds.
- Hybrid retrieval: independent BM25 and vector search, Reciprocal Rank Fusion (RRF),
  then a MiniLM cross-encoder reranker.
- Deterministic query decomposition for comparison and multi-part questions.
- Gemini-backed structured answers validated with Pydantic, retrieved-evidence citation
  validation, retries, provenance-only logs, and an abstention path.
- A committed 29-question golden set and an evaluation harness for deterministic
  retrieval diagnostics plus Gemini-judged RAGAS metrics.

Read [the business case](docs/business_case.md), [guardrail policy](docs/guardrail_policy.md),
[embedding selection note](docs/embedding_selection.md), and [acceptance criteria](specs/acceptance_criteria.md)
for the full rationale and traceability.

## Architecture

```text
Synthetic Markdown corpus
        |
section-aware chunks + metadata
        |
BGE embeddings -> persisted FAISS index
        |
query transform -> BM25 + vector search -> RRF -> cross-encoder reranker
        |
Gemini structured generation -> Pydantic + citation validation -> answer or abstention
        |
golden-set deterministic diagnostics + Gemini RAGAS evaluation
```

## Prerequisites

- Python **3.11**
- Internet access for the first download of the local embedding/reranker models
- A Gemini Developer API key for generation and live RAGAS evaluation

The corpus, chunk inspection, persisted-index inspection, retrieval tests, and
deterministic evaluation do **not** require a Gemini key.

## Setup

Run these commands from the repository root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Create your local credentials file from the committed template. Never commit `.env`.

```bash
cp .env.example .env
```

Then edit `.env` and add only your local key:

```text
GOOGLE_API_KEY=your_key_here
```

To verify the installed project before any model download or API call:

```bash
python -m pytest -q
```

## Pipeline verification, stage by stage

The following is the recommended office-laptop verification sequence. Run stages 1–5
first; they prove ingestion and hybrid retrieval without requiring Gemini.

### One-command end-to-end run (NFR-02)

After adding `GOOGLE_API_KEY` to `.env`, run the complete local workflow with committed
sample data:

```bash
python -m src.pipeline --run-all
```

It builds or reuses the persisted index, generates one cited answer using Q01 from the
committed golden set, and writes a full deterministic retrieval evaluation to
`eval/reports/pipeline_run_evaluation_report.json`. The default deliberately does not
run live RAGAS, because that stage consumes Gemini quota. To include it explicitly:

```bash
python -m src.pipeline --run-all --with-ragas
```

### 1. Inspect clause-preserving chunks

```bash
python -m src.inspect_artifacts chunks --limit 3
```

Expected: JSON records with source metadata, `clause_id`, section heading, chunk ID, and
chunk text. This verifies corpus loading and section-aware chunk construction.

### 2. Build or reuse the local FAISS index

```bash
python -m src.pipeline --build-index
```

Expected: `Index is ready.` On first use, BGE embedding weights may download. On a
repeat run with unchanged corpus/configuration, the persisted index is reused rather than
duplicated.

### 3. Inspect stored vectors and metadata

```bash
python -m src.inspect_artifacts index --limit 2
```

Expected: the index manifest, sample chunk metadata, and sample vectors. Check that
`chunk_count` and `vector_count` agree in the manifest.

### 4. Inspect query transformation, fusion, and reranking

```bash
python -m src.inspect_artifacts retrieve \
  "Compare the second-line management of APES and Chronic Rhythm Irregularity Disorder (CRID)."
```

Expected: JSON containing a `query_plan`, the RRF `fused` candidates, and the final
`reranked` candidates. The query plan should show comparison decomposition; final
candidates should have `source: "cross_encoder"`.

### 5. Run focused offline tests for each stage

```bash
python -m pytest -q tests/test_ingestion.py
python -m pytest -q tests/test_retrieval.py
python -m pytest -q tests/test_pipeline.py
python -m pytest -q tests/test_generation.py
python -m pytest -q tests/test_eval_set.py tests/test_eval_harness.py
```

Run the entire suite before submission:

```bash
python -m pytest -q
```

### 6. Verify Gemini connectivity and structured JSON

Use the configured current model explicitly for this diagnostic:

```bash
GEMINI_MODEL=gemini-3.6-flash python -m src.diagnose_gemini
```

Expected: a basic connection confirmation and `Structured JSON/Pydantic request
succeeded.` This command is terminal-only and does not log credentials or prompts.

### 7. Run one complete answer request

```bash
python -m src.pipeline "What is the first-line management of APES?"
```

Expected: a JSON object containing `answer`, `citations`, `guideline_source`, `section`,
`strength_of_recommendation`, `grounding_confidence`, and `grounded`. A grounded answer
must include clause-level citations. An unsupported question should return the structured
abstention response instead of a fabricated answer.

## Evaluation

### Deterministic retrieval diagnostics (no Gemini calls)

One golden question:

```bash
python -m eval.ragas_eval --question-id Q01 --skip-ragas
```

All golden questions:

```bash
python -m eval.ragas_eval --skip-ragas
```

The full deterministic report is written to `eval/reports/ragas_report.json`. A
single-question smoke run writes `eval/reports/ragas_smoke_report.json` and does not
overwrite the full report.

### Live Gemini RAGAS evaluation

Run one question first to verify provider access and quota:

```bash
python -m eval.ragas_eval --question-id Q01
```

Then run the complete committed golden set:

```bash
python -m eval.ragas_eval
```

After a complete report is present, generate the retrieval-versus-grounding-versus-synthesis
failure analysis:

```bash
python -m eval.failure_taxonomy
```

The taxonomy, thresholds, and remediation rules are documented in
[docs/failure_taxonomy.md](docs/failure_taxonomy.md).

### Two-model Gemini comparison

After live RAGAS succeeds for one smoke question, compare the two configured generation
models on the same full golden set and fixed Gemini judge:

```bash
python -m eval.llm_comparison
```

This writes a per-model RAGAS report and `eval/reports/llm_comparison_report.json`. It
selects a model only when both full runs complete, using the committed weighting that
prioritises faithfulness. To test the wiring without Gemini calls, use:

```bash
python -m eval.llm_comparison --question-id Q01 --skip-ragas
```

The harness records per-question and aggregate context precision, context recall,
faithfulness, and answer relevancy. It uses one RAGAS worker and retry/backoff settings
from `config/settings.yaml` to reduce rate-limit pressure.

Live RAGAS consumes Gemini quota. A `429 RESOURCE_EXHAUSTED` response is a provider
quota limit; a `503 UNAVAILABLE` response is temporary provider demand. Neither is a
valid completed report. Check the active Gemini project/model availability and quota in
Google AI Studio before rerunning.

### Instructor-authorized Groq RAGAS judge

The application continues to generate answers with Gemini. If the instructor authorizes
Groq specifically for RAGAS judging, install dependencies, add `GROQ_API_KEY` only to
your untracked `.env`, then run:

```bash
python -m pip install -r requirements.txt
python -m eval.ragas_groq_eval --question-id Q01
python -m eval.ragas_groq_eval
```

If a run is interrupted, inspect the resumable Gemini-generation checkpoint without
making any provider call:

```bash
python -m eval.ragas_groq_eval --status
```

The command reports completed grounded-answer IDs out of the 27 answerable golden-set
questions. Do not use `--reset-checkpoint` when resuming.

The Groq run writes `ragas_groq_report.json` and a privacy-safe
`groq_ragas_generation_audit.json` that counts every pipeline answer generated for the
run (grounded or abstained) and records only its ID, confidence, and citation clause IDs.
It also writes an ignored `groq_ragas_generation_checkpoint.json` after each grounded
Gemini answer. If Gemini fails at (for example) Q17, restore provider access and rerun
the same command: Q01–Q16 are reused, not regenerated. Use `--reset-checkpoint` only to
intentionally discard that local cache.
Do not replace the Gemini application model or the two-Gemini-model comparison candidates
with Groq unless the brief itself is formally changed.

After a completed full Groq report, generate the failure taxonomy from the same
provider-labelled report:

```bash
python -m eval.failure_taxonomy \
  --input eval/reports/ragas_groq_report.json \
  --output eval/reports/failure_taxonomy_report.json
```

## Configuration

All operational settings are in [config/settings.yaml](config/settings.yaml), including:

- chunk length and overlap;
- embedding and reranker models;
- BM25/vector candidate counts, RRF constant, final context count, and confidence
  threshold;
- Gemini generation model and retries; and
- RAGAS judge model, embedding model, worker count, timeout, and backoff settings.

Use `GEMINI_MODEL` only for a local generation-model override. The RAGAS judge is
selected by `evaluation.judge_model` in the settings file.

## Safety and data handling

- The entire corpus is fabricated for this project. Do not ingest real EHR, patient, or
  confidential organisational data.
- Non-abstained answers are validated against the retrieved citations before return.
- Logs contain only provenance metadata, confidence, cited clause IDs, and error type;
  not raw questions, prompts, model responses, or API keys.
- `.env` is deliberately ignored. Commit `.env.example` only.

## Repository map

```text
config/settings.yaml       Runtime and evaluation configuration
data/raw/                  30 synthetic guideline documents
data/index/                Persisted FAISS index, chunks, and manifest
data/golden_eval_set.json  Committed golden evaluation set
src/ingestion/             Loaders, chunker, index builder
src/retrieval/             BM25, vector search, RRF, reranker, query transform
src/generation/            Gemini prompting, schema, citation validation
src/pipeline.py            CLI pipeline entry point
src/inspect_artifacts.py   Read-only chunk/index/retrieval inspection CLI
eval/ragas_eval.py         Deterministic + live RAGAS evaluation CLI
tests/                     Acceptance and regression tests
```

## Current delivery status

Offline ingestion, retrieval, structured-output, logging, golden-set, and configuration
tests are implemented. Live RAGAS reporting requires a successful Gemini run with enough
quota. The two-model comparison report, final failure taxonomy, and final Git/PR evidence
remain delivery tasks until their respective committed artefacts are completed.
