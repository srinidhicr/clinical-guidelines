# Guardrail Policy

## Purpose and scope

This assistant is a synthetic-guideline reference tool. It retrieves and summarises only
the fictional documents committed under `data/raw/`; it is not a diagnostic service,
medical device, prescribing system, or source of individual clinical advice.

## Required response controls

1. **Citation required.** Each non-abstained answer must cite one or more retrieved
   source document and clause-ID pairs. Citations are validated against the loaded corpus
   before the response is returned.
2. **Grounded-only generation.** The generator may use only supplied retrieved passages
   for factual claims. It may not use general medical knowledge to fill a gap.
3. **Abstain by default.** The assistant abstains when the question is unsupported,
   retrieval confidence is below the configured threshold, retrieved evidence conflicts,
   or a claimed citation is not among the retrieved clauses.
4. **No patient-specific decisions.** Requests framed around a real person, their test
   values, diagnosis, treatment selection, or dosage decision receive a scope-limited
   refusal. The system can explain what a cited synthetic guideline says in general.
5. **Synthetic-data boundary.** Real EHR, patient, provider, and confidential
   organisational data must not be ingested. Logs store only a request ID, timestamp,
   outcome, confidence, cited clause IDs, and provider-error type; they never serialise
   raw questions, model prompts, model responses, or provider error messages.

## Abstention response contract

An abstention returns the Pydantic response schema with:

- a clear statement that the synthetic guideline corpus does not support an answer;
- an empty citation list and null recommendation grade;
- `grounded=false` and confidence below the configured threshold; and
- no inferred dose, diagnosis, referral, or patient-specific action.

## Enforcement points

| Control | Enforcement layer | Evidence |
|---|---|---|
| Scope and patient-specific refusal | query transformation / generator | AC-05 tests |
| Evidence threshold | pipeline | config + AC-05 test |
| Citation existence and provenance | schema validation | AC-02 and AC-06 tests |
| Retry then safe failure | Gemini client | NFR-05 test |
| Provenance-only audit logging | logging utility | NFR-08 test |

This policy is reviewed with every corpus change and before publishing an evaluation
report.

## Local diagnostics exception

`python -m src.diagnose_gemini` is an interactive terminal-only connectivity check. It
may display the provider's error message to the developer to diagnose API configuration;
it never writes that message to the audit log. Credentials must never be copied into its
output or a support request.
