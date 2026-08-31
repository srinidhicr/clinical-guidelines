# Corpus Design and Quality Review

## Current corpus

The corpus contains 30 markdown documents: five document types for each of six synthetic
specialties (Cardiology, Endocrinology, Respiratory Medicine, Infectious Disease,
Nephrology, and Rheumatology). Every document has a source, specialty, version, issuing
body, and stable `CLAUSE-ID`-style marker (for example `CARD-001-S2`). The committed
golden set currently has 29 entries: 24 single-clause, three multi-clause, and two
abstention cases. All expected clause IDs resolve to a source document.

No files should be removed at this stage. The corpus meets the 30-document threshold,
has no byte-identical files, and has a sound citation convention.

## Deliberate retrieval challenge

The baseline documents are structurally uniform. That makes section-aware chunking easy
to demonstrate, but can make the benchmark look templated and overly easy. Before
indexing, retain the same document count while enriching at least 12 existing documents
by hand with the following controlled variations:

- Add inclusion/exclusion criteria, exception notes, and negative statements that must
  not be mistaken for recommendations.
- Use contrast pairs across specialties: similar words such as *urgent*, *monitor*,
  *review*, and *avoid*, but different thresholds, actions, and clause IDs.
- Include genuine multi-clause questions spanning care-pathway + referral/contraindication combinations (e.g., Q28), same-document dual-clause synthesis (e.g., Q17), and cross-condition comparative pathways (e.g., Q22).
- Add hard-negative/abstention questions (unknown condition, unsupported dose,
  patient-specific decision, and non-clinical request) while preserving at least 20 total
  golden examples.
- Give each specialty one distinctive document shape (a decision table, exception list,
  monitoring schedule, or escalation ladder) rather than repeating four identical
  section headers everywhere.

All diseases, medicines, thresholds, and organisations remain expressly fictional. This
variation makes the solution recognisably yours without compromising the requirement that
the corpus be hand-authored synthetic data.

## Chunking contract

One citation clause is the smallest retrieval unit. A chunk contains its document header
metadata, section heading, clause ID, and the full section body. Very long sections may
be split only at paragraph or table-row boundaries; child chunks retain the same clause ID
with an ordinal suffix. This preserves human-readable citation anchors and avoids
separating a dose from its contraindication table header.
