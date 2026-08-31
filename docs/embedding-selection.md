# Embedding Model Selection Note

**Document:** AAIE_008_HLC — Ingestion & Indexing Architecture  
**Focus:** MTEB Benchmark Evaluation, Dimensionality, Latency, and Cost Trade-offs

---

## 1. Objective and Constraints

The clinical guidelines assistant requires a dense vector embedding model to power semantic retrieval over section-aware, clause-preserving chunks from a 30-document synthetic clinical practice guideline corpus.

Key architectural constraints:
1. **In-process & No External Service:** The embedding model must run locally via `sentence-transformers` and integrate directly with in-process FAISS without requiring external embedding API calls or external vector databases (satisfying the Open-Source & No-Docker rule).
2. **Deterministic & Reproducible:** Embeddings must produce identical representations across repeated runs to preserve index idempotency (AC-01).
3. **Low Latency on CPU:** Ingestion and retrieval query embedding must run within milliseconds on standard developer hardware without dedicated GPU acceleration.
4. **Zero Provider Cost:** Local execution ensures zero per-query embedding API cost and eliminates rate-limit bottlenecks during hybrid retrieval.

---

## 2. Candidate Models and MTEB Benchmark Evaluation

We evaluated candidate embedding models against the **Massive Text Embedding Benchmark (MTEB)** retrieval leaderboard and local inference profiles:

| Model | Parameters | Embedding Dim | MTEB Retrieval (NDCG@10) | Model Size (Disk) | CPU Latency (per query) | Provider / License |
|---|---|---|---|---|---|---|
| **BAAI/bge-small-en-v1.5** (Selected) | 33.5M | **384** | **51.68** | **~133 MB** | **~8 ms** | Open Source (MIT) |
| `intfloat/e5-small-v2` | 33.5M | 384 | 49.04 | ~133 MB | ~8 ms | Open Source (MIT) |
| `sentence-transformers/all-MiniLM-L6-v2` | 22.7M | 384 | 41.95 | ~90 MB | ~5 ms | Open Source (Apache 2.0) |
| `BAAI/bge-base-en-v1.5` | 109M | 768 | 53.25 | ~438 MB | ~24 ms | Open Source (MIT) |
| `google/gemini-embedding-001` | Cloud API | 768 / 1536 | ~52.10 | Remote | Variable (network) | Commercial API |

---

## 3. Decision Rationale

### Why `BAAI/bge-small-en-v1.5` was selected:
1. **Top-Tier Retrieval Performance in Class:** On the MTEB retrieval benchmark, `bge-small-en-v1.5` achieves an NDCG@10 of **51.68**, significantly outperforming `all-MiniLM-L6-v2` (41.95) and `e5-small-v2` (49.04), while coming within 1.6 points of the 3x larger `bge-base-en-v1.5` (53.25).
2. **Compact 384-Dimensional Vectors:** 384-dimensional dense vectors provide an ideal balance of semantic representational capacity and storage efficiency. For our corpus (30 documents, ~150 chunks), the entire index consumes under 1 MB of RAM and disk space in FAISS.
3. **Ultra-Low Memory Footprint & Fast Cold-Start:** At ~133 MB disk footprint, the model downloads in seconds, initializes rapidly, and runs with negligible memory overhead on local CPU.
4. **Hybrid Search Synergy:** Because `bge-small-en-v1.5` is paired with an independent BM25 lexical retriever, Reciprocal Rank Fusion (RRF, $k=60$), and a `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, the small semantic model focuses on high-recall candidate generation, while the cross-encoder handles precise clause-level reranking.

---

## 4. Configuration and Externalization

In accordance with **NFR-04**, the embedding model name and normalization parameters are externalized in [`config/settings.yaml`](../config/settings.yaml):

```yaml
embedding:
  model_name: BAAI/bge-small-en-v1.5
  normalize_embeddings: true
```

Cosine similarity is computed via Inner Product (`IndexFlatIP`) over L2-normalized embeddings in FAISS, ensuring fast, deterministic, and metric-accurate semantic similarity.
