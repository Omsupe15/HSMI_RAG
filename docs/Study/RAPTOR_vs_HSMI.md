# RAPTOR vs. HSMI: A Framework Comparison

*Based on the source conversation: "Understanding the RAPTOR RAG Framework"*

---

## 1. What is RAPTOR?

**RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) is an indexing and retrieval framework for RAG systems, designed to fix a core weakness of standard chunk-based RAG: it cannot synthesize information that is scattered across a long document.

### The problem it solves
Standard RAG splits documents into fixed-size chunks (~200–500 tokens) and retrieves flat, independent pieces. This causes two failure modes:

- **Context Fragmentation** – a concept developed across multiple chapters or sections can't be retrieved as one coherent answer.
- **Mismatched Semantic Granularity** – a broad question ("What were the primary economic drivers of the war?") only matches isolated sentences instead of a synthesized, high-level answer.

RAPTOR fixes this by building an index that contains both hyper-specific facts *and* broad thematic summaries.

### How the tree is built (bottom-up, recursive)
1. **Chunking & Dense Embedding** – the corpus is split into short segments (~100 tokens in the original paper), each embedded as a vector. This forms the **leaf layer**.
2. **Dimensionality Reduction (UMAP)** – embeddings are projected into lower-dimensional space, preserving both local and global structure before clustering.
3. **Soft Clustering (Gaussian Mixture Models)** – instead of hard partitioning (like k-means), GMMs (with BIC for model selection) let a chunk belong to multiple clusters if its membership probability clears a threshold (~0.1). This avoids splitting multi-topic content into one artificial branch.
4. **Abstractive LLM Summarization** – an LLM writes a free-form prose summary for each cluster, forming a parent node that represents a higher-level concept.
5. **Recursion** – summaries are re-embedded and fed back into the same pipeline. This repeats until nodes stop meaningfully clustering, producing a root summary layer.

### Retrieval strategies at query time

| Retrieval Mode | Mechanism | Best Used For |
|---|---|---|
| **Collapsed Tree (Flattened)** | Merges every layer (leaf chunks + intermediate summaries + root summaries) into one index; ranks all candidates by cosine similarity until the token budget fills | General QA & complex queries — best benchmark accuracy from mixing granular facts with macro context |
| **Tree Traversal (Top-Down)** | Starts at the root, finds the top‑*k* relevant summaries, and beam-searches down through child nodes | Controlled drill-down tasks, hierarchical filtering where irrelevant sub-trees should be pruned early |

### RAPTOR's trade-offs
- **High Ingestion Cost** – building the tree needs multiple LLM calls per layer, per document.
- **Dynamic Updates** – because parents depend on clusters of children, adding/editing text usually forces re-clustering and re-summarizing entire branches.
- **Cascading Hallucinations** – a factual distortion or dropped caveat in an early summary propagates and compounds as it's summarized again at each level up.

### Where RAPTOR is deployed
RAPTOR is best suited to **static, complex, long-form corpora** where users ask document-wide, synthesis-heavy questions: legal discovery, books, extensive financial filings, and technical documentation.

---

## 2. What is HSMI (Hierarchical Structured Manifest Index)?

HSMI is a hybrid design that keeps RAPTOR's hierarchical, tree-shaped aggregation but replaces its free-form LLM prose at every node with a **structured semantic manifest** (JSON, or Markdown front matter). The goal is to directly target RAPTOR's three biggest pain points: runaway ingestion costs, cascading hallucinations, and index rigidity on updates.

### Architecture: a 3-tier structured tree

```
[ Root Manifest ]     <-- Document-level schema: global themes, all covered entities
       |
[ Cluster Manifests ] <-- Clustered schemas: sub-system/thematic scope + rolled-up facts
       |
[ Leaf Manifests ]    <-- Granular chunk schemas (JSON/Markdown) paired with raw text
```

### Layer 0 — Leaf Manifest Generation
The document is sliced into chunks (200–400 tokens). Instead of a generic prose summary, an SLM (small language model) or constrained LLM pipeline extracts a **strict schema** per chunk:

```json
{
  "chunk_id": "doc_v1_042",
  "parent_section": "Authentication / OAuth2 Flow",
  "primary_topics": ["Token Refresh", "Session Expiry", "PKCE"],
  "named_entities": ["OAuth 2.1", "Auth0", "Refresh Token"],
  "claims_and_rules": [
    "Refresh tokens expire after 30 days of inactivity.",
    "PKCE is mandatory for all public clients."
  ],
  "semantic_digest": "Explains OAuth 2.1 refresh token rotation policies and mandatory PKCE requirements for public clients.",
  "prerequisites": ["doc_v1_040"]
}
```

**What gets embedded:** not the raw chunk text (which can be boilerplate or repetitive), but the concatenation of `primary_topics + semantic_digest` — a higher signal-to-noise vector.

### Layer 1 — Structured Clustering & Aggregation
Leaf manifest embeddings are clustered exactly as in RAPTOR — **UMAP + GMM**. The difference is *how the parent node is built*, via a **Schema Roll-Up** instead of expensive free-form generation:

1. **Deterministic Aggregation** – take the mathematical set union of `named_entities` and `primary_topics` across every child chunk in the cluster.
2. **Constrained Synthesis** – prompt a lightweight model with structured inputs only, e.g.: *"Given these 5 child manifests, output a 2-sentence macro-summary and extract the 3 primary cross-cutting insights."*

### Layer 2 — Incremental Dynamic Updating
When a new chunk arrives:
1. Generate its leaf manifest.
2. Compute cosine distance to existing Layer-1 cluster centroids.
3. If similarity clears threshold **τ**, append the chunk ID to that cluster, union its entities into the cluster manifest, and update the cluster's short summary.
4. If similarity is below **τ**, the chunk seeds a new micro-cluster.

**Result: zero full-index recomputations.**

### Query-Time Retrieval Pipeline
Storing structured manifests alongside vector embeddings enables **multi-stage pruning**:

1. **Query Decomposition** – e.g. for *"What is the session expiry rule for OAuth clients using PKCE?"*, an extractor pulls a target entity (`["OAuth 2.1", "PKCE"]`) and a target intent (`Token/Session Expiry`).
2. **Top-Down Tree Filtering**
   - *Stage 1 (cluster level):* filter clusters where `named_entities` intersect the query entities, or where vector similarity against the cluster digest is high.
   - *Stage 2 (leaf level):* search only inside candidate clusters — dense search over `semantic_digest`, combined with BM25 keyword scoring over `claims_and_rules`.
3. **Context Assembly** – retrieve the full raw text of the winning leaf chunks, prepending the cluster-level manifest digest as a macro-context header to orient the generator.

---

## 3. RAPTOR vs. HSMI — Key Differences

| Dimension | Vanilla RAPTOR | HSMI (Structured Manifest Hybrid) |
|---|---|---|
| Node representation | Free-form LLM prose summary | Strict structured manifest (JSON/schema fields) |
| Parent-node creation | Full LLM re-summarization of raw child text | Deterministic set-union of entities/topics + lightweight constrained synthesis |
| Clustering method | UMAP + GMM | Same — UMAP + GMM (unchanged) |
| Update mechanism | Re-cluster & regenerate branch prose | Incremental centroid-matching; local patch only |
| Retrieval mechanism | Pure dense vector similarity | Hybrid: vector similarity **+** deterministic metadata/SQL-style filtering (entities, tags, timeframes) |
| Underlying storage | Vector database only | Hybrid store: vector DB + document/relational store (JSON) + graph/DAG for parent-child links |

---

## 4. RAPTOR Trade-offs Solved by HSMI

| RAPTOR Limitation | Why It Happened in Vanilla RAPTOR | How HSMI Fixes It |
|---|---|---|
| **Incremental updates & re-indexing** (O(N) → O(log N)) | Adding/editing/deleting one chunk alters global UMAP projections and GMM cluster boundaries, forcing periodic full-tree rebuilds | Deterministic document hierarchy + explicit DAG pointers mean modifying chunk *Cᵢ* only invalidates its direct ancestor branch — the rest of the tree stays untouched |
| **Cascading hallucinations / information drift** | Abstractive prose summaries smooth over edge cases, negative constraints, numbers, and dates; higher nodes then summarize these already-flawed summaries | Explicit schema fields (`metrics_and_constants`, `key_entities`, `system_invariants`) turn parent synthesis into a semi-deterministic rollup — with union, dedup, and bounds-checking on child keys before any high-level summary is generated |
| **Retrieval ambiguity (semantic overshoot)** | Dense vectors of broad abstract summaries can score high cosine similarity with specific queries that merely share thematic vocabulary, injecting noisy macro-context | Macro-understanding is decoupled from dense vectors — deterministic metadata filters (e.g. `entities CONTAINS "X" AND metrics.timeout IS NOT NULL`) run alongside vector search, eliminating false positives from broad summaries |
| **Ingestion prompt token overhead** | Summarizing an intermediate cluster requires concatenating the *full raw text* of every child chunk into the LLM's context window | Parent nodes only ingest compact JSON metadata blocks from their children — cutting prompt token volume for non-leaf layers by roughly **60–75%** |

---

## 5. Trade-offs Newly Introduced by HSMI

| New Trade-off | What Causes It |
|---|---|
| **Schema fragility & domain rigidity** | Fixed keys like `metrics_and_constants` or `system_invariants` work well for technical specs, financial tables, and API documentation — but break down on conversational transcripts, literature, narrative history, or open-ended legal discourse. This forces a choice between maintaining domain-specific schemas (ongoing engineering cost) or using generic fields (which erodes precision). |
| **Query-time translation latency** | Running structured filters requires parsing the raw user query into matching schema attributes (entities, constraints, target intent) via an extra LLM/NER inference step in the critical query path — adding roughly **150–500 ms** of latency before retrieval even starts. |
| **Vulnerability to poorly formatted raw documents** | The hybrid tree construction leans heavily on structural hierarchy (Markdown headers, sections, DOM trees) to establish baseline groupings. Unstructured PDFs, OCR scans, or flat continuous prose without clear section markers cause this grouping to collapse, forcing a fallback to heuristic clustering that loses the deterministic-update advantage. |
| **Operational & storage complexity** | Vanilla RAPTOR needs only a vector database. HSMI needs a **hybrid data store**: a vector database for dense summaries, a document/relational store for JSON querying and filtering, and a graph/DAG layer to trace parent-child relationships. Keeping transactional integrity across all three layers adds real operational overhead. |

---

## 6. Trade-offs That Still Remain

| Problem | Status | Why It Persists |
|---|---|---|
| **Ingestion cost vs. vanilla flat RAG** | **Unsolved** | Even though parent rollup is cheaper than in vanilla RAPTOR, HSMI still requires an LLM call *for every single leaf chunk* to extract the initial JSON schema. Vanilla flat RAG requires zero LLM calls at ingestion — only an embedding pass. |
| **Extractor capability bottleneck** | **Shifted** | High-fidelity JSON extraction demands strong instruction-following. Small local models (1B–4B parameters) often produce formatting errors, miss subtle entities, or drop implied constraints — this shifts (rather than eliminates) the hallucination problem from prose generation to schema extraction. |
| **Context window saturation** | **Partially solved** | Deciding exactly what context to return to the final generator is still non-trivial. Returning parent metadata alongside multiple raw child chunks consumes significant context-window budget, risking token limits or "lost-in-the-middle" attention degradation. |

**Possible mitigation (serving-layer only, doesn't alter HSMI itself):** Map-reduce generation for high-recall queries. When many leaf chunks match, get short per-chunk (or per-cluster) extractive answers first, then synthesize those short answers — instead of cramming every raw chunk into one prompt.

---

## 7. Where Each Framework Is Used / Targeted

**RAPTOR** is deployed for **static, complex, long-form corpora** where questions require document-wide synthesis rather than pinpoint fact lookup:
- Legal discovery
- Books
- Extensive financial filings
- Technical documentation
- General, open-ended QA and multi-hop reasoning over large document sets

**HSMI** is targeted at domains that are both **structured/technical** *and* **operationally demanding** — i.e., places where RAPTOR's ingestion cost, hallucination drift, or update rigidity would be unacceptable in production:
- Technical specifications
- Financial tables and data-heavy filings
- API documentation
- Compliance/policy documents with explicit rules, constraints, and named entities (e.g., auth flows, SLAs, regulatory clauses)
- Any RAG deployment that needs frequent incremental updates (living documentation, evolving codebases) and deterministic, auditable filtering rather than pure semantic similarity

HSMI is explicitly **not** well-suited to conversational transcripts, literature, narrative history, open-ended legal discourse, or poorly structured/unstructured source documents — these are exactly the cases where its schema-based assumptions break down (see Section 5).

---

## Summary

HSMI doesn't replace RAPTOR's core insight (recursive, tree-shaped aggregation via UMAP + GMM clustering) — it re-implements *what gets stored and generated at each node*, swapping free-form LLM prose for structured, schema-bound manifests. This trades away some of RAPTOR's flexibility and generality (it needs structured or well-formatted source documents, and pays a latency cost at query time) in exchange for cheaper, safer parent-node construction and near-free incremental updates. It does **not** solve RAPTOR's fundamental ingestion cost problem relative to plain flat RAG — it only reduces it — and it introduces a new dependency on small-model extraction quality as its principal failure point.
