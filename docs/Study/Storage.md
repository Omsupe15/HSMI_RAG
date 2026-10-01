# PART 3 — STORAGE

Storage decides **how knowledge is represented and searched**. Each storage type answers a different kind of question well and others poorly. That is why mature systems usually combine several rather than pick one.

| Storage type | Represents knowledge as | Best at |
|---|---|---|
| Vector | Points in embedding space | "Find text that *means* something similar" |
| Graph | Entities and typed relationships | "How are these things *connected*?" |
| Keyword / relational | Exact tokens and structured rows | "Find this *exact* string / filter by this *field*" |
| Hybrid | Vector + keyword together | Production-grade general search |

---

## 3.1 Vector Storage

**From your overview:** The most common RAG storage. It stores text chunks alongside high-dimensional embeddings for semantic similarity search, either in dedicated vector databases (Pinecone, Milvus, Qdrant, using HNSW-style indexes) or as extensions of existing databases (pgvector, Redis, MongoDB).

### How it works

**At indexing time**
1. Each chunk goes through an **embedding model**, producing a vector (commonly several hundred to a few thousand dimensions).
2. The vector is stored together with the chunk text and **metadata** (source, page, date, tenant, access tags).
3. An **ANN (approximate nearest neighbor) index** is built over the vectors.

**At query time**
1. The query is embedded with the **same model** (some models require a specific query prefix or instruction, so check the model card).
2. The index returns the k vectors closest to the query by cosine similarity, dot product, or L2 distance.
3. The chunk text and metadata for those vectors are returned.

**Why ANN instead of exact search?** Exact ("flat") search compares the query against every vector, so cost grows linearly with corpus size. ANN indexes give up a small amount of recall to get near-logarithmic lookups.

| Index type | Idea | Trade-off |
|---|---|---|
| **HNSW** | A multi-layer proximity graph. Sparse upper layers allow long "jumps"; the dense bottom layer refines. Search enters at the top and greedily descends toward the query. Key knobs: `M` (links per node), `efConstruction` (build quality), `efSearch` (query-time recall vs. speed) | Excellent recall/speed; **memory-hungry** (graph + vectors usually in RAM) |
| **IVF** | Cluster vectors; at query time search only the `nprobe` nearest clusters | Smaller memory than HNSW; lower recall unless nprobe is raised |
| **Quantization (PQ / SQ / binary)** | Compress vectors into fewer bytes | Big memory savings; small accuracy loss; often combined with IVF or HNSW |
| **DiskANN-style** | Graph index designed to live on SSD | Billion-scale on modest RAM; higher latency than in-memory |

**Metadata filtering** (e.g., "only documents from tenant 42 after 2024") is a first-class concern. Naive *post-filtering* (search first, filter after) can return fewer than k results; good engines implement filtering *during* graph traversal. Test your database's behavior under selective filters.

### Dedicated vs. extension

| | Dedicated vector DB | Extension to an existing DB |
|---|---|---|
| Examples | **Pinecone** (managed, serverless), **Milvus** (open-source, distributed), **Qdrant** (strong filtering, sparse vector support), **Weaviate** (built-in hybrid search), **Chroma** and **LanceDB** (lightweight/embedded) | **pgvector** (PostgreSQL), **Redis** vector search, **MongoDB Atlas Vector Search**, Elasticsearch/OpenSearch k-NN, **sqlite-vec** |
| Strengths | Built for scale, QPS, filtering, tuning knobs | One system to run; transactional consistency with your other data; reuse existing ops and security |
| Weaknesses | Another system to operate and secure | Performance ceilings and fewer tuning options at very large scale |

**Libraries (not databases):** **FAISS**, **hnswlib**, and **ScaNN** provide the index algorithms in-process. They are great for prototypes and embedded use, but they don't give you persistence, replication, or access control by themselves.

> **Rule of thumb:** small corpus or prototype → in-memory/embedded; already on Postgres with moderate scale → pgvector; large scale, heavy filtering, or high QPS → a dedicated database. Benchmark at *your* scale before deciding.

### Where it is useful

- Natural-language questions over unstructured text (documentation, support tickets, articles).
- Paraphrase tolerance: matching "car won't start" with "engine fails to turn over".
- Multilingual and cross-lingual retrieval (with a multilingual embedding model).
- Multimodal search (image/text) if you use a multimodal embedder.

### Where it is not enough

- **Exact tokens**: serial numbers, SKUs, error codes, rare names. Embeddings blur them. Add a lexical leg (3.3/3.4).
- **Relationships** and multi-hop questions (3.2).
- Numeric aggregation and structured filters (use SQL).

### Important architectural coupling

The **embedding model is part of the storage contract.** Changing the model (or even its dimensionality) means **re-embedding the entire corpus**. Choose it early, version it, and store the model name alongside the index.

**Classification:** **Architectural.** Mechanism: **Learned model** (embeddings) + **Deterministic** ANN algorithms (approximate, but repeatable for a given index).

---

## 3.2 Graph Storage

**From your overview:** Information is mapped as explicit networks of entities and relationships rather than isolated text chunks. Nodes are entities/concepts/people; edges are the relationships between them. Graph stores such as Neo4j let an LLM navigate multi-hop paths (e.g., Document A connects to Document C through a shared entity in Document B).

### How it works

**At indexing time**
1. **Extraction.** Parse each chunk into **entities** (nodes) and **relations** (edges), stored as triples such as `(Acme Corp) —[ACQUIRED]→ (Beta Inc)`. This is done by an LLM prompted with a schema, or, more deterministically, by NER models (e.g., spaCy) plus rules.
2. **Entity resolution.** Merge aliases ("IBM", "International Business Machines") into one node. This step is easy to underestimate. Poor resolution silently fragments the graph.
3. **Store** as a property graph: typed nodes, typed edges, properties, and a pointer back to the **source chunk** for provenance. Many setups also embed nodes or chunks for vector entry points.

**At query time**
1. **Entity linking:** identify which graph nodes the question mentions.
2. **Traversal:** expand from those nodes along edges (k-hop neighborhood, path finding, or a generated graph query such as Cypher).
3. **Assemble context:** collect connected facts and their source chunks, and give them to the LLM.

*Reasoning:* a multi-hop question ("Which suppliers of the company Beta Inc acquired are in region X?") needs *links between facts*. Vector search retrieves facts that each look relevant in isolation; a graph walks the connection explicitly.

### Common patterns

- **Vector-seeded traversal:** vector search finds entry nodes, then the graph expands outward.
- **Text-to-graph-query:** the LLM writes the Cypher query. Powerful but non-deterministic; validate and constrain with the schema.
- **Community-based GraphRAG** (Microsoft GraphRAG): detect clusters of tightly connected entities, have an LLM write a summary per community, and answer *dataset-wide* questions ("what are the main themes across these reports?") from those summaries. Entity-centric "local" search handles specific questions.
- **Other graph-RAG designs:** LightRAG, HippoRAG (graph walks inspired by personalized PageRank).

### Tools

| Category | Tools |
|---|---|
| Graph databases | **Neo4j** (Cypher; built-in vector indexes; official GraphRAG Python package), **Memgraph**, **FalkorDB**, **Amazon Neptune**, **NebulaGraph**, **ArangoDB** |
| Graph-RAG frameworks | **Microsoft GraphRAG**, **LlamaIndex** `PropertyGraphIndex`, **LangChain** `LLMGraphTransformer` + Neo4j integration, **LightRAG** |
| Extraction | LLM-based extractors (flexible, costly, non-deterministic); spaCy NER/rules (cheaper, narrower, repeatable) |

### Where it is useful

- **Multi-hop and relationship-centric** questions: supply chains, org charts, fraud rings, drug–gene–disease links, legal citation networks, software dependency maps.
- **Corpus-wide synthesis** (global themes) that no single chunk contains.
- **Explainability:** the retrieved path *is* the reasoning trail, with provenance to sources.

### Where it is not worth it

- Simple FAQ/lookup RAG where a single chunk holds the answer.
- Fast-changing corpora, since re-extraction costs LLM calls and entity resolution must keep up.
- Teams without a defined domain schema or the capacity to maintain one.

### A qualification on your overview

Your overview says graph storage makes systems "highly resilient against hallucinations." It is more accurate to say graphs **ground answers in explicit, traceable facts**, which reduces some errors. But the graph is only as good as its extraction: a *missed or wrong edge becomes a silent retrieval failure*, and an LLM can still misread what it is given. Indexing cost (LLM calls per chunk) is typically the dominant expense.

**Classification:** **Architectural** (it is a different data model). Mechanism is **mixed**: extraction by an LLM is **Generative LLM** (non-deterministic, prompt/schema-sensitive); traversal and graph queries are **Deterministic**; Text-to-Cypher brings the generative part back into query time.

---

## 3.3 Keyword & Relational Databases

**From your overview:** Modern RAG keeps pre-semantic systems alive next to vector storage. Lexical search (Elasticsearch/OpenSearch with BM25) retrieves exact matches for IDs, acronyms, and names that embeddings dilute. Relational databases store raw un-chunked text, access control lists (ACLs), and conversation histories linked to primary keys.

### How lexical search works

1. **Analysis:** the text is tokenized, lowercased, and optionally stemmed ("running" → "run") with stop words removed.
2. **Inverted index:** each unique term maps to a *postings list* — the documents containing it, with term frequencies (and often positions, enabling phrase search).
3. **Query:** the query is analyzed the same way; postings for each term are looked up and combined.
4. **Scoring with BM25**, which balances three ideas:

```
score(D, Q) = Σ over query terms t:
      IDF(t) × [ tf(t,D) × (k1 + 1) ] / [ tf(t,D) + k1 × (1 − b + b × |D| / avgdl) ]
```

| Component | Meaning | Reasoning |
|---|---|---|
| `IDF(t)` | Rarer terms across the corpus weigh more | "kubelet" says more than "the" |
| `tf` with saturation (`k1`, commonly ≈ 1.2) | Repeating a term helps, with diminishing returns | Stops keyword stuffing from dominating |
| Length normalization (`b`, commonly ≈ 0.75) | Long documents are penalized relative to the average length `avgdl` | A term in a short passage is stronger evidence than in a long one |

Because no model is involved, lexical search is **fast, cheap, explainable, and exact**.

**Its weakness is vocabulary mismatch:** "car" will not match "automobile," and there is no understanding of meaning.

### How relational storage supports RAG

| Role | How it is used |
|---|---|
| **Source of truth** | Stores the original un-chunked documents and the chunk→parent mapping |
| **Parent-document / small-to-big retrieval** | Search with small, precise chunks; then fetch the larger parent section by ID to give the LLM fuller context |
| **Access control (security trimming)** | ACLs filter what each user may see, ideally *inside* the retrieval query, not after the LLM has already read it |
| **Metadata filters** | Date, document type, tenant, language |
| **Conversation memory** | Chat history keyed by session, used by query rewriting (2.1) |
| **Structured questions** | Text-to-SQL for counts, sums, and filters, where SQL is exact and vector search is not |

### Tools

| Category | Tools |
|---|---|
| Lexical search engines | **Elasticsearch**, **OpenSearch**, **Apache Solr / Lucene**, **Tantivy** (Rust) |
| In-process BM25 (Python) | `rank_bm25`, `bm25s`, SQLite **FTS5** (`bm25()` ranking built in) |
| Full-text inside a relational DB | PostgreSQL full-text search (`tsvector` + GIN index; note its `ts_rank` is *not* true BM25), ParadeDB `pg_search` (BM25 for Postgres) |
| Relational databases | **PostgreSQL**, **MySQL**, **SQLite** |

### Where it is useful

- **Identifier-heavy content:** part numbers, error codes, legal citations, drug names, API/function names, medical codes.
- Exact phrase and acronym lookups.
- **Compliance and multi-tenant systems** needing hard access control.
- Questions that are really structured data queries.

### Where it is not enough

- Purely conceptual or paraphrased questions on their own. Pair it with vectors (3.4).

**Classification:** **Architectural.** Mechanism: **Deterministic** (BM25 and SQL involve no model at all, which is why it stays valuable beside the model-driven parts of the stack).

---

## 3.4 Hybrid Storage

**From your overview:** To offset the limits of vector and lexical lookup, enterprise setups pivot to hybrid layouts: sparse (lexical) and dense (semantic) embeddings in one entity, queried in parallel and merged with Reciprocal Rank Fusion (RRF).

### How it works

1. **Two retrieval legs run in parallel** over the same corpus:
   - **Dense:** semantic vector search.
   - **Sparse:** BM25, or a *learned* sparse representation (e.g., SPLADE, BGE-M3's sparse output) that adds term-expansion on top of keyword matching.
2. **Fuse the two ranked lists.** This is the crux, because the scores are not comparable (cosine similarity lives in a small bounded range; BM25 is unbounded and corpus-dependent).
3. **(Optional) Rerank** the fused top-N with a cross-encoder (1.1).

### Fusion methods

| Method | How | Pros / cons |
|---|---|---|
| **Reciprocal Rank Fusion (RRF)** | Score each document by Σ 1 / (k + rank) across lists (k ≈ 60 is a common default) | **Uses only ranks**, so no score normalization needed; robust and nearly tuning-free. Ignores score *magnitude* |
| **Weighted score fusion** | Normalize scores (min-max or z-score), then blend with a weight (often called `alpha`) | Tunable per use case; sensitive to normalization quality |
| **Distribution-based fusion** (e.g., Qdrant's DBSF) | Normalize using each list's score distribution before combining | Handles skewed score distributions |

**RRF reference implementation** (tested):

```python
def rrf(rankings, k=60):
    """rankings: list of ranked lists of doc ids (best first)."""
    scores = {}
    for ranked in rankings:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

dense   = ["d1", "d2", "d3", "d4"]
lexical = ["d3", "d9", "d1", "d7"]
print(rrf([dense, lexical]))   # ['d1', 'd3', 'd2', 'd9', 'd4', 'd7']
```

*Reading the output:* `d1` and `d3` appear in **both** lists, so they float to the top. Agreement between two very different retrieval methods is strong evidence of relevance. `d9` (lexical-only) outranks `d4` (dense-only) because of its rank-2 position in the lexical list.

### Unified vs. federated

- **Unified:** one engine holds dense and sparse representations and fuses internally (Elasticsearch, OpenSearch, Weaviate, Qdrant, Milvus, Pinecone's sparse-dense support, Azure AI Search, Vespa). Simplest to operate.
- **Federated:** separate systems (e.g., Elasticsearch for BM25 + a vector DB, or PostgreSQL with `pgvector` + full-text search), fused in application code or SQL. More flexible, more moving parts.

### Tools

| Tool | Hybrid support |
|---|---|
| **Elasticsearch** | RRF retriever combining lexical and vector retrievers |
| **OpenSearch** | Hybrid query with score normalization (RRF support in recent versions) |
| **Weaviate** | Built-in hybrid search with an `alpha` blend and selectable fusion |
| **Qdrant** | Query API with prefetch and server-side fusion (RRF / DBSF); sparse vectors |
| **Milvus** | `hybrid_search` with RRF or weighted rankers; sparse vectors |
| **Pinecone** | Sparse-dense search |
| **LangChain** `EnsembleRetriever` | Combines BM25 and vector retrievers with weighted rank fusion |
| **LlamaIndex** `QueryFusionRetriever` | Fuses multiple retrievers |
| **PostgreSQL** | `pgvector` + full-text (or ParadeDB BM25) with RRF written in SQL |

### Where it is useful

- Real-world corpora mixing **natural language and identifiers** (support KBs, API docs, code search, e-commerce catalogs, legal and medical text).
- The sensible **production default**: the two methods fail in *different* ways, so combining them covers each other's blind spots.
- Latency is roughly the slower of the two legs, because they run in parallel.

### Where it is more than you need

- Purely semantic corpora with no identifiers or exact-match needs.
- Early prototypes where the extra index and tuning aren't yet justified.

**Classification:** **Architectural.** Mechanism: dense leg = **Learned model**; BM25 leg = **Deterministic**; fusion (RRF) = **Deterministic**.

---
---
