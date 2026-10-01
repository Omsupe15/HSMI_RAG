# PART 1 — POST-RETRIEVAL TECHNIQUES

Post-retrieval techniques run **after** the database has returned candidates and **before** the LLM sees them. They improve *what the LLM reads*, not *what was fetched*.

> **A rule that applies to everything in this part:** a post-retrieval step can only work with what retrieval already returned. If the right chunk is not in the candidate set, reranking and compression cannot rescue it. Fix chunking, storage, or the query instead.

---

## 1.1 Reranking

**From your overview:** Vector databases find matches quickly using distance (e.g., cosine similarity) but miss fine-grained nuance, keyword matching, and complex logic. Reranking takes the top results (e.g., 25 chunks) and re-scores them with a cross-encoder so the best answers move to the top.

### How it works

**Step 1: Wide first-stage retrieval (optimize recall).**
A fast retriever (vector search, BM25, or hybrid) returns a generous candidate set, typically 25–100 chunks. *Reasoning:* the first stage only needs to make sure the right answer is *somewhere in the pile*; it does not need to order the pile well.

**Step 2: Pairwise scoring by a cross-encoder (optimize precision).**
For each candidate, the reranker receives the query and the chunk **concatenated as one input** (conceptually `[CLS] query [SEP] chunk [SEP]`). Self-attention runs across both texts at once, so every query token can attend to every chunk token. The model outputs one relevance score for that pair.
*Reasoning:* this joint view lets the model catch things a single vector cannot encode, such as negation ("*not* FDA-approved"), qualifiers ("before 2020"), numbers, and which entity is the subject of a sentence.

**Step 3: Sort and truncate.**
Candidates are sorted by score and the top-k (commonly 3–10) go to the LLM. Optionally, apply a minimum score threshold to drop clearly irrelevant chunks.

### Why a two-stage design (bi-encoder → cross-encoder)

| | Bi-encoder (vector search) | Cross-encoder (reranker) |
|---|---|---|
| Input | Query and document encoded **separately** | Query and document encoded **together** |
| Precomputable? | **Yes**: document vectors are computed once at indexing time | **No**: a score exists only for a specific (query, chunk) pair |
| Cost per query | One embedding + one ANN lookup (milliseconds, near-constant) | One model forward pass **per candidate** (grows linearly with candidates) |
| Accuracy on nuance | Lower | Higher |
| Role | Fast recall over millions of chunks | Slow precision over dozens of chunks |

Running a cross-encoder over the whole corpus would be far too slow, and that is the entire reason reranking is a *second* stage.

### Variants worth knowing

- **Pointwise cross-encoders** (the standard): score each (query, chunk) pair independently.
- **Listwise LLM rerankers**: an LLM (or a model like jina-reranker-v3) sees many candidates *together* and orders them. Highest quality on hard queries, highest latency and cost.
- **Late interaction (ColBERT-style)**: stores per-token vectors for each document and compares at the token level. It sits *between* bi- and cross-encoders in cost and quality, and is **not** a cross-encoder.

### Tools

| Category | Tools | Notes |
|---|---|---|
| Managed API | **Cohere Rerank** (current generation is Rerank 4.0, with Pro and Fast variants), **Voyage rerank**, **Jina Reranker API**, rerank endpoints built into Pinecone and Elasticsearch | One API call, no infrastructure; per-request or per-unit pricing |
| Open-source, self-hosted | **bge-reranker-v2-m3** (≈0.6B parameters, multilingual, the most common open baseline), **Qwen3-Reranker** (several sizes), **jina-reranker-v3** (listwise), **mxbai-rerank-v2** | Zero per-query fee, full data control; you provide the GPU/CPU. Check each model's license before commercial use |
| Lightweight / CPU | **FlashRank** (small ONNX models, no heavy ML dependencies) | Good for edge devices, serverless, or tight budgets |
| Frameworks | `sentence-transformers` `CrossEncoder`; LangChain (contextual-compression retriever with a reranker as the compressor); LlamaIndex (reranking node postprocessors) | Glue code to plug a reranker after any retriever |

> **Practical notes**
> - Relevance **scores are model-specific**. Cohere's own migration notes for Rerank 3.5 → 4.0 warn that scores differ and any hard-coded thresholds must be re-tuned. Never reuse a threshold across rerankers.
> - Many cross-encoders have limited input length (often 512 tokens for the pair), so overly long chunks get truncated, so make sure your chunking is compatible.
> - Vendor and leaderboard comparisons are volatile and often benchmark-specific. **Evaluate 2–3 candidates on your own queries** (metrics: nDCG@10, MRR, recall@k).

### Where it is useful

- Any RAG system where the correct chunk is retrieved but sits at rank 8–25 rather than rank 1–3.
- Queries with constraints, negations, dates, or numbers.
- **Hybrid search** outputs, where a reranker gives a single trustworthy ordering across two different scoring systems.
- Multilingual corpora (strong multilingual rerankers exist).
- Long-tail or ambiguous queries where the embedding model is "close but not exact".

### Where it is not worth it

- Tiny corpora, or cases where the top-3 from vector search is already almost always right (measure first).
- Hard latency budgets that cannot absorb tens to hundreds of milliseconds.
- When the real failure is **recall**, i.e., the right chunk is not in the candidate set. Widening the candidate pool, hybrid search, or better chunking is the fix.

### Trade-offs and failure modes

- Added latency and cost scale with the **number of candidates × chunk length**.
- Reranking cannot create relevance that retrieval never surfaced.
- Rerankers are trained on general web/QA data; highly specialized domains (legal, biomedical, code) may need a domain-tuned model or fine-tuning.

**Classification:** **Architectural** (it is a new pipeline stage; no prompt substitutes for it). **Learned model** (cross-encoder; deterministic for a fixed model and input). *Exception:* listwise LLM rerankers are **Generative LLM**.

---

## 1.2 Context Compression

**From your overview:** Even a highly relevant 500-word chunk may contain only one sentence that answers the question. Passing everything wastes money and clutters the LLM's context window. Compression extracts or shrinks the retrieved text so only the needed information reaches the generator.

### How it works

A compressor sits **between retrieval (or reranking) and generation** and has one signature: `(query, documents) → shorter documents`. Three families exist, and they differ a lot in mechanism:

#### A. LLM Chain Extractor (extractive, generative)

1. For each retrieved chunk, call a small, fast LLM with a prompt like: *"Copy verbatim only the sentences that help answer this question. If none, return NO_OUTPUT."*
2. Keep the extracted sentences; discard chunks that return nothing.
3. Pass the survivors to the generator.

*Reasoning:* because it **copies** text instead of rewriting it, wording and numbers stay faithful to the source, so hallucination risk is low. The cost is **one LLM call per chunk** (parallelizable, but it adds latency and spend). A cheaper sibling, the **LLM chain filter**, only answers "keep this chunk or drop it" and does not trim inside the chunk.

#### B. Embeddings-based filter (semantic compression, learned model)

1. Split each retrieved chunk into sentences.
2. Embed the query and every sentence with the same embedding model.
3. Compute cosine similarity of each sentence against the query.
4. Drop sentences below a threshold (or keep only the top-n).

*Reasoning:* no text generation is involved, so it is fast, cheap, and repeatable. The weakness: a sentence cut out of its surroundings can lose meaning (pronouns like "it" or "this" lose their referent), and thresholds are specific to the embedding model, so tune them on your data. A common mitigation is keeping each surviving sentence together with its immediate neighbors.

#### C. Token pruning vs. summarisation (your overview groups these; they are different things)

- **Token-level pruning** (e.g., the LLMLingua family): a small language model measures how *informative* each token is (surprisal/perplexity, or a trained token classifier in LLMLingua-2) and deletes low-information tokens. The output may look ungrammatical but remains readable to an LLM. Published results report substantial compression ratios; the achievable ratio depends heavily on the task and the settings, so treat "up to 50%" as a conservative planning figure, not a guarantee.
- **Summarisation** (abstractive): an LLM rewrites the content more briefly. It can compress hard, but it can **introduce content not in the source** and may drop exact figures or quotes.

#### Combining them: a compression pipeline

A typical ordering is cheap-to-expensive: **split → remove redundant/duplicate chunks → embedding-similarity filter → (optional) LLM extraction.** Each step shrinks the input to the next one.

### Tools

| Tool | What it provides |
|---|---|
| **LangChain** | `ContextualCompressionRetriever` (wraps any retriever) with compressors `LLMChainExtractor`, `LLMChainFilter`, `EmbeddingsFilter`, and `DocumentCompressorPipeline` to chain them. *These legacy retriever classes have moved between packages across LangChain releases; check your installed version's docs for current import paths.* |
| **LlamaIndex** | Node postprocessors such as `SentenceEmbeddingOptimizer` and similarity-cutoff postprocessors; a LongLLMLingua postprocessor; response synthesizers (compact/tree-summarize) that condense context |
| **LLMLingua / LongLLMLingua / LLMLingua-2** (Microsoft) | Open-source prompt-compression library for token-level pruning; LongLLMLingua is question-aware, which suits RAG |
| **RECOMP** (research) | Trains extractive and abstractive compressors specifically for retrieved context |
| **Any small LLM** | Haiku-class / "mini" hosted models or a local 3–8B model for the extractor step |

### Where it is useful

- Long chunks where only a sentence or two is relevant.
- Answers that need many sources, where raw context would overflow the window.
- High-volume, cost-sensitive applications (you pay per input token on every query).
- **Small-context or resource-limited models**, where long context is slow or memory-hungry.
- Reducing the "lost in the middle" effect: LLMs often use information in the middle of a long context less reliably than information at the start or end, and shorter, denser context helps.

### Where it is not worth it

- **Verbatim-critical domains** (legal, compliance, medical) unless you restrict yourself to *extractive* methods and keep source links. Avoid abstractive summarisation here.
- Chunks that are already short and focused.
- When compression cost exceeds the generation cost it saves. Running an LLM extractor over 20 chunks can cost more than simply sending them. Do the break-even math.

### Practical order of operations

**Rerank first, then compress.** Reranking cuts 50 candidates to ~5; compressing 5 chunks is much cheaper than compressing 50.

**Classification:** **Architectural slot** (a stage between retrieval and generation), with the mechanism varying by family: embeddings filter = **Learned model, deterministic**; LLMLingua pruning = **Learned model, deterministic**; LLM chain extractor and summarisation = **Generative LLM** (here the extraction prompt is a **prompting** concern: it determines what gets dropped).

---
---