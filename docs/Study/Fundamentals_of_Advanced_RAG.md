# How to read each topic

Every topic ends with a **Classification** line using two axes, so you can tell at a glance what kind of problem a technique solves and what kind of machinery it needs.

**Axis 1 — Layer: architectural vs. prompting**

| Label | Meaning |
|---|---|
| **Architectural** | Adds or changes a pipeline stage, data store, or data layout. No amount of prompt rewording can substitute for it. |
| **Prompting** | The result's quality is governed mainly by the instruction given to an LLM. Fixable by editing the prompt. |
| **Architectural slot, prompting-governed** | The *place* in the pipeline is an architectural decision, but the quality inside that slot is a prompt problem. |

**Axis 2 — Mechanism: deterministic vs. model intelligence**

| Label | Meaning |
|---|---|
| **Deterministic** | Rules or classical algorithms. Same input → same output, always. No model involved. |
| **Learned model** | An embedding model or cross-encoder scores or encodes text. It produces no new text, and for a fixed model and input the output is stable. |
| **Generative LLM** | An LLM writes new text. Output can vary between runs and depends on the prompt. This is the most expensive and least predictable mechanism. |

**Why it matters:** a retrieval failure that is *architectural* (e.g., the right chunk was never fetched) will not be fixed by a better prompt, and a *prompting* failure (e.g., a sloppy query-rewrite instruction) will not be fixed by buying a vector database.

---

## Pipeline map: where each topic sits

```
 INDEXING TIME (offline, once per document)        QUERY TIME (online, per question)

   Documents                                          User query
      │                                                  │
      ▼                                                  ▼
 [4] CHUNKING ───────►  [3] STORAGE  ◄────────  [2] PRE-RETRIEVAL
 split into pieces      embed + index            rewrite / expand /
                        (vector, graph,          decompose / HyDE
                         keyword, hybrid)             │
                              │                       │
                              └────►  RETRIEVAL  ◄────┘
                                          │
                                          ▼
                                 [1] POST-RETRIEVAL
                                 rerank → compress
                                          │
                                          ▼
                                    LLM GENERATION
```

Two consequences of this layout:

1. **Chunking and storage are decided first and are the most expensive to change**, because changing them means re-processing and re-embedding the whole corpus.
2. **Pre- and post-retrieval techniques are query-time add-ons.** You can add, remove, or tune them without touching the index.

---
---

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

# PART 2 — PRE-RETRIEVAL TECHNIQUES

Pre-retrieval techniques transform the **user's question** before it touches the index. They exist because user questions are often a poor search query: ambiguous, conversational, compound, or phrased very differently from the documents.

> **Common principle:** all four techniques below are *query-side* changes. They need no re-indexing, so they are the cheapest place to experiment. They also all add at least one LLM call, so each adds latency.

---

## 2.1 Query Rewriting / Rephrasing

**From your overview:** An LLM strips conversational fluff and reformulates ambiguous questions into context-rich search phrases, using recent conversation history.

### How it works

1. **Collect inputs:** the latest user message plus the last few turns of chat history (and optionally known context such as the active product or document).
2. **Prompt a small, fast LLM** to produce a *standalone* search query: resolve pronouns and references ("it", "that one"), drop pleasantries, keep all entities, numbers, and identifiers, and output only the query.
3. **Run retrieval with the rewritten query** (keep the original query as a fallback or as an additional search).

**Worked example**

| Turn | Text |
|---|---|
| User | "Tell me about the Kindle Paperwhite." |
| Assistant | *(answers)* |
| User | "How long does it last on a charge?" |
| Raw query to vector DB | "How long does it last on a charge?" ← "it" is meaningless to the index |
| Rewritten query | "Kindle Paperwhite battery life per charge" |

*Reasoning:* the vector database has no memory of the conversation. Without rewriting, follow-up questions are embedded without their referent and retrieve noise.

### Variants

- **Standalone-question condensation**: the classic conversational-RAG step described above.
- **Rewrite → Retrieve → Read**: rewriting treated as an explicit, optionally trainable stage (a small model can be trained to rewrite for the specific retriever).
- **Step-back prompting**: rewrite a very specific question into a broader one to retrieve background principles first ("Why did my pump fail at 80 psi?" → "What are the operating pressure limits and failure modes of this pump?").

### Tools

| Tool | Role |
|---|---|
| **LangChain** `create_history_aware_retriever` | Built-in condense-then-retrieve pattern |
| **LlamaIndex** `CondenseQuestionChatEngine` / `CondensePlusContextChatEngine` | Chat engines that rewrite follow-ups into standalone questions |
| **DSPy** | Lets you *optimize* the rewrite prompt against a metric instead of hand-tuning |
| **Any small LLM** | Haiku-class / "mini" hosted models or a local 3–8B model; speed and low cost matter more than raw capability here |

### Where it is useful

- Chatbots and assistants with multi-turn conversations (the biggest win).
- Noisy user input: typos, filler, voice transcripts, run-on questions.
- Vocabulary mismatch between how users ask and how documents are written.

### Where it hurts

- **Exact-match queries** (error codes, SKUs, citations): an aggressive rewrite can strip or paraphrase the very token a keyword index needs. Preserve identifiers verbatim, or search with both the original and the rewrite.
- Single-turn, already clear queries: skip it (a simple router or length/pronoun check can decide) to save latency.
- **Entity drift:** the LLM may "helpfully" add details the user never said. Use a low temperature and log original vs. rewritten queries for auditing.

**Classification:** **Architectural slot, prompting-governed.** Adding a pre-retrieval LLM step with access to chat history is an architectural decision; whether the rewrites are *good* is almost entirely a prompt-quality problem. **Generative LLM.**

---

## 2.2 Query Expansion

**From your overview:** One user query becomes several variations (typically 3–4 perspectives). The pipeline runs a parallel vector search for each variation to ensure comprehensive topic coverage.

### How it works

1. **Generate variants.** An LLM produces N alternative phrasings or angles of the same question. Example for "How do I reduce cloud costs?": *"AWS cost optimization strategies"*, *"right-sizing compute instances"*, *"reserved instances vs. spot pricing"*.
2. **Embed each variant** and run all searches **in parallel** (async), so latency is roughly one search rather than N.
3. **Merge results:** take the union, de-duplicate by chunk ID, and fuse rankings. **Reciprocal Rank Fusion (RRF)** is the usual choice; the multi-query + RRF combination is often called *RAG-Fusion*.
4. **(Optional) Rerank** the merged set against the *original* query to prevent drift (see 1.1).

*Reasoning:* one phrasing embeds to one point in vector space. Relevant documents worded differently may sit in a neighboring region, which a single search misses. Several phrasings probe several regions, which raises **recall**.

### Deterministic alternatives (so you can choose consciously)

LLM expansion is not the only kind. Classical expansion is rule-based and needs no model call:

| Method | How | Typical use |
|---|---|---|
| Synonym lists / thesauri / ontologies | Query terms are expanded using curated word lists (e.g., "MI" → "myocardial infarction") | Keyword (BM25) search in specialized domains |
| Pseudo-relevance feedback (e.g., RM3) | Take top results, extract their frequent terms, add them to the query | Keyword search, no LLM |

Use deterministic expansion when you need predictable, auditable behavior or the vocabulary is controlled; use LLM expansion for open-ended natural language.

### Tools

| Tool | Role |
|---|---|
| **LangChain** `MultiQueryRetriever` | Generates query variants and merges the results |
| **LlamaIndex** `QueryFusionRetriever` | Multi-query generation with reciprocal-rank fusion built in |
| **Elasticsearch / OpenSearch synonym filters** | Deterministic expansion at the keyword-index level |
| `asyncio` / thread pools | Run the N searches concurrently |

### Where it is useful

- Short, vague, or ambiguous queries.
- Exploratory research ("what's known about X?") where breadth beats precision.
- **Recall-critical** work: legal discovery, medical literature, compliance searches.
- Corpora with inconsistent terminology across documents.

### Where it hurts

- Precise lookups: extra variants add noise (**query drift**), pushing in off-topic chunks.
- Tight latency or cost budgets: you pay one LLM call plus N retrievals.
- Without a reranker or RRF, a large union of results can *lower* precision.

**Classification:** **Mixed.** The fan-out-and-fuse structure is **architectural**; the diversity and quality of the variants is **prompting-governed**. Mechanism: variant generation is **Generative LLM**; the merge/RRF step is **Deterministic**.

---

## 2.3 Query Decomposition

**From your overview:** Compound questions are split by an orchestrator into isolated sub-queries (e.g., "Compare Q3 results of Company A and Company B" → "Company A Q3 results" and "Company B Q3 results"), each retrieved independently, then aggregated.

### How it works

1. **Detect** whether the query is compound or multi-hop. This is often a cheap router step; simple queries skip decomposition entirely.
2. **Plan.** An LLM outputs a structured list of sub-queries (ideally JSON via structured output), marking dependencies between them.
3. **Execute.**
   - *Independent* sub-queries (like the A-vs-B example) run **in parallel**.
   - *Dependent* sub-queries (multi-hop, e.g., "What is the population of the city where the company that makes the Kindle is headquartered?": first find the company, then its headquarters city, then that city's population) run **sequentially**, with each answer feeding the next query.
4. **Retrieve per sub-query**, each with its own top-k and, if useful, its own metadata filter (e.g., `company = "A"`).
5. **Aggregate.** Either answer each sub-question separately and synthesize, or pool all retrieved contexts, labeled by sub-query, into one final generation call.

*Reasoning:* a single embedding of "compare A and B" lands *between* the two topics and may match documents about neither well. It also lets one entity monopolize the top-k. Decomposition gives each part of the question its own focused search and its own share of the retrieval budget.

### Tools

| Tool | Role |
|---|---|
| **LlamaIndex** `SubQuestionQueryEngine` (with one `QueryEngineTool` per data source) | Ready-made decompose → route → answer → synthesize pipeline |
| **LangChain / LangGraph** | Structured-output "query analysis" step; LangGraph for loops, branching, and parallel execution |
| **DSPy** | Programmatic, optimizable multi-step retrieval |
| **Structured output / JSON-schema mode** | Makes the plan machine-parseable so the *orchestration itself* stays deterministic code |

### Where it is useful

- Comparisons ("A vs. B"), multi-entity questions, "list all X that…" aggregation.
- **Multi-hop** reasoning where the answer to one lookup is the input to the next.
- Questions spanning different sources (a document store *and* a SQL database).

### Where it hurts

- Single-fact questions, or any question one chunk can answer. It only adds latency and cost.
- **Error compounding:** a bad decomposition, or a wrong answer in step 1 of a dependent chain, poisons later steps. Cap the number of sub-queries and steps.
- Implicit links *between* the sub-answers can be lost if the final synthesis step doesn't see the raw evidence.

**Classification:** **Architectural**: an orchestrator, executors (parallel/sequential), and an aggregator are real pipeline components. The planner prompt is a **prompting** concern within it. Mechanism: planning is **Generative LLM**; orchestration and aggregation plumbing is **Deterministic** code.

---

## 2.4 Hypothetical Document Embeddings (HyDE)

**From your overview:** An LLM generates a fake, idealized answer. The system embeds that fake answer instead of the query and uses it to find documents with a similar factual structure.

### How it works

1. **Generate a hypothetical document.** Prompt an LLM: *"Write a short passage that answers this question."* It may contain wrong facts, and that is acceptable.
2. **Embed the passage** using the *same* embedding model as the index. (In the original HyDE paper, the embedding step acted as a "filter" that discards much of the invented detail and keeps the general content shape.)
3. **Search the index** with that embedding.
4. **Return real documents.** The fake passage is thrown away; only genuine retrieved documents go to the generator.

*Reasoning:* questions and answers have different linguistic form: a question is short and interrogative, a document passage is declarative and detailed. Embedding models narrow that gap but do not remove it. A hypothetical *answer* looks like a document, so the search becomes **document-to-document** similarity rather than **question-to-document**.

### Variants and tuning

- Generate several hypothetical documents and **average** their embeddings (optionally with the query embedding) for stability.
- Steer style with the prompt ("write it like a clinical-trial abstract", "like a Python docs page") so the fake answer resembles your corpus.

### Tools

| Tool | Role |
|---|---|
| **LangChain** `HypotheticalDocumentEmbedder` | Wraps an LLM and embeddings model into a HyDE embedder |
| **LlamaIndex** `HyDEQueryTransform` (used with a transform query engine) | Applies HyDE to any query engine |
| **Any LLM + any embedding model** | The technique is only a few lines of glue code |

### Where it is useful

- **Zero-shot or cold-start** domains: no labeled query-document pairs, no budget to fine-tune the embedder.
- Short, abstract questions against long, technical documents (scientific papers, specs).
- A visible *style mismatch* between questions and corpus text.

### Where it hurts

- **The LLM doesn't know the domain.** For niche, proprietary, or very recent topics the hypothetical answer can point the search at the wrong region of the index, which is worse than using the raw query.
- Exact-keyword or identifier queries gain nothing.
- Adds an LLM call of latency. With strong modern query-document embedding models, gains over plain search vary, so **A/B test** instead of assuming.

**Classification:** **Architectural slot, prompting-governed** (the generation prompt controls how useful the fake document is). Mechanism: **Generative LLM** (for the fake document) + **Learned model** (embedder).

---

## 2.5 Pre-retrieval techniques at a glance

| Technique | Problem it solves | LLM calls | Searches | Main risk |
|---|---|---|---|---|
| Rewriting | Ambiguous / context-dependent / noisy queries | 1 | 1 | Dropping exact identifiers; entity drift |
| Expansion | Vocabulary mismatch; low recall | 1 | N (parallel) | Query drift; noise without fusion/rerank |
| Decomposition | Compound or multi-hop questions | 1+ (planner, maybe synthesis) | One per sub-query | Error compounding; over-splitting |
| HyDE | Question-vs-document style gap | 1 | 1 | Wrong hypothetical answer misleads search |

They are **composable**: a common production chain is *rewrite → (decompose if compound) → expand each sub-query → hybrid retrieve → rerank*.

---
---

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

# PART 4 — CHUNKING STRATEGIES

Chunking is the process of breaking down large documents into smaller, coherent text units before generating embeddings and inserting them into storage. The size, structure, and boundaries of chunks fundamentally dictate retrieval precision and context quality for generation.

> **The Goldilocks Problem:** Chunks that are too small lose surrounding context and semantic continuity; chunks that are too large dilute specific facts, exceed embedding context limits, and inject irrelevant noise into generation.

| Chunking Strategy | Splitting Mechanism | Pros | Cons / Risks |
|---|---|---|---|
| **Fixed Length** | Character / token count with optional overlap | Fast, predictable, zero model overhead | Mid-sentence truncations, severed context |
| **Content-Aware / Structural** | Document formatting (Markdown headers, HTML DOM, code AST, LaTeX) | Preserves author hierarchy and document logic | Inconsistent chunk sizes; requires structured source files |
| **Sentence / Paragraph** | NLP boundary rules (punctuation, spaCy, NLTK) | Preserves complete semantic propositions and thoughts | Irregular chunk lengths; long paragraphs may exceed limits |
| **Embedding-Based Semantic** | Cosine distance / semantic drift between adjacent sentences | Highly coherent topic boundaries, self-adapting | Slower indexing, additional embedding model calls |

---

## 4.1 Fixed-Length Chunking

**Overview:** Splits text into uniform blocks based on a strict count of characters or tokens (e.g., 512 tokens per chunk), typically maintaining a sliding window with an overlap buffer (e.g., 10–20% overlap). It is highly predictable and computationally cheap, but it frequently cuts thoughts mid-sentence or mid-word.

### How it works

1. **Count units:** The text stream is scanned by character count, word count, or model-specific tokenizer tokens (e.g., `tiktoken` for OpenAI models).
2. **Slice at fixed boundaries:** When the window reaches the target limit (e.g., 512 tokens), the chunk is emitted.
3. **Sliding overlap:** The start pointer moves forward by `Chunk Size - Overlap` (e.g., 512 - 50 = 462 tokens), retaining overlapping tokens so words spanning boundaries aren't completely lost.

### Tools

| Tool | Role |
|---|---|
| **LangChain** `CharacterTextSplitter` / `TokenTextSplitter` | Fixed character/token slicing |
| **LangChain** `RecursiveCharacterTextSplitter` | Slices with fallback separators (`\n\n`, `\n`, ` `) before hard cutting |
| **LlamaIndex** `TokenTextSplitter` / `SentenceSplitter` | Fixed token chunking with overlap |
| **tiktoken** / **Hugging Face Tokenizers** | Low-level token counting and slicing |

### Where it is useful

- Baseline prototyping and fast MVP setup.
- Unstructured raw text dumps with no consistent layout, punctuation, or markup.
- Computationally constrained offline ingestion pipelines needing zero-overhead indexing.

### Where it is not enough

- Frequently cuts thoughts mid-sentence or mid-word if token boundaries fall arbitrarily.
- Semantic units (tables, bulleted lists, arguments) get split across multiple chunks, degrading embedding quality.

**Classification:** **Architectural.** Mechanism: **Deterministic**.

---

## 4.2 Content-Aware & Structural Chunking

**Overview:** Splits documents along their formatting markers, such as Markdown headers (`#`, `##`, `###`), HTML tags (`<h1>`, `<p>`, `<table>`, DOM hierarchy), JSON/YAML keys, code Abstract Syntax Trees (ASTs), or LaTeX tags.

### How it works

1. **Parse document syntax:** A parser analyzes the hierarchical structure of the document (DOM tree, Markdown AST, code AST, or LaTeX environment).
2. **Segment by logical blocks:** The document is split along structural nodes (e.g., headers or sections), keeping related tables, code snippets, lists, and subsections intact.
3. **Attach breadcrumb metadata:** Hierarchical context (e.g., `{"h1": "API Reference", "h2": "Authentication", "h3": "OAuth2"}`) is injected into the chunk's metadata or prepended to the text to preserve parent context.

### Tools

| Tool | Role |
|---|---|
| **LangChain** `MarkdownHeaderTextSplitter` | Splits markdown on header boundaries and attaches header metadata |
| **LangChain** `HTMLHeaderTextSplitter` / `HTMLSectionSplitter` | Splits HTML by structural tags |
| **LlamaIndex** `MarkdownNodeParser` / `HTMLNodeParser` | Extracts document hierarchy into nodes |
| **Unstructured.io** / **Docling** | Layout-aware document parsers that extract tables, sections, and structural headers into self-contained elements |
| **Tree-sitter** / **LangChain** `PythonCodeTextSplitter` | Language-aware AST code chunking for functions and classes |

### Where it is useful

- Highly structured documentation (API docs, user manuals, wikis, Markdown notes).
- Web scraping pipelines (HTML DOM hierarchy and table extraction).
- Source code repositories where function/class boundaries must remain unbroken.

### Where it is not enough

- Messy, scanned PDFs or unstructured free-form text lacking consistent semantic markup.
- Sections that are exceptionally long (a single Markdown section might span 5,000 words), which still require secondary sub-chunking.

**Classification:** **Architectural.** Mechanism: **Deterministic**.

---

## 4.3 Sentence-Based / Paragraph-Based Chunking

**Overview:** Uses natural language libraries (like NLTK or spaCy) or regex rules to split text strictly at sentence or paragraph boundaries. This preserves full thoughts but results in irregular chunk lengths.

### How it works

1. **Rule / Model-based sentence boundary detection:** NLP tokenizers detect sentence terminators while handling edge cases like abbreviations ("e.g.", "Dr.", "U.S."), numbers ("3.14"), and quotation marks.
2. **Accumulate sentences:** Sentences are bundled together until a target length threshold is reached without breaking any individual sentence.
3. **Paragraph boundaries:** Splitting occurs primarily on double line breaks (`\n\n`), keeping paragraphs as unified conceptual units.

### Tools

| Tool | Role |
|---|---|
| **NLTK** `nltk.tokenize.sent_tokenize` | Rule-based Punkt sentence tokenizer |
| **spaCy** `nlp.create_pipe('sentencizer')` | Fast rule-based or statistical sentence segmentation |
| **LangChain** `SpacyTextSplitter` / `NLTKTextSplitter` | Sentence-aware chunking integrations |
| **LlamaIndex** `SentenceSplitter` | Packs full sentences up to chunk size limits |

### Where it is useful

- Narrative text, academic papers, prose, legal contracts, and news articles where sentences contain complete, self-contained propositions.
- Preserving grammatical integrity and complete thoughts for generation and reranking.

### Where it is not enough

- High variance in chunk length: paragraphs can range from 10 words to 1,000 words.
- Does not understand whether consecutive sentences actually belong to the same topic.

**Classification:** **Architectural.** Mechanism: **Deterministic** (rule-based NLP / statistical tokenization).

---

## 4.4 Embedding-Based Semantic Chunking

**Overview:** Computes the vector embeddings of individual sequential sentences and measures the distance (or cosine similarity) between them. The pipeline dynamically places a breakpoint only when the semantic drift between adjacent sentences exceeds a specific threshold, meaning a new topic has started.

### How it works

1. **Sentence segmentation:** The text is first broken down into sequential sentences or small micro-windows.
2. **Embed sentences:** Each sentence (or a sliding window of combined adjacent sentences) is passed through an embedding model to generate dense vector representations.
3. **Compute semantic distance:** The cosine distance between sentence $i$ and sentence $i+1$ is calculated across the sequence:
   $$\text{Distance} = 1 - \text{CosineSimilarity}(e_i, e_{i+1})$$
4. **Detect threshold breakpoints:** Breakpoints are identified where distance spikes (e.g., exceeding a fixed percentile threshold such as the 95th percentile of distances, or a standard deviation cutoff). Chunks are partitioned at these inflection points.

```
Sentence:   S1 ─── S2 ─── S3 ─────── S4 ─── S5 ─── S6
Similarity:   0.88   0.85     0.32      0.89   0.87
                       ▲
            [ Semantic Drop / Breakpoint ]
Chunk 1: {S1, S2, S3}
Chunk 2: {S4, S5, S6}
```

### Tools

| Tool | Role |
|---|---|
| **LangChain** `SemanticChunker` (`langchain_experimental.text_splitter`) | Semantic chunker with percentile, standard deviation, and interquartile threshold methods |
| **LlamaIndex** `SemanticSplitterNodeParser` | Dynamic semantic similarity node splitter |
| **FastEmbed** / **Sentence-Transformers** | In-process sentence embedding generation for local, low-latency semantic splitting |
| **Chonkie** / **Semantic Router** | Lightweight dedicated semantic chunking libraries |

### Where it is useful

- Multi-topic conversational transcripts, meeting minutes, interview transcripts, and lecture notes where topic transitions occur without explicit headers or formatting cues.
- High-value knowledge bases where retrieval precision is prioritized over ingestion speed.

### Where it is not enough / Trade-offs

- **High indexing cost & latency:** Requires generating embeddings for every single sentence during ingestion, multiplying API costs or compute time.
- **Hyperparameter sensitivity:** Threshold tuning is critical; too aggressive creates fragmented tiny chunks, while too loose merges disparate topics.

**Classification:** **Architectural.** Mechanism: **Learned model** (sentence embeddings) + **Deterministic** (distance thresholding).

---

## 4.5 Chunking strategies at a glance

| Strategy | Splitting Trigger | Overhead | Best Suited For | Main Failure Mode |
|---|---|---|---|---|
| **Fixed Length** | Character / token count | Minimal (CPU only) | Raw text dumps, quick baselines | Broken words/sentences, severed context |
| **Content-Aware / Structural** | Headers, HTML tags, AST | Low (Syntax parser) | Codebases, API documentation, Markdown docs | Oversized sections requiring secondary splitting |
| **Sentence / Paragraph** | NLP punctuation rules (`\n\n`, `.`) | Low (NLP tokenizer) | Prose, books, research papers, legal text | Wide chunk size variance |
| **Semantic (Embedding-based)** | Cosine distance spike between sentences | High (Embeddings per sentence) | Transcripts, podcasts, unstructured multi-topic text | High indexing cost, sensitivity to threshold hyperparameters |
