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
