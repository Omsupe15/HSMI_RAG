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
