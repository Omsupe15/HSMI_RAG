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

