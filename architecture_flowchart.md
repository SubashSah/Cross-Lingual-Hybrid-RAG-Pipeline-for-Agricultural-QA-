# System Architecture & Flowchart: Cross-Lingual Hybrid RAG Pipeline

This document details the working architecture and dataflow of the Tomato Cultivation RAG system. It covers offline ingestion, cross-lingual dense/sparse indexing, multi-branch retrieval with Score-Based Fusion (SBF), and LLM generation.

---

## 1. End-to-End Architecture Flowchart

```mermaid
flowchart TD
    %% Phase 1: Ingestion & Indexing
    subgraph P1["Phase 1: Ingestion and Indexing (Offline)"]
        DOC["Source Document<br/>data/tomato.md"] --> CHUNK["Chunking<br/>chunk_size = 2500, overlap = 300"]
        CHUNK --> EN_CHUNKS["English Text Chunks<br/>original_chunks.pkl"]
        
        %% Dense Branch
        EN_CHUNKS --> EMB["Embedding Model<br/>BAAI/bge-m3"]
        EMB --> FAISS_DB["Dense Vector Store<br/>FAISS"]
        
        %% Sparse Branch
        EN_CHUNKS --> TRANS["Translate to Nepali<br/>translation_service.py"]
        TRANS --> BM25_IDX["BM25 Keyword Index<br/>storage/bm25"]
    end

    %% Phase 2: Query Input
    Q["User Query<br/>(Nepali / English)"]

    %% Phase 3: Retrieval & Fusion
    subgraph P2["Phase 2: Multi-Branch Retrieval and Fusion"]
        %% Branch 1: Dense Semantic
        S_RET["Dense Semantic Search<br/>Top 8 Chunks"]
        S_DOCS["8 Semantic Chunks"]
        S_RET --> S_DOCS

        %% Branch 2: Hybrid SBF
        H_ENG["Hybrid Retrieval Engine<br/>Score-Based Fusion"]
        H_SEM["Dense Retrieval (FAISS)<br/>Top 30 Candidates"]
        H_SEM_NORM["Score Normalization (0 to 1)<br/>Min-Max"]
        H_BM25["Keyword Retrieval (BM25)<br/>Top 30 Candidates"]
        H_KW_NORM["Score Normalization (0 to 1)<br/>Min-Max"]
        SBF["Score-Based Fusion (SBF)<br/>Score = 0.7 * S_sem + 0.3 * S_kw"]
        H_RANK["Sort and Top-k Selection (k=8)"]
        H_DOCS["8 Hybrid Fused Chunks"]

        H_ENG --> H_SEM
        H_SEM --> H_SEM_NORM
        H_ENG --> H_BM25
        H_BM25 --> H_KW_NORM
        H_SEM_NORM --> SBF
        H_KW_NORM --> SBF
        SBF --> H_RANK
        H_RANK --> H_DOCS

        %% Branch 3: Whole Document
        FULL_CTX["Complete Document Context<br/>tomato.md full text"]
    end

    %% Connecting Query to Retrieval branches
    Q --> S_RET
    Q --> H_ENG
    Q --> FULL_CTX

    %% Phase 4: Generation
    subgraph P3["Phase 3: LLM Generation"]
        PROMPT["RAG Prompt Template<br/>prompt_service.py"]
        LLM["WiseAI vLLM Server<br/>ChatOpenAI"]

        PROMPT --> LLM
    end

    S_DOCS --> PROMPT
    H_DOCS --> PROMPT
    FULL_CTX --> PROMPT

    %% Output Answers
    LLM --> ANS1["1. Semantic Answer"]
    LLM --> ANS2["2. Hybrid SBF Answer"]
    LLM --> ANS3["3. Full Document Answer"]
```

---

## 2. Score-Based Fusion (SBF) Mechanism

```mermaid
flowchart LR
    Q["User Question"] --> D1["Dense Retrieval<br/>Top 30 Candidates"]
    D1 --> D2["Score Normalization (0 to 1)<br/>Min-Max"]

    Q --> S1["Sparse Retrieval<br/>Top 30 Candidates"]
    S1 --> S2["Score Normalization (0 to 1)<br/>Min-Max"]

    D2 --> F["Score-Based Fusion<br/>Score = 0.7 * S_sem + 0.3 * S_kw"]
    S2 --> F
    F --> R["Sort by Final Score"]
    R --> TOP["Top 8 Fused Chunks"]
```

---

### Mathematical Formulation:

1. **Min-Max Score Normalization**:
   For any retrieved candidate list with scores $S$:
   $$S_{\text{norm}}(d) = \frac{S(d) - \min(S)}{\max(S) - \min(S)}$$
   *(If $\max(S) == \min(S)$, all nonzero scores normalize to $1.0$)*

2. **Weighted Score-Based Fusion (SBF)**:
   $$\text{FinalScore}(d) = \alpha \cdot S_{\text{sem\_norm}}(d) + (1 - \alpha) \cdot S_{\text{kw\_norm}}(d)$$
   - $\alpha = 0.7$ (70% weight on semantic relevance, 30% on keyword matching).
   - Unretrieved items in either modality receive a score of $0.0$.

---

## 3. Core Architecture Phases

### Phase 1: Ingestion & Cross-Lingual Indexing (`ingestion.py`)
1. **Document Loading**: Source guide (`data/tomato.md`) is ingested.
2. **Chunking**: Chunked with `RecursiveCharacterTextSplitter` using **`chunk_size = 2500`** and `chunk_overlap = 300`.
3. **Dense Vector Store**:
   - Chunks are vectorized using `BAAI/bge-m3` (multilingual embeddings) and stored in the FAISS vector index.
4. **Cross-Lingual Sparse Index (BM25)**:
   - English chunks are translated into Nepali to bridge the vocabulary gap.
   - Tokenized Nepali text is indexed using `BM25Okapi` in `storage/bm25`.

### Phase 2: Multi-Branch Retrieval & Fusion (`retriever_service.py`)
- **Semantic Search**: Fetches top 8 nearest neighbors directly from the dense vector index.
- **Hybrid Search (SBF)**: Fetches 30 semantic + 30 keyword candidates, normalizes both candidate pools into $[0, 1]$, fuses them using $\alpha = 0.7$, and selects the top 8 chunks.
- **Full Context**: Uses complete document text for benchmark comparison.

### Phase 3: LLM Generation (`services/llm_service.py`)
- Uses `create_rag_prompt()` with strict grounding instructions.
- WiseAI vLLM server generates 3 answers:
  1. **Semantic Answer**
  2. **Hybrid SBF Answer**
  3. **Full Document Answer**
