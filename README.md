# Quiz-Project

Standalone quiz generator that uses a pre-built Chroma vector store from the sibling RAG-Workflow project.

## Prereqs
- A built vector store at ../.chroma (produced by RAG-Workflow/scripts/rag/vector_store_build.py)
- Ollama for local LLMs (the project is Ollama-only)

## Quick start (Quiz)

Generate a quiz and test your skills. 

```sh
./scripts/bin/run_venv.sh ./master.py prepare
```

Auto-validate quiz answers using LLM with RAG context. This writes validated questions to both JSON and SQLite database (no user prompts required).
```sh
./scripts/bin/run_venv.sh ./master.py validate
```

## Quick start (Chat)

Start an interactive chat that keeps a sliding window of the last N Q/A pairs in memory, augments each turn with vector-store context (RAG), and answers using an Ollama model.

```sh
./scripts/bin/run_venv.sh ./master.py chat
```

## Config
Use these project files:

- params.yaml: the single source of truth for all configuration. Inline comments document every setting and override rule.
- schema.sql: SQLite database schema for storing validated quiz results.
- scripts/bin/run_venv.sh: helper to run commands inside the project's virtual environment.
- master.py: entry point to run workflows: prepare, validate, chat.
- requirements.txt: Python dependencies used by the project.

## Architecture Overview

The Quiz-Project follows a modular architecture with three main workflows that share common components for LLM interaction and RAG-based knowledge retrieval.

```mermaid
flowchart TB
    subgraph Input["Configuration & Input"]
        PARAMS[params.yaml<br/>Global Config]
        SCHEMA[schema.sql<br/>DB Schema]
        VECTOR[../.chroma<br/>Vector Store]
    end

    subgraph Master["Master Controller"]
        MAIN[master.py<br/>Entry Point]
    end

    subgraph Workflows["Core Workflows"]
        PREP[prepare<br/>Quiz Generation]
        VALID[validate<br/>LLM Auto-Validation]
        CHAT[chat<br/>Interactive Q&A]
    end

    subgraph Components["Shared Components"]
        LLM[LLM Client<br/>Ollama HTTP]
        RAG[RAG System<br/>Chroma + Embeddings]
        UTILS[Utils<br/>Parsing & I/O]
    end

    subgraph Storage["Data Storage"]
        JSON[JSON Files<br/>quiz.json, answers.json, validated_quiz.json]
        SQLITE[SQLite Database<br/>test_questions, test_answers]
        HISTORY[.quiz_history.json<br/>Deduplication]
    end

    PARAMS --> MAIN
    SCHEMA --> MAIN
    VECTOR --> RAG
    
    MAIN --> PREP
    MAIN --> VALID
    MAIN --> CHAT
    
    PREP --> LLM
    PREP --> RAG
    PREP --> UTILS
    
    VALID --> LLM
    VALID --> RAG
    VALID --> UTILS
    
    CHAT --> LLM
    CHAT --> RAG
    
    PREP --> JSON
    PREP --> HISTORY
    
    VALID --> JSON
    VALID --> SQLITE
    
    style MAIN fill:#e1f5fe
    style PREP fill:#f3e5f5
    style VALID fill:#e8f5e8
    style CHAT fill:#fff3e0
```

## Validation Flow Details

The validate workflow is fully automated and uses LLM-driven validation with RAG context:

```mermaid
flowchart LR
    subgraph Input["Input Files"]
        QUIZ[quiz.json<br/>Generated Questions]
        ANSWERS[answer_key.json<br/>Answer Explanations]
    end

    subgraph Validation["LLM Validation Process"]
        LOAD[Load Question<br/>& Answer Key]
        RAG_Q[RAG Query<br/>Question Text Only]
        CONTEXT[Retrieved Context<br/>from Vector Store]
        PROMPT[Build Validation Prompt<br/>Context + Question + Options]
        LLM_JUDGE[LLM Judgment<br/>True/False Decision]
    end

    subgraph Output["Validated Output"]
        FILTER[Filter True<br/>Responses Only]
        ENRICH[Enrich with<br/>Answer Text + Explanation]
        JSON_OUT[validated_quiz.json<br/>Clean Questions]
        DB_OUT[SQLite Database<br/>Structured Storage]
    end

    QUIZ --> LOAD
    ANSWERS --> LOAD
    LOAD --> RAG_Q
    RAG_Q --> CONTEXT
    CONTEXT --> PROMPT
    PROMPT --> LLM_JUDGE
    LLM_JUDGE --> FILTER
    FILTER --> ENRICH
    ENRICH --> JSON_OUT
    ENRICH --> DB_OUT
```

## Database Schema

The validate flow creates a SQLite database (configured via `sqlite_db` in params.yaml) with this schema:

- **test_questions**: question text, random UUID, explanation
- **test_answers**: one row per option, with true/false flag and FK to parent question

Access the database from command line:
```bash
sqlite3 quiz.db
.tables
.schema test_questions
SELECT * FROM test_questions LIMIT 5;
.quit
```

---or that uses a pre-built Chroma vector store from the sibling RAG-Workflow project.

## Prereqs
- A built vector store at ../.chroma (produced by RAG-Workflow/scripts/rag/vector_store_build.py)
- Ollama for local LLMs (the project is Ollama-only)

## Quick start (Quiz)

Generate a quiz and test your skills. 

```sh
./scripts/bin/run_venv.sh ./master.py prepare
```

Auto-validate quiz answers using LLM with RAG context. This writes validated questions to both JSON and SQLite database (no user prompts required).
```sh
./scripts/bin/run_venv.sh ./master.py validate
```

## Quick start (Chat)

Start an interactive chat that keeps a sliding window of the last N Q/A pairs in memory, augments each turn with vector-store context (RAG), and answers using an Ollama model.

```sh
./scripts/bin/run_venv.sh ./master.py chat
```


## Config
Use these project files:

- params.yaml: the single source of truth for all configuration. Inline comments document every setting and override rule.
- scripts/bin/run_venv.sh: helper to run commands inside the project’s virtual environment.
- master.py: entry point to run workflows: prepare, validate, chat.
- requirements.txt: Python dependencies used by the project.

---



## Diagram: Where ChromaEmbeddingRetriever Fits

```mermaid
flowchart LR
    subgraph Client["Client or Application Layer"]
        UQ["User Query: 'What is vector search?'"]
    end

    subgraph RAG["RAG Pipeline (Python Layer)"]
        EMB["SentenceTransformersTextEmbedder\n→ Create query embedding"]
        RET["ChromaEmbeddingRetriever\n→ Retrieve top-K documents"]
        DOCS["Top-K Documents with metadata"]
    end

    subgraph VectorStore["Chroma Document Store (Vector DB)"]
        VECS["Stored document embeddings\n(Chroma SQLite / .parquet)"]
        META["Metadata (tags, source, section, etc.)"]
    end

    subgraph LLM["LLM (Generator)"]
        PROMPT["Augmented Prompt:\n(Context + Query)"]
        RESP["Grounded Response"]
    end

    UQ --> EMB --> RET
    RET --> VECS
    VECS --> RET
    RET --> DOCS
    DOCS --> PROMPT
    PROMPT --> RESP
```

---

## In Code Terms

| Component | Class | Responsibility |
|------------|--------|----------------|
| **Embedder** | `SentenceTransformersTextEmbedder` | Converts query → vector |
| **Retriever** | `ChromaEmbeddingRetriever` | Finds similar vectors in Chroma |
| **Vector Store** | `ChromaDocumentStore` | Stores embeddings and metadata |
| **Generator (LLM)** | Ollama model or local LLM | Generates response based on context |

---

## Typical Chroma Stack Setup

When initialized:
```python
self._document_store = ChromaDocumentStore(persist_path=self.cfg.rag_persist)
self._retriever = ChromaEmbeddingRetriever(document_store=self._document_store)
```

- `ChromaDocumentStore` handles **storage** (via SQLite + .parquet).
- `ChromaEmbeddingRetriever` uses **similarity search** APIs to fetch results.
- Together, they implement **dense retrieval** (vector-based) inside your RAG pipeline.

---

## Concept Summary

| Stage | Description |
|-------|--------------|
| 1️⃣ Query Embedding | Encode user text into a high-dimensional vector |
| 2️⃣ Retrieval | `ChromaEmbeddingRetriever` finds nearest neighbors in vector space |
| 3️⃣ Context Assembly | Combine retrieved docs into prompt |
| 4️⃣ Generation | Send context + query to LLM for final answer |