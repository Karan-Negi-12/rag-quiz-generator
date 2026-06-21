# Enterprise RAG with Gemini + ChromaDB

## Features

### 1. Document Ingestion Pipeline
- Load TXT documents
- Chunk documents using CharacterTextSplitter
- Generate embeddings using Gemini Embeddings
- Store vectors in ChromaDB
- Batch processing for large datasets

### 2. Retrieval-Augmented Generation (RAG)
- User query embedding
- Semantic similarity search
- Top-K chunk retrieval
- Gemini 2.5 Flash answer generation
- Context-aware responses

### 3. Quiz Generation Pipeline
- Read chunks directly from ChromaDB
- Random chunk selection
- No vector search required
- No query embedding required
- Generate MCQs from random knowledge chunks
- Return quiz in JSON format

---

## Architecture

### RAG Pipeline

Documents
↓
Chunking
↓
Gemini Embeddings
↓
ChromaDB
↓
User Query
↓
Query Embedding
↓
Vector Similarity Search
↓
Top K Chunks
↓
Gemini 2.5 Flash
↓
Answer

---

### Quiz Generation Pipeline

Documents
↓
Chunking
↓
Gemini Embeddings
↓
ChromaDB
↓
Read Stored Chunks
↓
Random Chunk Selection
↓
Gemini 2.5 Flash
↓
MCQ Generation (JSON)

Note:
The Quiz Generator does not perform semantic retrieval.
It directly reads existing chunks from ChromaDB and randomly samples chunks to generate assessments.

---

## Tech Stack

- Python
- LangChain
- Gemini Embeddings (gemini-embedding-001)
- Gemini 2.5 Flash
- ChromaDB
- dotenv

---

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`

```env
GOOGLE_API_KEY=your_api_key
```

Run Ingestion:

```bash
python injestion_pipeline.py
```

Run RAG Retrieval:

```bash
python retrieval_pipeline.py
```

Run Quiz Generator:

```bash
python quiz_generation_pipeline.py
```

## Concepts Demonstrated

- Document Chunking
- Vector Embeddings
- Vector Databases
- Semantic Search
- Cosine Similarity
- HNSW Retrieval
- Retrieval-Augmented Generation (RAG)
- Random Knowledge Sampling
- Automated MCQ Generation