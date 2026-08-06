## Policy Q&A Agent

An AI-powered Policy Q&A Agent that enables employees to ask natural language questions about company policies and receive accurate, citation-backed answers using Hierarchical Retrieval-Augmented Generation (Hierarchical RAG).

Instead of searching through lengthy employee handbooks manually, the system automatically extracts the document hierarchy and retrieves information through Document → Section → Chunk retrieval before generating an answer with an LLM.

#### Features
Upload company policy PDFs
Automatic policy hierarchy extraction
Hierarchical indexing of documents
Semantic search using embeddings
Hierarchical Retrieval (Document → Section → Chunk)
Citation-backed responses
Response latency measurement
Supports employee handbooks, HR policies, SOPs, compliance documents, and company manuals
Architecture
                    Company Policy PDF
                            │
                            ▼
                PDF Processing Engine
                            │
                            ▼
                Hierarchy Detection
          (Document → Section → Chunk)
                            │
                            ▼
              Embedding Generation
                            │
                            ▼
                Vector Database Indexes
      ┌──────────────┬──────────────┬──────────────┐
      │              │              │
      ▼              ▼              ▼
 Document Index   Section Index   Chunk Index
                            │
                            ▼
                     User Question
                            │
                            ▼
                 Document Retrieval
                            │
                            ▼
                  Section Retrieval
                            │
                            ▼
                   Chunk Retrieval
                            │
                            ▼
                  LLM Answer Generator
                            │
                            ▼
          Grounded Answer with Policy Citations