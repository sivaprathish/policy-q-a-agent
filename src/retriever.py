from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import (
    CHUNK_TOP_K,
    DOCUMENT_TOP_K,
    SECTION_TOP_K,
    VECTOR_DB_DIR,
)

from src.vector_store import (
    SearchResult,
    SimpleVectorStore,
)


@dataclass
class RetrievalResult:
    documents: list[SearchResult]
    sections: list[SearchResult]
    chunks: list[SearchResult]

    @property
    def context(self) -> str:
        """
        Convert retrieved chunks into LLM context.
        """
        context_parts: list[str] = []

        for index, result in enumerate(
            self.chunks,
            start=1,
        ):
            record = result.record

            citation = record.get(
                "metadata",
                {},
            ).get(
                "citation",
                (
                    f"Section "
                    f"{record.get('section_number', '')}"
                ),
            )

            chunk_text = record.get(
                "raw_text",
                record.get("text", ""),
            )

            context_parts.append(
                f"[Evidence {index}]\n"
                f"Citation: {citation}\n"
                f"Similarity score: {result.score:.4f}\n"
                f"Policy text:\n{chunk_text}"
            )

        return "\n\n".join(
            context_parts
        )


class HierarchicalRetriever:
    """
    Performs:

    Document retrieval
        ↓
    Section retrieval within selected documents
        ↓
    Chunk retrieval within selected sections
    """

    def __init__(
        self,
        vector_directory: Path = VECTOR_DB_DIR,
    ) -> None:
        self.document_store = SimpleVectorStore(
            directory=vector_directory,
            name="documents",
        )

        self.section_store = SimpleVectorStore(
            directory=vector_directory,
            name="sections",
        )

        self.chunk_store = SimpleVectorStore(
            directory=vector_directory,
            name="chunks",
        )

    def retrieve(
        self,
        query: str,
        document_top_k: int = DOCUMENT_TOP_K,
        section_top_k: int = SECTION_TOP_K,
        chunk_top_k: int = CHUNK_TOP_K,
    ) -> RetrievalResult:
        if not query.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        # Stage 1: retrieve relevant documents
        document_results = (
            self.document_store.search(
                query=query,
                top_k=document_top_k,
            )
        )

        document_ids = {
            str(
                result.record["document_id"]
            )
            for result in document_results
        }

        if not document_ids:
            return RetrievalResult(
                documents=[],
                sections=[],
                chunks=[],
            )

        # Stage 2: retrieve sections only
        # from selected documents
        section_results = (
            self.section_store.search(
                query=query,
                top_k=section_top_k,
                filters={
                    "document_id": document_ids
                },
            )
        )

        section_ids = {
            str(
                result.record["section_id"]
            )
            for result in section_results
        }

        if not section_ids:
            return RetrievalResult(
                documents=document_results,
                sections=[],
                chunks=[],
            )

        # Stage 3: retrieve chunks only
        # from selected sections
        chunk_results = (
            self.chunk_store.search(
                query=query,
                top_k=chunk_top_k,
                filters={
                    "section_id": section_ids
                },
            )
        )

        return RetrievalResult(
            documents=document_results,
            sections=section_results,
            chunks=chunk_results,
        )