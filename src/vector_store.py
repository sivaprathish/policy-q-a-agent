from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.embedder import (
    embed_query,
    embed_texts,
)


@dataclass
class SearchResult:
    score: float
    record: dict[str, Any]


class SimpleVectorStore:
    """
    Small local vector store using NumPy.

    Embeddings are normalized, so the dot product
    acts as cosine similarity.
    """

    def __init__(
        self,
        directory: Path,
        name: str,
    ) -> None:
        self.directory = directory
        self.name = name

        self.vector_path = (
            directory / f"{name}_vectors.npy"
        )

        self.records_path = (
            directory / f"{name}_records.json"
        )

        self.vectors: np.ndarray | None = None
        self.records: list[dict[str, Any]] = []

    def build(
        self,
        records: list[dict[str, Any]],
        text_field: str = "text",
    ) -> None:
        """
        Create embeddings and save the vector index.
        """
        valid_records = [
            record
            for record in records
            if str(
                record.get(text_field, "")
            ).strip()
        ]

        if not valid_records:
            raise ValueError(
                f"No valid records for index: {self.name}"
            )

        texts = [
            str(record[text_field])
            for record in valid_records
        ]

        vectors = embed_texts(texts)

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            self.vector_path,
            vectors,
        )

        with self.records_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                valid_records,
                file,
                indent=2,
                ensure_ascii=False,
            )

        self.vectors = vectors
        self.records = valid_records

    def load(self) -> None:
        """
        Load vectors and their corresponding metadata.
        """
        if not self.vector_path.exists():
            raise FileNotFoundError(
                f"Vector file missing: {self.vector_path}"
            )

        if not self.records_path.exists():
            raise FileNotFoundError(
                f"Record file missing: {self.records_path}"
            )

        self.vectors = np.load(
            self.vector_path
        )

        with self.records_path.open(
            encoding="utf-8",
        ) as file:
            self.records = json.load(file)

        if len(self.vectors) != len(self.records):
            raise ValueError(
                f"Vector and record counts do not match "
                f"for index {self.name}."
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, set[str]] | None = None,
    ) -> list[SearchResult]:
        """
        Search the index and optionally apply metadata filters.
        """
        if self.vectors is None:
            self.load()

        assert self.vectors is not None

        query_vector = embed_query(query)

        candidate_indices: list[int] = []

        for index, record in enumerate(
            self.records
        ):
            if filters and not self._matches_filters(
                record,
                filters,
            ):
                continue

            candidate_indices.append(index)

        if not candidate_indices:
            return []

        candidate_vectors = self.vectors[
            candidate_indices
        ]

        scores = (
            candidate_vectors @ query_vector
        )

        ranked_positions = np.argsort(
            scores
        )[::-1][:top_k]

        results: list[SearchResult] = []

        for position in ranked_positions:
            original_index = candidate_indices[
                int(position)
            ]

            results.append(
                SearchResult(
                    score=float(
                        scores[position]
                    ),
                    record=self.records[
                        original_index
                    ],
                )
            )

        return results

    @staticmethod
    def _matches_filters(
        record: dict[str, Any],
        filters: dict[str, set[str]],
    ) -> bool:
        """
        Check whether a record satisfies all filters.
        """
        for field, allowed_values in filters.items():
            value = record.get(field)

            if value is None:
                return False

            if str(value) not in allowed_values:
                return False

        return True


def load_json_records(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            f"Expected JSON list in: {path}"
        )

    return records