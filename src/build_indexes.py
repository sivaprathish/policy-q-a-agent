from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    PROCESSED_DIR,
    VECTOR_DB_DIR,
)

from src.vector_store import (
    SimpleVectorStore,
    load_json_records,
)


def build_all_indexes(
    processed_directory: Path = PROCESSED_DIR,
    vector_directory: Path = VECTOR_DB_DIR,
) -> None:
    """
    Build document, section and chunk indexes.
    """
    document_records = load_json_records(
        processed_directory
        / "document_index.json"
    )

    section_records = load_json_records(
        processed_directory
        / "section_index.json"
    )

    chunk_records = load_json_records(
        processed_directory
        / "chunk_index.json"
    )

    document_store = SimpleVectorStore(
        directory=vector_directory,
        name="documents",
    )

    section_store = SimpleVectorStore(
        directory=vector_directory,
        name="sections",
    )

    chunk_store = SimpleVectorStore(
        directory=vector_directory,
        name="chunks",
    )

    document_store.build(
        records=document_records,
        text_field="text",
    )

    section_store.build(
        records=section_records,
        text_field="text",
    )

    chunk_store.build(
        records=chunk_records,
        text_field="text",
    )

    print("=" * 60)
    print("VECTOR INDEXES CREATED")
    print("=" * 60)
    print(
        f"Documents: {len(document_records)}"
    )
    print(
        f"Sections:  {len(section_records)}"
    )
    print(
        f"Chunks:    {len(chunk_records)}"
    )
    print(
        f"Location:  {vector_directory.resolve()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build policy vector indexes."
    )

    parser.add_argument(
        "--processed",
        type=Path,
        default=PROCESSED_DIR,
    )

    parser.add_argument(
        "--vector-db",
        type=Path,
        default=VECTOR_DB_DIR,
    )

    arguments = parser.parse_args()

    build_all_indexes(
        processed_directory=arguments.processed,
        vector_directory=arguments.vector_db,
    )


if __name__ == "__main__":
    main()