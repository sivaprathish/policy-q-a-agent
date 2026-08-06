from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import streamlit as st

from config import (
    PROCESSED_DIR,
    RAW_DIR,
    VECTOR_DB_DIR,
)
from src.build_indexes import build_all_indexes
from src.pdf_processor import process_pdf
from src.qa_agent import generate_answer
from src.retriever import HierarchicalRetriever


# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 220
CHUNK_OVERLAP = 40


# ============================================================
# Streamlit page
# ============================================================

st.set_page_config(
    page_title="Policy Q&A",
    page_icon="📘",
    layout="wide",
)

st.title("Policy Q&A Agent")

st.caption(
    "Upload a company policy document and ask questions "
    "using hierarchical retrieval."
)


# ============================================================
# Session state
# ============================================================

def initialize_state() -> None:
    defaults = {
        "is_ready": False,
        "retriever": None,
        "messages": [],
        "processed_filename": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


initialize_state()


# ============================================================
# File utilities
# ============================================================

def save_json(
    data: Any,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def clear_directory(
    directory: Path,
) -> None:
    if not directory.exists():
        return

    for path in directory.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


# ============================================================
# Processed index handling
# ============================================================

def save_processed_indexes(
    result: dict[str, Any],
    processed_directory: Path,
) -> None:
    """
    Save the JSON files expected by build_all_indexes().

    Supports processor outputs containing either:

    document_index, section_index, chunk_index

    or:

    document, sections, chunks
    """
    processed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_records = result.get(
        "document_index"
    )

    if document_records is None:
        document = result.get(
            "document",
            {},
        )

        document_title = document.get(
            "title",
            "Company Policy",
        )

        document_records = [
            {
                "document_id": document.get(
                    "document_id",
                    "",
                ),
                "level": "document",
                "title": document_title,
                "text": (
                    f"Policy document: "
                    f"{document_title}"
                ),
                "source_file": document.get(
                    "source_file",
                    "",
                ),
                "metadata": document.get(
                    "metadata",
                    {},
                ),
            }
        ]

    section_records = (
        result.get("section_index")
        or result.get("sections")
        or []
    )

    chunk_records = (
        result.get("chunk_index")
        or result.get("chunks")
        or []
    )

    if not document_records:
        raise ValueError(
            "No document records were generated."
        )

    if not section_records:
        raise ValueError(
            "No policy sections were generated."
        )

    if not chunk_records:
        raise ValueError(
            "No policy chunks were generated."
        )

    normalized_sections: list[dict[str, Any]] = []

    for section in section_records:
        section_copy = dict(section)

        section_text = (
            section_copy.get("text")
            or section_copy.get("summary")
            or section_copy.get("raw_text")
            or ""
        )

        if not str(section_text).strip():
            section_number = section_copy.get(
                "section_number",
                "",
            )

            section_title = section_copy.get(
                "title",
                "Policy section",
            )

            section_text = (
                f"Section {section_number}: "
                f"{section_title}"
            )

        section_copy["text"] = section_text
        normalized_sections.append(
            section_copy
        )

    normalized_chunks: list[dict[str, Any]] = []

    for chunk in chunk_records:
        chunk_copy = dict(chunk)

        chunk_text = (
            chunk_copy.get("text")
            or chunk_copy.get("raw_text")
            or ""
        )

        if not str(chunk_text).strip():
            continue

        chunk_copy["text"] = chunk_text
        normalized_chunks.append(
            chunk_copy
        )

    if not normalized_chunks:
        raise ValueError(
            "All generated policy chunks are empty."
        )

    save_json(
        document_records,
        processed_directory
        / "document_index.json",
    )

    save_json(
        normalized_sections,
        processed_directory
        / "section_index.json",
    )

    save_json(
        normalized_chunks,
        processed_directory
        / "chunk_index.json",
    )


def validate_processed_files(
    processed_directory: Path,
) -> None:
    required_files = [
        processed_directory
        / "document_index.json",

        processed_directory
        / "section_index.json",

        processed_directory
        / "chunk_index.json",
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file was not created: "
                f"{file_path}"
            )

        with file_path.open(
            encoding="utf-8",
        ) as file:
            records = json.load(file)

        if not isinstance(records, list):
            raise ValueError(
                f"{file_path.name} must contain "
                f"a JSON list."
            )

        if not records:
            raise ValueError(
                f"{file_path.name} is empty."
            )


# ============================================================
# Existing index detection
# ============================================================

def existing_indexes_available() -> bool:
    required_files = [
        PROCESSED_DIR
        / "document_index.json",

        PROCESSED_DIR
        / "section_index.json",

        PROCESSED_DIR
        / "chunk_index.json",

        VECTOR_DB_DIR
        / "documents_vectors.npy",

        VECTOR_DB_DIR
        / "documents_records.json",

        VECTOR_DB_DIR
        / "sections_vectors.npy",

        VECTOR_DB_DIR
        / "sections_records.json",

        VECTOR_DB_DIR
        / "chunks_vectors.npy",

        VECTOR_DB_DIR
        / "chunks_records.json",
    ]

    return all(
        file_path.exists()
        for file_path in required_files
    )


def load_existing_retriever() -> None:
    if st.session_state.is_ready:
        return

    if not existing_indexes_available():
        return

    try:
        st.session_state.retriever = (
            HierarchicalRetriever(
                vector_directory=VECTOR_DB_DIR,
            )
        )

        st.session_state.is_ready = True

    except Exception:
        st.session_state.is_ready = False
        st.session_state.retriever = None


load_existing_retriever()


# ============================================================
# PDF upload
# ============================================================

uploaded_pdf = st.file_uploader(
    "Upload a company policy PDF",
    type=["pdf"],
)


# ============================================================
# PDF processing
# ============================================================

if uploaded_pdf is not None:
    if st.button(
        "Process and index policy",
        type="primary",
    ):
        try:
            RAW_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            PROCESSED_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            VECTOR_DB_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            clear_directory(
                PROCESSED_DIR
            )

            clear_directory(
                VECTOR_DB_DIR
            )

            safe_filename = Path(
                uploaded_pdf.name
            ).name

            pdf_path = (
                RAW_DIR
                / safe_filename
            )

            with pdf_path.open(
                "wb",
            ) as file:
                shutil.copyfileobj(
                    uploaded_pdf,
                    file,
                )

            processing_start = time.perf_counter()

            with st.status(
                "Processing policy...",
                expanded=True,
            ) as status:
                st.write(
                    "Extracting policy text "
                    "and detecting headings"
                )

                result = process_pdf(
                    pdf_path=pdf_path,
                    output_directory=PROCESSED_DIR,
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                )

                st.write(
                    "Saving document, section "
                    "and chunk indexes"
                )

                save_processed_indexes(
                    result=result,
                    processed_directory=(
                        PROCESSED_DIR
                    ),
                )

                validate_processed_files(
                    PROCESSED_DIR
                )

                st.write(
                    "Generating embeddings"
                )

                build_all_indexes(
                    processed_directory=(
                        PROCESSED_DIR
                    ),
                    vector_directory=(
                        VECTOR_DB_DIR
                    ),
                )

                st.write(
                    "Loading policy retriever"
                )

                st.session_state.retriever = (
                    HierarchicalRetriever(
                        vector_directory=(
                            VECTOR_DB_DIR
                        ),
                    )
                )

                st.session_state.is_ready = True

                st.session_state.processed_filename = (
                    safe_filename
                )

                st.session_state.messages = []

                processing_time = (
                    time.perf_counter()
                    - processing_start
                )

                status.update(
                    label="Policy is ready",
                    state="complete",
                )

            st.success(
                f"Policy processed successfully "
                f"in {processing_time:.2f} seconds."
            )

        except Exception as error:
            st.session_state.is_ready = False
            st.session_state.retriever = None

            st.exception(error)


# ============================================================
# Ready status
# ============================================================

if st.session_state.is_ready:
    if st.session_state.processed_filename:
        st.success(
            f"Policy ready: "
            f"{st.session_state.processed_filename}"
        )
    else:
        st.success(
            "The existing policy index is ready."
        )


# ============================================================
# Chat
# ============================================================

if st.session_state.is_ready:
    st.divider()

    question = st.chat_input(
        "Ask a question about the company policy"
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        try:
            response_start = time.perf_counter()

            with st.spinner(
                "Searching the policy..."
            ):
                retrieval = (
                    st.session_state
                    .retriever
                    .retrieve(question)
                )

                answer = generate_answer(
                    question=question,
                    retrieval=retrieval,
                )

            response_time = (
                time.perf_counter()
                - response_start
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "retrieval": retrieval,
                    "response_time": response_time,
                }
            )

        except Exception as error:
            response_time = (
                time.perf_counter()
                - response_start
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "An error occurred while "
                        "answering the question."
                    ),
                    "response_time": response_time,
                }
            )

            st.exception(error)


# ============================================================
# Chat history
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if message["role"] != "assistant":
            continue

        response_time = message.get(
            "response_time"
        )

        if response_time is not None:
            st.caption(
                f"Response time: "
                f"{response_time:.2f} seconds"
            )

        retrieval = message.get(
            "retrieval"
        )

        if not retrieval:
            continue

        if retrieval.chunks:
            with st.expander(
                "View supporting policy evidence"
            ):
                for index, result in enumerate(
                    retrieval.chunks,
                    start=1,
                ):
                    record = result.record

                    metadata = record.get(
                        "metadata",
                        {},
                    )

                    citation = metadata.get(
                        "citation",
                        (
                            f"Section "
                            f"{record.get('section_number', '')}"
                        ),
                    )

                    evidence_text = (
                        record.get("raw_text")
                        or record.get(
                            "text",
                            "",
                        )
                    )

                    st.markdown(
                        f"**Evidence {index}**"
                    )

                    st.caption(
                        f"{citation} · "
                        f"Similarity: "
                        f"{result.score:.3f}"
                    )

                    st.write(
                        evidence_text
                    )

                    if (
                        index
                        < len(retrieval.chunks)
                    ):
                        st.divider()

else:
    st.info(
        "Upload and process a company policy "
        "PDF to begin."
    )