from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz


# ============================================================
# Configuration
# ============================================================

HEADING_PATTERN = re.compile(
    r"""
    ^
    (?P<number>\d+(?:\.\d+)*)
    \s*
    [-–—.]
    \s*
    (?P<title>.+?)
    \s*$
    """,
    re.VERBOSE,
)

MAIN_HEADING_PATTERN = re.compile(
    r"""
    ^
    (?P<number>\d+)
    \s*
    [-–—.]
    \s*
    (?P<title>[A-Z][A-Z\s/&(),'-]+)
    $
    """,
    re.VERBOSE,
)

PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d+\s*$")

DEFAULT_CHUNK_SIZE = 220
DEFAULT_OVERLAP = 40


# ============================================================
# Data models
# ============================================================

@dataclass
class PDFLine:
    text: str
    page_number: int
    font_size: float
    font_name: str
    is_bold: bool
    bbox: tuple[float, float, float, float]


@dataclass
class PolicySection:
    section_id: str
    document_id: str
    section_number: str
    title: str
    level: int
    parent_section_id: str | None
    heading_path: list[str]
    page_start: int
    page_end: int
    text: str


@dataclass
class PolicyChunk:
    chunk_id: str
    document_id: str
    section_id: str
    section_number: str
    section_title: str
    heading_path: list[str]
    page_start: int
    page_end: int
    chunk_index: int
    text: str
    raw_text: str
    word_count: int
    metadata: dict[str, Any]


# ============================================================
# Utilities
# ============================================================

def stable_id(prefix: str, *values: str) -> str:
    combined = "|".join(values)
    digest = hashlib.sha256(
        combined.encode("utf-8")
    ).hexdigest()[:16]

    return f"{prefix}_{digest}"


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\uf0b7", "•")
    text = text.replace("\ufffe", "")
    text = text.replace("￾", "")

    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_bold_font(font_name: str) -> bool:
    font_name = font_name.lower()

    return any(
        marker in font_name
        for marker in (
            "bold",
            "black",
            "heavy",
            "semibold",
            "demibold",
        )
    )


def section_level(section_number: str) -> int:
    return section_number.count(".") + 1


def normalize_heading_title(title: str) -> str:
    title = clean_text(title)
    title = title.strip(" -–—.")
    return title


# ============================================================
# Line-level PDF extraction
# ============================================================

def extract_pdf_lines(pdf_path: Path) -> tuple[list[PDFLine], dict[str, Any]]:
    document = fitz.open(pdf_path)

    lines: list[PDFLine] = []

    metadata = {
        "title": document.metadata.get("title") or "",
        "author": document.metadata.get("author") or "",
        "subject": document.metadata.get("subject") or "",
        "page_count": document.page_count,
    }

    for page_index in range(document.page_count):
        page = document[page_index]
        page_dict = page.get_text("dict", sort=True)

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for raw_line in block.get("lines", []):
                spans = raw_line.get("spans", [])

                if not spans:
                    continue

                text_parts: list[str] = []
                font_sizes: list[float] = []
                font_names: list[str] = []

                for span in spans:
                    span_text = clean_text(
                        span.get("text", "")
                    )

                    if not span_text:
                        continue

                    text_parts.append(span_text)
                    font_sizes.append(
                        float(span.get("size", 0))
                    )
                    font_names.append(
                        span.get("font", "")
                    )

                line_text = clean_text(
                    " ".join(text_parts)
                )

                if not line_text:
                    continue

                if PAGE_NUMBER_PATTERN.fullmatch(line_text):
                    continue

                average_font_size = (
                    sum(font_sizes) / len(font_sizes)
                    if font_sizes
                    else 0
                )

                main_font = (
                    max(
                        set(font_names),
                        key=font_names.count,
                    )
                    if font_names
                    else ""
                )

                bbox = tuple(
                    raw_line.get(
                        "bbox",
                        block.get("bbox", (0, 0, 0, 0)),
                    )
                )

                lines.append(
                    PDFLine(
                        text=line_text,
                        page_number=page_index + 1,
                        font_size=round(
                            average_font_size,
                            2,
                        ),
                        font_name=main_font,
                        is_bold=any(
                            is_bold_font(font)
                            for font in font_names
                        ),
                        bbox=bbox,
                    )
                )

    document.close()

    return lines, metadata


# ============================================================
# Heading detection
# ============================================================

def parse_heading(
    line: PDFLine,
) -> tuple[str, str, int] | None:
    """
    Detect headings such as:

    1 - INTRODUCTION
    1.1 - WELCOME NEW EMPLOYEES
    4.3 - PAID TIME OFF (PTO)
    9.10 - SOCIAL MEDIA
    """
    text = clean_text(line.text)

    match = HEADING_PATTERN.match(text)

    if not match:
        return None

    number = match.group("number")
    title = normalize_heading_title(
        match.group("title")
    )

    if not title:
        return None

    # Avoid treating numbered list items as headings.
    # Policy headings are normally bold, uppercase, or short.
    words = title.split()

    mostly_uppercase = (
        sum(
            1
            for character in title
            if character.isalpha()
            and character.isupper()
        )
        >= max(
            1,
            int(
                sum(
                    1
                    for character in title
                    if character.isalpha()
                )
                * 0.6
            ),
        )
    )

    heading_like = (
        line.is_bold
        or mostly_uppercase
        or line.font_size >= 10
    )

    if not heading_like:
        return None

    if len(words) > 18:
        return None

    level = section_level(number)

    return number, title, level


# ============================================================
# Remove table of contents
# ============================================================

def find_content_start_page(
    lines: list[PDFLine],
) -> int:
    """
    Find the real first section instead of the table of contents.

    In this handbook, the TOC is on pages 2–3 and the real
    '1 - INTRODUCTION' begins on page 4.
    """
    candidates: list[int] = []

    for line in lines:
        normalized = line.text.upper().strip()

        if normalized in {
            "1 - INTRODUCTION",
            "1. INTRODUCTION",
        }:
            candidates.append(line.page_number)

    if not candidates:
        return 1

    # The last occurrence is normally the real section,
    # after its occurrence in the table of contents.
    return max(candidates)


# ============================================================
# Section construction
# ============================================================

def build_policy_sections(
    lines: list[PDFLine],
    document_id: str,
) -> list[PolicySection]:
    content_start_page = find_content_start_page(lines)

    content_lines = [
        line
        for line in lines
        if line.page_number >= content_start_page
    ]

    sections: list[PolicySection] = []
    section_stack: list[dict[str, Any]] = []

    current_heading: dict[str, Any] | None = None
    current_content: list[PDFLine] = []

    def finish_current_section() -> None:
        nonlocal current_heading
        nonlocal current_content

        if current_heading is None:
            return

        section_text = merge_lines(current_content)

        page_numbers = [
            line.page_number
            for line in current_content
        ]

        page_start = current_heading["page_number"]
        page_end = (
            max(page_numbers)
            if page_numbers
            else page_start
        )

        sections.append(
            PolicySection(
                section_id=current_heading["section_id"],
                document_id=document_id,
                section_number=current_heading[
                    "section_number"
                ],
                title=current_heading["title"],
                level=current_heading["level"],
                parent_section_id=current_heading[
                    "parent_section_id"
                ],
                heading_path=current_heading[
                    "heading_path"
                ],
                page_start=page_start,
                page_end=page_end,
                text=section_text,
            )
        )

        current_content = []

    for line in content_lines:
        parsed_heading = parse_heading(line)

        if parsed_heading is None:
            if current_heading is not None:
                current_content.append(line)
            continue

        finish_current_section()

        number, title, level = parsed_heading

        while (
            section_stack
            and section_stack[-1]["level"] >= level
        ):
            section_stack.pop()

        parent = (
            section_stack[-1]
            if section_stack
            else None
        )

        heading_label = f"{number} - {title}"

        heading_path = (
            [
                *parent["heading_path"],
                heading_label,
            ]
            if parent
            else [heading_label]
        )

        section_id = stable_id(
            "section",
            document_id,
            number,
            title,
        )

        current_heading = {
            "section_id": section_id,
            "section_number": number,
            "title": title,
            "level": level,
            "parent_section_id": (
                parent["section_id"]
                if parent
                else None
            ),
            "heading_path": heading_path,
            "page_number": line.page_number,
        }

        section_stack.append(current_heading)

    finish_current_section()

    return sections


# ============================================================
# Paragraph reconstruction
# ============================================================

def merge_lines(lines: list[PDFLine]) -> str:
    """
    Reconstruct paragraphs from individual PDF lines.
    """
    if not lines:
        return ""

    paragraphs: list[str] = []
    current_paragraph: list[str] = []
    previous_line: PDFLine | None = None

    for line in lines:
        text = clean_text(line.text)

        if not text:
            continue

        is_bullet = bool(
            re.match(
                r"^(?:[-•▪●]|\([a-zA-Z0-9]+\))\s+",
                text,
            )
        )

        if previous_line is not None:
            vertical_gap = (
                line.bbox[1] - previous_line.bbox[3]
            )

            new_paragraph = (
                vertical_gap > max(
                    4,
                    previous_line.font_size * 0.65,
                )
                or is_bullet
            )

            if new_paragraph and current_paragraph:
                paragraphs.append(
                    join_paragraph_lines(
                        current_paragraph
                    )
                )
                current_paragraph = []

        current_paragraph.append(text)
        previous_line = line

    if current_paragraph:
        paragraphs.append(
            join_paragraph_lines(current_paragraph)
        )

    return "\n\n".join(paragraphs)


def join_paragraph_lines(lines: list[str]) -> str:
    result = ""

    for line in lines:
        if not result:
            result = line
            continue

        if result.endswith("-"):
            result = result[:-1] + line
        else:
            result += " " + line

    return clean_text(result)


# ============================================================
# Chunking
# ============================================================

def split_words_with_overlap(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    words = text.split()

    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(
            start + chunk_size,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def create_chunks(
    sections: list[PolicySection],
    document_title: str,
    source_file: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []

    for section in sections:
        if not section.text.strip():
            continue

        section_chunks = split_words_with_overlap(
            text=section.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_index, raw_text in enumerate(
            section_chunks
        ):
            contextual_text = (
                f"Document: {document_title}\n"
                f"Section: "
                f"{' > '.join(section.heading_path)}\n"
                f"Pages: {section.page_start}-"
                f"{section.page_end}\n\n"
                f"{raw_text}"
            )

            chunk_id = stable_id(
                "chunk",
                section.section_id,
                str(chunk_index),
                raw_text,
            )

            chunks.append(
                PolicyChunk(
                    chunk_id=chunk_id,
                    document_id=section.document_id,
                    section_id=section.section_id,
                    section_number=section.section_number,
                    section_title=section.title,
                    heading_path=section.heading_path,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    chunk_index=chunk_index,
                    text=contextual_text,
                    raw_text=raw_text,
                    word_count=len(
                        contextual_text.split()
                    ),
                    metadata={
                        "source_file": source_file,
                        "citation": (
                            f"{document_title}, "
                            f"Section "
                            f"{section.section_number} - "
                            f"{section.title}, "
                            f"pages "
                            f"{section.page_start}-"
                            f"{section.page_end}"
                        ),
                    },
                )
            )

    return chunks


# ============================================================
# Complete processing function
# ============================================================

def process_pdf(
    pdf_path: Path,
    output_directory: Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> dict[str, Any]:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    lines, pdf_metadata = extract_pdf_lines(
        pdf_path
    )

    if not lines:
        raise ValueError(
            "No selectable text was extracted. "
            "The PDF may require OCR."
        )

    document_title = (
        clean_text(pdf_metadata["title"])
        or pdf_path.stem
    )

    document_id = stable_id(
        "document",
        pdf_path.name,
        document_title,
    )

    sections = build_policy_sections(
        lines=lines,
        document_id=document_id,
    )

    chunks = create_chunks(
        sections=sections,
        document_title=document_title,
        source_file=pdf_path.name,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    result = {
        "document": {
            "document_id": document_id,
            "title": document_title,
            "source_file": pdf_path.name,
            "page_count": pdf_metadata[
                "page_count"
            ],
            "content_start_page": (
                find_content_start_page(lines)
            ),
            "section_count": len(sections),
            "chunk_count": len(chunks),
            "metadata": pdf_metadata,
        },
        "sections": [
            asdict(section)
            for section in sections
        ],
        "chunks": [
            asdict(chunk)
            for chunk in chunks
        ],
    }

    if output_directory is not None:
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_directory
            / f"{pdf_path.stem}_processed.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=False,
            )

        result["output_path"] = str(
            output_path.resolve()
        )

    return result