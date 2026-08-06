from pathlib import Path

from doc_process.pdf import process_pdf


PDF_PATH = Path(
    r"src/test/attachment-a-2023-handbook-revised-11-2023.pdf"
)

OUTPUT_DIRECTORY = Path("data/processed")


def main() -> None:
    result = process_pdf(
        pdf_path=PDF_PATH,
        output_directory=OUTPUT_DIRECTORY,
        chunk_size=220,
        overlap=40,
    )

    document = result["document"]

    print("=" * 80)
    print("DOCUMENT DETAILS")
    print("=" * 80)
    print("Title:", document["title"])
    print("Pages:", document["page_count"])
    print(
        "Content starts on page:",
        document["content_start_page"],
    )
    print(
        "Sections:",
        document["section_count"],
    )
    print(
        "Chunks:",
        document["chunk_count"],
    )

    print("\n" + "=" * 80)
    print("DETECTED HIERARCHY")
    print("=" * 80)

    for section in result["sections"]:
        indent = "    " * (
            section["level"] - 1
        )

        print(
            f"{indent}"
            f"{section['section_number']} - "
            f"{section['title']} "
            f"[pages "
            f"{section['page_start']}-"
            f"{section['page_end']}]"
        )

    print("\n" + "=" * 80)
    print("SAMPLE CHUNKS")
    print("=" * 80)

    for chunk in result["chunks"][:5]:
        print("\nChunk ID:", chunk["chunk_id"])
        print(
            "Section:",
            chunk["section_number"],
            "-",
            chunk["section_title"],
        )
        print(
            "Path:",
            " > ".join(
                chunk["heading_path"]
            ),
        )
        print(
            "Pages:",
            chunk["page_start"],
            "-",
            chunk["page_end"],
        )
        print("Text:")
        print(chunk["raw_text"][:500])
        print("-" * 80)

    print(
        "\nSaved to:",
        result.get("output_path"),
    )


if __name__ == "__main__":
    main()