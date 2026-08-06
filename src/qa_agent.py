from __future__ import annotations

import requests

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)

from src.retriever import RetrievalResult


SYSTEM_PROMPT = """
You are a company policy Q&A assistant.

Follow these rules:

1. Answer only from the provided policy evidence.
2. Do not invent policies, limits, dates, approvals or exceptions.
3. If the evidence is insufficient, clearly say so.
4. Include the policy section or citation supporting the answer.
5. Mention important eligibility conditions, limits and exceptions.
6. Keep the answer practical and easy to understand.
""".strip()


def build_user_prompt(
    question: str,
    retrieval: RetrievalResult,
) -> str:
    return f"""
Employee question:

{question}

Retrieved policy evidence:

{retrieval.context}

Return the answer in this format:

Answer:
<direct answer>

Conditions or exceptions:
<important conditions, approvals or exceptions>

Sources:
<supporting policy citations>
""".strip()


def generate_answer(
    question: str,
    retrieval: RetrievalResult,
) -> str:
    """
    Generate an evidence-grounded answer.
    """
    if not retrieval.chunks:
        return (
            "I could not find sufficient policy evidence "
            "to answer this question."
        )

    if not LLM_API_KEY:
        return build_evidence_only_response(
            question=question,
            retrieval=retrieval,
        )

    url = (
        f"{LLM_BASE_URL.rstrip('/')}"
        f"/chat/completions"
    )

    response = requests.post(
        url,
        headers={
            "Authorization": (
                f"Bearer {LLM_API_KEY}"
            ),
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "temperature": 0.1,
            "max_tokens": 900,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_user_prompt(
                        question=question,
                        retrieval=retrieval,
                    ),
                },
            ],
        },
        timeout=90,
    )

    response.raise_for_status()

    payload = response.json()

    return payload[
        "choices"
    ][0][
        "message"
    ][
        "content"
    ].strip()


def build_evidence_only_response(
    question: str,
    retrieval: RetrievalResult,
) -> str:
    """
    Show retrieval results when no LLM key is configured.
    """
    response_parts = [
        "No LLM API key is configured.",
        "",
        f"Question: {question}",
        "",
        "Most relevant policy evidence:",
        "",
    ]

    for index, result in enumerate(
        retrieval.chunks[:4],
        start=1,
    ):
        record = result.record

        citation = record.get(
            "metadata",
            {},
        ).get(
            "citation",
            "Policy evidence",
        )

        text = record.get(
            "raw_text",
            record.get("text", ""),
        )

        response_parts.extend(
            [
                f"Evidence {index}",
                f"Source: {citation}",
                text,
                "",
            ]
        )

    return "\n".join(
        response_parts
    ).strip()