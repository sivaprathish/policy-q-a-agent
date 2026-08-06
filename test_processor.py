"""
Self-contained MOCK demo of the RAG latency/token benchmark.

This does NOT hit a real vector store or LLM. It simulates a
"Flat RAG + LLM" run and a "Hierarchical RAG + LLM" run with
fabricated (but internally consistent) retrieval/answer text and
timing, then reuses the same reporting logic/format as the real
benchmark script to print the summary table + comparison.

Run:
    python benchmark_demo.py
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass


# ============================================================
# Benchmark models (same shape as the real script)
# ============================================================

@dataclass
class LatencyMeasurement:
    question: str
    latency_ms: float
    result_count: int
    input_tokens: int = 0
    context_tokens: int = 0
    output_tokens: int = 0
    input_words: int = 0
    context_words: int = 0
    output_words: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.context_tokens + self.output_tokens

    @property
    def total_words(self) -> int:
        return self.input_words + self.context_words + self.output_words


@dataclass
class BenchmarkSummary:
    name: str
    measurements: list[LatencyMeasurement]

    @property
    def mean_ms(self) -> float:
        return statistics.mean(m.latency_ms for m in self.measurements)


# ============================================================
# Fake retrieval + fake LLM
# ============================================================

QUESTION = "How many PTO days can an employee carry over?"


# Fixed targets for the demo. In the real benchmark these numbers
# come from actually counting the retrieved context + LLM answer;
# here they're pinned directly so the printed table is reproducible.
FLAT_TARGETS = dict(
    input_tokens=12, context_tokens=842, output_tokens=167,
    input_words=9, context_words=600, output_words=157,
    latency_ms=1988.14,
)
HIERARCHICAL_TARGETS = dict(
    input_tokens=12, context_tokens=421, output_tokens=170,
    input_words=9, context_words=300, output_words=143,
    latency_ms= 2150.31,
)


def simulate_flat_rag(question: str) -> dict:
    """Flat RAG: searches all chunks directly -> larger, noisier context."""
    time.sleep(0.05)  # pretend retrieval work
    time.sleep(0.05)  # pretend LLM call
    return FLAT_TARGETS


def simulate_hierarchical_rag(question: str) -> dict:
    """Hierarchical RAG: narrows document -> section -> chunk first."""
    time.sleep(0.02)
    time.sleep(0.03)
    return HIERARCHICAL_TARGETS


# ============================================================
# Run one "end-to-end" measurement per retriever
# ============================================================

def run_demo(name: str, simulate_fn, question: str) -> BenchmarkSummary:
    targets = simulate_fn(question)

    measurement = LatencyMeasurement(
        question=question,
        latency_ms=targets["latency_ms"],
        result_count=5,
        input_tokens=targets["input_tokens"],
        context_tokens=targets["context_tokens"],
        output_tokens=targets["output_tokens"],
        input_words=targets["input_words"],
        context_words=targets["context_words"],
        output_words=targets["output_words"],
    )

    return BenchmarkSummary(name=name, measurements=[measurement])


# ============================================================
# Reporting (mirrors print_end_to_end_summary from benchmark.py)
# ============================================================

# ============================================================
# Reporting — boxed table + comparison panel
# ============================================================

COLUMNS = [
    ("Retriever", 24, "<"),
    ("Input Tok", 10, ">"),
    ("Context Tok", 12, ">"),
    ("Output Tok", 11, ">"),
    ("Total Tok", 10, ">"),
    ("Total Words", 12, ">"),
    ("Latency (ms)", 13, ">"),
]


def _row(values: list[str]) -> str:
    cells = [f"{val:{align}{width}}" for (_, width, align), val in zip(COLUMNS, values)]
    return "│ " + " │ ".join(cells) + " │"


def _rule(char: str = "─", junction: str = "┼") -> str:
    segments = [char * (width + 2) for _, width, _ in COLUMNS]
    return "├" + junction.join(segments) + "┤" if junction else "─" * len(segments)


def _top_rule() -> str:
    segments = ["─" * (width + 2) for _, width, _ in COLUMNS]
    return "┌" + "┬".join(segments) + "┐"


def _bottom_rule() -> str:
    segments = ["─" * (width + 2) for _, width, _ in COLUMNS]
    return "└" + "┴".join(segments) + "┘"


def print_end_to_end_summary(summaries: list[BenchmarkSummary]) -> None:
    title = "END-TO-END TOKEN, WORD, AND LATENCY SUMMARY"
    width = sum(w + 3 for _, w, _ in COLUMNS) + 1

    print()
    print(title.center(width))
    print()
    print(_top_rule())
    print(_row([name for name, _, _ in COLUMNS]))
    print(_rule())

    stats_by_name: dict[str, dict[str, float]] = {}

    for summary in summaries:
        m = summary.measurements
        stats = {
            "input_tokens": statistics.mean(x.input_tokens for x in m),
            "context_tokens": statistics.mean(x.context_tokens for x in m),
            "output_tokens": statistics.mean(x.output_tokens for x in m),
            "total_tokens": statistics.mean(x.total_tokens for x in m),
            "total_words": statistics.mean(x.total_words for x in m),
            "latency_ms": summary.mean_ms,
        }
        stats_by_name[summary.name] = stats

        print(_row([
            summary.name,
            f"{stats['input_tokens']:.0f}",
            f"{stats['context_tokens']:.0f}",
            f"{stats['output_tokens']:.0f}",
            f"{stats['total_tokens']:.0f}",
            f"{stats['total_words']:.0f}",
            f"{stats['latency_ms']:.2f}",
        ]))

    print(_bottom_rule())

    if len(summaries) < 2:
        return

    flat_name, hier_name = summaries[0].name, summaries[1].name
    flat, hier = stats_by_name[flat_name], stats_by_name[hier_name]

    token_diff = flat["total_tokens"] - hier["total_tokens"]
    word_diff = flat["total_words"] - hier["total_words"]
    latency_diff = flat["latency_ms"] - hier["latency_ms"]

    token_pct = (token_diff / flat["total_tokens"] * 100) if flat["total_tokens"] else 0.0
    word_pct = (word_diff / flat["total_words"] * 100) if flat["total_words"] else 0.0
    latency_pct = (latency_diff / flat["latency_ms"] * 100) if flat["latency_ms"] else 0.0

    latency_verb = "faster" if latency_diff >= 0 else "slower"
    latency_val = abs(latency_diff)
    latency_pct_val = abs(latency_pct)

    lines = [
        f"Tokens:   {token_diff:.0f} fewer  ({token_pct:.1f}% reduction)",
        f"Words:    {word_diff:.0f} fewer  ({word_pct:.1f}% reduction)",
        f"Latency:  {latency_val:.2f} ms {latency_verb}  ({latency_pct_val:.1f}%)",
    ]

    panel_width = max(len(f" {hier_name} vs. {flat_name} "), *(len(f" {l} ") for l in lines))

    print()
    print("┌" + "─" * panel_width + "┐")
    print("│" + f" {hier_name} vs. {flat_name}".ljust(panel_width) + "│")
    print("├" + "─" * panel_width + "┤")
    for line in lines:
        print("│" + f" {line}".ljust(panel_width) + "│")
    print("└" + "─" * panel_width + "┘")


# ============================================================
# Main
# ============================================================

def main() -> None:
    flat_summary = run_demo("Flat RAG + LLM", simulate_flat_rag, QUESTION)
    hierarchical_summary = run_demo(
        "Hierarchical RAG + LLM", simulate_hierarchical_rag, QUESTION
    )

    print_end_to_end_summary([flat_summary, hierarchical_summary])


if __name__ == "__main__":
    main()