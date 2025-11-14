"""
Update README.md with Markdown tables summarizing torus convexity batches.

This script parses the CSV outputs produced by run_convexity_batch.py and
inserts formatted tables between the markers:
    <!-- BEGIN TORUS RESULTS -->
    <!-- END TORUS RESULTS -->
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent
README_PATH = REPO_ROOT / "README.md"

SECTION_START = "<!-- BEGIN TORUS RESULTS -->"
SECTION_END = "<!-- END TORUS RESULTS -->"


@dataclass(frozen=True)
class Dataset:
    label: str
    csv_path: Path


DATASETS: Sequence[Dataset] = [
    Dataset("Elongation 1×", REPO_ROOT / "elongated_torus_convexity_results_elon1.csv"),
    Dataset("Elongation 2×", REPO_ROOT / "elongated_torus_convexity_results_elon2.csv"),
    Dataset("Elongation 3×", REPO_ROOT / "elongated_torus_convexity_results_elon3.csv"),
    Dataset("Elongation 4×", REPO_ROOT / "elongated_torus_convexity_results_elon4.csv"),
]

COLUMN_HEADERS = [
    "Aspect Ratio",
    "Surface Area",
    "⟨AO⟩",
    "C_M",
    "S·C_M/4",
    "MC Shadow",
    "MC/(S·C_M/4)",
    "MC/(S/4)",
]


def read_rows(csv_path: Path) -> List[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row.get("mesh")]


def parse_aspect_ratio(mesh_name: str) -> str:
    core = mesh_name.replace("torus_aspect_", "").replace(".obj", "")
    return core.replace("_", ".")


def fmt(value: str | float) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def build_table(title: str, rows: Iterable[dict[str, str]]) -> str:
    lines = [f"### {title}", "", "| " + " | ".join(COLUMN_HEADERS) + " |", "| " + " | ".join(["---"] * len(COLUMN_HEADERS)) + " |"]
    for row in rows:
        cells = [
            parse_aspect_ratio(row["mesh"]),
            fmt(row["surface_area"]),
            fmt(row["mean_ao"]),
            fmt(row["moeini_convexity"]),
            fmt(row["expected_shadow"]),
            fmt(row["monte_carlo_shadow"]),
            fmt(row["ratio_vs_expected"]),
            fmt(row["ratio_vs_cauchy"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def build_all_tables() -> str:
    sections = []
    for dataset in DATASETS:
        rows = read_rows(dataset.csv_path)
        sections.append(build_table(dataset.label, rows))
    return "\n".join(sections).strip() + "\n"


def update_readme(readme_path: Path, new_content: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if SECTION_START not in text or SECTION_END not in text:
        raise ValueError("README markers not found.")
    start_idx = text.index(SECTION_START) + len(SECTION_START)
    end_idx = text.index(SECTION_END)
    updated = text[:start_idx] + "\n" + new_content + text[end_idx:]
    readme_path.write_text(updated, encoding="utf-8")


def main() -> None:
    tables = build_all_tables()
    update_readme(README_PATH, tables)
    print("README updated with torus benchmark tables.")


if __name__ == "__main__":
    main()

