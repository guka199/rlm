from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FAILURE_SPLIT_PATTERN = re.compile(r"[\n;,|]+")
PROCESS_STAGE_ALIASES = {"process/stage", "process", "stage", "step"}
FAILURE_MODE_ALIASES = {"failure modes", "failure mode", "failure_modes", "failure_mode", "failure"}


class FrameValues(list[list[str]]):
    def tolist(self) -> list[list[str]]:
        return list(self)


@dataclass
class LabelFrame:
    columns: list[str]
    rows: list[list[str]]

    def to_dict(self, orient: str) -> list[dict[str, str]]:
        if orient != "records":
            raise ValueError("only records orientation is supported")
        return [dict(zip(self.columns, row, strict=False)) for row in self.rows]

    def astype(self, _type: type[str]) -> LabelFrame:
        return self

    @property
    def values(self) -> FrameValues:
        return FrameValues(self.rows)


def normalize_label_file(
    path: Path,
    *,
    sheet_name: str | int | None = None,
    dataset: str | None = None,
    config: str | None = None,
) -> list[dict[str, str]]:
    frame = read_label_frame(path, sheet_name=sheet_name)
    direct_rows = normalize_direct_columns(frame, path=path, dataset=dataset, config=config)
    if direct_rows:
        return direct_rows
    return normalize_alternating_rows(frame, path=path, dataset=dataset, config=config)


def read_label_frame(path: Path, *, sheet_name: str | int | None) -> Any:
    if path.suffix.lower() == ".csv":
        return read_csv_label_frame(path)

    import pandas as pd

    effective_sheet = 0 if sheet_name is None else sheet_name
    return pd.read_excel(path, sheet_name=effective_sheet, dtype=str).fillna("")


def read_csv_label_frame(path: Path) -> LabelFrame:
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.reader(input_file))
    if not rows:
        return LabelFrame([], [])
    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    return LabelFrame(padded_rows[0], padded_rows[1:])


def normalize_direct_columns(
    frame: Any,
    *,
    path: Path,
    dataset: str | None,
    config: str | None,
) -> list[dict[str, str]]:
    step_column = find_column(frame.columns, PROCESS_STAGE_ALIASES)
    failure_column = find_column(frame.columns, FAILURE_MODE_ALIASES)
    if step_column is None or failure_column is None:
        return []

    rows: list[dict[str, str]] = []
    for row_number, row in enumerate(frame.to_dict("records"), start=2):
        step = clean_cell(row.get(step_column))
        labels = split_failure_modes(row.get(failure_column))
        for label in labels:
            rows.append(normalized_row(path, row_number, step, label, dataset, config))
    return rows


def normalize_alternating_rows(
    frame: Any,
    *,
    path: Path,
    dataset: str | None,
    config: str | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current_step = ""

    raw_rows = [list(frame.columns), *frame.astype(str).values.tolist()]
    for row_number, values in enumerate(raw_rows, start=1):
        cells = [clean_cell(cell) for cell in values if clean_cell(cell)]
        if not cells:
            continue

        first_cell = normalize_heading(cells[0])
        if first_cell in PROCESS_STAGE_ALIASES:
            current_step = clean_cell(" ".join(cells[1:]))
            continue

        if first_cell in FAILURE_MODE_ALIASES:
            for label in split_failure_modes(" ".join(cells[1:])):
                rows.append(normalized_row(path, row_number, current_step, label, dataset, config))
            continue

        step = infer_step_from_row(cells)
        if step:
            current_step = step
            continue

        for label in split_failure_modes(" ".join(cells)):
            rows.append(normalized_row(path, row_number, current_step, label, dataset, config))

    return rows


def find_column(columns: Iterable[Any], aliases: set[str]) -> str | None:
    for column in columns:
        column_name = str(column)
        if normalize_heading(column_name) in aliases:
            return column_name
    return None


def infer_step_from_row(cells: list[str]) -> str:
    if len(cells) == 1 and normalize_heading(cells[0]) not in FAILURE_MODE_ALIASES:
        return cells[0]
    return ""


def split_failure_modes(value: Any) -> list[str]:
    cell = clean_cell(value)
    if not cell:
        return []

    labels: list[str] = []
    seen: set[str] = set()
    for label in FAILURE_SPLIT_PATTERN.split(cell):
        cleaned_label = clean_cell(label)
        normalized_label = normalize_label(cleaned_label)
        if cleaned_label and normalized_label not in seen:
            labels.append(cleaned_label)
            seen.add(normalized_label)
    return labels


def normalized_row(
    path: Path,
    row_number: int,
    step: str,
    failure_mode: str,
    dataset: str | None,
    config: str | None,
) -> dict[str, str]:
    return {
        "dataset": dataset or "",
        "config": config or "",
        "source_file": str(path),
        "row_number": str(row_number),
        "step": step,
        "failure_mode": failure_mode,
    }


def failure_counts_by_step(rows: Iterable[dict[str, str]]) -> list[dict[str, str | int]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        counts[(row["step"], row["failure_mode"])] += 1
    return [
        {"step": step, "failure_mode": failure_mode, "count": count}
        for (step, failure_mode), count in sorted(counts.items())
    ]


def aggregate_failure_counts(rows: Iterable[dict[str, str]]) -> list[dict[str, str | int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[row["failure_mode"]] += 1
    return [
        {"failure_mode": failure_mode, "count": count}
        for failure_mode, count in sorted(counts.items())
    ]


def write_failure_metrics(
    input_paths: Iterable[Path],
    *,
    output_dir: Path,
    sheet_name: str | int | None = None,
    dataset: str | None = None,
    config: str | None = None,
) -> tuple[Path, Path, Path]:
    rows: list[dict[str, str]] = []
    for path in input_paths:
        rows.extend(
            normalize_label_file(path, sheet_name=sheet_name, dataset=dataset, config=config)
        )

    metrics_dir = output_dir / "failure_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = metrics_dir / "normalized_labels.csv"
    by_step_path = metrics_dir / "failure_counts_by_step.csv"
    aggregate_path = metrics_dir / "aggregate_failure_counts.csv"

    write_csv(
        normalized_path,
        rows,
        ["dataset", "config", "source_file", "row_number", "step", "failure_mode"],
    )
    write_csv(by_step_path, failure_counts_by_step(rows), ["step", "failure_mode", "count"])
    write_csv(aggregate_path, aggregate_failure_counts(rows), ["failure_mode", "count"])
    return normalized_path, by_step_path, aggregate_path


def write_csv(path: Path, rows: Iterable[dict[str, str | int]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize labeled failure sheets and compute metrics."
    )
    parser.add_argument(
        "input_paths", nargs="+", type=Path, help="Excel or CSV label sheets to read."
    )
    parser.add_argument(
        "--sheet-name", default=None, help="Excel sheet name or index. Defaults to first sheet."
    )
    parser.add_argument(
        "--dataset", default=None, help="Dataset label to include in normalized output."
    )
    parser.add_argument(
        "--config", default=None, help="Config label to include in normalized output."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Base directory for failure metric outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sheet_name: str | int | None = args.sheet_name
    if isinstance(sheet_name, str) and sheet_name.isdigit():
        sheet_name = int(sheet_name)
    write_failure_metrics(
        args.input_paths,
        output_dir=args.output_dir,
        sheet_name=sheet_name,
        dataset=args.dataset,
        config=args.config,
    )


if __name__ == "__main__":
    main()
