from __future__ import annotations

import csv

from scripts.analysis.failure_metrics import (
    aggregate_failure_counts,
    failure_counts_by_step,
    normalize_label_file,
    split_failure_modes,
    write_failure_metrics,
)


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerows(rows)


def read_csv_rows(path):
    with path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def test_blank_failure_cells_and_repeated_labels_are_normalized(tmp_path):
    label_path = tmp_path / "labels.csv"
    write_rows(
        label_path,
        [
            ["Process/Stage", "Failure modes"],
            ["Plan", "Bad prompt, Bad prompt; Tool error"],
            ["Plan", ""],
            ["Execute", "Bad prompt"],
        ],
    )

    rows = normalize_label_file(label_path, dataset="tau", config="baseline")

    assert [(row["step"], row["failure_mode"]) for row in rows] == [
        ("Plan", "Bad prompt"),
        ("Plan", "Tool error"),
        ("Execute", "Bad prompt"),
    ]
    assert {row["dataset"] for row in rows} == {"tau"}
    assert {row["config"] for row in rows} == {"baseline"}


def test_alternating_process_stage_and_failure_mode_rows(tmp_path):
    label_path = tmp_path / "alternating.csv"
    write_rows(
        label_path,
        [
            ["Process/Stage", "Plan"],
            ["Failure modes", "Bad prompt\nTool error"],
            ["Process/Stage", "Execute"],
            ["Failure modes", "Tool error"],
        ],
    )

    rows = normalize_label_file(label_path)

    assert [(row["step"], row["failure_mode"]) for row in rows] == [
        ("Plan", "Bad prompt"),
        ("Plan", "Tool error"),
        ("Execute", "Tool error"),
    ]


def test_failure_counts_by_step_and_aggregate_counts():
    rows = [
        {"step": "Plan", "failure_mode": "Bad prompt"},
        {"step": "Plan", "failure_mode": "Bad prompt"},
        {"step": "Execute", "failure_mode": "Tool error"},
    ]

    assert failure_counts_by_step(rows) == [
        {"step": "Execute", "failure_mode": "Tool error", "count": 1},
        {"step": "Plan", "failure_mode": "Bad prompt", "count": 2},
    ]
    assert aggregate_failure_counts(rows) == [
        {"failure_mode": "Bad prompt", "count": 2},
        {"failure_mode": "Tool error", "count": 1},
    ]


def test_write_failure_metric_outputs(tmp_path):
    label_path = tmp_path / "labels.csv"
    write_rows(
        label_path,
        [
            ["Process/Stage", "Failure modes"],
            ["Plan", "Bad prompt"],
        ],
    )

    normalized_path, by_step_path, aggregate_path = write_failure_metrics(
        [label_path],
        output_dir=tmp_path / "outputs",
    )

    assert normalized_path.name == "normalized_labels.csv"
    assert by_step_path.name == "failure_counts_by_step.csv"
    assert aggregate_path.name == "aggregate_failure_counts.csv"
    assert read_csv_rows(normalized_path)[0]["failure_mode"] == "Bad prompt"
    assert read_csv_rows(by_step_path)[0]["count"] == "1"
    assert read_csv_rows(aggregate_path)[0]["count"] == "1"


def test_split_failure_modes_handles_blank_cells():
    assert split_failure_modes("") == []
    assert split_failure_modes(None) == []
