from __future__ import annotations

import pytest

from scripts.analysis.failure_analysis import analyze_record


def test_failure_onset_duration_and_normalized_position() -> None:
    result = analyze_record(
        {
            "trace_id": "trace-1",
            "task_id": "task-1",
            "first_failure_onset": 3,
            "total_trace_steps": 10,
            "primary_label": "Looping",
        }
    )
    assert result["first_failure_onset"] == 3
    assert result["failure_duration"] == 8
    assert result["failure_persistence"] == 8
    assert result["failure_duration_percentage"] == pytest.approx(0.8)
    assert result["normalized_position_within_trace"] == pytest.approx(0.3)


def test_failure_duration_uses_annotated_end_step_inclusively() -> None:
    result = analyze_record(
        {
            "onset_step": "4",
            "end_step": "6",
            "total_steps": "10",
        }
    )
    assert result["failure_duration"] == 3
    assert result["failure_duration_percentage"] == pytest.approx(0.3)


def test_manual_duration_and_normalized_position_are_preserved() -> None:
    result = analyze_record(
        {
            "first_failure_onset": 2,
            "failure_duration": 4,
            "normalized_position_within_trace": 0.25,
            "failure_duration_percentage": 0.5,
            "total_trace_steps": 8,
            "human_revision": "Reviewed by annotator",
        }
    )
    assert result["failure_duration"] == 4
    assert result["normalized_position_within_trace"] == 0.25
    assert result["failure_duration_percentage"] == 0.5
    assert result["human_revision"] == "Reviewed by annotator"


def test_trace_length_can_be_inferred_from_iterations() -> None:
    result = analyze_record(
        {
            "first_failure_onset": 2,
            "iterations": [{"iteration": 1}, {"iteration": 2}, {"iteration": 3}],
        }
    )
    assert result["total_trace_steps"] == 3
    assert result["failure_duration"] == 2
    assert result["normalized_position_within_trace"] == pytest.approx(2 / 3)


def test_failure_onset_cannot_exceed_trace_length() -> None:
    with pytest.raises(ValueError, match="exceeds trace length"):
        analyze_record({"first_failure_onset": 5, "total_trace_steps": 4})
