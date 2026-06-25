from __future__ import annotations

import pytest

from scripts.analysis.eval_metrics import (
    evaluate_record,
    exact_match,
    partial_match_accuracy,
    rouge_l,
    rouge_n,
)


def test_rouge_1_calculation() -> None:
    score = rouge_n("the cat sat", "the cat")
    assert score["precision"] == pytest.approx(2 / 3)
    assert score["recall"] == 1.0
    assert score["f1"] == pytest.approx(0.8)


def test_rouge_l_calculation() -> None:
    score = rouge_l("a b c d", "a c d")
    assert score["precision"] == 0.75
    assert score["recall"] == 1.0
    assert score["f1"] == pytest.approx(6 / 7)


def test_exact_match_is_case_and_whitespace_normalized() -> None:
    assert exact_match("  Answer   Forty-Two ", "answer forty-two") == 1.0
    assert exact_match("42", "forty-two") == 0.0


def test_partial_match_uses_unique_words_and_removes_nltk_stopwords() -> None:
    score = partial_match_accuracy("The red train, and the red bus.", "A red train of Cambridge.")
    assert score == pytest.approx(2 / 4)


def test_partial_match_returns_zero_for_only_stopwords_or_no_overlap() -> None:
    assert partial_match_accuracy("the and of", "of the and") == 0.0
    assert partial_match_accuracy("alpha", "beta") == 0.0


def test_multiwoz_exact_match_is_disabled_by_default() -> None:
    result = evaluate_record(
        {
            "dataset": "MultiWOZ",
            "final_response": "correct",
            "ground_truth": "correct",
        }
    )
    assert result["metrics"]["exact_match"] is None
    assert result["metrics"]["exact_match_enabled"] is False


def test_multiwoz_dialogue_id_disables_exact_match_without_dataset_field() -> None:
    result = evaluate_record(
        {
            "dialogue_id": "PMUL2597.json",
            "final_response": "correct",
            "ground_truth": "correct",
        }
    )
    assert result["metrics"]["exact_match"] is None


def test_multiwoz_exact_match_can_be_enabled() -> None:
    result = evaluate_record(
        {
            "dataset": "MultiWOZ",
            "final_response": "correct",
            "ground_truth": "correct",
        },
        enable_multiwoz_exact_match=True,
    )
    assert result["metrics"]["exact_match"] == 1.0


def test_empty_or_malformed_final_response_scores_zero() -> None:
    for response in (None, "", {"unexpected": "object"}):
        result = evaluate_record({"final_response": response, "ground_truth": "expected answer"})
        assert result["metrics"]["rouge_1"]["f1"] == 0.0
        assert result["metrics"]["rouge_l"]["f1"] == 0.0
        assert result["metrics"]["exact_match"] == 0.0
        assert result["metrics"]["partial_match_jaccard"] == 0.0
