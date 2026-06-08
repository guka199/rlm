from __future__ import annotations

import json

from scripts.analysis.extract_responses import (
    extract_responses_from_record,
    extract_trace_paths,
    read_trace_records,
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_extracts_common_response_keys_from_valid_json(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "response": "direct",
                "assistant_response": "assistant",
                "model_response": "model",
                "message": {"content": "message"},
                "choices": [{"message": {"content": "choice"}}],
            }
        ),
        encoding="utf-8",
    )

    extracted_path, invalid_path = extract_trace_paths([trace_path], tmp_path / "outputs")

    rows = read_jsonl(extracted_path)
    assert invalid_path.read_text(encoding="utf-8") == ""
    assert {row["response"] for row in rows} == {
        "assistant",
        "choice",
        "direct",
        "message",
        "model",
    }


def test_extracts_jsonl_records(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"output": "first"}),
                json.dumps({"content": "second"}),
            ]
        ),
        encoding="utf-8",
    )

    extracted_path, invalid_path = extract_trace_paths([trace_path], tmp_path / "outputs")

    assert [row["response"] for row in read_jsonl(extracted_path)] == ["first", "second"]
    assert invalid_path.read_text(encoding="utf-8") == ""


def test_extracts_jsonl_records_with_utf8_bom(tmp_path):
    trace_path = tmp_path / "trace_bom.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"output": "first"}),
                json.dumps({"content": "second"}),
            ]
        ),
        encoding="utf-8-sig",
    )

    records, invalid_records = read_trace_records(trace_path)

    assert records == [{"output": "first"}, {"content": "second"}]
    assert invalid_records == []


def test_extracts_new_rlm_trace_jsonl_and_skips_metadata(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "metadata", "run_id": "abc"}),
                json.dumps(
                    {
                        "type": "iteration",
                        "iteration": 1,
                        "timestamp": "2026-06-02T00:00:00Z",
                        "prompt": [{"role": "user", "content": "do not extract prompt"}],
                        "response": "The next step is to examine the `context` variable...",
                        "code_blocks": [],
                        "final_answer": None,
                        "iteration_time": 3.97,
                    }
                ),
                json.dumps({"assistant_response": "old assistant"}),
                json.dumps({"choices": [{"message": {"content": "old choice"}}]}),
            ]
        ),
        encoding="utf-8-sig",
    )

    extracted_path, invalid_path = extract_trace_paths([trace_path], tmp_path / "outputs")

    rows = read_jsonl(extracted_path)
    assert invalid_path.read_text(encoding="utf-8") == ""
    assert [
        {
            "response_key": row["response_key"],
            "response": row["response"],
            **({"step_idx": row["step_idx"]} if "step_idx" in row else {}),
        }
        for row in rows
    ] == [
        {
            "response_key": "response",
            "response": "The next step is to examine the `context` variable...",
            "step_idx": 1,
        },
        {"response_key": "assistant_response", "response": "old assistant"},
        {"response_key": "choices[0].message.content", "response": "old choice"},
    ]


def test_extracts_iteration_final_answer(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "type": "iteration",
                "iteration": 2,
                "response": "FINAL_VAR(answer)",
                "final_answer": "resolved answer",
            }
        ),
        encoding="utf-8",
    )

    extracted_path, invalid_path = extract_trace_paths([trace_path], tmp_path / "outputs")

    rows = read_jsonl(extracted_path)
    assert invalid_path.read_text(encoding="utf-8") == ""
    assert [(row["response_key"], row["response"], row["step_idx"]) for row in rows] == [
        ("response", "FINAL_VAR(answer)", 2),
        ("final_answer", "resolved answer", 2),
    ]


def test_extracts_response_values_from_fenced_json_string(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "response": """```json
{
  "assistant_response": "nested assistant",
  "choices": [
    {"message": {"content": "nested choice"}}
  ]
}
```"""
            }
        ),
        encoding="utf-8",
    )

    extracted_path, invalid_path = extract_trace_paths([trace_path], tmp_path / "outputs")

    rows = read_jsonl(extracted_path)
    assert invalid_path.read_text(encoding="utf-8") == ""
    assert [(row["response_key"], row["response"]) for row in rows] == [
        (
            "response",
            """```json
{
  "assistant_response": "nested assistant",
  "choices": [
    {"message": {"content": "nested choice"}}
  ]
}
```""",
        ),
        ("response.json.choices[0].message.content", "nested choice"),
        ("response.json.assistant_response", "nested assistant"),
    ]


def test_invalid_json_is_reported_without_crashing(tmp_path):
    trace_path = tmp_path / "broken.json"
    trace_path.write_text("{bad json", encoding="utf-8")

    records, invalid_records = read_trace_records(trace_path)
    extracted_path, invalid_path = extract_trace_paths([trace_path], tmp_path / "outputs")

    assert records == []
    assert invalid_records[0]["source_file"] == str(trace_path)
    assert "error" in invalid_records[0]
    assert extracted_path.read_text(encoding="utf-8") == ""
    assert read_jsonl(invalid_path)[0]["source_file"] == str(trace_path)


def test_extract_responses_from_nested_record():
    rows = extract_responses_from_record({"events": [{"message": {"content": "nested"}}]})

    assert rows == [{"response_key": "events.[0].message.content", "response": "nested"}]
