from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DIRECT_RESPONSE_KEYS = {
    "assistant_response",
    "content",
    "final_answer",
    "model_response",
    "output",
    "response",
}


def read_trace_records(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []

    if path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_records.append(invalid_trace(path, str(exc), line_number=line_number))
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                invalid_records.append(
                    invalid_trace(
                        path, "trace record is not a JSON object", line_number=line_number
                    )
                )
        return records, invalid_records

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return [], [invalid_trace(path, str(exc))]

    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, dict):
                records.append(item)
            else:
                invalid_records.append(
                    invalid_trace(path, "trace record is not a JSON object", index=index)
                )
        return records, invalid_records

    if isinstance(value, dict):
        return [value], []

    return [], [invalid_trace(path, "trace file must contain a JSON object or list of objects")]


def invalid_trace(
    path: Path,
    error: str,
    *,
    line_number: int | None = None,
    index: int | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {"source_file": str(path), "error": error}
    if line_number is not None:
        report["line_number"] = line_number
    if index is not None:
        report["record_index"] = index
    return report


def extract_responses_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("type") == "metadata":
        return []
    if record.get("type") == "iteration":
        return extract_iteration_responses(record)

    responses: list[dict[str, Any]] = []
    collect_response_values(record, (), responses)
    return responses


def extract_iteration_responses(record: dict[str, Any]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    step_idx = record.get("iteration")
    for key in ("response", "final_answer"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        responses.append({"response_key": key, "response": value, "step_idx": step_idx})
        collect_json_response_values(value, (key, "json"), responses)
    return responses


def collect_response_values(
    value: Any,
    path: tuple[str, ...],
    responses: list[dict[str, Any]],
) -> None:
    if isinstance(value, dict):
        append_message_content(value, path, responses)
        append_choice_message_content(value, path, responses)

        for key, nested_value in value.items():
            if (
                key in DIRECT_RESPONSE_KEYS
                and isinstance(nested_value, str)
                and nested_value.strip()
            ):
                responses.append({"response_key": ".".join((*path, key)), "response": nested_value})
                collect_json_response_values(nested_value, (*path, key, "json"), responses)
            if key in {"choices", "message"}:
                continue
            collect_response_values(nested_value, (*path, key), responses)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            collect_response_values(item, (*path, f"[{index}]"), responses)


def collect_json_response_values(
    value: str,
    path: tuple[str, ...],
    responses: list[dict[str, Any]],
) -> None:
    parsed = parse_json_text(value)
    if isinstance(parsed, dict | list):
        collect_response_values(parsed, path, responses)


def parse_json_text(value: str) -> Any:
    text = strip_json_code_fence(value.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_fragment = extract_json_fragment(text)
    if json_fragment is None:
        return None
    try:
        return json.loads(json_fragment)
    except json.JSONDecodeError:
        return None


def strip_json_code_fence(value: str) -> str:
    lines = value.splitlines()
    if len(lines) < 2:
        return value
    opening = lines[0].strip().lower()
    closing = lines[-1].strip()
    if opening in {"```", "```json"} and closing == "```":
        return "\n".join(lines[1:-1]).strip()
    return value


def extract_json_fragment(value: str) -> str | None:
    for opener, closer in (("{", "}"), ("[", "]")):
        start = value.find(opener)
        end = value.rfind(closer)
        if start != -1 and end > start:
            return value[start : end + 1]
    return None


def append_message_content(
    value: dict[str, Any],
    path: tuple[str, ...],
    responses: list[dict[str, Any]],
) -> None:
    message = value.get("message")
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        responses.append(
            {"response_key": ".".join((*path, "message.content")), "response": content}
        )


def append_choice_message_content(
    value: dict[str, Any],
    path: tuple[str, ...],
    responses: list[dict[str, Any]],
) -> None:
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        responses.append(
            {"response_key": ".".join((*path, "choices[0].message.content")), "response": content}
        )


def extract_trace_paths(paths: Iterable[Path], output_dir: Path) -> tuple[Path, Path]:
    extracted_dir = output_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    extracted_path = extracted_dir / "extracted_responses.jsonl"
    invalid_path = extracted_dir / "invalid_traces.jsonl"

    with (
        extracted_path.open("w", encoding="utf-8") as extracted_file,
        invalid_path.open("w", encoding="utf-8") as invalid_file,
    ):
        for path in paths:
            records, invalid_records = read_trace_records(path)
            for invalid_record in invalid_records:
                invalid_file.write(json.dumps(invalid_record) + "\n")
            for record_index, record in enumerate(records):
                for response in extract_responses_from_record(record):
                    row = {
                        "source_file": str(path),
                        "record_index": record_index,
                        **response,
                    }
                    extracted_file.write(json.dumps(row) + "\n")

    return extracted_path, invalid_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract model responses from RLM trace JSON/JSONL files."
    )
    parser.add_argument(
        "input_paths", nargs="+", type=Path, help="Trace JSON or JSONL files to read."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Base directory for extracted outputs.",
    )
    parser.add_argument(
        "--dataset", default=None, help="Dataset label for downstream run metadata."
    )
    parser.add_argument("--config", default=None, help="Config label for downstream run metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_trace_paths(args.input_paths, args.output_dir)


if __name__ == "__main__":
    main()
