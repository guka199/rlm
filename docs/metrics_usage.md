# RLM Trace and Failure Metrics

This repository includes small analysis scripts for extracting model responses from RLM trace logs and computing failure-mode metrics from labeled spreadsheets.

## Extract Model Responses

Run the response extractor on one or more JSON or JSONL trace files:

```bash
uv run python scripts/analysis/extract_responses.py path/to/trace.json path/to/run.jsonl \
  --dataset tau-bench \
  --config baseline \
  --output-dir outputs
```

The extractor searches each trace record for common response fields:

- `response`
- `output`
- `content`
- `message.content`
- `choices[0].message.content`
- `assistant_response`
- `model_response`

Malformed JSON is written to an invalid trace report instead of raising and stopping the run.

Outputs:

- `outputs/extracted/extracted_responses.jsonl`
- `outputs/extracted/invalid_traces.jsonl`

## Normalize Labels and Compute Failure Metrics

Run the failure metrics script on one or more labeled Excel or CSV files:

```bash
uv run python scripts/analysis/failure_metrics.py path/to/labels.xlsx \
  --sheet-name Sheet1 \
  --dataset tau-bench \
  --config baseline \
  --output-dir outputs
```

Supported label layouts:

- Direct columns such as `Process/Stage` and `Failure modes`.
- Alternating rows where one row names `Process/Stage` and the next row names `Failure modes`.

Failure-mode cells may contain multiple labels separated by commas, semicolons, pipes, or newlines. Blank failure cells are skipped. Repeated labels within a single cell are counted once for that row.

Outputs:

- `outputs/failure_metrics/normalized_labels.csv`
- `outputs/failure_metrics/failure_counts_by_step.csv`
- `outputs/failure_metrics/aggregate_failure_counts.csv`

## Development Checks

Before opening a PR, run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
