"""
LongBench-v2 evaluation pipeline.

Usage:
    uv run python examples/Eval.py
    uv run python examples/Eval.py --log-dir examples/logs_LongBenchPro --out examples/logs_LongBenchPro/eval_report.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Samples by context length
multidoc_qa_sample_long   = [7, 10, 12, 18, 28, 31, 46, 53, 57, 58]
multidoc_qa_sample_medium = [3, 8, 11, 17, 29, 36, 42, 44, 50, 61]
multidoc_qa_sample_short  = [1, 27, 40, 51, 52, 66, 67, 72, 86, 88]

_CONTEXT_LENGTH_MAP: dict[int, str] = {
    idx: label
    for label, group in (
        ("long",   multidoc_qa_sample_long),
        ("medium", multidoc_qa_sample_medium),
        ("short",  multidoc_qa_sample_short),
    )
    for idx in group
}

_EXP_LABEL_MAP: dict[str, str] = {
    "base":     "base (d=1, i=10)",
    "exp1":     "exp1 (d=1, i=30)",
    "baseline": "baseline (d=0, i=1)",
}

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMetadata:
    sample_idx: int
    exp_name: str
    depth: int
    max_iters: int
    model: str
    source: str           # "rlm" | "baseline"
    timestamp: str = ""
    num_iters_used: int = 0
    context_length: str = "unknown"   # "long" | "medium" | "short" | "unknown"
    exp_label: str = ""               # human-readable experiment description


@dataclass
class EvalResult:
    metadata: ExperimentMetadata
    predicted_letter: str
    ground_truth: str
    final_answer: str     # raw model output used for evaluation
    exact_match: bool = False
    partial_match: bool = False
    invalid_final_answer: bool = False  # True when model produced no final answer

    @property
    def sample_idx(self) -> int:
        return self.metadata.sample_idx


# ---------------------------------------------------------------------------
# Metric evaluator  (Open/Closed: subclass to add new metrics)
# ---------------------------------------------------------------------------

class MetricEvaluator:
    """Evaluates an EvalResult in-place and returns it."""

    def exact_match(self, predicted: str, ground_truth: str) -> bool:
        return predicted.strip().upper() == ground_truth.strip().upper()

    def partial_match(self, response_text: str, ground_truth: str) -> bool:
        return bool(re.search(rf'\b{ground_truth.upper()}\b', response_text.upper()))

    def evaluate(self, result: EvalResult) -> EvalResult:
        result.exact_match = self.exact_match(result.predicted_letter, result.ground_truth)
        result.partial_match = self.partial_match(result.final_answer, result.ground_truth)
        return result


# ---------------------------------------------------------------------------
# Log parsers  (Single Responsibility: one class per log format)
# ---------------------------------------------------------------------------

# sample10_base_d1_i10_2026-05-26_22-26-05_af2c4cba.jsonl
_RLM_FNAME_RE = re.compile(
    r"^sample(\d+)_([^_]+)_d(\d+)_i(\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_[a-f0-9]+\.jsonl$"
)


def _extract_letter(text: str) -> str:
    """Extract first A/B/C/D answer letter from a response string."""
    if not text:
        return ""
    text = text.strip()
    m = re.search(r'\b(?:answer(?:\s+is)?|correct\s+answer\s+is)[:\s]+([A-D])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.match(r'^([A-D])[^a-z]', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'\b([A-D])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return ""


class LogParser(ABC):
    """Abstract base — parse a log file into EvalResult objects."""

    @abstractmethod
    def can_parse(self, path: Path) -> bool: ...

    @abstractmethod
    def parse(self, path: Path) -> list[EvalResult]: ...


class RLMLogParser(LogParser):
    def can_parse(self, path: Path) -> bool:
        return bool(_RLM_FNAME_RE.match(path.name))

    def parse(self, path: Path) -> list[EvalResult]:
        m = _RLM_FNAME_RE.match(path.name)
        if not m:
            return []
        sample_idx, exp_name, depth, max_iters, timestamp = m.groups()

        iterations = []
        metadata_obj: dict = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") == "metadata":
                    metadata_obj = obj
                elif obj.get("type") == "iteration":
                    iterations.append(obj)

        if not iterations:
            return []

        last = iterations[-1]
        final_answer = last.get("final_answer") or ""
        response = last.get("response") or ""
        if isinstance(response, list):
            response = " ".join(
                msg.get("content", "") for msg in response if isinstance(msg, dict)
            )

        text = final_answer or response
        meta = ExperimentMetadata(
            sample_idx=int(sample_idx),
            exp_name=exp_name,
            depth=int(depth),
            max_iters=int(max_iters),
            model=metadata_obj.get("root_model", "unknown"),
            source="rlm",
            timestamp=timestamp,
            num_iters_used=len(iterations),
        )
        return [EvalResult(
            metadata=meta,
            predicted_letter=_extract_letter(text),
            ground_truth="",    # filled in by EvalRunner
            final_answer=text,
        )]


class BaselineLogParser(LogParser):
    def can_parse(self, path: Path) -> bool:
        return path.name == "baseline_gpt5.jsonl"

    def parse(self, path: Path) -> list[EvalResult]:
        results = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                response_text = obj.get("step1", {}).get("response", "")
                meta = ExperimentMetadata(
                    sample_idx=obj["sample_idx"],
                    exp_name="baseline",
                    depth=0,
                    max_iters=1,
                    model="gpt-5",
                    source="baseline",
                    num_iters_used=1,
                )
                results.append(EvalResult(
                    metadata=meta,
                    predicted_letter=obj.get("prediction") or _extract_letter(response_text),
                    ground_truth=obj.get("ground_truth", ""),
                    final_answer=response_text,
                ))
        return results


# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------

class ResultsDB:
    """Persists EvalResult objects to a SQLite database, one row per (sample_idx, exp_name)."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                sample_idx           INTEGER,
                exp_name             TEXT,
                depth                INTEGER,
                max_iters            INTEGER,
                model                TEXT,
                source               TEXT,
                timestamp            TEXT,
                num_iters_used       INTEGER,
                context_length       TEXT,
                exp_label            TEXT,
                predicted_letter     TEXT,
                ground_truth         TEXT,
                exact_match          INTEGER,
                partial_match        INTEGER,
                invalid_final_answer INTEGER,
                PRIMARY KEY (sample_idx, exp_name)
            )
        """)
        self.conn.commit()

    def upsert(self, r: EvalResult) -> None:
        m = r.metadata
        self.conn.execute("""
            INSERT OR REPLACE INTO eval_results VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            m.sample_idx, m.exp_name, m.depth, m.max_iters, m.model,
            m.source, m.timestamp, m.num_iters_used, m.context_length, m.exp_label,
            r.predicted_letter, r.ground_truth,
            int(r.exact_match), int(r.partial_match), int(r.invalid_final_answer),
        ))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# Eval runner  (Dependency Inversion: accepts any parsers / evaluator)
# ---------------------------------------------------------------------------

class EvalRunner:
    def __init__(
        self,
        parsers: list[LogParser] | None = None,
        evaluator: MetricEvaluator | None = None,
        db: ResultsDB | None = None,
    ):
        self.parsers = parsers or [RLMLogParser(), BaselineLogParser()]
        self.evaluator = evaluator or MetricEvaluator()
        self.db = db

    def run(self, log_dir: Path, dataset) -> list[EvalResult]:
        raw: dict[tuple, EvalResult] = {}  # dedup RLM logs, keep latest timestamp

        for path in sorted(log_dir.glob("*.jsonl")):
            for parser in self.parsers:
                if not parser.can_parse(path):
                    continue
                for result in parser.parse(path):
                    if not result.ground_truth:
                        result.ground_truth = dataset[result.sample_idx]["answer"]

                    result.invalid_final_answer = not bool(result.final_answer)

                    m = result.metadata
                    m.context_length = _CONTEXT_LENGTH_MAP.get(m.sample_idx, "unknown")
                    m.exp_label = _EXP_LABEL_MAP.get(m.exp_name, m.exp_name)

                    self.evaluator.evaluate(result)

                    if self.db:
                        self.db.upsert(result)

                    key = (result.metadata.sample_idx, result.metadata.exp_name)
                    existing = raw.get(key)
                    if existing is None or result.metadata.timestamp > existing.metadata.timestamp:
                        raw[key] = result

        return sorted(raw.values(), key=lambda r: (r.sample_idx, r.metadata.exp_name))


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

class ReportWriter:
    def write(self, results: list[EvalResult], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("LONGBENCH-V2 EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Per-run table
            header = f"{'idx':>4}  {'exp':<10}  {'d':>2}  {'i':>3}  {'model':<14}  {'pred':>4}  {'gt':>2}  {'EM':>3}  {'PM':>3}  {'iters':>5}  {'note':<22}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            for r in results:
                m = r.metadata
                note = "INVALID_FINAL_ANSWER" if r.invalid_final_answer else ""
                f.write(
                    f"{r.sample_idx:>4}  {m.exp_name:<10}  {m.depth:>2}  {m.max_iters:>3}  "
                    f"{m.model:<14}  {r.predicted_letter or '?':>4}  {r.ground_truth:>2}  "
                    f"{'✓' if r.exact_match else '✗':>3}  "
                    f"{'✓' if r.partial_match else '✗':>3}  "
                    f"{m.num_iters_used:>5}  {note:<22}\n"
                )

            # Summary by experiment
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("SUMMARY BY EXPERIMENT\n")
            f.write("=" * 80 + "\n\n")

            by_exp: dict[str, list[EvalResult]] = {}
            for r in results:
                by_exp.setdefault(r.metadata.exp_name, []).append(r)

            for exp, group in sorted(by_exp.items()):
                n = len(group)
                em = sum(r.exact_match for r in group) / n
                pm = sum(r.partial_match for r in group) / n
                avg_iters = sum(r.metadata.num_iters_used for r in group) / n
                f.write(f"  {exp:<12}  n={n}  EM={em:.2f}  PM={pm:.2f}  avg_iters={avg_iters:.1f}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="examples/logs_LongBenchPro")
    parser.add_argument("--out",     default="examples/logs_LongBenchPro/eval_report.txt")
    parser.add_argument("--db",      default="examples/logs_LongBenchPro/eval_results.db")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    dataset = load_dataset("zai-org/LongBench-v2", split="train")

    db = ResultsDB(Path(args.db))
    runner = EvalRunner(db=db)
    results = runner.run(log_dir, dataset)
    db.close()

    if not results:
        print("No log files found.")
        return

    for r in results:
        m = r.metadata
        print(
            f"  sample={r.sample_idx:>3}  {m.exp_name:<10}  d={m.depth}  i={m.max_iters:>2}  "
            f"pred={r.predicted_letter or '?'}  gt={r.ground_truth}  "
            f"EM={'✓' if r.exact_match else '✗'}  PM={'✓' if r.partial_match else '✗'}  "
            f"iters={m.num_iters_used}"
        )

    writer = ReportWriter()
    writer.write(results, Path(args.out))
    print(f"\nReport written to {args.out}")


def exploratory_analysis(log_dir: Path, out_excel: Path, dataset=None) -> pd.DataFrame:
    log_dir = Path(log_dir)
    if dataset is None:
        dataset = load_dataset("zai-org/LongBench-v2", split="train")

    runner = EvalRunner()
    results = runner.run(log_dir, dataset)
    if not results:
        print("No log files found.")
        return pd.DataFrame()

    summary_rows: list[dict[str, object]] = []

    def append_summary(group: list[EvalResult], category_type: str, category_value: str, exp_name: str) -> None:
        n = len(group)
        invalid_count = sum(r.invalid_final_answer for r in group)
        summary_rows.append({
            "category_type": category_type,
            "category_value": category_value,
            "exp_name": exp_name,
            "n": n,
            "exact_match_rate": sum(r.exact_match for r in group) / n if n else 0.0,
            "partial_match_rate": sum(r.partial_match for r in group) / n if n else 0.0,
            "invalid_final_count": invalid_count,
            "invalid_final_rate": invalid_count / n if n else 0.0,
        })

    append_summary(results, "overall", "all", "all")
    for exp_name in ("base", "exp1", "baseline"):
        subset = [r for r in results if r.metadata.exp_name == exp_name]
        append_summary(subset, "overall", "all", exp_name)

    for length in ("short", "medium", "long"):
        for exp_name in ("base", "exp1", "baseline"):
            subset = [
                r for r in results
                if r.metadata.context_length == length and r.metadata.exp_name == exp_name
            ]
            append_summary(subset, "context_length", length, exp_name)

    df = pd.DataFrame(summary_rows)
    out_excel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="analysis", index=False)

    print(f"Exploratory analysis written to {out_excel}")
    return df


if __name__ == "__main__":
    main()
