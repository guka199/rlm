"""
LongBench-v2 result analysis.

Usage:
    uv run python examples/LongBenchResultAnalasys.py
    uv run python examples/LongBenchResultAnalasys.py --log-dir examples/logs_LongBenchPro --out examples/logs_LongBenchPro/analysis.txt
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class Metrics:
    def exact_match(self, predicted_letter: str, ground_truth: str) -> bool:
        """Predicted letter == ground truth letter."""
        return predicted_letter.strip().upper() == ground_truth.strip().upper()

    def partial_match(self, response_text: str, ground_truth: str) -> bool:
        """Ground truth letter appears as a standalone letter anywhere in response."""
        return bool(re.search(rf'\b{ground_truth.upper()}\b', response_text.upper()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# sample10_base_d1_i10_2026-05-26_22-26-05_af2c4cba.jsonl
_RLM_FNAME_RE = re.compile(
    r"^sample(\d+)_([^_]+)_d(\d+)_i(\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_[a-f0-9]+\.jsonl$"
)

def extract_letter(text: str) -> str:
    """Extract the first A/B/C/D answer letter from a response string."""
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


def parse_rlm_log(path: Path) -> dict | None:
    m = _RLM_FNAME_RE.match(path.name)
    if not m:
        return None

    sample_idx, exp_name, depth, iters, timestamp = m.groups()

    iterations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "iteration":
                iterations.append(obj)

    if not iterations:
        return None

    last = iterations[-1]
    final_answer = last.get("final_answer") or ""
    response = last.get("response") or ""
    if isinstance(response, list):
        response = " ".join(msg.get("content", "") for msg in response if isinstance(msg, dict))

    return {
        "source":           "rlm",
        "sample_idx":       int(sample_idx),
        "exp_name":         exp_name,
        "max_depth":        int(depth),
        "max_iters":        int(iters),
        "timestamp":        timestamp,
        "num_iters":        len(iterations),
        "final_answer":     final_answer,
        "response":         response,
        "predicted_letter": extract_letter(final_answer) or extract_letter(response),
    }


def parse_baseline_log(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            response_text = obj.get("step1", {}).get("response", "")
            records.append({
                "source":           "baseline",
                "sample_idx":       obj["sample_idx"],
                "exp_name":         "baseline",
                "max_depth":        0,
                "max_iters":        1,
                "timestamp":        "",
                "num_iters":        1,
                "final_answer":     response_text,
                "response":         response_text,
                "predicted_letter": obj.get("prediction") or extract_letter(response_text),
                "ground_truth":     obj.get("ground_truth", ""),
            })
    return records


def collect_runs(log_dir: Path) -> list[dict]:
    seen: dict[tuple, dict] = {}

    for p in sorted(log_dir.glob("*.jsonl")):
        if p.name in ("baseline_gpt5.jsonl", "experiment_results.json"):
            continue
        run = parse_rlm_log(p)
        if run is None:
            continue
        key = (run["sample_idx"], run["exp_name"], run["max_depth"], run["max_iters"])
        if key not in seen or run["timestamp"] > seen[key]["timestamp"]:
            seen[key] = run

    runs = list(seen.values())

    baseline_path = log_dir / "baseline_gpt5.jsonl"
    if baseline_path.exists():
        runs.extend(parse_baseline_log(baseline_path))

    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(log_dir: str, out_path: str) -> None:
    log_dir = Path(log_dir)
    dataset = load_dataset("zai-org/LongBench-v2", split="train")
    metrics = Metrics()

    runs = collect_runs(log_dir)
    if not runs:
        print("No log files found.")
        return

    print(f"Found {len(runs)} runs in {log_dir}\n")

    results = []
    for run in sorted(runs, key=lambda r: (r["sample_idx"], r["exp_name"])):
        idx = run["sample_idx"]
        ground_truth = run.get("ground_truth") or dataset[idx]["answer"]

        pred = run["predicted_letter"]
        response_text = run["final_answer"] or run["response"]

        em = metrics.exact_match(pred, ground_truth)
        pm = metrics.partial_match(response_text, ground_truth)

        results.append({**run, "ground_truth": ground_truth, "exact_match": em, "partial_match": pm})

        print(
            f"  sample={idx:>3}  {run['exp_name']:<10}  d={run['max_depth']}  i={run['max_iters']:>2}  "
            f"pred={pred or '?'}  gt={ground_truth}  EM={'✓' if em else '✗'}  PM={'✓' if pm else '✗'}  "
            f"iters={run['num_iters']}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LONGBENCH-V2 RESULT ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'idx':>4}  {'exp':<10}  {'d':>2}  {'i':>3}  {'pred':>4}  {'gt':>2}  {'EM':>3}  {'PM':>3}  {'iters':>5}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(
                f"{r['sample_idx']:>4}  {r['exp_name']:<10}  {r['max_depth']:>2}  {r['max_iters']:>3}  "
                f"{r['predicted_letter'] or '?':>4}  {r['ground_truth']:>2}  "
                f"{'✓' if r['exact_match'] else '✗':>3}  "
                f"{'✓' if r['partial_match'] else '✗':>3}  "
                f"{r['num_iters']:>5}\n"
            )

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("SUMMARY BY EXPERIMENT\n")
        f.write("=" * 80 + "\n\n")

        by_exp: dict[str, list] = {}
        for r in results:
            by_exp.setdefault(r["exp_name"], []).append(r)

        for exp, group in sorted(by_exp.items()):
            n = len(group)
            em_acc = sum(r["exact_match"] for r in group) / n
            pm_acc = sum(r["partial_match"] for r in group) / n
            avg_iters = sum(r["num_iters"] for r in group) / n
            f.write(f"  {exp:<12}  n={n}  EM={em_acc:.2f}  PM={pm_acc:.2f}  avg_iters={avg_iters:.1f}\n")

    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="examples/logs_LongBenchPro")
    parser.add_argument("--out",     default="examples/logs_LongBenchPro/analysis.txt")
    args = parser.parse_args()
    analyze(args.log_dir, args.out)
