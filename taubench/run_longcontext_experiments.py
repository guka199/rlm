"""
run_longcontext_experiments.py
==============================
RLM next-action prediction on the long-context τ-bench airline dataset.
Replays from padded JSONL datasets; prompt is split so the conversation
context goes to `prompt` and the prediction question goes to `root_prompt`.

Experiments: depth1_iter10, depth1_iter30
Context variants: 8k_start, 8k_end, 66k_start, 66k_end  (8 total runs)

Build the dataset first:
    python build_longcontext_dataset.py

Usage:
    export AZURE_OPENAI_API_KEY=...
    export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
    export AZURE_OPENAI_API_VERSION=2024-02-15-preview

    python run_longcontext_experiments.py                       # all 8 runs
    python run_longcontext_experiments.py --variant 8k_start    # one variant
    python run_longcontext_experiments.py --experiment iter10   # one experiment
    python run_longcontext_experiments.py --num-samples 5       # smoke test
"""

import argparse
import re
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {"name": "iter10", "max_depth": 1, "max_iterations": 10, "model": "gpt-5"},
    {"name": "iter30", "max_depth": 1, "max_iterations": 30, "model": "gpt-5"},
]

# ---------------------------------------------------------------------------
# Context variants
# ---------------------------------------------------------------------------

CONTEXT_VARIANTS = [
    {"name": "8k_start",  "file": "data/longcontext_8k.jsonl",  "position": "start"},
    {"name": "8k_end",    "file": "data/longcontext_8k.jsonl",  "position": "end"},
    {"name": "66k_start", "file": "data/longcontext_66k.jsonl", "position": "start"},
    {"name": "66k_end",   "file": "data/longcontext_66k.jsonl", "position": "end"},
]

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

RESULTS_DIR      = Path("results")
LOGS_DIR         = Path("logs")
ALL_RESULTS_PATH = RESULTS_DIR / "longcontext_results.jsonl"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a next-action predictor for an airline customer service agent.
Given a conversation history and a list of available tools, your only job is to decide the single next action the agent should take.
Use FINAL(tool_name) to submit your answer. One answer only.\
"""

AVAILABLE_TOOLS = [
    "get_user_details",
    "get_reservation_details",
    "search_direct_flight",
    "search_onestop_flight",
    "list_all_airports",
    "calculate",
    "think",
    "book_reservation",
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "update_reservation_passengers",
    "send_certificate",
    "transfer_to_human_agents",
]

ROOT_PROMPT = (
    "Given the conversation above, what is the single next tool the agent should call?\n"
    f"Available tools: {', '.join(AVAILABLE_TOOLS)}\n"
    "Use FINAL(tool_name) to answer with one tool name only."
)


# ---------------------------------------------------------------------------
# Context formatting  (prompt = context, root_prompt = question)
# ---------------------------------------------------------------------------

def build_context(padded_context: list[dict]) -> str:
    """
    Format padded_context turns into the `prompt` string passed to RLM.
    System turns are skipped — the agent role is set via custom_system_prompt.
    """
    lines = []
    for turn in padded_context:
        role    = turn.get("role", "unknown")
        content = turn.get("content") or ""

        if role == "system":
            continue
        elif role == "user":
            lines.append(f"USER: {content}")
        elif role == "tool":
            tool_name = turn.get("name", "tool")
            lines.append(f"TOOL RESULT ({tool_name}): {content}")
        elif role == "assistant":
            tool_calls = turn.get("tool_calls") or []
            if tool_calls:
                fn   = tool_calls[0]["function"]["name"]
                args = tool_calls[0]["function"].get("arguments", "{}")
                lines.append(f"AGENT called: {fn}({args})")
            else:
                lines.append(f"AGENT: {content}")

    conv = "\n".join(lines)
    return (
        "CONVERSATION:\n"
        "----------------------------------------\n"
        f"{conv}\n"
        "----------------------------------------"
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def full_exp_name(exp_name: str, variant_name: str) -> str:
    return f"{exp_name}_{variant_name}"


def load_done(exp_name: str, variant_name: str) -> set[str]:
    key  = full_exp_name(exp_name, variant_name)
    done = set()
    if ALL_RESULTS_PATH.exists():
        with open(ALL_RESULTS_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("experiment") == key:
                        done.add(r["qa_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


# ---------------------------------------------------------------------------
# Single experiment × variant run
# ---------------------------------------------------------------------------

def run_experiment(
    exp: dict,
    variant: dict,
    qa_pairs: list[dict],
    backend: str,
    backend_kwargs: dict,
) -> dict:
    exp_name     = exp["name"]
    variant_name = variant["name"]
    name         = full_exp_name(exp_name, variant_name)

    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    done      = load_done(exp_name, variant_name)
    remaining = [q for q in qa_pairs if q["qa_id"] not in done]

    print(f"\n{'='*60}")
    print(f"Experiment : {name}")
    print(f"  max_depth={exp['max_depth']}, max_iterations={exp['max_iterations']}")
    print(f"  Total QA pairs : {len(qa_pairs)}")
    print(f"  Already done   : {len(done)}")
    print(f"  To run         : {len(remaining)}")

    if not remaining:
        print("  Nothing to do — fully checkpointed.")
        return summarize(exp_name, variant_name)

    logger = RLMLogger()
    rlm = RLM(
        backend=backend,
        backend_kwargs={**backend_kwargs, "model_name": exp["model"]},
        max_depth=exp["max_depth"],
        max_iterations=exp["max_iterations"],
        custom_system_prompt=SYSTEM_PROMPT,
        logger=logger,
        verbose=True,
    )

    log_dir = LOGS_DIR / name
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(ALL_RESULTS_PATH, "a") as results_out:
        for i, qa in enumerate(remaining):
            try:
                completion = rlm.completion(qa["padded_context"], root_prompt=ROOT_PROMPT)
                raw        = completion.response.strip()
                m          = re.search(r"FINAL\(([^)]+)\)", raw)
                predicted  = m.group(1).strip() if m else raw.split()[0]
                error      = None
            except Exception as e:
                predicted = ""
                error     = str(e)

            correct = predicted == qa["ground_truth"]

            record = {
                # provenance — matches longcontext JSONL schema for easy join
                "qa_id":           qa["qa_id"],
                "traj_id":         qa["traj_id"],
                "task_id":         qa["task_id"],
                "turn_index":      qa["turn_index"],
                "step":            qa["step"],
                "traj_reward":     qa["traj_reward"],
                "ground_truth":    qa["ground_truth"],
                "position":        qa["position"],
                "context_size":    qa["context_size"],
                "total_tokens":    qa["total_tokens"],
                # prediction
                "predicted":       predicted,
                "correct":         correct,
                "error":           error,
                # experiment tags
                "experiment":      name,
                "exp_name":        exp_name,
                "variant_name":    variant_name,
                "max_depth":       exp["max_depth"],
                "max_iterations":  exp["max_iterations"],
                "timestamp":       datetime.now().isoformat(),
            }
            results_out.write(json.dumps(record) + "\n")
            results_out.flush()

            trajectory = logger.get_trajectory()
            if trajectory:
                log_path = log_dir / f"{qa['qa_id']}.jsonl"
                with open(log_path, "w") as log_f:
                    log_f.write(json.dumps({
                        "type": "metadata",
                        "timestamp": record["timestamp"],
                        **trajectory["run_metadata"],
                    }) + "\n")
                    for iteration in trajectory["iterations"]:
                        log_f.write(json.dumps(iteration) + "\n")

            status = "✓" if correct else "✗"
            print(
                f"  [{i+1}/{len(remaining)}] {status}"
                f"  gt={qa['ground_truth']:<35} pred={predicted}"
            )

    return summarize(exp_name, variant_name)


def summarize(exp_name: str, variant_name: str) -> dict:
    key     = full_exp_name(exp_name, variant_name)
    records = []
    if ALL_RESULTS_PATH.exists():
        with open(ALL_RESULTS_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("experiment") == key:
                        records.append(r)
                except json.JSONDecodeError:
                    pass

    total   = len(records)
    correct = sum(1 for r in records if r["correct"])
    errors  = sum(1 for r in records if r.get("error"))
    acc     = correct / total if total else 0.0

    print(f"\n  --- {key} ---")
    print(f"  Accuracy : {correct}/{total}  ({acc:.1%})")
    if errors:
        print(f"  Errors   : {errors}")

    return {"experiment": key, "accuracy": acc, "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# Azure setup
# ---------------------------------------------------------------------------

def configure_azure() -> tuple[str, dict]:
    api_key     = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint    = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not api_key:
        print("ERROR: AZURE_OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    if not endpoint:
        print("ERROR: AZURE_OPENAI_ENDPOINT not set.", file=sys.stderr)
        sys.exit(1)

    os.environ["AZURE_API_KEY"]     = api_key
    os.environ["AZURE_API_BASE"]    = endpoint.rstrip("/")
    os.environ["AZURE_API_VERSION"] = api_version

    print(f"Azure OpenAI: endpoint={endpoint}  api_version={api_version}")
    return "azure_openai", {
        "api_key":        api_key,
        "azure_endpoint": endpoint.rstrip("/"),
        "api_version":    api_version,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", default=None,
        choices=[v["name"] for v in CONTEXT_VARIANTS],
        help="Run only this context variant. Default: all 4.",
    )
    parser.add_argument(
        "--experiment", default=None,
        choices=[e["name"] for e in EXPERIMENTS],
        help="Run only this experiment config. Default: both.",
    )
    parser.add_argument(
        "--successful-only", action="store_true",
        help="Only QA pairs from reward=1.0 airline trajectories.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Cap QA pairs per variant (smoke test).",
    )
    args = parser.parse_args()

    backend, backend_kwargs = configure_azure()

    variants = (
        [v for v in CONTEXT_VARIANTS if v["name"] == args.variant]
        if args.variant else CONTEXT_VARIANTS
    )
    exps = (
        [e for e in EXPERIMENTS if e["name"] == args.experiment]
        if args.experiment else EXPERIMENTS
    )

    summaries = []

    for variant in variants:
        variant_path = Path(variant["file"])
        if not variant_path.exists():
            print(
                f"ERROR: {variant_path} not found. "
                "Run build_longcontext_dataset.py first.",
                file=sys.stderr,
            )
            sys.exit(1)

        qa_pairs = []
        with open(variant_path) as f:
            for line in f:
                try:
                    qa_pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        qa_pairs = [q for q in qa_pairs if q["position"] == variant["position"]]

        if args.successful_only:
            qa_pairs = [q for q in qa_pairs if q["traj_reward"] == 1.0]
            print(f"[{variant['name']}] Filtered to {len(qa_pairs)} successful pairs")

        if args.num_samples:
            qa_pairs = qa_pairs[:args.num_samples]

        print(f"\n[{variant['name']}] {len(qa_pairs)} QA pairs")

        for exp in exps:
            s = run_experiment(exp, variant, qa_pairs, backend, backend_kwargs)
            summaries.append(s)

    if len(summaries) > 1:
        print(f"\n{'='*60}")
        print("FINAL COMPARISON")
        print(f"{'Experiment':<35} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
        print("-" * 65)
        for s in summaries:
            print(
                f"{s['experiment']:<35} {s['accuracy']:>9.1%}"
                f" {s['correct']:>10} {s['total']:>8}"
            )


if __name__ == "__main__":
    main()
