"""
run_experiments.py
==================
Runs RLM experiment configurations on τ-bench airline QA datasets.

Experiments:
    1. max_depth=1, max_iterations=10
    2. max_depth=1, max_iterations=30
    3. max_depth=2, max_iterations=10

Long-context experiments (pass --input and --context-variant):
    python run_experiments.py --input data/longcontext_8k.jsonl  --context-variant 8k_start  --gold-position start
    python run_experiments.py --input data/longcontext_8k.jsonl  --context-variant 8k_end    --gold-position end
    python run_experiments.py --input data/longcontext_66k.jsonl --context-variant 66k_start --gold-position start
    python run_experiments.py --input data/longcontext_66k.jsonl --context-variant 66k_end   --gold-position end

Usage:
    export AZURE_OPENAI_API_KEY=...
    export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
    export AZURE_OPENAI_DEPLOYMENT=gpt-5
    export AZURE_OPENAI_API_VERSION=2024-02-15-preview

    python run_experiments.py                       # all 3 experiments
    python run_experiments.py --successful-only     # reward=1.0 trajectories only
    python run_experiments.py --experiment 1        # single experiment
    python run_experiments.py --num-samples 5       # smoke test

Note on depth=2:
    The upstream rlm library marks max_depth > 1 as a TODO (not yet implemented).
    Experiment 3 is included as specified but may behave identically to depth=1.
    Check with your team if guka199/rlm has patched this.
"""

import argparse
import re
import json
import os
import sys
from datetime import datetime
from pathlib import Path


from rlm import RLM
from rlm.logger import RLMLogger


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {"id": 1, "name": "depth1_iter10", "max_depth": 1, "max_iterations": 10},
    {"id": 2, "name": "depth1_iter30", "max_depth": 1, "max_iterations": 30},
    {"id": 3, "name": "depth2_iter10", "max_depth": 2, "max_iterations": 10},
]

RESULTS_DIR = Path("results")
LOGS_DIR = Path("logs")

ALL_RESULTS_PATH = RESULTS_DIR / "all_results.jsonl"


# ---------------------------------------------------------------------------
# Azure setup
# ---------------------------------------------------------------------------

def configure_azure() -> tuple[str, dict]:
    """
    Configure Azure OpenAI via LiteLLM backend.
    LiteLLM has native Azure support via the "azure/<deployment>" model string.

    Returns (backend, backend_kwargs) ready to pass to RLM().
    """
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get(
        "AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    deployment = "gpt-5"

    if not api_key:
        print("ERROR: AZURE_OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    if not endpoint:
        print("ERROR: AZURE_OPENAI_ENDPOINT not set.", file=sys.stderr)
        sys.exit(1)

    # LiteLLM reads these env vars automatically
    os.environ["AZURE_API_KEY"] = api_key
    os.environ["AZURE_API_BASE"] = endpoint.rstrip("/")
    os.environ["AZURE_API_VERSION"] = api_version

    print(f"Azure OpenAI configured (via litellm)")
    print(f"  endpoint   : {endpoint}")
    print(f"  deployment : {deployment}")
    print(f"  api_version: {api_version}")

    return "azure_openai", {
        "model_name":     deployment,
        "api_key":        api_key,
        "azure_endpoint": endpoint.rstrip("/"),
        "api_version":    api_version,
    }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a next-action classifier for an airline customer service agent.
{custom_tools_section}
Your ONLY job: inspect the conversation in the `context` variable and output the single tool name the agent should call next. Use FINAL(tool_name) to submit your answer — one word, the tool name only.

Available tools:
get_user_details, get_reservation_details, search_direct_flight,
search_onestop_flight, list_all_airports, calculate, think,
book_reservation, cancel_reservation, update_reservation_flights,
update_reservation_baggages, update_reservation_passengers,
send_certificate, transfer_to_human_agents\
"""


ROOT_PROMPT = (
    "What is the next tool the agent should call? "
    "Available tools: get_user_details, get_reservation_details, search_direct_flight, "
    "search_onestop_flight, list_all_airports, calculate, think, book_reservation, "
    "cancel_reservation, update_reservation_flights, update_reservation_baggages, "
    "update_reservation_passengers, send_certificate, transfer_to_human_agents. "
    "Use FINAL(tool_name) to answer with one tool name only."
)


def build_context(context: list[dict]) -> str:
    """Build the conversation context string passed as `prompt` to RLM."""
    conv_lines = []
    for turn in context:
        role = turn.get("role", "unknown")
        content = turn.get("content") or ""

        if role == "system":
            continue  # skip the long policy doc — not needed for classification
        elif role == "user":
            conv_lines.append(f"USER: {content}")
        elif role == "tool":
            tool_name = turn.get("name", "tool")
            conv_lines.append(f"TOOL RESULT ({tool_name}): {content}")
        elif role == "assistant":
            tool_calls = turn.get("tool_calls") or []
            if tool_calls:
                fn = tool_calls[0]["function"]["name"]
                args = tool_calls[0]["function"].get("arguments", "{}")
                conv_lines.append(f"AGENT called: {fn}({args})")
            else:
                conv_lines.append(f"AGENT: {content}")

    conv = "\n".join(conv_lines)

    return (
        "AIRLINE CUSTOMER SERVICE CONVERSATION:\n"
        "----------------------------------------\n"
        f"{conv}\n"
        "----------------------------------------"
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_done(experiment_name: str) -> set[str]:
    done = set()
    if ALL_RESULTS_PATH.exists():
        with open(ALL_RESULTS_PATH) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("experiment") == experiment_name:
                        done.add(record["qa_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_experiment(config: dict, qa_pairs: list[dict], backend: str, backend_kwargs: dict):
    name = config["name"]
    max_depth = config["max_depth"]
    max_iters = config["max_iterations"]

    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    done = load_done(name)
    remaining = [q for q in qa_pairs if q["qa_id"] not in done]

    print(f"\n{'='*60}")
    print(f"Experiment {config['id']}: {name}")
    print(f"  max_depth={max_depth}, max_iterations={max_iters}")
    print(f"  Total QA pairs : {len(qa_pairs)}")
    print(f"  Already done   : {len(done)}")
    print(f"  To run         : {len(remaining)}")
    print(f"  Results        → {ALL_RESULTS_PATH}")
    print(f"  Logs           → {LOGS_DIR / name}/<qa_id>.jsonl")

    if not remaining:
        print("  Nothing to do — fully checkpointed.")
        return summarize(name)

    logger = RLMLogger()
    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        max_depth=max_depth,
        max_iterations=max_iters,
        custom_system_prompt=SYSTEM_PROMPT,
        logger=logger,
        verbose=True,
    )

    qa_log_dir = LOGS_DIR / name
    qa_log_dir.mkdir(parents=True, exist_ok=True)

    with open(ALL_RESULTS_PATH, "a") as results_out:
        for i, qa in enumerate(remaining):
            prompt = build_context(qa["context"])
            try:
                completion = rlm.completion(
                    prompt, root_prompt=ROOT_PROMPT)
                raw = completion.response.strip()
                m = re.search(r'FINAL\(([^)]+)\)', raw)
                predicted = m.group(1).strip() if m else raw.split()[0]
                error = None
            except Exception as e:
                predicted = ""
                error = str(e)

            correct = predicted == qa["next_action"]

            record = {
                # provenance
                "qa_id":          qa["qa_id"],
                "traj_id":        qa["traj_id"],
                "task_id":        qa["task_id"],
                "trial":          qa["trial"],
                "traj_reward":    qa["traj_reward"],
                "turn_index":     qa["turn_index"],
                "step":           qa["step"],
                # prediction
                "predicted":      predicted,
                "ground_truth":   qa["next_action"],
                "correct":        correct,
                "error":          error,
                # experiment tag
                "experiment":     name,
                "max_depth":      max_depth,
                "max_iterations": max_iters,
            }
            results_out.write(json.dumps(record) + "\n")
            results_out.flush()

            trajectory = logger.get_trajectory()
            if trajectory:
                qa_log_path = qa_log_dir / f"{qa['qa_id']}.jsonl"
                with open(qa_log_path, "w") as log_f:
                    log_f.write(json.dumps({
                        "type": "metadata",
                        "timestamp": datetime.now().isoformat(),
                        **trajectory["run_metadata"],
                    }) + "\n")
                    for iteration in trajectory["iterations"]:
                        log_f.write(json.dumps(iteration) + "\n")

            status = "✓" if correct else "✗"
            print(
                f"  [{i+1}/{len(remaining)}] {status}  gt={qa['next_action']:<35} pred={predicted}")

    return summarize(name)


def summarize(name: str) -> dict:
    records = []
    if ALL_RESULTS_PATH.exists():
        with open(ALL_RESULTS_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("experiment") == name:
                        records.append(r)
                except json.JSONDecodeError:
                    pass

    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    errors = sum(1 for r in records if r.get("error"))
    acc = correct / total if total else 0.0

    print(f"\n  --- {name} summary ---")
    print(f"  Accuracy : {correct}/{total}  ({acc:.1%})")
    if errors:
        print(f"  Errors   : {errors}")

    return {"experiment": name, "accuracy": acc, "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",           default="data/taubench_airline_qa.jsonl")
    parser.add_argument("--experiment",      type=int, choices=[1, 2, 3], default=None,
                        help="Run only this experiment (1, 2, or 3). Default: all.")
    parser.add_argument("--successful-only", action="store_true",
                        help="Only use QA pairs from successful trajectories (reward=1.0).")
    parser.add_argument("--num-samples",     type=int, default=None,
                        help="Cap number of QA pairs (useful for smoke tests).")
    parser.add_argument("--gold-position",   choices=["start", "end"], default=None,
                        help="Filter long-context records by gold context position.")
    parser.add_argument("--context-variant", default=None,
                        help="Suffix appended to experiment names (e.g. '8k_start'). "
                             "Use when running on a long-context dataset so results "
                             "are checkpointed separately from the baseline.")
    args = parser.parse_args()

    # --- configure Azure ---
    backend, backend_kwargs = configure_azure()

    # --- load dataset ---
    input_path = Path(args.input)
    if not input_path.exists():
        print(
            f"ERROR: {input_path} not found. Run build_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    qa_pairs = []
    with open(input_path) as f:
        for line in f:
            try:
                qa_pairs.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if args.successful_only:
        qa_pairs = [q for q in qa_pairs if q["traj_reward"] == 1.0]
        print(f"Filtered to successful trajectories: {len(qa_pairs)} pairs")

    if args.gold_position:
        qa_pairs = [q for q in qa_pairs if q.get("gold_position") == args.gold_position]
        print(f"Filtered to gold_position={args.gold_position}: {len(qa_pairs)} pairs")

    if args.num_samples:
        qa_pairs = qa_pairs[:args.num_samples]
        print(f"Capped to {len(qa_pairs)} samples")

    # Apply context variant suffix to experiment names for separate checkpointing
    configs_base = EXPERIMENTS if args.experiment is None else [
        c for c in EXPERIMENTS if c["id"] == args.experiment
    ]
    if args.context_variant:
        configs = [
            {**c, "name": f"{c['name']}_{args.context_variant}"}
            for c in configs_base
        ]
    else:
        configs = configs_base

    summaries = []
    for config in configs:
        s = run_experiment(config, qa_pairs, backend, backend_kwargs)
        summaries.append(s)

    # --- final comparison ---
    if len(summaries) > 1:
        print(f"\n{'='*60}")
        print("FINAL COMPARISON")
        print(f"{'Experiment':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
        print("-" * 50)
        for s in summaries:
            print(
                f"{s['experiment']:<20} {s['accuracy']:>9.1%} {s['correct']:>10} {s['total']:>8}")


if __name__ == "__main__":
    main()
