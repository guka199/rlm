"""
build_dataset.py
================
Converts raw τ-bench airline trajectory JSON into a flat JSONL dataset of
next-action-prediction QA pairs.

Each QA record is fully self-contained and tagged back to its source
trajectory so results can be grouped/filtered by task, trial, or outcome.

Setup:
    git clone https://github.com/sierra-research/tau-bench
    python build_dataset.py \
        --input  tau-bench/historical_trajectories/gpt-4o-airline.json \
        --output data/taubench_airline_qa.jsonl

Output schema (one JSON object per line):
    {
      # --- trajectory provenance ---
      "traj_id":          str,   # "{task_id}_{trial}"  — unique per trajectory
      "task_id":          int,
      "trial":            int,
      "traj_reward":      float, # 0.0 or 1.0 — did the full trajectory succeed?

      # --- QA pair ---
      "qa_id":            str,   # "{task_id}_{trial}_{turn_index}" — unique per pair
      "turn_index":       int,   # index of the tool-calling assistant turn in the traj
      "step":             int,   # 1-based counter of tool calls within this trajectory
      "context":          list,  # all turns BEFORE this assistant turn (system, user, assistant, tool)
      "next_action":      str,   # ground-truth tool name (e.g. "get_user_details")
      "next_action_args": dict,  # parsed tool arguments
    }
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_qa_pairs(trajectories: list[dict]) -> list[dict]:
    """
    Walk every trajectory and emit one QA record per assistant turn
    that issues a tool call.
    """
    qa_pairs = []

    for traj in trajectories:
        task_id = traj.get("task_id", -1)
        trial = traj.get("trial", 0)
        reward = float(traj.get("reward", 0.0))
        turns = traj.get("traj", [])

        traj_id = f"{task_id}_{trial}"
        step_ctr = 0  # counts tool calls within this trajectory

        for i, turn in enumerate(turns):
            if turn.get("role") != "assistant":
                continue

            tool_calls = turn.get("tool_calls") or []
            if not tool_calls:
                continue  # assistant message with no tool call — skip

            # Use only the first tool call (τ-bench agents call one tool at a time)
            tc = tool_calls[0]
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"].get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                tool_args = {}

            step_ctr += 1

            qa_pairs.append({
                # provenance
                "traj_id":          traj_id,
                "task_id":          task_id,
                "trial":            trial,
                "traj_reward":      reward,
                # QA pair
                "qa_id":            f"{task_id}_{trial}_{i}",
                "turn_index":       i,
                "step":             step_ctr,
                "context":          turns[:i],   # everything before this turn
                "next_action":      tool_name,
                "next_action_args": tool_args,
            })

    return qa_pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build τ-bench airline QA dataset from raw trajectories."
    )
    parser.add_argument(
        "--input", "-i",
        default="tau-bench/historical_trajectories/gpt-4o-airline.json",
        help="Path to gpt-4o-airline.json",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/taubench_airline_qa.jsonl",
        help="Output .jsonl path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # --- load ---
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        print(
            "Clone tau-bench first:\n"
            "  git clone https://github.com/sierra-research/tau-bench",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading trajectories from {input_path} ...")
    with open(input_path) as f:
        trajectories = json.load(f)
    print(f"  {len(trajectories)} trajectories loaded.")

    # --- extract ---
    qa_pairs = extract_qa_pairs(trajectories)
    print(f"  {len(qa_pairs)} QA pairs extracted.")

    # --- stats ---
    n_success = sum(1 for q in qa_pairs if q["traj_reward"] == 1.0)
    actions = {}
    for q in qa_pairs:
        actions[q["next_action"]] = actions.get(q["next_action"], 0) + 1
    print(
        f"  From successful trajectories: {n_success} / {len(qa_pairs)} pairs")
    print(f"  Unique actions: {len(actions)}")
    for action, count in sorted(actions.items(), key=lambda x: -x[1]):
        print(f"    {action:<40} {count}")

    # --- write ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in qa_pairs:
            f.write(json.dumps(record) + "\n")

    print(
        f"\nSaved → {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
