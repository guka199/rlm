"""
τ-bench experiments - Andrew
3 experiments x 15 samples = 45 traces total
"""

import json
import os
from rlm import RLM
from rlm.logger import RLMLogger
from taubench_dataset import TauBenchDataset

EXPERIMENTS = [
    {"max_depth": 1, "max_iterations": 10, "name": "base_depth1_iter10"},
    {"max_depth": 1, "max_iterations": 30, "name": "exp1_depth1_iter30"},
    {"max_depth": 2, "max_iterations": 10, "name": "exp2_depth2_iter10"},
]

NUM_SAMPLES = 15
DOMAIN = "airline"


def build_prompt(qa_pair: dict) -> str:
    lines = []

    for turn in qa_pair["context"]:
        role = turn["role"]

        if role == "system":
            policy = turn.get("content", "")
            if len(policy) > 500:
                policy = policy[:500] + "..."
            lines.append(f"[POLICY]\n{policy}\n")

        elif role == "user":
            lines.append(f"[USER]: {turn['content']}")

        elif role == "assistant":
            if turn.get("tool_calls"):
                tc = turn["tool_calls"][0]
                lines.append(
                    f"[AGENT ACTION]: {tc['function']['name']}({tc['function']['arguments']})")
            elif turn.get("content"):
                lines.append(f"[AGENT]: {turn['content']}")

        elif role == "tool":
            content = turn.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"[TOOL RESULT ({turn.get('name', '?')})]: {content}")

    # Available tools for this domain
    available_tools = [
        "get_user_details",
        "get_reservation_details",
        "search_direct_flight",
        "search_onestop_flight",
        "book_reservation",
        "cancel_reservation",
        "update_reservation_flights",
        "update_reservation_baggages",
        "update_reservation_passengers",
        "send_certificate",
        "list_all_airports",
        "calculate",
        "think",
        "transfer_to_human_agents",
    ]

    lines.append(
        f"\nAvailable tools: {', '.join(available_tools)}\n"
        f"\nGiven the conversation above, what is the agent's NEXT tool call?\n"
        f"Return the tool name as your final answer using FINAL_VAR('tool_name')."
    )

    return "\n".join(lines)


def main():
    os.makedirs("./logs", exist_ok=True)

    print(f"Loading τ-bench dataset (domain={DOMAIN})...")
    ds = TauBenchDataset(domain=DOMAIN, seed=42)
    ds.load_data()

    samples = ds.sample_data(n=NUM_SAMPLES)
    print(f"Using {len(samples)} samples\n")
    for i, s in enumerate(samples):
        print(f"  Sample {i+1}: task_id={s['task_id']} | turn={s['turn_index']} | "
              f"next_action={s['next_action']} | reward={s['reward']}")

    results_summary = []

    for s_idx, sample in enumerate(samples):
        prompt = build_prompt(sample)
        ground_truth = sample["next_action"]

        print(f"\n{'='*60}")
        print(
            f"Sample {s_idx+1}: task_id={sample['task_id']} | turn={sample['turn_index']}")
        print(f"Ground truth: {ground_truth}")
        print(f"{'='*60}")

        for exp in EXPERIMENTS:
            log_dir = f"./logs/s{s_idx+1}_{exp['name']}"

            if os.path.exists(log_dir) and os.listdir(log_dir):
                print(f"Skipping {log_dir} (already exists)")
                results_summary.append({
                    "sample": s_idx + 1,
                    "task_id": sample["task_id"],
                    "turn_index": sample["turn_index"],
                    "experiment": exp["name"],
                    "predicted": "SKIPPED",
                    "ground_truth": ground_truth,
                    "correct": None,
                })
                continue

            os.makedirs(log_dir, exist_ok=True)
            logger = RLMLogger(log_dir=log_dir)

            print(f"\n--- Experiment: {exp['name']} ---")

            rlm = RLM(
                backend="azure_openai",
                backend_kwargs={"model_name": "gpt-5"},
                max_depth=exp["max_depth"],
                max_iterations=exp["max_iterations"],
                logger=logger,
                verbose=True,
            )

            result = rlm.completion(prompt)

            prediction = result.response.strip().lower()
            correct = prediction == ground_truth.lower()

            print(f"Predicted: {prediction}")
            print(f"Correct:   {correct}")

            results_summary.append({
                "sample": s_idx + 1,
                "task_id": sample["task_id"],
                "turn_index": sample["turn_index"],
                "context_turns": len(sample["context"]),
                "experiment": exp["name"],
                "predicted": prediction,
                "ground_truth": ground_truth,
                "correct": correct,
                "traj_reward": sample["reward"],
            })

    with open("./logs/results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results_summary:
        if r["correct"] is None:
            status = "-"
        elif r["correct"]:
            status = "✓"
        else:
            status = "✗"
        print(f"{status} S{r['sample']} (task={r['task_id']}) | {r['experiment']} | "
              f"predicted={r['predicted']} | truth={r['ground_truth']}")

    completed = [r for r in results_summary if r["correct"] is not None]
    correct_count = sum(1 for r in completed if r["correct"])
    print(f"\nTotal correct: {correct_count}/{len(completed)} completed runs")

    # Per-experiment breakdown
    print(f"\n--- Per-experiment accuracy ---")
    for exp in EXPERIMENTS:
        exp_results = [r for r in completed if r["experiment"] == exp["name"]]
        if exp_results:
            acc = sum(
                1 for r in exp_results if r["correct"]) / len(exp_results)
            print(
                f"  {exp['name']}: {acc:.0%} ({sum(1 for r in exp_results if r['correct'])}/{len(exp_results)})")


if __name__ == "__main__":
    main()
