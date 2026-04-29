"""
LongBench Pro experiments - Andrew
4 experiments x 3 questions = 12 traces total
"""

import json
import os
from datasets import load_dataset
from rlm import RLM
from rlm.logger import RLMLogger

EXPERIMENTS = [
    {"max_depth": 1, "max_iterations": 50,  "name": "depth1_iter10"},
    {"max_depth": 1, "max_iterations": 30,  "name": "depth1_iter30"},
    {"max_depth": 2, "max_iterations": 10,  "name": "depth2"},
    {"max_depth": 5, "max_iterations": 10,  "name": "depth5_recursive"},
]

def build_prompt(question, choices, context):
    return f"""You are answering a multiple choice question. Use the following context to find the answer.

<context>
{context}
</context>

Question: {question}

Choices:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Return your final answer as a single letter (A, B, C, or D) using FINAL_VAR('answer').
"""

def main():
    os.makedirs("./logs", exist_ok=True)

    print("Loading LongBench Pro dataset...")
    ds = load_dataset("THUDM/LongBench-v2", split="train")

    all_domains = set(ex["domain"] for ex in ds)
    print(f"Available domains: {all_domains}")

    questions = [
        ex for ex in ds
        if ex.get("length") == "long"
    ][:3]

    print(f"\nUsing {len(questions)} questions")
    for i, q in enumerate(questions):
        print(f"  Q{i+1}: domain={q['domain']} | sub_domain={q['sub_domain']} | difficulty={q['difficulty']}")

    results_summary = []

    for q_idx, example in enumerate(questions):
        context = example["context"]
        question = example["question"]
        choices = [example["choice_A"], example["choice_B"], example["choice_C"], example["choice_D"]]
        ground_truth = example["answer"]

        print(f"\n{'='*60}")
        print(f"Question {q_idx+1}: {question[:120]}...")
        print(f"Ground truth: {ground_truth}")
        print(f"{'='*60}")

        for exp in EXPERIMENTS:
            log_dir = f"./logs/q{q_idx+1}_{exp['name']}"

            if os.path.exists(log_dir) and os.listdir(log_dir):
                print(f"Skipping {log_dir} (already exists)")
                results_summary.append({
                    "question": q_idx + 1,
                    "domain": example["domain"],
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

            prompt = build_prompt(question, choices, context)
            result = rlm.completion(prompt)

            prediction = result.response.strip()
            correct = prediction.upper() == ground_truth.upper()

            print(f"Predicted: {prediction}")
            print(f"Correct: {correct}")

            results_summary.append({
                "question": q_idx + 1,
                "domain": example["domain"],
                "experiment": exp["name"],
                "predicted": prediction,
                "ground_truth": ground_truth,
                "correct": correct,
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
        print(f"{status} Q{r['question']} ({r['domain']}) | {r['experiment']} | predicted={r['predicted']} | truth={r['ground_truth']}")

    completed = [r for r in results_summary if r["correct"] is not None]
    correct_count = sum(1 for r in completed if r["correct"])
    print(f"\nTotal correct: {correct_count}/{len(completed)} completed runs")

if __name__ == "__main__":
    main()
