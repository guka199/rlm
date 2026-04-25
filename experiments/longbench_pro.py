"""
LongBench Pro experiments - Andrew
4 experiments x 3 questions = 12 traces total
Modified for RLM signature: root_prompt = objective, prompt = context
Model: GPT-5 Nano (gpt-5.4-nano)
"""

import json
import os
from datasets import load_dataset
from rlm import RLM
from rlm.logger import RLMLogger

# Experiments as defined in your current vim buffer
EXPERIMENTS = [
    {"max_depth": 1, "max_iterations": 10,  "name": "depth1_iter10"},
    {"max_depth": 1, "max_iterations": 30, "name": "depth1_iter100"},
    {"max_depth": 2, "max_iterations": 10,  "name": "depth2"},
    {"max_depth": 5, "max_iterations": 10,  "name": "depth3"},
]

def build_root_prompt(question, choices):
    """
    The root_prompt acts as the persistent objective for the RLM controller.
    """
    return f"""You are answering a multiple choice question. Use the provided context to find the answer.

Question: {question}

Choices:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Final Instruction: Identify the correct answer based on the context provided in the workspace. 
Return your final answer as a single letter (A, B, C, or D) using FINAL_VAR('answer').
"""

def main():
    os.makedirs("./logs", exist_ok=True)

    print("Loading LongBench Pro dataset...")
    ds = load_dataset("THUDM/LongBench-v2", split="train")

    # Filtering for long context questions as per your experiment design
    questions = [
        ex for ex in ds
        if ex.get("length") == "long"
    ][:3]

    print(f"\nUsing {len(questions)} questions")
    for i, q in enumerate(questions):
        print(f"  Q{i+1}: domain={q['domain']} | difficulty={q['difficulty']}")

    results_summary = []

    for q_idx, example in enumerate(questions):
        context = example["context"]
        question = example["question"]
        choices = [example["choice_A"], example["choice_B"], example["choice_C"], example["choice_D"]]
        ground_truth = example["answer"]

        # Context wrapper for the workspace
        context_prompt = f"<context>\n{context}\n</context>"
        root_prompt = build_root_prompt(question, choices)

        print(f"\n{'='*60}")
        print(f"Question {q_idx+1}: {question[:120]}...")
        print(f"Ground truth: {ground_truth}")
        print(f"{'='*60}")

        for exp in EXPERIMENTS:
            log_dir = f"./logs/q{q_idx+1}_{exp['name']}"

            if os.path.exists(log_dir) and os.listdir(log_dir):
                print(f"Skipping {log_dir} (already exists)")
                continue

            os.makedirs(log_dir, exist_ok=True)
            logger = RLMLogger(log_dir=log_dir)

            print(f"\n--- Experiment: {exp['name']} | Model: gpt-5.4-nano ---")

            rlm = RLM(
                backend="azure_openai",
                backend_kwargs={"model_name": "gpt-5-nano"}, # Switched from gpt4
                max_depth=exp["max_depth"],
                max_iterations=exp["max_iterations"],
                logger=logger,
                verbose=True,
            )

            # prompt (context) comes first, root_prompt (question) second
            result = rlm.completion(
                prompt=context_prompt, 
                root_prompt=root_prompt
            )

            prediction = result.response.strip()
            correct = prediction.upper() == ground_truth.upper()

            print(f"Predicted: {prediction} | Correct: {correct}")

            results_summary.append({
                "question": q_idx + 1,
                "domain": example["domain"],
                "experiment": exp["name"],
                "predicted": prediction,
                "ground_truth": ground_truth,
                "correct": correct,
            })

    # Save summary to disk
    with open("./logs/results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'='*60}\nFINAL RESULTS SUMMARY\n{'='*60}")
    for r in results_summary:
        status = "✓" if r["correct"] else "✗"
        print(f"{status} Q{r['question']} | {r['experiment']} | {r['predicted']} (Truth: {r['ground_truth']})")

if __name__ == "__main__":
    main()
