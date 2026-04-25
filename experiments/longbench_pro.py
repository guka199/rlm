"""
LongBench Pro experiments - Andrew
Strategy: Question-First (Separated Objective/Context)
Model: GPT-5 Nano (gpt-5-nano)
"""

import json
import os
import time
from datasets import load_dataset
from rlm import RLM
from rlm.logger import RLMLogger

# Configuration for the 4 traces
EXPERIMENTS = [
    {"max_depth": 1, "max_iterations": 10, "name": "depth1_iter10"},
    {"max_depth": 1, "max_iterations": 30, "name": "depth1_iter30"},
    {"max_depth": 2, "max_iterations": 10, "name": "depth2"},
    {"max_depth": 5, "max_iterations": 10, "name": "depth5_recursive"},
]

def build_root_prompt(question, choices):
    """
    This becomes the persistent objective in the RLM's system context.
    Separating this from the context prevents 'Instruction Drift'.
    """
    return f"""Objective: Answer the following multiple-choice question based on the provided context.

Question: {question}

Choices:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Final Instruction: Use the REPL to analyze the source context and identify the correct answer. 
Return your final answer as a single letter (A, B, C, or D) using FINAL_VAR('answer').
"""

def main():
    # Ensure local directories exist to avoid Vim E212 errors
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./experiment", exist_ok=True)

    print("Loading LongBench Pro (v2) dataset...")
    ds = load_dataset("THUDM/LongBench-v2", split="train")

    # Filter for 'long' context questions for high-stress RLM testing
    questions = [
        ex for ex in ds
        if ex.get("length") == "long"
    ][:3]

    print(f"\nUsing {len(questions)} long-context questions.")
    results_summary = []

    for q_idx, example in enumerate(questions):
        context = example["context"]
        question = example["question"]
        choices = [example["choice_A"], example["choice_B"], example["choice_C"], example["choice_D"]]
        ground_truth = example["answer"]

        # Strategy: Separate Objective from Context
        objective = build_root_prompt(question, choices)
        source_context = f"<context>\n{context}\n</context>"

        print(f"\n{'='*60}")
        print(f"Question {q_idx+1}: {question[:100]}...")
        print(f"Ground truth: {ground_truth}")
        print(f"{'='*60}")

        for exp in EXPERIMENTS:
            log_dir = f"./logs/q{q_idx+1}_{exp['name']}"

            if os.path.exists(log_dir) and os.listdir(log_dir):
                print(f"Skipping {exp['name']} (logs already exist)")
                continue

            os.makedirs(log_dir, exist_ok=True)
            logger = RLMLogger(log_dir=log_dir)

            print(f"\n--- Running: {exp['name']} | Model: gpt-5-nano ---")

            # Initialize RLM with GPT-5 Nano
            rlm = RLM(
                backend="azure_openai",
                backend_kwargs={
                    "model_name": "gpt-5-nano",
                    "temperature": 0.0 # Keep it deterministic for experiments
                },
                max_depth=exp["max_depth"],
                max_iterations=exp["max_iterations"],
                logger=logger,
                verbose=True,
            )

            # Updated Library Call: prompt=Data, root_prompt=Objective
            try:
                result = rlm.completion(
                    prompt=source_context, 
                    root_prompt=objective
                )

                prediction = result.response.strip()
                # Use upper() for robustness against model formatting quirks
                correct = prediction.upper() == ground_truth.upper()

                print(f"Predicted: {prediction} | Correct: {correct}")

                results_summary.append({
                    "question": q_idx + 1,
                    "domain": example["domain"],
                    "experiment": exp["name"],
                    "predicted": prediction,
                    "ground_truth": ground_truth,
                    "correct": correct,
                    "time": result.execution_time
                })

            except Exception as e:
                print(f"Error during {exp['name']}: {e}")
                continue

    # Final Save
    output_path = "./logs/results_summary.json"
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE: Results saved to {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
