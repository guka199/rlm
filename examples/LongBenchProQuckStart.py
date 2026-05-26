import os
import json
from dotenv import load_dotenv
from datasets import load_dataset
from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

dataset = load_dataset("zai-org/LongBench-v2", split="train")

# Indices of multidoc_qa samples (256k, English) identified from dataset inspection
# # multidoc_qa_samples = [2, 8, 15, 16, 18, 19]
# multidoc_qa_samples = [16, 18, 19]


# for i in range(100):
#     s = dataset[i]
#     if "multi" in str(s.get("domain", "")).lower() and str(s.get("difficulty", "")).lower() == "easy":
#         print(f"  idx={i}  difficulty={s.get('difficulty')}  Q: {s.get('question', '')[:120]}")
# breakpoint()

# SIMPLE MULTIDOC QUESTONS
multidoc_qa_samples = [8, 18, 27, 28, 40, 52, 66, 69]
multidoc_qa_samples = [69]
selected_samples = [{"sample_idx": i, **dataset[i]} for i in multidoc_qa_samples]


print(f"\nRunning {len(selected_samples)} multidoc QA samples:")
for s in selected_samples:
    print(f"  idx={s['sample_idx']}  domain={s.get('domain')}  sub_domain={s.get('sub_domain')}  length={s.get('length')}")


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def run_rlm_evaluation(
    sample: dict,
    max_depth: int = 1,
    max_iterations: int = 10,
    exp_name: str = "base",
    model_name: str = "gpt-5",
    log_dir: str = "./logs_LongBenchPro",
) -> dict:
    """
    Run a single RLM completion on a LongBench-v2 sample.

    Parameters
    ----------
    sample        : dataset row (must have context/question/choice_*/answer + sample_idx)
    max_depth     : RLM max_depth (1 = single sub-call)
    max_iterations: maximum iterations per completion
    exp_name      : label for this experiment run (used in log filename)
    model_name    : Azure OpenAI model name
    log_dir       : root directory for RLMLogger output
    """
    sample_idx = sample.get("sample_idx", "?")
    log_name = f"sample{sample_idx}_{exp_name}_d{max_depth}_i{max_iterations}"

    rlm = RLM(
        backend="azure_openai",
        backend_kwargs={
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            "model_name": model_name,
        },
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        logger=RLMLogger(log_dir=log_dir, file_name=log_name),
        verbose=True,
    )

    context      = sample["context"]
    question     = sample["question"]
    choice_A     = sample["choice_A"]
    choice_B     = sample["choice_B"]
    choice_C     = sample["choice_C"]
    choice_D     = sample["choice_D"]
    ground_truth = sample["answer"]

    prompt = f"""You are given a context and a multiple-choice question. \
Read the context carefully and choose the best answer.
Question:
{question}


Choices:
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}

Context:
{context}


Reply with only the letter of the correct answer (A, B, C, or D):"""

    result = rlm.completion(prompt)

    prediction = result.response.strip().upper().replace('"', "")
    predicted_letter = prediction[0] if prediction else ""
    correct = predicted_letter == ground_truth

    return {
        "sample_idx":     sample_idx,
        "experiment":     exp_name,
        "max_depth":      max_depth,
        "max_iterations": max_iterations,
        "question":       question,
        "prediction":     predicted_letter,
        "ground_truth":   ground_truth,
        "correct":        correct,
        "log_file":       log_name,
        "metadata":       result.metadata,
    }


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {"name": "base", "max_depth": 1, "max_iterations": 10},
    {"name": "exp1", "max_depth": 1, "max_iterations": 30},
    # {"name": "exp2", "max_depth": 2, "max_iterations": 10},
    # {"name": "exp3", "max_depth": 5, "max_iterations": 10},
]

all_results: list[dict] = []

for sample in selected_samples:
    for exp in EXPERIMENTS:
        print(
            f"\n{'='*60}\n"
            f"  sample_idx={sample['sample_idx']}  {exp['name'].upper()}  "
            f"depth={exp['max_depth']}  iters={exp['max_iterations']}\n"
            f"{'='*60}"
        )
        res = run_rlm_evaluation(
            sample,
            max_depth=exp["max_depth"],
            max_iterations=exp["max_iterations"],
            exp_name=exp["name"],
        )
        all_results.append(res)

        print(f"  Q: {res['question'][:120]}...")
        print(f"  Pred: {res['prediction']}  GT: {res['ground_truth']}  Correct: {res['correct']}")
        print(f"  Log: ./logs_LongBenchPro/{res['log_file']}_*.jsonl")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
header = f"{'idx':>4} {'Exp':<6} {'depth':>5} {'iters':>5} {'Pred':>4} {'GT':>2} {'OK':>3}"
print(header)
print("-" * len(header))
for r in all_results:
    print(
        f"{r['sample_idx']:>4} {r['experiment']:<6} "
        f"{r['max_depth']:>5} {r['max_iterations']:>5} "
        f"{r['prediction']:>4} {r['ground_truth']:>2} {'✓' if r['correct'] else '✗':>3}"
    )

output_path = "./logs_LongBenchPro/experiment_results.json"
os.makedirs("./logs_LongBenchPro", exist_ok=True)
with open(output_path, "w") as f:
    serializable = [{k: v for k, v in r.items() if k != "metadata"} for r in all_results]
    json.dump(serializable, f, indent=2)
print(f"\nResults saved to {output_path}")
