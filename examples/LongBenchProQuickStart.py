import os
from dotenv import load_dotenv
from datasets import load_dataset
from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

dataset = load_dataset("zai-org/LongBench-v2", split="train")
multidoc_qa_sample_256k = [7, 10, 12, 18, 28, 31, 46, 53, 57, 58]
multidoc_qa_sample_medium = [3, 8, 11, 17, 29, 36, 42, 44, 50, 61]

multidoc_qa_sample_short = [1, 27, 40, 51, 52, 66, 67, 72, 86, 88]

# Indices of multidoc_qa samples (256k, English) identified from dataset inspection
# multidoc_qa_samples = [2, 8, 15, 16, 18, 19]
multidoc_qa_samples = [16, 18, 19]


# for i in range(100):
#     s = dataset[i]
#     if "multi" in str(s.get("domain", "")).lower():
#         print(f"  idx={i}  difficulty={s.get('difficulty')}  Q: {s.get('question', '')[:120]}")


# # Finding long context questions
# for i in range(100):
#     s = dataset[i]
#     print(str(s.get("length", "")))
#     if "long" in str(s.get("length", "")).lower() and str(s.get("difficulty", "")).lower() == "easy":
#         print(f"  idx={i}  difficulty={s.get('difficulty')}  Q: {s.get('question', '')[:120]}")
#         multidoc_qa_sample_256k.append(i)
#     if len(multidoc_qa_sample_256k) >= 10:
#         break

# # finding medium context questions
# for i in range(100):
#     s = dataset[i]
#     print(str(s.get("length", "")))
#     if "medium" in str(s.get("length", "")).lower() and str(s.get("difficulty", "")).lower() == "easy":
#         print(f"  idx={i}  difficulty={s.get('difficulty')}  Q: {s.get('question', '')[:120]}")
#         multidoc_qa_sample_medium.append(i)
#     if len(multidoc_qa_sample_medium) >= 10:
#         break

# # finding short context questions
# for i in range(100):
#     s = dataset[i]
#     print(str(s.get("length", "")))
#     if "short" in str(s.get("length", "")).lower() and str(s.get("difficulty", "")).lower() == "easy":
#         print(f"  idx={i}  difficulty={s.get('difficulty')}  Q: {s.get('question', '')[:120]}")
#         multidoc_qa_sample_short.append(i)
#     if len(multidoc_qa_sample_short) >= 10:
#         break



# print(multidoc_qa_sample_medium)
# print(multidoc_qa_sample_short)
# breakpoint()

# print(f"\nmultidoc_qa_sample_256k = {multidoc_qa_sample_256k}")

# SIMPLE MULTIDOC QUESTONS
multidoc_qa_samples = [8, 18, 27, 28, 40, 52, 66, 69]
multidoc_qa_samples = multidoc_qa_sample_medium
multidoc_qa_samples = multidoc_qa_sample_short

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
        print(f"  Q: {res['question'][:120]}...")
        print(f"  Pred: {res['prediction']}  GT: {res['ground_truth']}  Correct: {res['correct']}")
        print(f"  Log: logs_LongBenchPro/{res['log_file']}_*.jsonl")
