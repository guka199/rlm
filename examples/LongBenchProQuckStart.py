import os
import random
import string

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger
from datasets import load_dataset

dataset = load_dataset("zai-org/LongBench-v2", split="train")
load_dotenv()
print("AZURE_OPENAI_API_KEY:", os.getenv("AZURE_OPENAI_API_KEY"))
print("AZURE_OPENAI_ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("AZURE_OPENAI_API_VERSION:", os.getenv("AZURE_OPENAI_API_VERSION"))
print("AZURE_OPENAI_DEPLOYMENT:", os.getenv("AZURE_OPENAI_DEPLOYMENT"))

# Generate a large text file with a hidden secret number
secret_number = random.randint(10_000_000, 99_999_999)
filler_lines = ["".join(random.choices(string.ascii_lowercase + " ", k=120)) for _ in range(50_000)]
insert_at = random.randint(len(filler_lines) // 3, 2 * len(filler_lines) // 3)
filler_lines.insert(insert_at, f"SECRET_NUMBER={secret_number}")
haystack = "\n".join(filler_lines)


if False: #GPT 4o if true and gpt 5 nano otherwise
    rlm = RLM(
        backend="azure_openai",
        backend_kwargs={
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            # "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT"),  
            "model_name": "gpt-4o",
        },
        environment="local",
        max_iterations=10,
        logger=RLMLogger(log_dir="./logs"),
        verbose=True,
    )
else:
    rlm = RLM(
    backend="azure_openai",
    backend_kwargs={
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        # "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT"),  
        "model_name": "gpt-5-nano",
    },
    environment="local",
    max_iterations=10,
    logger=RLMLogger(log_dir="./logs_LongBenchPro"),
    verbose=True,
    )
# result = rlm.completion(
#     "The context contains ~50k lines of random text with a single line "
#     "matching the pattern SECRET_NUMBER=<digits>. Find and return ONLY the "
#     f"numeric value.\n\n{haystack}"
# )

# print(f"\nModel found: {result.response}")
# print(f"Actual number: {secret_number}")
# print(f"Correct: {str(secret_number) in result.response}")
# --- Run evaluation on LongBench ---
results = []

num_samples = 10  # change if needed

for i in range(num_samples):
    print(f"\n{'='*60}\n  QUESTION {i + 1} / {num_samples}\n{'='*60}")
    sample = dataset[i]

    context = sample["context"]
    question = sample["question"]
    choice_A = sample["choice_A"]
    choice_B = sample["choice_B"]
    choice_C = sample["choice_C"]
    choice_D = sample["choice_D"]
    ground_truth = sample["answer"]  # single letter: "A", "B", "C", or "D"

    prompt = f"""
    You are given a context and a multiple-choice question. Read the context carefully and choose the best answer.

    Context:
    {context}

    Question:
    {question}

    Choices:
    A. {choice_A}
    B. {choice_B}
    C. {choice_C}
    D. {choice_D}

    Reply with only the letter of the correct answer (A, B, C, or D):
    """

    result = rlm.completion(prompt)

    prediction = result.response.strip().upper()
    # Extract just the first letter in case the model returns "A." or "A) ..."
    predicted_letter = prediction[0] if prediction else ""

    correct = predicted_letter == ground_truth

    results.append({
        "id": i,
        "question": question,
        "prediction": predicted_letter,
        "ground_truth": ground_truth,
        "correct": correct,
        "trajectory": result.metadata
    })

    print(f"\n--- Sample {i} ---")
    print(f"Q: {question}")
    print(f"Pred: {predicted_letter}")
    print(f"GT: {ground_truth}")
    print(f"Correct: {correct}")