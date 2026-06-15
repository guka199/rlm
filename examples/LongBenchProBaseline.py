import json
import os

from datasets import load_dataset
from dotenv import load_dotenv

from rlm.clients.azure_openai import AzureOpenAIClient

load_dotenv()

multidoc_qa_sample_long   = [7, 10, 12, 18, 28, 31, 46, 53, 57, 58]
multidoc_qa_sample_medium = [3, 8, 11, 17, 29, 36, 42, 44, 50, 61]
multidoc_qa_sample_short  = [1, 27, 40, 51, 52, 66, 67, 72, 86, 88]

multidoc_qa_samples = sorted(set(
    multidoc_qa_sample_long + multidoc_qa_sample_medium + multidoc_qa_sample_short
))

TASK_PROMPT = (
    "You are given a context and a multiple-choice question. "
    "Read the context carefully and choose the best answer.\n\n"
    "Question:\n{question}\n\n"
    "Choices:\n"
    "A. {choice_A}\n"
    "B. {choice_B}\n"
    "C. {choice_C}\n"
    "D. {choice_D}\n\n"
    "Context:\n{context}\n\n"
    "Reply with only the letter of the correct answer (A, B, C, or D):"
)

REVIEW_PROMPT = (
    "Review your answer carefully. Verify that it is grounded in the conversations provided and not fabricated. "
    "Check that you found the relevant conversation and not an unrelated one. "
    "Check that all parts of the question have been answered."
)

CORRECTION_PROMPT = (
    "Produce a corrected final solution based on the critique."
)


def run_baseline(output_path: str) -> None:
    dataset = load_dataset("zai-org/LongBench-v2", split="train")

    # Skip samples already present in the output file
    already_done: set[int] = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    already_done.add(json.loads(line)["sample_idx"])
    print(f"Already done: {sorted(already_done)}")

    remaining = [i for i in multidoc_qa_samples if i not in already_done]
    print(f"To run: {remaining} ({len(remaining)} samples)\n")
    records = [{"sample_idx": i, **dataset[i]} for i in remaining]

    if not records:
        print("Nothing to run — all samples already done.")
        return

    client = AzureOpenAIClient(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        model_name="gpt-5",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "a") as out:
        for i, record in enumerate(records):
            print(f"[{i + 1}/{len(records)}] sample_idx={record['sample_idx']}")

            prompt = TASK_PROMPT.format(
                question=record["question"],
                choice_A=record["choice_A"],
                choice_B=record["choice_B"],
                choice_C=record["choice_C"],
                choice_D=record["choice_D"],
                context=record["context"],
            )
            messages = [{"role": "user", "content": prompt}]

            try:
                response = client.completion(messages)
                usage = client.get_last_usage()

                prediction = response.strip().upper().replace('"', "")
                predicted_letter = prediction[0] if prediction else ""
                ground_truth = record["answer"]

                trace = {
                    "sample_idx":   record["sample_idx"],
                    "question":     record["question"],
                    "ground_truth": ground_truth,
                    "prediction":   predicted_letter,
                    "correct":      predicted_letter == ground_truth,
                    "step1": {
                        "prompt":        prompt,
                        "response":      response,
                        "input_tokens":  usage.total_input_tokens,
                        "output_tokens": usage.total_output_tokens,
                    },
                }
                print(f"  pred={predicted_letter}  gt={ground_truth}  correct={predicted_letter == ground_truth}")
            except Exception as e:
                token_limit_exceeded = any(
                    kw in str(e).lower() for kw in ("token", "context length", "max_tokens", "too long")
                )
                error_msg = "TOKEN_LIMIT_EXCEEDED" if token_limit_exceeded else f"ERROR: {e}"
                print(f"  {error_msg}")
                trace = {
                    "sample_idx":   record["sample_idx"],
                    "question":     record["question"],
                    "ground_truth": record["answer"],
                    "prediction":   None,
                    "correct":      False,
                    "error":        error_msg,
                }

            out.write(json.dumps(trace) + "\n")
            out.flush()

    print(f"\nDone. Traces written to {output_path}")


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "logs_LongBenchPro/baseline_gpt5.jsonl")
    run_baseline(output_path)
