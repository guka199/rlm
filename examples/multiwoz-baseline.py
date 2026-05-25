import json
import os

from dotenv import load_dotenv

from rlm.clients.azure_openai import AzureOpenAIClient

load_dotenv()

TASK_PROMPT = (
    "You are a helpful assistant. Answer the question below using only the information "
    "provided in the conversations.\n\n"
    "Conversations:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

REVIEW_PROMPT = (
    "Review your answer carefully. Verify that it is grounded in the conversations provided and not fabricated. "
    "Check that you found the relevant conversation and not an unrelated one. "
    "Check that all parts of the question have been answered."
)

CORRECTION_PROMPT = (
    "Produce a corrected final solution based on the critique."
)


def run_baseline(jsonl_path: str, output_path: str) -> None:
    client = AzureOpenAIClient(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        model_name="gpt-5",
    )

    with open(jsonl_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as out:
        for i, record in enumerate(records):
            print(f"[{i + 1}/{len(records)}] {record['dialogue_id']}")

            prompt = TASK_PROMPT.format(
                context=record["context"],
                question=record["question"],
            )
            messages = [{"role": "user", "content": prompt}]

            response = client.completion(messages)
            usage = client.get_last_usage()

            # # Step 2: review
            # review_messages = messages + [
            #     {"role": "assistant", "content": response},
            #     {"role": "user", "content": REVIEW_PROMPT},
            # ]
            # review_response = client.completion(review_messages)
            # usage_2 = client.get_last_usage()

            # # Step 3: correction
            # correction_messages = review_messages + [
            #     {"role": "assistant", "content": review_response},
            #     {"role": "user", "content": CORRECTION_PROMPT},
            # ]
            # response_2 = client.completion(correction_messages)
            # usage_3 = client.get_last_usage()

            trace = {
                "dialogue_id": record["dialogue_id"],
                "question": record["question"],
                "answer": record["answer"],
                "target_tokens": record.get("target_tokens"),
                "position": record.get("position"),
                "step1": {
                    "prompt": prompt,
                    "response": response,
                    "input_tokens": usage.total_input_tokens,
                    "output_tokens": usage.total_output_tokens,
                },
            }
            out.write(json.dumps(trace) + "\n")
            out.flush()
            print(f"  response: {response[:120]}...")

    print(f"\nDone. Traces written to {output_path}")


if __name__ == "__main__":
    jsonl_path = os.path.join(os.path.dirname(__file__), "../data/multiwoz_qa_8k_middle.jsonl")
    output_path = os.path.join(os.path.dirname(__file__), "../baseline-logs/8k_middle_singleCall.jsonl")
    run_baseline(jsonl_path, output_path)
