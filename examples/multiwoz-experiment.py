import asyncio
import json
import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

EXPERIMENTS = [
    {"context_size_k": 8,   "gold_label": "gold1", "position": "beginning", "max_iterations": 10},
    {"context_size_k": 8,   "gold_label": "gold1", "position": "middle",    "max_iterations": 10},
    {"context_size_k": 8,   "gold_label": "gold1", "position": "end",       "max_iterations": 10},
]


def make_rlm(file_name: str, max_iterations: int) -> RLM:
    return RLM(
        backend="azure_openai",
        backend_kwargs={
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            "model_name": "gpt-5",
        },
        environment="local",
        max_iterations=max_iterations,
        max_depth=1,
        logger=RLMLogger(log_dir="./mutlti-logs", file_name=file_name),
        verbose=True,
    )


async def call_with_retry(rlm: RLM, record: dict, max_retries: int = 5, base_delay: float = 10.0):
    loop = asyncio.get_event_loop()
    for attempt in range(max_retries):
        try:
            result = await loop.run_in_executor(
                None,
                lambda r=record: rlm.completion(prompt=r["context"], root_prompt=r["question"]),
            )
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  [retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e} — waiting {delay:.0f}s")
            await asyncio.sleep(delay)


async def run_experiment(exp: dict) -> None:
    context_size_k = exp["context_size_k"]
    gold_label = exp["gold_label"]
    position = exp["position"]
    max_iterations = exp["max_iterations"]

    file_name = f"rlm_iter{max_iterations}_{context_size_k}k_{gold_label}_{position}"
    data_path = os.path.join(
        os.path.dirname(__file__),
        f"../data/multiwoz_{context_size_k}k_{gold_label}_{position}_30.jsonl",
    )

    print(f"\n{'=' * 60}")
    print(f"Experiment: {file_name}")
    print(f"Data:       {data_path}")
    print(f"{'=' * 60}")

    rlm = make_rlm(file_name, max_iterations)

    with open(data_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    for i, record in enumerate(records):
        dialogue_id = record["dialogue_id"]
        print(f"\n[{position}] {i + 1}/{len(records)} {dialogue_id}")
        try:
            result = await call_with_retry(rlm, record)
            print(f"  Q: {record['question']}")
            print(f"  A (model):    {result.response[:120]}")
            print(f"  A (expected): {record['answer']}")
        except Exception as e:
            print(f"  FAILED after all retries: {e} — skipping record")


async def main() -> None:
    for exp in EXPERIMENTS:
        await run_experiment(exp)
    print("\nAll experiments complete.")


if __name__ == "__main__":
    asyncio.run(main())
