import json
import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

max_iterations = 10
context_size_k = 8
gold_label = "gold1"
position = "beginning"

file_name = f"rlm_iter{max_iterations}_{context_size_k}k_{gold_label}_{position}"
jsonl_path = os.path.join(os.path.dirname(__file__), f"../data/multiwoz_{context_size_k}k_{gold_label}_{position}_30.jsonl")

if False:  # GPT 4o if true and gpt 5 nano otherwise
    rlm = RLM(
        backend="azure_openai",
        backend_kwargs={
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            "model_name": "gpt-4o",
        },
        environment="local",
        max_iterations=max_iterations,
        logger=RLMLogger(log_dir="./logs", file_name=file_name),
        verbose=True,
    )
else:
    rlm = RLM(
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

with open(jsonl_path) as f:
    records = [json.loads(line) for line in f if line.strip()]

for record in records:
    result = rlm.completion(
        prompt=record['context'],
        root_prompt=record['question']
    )

    print(f"\n--- {record['dialogue_id']} ---")
    print(f"Question:        {record['question']}")
    print(f"Model answer:    {result.response}")
    print(f"Expected answer: {record['answer']}")
    # print(prompt)

print(f"\nDone. Log written to: {rlm.logger.log_file_path}")
