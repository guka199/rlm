import json
import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

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
            "model_name": "gpt-5",
        },
        environment="local",
        max_iterations=30,
        max_depth = 1,
        logger=RLMLogger(log_dir="./mutlti-logs"),
        verbose=True,
    )

jsonl_path = os.path.join(os.path.dirname(__file__), "../data/multiwoz_qa_8k_middle.jsonl")

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
