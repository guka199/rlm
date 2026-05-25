import json

JSONL_PATH = "baseline-logs/8k_middle_singleCall.jsonl"

with open(JSONL_PATH, "r") as f:
    for i, line in enumerate(f):
        entry = json.loads(line)

        print(f"\n{'='*80}")
        print(f"Entry {i+1} | dialogue_id: {entry['dialogue_id']}")
        print(f"Question: {entry['question']}")
        print(f"Expected answer: {entry['answer']}")
        print(f"\n--- Step 1: Initial response ---")
        print(entry["step1"]["response"])
        # print(f"\n--- Step 2: Review response ---")
        # print(entry["step2"]["response"])
        # print(f"\n--- Step 3: Final solution ---")
        # print(entry["step3"]["response"])
        # print(f"Expected answer: {entry['answer']}")

        input("\nPress Enter for next entry (Ctrl+C to quit)...")
