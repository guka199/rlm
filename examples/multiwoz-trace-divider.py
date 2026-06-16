import json
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), "../mutlti-logs/newPromptShreyas.jsonl")
DATA_FILE = os.path.join(os.path.dirname(__file__), "../data/multiwoz_qa_512k_middle.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../mutlti-logs/512k_iter30_split")

with open(LOG_FILE) as f:
    entries = [json.loads(line) for line in f if line.strip()]

# Separate the single metadata header, then split on iteration resets to 1
metadata_entry = next((e for e in entries if e.get("type") == "metadata"), None)
iteration_entries = [e for e in entries if e.get("type") != "metadata"]

groups = []
current = []
for entry in iteration_entries:
    if entry.get("iteration") == 1 and current:
        groups.append(current)
        current = []
    current.append(entry)
if current:
    groups.append(current)

# Prepend metadata to each group so every file has the run config
if metadata_entry:
    groups = [[metadata_entry] + g for g in groups]

# Load dialogue IDs in order to name the output files (falls back to sample_N if file missing)
records = []
if os.path.exists(DATA_FILE):
    with open(DATA_FILE) as f:
        records = [json.loads(line) for line in f if line.strip()]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, group in enumerate(groups):
    if i < len(records):
        sample_id = records[i]["dialogue_id"].replace(".json", "")
    else:
        sample_id = f"sample_{i}"
    out_path = os.path.join(OUTPUT_DIR, f"{sample_id}.jsonl")
    with open(out_path, "w") as out:
        for entry in group:
            out.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(group)} entries -> {out_path}")
