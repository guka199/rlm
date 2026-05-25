import json
import os
import random
from typing import Literal

from datasets import load_dataset

CHARS_PER_TOKEN = 4
CONVERSATION_SEPARATOR = "\n\n---\n\n"


def flatten_dialogue(turns: dict) -> str:
    """Flatten parallel turn arrays into a single conversation string."""
    speakers = turns["speaker"]
    utterances = turns["utterance"]
    lines = []
    for speaker, utterance in zip(speakers, utterances):
        label = "USER" if speaker == 0 else "SYSTEM"
        lines.append(f"{label}: {utterance}")
    return "\n".join(lines)


def build_id_to_services(dataset) -> dict[str, list[str]]:
    lookup = {}
    for split in ["train", "validation", "test"]:
        for dialogue in dataset[split]:
            lookup[dialogue["dialogue_id"]] = dialogue["services"]
    return lookup


def load_filler_pool(filler_jsonl_path: str) -> list[dict]:
    with open(filler_jsonl_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_filler_context(
    gold_context: str,
    gold_dialogue_id: str,
    gold_services: list[str],
    filler_pool: list[dict],
    target_tokens: int,
    position: Literal["beginning", "middle", "end"],
    rng: random.Random | None = None,
) -> str:
    if rng is None:
        rng = random.Random()

    target_chars = target_tokens * CHARS_PER_TOKEN
    gold_chars = len(gold_context)
    filler_budget = target_chars - gold_chars

    gold_service_set = set(gold_services)
    candidates = [
        r for r in filler_pool
        if r["dialogue_id"] != gold_dialogue_id
        and not (set(r.get("services", [])) & gold_service_set)
    ]
    rng.shuffle(candidates)

    filler_convos = []
    accumulated = 0
    for record in candidates:
        if accumulated >= filler_budget:
            break
        filler_convos.append(record["context"])
        accumulated += len(record["context"]) + len(CONVERSATION_SEPARATOR)

    if position == "beginning":
        parts = [gold_context] + filler_convos
    elif position == "end":
        parts = filler_convos + [gold_context]
    else:  # middle
        mid = len(filler_convos) // 2
        parts = filler_convos[:mid] + [gold_context] + filler_convos[mid:]

    return CONVERSATION_SEPARATOR.join(parts)


def build_filler_jsonl(gold_jsonl_path: str, output_path: str, dataset=None) -> None:
    with open(gold_jsonl_path) as f:
        gold_ids = {json.loads(line)["dialogue_id"] for line in f if line.strip()}

    if dataset is None:
        dataset = load_dataset("pfb30/multi_woz_v22", trust_remote_code=True)

    written = 0
    with open(output_path, "w") as out:
        for dialogue in dataset["train"]:
            dialogue_id = dialogue.get("dialogue_id", "")
            if dialogue_id in gold_ids:
                continue
            context = flatten_dialogue(dialogue["turns"])
            out.write(json.dumps({
                "dialogue_id": dialogue_id,
                "services": dialogue["services"],
                "context": context,
            }) + "\n")
            written += 1

    print(f"Wrote {written} filler conversations to {output_path}")


def enrich_gold_with_services(gold_jsonl_path: str, id_to_services: dict[str, list[str]]) -> None:
    with open(gold_jsonl_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    with open(gold_jsonl_path, "w") as out:
        for record in records:
            record["services"] = id_to_services.get(record["dialogue_id"], [])
            out.write(json.dumps(record) + "\n")

    print(f"Enriched {len(records)} gold records with services")


def build_expanded_datasets(
    gold_jsonl_path: str,
    filler_jsonl_path: str,
    output_dir: str,
    target_tokens: list[int] | None = None,
    positions: list[Literal["beginning", "middle", "end"]] | None = None,
    seed: int = 42,
) -> None:
    if target_tokens is None:
        target_tokens = [8000, 33000]
    if positions is None:
        positions = ["beginning", "middle", "end"]

    with open(gold_jsonl_path) as f:
        gold_records = [json.loads(line) for line in f if line.strip()]

    filler_pool = load_filler_pool(filler_jsonl_path)

    os.makedirs(output_dir, exist_ok=True)

    for tokens in target_tokens:
        for position in positions:
            filename = f"multiwoz_qa_{tokens // 1000}k_{position}.jsonl"
            out_path = os.path.join(output_dir, filename)
            rng = random.Random(seed)
            with open(out_path, "w") as out:
                for record in gold_records:
                    expanded_context = build_filler_context(
                        gold_context=record["context"],
                        gold_dialogue_id=record["dialogue_id"],
                        gold_services=record.get("services", []),
                        filler_pool=filler_pool,
                        target_tokens=tokens,
                        position=position,
                        rng=rng,
                    )
                    out.write(json.dumps({
                        "dialogue_id": record["dialogue_id"],
                        "question": record["question"],
                        "answer": record["answer"],
                        "context": expanded_context,
                        "target_tokens": tokens,
                        "position": position,
                    }) + "\n")
            print(f"Wrote {len(gold_records)} records to {out_path}")


def main():
    dataset = load_dataset("pfb30/multi_woz_v22", trust_remote_code=True)

    train = dataset["train"]
    for i, dialogue in enumerate(train):
        if i >= 10:
            break
        dialogue_id = dialogue.get("dialogue_id", f"dialogue_{i}")
        context = flatten_dialogue(dialogue["turns"])
        print(f"--- {dialogue_id} ---")
        print(context)
        print()


if __name__ == "__main__":
    gold_path = os.path.join(os.path.dirname(__file__), "../data/multiwoz_qa.jsonl")
    filler_path = os.path.join(os.path.dirname(__file__), "../data/multiwoz_filler.jsonl")
    output_dir = os.path.join(os.path.dirname(__file__), "../data")

    print("Loading HuggingFace dataset...")
    dataset = load_dataset("pfb30/multi_woz_v22", trust_remote_code=True)

    print("Building id→services lookup...")
    id_to_services = build_id_to_services(dataset)

    print("Rebuilding filler JSONL with services...")
    build_filler_jsonl(gold_path, filler_path, dataset=dataset)

    print("Enriching gold records with services...")
    enrich_gold_with_services(gold_path, id_to_services)

    print("Rebuilding expanded datasets...")
    build_expanded_datasets(gold_path, filler_path, output_dir)
