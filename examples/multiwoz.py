import itertools
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
    gold_percent: float | None = None,
) -> str:
    if rng is None:
        rng = random.Random()

    target_chars = target_tokens * CHARS_PER_TOKEN

    if gold_percent is not None:
        gold_target_chars = int(target_chars * gold_percent)
        gold_parts = []
        accumulated_gold = 0
        while accumulated_gold < gold_target_chars:
            gold_parts.append(gold_context)
            accumulated_gold += len(gold_context) + len(CONVERSATION_SEPARATOR)
        gold_block = CONVERSATION_SEPARATOR.join(gold_parts)
    else:
        gold_block = gold_context

    filler_budget = target_chars - len(gold_block)

    gold_service_set = set(gold_services)
    candidates = [
        r for r in filler_pool
        if r["dialogue_id"] != gold_dialogue_id
        and not (set(r.get("services", [])) & gold_service_set)
    ]
    rng.shuffle(candidates)

    filler_convos = []
    accumulated = 0
    for record in itertools.cycle(candidates):
        if accumulated >= filler_budget:
            break
        filler_convos.append(record["context"])
        accumulated += len(record["context"]) + len(CONVERSATION_SEPARATOR)

    if position == "beginning":
        parts = [gold_block] + filler_convos
    elif position == "end":
        parts = filler_convos + [gold_block]
    else:  # middle
        mid = len(filler_convos) // 2
        parts = filler_convos[:mid] + [gold_block] + filler_convos[mid:]

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
    gold_percentages: list[float] | None = None,
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

    percentages = gold_percentages if gold_percentages is not None else [None]

    for tokens in target_tokens:
        for gold_pct in percentages:
            for position in positions:
                if gold_pct is None:
                    filename = f"multiwoz_qa_{tokens // 1000}k_{position}.jsonl"
                else:
                    pct_label = f"gold{int(gold_pct * 100)}pct"
                    filename = f"multiwoz_qa_{tokens // 1000}k_{pct_label}_{position}.jsonl"
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
                            gold_percent=gold_pct,
                        )
                        out.write(json.dumps({
                            "dialogue_id": record["dialogue_id"],
                            "question": record["question"],
                            "answer": record["answer"],
                            "context": expanded_context,
                            "target_tokens": tokens,
                            "position": position,
                            "gold_percent": gold_pct,
                        }) + "\n")
                print(f"Wrote {len(gold_records)} records to {out_path}")


def sample_hotel_dialogues(
    output_path: str,
    n: int = 30,
    seed: int = 42,
    dataset=None,
) -> None:
    if dataset is None:
        dataset = load_dataset("pfb30/multi_woz_v22", trust_remote_code=True)

    candidates = []
    for split in ["train", "validation", "test"]:
        for dialogue in dataset[split]:
            if "hotel" in dialogue["services"]:
                candidates.append(dialogue)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as out:
        for dialogue in selected:
            out.write(json.dumps({
                "dialogue_id": dialogue["dialogue_id"],
                "context": flatten_dialogue(dialogue["turns"]),
                "services": dialogue["services"],
                "question": "",
                "answer": "",
            }) + "\n")

    print(f"Wrote {len(selected)} hotel dialogues to {output_path}")


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
    gold_path_30 = os.path.join(os.path.dirname(__file__), "../data/multiwoz_qa_30.jsonl")
    filler_path_30 = os.path.join(os.path.dirname(__file__), "../data/multiwoz_filler_30.jsonl")
    output_dir = os.path.join(os.path.dirname(__file__), "../data")

    print("Rebuilding expanded datasets (512k, 30 samples)...")
    build_expanded_datasets(gold_path_30, filler_path_30, output_dir, target_tokens=[512000])
