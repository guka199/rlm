"""
build_longcontext_dataset.py
============================
Pads τ-bench airline QA pairs with retail filler context to hit 8k and 66k
token targets, then creates gold-at-start and gold-at-end variants.

Each record preserves the original airline QA fields so results from
run_longcontext_experiments.py can be joined back to the baseline by qa_id.

Output schema per record:
    qa_id          str   — original QA id (e.g. "0_0_6"), unchanged for joining
    traj_id        str
    task_id        int
    turn_index     int
    step           int
    traj_reward    float
    ground_truth   str   — next tool the agent should call (was "next_action")
    padded_context list  — full padded conversation (system + turns + filler)
    position       str   — "start" (gold first) or "end" (gold last)
    total_tokens   int   — approx token count of the non-system padded turns
    context_size   str   — "8k" or "66k"

Output files:
    data/longcontext_8k.jsonl   — both positions at ~8k tokens
    data/longcontext_66k.jsonl  — both positions at ~66k tokens

Usage:
    python build_longcontext_dataset.py
    python build_longcontext_dataset.py --successful-only
    python build_longcontext_dataset.py --num-samples 10
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Token counting — tiktoken if available, else 4 chars ≈ 1 token fallback
# ---------------------------------------------------------------------------

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
    print("Token counting: tiktoken cl100k_base", file=sys.stderr)
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text) // 4
    print(
        "Token counting: character fallback (len // 4) — install tiktoken for accuracy",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# Paths and targets
# ---------------------------------------------------------------------------

AIRLINE_QA_PATH = Path("data/taubench_airline_qa.jsonl")
RETAIL_TRAJ_PATH = Path("tau-bench/historical_trajectories/gpt-4o-retail.json")
OUTPUT_DIR = Path("data")

TOKEN_TARGETS = {"8k": 8_000, "66k": 66_000}


# ---------------------------------------------------------------------------
# Turn serialization (matches build_context in run_longcontext_experiments.py)
# ---------------------------------------------------------------------------

def turn_to_text(turn: dict) -> str:
    role = turn.get("role", "unknown")
    content = turn.get("content") or ""

    if role == "system":
        return ""
    elif role == "user":
        return f"USER: {content}"
    elif role == "tool":
        return f"TOOL RESULT ({turn.get('name', 'tool')}): {content}"
    elif role == "assistant":
        tool_calls = turn.get("tool_calls") or []
        if tool_calls:
            fn = tool_calls[0]["function"]["name"]
            args = tool_calls[0]["function"].get("arguments", "{}")
            return f"AGENT called: {fn}({args})"
        return f"AGENT: {content}"
    return ""


def turns_token_count(turns: list[dict]) -> int:
    """Token count of all non-system turns as they will appear in the prompt."""
    return sum(count_tokens(turn_to_text(t)) for t in turns if t.get("role") != "system")


# ---------------------------------------------------------------------------
# Load retail filler pool
# ---------------------------------------------------------------------------

def load_filler_pool(retail_path: Path) -> list[dict]:
    """All non-system turns from reward=1.0 retail trajectories."""
    with open(retail_path) as f:
        data = json.load(f)
    turns = []
    for traj in data:
        if traj.get("reward") != 1.0:
            continue
        for turn in traj.get("traj", []):
            if turn.get("role") != "system":
                turns.append(turn)
    return turns


# ---------------------------------------------------------------------------
# Padding logic
# ---------------------------------------------------------------------------

def collect_filler(
    filler_pool: list[dict],
    start_idx: int,
    token_gap: int,
) -> tuple[list[dict], int]:
    """
    Pull turns from filler_pool (cycling) until accumulated tokens >= token_gap.
    Returns (filler_turns, next_start_idx).
    """
    n = len(filler_pool)
    filler_turns: list[dict] = []
    accumulated = 0
    idx = start_idx
    while accumulated < token_gap:
        turn = filler_pool[idx % n]
        accumulated += count_tokens(turn_to_text(turn))
        filler_turns.append(turn)
        idx += 1
    return filler_turns, idx


def assemble_context(
    gold_context: list[dict],
    filler_turns: list[dict],
    position: str,
) -> list[dict]:
    """
    position="start": system + gold_turns + filler_turns
    position="end":   system + filler_turns + gold_turns
    """
    system_turns = [t for t in gold_context if t.get("role") == "system"]
    gold_turns = [t for t in gold_context if t.get("role") != "system"]
    if position == "start":
        return system_turns + gold_turns + filler_turns   # gold first, filler at end
    return system_turns + filler_turns + gold_turns       # filler first, gold at end


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airline-qa",      default=str(AIRLINE_QA_PATH))
    parser.add_argument("--retail-traj",     default=str(RETAIL_TRAJ_PATH))
    parser.add_argument("--output-dir",      default=str(OUTPUT_DIR))
    parser.add_argument("--successful-only", action="store_true",
                        help="Only QA pairs from reward=1.0 airline trajectories")
    parser.add_argument("--num-samples",     type=int, default=None,
                        help="Cap QA pairs (smoke test)")
    args = parser.parse_args()

    airline_path = Path(args.airline_qa)
    retail_path = Path(args.retail_traj)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load airline QA pairs
    qa_pairs = []
    with open(airline_path) as f:
        for line in f:
            try:
                qa_pairs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"Loaded {len(qa_pairs)} airline QA pairs")

    if args.successful_only:
        qa_pairs = [q for q in qa_pairs if q["traj_reward"] == 1.0]
        print(f"Filtered to {len(qa_pairs)} from successful trajectories")

    if args.num_samples:
        qa_pairs = qa_pairs[:args.num_samples]
        print(f"Capped to {len(qa_pairs)} samples")

    filler_pool = load_filler_pool(retail_path)
    print(
        f"Retail filler pool: {len(filler_pool)} turns from successful trajectories")

    for context_size, token_target in TOKEN_TARGETS.items():
        output_path = output_dir / f"longcontext_{context_size}.jsonl"
        records = []
        filler_idx = 0  # advances globally so adjacent QA pairs get varied filler

        for i, qa in enumerate(qa_pairs):
            gold_tokens = turns_token_count(qa["context"])
            token_gap = max(0, token_target - gold_tokens)

            # Collect filler once; reuse for both positions so they're comparable
            filler_turns, filler_idx = collect_filler(
                filler_pool, filler_idx, token_gap)
            filler_tokens = turns_token_count(filler_turns)
            total_tokens = gold_tokens + filler_tokens

            for position in ("start", "end"):
                padded = assemble_context(
                    qa["context"], filler_turns, position)
                records.append({
                    # original fields — preserved for joining back to baseline results
                    "qa_id":          qa["qa_id"],
                    "traj_id":        qa["traj_id"],
                    "task_id":        qa["task_id"],
                    "turn_index":     qa["turn_index"],
                    "step":           qa["step"],
                    "traj_reward":    qa["traj_reward"],
                    "ground_truth":   qa["next_action"],
                    # padded context and metadata
                    "padded_context": padded,
                    "position":       position,
                    "total_tokens":   total_tokens,
                    "context_size":   context_size,
                })

            if (i + 1) % 200 == 0:
                print(f"  [{context_size}] {i+1}/{len(qa_pairs)} processed ...")

        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        start_recs = [r for r in records if r["position"] == "start"]
        avg_gold = sum(turns_token_count(q["context"])
                       for q in qa_pairs) / len(qa_pairs)
        avg_total = sum(r["total_tokens"]
                        for r in start_recs) / len(start_recs)
        min_total = min(r["total_tokens"] for r in start_recs)
        max_total = max(r["total_tokens"] for r in start_recs)
        size_kb = output_path.stat().st_size / 1024

        print(f"\n{context_size} → {output_path}  ({size_kb:,.0f} KB)")
        print(
            f"  Records       : {len(records)} ({len(start_recs)} start + {len(start_recs)} end)")
        print(f"  Token target  : {token_target:,}")
        print(f"  Avg gold toks : {avg_gold:,.0f}")
        print(
            f"  Avg total toks: {avg_total:,.0f}  [min={min_total:,}, max={max_total:,}]")


if __name__ == "__main__":
    main()
