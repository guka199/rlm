"""
run_baseline.py
===============
Plain LM baseline — no RLM, no REPL, just a direct GPT-4o completion call
for each QA pair. Used to identify whether failures are RLM-specific or
just hard questions in general.

Usage:
    export AZURE_OPENAI_API_KEY=...
    export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o
    export AZURE_OPENAI_API_VERSION=2024-02-15-preview

    python run_baseline.py --num-samples 10 --successful-only
    python run_baseline.py --num-samples 10 --successful-only --model gpt-5  # when available
"""

import argparse
import json
import os
import sys
from pathlib import Path

import openai

RESULTS_DIR = Path("results")

SYSTEM_PROMPT = """\
You are a next-action classifier for an airline customer service agent.
Given the conversation history, output the single next tool the agent should call.
Respond with ONLY the tool name — one word, nothing else.

Available tools:
get_user_details, get_reservation_details, search_direct_flight,
search_onestop_flight, list_all_airports, calculate, think,
book_reservation, cancel_reservation, update_reservation_flights,
update_reservation_baggages, update_reservation_passengers,
send_certificate, transfer_to_human_agents\
"""


def configure_azure() -> openai.AzureOpenAI:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get(
        "AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    if not api_key:
        print("ERROR: AZURE_OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    if not endpoint:
        print("ERROR: AZURE_OPENAI_ENDPOINT not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Azure OpenAI configured")
    print(f"  endpoint   : {endpoint}")
    print(f"  deployment : {deployment}")
    print(f"  api_version: {api_version}")

    client = openai.AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint.rstrip("/"),
        api_version=api_version,
        azure_deployment=deployment,
    )
    return client, deployment


def build_prompt(context: list[dict]) -> str:
    conv_lines = []
    for turn in context:
        role = turn.get("role", "unknown")
        content = turn.get("content") or ""

        if role == "system":
            continue
        elif role == "user":
            conv_lines.append(f"USER: {content}")
        elif role == "tool":
            tool_name = turn.get("name", "tool")
            conv_lines.append(f"TOOL RESULT ({tool_name}): {content}")
        elif role == "assistant":
            tool_calls = turn.get("tool_calls") or []
            if tool_calls:
                fn = tool_calls[0]["function"]["name"]
                args = tool_calls[0]["function"].get("arguments", "{}")
                conv_lines.append(f"AGENT called: {fn}({args})")
            else:
                conv_lines.append(f"AGENT: {content}")

    conv = "\n".join(conv_lines)
    return (
        "AIRLINE CUSTOMER SERVICE CONVERSATION:\n"
        "----------------------------------------\n"
        f"{conv}\n"
        "----------------------------------------\n"
        "What is the single next tool the agent should call? One tool name only."
    )


def load_done(results_path: Path) -> set[str]:
    done = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["qa_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",           default="data/taubench_airline_qa.jsonl")
    parser.add_argument("--successful-only", action="store_true")
    parser.add_argument("--num-samples",     type=int, default=None)
    parser.add_argument("--model",           default=None,
                        help="Override deployment name (e.g. gpt-5)")
    args = parser.parse_args()

    client, deployment = configure_azure()
    model = args.model or deployment

    # --- load dataset ---
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.", file=sys.stderr)
        sys.exit(1)

    qa_pairs = []
    with open(input_path) as f:
        for line in f:
            try:
                qa_pairs.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if args.successful_only:
        qa_pairs = [q for q in qa_pairs if q["traj_reward"] == 1.0]
        print(f"Filtered to successful trajectories: {len(qa_pairs)} pairs")

    if args.num_samples:
        qa_pairs = qa_pairs[:args.num_samples]
        print(f"Capped to {len(qa_pairs)} samples")

    # --- run ---
    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / f"baseline_{model.replace('-', '_')}.jsonl"
    done = load_done(results_path)
    remaining = [q for q in qa_pairs if q["qa_id"] not in done]

    print(f"\nBaseline: {model}")
    print(f"  Total QA pairs : {len(qa_pairs)}")
    print(f"  Already done   : {len(done)}")
    print(f"  To run         : {len(remaining)}")
    print(f"  Results        → {results_path}")

    correct = sum(1 for q in qa_pairs if q["qa_id"] in done and
                  any(json.loads(l)["correct"]
                      for l in open(results_path) if q["qa_id"] in l))

    with open(results_path, "a") as out:
        for i, qa in enumerate(remaining):
            prompt = build_prompt(qa["context"])
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=20,
                    temperature=0,
                )
                predicted = response.choices[0].message.content.strip().split()[
                    0]
                error = None
            except Exception as e:
                predicted = ""
                error = str(e)

            correct_flag = predicted == qa["next_action"]

            record = {
                "qa_id":        qa["qa_id"],
                "traj_id":      qa["traj_id"],
                "task_id":      qa["task_id"],
                "trial":        qa["trial"],
                "traj_reward":  qa["traj_reward"],
                "turn_index":   qa["turn_index"],
                "step":         qa["step"],
                "predicted":    predicted,
                "ground_truth": qa["next_action"],
                "correct":      correct_flag,
                "error":        error,
                "model":        model,
                "experiment":   "baseline",
            }
            out.write(json.dumps(record) + "\n")
            out.flush()

            status = "✓" if correct_flag else "✗"
            print(
                f"  [{i+1}/{len(remaining)}] {status}  gt={qa['next_action']:<35} pred={predicted}")

    # --- summary ---
    records = [json.loads(l) for l in open(results_path)]
    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    print(f"\n--- baseline {model} summary ---")
    print(f"Accuracy: {correct}/{total}  ({correct/total:.1%})")


if __name__ == "__main__":
    main()
