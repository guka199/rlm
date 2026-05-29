"""
tau-bench live environment runner — uses RLM as the agent
Mirrors the structure of the multiwoz baseline but drives the real
tau-bench airline env (reset → step loop) instead of replaying
historical trajectories.

Usage:
    python taubench_run.py [--tasks 0 1 2] [--trials 1] [--exp base]
"""

import argparse
import json
import os
import traceback
from datetime import datetime
from dotenv import load_dotenv

from tau_bench.envs import get_env
from tau_bench.types import Action, RESPOND_ACTION_NAME

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

# ---------------------------------------------------------------------------
# Experiment configs  (mirrors your existing EXPERIMENTS list)
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {"name": "depth1_iter10", "max_depth": 1,
        "max_iterations": 10, "model": "gpt-5"},
    {"name": "depth1_iter30", "max_depth": 1,
        "max_iterations": 30, "model": "gpt-5"},
    # Uncomment to add depth-2:
    # {"name": "depth2_iter10", "max_depth": 2, "max_iterations": 10, "model": "gpt-5"},
]

# ---------------------------------------------------------------------------
# Build the RLM prompt from a live tau-bench conversation history
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = [
    "get_user_details",
    "get_reservation_details",
    "search_direct_flight",
    "search_onestop_flight",
    "book_reservation",
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "update_reservation_passengers",
    "send_certificate",
    "list_all_airports",
    "calculate",
    "think",
    "transfer_to_human_agents",
    "respond",
]


def build_prompt(wiki: str, messages: list[dict]) -> str:
    """
    Convert the running message history into a prompt for the RLM.
    messages is the list we accumulate ourselves (role/content dicts).
    """
    lines = []

    # Policy / wiki block (truncated so it doesn't swamp the context)
    if wiki:
        wiki_trunc = wiki[:800] + "..." if len(wiki) > 800 else wiki
        lines.append(f"[POLICY]\n{wiki_trunc}\n")

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "assistant":
            # A previous tool call the agent made
            if msg.get("tool_call"):
                tc = msg["tool_call"]
                lines.append(
                    f"[AGENT ACTION]: {tc['name']}({json.dumps(tc['kwargs'])})")
            elif content:
                lines.append(f"[AGENT]: {content}")
        elif role == "user":
            lines.append(f"[USER]: {content}")
        elif role == "tool":
            result = content
            if len(result) > 300:
                result = result[:300] + "..."
            lines.append(f"[TOOL RESULT ({msg.get('name', '?')})]: {result}")

    lines.append(f"\nAvailable tools: {', '.join(AVAILABLE_TOOLS)}")
    lines.append(
        "\nGiven the conversation so far, what is the agent's NEXT action?"
    )
    lines.append(
        "If you want to call a tool, return FINAL(tool_name)."
        " If you want to reply to the user, return FINAL(respond)."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parse the RLM response into a tau-bench Action
# ---------------------------------------------------------------------------

def parse_action(response: str, tools_info: list[dict]) -> Action:
    """
    Extract a tool name from the RLM response and return a minimal Action.
    For real arg generation you'd need a second pass; here we do best-effort
    argument extraction or return empty kwargs (tau-bench tools tolerate this
    for smoke-testing).
    """
    raw = response.strip().lower()

    # Try FINAL(tool_name) extraction first
    import re
    m = re.search(r"final\(([^)]+)\)", raw)
    tool_name = m.group(1).strip() if m else None

    # Fall back to substring match against known tools
    if not tool_name or tool_name not in AVAILABLE_TOOLS:
        tool_name = next(
            (t for t in AVAILABLE_TOOLS if t.lower() in raw),
            None,
        )

    if tool_name is None or tool_name == RESPOND_ACTION_NAME:
        # Default: respond to the user with whatever the model said
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": response})

    return Action(name=tool_name, kwargs={})


# ---------------------------------------------------------------------------
# Run one task end-to-end with a given RLM config
# ---------------------------------------------------------------------------

MAX_AGENT_TURNS = 30  # hard cap to avoid runaway loops


def run_task(env, task_index: int, exp: dict, log_dir: str) -> dict:
    """
    Reset the env for task_index, run the agent loop, return a result dict.
    """
    reset_resp = env.reset(task_index=task_index)
    observation = reset_resp.observation  # first user message
    task = reset_resp.info.task

    messages: list[dict] = [{"role": "user", "content": observation}]

    logger = RLMLogger(log_dir=log_dir)
    rlm = RLM(
        backend="azure_openai",
        backend_kwargs={
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            "model_name": exp["model"],
        },
        environment="local",
        max_depth=exp["max_depth"],
        max_iterations=exp["max_iterations"],
        logger=logger,
        verbose=True,
    )

    reward = 0.0
    done = False
    turn = 0
    traj_log = []

    while not done and turn < MAX_AGENT_TURNS:
        turn += 1
        prompt = build_prompt(env.wiki, messages)

        print(f"  [turn {turn}] calling RLM...")
        result = rlm.completion(prompt=prompt)
        raw_response = result.response

        action = parse_action(raw_response, env.tools_info)
        print(f"  [turn {turn}] action={action.name}")

        # Record agent turn
        messages.append({
            "role": "assistant",
            "tool_call": {"name": action.name, "kwargs": action.kwargs},
        })

        # Step the environment
        env_resp = env.step(action)
        observation = env_resp.observation
        reward = env_resp.reward
        done = env_resp.done

        # Record tool/user result
        messages.append({
            "role": "tool",
            "name": action.name,
            "content": observation,
        })

        traj_log.append({
            "turn": turn,
            "action": action.name,
            "kwargs": action.kwargs,
            "observation": observation[:300],
            "done": done,
            "reward": reward,
        })

        print(
            f"  [turn {turn}] obs={observation[:80]}... done={done} reward={reward}")

    return {
        "task_id": task_index,
        "experiment": exp["name"],
        "reward": reward,
        "turns": turn,
        "done": done,
        "traj": traj_log,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", type=int, default=None,
                        help="Task indices to run (default: 0–4)")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of trials per task")
    parser.add_argument("--exp", type=str, default=None,
                        help="Run a single experiment by name (default: all)")
    parser.add_argument("--log-dir", type=str, default="./tau-logs")
    args = parser.parse_args()

    task_ids = args.tasks if args.tasks is not None else list(range(5))
    exps = (
        [e for e in EXPERIMENTS if e["name"] == args.exp]
        if args.exp else EXPERIMENTS
    )
    if not exps:
        raise ValueError(f"Unknown experiment: {args.exp}")

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    results_path = os.path.join(args.log_dir, f"results_{timestamp}.jsonl")
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"Tasks:       {task_ids}")
    print(f"Experiments: {[e['name'] for e in exps]}")
    print(f"Trials:      {args.trials}")
    print(f"Logging to:  {args.log_dir}\n")

    all_results = []

    for trial in range(args.trials):
        for task_index in task_ids:
            for exp in exps:
                run_id = f"task{task_index}_trial{trial}_{exp['name']}"
                log_dir = os.path.join(args.log_dir, run_id)

                # Skip-if-exists checkpoint (same pattern as taubench_run.py)
                done_marker = os.path.join(log_dir, "result.json")
                if os.path.exists(done_marker):
                    print(f"Skipping {run_id} (already done)")
                    with open(done_marker) as f:
                        all_results.append(json.load(f))
                    continue

                os.makedirs(log_dir, exist_ok=True)
                print(f"\n{'='*60}")
                print(f"  {run_id}")
                print(f"{'='*60}")

                # Fresh env per run (tau-bench pattern from run.py)
                env = get_env(
                    "airline",
                    user_strategy="llm",
                    user_model="gpt-4o",
                    user_provider="azure",
                    task_split="test",
                    task_index=task_index,
                )

                try:
                    result = run_task(env, task_index, exp, log_dir)
                    result["trial"] = trial
                except KeyboardInterrupt:
                    print(
                        f"\nInterrupted on {run_id} — writing partial result")
                    result = {
                        "task_id": task_index,
                        "trial": trial,
                        "experiment": exp["name"],
                        "reward": 0.0,
                        "error": "interrupted",
                    }
                except Exception as e:
                    print(f"Error on {run_id}: {e}")
                    result = {
                        "task_id": task_index,
                        "trial": trial,
                        "experiment": exp["name"],
                        "reward": 0.0,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }

                # Persist per-run result so we can resume
                with open(done_marker, "w") as f:
                    json.dump(result, f, indent=2)

                # Append to rolling JSONL
                with open(results_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

                all_results.append(result)

                status = "✅" if result.get("reward", 0) == 1.0 else "❌"
                print(f"{status} {run_id} reward={result.get('reward', '?')}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    completed = [r for r in all_results if "error" not in r]

    for exp in exps:
        exp_results = [r for r in completed if r["experiment"] == exp["name"]]
        if not exp_results:
            continue
        avg_reward = sum(r["reward"] for r in exp_results) / len(exp_results)
        pass_at_1 = sum(1 for r in exp_results if r["reward"] == 1.0)
        print(
            f"  {exp['name']:25s}  pass@1={pass_at_1}/{len(exp_results)}"
            f"  avg_reward={avg_reward:.3f}"
        )

    errors = [r for r in all_results if "error" in r]
    if errors:
        print(f"\n  {len(errors)} error(s):")
        for r in errors:
            print(
                f"    task{r['task_id']} trial{r['trial']} {r['experiment']}: {r['error']}")

    print(f"\nFull results: {results_path}")


if __name__ == "__main__":
    main()
