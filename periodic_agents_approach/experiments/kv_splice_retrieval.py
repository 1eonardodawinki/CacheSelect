# Experiment 2b: same splice-and-generate mechanism as kv_splice_generation.py,
# but with a different, much simpler behavioural probe. Experiment 2 asked
# the model to make a threshold judgement (ALERT vs STATUS: nominal) -- a
# task it turned out to be unreliable at even with a fully correct,
# unspliced prompt (missed an unmistakable 60% crash three times), which
# made "does splicing change the answer" untestable: a model that ignores
# real signal will obviously also ignore injected noise.
#
# This version sidesteps that weakness by asking something the model should
# reliably get right at baseline: "what was the most recent price check for
# <route>?" -- plain retrieval, not threshold reasoning. If splicing an
# older, supposedly-irrelevant check corrupts even this, that's clean
# evidence contamination matters; if it doesn't, it's a trustworthy
# positive result, since (unlike Experiment 2) baseline correctness here
# isn't in doubt to begin with.
#
# Same price-table content as kv_splice_generation.py, so the check token
# spans and splice mechanics are identical -- only the system prompt and
# the trailing instruction change.
#
# Usage (from the repo root):
#     python -m periodic_agents_approach.experiments.kv_splice_retrieval

import csv
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from periodic_agents_approach.agents.travel_planner_trace import DEFAULT_ROUTES, TOOLS_DESCRIPTION, generate_trace
from periodic_agents_approach.experiments.kv_contamination import find_route_block, pick_device
from periodic_agents_approach.experiments.kv_splice_generation import get_check_token_spans, run_generation, run_spliced_generation

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FOCUS_ROUTE = DEFAULT_ROUTES[0]
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "experiment2b_kv_splice_retrieval.csv")

RETRIEVAL_SYSTEM_PROMPT = (
    "You are a travel monitoring agent. You track flight prices for a set of "
    "routes ({routes_str}). " + TOOLS_DESCRIPTION + " "
    "When asked for the most recent price check for a route, answer with only "
    "the dollar amount, in the format $X.XX, and nothing else."
)

OLD_TRAILING_INSTRUCTION = "\n\nReview the latest check for each route and report any threshold breaches."


def build_retrieval_messages(activation, routes, route):
    # Same user content (the price tables) as the real agent prompt -- only
    # the system prompt and the trailing question change, so check token
    # spans located the same way as kv_splice_generation.py stay valid.
    system_content = RETRIEVAL_SYSTEM_PROMPT.format(routes_str=", ".join(routes))
    original_user = next(m["content"] for m in activation.messages if m["role"] == "user")
    assert OLD_TRAILING_INSTRUCTION in original_user, "Trailing instruction text changed upstream -- update this."
    new_user = original_user.replace(
        OLD_TRAILING_INSTRUCTION,
        f"\n\nWhat was the most recent price check for {route}?",
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": new_user},
    ]


def get_ground_truth_price(activation, route, latest_check_idx):
    user_content = next(m["content"] for m in activation.messages if m["role"] == "user")
    _, block_text = find_route_block(user_content, route)
    match = re.search(rf"  check {latest_check_idx}: (\$[\d,]+\.\d+)", block_text)
    assert match, f"Could not find check {latest_check_idx} in the {route} block."
    return match.group(1)


def main():
    device = pick_device()
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME} (tokenizer + weights) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    # No anomaly needed -- retrieval correctness doesn't depend on whether
    # the price happens to be a big move, just on stating whatever it
    # actually is. window_size=15 kept for continuity with Experiment 2
    # (more shared checks -> more distance data points).
    trace = generate_trace(num_activations=2, window_size=15, stride=1, seed=1)
    activation_t, activation_t1 = trace.activations[0], trace.activations[1]
    routes = trace.tickers_or_entities
    latest_check_idx = 15  # window_size -- the newest check in t+1's window
    ground_truth_price = get_ground_truth_price(activation_t1, FOCUS_ROUTE, latest_check_idx)
    print(f"Ground truth: latest {FOCUS_ROUTE} price = {ground_truth_price}")

    messages_t = build_retrieval_messages(activation_t, routes, FOCUS_ROUTE)
    messages_t1 = build_retrieval_messages(activation_t1, routes, FOCUS_ROUTE)

    input_ids_t, spans_t = get_check_token_spans(tokenizer, messages_t, FOCUS_ROUTE)
    input_ids_t1, spans_t1 = get_check_token_spans(tokenizer, messages_t1, FOCUS_ROUTE)
    shared_checks = sorted(set(spans_t) & set(spans_t1))
    print(f"Shared checks: {shared_checks}")

    with torch.no_grad():
        cache_t_out = model(torch.tensor([input_ids_t], device=device), use_cache=True)
    cache_t = cache_t_out.past_key_values

    print("Running baseline (unspliced) generation for t+1 ...")
    baseline_text = run_generation(model, tokenizer, device, input_ids_t1)
    baseline_correct = ground_truth_price in baseline_text
    print(f"Baseline output: {baseline_text!r} -> correct={baseline_correct}")

    if not baseline_correct:
        print(
            "\nWARNING: the model got the UNSPLICED retrieval baseline wrong -- "
            "same problem as Experiment 2, just at the retrieval task instead of "
            "the alert task. Results below are not meaningful if so.\n"
        )

    rows = []
    for check_idx in shared_checks:
        spliced_text = run_spliced_generation(
            model, tokenizer, device, input_ids_t1, spans_t1, check_idx, cache_t, spans_t
        )
        spliced_correct = ground_truth_price in spliced_text
        answer_changed = spliced_text.strip() != baseline_text.strip()
        print(f"check {check_idx} (distance {check_idx}): spliced output: {spliced_text!r} "
              f"-> correct={spliced_correct}, changed={answer_changed}")
        rows.append({
            "check_index": check_idx,
            "distance_from_drop": check_idx,
            "baseline_output": baseline_text,
            "baseline_correct": baseline_correct,
            "spliced_output": spliced_text,
            "spliced_correct": spliced_correct,
            "answer_changed": answer_changed,
        })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")

    flips = sum(1 for r in rows if not r["spliced_correct"] and r["baseline_correct"])
    print(f"\n{flips}/{len(rows)} spliced checks flipped a correct baseline answer to incorrect.")


if __name__ == "__main__":
    main()
