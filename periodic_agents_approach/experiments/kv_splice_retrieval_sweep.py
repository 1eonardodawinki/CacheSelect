# Experiment 2c: does Experiment 2b's finding hold up across more than one
# random example? Experiment 2b (kv_splice_retrieval.py) found real
# evidence that splicing stale K,V can corrupt the model's answer -- but
# from a single trace, a single seed, a single route. That's one anecdote,
# not a rate. This reruns the same splice-and-generate retrieval probe
# across several independent seeds and aggregates the results, so instead
# of "it happened once" we get "it happens X% of the time at distance N."
#
# Classification is now built into the script itself (not a one-off
# post-hoc analysis, per Experiment 2b's own "next step" note): every
# output is checked against the full table of real check values for that
# trace and labelled CORRECT (matches the true latest check), WRONG_CHECK_N
# (a real value, just the wrong check), or INVALID (matches no real value
# at all -- the clearest sign of outright corruption).
#
# Also runs a control alongside the real (stale) splice: a SELF-splice,
# where the "stale" K,V for a check is pulled from activation t+1's own
# true cache at that same check -- mathematically a no-op (attention is
# causal, so chunking a forward pass into [prefix][suffix] and rejoining
# via the KV cache should reproduce the single-pass result exactly, modulo
# floating-point noise). If the middle-distance danger zone shows up in
# the self-splice control too, it's a generic effect of chunked processing
# or of that position in a long list (e.g. "lost in the middle"), nothing
# to do with cache staleness. If it only shows up with genuinely stale
# (activation t) content, that confirms it's a real caching effect.
#
# Same local, no-vLLM, no-GPU-cluster setup as Experiments 1/2/2b.
# Usage (from the repo root):
#     python -m periodic_agents_approach.experiments.kv_splice_retrieval_sweep

import csv
import os
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from periodic_agents_approach.agents.travel_planner_trace import DEFAULT_ROUTES, generate_trace
from periodic_agents_approach.experiments.kv_contamination import find_route_block, pick_device
from periodic_agents_approach.experiments.kv_splice_generation import get_check_token_spans, run_generation, run_spliced_generation
from periodic_agents_approach.experiments.kv_splice_retrieval import build_retrieval_messages

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FOCUS_ROUTE = DEFAULT_ROUTES[0]
WINDOW_SIZE = 15
SEEDS = list(range(1, 9))  # 8 independent traces -- adjust for a faster/slower sweep
DETAIL_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "experiment2c_sweep_detail.csv")
SUMMARY_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "experiment2c_sweep_summary.csv")


def get_all_real_values(activation, route):
    # Every check's real dollar value for this route in this activation,
    # keyed by check index -- used to classify a generated answer as
    # correct / a different real check / outright invalid.
    user_content = next(m["content"] for m in activation.messages if m["role"] == "user")
    _, block_text = find_route_block(user_content, route)
    return {int(idx): val for idx, val in re.findall(r"check (\d+): (\$[\d,]+\.\d+)", block_text)}


def classify_output(output_text, real_values, latest_check_idx):
    match = re.search(r"\$[\d,]+\.\d+", output_text)
    if not match:
        return "NO_NUMBER"
    val = match.group(0)
    for idx, v in real_values.items():
        if v == val:
            return "CORRECT" if idx == latest_check_idx else f"WRONG_CHECK_{idx}"
    return "INVALID"


def run_one_seed(model, tokenizer, device, seed):
    trace = generate_trace(num_activations=2, window_size=WINDOW_SIZE, stride=1, seed=seed)
    activation_t, activation_t1 = trace.activations[0], trace.activations[1]
    routes = trace.tickers_or_entities
    latest_check_idx = WINDOW_SIZE

    real_values = get_all_real_values(activation_t1, FOCUS_ROUTE)
    ground_truth_price = real_values[latest_check_idx]

    messages_t = build_retrieval_messages(activation_t, routes, FOCUS_ROUTE)
    messages_t1 = build_retrieval_messages(activation_t1, routes, FOCUS_ROUTE)
    input_ids_t, spans_t = get_check_token_spans(tokenizer, messages_t, FOCUS_ROUTE)
    input_ids_t1, spans_t1 = get_check_token_spans(tokenizer, messages_t1, FOCUS_ROUTE)
    shared_checks = sorted(set(spans_t) & set(spans_t1))

    with torch.no_grad():
        cache_t_out = model(torch.tensor([input_ids_t], device=device), use_cache=True)
        cache_t1_out = model(torch.tensor([input_ids_t1], device=device), use_cache=True)
    cache_t = cache_t_out.past_key_values  # source of STALE values (the real condition)
    cache_t1_full = cache_t1_out.past_key_values  # source of TRUE values (the control)

    baseline_text = run_generation(model, tokenizer, device, input_ids_t1)
    baseline_class = classify_output(baseline_text, real_values, latest_check_idx)

    rows = []
    for check_idx in shared_checks:
        stale_text = run_spliced_generation(
            model, tokenizer, device, input_ids_t1, spans_t1, check_idx, cache_t, spans_t
        )
        stale_class = classify_output(stale_text, real_values, latest_check_idx)

        # Control: "splice" with the check's own true value from t+1 itself.
        # Same chunked-forward-pass mechanics, mathematically a no-op.
        control_text = run_spliced_generation(
            model, tokenizer, device, input_ids_t1, spans_t1, check_idx, cache_t1_full, spans_t1
        )
        control_class = classify_output(control_text, real_values, latest_check_idx)

        rows.append({
            "seed": seed,
            "check_index": check_idx,
            "distance_from_drop": check_idx,
            "ground_truth_price": ground_truth_price,
            "baseline_output": baseline_text,
            "baseline_class": baseline_class,
            "stale_spliced_output": stale_text,
            "stale_spliced_class": stale_class,
            "stale_changed_from_baseline": stale_text.strip() != baseline_text.strip(),
            "control_spliced_output": control_text,
            "control_spliced_class": control_class,
            "control_changed_from_baseline": control_text.strip() != baseline_text.strip(),
        })
    return rows


def _pct(rows, class_key, predicate):
    n = len(rows)
    return round(100 * sum(1 for r in rows if predicate(r[class_key])) / n, 1)


def summarize(all_rows):
    # One row per distance, aggregated across every seed that had a check
    # at that distance (all seeds do here, since WINDOW_SIZE is fixed, but
    # written generically in case that changes). Reports stale-splice
    # (the real condition) and control-splice (the no-op check) side by
    # side, so the two are directly comparable at every distance.
    by_distance = defaultdict(list)
    for r in all_rows:
        by_distance[r["distance_from_drop"]].append(r)

    is_invalid = lambda c: c == "INVALID"
    is_wrong_check = lambda c: c.startswith("WRONG_CHECK")
    is_correct = lambda c: c == "CORRECT"

    summary = []
    for distance in sorted(by_distance):
        rows = by_distance[distance]
        n = len(rows)
        summary.append({
            "distance_from_drop": distance,
            "n_seeds": n,
            "stale_pct_correct": _pct(rows, "stale_spliced_class", is_correct),
            "stale_pct_wrong_check": _pct(rows, "stale_spliced_class", is_wrong_check),
            "stale_pct_invalid": _pct(rows, "stale_spliced_class", is_invalid),
            "stale_pct_changed": round(100 * sum(1 for r in rows if r["stale_changed_from_baseline"]) / n, 1),
            "control_pct_correct": _pct(rows, "control_spliced_class", is_correct),
            "control_pct_wrong_check": _pct(rows, "control_spliced_class", is_wrong_check),
            "control_pct_invalid": _pct(rows, "control_spliced_class", is_invalid),
            "control_pct_changed": round(100 * sum(1 for r in rows if r["control_changed_from_baseline"]) / n, 1),
        })
    return summary


def main():
    device = pick_device()
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME} (tokenizer + weights) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    all_rows = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        rows = run_one_seed(model, tokenizer, device, seed)
        baseline_class = rows[0]["baseline_class"] if rows else "?"
        print(f"baseline: {rows[0]['baseline_output']!r} -> {baseline_class}")
        for r in rows:
            print(f"  check {r['check_index']:>2} (dist {r['distance_from_drop']:>2}): "
                  f"stale={r['stale_spliced_output']!r} -> {r['stale_spliced_class']:<14} | "
                  f"control={r['control_spliced_output']!r} -> {r['control_spliced_class']}")
        all_rows.extend(rows)

    os.makedirs(os.path.dirname(DETAIL_CSV), exist_ok=True)
    with open(DETAIL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {DETAIL_CSV}")

    summary = summarize(all_rows)
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {len(summary)} rows to {SUMMARY_CSV}\n")

    print("                |  STALE (real condition)               |  CONTROL (self-splice, should be a no-op)")
    print("distance | n | %correct | %wrong-check | %invalid | %changed | %correct | %wrong-check | %invalid | %changed")
    for s in summary:
        print(f"{s['distance_from_drop']:>8} | {s['n_seeds']:>1} | "
              f"{s['stale_pct_correct']:>8} | {s['stale_pct_wrong_check']:>12} | "
              f"{s['stale_pct_invalid']:>8} | {s['stale_pct_changed']:>8} | "
              f"{s['control_pct_correct']:>8} | {s['control_pct_wrong_check']:>12} | "
              f"{s['control_pct_invalid']:>8} | {s['control_pct_changed']:>8}")


if __name__ == "__main__":
    main()
