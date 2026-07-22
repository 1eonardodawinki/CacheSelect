# Experiment 1: how much does a token's TRUE K,V representation change when
# an earlier block ages out of a sliding window? This is the attention-
# contamination question from the interim report's known-issue note --
# under causal self-attention, a token's K,V (layer 2 onward) already
# encodes everything before it, so reusing a cached block after its
# preceding context changed is an approximation, not a guaranteed-correct
# cache hit. This script measures how big that approximation actually is,
# before any of it gets built into vLLM.
#
# Runs entirely locally via plain `transformers` (CPU or Apple MPS) -- no
# vLLM, no CUDA, no GPU cluster. vLLM requires a CUDA GPU and isn't used
# here at all; this only needs a single forward pass per prompt.
#
# Method: generate two consecutive Travel Planner activations with a small
# window (t = checks 0..5, t+1 = checks 1..6 -- check 0 drops off the front,
# check 6 is genuinely new, checks 1..5 are byte-identical content in both,
# just shifted one slot earlier). Run both full prompts through the model
# with use_cache=True, locate each shared check's token span in both
# prompts via character-offset mapping, and compare their K,V vectors
# layer by layer.
#
# V is the primary signal: RoPE never rotates V (see kv_cache_architecture
# artifact), so any V deviation is purely attention-context contamination,
# not a byproduct of checks 1..5 shifting position between the two prompts.
# K is reported too, but conflates contamination with the position/RoPE
# change, so it's expected to deviate more than V -- that gap is itself a
# sanity check that the measurement is working as intended.
#
# Usage (from the repo root):
#     python -m experiments.kv_contamination

import csv
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agents.travel_planner_trace import DEFAULT_ROUTES, generate_trace

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FOCUS_ROUTE = DEFAULT_ROUTES[0]  # keep the first pass to one route -- simplest
                                   # thing that gives a real signal; extending
                                   # to all three routes is a trivial follow-up
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "experiment1_kv_contamination.csv")


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_route_block(user_content, route):
    # The user message is built (see travel_planner_trace.format_price_window)
    # as "ROUTE:\n  check N: $X.XX\n  check N+1: ...", one block per route,
    # blocks separated by blank lines. Grabs one route's block verbatim.
    pattern = re.escape(route) + r":\n((?:  check \d+: \$[\d,]+\.\d+\n)+)"
    match = re.search(pattern, user_content)
    if not match:
        raise ValueError(f"Could not find a block for route {route!r} in the user message.")
    return match.start(1), match.group(1)


def find_check_spans(block_text, block_start_offset):
    # Within one route's block, finds the (start, end) character span of
    # each individual "  check N: $X.XX" line, as absolute offsets into the
    # full user message -- and the check index each span belongs to.
    #
    # Deliberately excludes the trailing newline(s) from the span. The last
    # check in a window is followed by "\n\n" (next route block) while the
    # same check, when it isn't last in the other window, is followed by
    # just "\n" -- an artifact of which check happens to be last, nothing to
    # do with attention contamination. Stopping at the last digit keeps the
    # comparison to content that's genuinely identical between the two
    # windows.
    spans = {}
    for m in re.finditer(r"  check (\d+): \$[\d,]+\.\d+", block_text):
        check_idx = int(m.group(1))
        spans[check_idx] = (block_start_offset + m.start(), block_start_offset + m.end())
    return spans


def char_span_to_token_indices(offset_mapping, char_start, char_end):
    # offset_mapping[i] = (tok_start, tok_end) for token i, both char offsets
    # into the original string. Returns every token index whose span
    # overlaps [char_start, char_end).
    indices = []
    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end:
            continue  # special tokens often report an empty (0,0)-style span
        if tok_start < char_end and tok_end > char_start:
            indices.append(i)
    return indices


def extract_layer_kv(past_key_values, layer_idx):
    # Normalizes across transformers versions. Current (transformers 5.x)
    # Cache objects store per-layer state as past_key_values.layers[i], with
    # .keys/.values attributes; older versions returned a plain
    # tuple-of-(key,value)-tuples, indexable directly. Either way this
    # returns (key, value), each shaped [batch=1, num_kv_heads, seq_len, head_dim].
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values
    layer = past_key_values[layer_idx]
    return layer[0], layer[1]


def pooled_vector(tensor, token_indices):
    # tensor: [1, num_kv_heads, seq_len, head_dim]. Mean-pools over the given
    # token positions, then flattens heads and head_dim into one vector --
    # simplest comparable representation for "this check's K (or V), in this
    # layer, in this prompt."
    selected = tensor[0, :, token_indices, :]  # [num_kv_heads, len(token_indices), head_dim]
    pooled = selected.mean(dim=1)  # [num_kv_heads, head_dim]
    return pooled.flatten().float()


def compare(vec_a, vec_b):
    cos_sim = torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=0).item()
    rel_l2 = (torch.norm(vec_a - vec_b) / torch.norm(vec_a)).item()
    return cos_sim, rel_l2


def build_prompt_and_token_spans(tokenizer, messages, route):
    # Renders the full chat-formatted prompt, locates the route's check
    # lines within the raw user content, then maps those to token-index
    # spans in the fully rendered (and tokenized) prompt string.
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    user_offset_in_rendered = rendered.find(user_content)
    if user_offset_in_rendered == -1:
        raise ValueError("Chat template altered the user content unexpectedly -- can't locate it verbatim.")

    block_start, block_text = find_route_block(user_content, route)
    check_char_spans = find_check_spans(block_text, block_start + user_offset_in_rendered)

    encoded = tokenizer(rendered, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoded["offset_mapping"]

    check_token_spans = {
        check_idx: char_span_to_token_indices(offset_mapping, start, end)
        for check_idx, (start, end) in check_char_spans.items()
    }
    return rendered, encoded["input_ids"], check_token_spans


def main():
    device = pick_device()
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME} (tokenizer + weights) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # bf16, not fp16 -- matches the real GPU baseline run's precision.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    num_layers = model.config.num_hidden_layers

    # Small window on purpose: enough checks to see a trend across distance
    # from the drop point, small enough to keep the forward pass and the
    # manual bookkeeping fast for a first pass.
    trace = generate_trace(num_activations=2, window_size=6, stride=1, seed=1)
    activation_t, activation_t1 = trace.activations[0], trace.activations[1]

    rendered_t, ids_t, spans_t = build_prompt_and_token_spans(tokenizer, activation_t.messages, FOCUS_ROUTE)
    rendered_t1, ids_t1, spans_t1 = build_prompt_and_token_spans(tokenizer, activation_t1.messages, FOCUS_ROUTE)

    # Only checks present in BOTH windows are comparable -- check 0 aged out
    # of t+1, check 6 is genuinely new in t+1, neither has a counterpart.
    shared_checks = sorted(set(spans_t) & set(spans_t1))
    print(f"Shared checks (present in both windows): {shared_checks}")

    with torch.no_grad():
        input_ids_t = torch.tensor([ids_t], device=device)
        input_ids_t1 = torch.tensor([ids_t1], device=device)
        out_t = model(input_ids_t, use_cache=True)
        out_t1 = model(input_ids_t1, use_cache=True)

    rows = []
    for layer_idx in range(num_layers):
        key_t, value_t = extract_layer_kv(out_t.past_key_values, layer_idx)
        key_t1, value_t1 = extract_layer_kv(out_t1.past_key_values, layer_idx)

        for check_idx in shared_checks:
            tok_idx_t = spans_t[check_idx]
            tok_idx_t1 = spans_t1[check_idx]

            v_pooled_t = pooled_vector(value_t, tok_idx_t)
            v_pooled_t1 = pooled_vector(value_t1, tok_idx_t1)
            v_cos, v_rel_l2 = compare(v_pooled_t, v_pooled_t1)

            k_pooled_t = pooled_vector(key_t, tok_idx_t)
            k_pooled_t1 = pooled_vector(key_t1, tok_idx_t1)
            k_cos, k_rel_l2 = compare(k_pooled_t, k_pooled_t1)

            # distance from the drop point: check 0 (dropped) sat right
            # before check 1, so check 1 is 1 slot downstream of the change,
            # check 5 is 5 slots downstream -- tests whether contamination
            # decays with distance, as attention weight often does.
            distance_from_drop = check_idx  # check_idx IS the distance here, since check 0 was dropped

            rows.append({
                "layer": layer_idx,
                "check_index": check_idx,
                "distance_from_drop": distance_from_drop,
                "v_cosine_similarity": round(v_cos, 4),
                "v_relative_l2": round(v_rel_l2, 4),
                "k_cosine_similarity": round(k_cos, 4),
                "k_relative_l2": round(k_rel_l2, 4),
            })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    # Quick eyeball summary: V deviation (the clean contamination signal,
    # unconfounded by RoPE/position) averaged per check, across all layers.
    print("\nMean V relative-L2 deviation per check, averaged across all layers:")
    for check_idx in shared_checks:
        vals = [r["v_relative_l2"] for r in rows if r["check_index"] == check_idx]
        print(f"  check {check_idx} (distance {check_idx} from drop): {sum(vals) / len(vals):.4f}")


if __name__ == "__main__":
    main()
