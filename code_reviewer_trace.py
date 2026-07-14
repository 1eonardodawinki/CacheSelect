# Code Reviewer periodic-agent trace generator.
#
# In simple terms: this agent pretends to watch test coverage for a few
# code files. Every time there's a new commit, it checks how well-tested
# each file is, and if one file's coverage just dropped a lot (e.g.
# someone deleted its tests), it sends an alert so the regression gets
# caught early. Otherwise it just says everything looks normal. It repeats
# this check after every commit, each time seeing a slightly updated
# window of recent coverage history.
#
# Reimplements the "Code Reviewer" agent from Norgren (arXiv:2605.26289,
# Section 4.1) -- tools: analyze_code, run_tests, check_coverage, lint_code,
# review_pr -- restructured for periodic reactivation: instead of reviewing
# one PR once, the agent re-checks test coverage for a set of tracked files
# after every new commit to the repo. Same sliding-window / seeded-anomaly
# mechanism as travel_planner_trace.py, just with per-file coverage %
# instead of per-route flight prices.

import numpy as np

from trace_common import Activation, ActivationGroundTruth, TraceBundle

# Plausible tracked files. Arbitrary but realistic-looking, same spirit as
# travel_planner_trace.py's DEFAULT_ROUTES.
DEFAULT_FILES = ["src/auth.py", "src/payment.py", "src/api_routes.py"]

TOOLS_DESCRIPTION = (
    "Tools: analyze_code(file), run_tests(file), check_coverage(file), "
    "lint_code(file), review_pr(pr_id)."
)


def build_system_prompt(files, coverage_drop_threshold_pct=15.0):
    # Builds the fixed instruction block that's identical across every
    # activation (the part that should be near-perfectly reusable by prefix
    # caching -- it never changes). Ends with a strict, rigid output format
    # ("ALERT: ..." / "STATUS: nominal") so scoring in run_baseline.py can
    # just check for an exact substring instead of parsing free-form prose.
    files_str = ", ".join(files)
    return (
        "You are a code review agent. You track test coverage for a set of "
        f"files ({files_str}) and alert when a tracked file's coverage "
        "drops significantly, so a regression can be caught before it "
        f"ships. {TOOLS_DESCRIPTION} "
        "Assess risk using only the most recent coverage check for each "
        "file, compared to the check before it -- do not consider older "
        "history. "
        f"If any file's latest coverage has dropped by more than {coverage_drop_threshold_pct:.0f}% "
        "since the prior check, output exactly one line: "
        "ALERT: <FILE> <reason>. If no file qualifies, output exactly: STATUS: nominal."
    )


def _draw_coverage_walk(rng, num_checks, drift=0.0, volatility=0.02):
    # Draws one random % change per commit (a "random walk"), so coverage
    # wiggles realistically instead of moving in a straight line.
    return rng.normal(drift, volatility, num_checks)


def _inject_anomaly_return(returns, check_index, pct_move):
    # Takes an array of normal, random coverage-change values and overwrites
    # exactly one of them with our deliberately large, planted drop. This is
    # the "answer key" mechanism: we know exactly which commit has the
    # anomaly because we're the ones who put it there. Returns a copy so the
    # caller's original array is untouched.
    out = returns.copy()
    out[check_index] = pct_move
    return out


def generate_coverage_series(file, num_checks, start_coverage, seed,
                              drift=0.0, volatility=0.02,
                              anomaly_check_index=None, anomaly_pct_move=None):
    # Same mechanism as travel_planner_trace.py's generate_price_series:
    # seeded random walk of % changes, compounded onto the starting coverage,
    # clamped to a valid percentage range since coverage can't go below 0%
    # or above 100%.
    rng = np.random.default_rng(seed)
    returns = _draw_coverage_walk(rng, num_checks, drift, volatility)
    if anomaly_check_index is not None and anomaly_pct_move is not None:
        returns = _inject_anomaly_return(returns, anomaly_check_index, anomaly_pct_move)
    coverage = start_coverage * np.cumprod(1 + returns)
    coverage = np.clip(coverage, 0.0, 100.0)
    return coverage.tolist()


def format_coverage_window(file, coverage_values, window_start_idx):
    # Renders one file's visible window of coverage checks as plain text for
    # the prompt, e.g. "src/auth.py:\n  commit 5: coverage 87.3%".
    # `window_start_idx` is added on so the commit numbers shown reflect
    # their true position in the full coverage history, not just their
    # position within this particular window.
    lines = [f"{file}:"]
    for offset, cov in enumerate(coverage_values):
        check_idx = window_start_idx + offset
        lines.append(f"  commit {check_idx}: coverage {cov:.1f}%")
    return "\n".join(lines)


def build_activation_messages(system_prompt, file_windows, window_start_idx):
    # Assembles the two-message chat payload for one activation: the fixed
    # system prompt (built once, reused every activation) plus a user
    # message listing every tracked file's current coverage window.
    blocks = [
        format_coverage_window(file, values, window_start_idx)
        for file, values in file_windows.items()
    ]
    user_content = (
        "Latest coverage checks:\n\n" + "\n\n".join(blocks) +
        "\n\nReview the latest check for each file and report any threshold breaches."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def generate_trace(
    routes=None,  # accepted for signature compatibility with run_baseline.py's generic call;
                  # "routes" here means "files" -- see DEFAULT_FILES
    num_activations=20,
    window_size=30,
    stride=1,
    seed=42,
    anomaly_activation_index=None,
    anomaly_route=None,  # same compatibility note -- this is the anomaly file
    anomaly_pct_move=-0.40,
    volatility=0.02,
    start_prices=None,  # same compatibility note -- starting coverage % per file
    price_drop_threshold_pct=15.0,
):
    # Pure function, same determinism guarantee as travel_planner_trace.py.
    files = list(routes) if routes else list(DEFAULT_FILES)
    if anomaly_activation_index is None:
        anomaly_activation_index = num_activations // 2  # default: plant it roughly in the middle
    if anomaly_route is None:
        anomaly_route = files[1] if len(files) > 1 else files[0]
    if start_prices is None:
        # Plausible starting coverage per file, comfortably below 100% so
        # the random walk has room to move in both directions.
        start_prices = {f: 90.0 - 3.0 * i for i, f in enumerate(files)}

    # Total commit history needed to cover every activation's window. E.g.
    # window_size=30, stride=1, num_activations=20 -> 49 commits total, since
    # activation 19's window is commits [19, 49).
    num_checks = window_size + stride * (num_activations - 1)

    # Convert "the anomaly should be the NEWEST commit visible in activation
    # N's window" into an absolute index into the full commit history. This
    # is what makes ground truth unambiguous: by construction, no other
    # activation has this commit as its newest one, even though a sliding
    # window means this commit is still technically visible in several
    # activations before and after.
    anomaly_check_index = window_size - 1 + anomaly_activation_index * stride

    system_prompt = build_system_prompt(files, price_drop_threshold_pct)

    # Generate one full coverage history per file. Only `anomaly_route` gets
    # the anomaly injected; the other files are just normal random walks.
    coverage_series = {}
    for i, file in enumerate(files):
        is_anomaly_file = file == anomaly_route
        coverage_series[file] = generate_coverage_series(
            file=file,
            num_checks=num_checks,
            start_coverage=start_prices[file],
            seed=seed + i,  # different-but-deterministic seed per file, so
                             # files don't all move in lockstep
            volatility=volatility,
            anomaly_check_index=anomaly_check_index if is_anomaly_file else None,
            anomaly_pct_move=anomaly_pct_move if is_anomaly_file else None,
        )

    # Slice out each activation's window and build its prompt + ground truth.
    activations = []
    for act_idx in range(num_activations):
        window_start = act_idx * stride
        window_end = window_start + window_size
        file_windows = {f: coverage_series[f][window_start:window_end] for f in files}
        messages = build_activation_messages(system_prompt, file_windows, window_start)
        # Flat string version, purely for human-readable dumps (see trace_common.py).
        prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]

        # True on exactly one activation -- the one whose newest commit is the
        # planted anomaly. Every other activation should produce "STATUS: nominal".
        expect_flag = act_idx == anomaly_activation_index
        ground_truth = ActivationGroundTruth(
            activation_index=act_idx,
            expect_flag=expect_flag,
            anomaly_entity=anomaly_route if expect_flag else None,
            anomaly_bar_index=anomaly_check_index if expect_flag else None,
            anomaly_magnitude=anomaly_pct_move if expect_flag else None,
        )
        activations.append(Activation(index=act_idx, prompt=prompt, messages=messages, ground_truth=ground_truth))

    return TraceBundle(
        agent_name="code_reviewer",
        tickers_or_entities=files,
        num_activations=num_activations,
        window_size=window_size,
        stride=stride,
        seed=seed,
        anomaly_activation_index=anomaly_activation_index,
        anomaly_entity=anomaly_route,
        anomaly_magnitude=anomaly_pct_move,
        # Every argument used to build this trace, saved for reproducibility
        # -- so a saved trace file fully documents how to regenerate it.
        generation_params=dict(
            files=files, num_activations=num_activations, window_size=window_size,
            stride=stride, seed=seed, anomaly_activation_index=anomaly_activation_index,
            anomaly_file=anomaly_route, anomaly_pct_move=anomaly_pct_move,
            volatility=volatility, start_prices=start_prices,
            coverage_drop_threshold_pct=price_drop_threshold_pct,
        ),
        activations=activations,
    )


if __name__ == "__main__":
    # Quick manual sanity check, same pattern as the other two agents.
    bundle = generate_trace(num_activations=5, window_size=10, stride=1, seed=1)
    for a in bundle.activations:
        print(f"=== activation {a.index} (expect_flag={a.ground_truth.expect_flag}) ===")
        print(a.prompt)
        print()
