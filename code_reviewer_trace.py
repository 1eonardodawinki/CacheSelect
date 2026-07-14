# Code Reviewer periodic-agent trace generator.
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
