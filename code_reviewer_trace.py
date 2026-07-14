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
