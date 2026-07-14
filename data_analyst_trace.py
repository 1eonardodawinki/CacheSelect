# Data Analyst periodic-agent trace generator.
#
# Reimplements the "Data Analyst" agent from Norgren (arXiv:2605.26289,
# Section 4.1) -- tools: query_db, create_chart, export_data, run_pipeline,
# compute_stats -- restructured for periodic reactivation, same as
# travel_planner_trace.py and code_reviewer_trace.py.
#
# Hybrid real/synthetic design: unlike the other two agents, this one's
# FIXED reference material (the part that's identical across every
# activation, analogous to a system prompt) is a REAL excerpt from Apple's
# fiscal 2004 10-K filing, taken verbatim from the FinQA dataset (Chen et
# al., EMNLP 2021, arXiv:2109.00122, github.com/czyssrs/FinQA) -- example
# id "AAPL/2004/page_36.pdf-2". FinQA itself is a static one-document-one-
# question benchmark with no time-series structure, so it can't drive the
# periodic/sliding-window part of this agent on its own. Instead: the real
# filing is the large, realistic, unchanging reference context every
# activation includes (testing whether DeltaCache holds up when the STABLE
# portion is a genuine real document, not just a short instruction block),
# and the periodic monitoring layer on top is synthetic -- a random walk
# seeded from Apple's real 2004 values, continuing the same three metrics
# (net sales, cost of sales, gross margin %) forward in simulated time,
# with one seeded anomaly for ground truth. Same mechanism as
# travel_planner_trace.py's price walk, just applied to financial metrics
# instead of flight prices.

import numpy as np

from trace_common import Activation, ActivationGroundTruth, TraceBundle
