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

# Verbatim from FinQA dev.json, example id "AAPL/2004/page_36.pdf-2"
# (github.com/czyssrs/FinQA, dataset/dev.json). Real filing text, not
# generated or paraphrased.
FINQA_PRE_TEXT = (
    "Net sales of the retail segment grew to $1.185 billion during 2004 from "
    "$621 million and $283 million, in 2003 and 2002, respectively. The "
    "increases in net sales during both 2004 and 2003 reflect the impact of "
    "new store openings for each fiscal year, including the opening of 21 "
    "new stores in 2004 and 25 new stores in 2003. Gross margin for the "
    "three fiscal years ended September 25, 2004 are as follows "
    "(in millions, except gross margin percentages):"
)

FINQA_TABLE = (
    "                          2004      2003      2002\n"
    "net sales               $8279     $6207     $5742\n"
    "cost of sales             6020      4499      4139\n"
    "gross margin             $2259     $1708     $1603\n"
    "gross margin percentage   27.3%     27.5%     27.9%"
)

FINQA_POST_TEXT = (
    "Gross margin declined in fiscal 2004 to 27.3% of net sales from 27.5% "
    "of net sales in 2003. The company's gross margin during fiscal 2004 "
    "declined due to an increase in mix towards lower margin iPod and iBook "
    "sales, pricing actions on certain Power Macintosh G5 models that were "
    "transitioned during the beginning of 2004, higher warranty costs on "
    "certain portable Macintosh products, and higher freight and duty costs "
    "during fiscal 2004."
)

FINQA_REFERENCE = (
    "--- Reference filing excerpt: Apple Inc., FY2004 10-K "
    "(source: FinQA dataset, github.com/czyssrs/FinQA, id AAPL/2004/page_36.pdf-2) ---\n"
    f"{FINQA_PRE_TEXT}\n\n{FINQA_TABLE}\n\n{FINQA_POST_TEXT}\n"
    "--- end reference excerpt ---"
)

# The three tracked metrics, seeded from Apple's real FY2004 values in the
# reference filing above ($ millions, except the percentage).
DEFAULT_METRICS = ["net_sales", "cost_of_sales", "gross_margin_pct"]
REAL_2004_VALUES = {
    "net_sales": 8279.0,
    "cost_of_sales": 6020.0,
    "gross_margin_pct": 27.3,
}

TOOLS_DESCRIPTION = (
    "Tools: query_db(metric), create_chart(metric), export_data(metric), "
    "run_pipeline(), compute_stats(metric)."
)
