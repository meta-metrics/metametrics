
# !git clone https://github.com/google-research/mt-metrics-eval.git && cd mt-metrics-eval && pip install .
from metric_report import evaluate_metric, write_results

from mt_metrics_eval import data


import numpy as np

def NewMetric(
    level: str,
    lp: str,
    domains: dict[str, list[list[int]]],
    docs: dict[str, list[int]],
    src: list[str],
    ref: list[str],
    hyps: dict[list[str]]
) -> dict[str, list[float]]:
    """
    Generate metric scores.
    
    Args:
    level: Level for which to produce scores, 'sys' or 'seg'.
    lp: Language pair, eg 'en-de'.
    domains: Map from domain name to [[beg, end+1], ...] segment position lists.
    docs: Map from doc name to [beg, end+1] segment positions.
    src: List of source segments.
    ref: List of reference segments.
    hyps: Map from MT system name to output segments for that system.
    
    Returns:
    Map from system name to scores, a list of segment-level scores if level is
    'seg', or a list containing a single score if level is 'sys'.
    """
    # Sample metric just computes a length match between each hypothesis and the
    # reference. It ignores lp, domains, docs, and source.
    
    del lp, domains, docs, src
    
    ref_lens = np.array([len(r) for r in ref])
    scores = {}
    for sysname, hyp in hyps.items():
        hyp_lens = np.array([len(h) for h in hyp])
        deltas = np.abs(ref_lens - hyp_lens) / (ref_lens + 1)
        scores[sysname] = -deltas if level == 'seg' else [-deltas.mean()]
    
    return scores

def main():
    data.Download()
    result, wts = evaluate_metric(NewMetric, "lendiff")
    write_results(result, wts, "lendiff")

if __name__ == "__main__":
    main()