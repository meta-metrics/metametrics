import os
import csv
from meta_metrics import MetaMetrics

os.system("!git clone https://github.com/google-research/mt-metrics-eval.git && cd mt-metrics-eval && pip install .")

# @title Imports

from mt_metrics_eval import meta_info
from mt_metrics_eval import data
from mt_metrics_eval import tasks

# @title Download data

data.Download()  # Copies about 2G onto local machine.

# @title Define the metric

import numpy as np

# Replace this function with your own metric.

def NewMetric(
    # level: str,
    metric,
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
    # level: Level for which to produce scores, 'sys' or 'seg'.
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

  segment_scores = {}
  system_scores = {}
  for sysname, hyp in hyps.items():
    outputs = np.array(metric.score(hyp, ref, src))

    segment_scores[sysname] = outputs
    system_scores[sysname] = [outputs.mean()]

  return segment_scores, system_scores

# @title Load EvalSets

wmt23_lps = ['en-de', 'he-en', 'zh-en']
evs_dict = {('wmt23', lp): data.EvalSet('wmt23', lp, True) for lp in wmt23_lps}

# @title Add metric scores to EvalSets

# Compute scores for each language pair, and add to the appropriate EvalSet.
# Setting replace=True makes this work if we want to iterate over different
# versions of the metric.

metrics_configs = {
    ("metricx", {"model_name": "google/metricx-23-xxl-v2p0", "batch_size": 1, 'is_qe': False, 'tokenizer_name': "google/mt5-xxl", 'max_input_length': 1024, "bf16": True}, False),
    ("comet", {"hf_token": "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", "batch_size": 1}, False),
    ("xcomet-xl", {"hf_token": "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", "batch_size": 1}, False),
}

# "params": {
#       "metricx-23-xxl-v2p0": 1.0,
#       "wmt22-comet-da": 0.2055307813370211,
#       "xcomet-xl": 0.27327721603913696,
#     }

metric_name = 'metametrics'
metric = MetaMetrics(metrics_configs, weights=[1,0.2055307813370211,0.27327721603913696])

for lp in wmt23_lps:
  print(">>>", lp)
  evs = evs_dict[('wmt23', lp)]

  with open(f"scores_{lp}_segment.tsv", "w+") as f_out_segment:
    with open(f"scores_{lp}_system.tsv", "w+") as f_out_system:
        writer = csv.writer(f_out_segment, delimiter='\t')
        writer_system = csv.writer(f_out_system, delimiter='\t')

        for refname, ref in evs.all_refs.items():
            print(">>>", refname)
            seg_scores, sys_scores = NewMetric(metric, evs.lp, evs.domains, evs.docs, evs.src, ref, evs.sys_outputs)
            evs.AddMetric(metric_name, {refname}, 'sys', sys_scores, replace=True)
            evs.AddMetric(metric_name, {refname}, 'seg', seg_scores, replace=True)

            f_out_segment.writerow(["sys_name","lp","segment_id","src","ref","mt","score"])

            for sys_name in range(seg_scores):
                for _id in range(len(seg_scores[sys_name])):
                    f_out_segment.writerow([sys_name, evs.lp, _id, evs.src[_id], ref[sys_name][_id], evs.sys_outputs[_id], seg_scores[sys_name][_id]])

# Add new metric to the primary lists, so it will get picked up when tasks get
# run with primary=True (avoiding having to evaluate all contrastive
# submissions as well).

for evs in evs_dict.values():
  evs.SetPrimaryMetrics(evs.primary_metrics | {metric_name})


# @title Generate results with new metric

# For a first pass we turn off significance testing.

wmt23_tasks, wts = tasks.WMT23(wmt23_lps, k=0)

# Takes about 3 minutes.
new_results = wmt23_tasks.Run(eval_set_dict=evs_dict)

# @title Print results

# Results show all primary metrics, along with the new 'lendiff' metric.

avg_corrs = new_results.AverageCorrs(wts)

table = new_results.Table(
    metrics=list(avg_corrs),
    initial_column=avg_corrs,
    initial_column_header='avg-corr',
    attr_list=['lang', 'level', 'corr_fcn'],
    nicknames={'KendallWithTiesOpt': 'acc-t'},
    fmt='text',
    baselines_metainfo=meta_info.WMT23)

print(table)