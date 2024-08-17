import os
import csv
from meta_metrics import MetaMetrics

os.system("git clone https://github.com/google-research/mt-metrics-eval.git && cd mt-metrics-eval && pip install .")

# @title Imports

from mt_metrics_eval import meta_info
from mt_metrics_eval import data
from mt_metrics_eval import tasks

objective = "pearson"
os.system(f"mkdir -p wmt23_outputs/{objective}")

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

  del lp, domains, docs

  segment_scores = {}
  system_scores = {}
  count = 0
  
  for sysname, hyp in hyps.items():
    count += 1
    print(f"######### {count} of {len(hyps)}")
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

#########################
###### PEARSON ##########
#########################

###### METAMETRICS ######

metrics_configs = [
    ("metricx", {"model_name": "google/metricx-23-xxl-v2p0", "batch_size": 1, 'is_qe': False, 'tokenizer_name': "google/mt5-xxl", 'max_input_length': 1024, "bf16": True}, False),
    ("comet", {"hf_token": "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", "batch_size": 8}, False),
]

metric_name = 'metametrics'
metric = MetaMetrics(metrics_configs, weights=[0.7847653360190587,1], normalize=True, cache_mode=True)

# metricx-23-xxl-v2p0
# wmt22-comet-da1

#########################
###### KENDALL ##########
#########################

###### METAMETRICS ######

# metrics_configs = [
#     ("metricx", {"model_name": "google/metricx-23-xxl-v2p0", "batch_size": 1, 'is_qe': False, 'tokenizer_name': "google/mt5-xxl", 'max_input_length': 1024, "bf16": True}, False),
#     ("comet", {"hf_token": "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", "batch_size": 8}, False),
#     ("xcomet-xl", {"hf_token": "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", "batch_size": 4}, False)
# ]

# metric_name = 'metametrics'
# metric = MetaMetrics(metrics_configs, weights=[1,0.2055307813370211,0.27327721603913696], normalize=True, cache_mode=True)

###### METAMETRICS-EN-SOURCE ######

# metrics_configs = [
#     ("bertscore", {"model_name": "microsoft/deberta-xlarge-mnli", "model_metric": "precision", "batch_size": 4}, False),
#     ("metricx", {"model_name": "google/metricx-23-xxl-v2p0", "batch_size": 1, 'is_qe': False, 'tokenizer_name': "google/mt5-xxl", 'max_input_length': 1024, "bf16": True}, False),
#     ("comet", {"hf_token": "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", "batch_size": 8}, False),
#     ("xcomet-xl", {"hf_token": "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", "batch_size": 4}, False)
# ]

# metric_name = 'metametrics-en-source'
# metric = MetaMetrics(metrics_configs, weights=[1,1,1,1], normalize=True, cache_mode=True)

# bertscore_precision 1
# metricx-23-xxl-v2p0 1
# wmt22-comet-da 1
# xcomet-xl 1


###### METAMETRICS-EN-TARGET ######

# metrics_configs = [
#     ("metricx", {"model_name": "google/metricx-23-xl-v2p0", "batch_size": 1, 'is_qe': False, 'tokenizer_name': "google/mt5-xl", 'max_input_length': 1024, "bf16": True}, False),
#     ("metricx", {"model_name": "google/metricx-23-xxl-v2p0", "batch_size": 1, 'is_qe': False, 'tokenizer_name': "google/mt5-xxl", 'max_input_length': 1024, "bf16": True}, False),
#     ("xcomet-xl", {"hf_token": "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", "batch_size": 4}, False)
# ]

# metric_name = 'metametrics-en-target'
# metric = MetaMetrics(metrics_configs, weights=[0.16295300667868262,1,1], normalize=True, cache_mode=True)

# metricx-23-xl-v2p00.16295300667868262
# metricx-23-xxl-v2p01
# xcomet-xl1

###### METAMETRICS-QE ######

# metrics_configs = [
#     ("metricx", {"model_name": "google/metricx-23-qe-xxl-v2p0", "batch_size": 1, 'is_qe': True, 'tokenizer_name': "google/mt5-xxl", 'max_input_length': 1024, "bf16": True}, True),
#     ("metricx", {"model_name": "google/metricx-23-qe-large-v2p0", "batch_size": 1, 'is_qe': True, 'tokenizer_name': "google/mt5-large", 'max_input_length': 1024, "bf16": True}, True),        
#     ("cometkiwi", {"hf_token": "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", "batch_size": 8}, True),
#     ("cometkiwi-xl", {"hf_token": "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", "batch_size": 1}, True),
# ]

# metric_name = 'metametrics-qe'
# metric = MetaMetrics(metrics_configs, weights=[0.9904603321616574,0.06564649056309636,0.1267047358620059,0.05844699223607353], normalize=True, cache_mode=True)

###### METAMETRICS-QE-EN-SOURCE ######

# metrics_configs = [
#     ("metricx", {"model_name": "google/metricx-23-qe-xxl-v2p0", "batch_size": 1, 'is_qe': True, 'tokenizer_name': "google/mt5-xxl", 'max_input_length': 1024, "bf16": True}, True),
#     ("cometkiwi", {"hf_token": "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", "batch_size": 8}, True),
#     ("cometkiwi-xl", {"hf_token": "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", "batch_size": 1}, True),
# ]

# metric_name = 'metametrics-qe-en-source'
# metric = MetaMetrics(metrics_configs, weights=[0.9206599602022377,0.22511976200589023,0.05474087559603472], normalize=True, cache_mode=True)

# metricx-23-qe-xxl-v2p0_reference_free0.9206599602022377
# wmt22-cometkiwi-da_reference_free0.22511976200589023
# wmt23-cometkiwi-da-xl_reference_free0.05474087559603472

###### METAMETRICS-QE-EN-TARGET ######

# metrics_configs = [
#     ("gemba_mqm", {"model": "gpt-4o-mini"}, True),
#     ("metricx", {"model_name": "google/metricx-23-qe-large-v2p0", "batch_size": 1, 'is_qe': True, 'tokenizer_name': "google/mt5-large", 'max_input_length': 1024, "bf16": True}, True),    
#     ("metricx", {"model_name": "google/metricx-23-qe-xl-v2p0", "batch_size": 1, 'is_qe': True, 'tokenizer_name': "google/mt5-xl", 'max_input_length': 1024, "bf16": True}, True),
#     ("metricx", {"model_name": "google/metricx-23-qe-xxl-v2p0", "batch_size": 1, 'is_qe': True, 'tokenizer_name': "google/mt5-xxl", 'max_input_length': 1024, "bf16": True}, True),
# ]

# metric_name = 'metametrics-qe-en-target'
# metric = MetaMetrics(metrics_configs, weights=[0.06460795131265709,0.25352367596996267,0.03973703688652602,1], normalize=True)

# GEMBA_score0.06460795131265709
# metricx-23-qe-large-v2p0_reference_free0.25352367596996267
# metricx-23-qe-xl-v2p0_reference_free0.03973703688652602
# metricx-23-qe-xxl-v2p0_reference_free1

###### METRICX-23-QE-XXL ######

# metrics_configs = [
#     ("metricx", {"model_name": "google/metricx-23-qe-xxl-v2p0", "batch_size": 1, 'is_qe': True, 'tokenizer_name': "google/mt5-xxl", 'max_input_length': 1024, "bf16": True}, True)
# ]

# metric_name = 'metricx-qe-xxl'
# metric = MetaMetrics(metrics_configs, weights=[1], normalize=True, cache_mode=True)

###### METRICX-23-QE-LARGE ######

# metrics_configs = [
#     ("metricx", {"model_name": "google/metricx-23-qe-large-v2p0", "batch_size": 1, 'is_qe': True, 'tokenizer_name': "google/mt5-large", 'max_input_length': 1024, "bf16": True}, True),        
# ]

# metric_name = 'metricx-qe-large'
# metric = MetaMetrics(metrics_configs, weights=[1], normalize=True, cache_mode=True)

###### COMETKIWI ######

# metrics_configs = [
#     ("cometkiwi", {"hf_token": "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", "batch_size": 8}, True)     
# ]

# metric_name = 'cometkiwi-qe'
# metric = MetaMetrics(metrics_configs, weights=[1], normalize=True, cache_mode=True)

###### COMETKIWI-XL ######

# metrics_configs = [
#     ("cometkiwi-xl", {"hf_token": "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", "batch_size": 1}, True)     
# ]

# metric_name = 'cometkiwi-xl-qe'
# metric = MetaMetrics(metrics_configs, weights=[1], normalize=True, cache_mode=True)


###### COMET ######

# metrics_configs = [
#     ("comet", {"hf_token": "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", "batch_size": 8}, False),
# ]

# metric_name = 'comet'
# metric = MetaMetrics(metrics_configs, weights=[1], normalize=True, cache_mode=True)

###### MetricX ######

# metrics_configs = [
#     ("metricx", {"model_name": "google/metricx-23-xxl-v2p0", "batch_size": 1, 'is_qe': False, 'tokenizer_name': "google/mt5-xxl", 'max_input_length': 1024, "bf16": True}, False),
# ]

# metric_name = 'metricx-23-xxl'
# metric = MetaMetrics(metrics_configs, weights=[1], normalize=True, cache_mode=True)

###### XCOMET-XL ######

# metrics_configs = [
#     ("xcomet-xl", {"hf_token": "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", "batch_size": 4}, False)
# ]

# metric_name = 'xcomet-xl'
# metric = MetaMetrics(metrics_configs, weights=[1], normalize=True, cache_mode=True)

for lp in wmt23_lps:
  print(">>>", lp)
  evs = evs_dict[('wmt23', lp)]

  with open(f"wmt23_outputs/{objective}/{metric_name}_scores_{lp}_segment.tsv", "w+") as f_out_segment:
    with open(f"wmt23_outputs/{objective}/{metric_name}_scores_{lp}_system.tsv", "w+") as f_out_system:
        writer = csv.writer(f_out_segment, delimiter='\t')
        writer_system = csv.writer(f_out_system, delimiter='\t')
        print(">>>>>>>>", len(evs.all_refs))
        writer.writerow(["refname","sys_name","lp","segment_id","src","ref","mt","score"])
        writer_system.writerow(["refname","sys_name","lp","score"])
        for refname, ref in evs.all_refs.items():
            print(">>>>>", refname)
            seg_scores, sys_scores = NewMetric(metric, evs.lp, evs.domains, evs.docs, evs.src, ref, evs.sys_outputs)
            evs.AddMetric(metric_name, {refname}, 'sys', sys_scores, replace=True)
            evs.AddMetric(metric_name, {refname}, 'seg', seg_scores, replace=True)
            for sys_name in sys_scores:
                writer_system.writerow([refname, sys_name, evs.lp, sys_scores[sys_name][0]])
            
            for sys_name in seg_scores:
                for _id in range(len(seg_scores[sys_name])):
                    writer.writerow([refname, sys_name, evs.lp, _id, evs.src[_id], ref[_id], evs.sys_outputs[sys_name][_id], seg_scores[sys_name][_id]])

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

with open(f"wmt23_outputs/{metric_name}_outputs.txt", "w") as f:
    f.write(str(table))