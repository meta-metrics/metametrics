# @title Imports

from mt_metrics_eval import meta_info
from mt_metrics_eval import data
from mt_metrics_eval import tasks

import sys
import os



def GenerateTaskSet(key_to_lp, seg_pairs, sys_pairs, primary=True, k=0, gold=None):
    """Generate the WMT23 task set and associated weight vector."""
    
    # Not strictly necessary to declare this, because setting human=True will
    # only score human outputs if any are  available, but we want to make the
    # human attribute reflect what actually got used, and also want to avoid
    # having to load the EvalSets at this point to get this info automatically.
    # lps_with_multiple_refs = {'en-he', 'he-en'}
    results = {}
    def Add(key, lp, level, corr_fcn, human, gold, **kw_args):
        ts.Append(tasks.Task(
            key, lp, level=level, corr_fcn=corr_fcn, human=human, gold=gold,
            primary=primary, k=k, **kw_args))
    for key in key_to_lp:
        lps = key_to_lp[key]
    
        ts = tasks.TaskSet()
        # # 1st task is pairwise accuracy across all lps.
        try:
            Add(key, ','.join(lps), 'sys', 'accuracy',
              human=False,
              gold=[gold] * len(lps) if gold else None)
        except:
            pass
        # System- and segment-level Pearson, and segment-level accuracy for all lps.
        for lp in lps:
            human = False
            if (key,lp) in sys_pairs:
                Add(key, lp, 'sys', 'pearson', human, gold)
            if (key,lp) in seg_pairs:
                Add(key, lp, 'seg', 'pearson', human, gold)
                Add(key, lp, 'seg', 'KendallWithTiesOpt', human, gold,
                    avg_by='item', perm_test='pairs', corr_fcn_args={'sample_rate': 1.0})
        
        weights = [len(lps)] + [1] * (len(ts) - 1)
        weights = [w / sum(weights) for w in weights]
        results[key] = ts, weights
      
    return results

def evaluate_metric(metric, metric_name):
    key_to_lps = {}
    evs_dict = {}
    for key in meta_info.DATA:
        print(key)
        key_to_lps[key] = meta_info.DATA[key].keys()
        for lp in meta_info.DATA[key]:
            evs_dict[(key, lp)] = data.EvalSet(key, lp, True)
    
    # @title Add metric scores to EvalSets
    
    # Compute scores for each language pair, and add to the appropriate EvalSet.
    # Setting replace=True makes this work if we want to iterate over different
    # versions of the metric.
    
    for key, lp in evs_dict:
        evs = evs_dict[(key,lp)]
        for refname, ref in evs.all_refs.items():
            sys_scores = metric(
                'sys', evs.lp, evs.domains, evs.docs, evs.src, ref, evs.sys_outputs)
            seg_scores = metric(
                'seg', evs.lp, evs.domains, evs.docs, evs.src, ref, evs.sys_outputs)
            evs.AddMetric(metric_name, {refname}, 'sys', sys_scores, replace=True)
            evs.AddMetric(metric_name, {refname}, 'seg', seg_scores, replace=True)
    
    # Add new metric to the primary lists, so it will get picked up when tasks get
    # run with primary=True (avoiding having to evaluate all contrastive
    # submissions as well).
    
    for evs in evs_dict.values():
        evs.SetPrimaryMetrics(evs.primary_metrics | {metric_name})

    sys_pairs = []
    seg_pairs = []
    for key in key_to_lps:
        for lp in key_to_lps[key]:
            if 'sys' in meta_info.DATA[key][lp].std_gold:
                sys_pairs.append((key,lp))
            if 'seg' in meta_info.DATA[key][lp].std_gold:
                seg_pairs.append((key,lp))

    tasks_set = GenerateTaskSet(key_to_lps, seg_pairs, sys_pairs, k = 0)
    # @title Generate results with new metric
    
    # For a first pass we turn off significance testing.
    new_results = {}
    new_wts = {}
    for key, (key_tasks, wts) in tasks_set.items():
        print(key)
        new_results[key] = key_tasks.Run(eval_set_dict=evs_dict)
        new_wts[key] = wts
    return new_results, new_wts

def write_results(new_results, new_wts, metric_name, fmt = 'tsv'):
    for key in new_results:
        wts = new_wts[key][1]
        avg_corrs = new_results[key].AverageCorrs(wts)
        
        table = new_results[key].Table(
            metrics=list(avg_corrs),
            initial_column=avg_corrs,
            initial_column_header='avg-corr',
            attr_list=['lang', 'level', 'corr_fcn'],
            nicknames={'KendallWithTiesOpt': 'acc-t'},
            fmt=fmt,
            baselines_metainfo=meta_info.WMT23)
        if not os.path.exists(metric_name):
            os.makedirs(metric_name)
        with open(f"{metric_name}/{key}.tsv", "w+") as f:
            f.write(table)
        # print(table)