import os
import pandas as pd
from tqdm import tqdm

from metametrics.utils.logging import get_logger
from metametrics.utils.constants import METAMETRICS_SCORE

logger = get_logger(__name__)

WMT23_LPS = ['en-de', 'he-en', 'zh-en']
WMT24_LPS = ['en-de', 'en-es', 'ja-zh']

def NewMetric(score_df, src: list[str], ref: list[str], hyps: dict[list[str]]) -> dict[str, list[float]]:
    """
    Generate metric scores.

    Args:
        score_df: MetaMetrics score
        src: List of source segments.
        ref: List of reference segments.
        hyps: Map from MT system name to output segments for that system.

    Returns:
        Map from system name to scores, a list of segment-level scores if level is
        'seg', or a list containing a single score if level is 'sys'.
    """
    segment_scores = {}
    system_scores = {}
  
    for sysname, hyp in tqdm(hyps.items()):
        filtered_score_df = []
        for h, r, s in zip(hyp, ref, src):
            # Filter score_df where hyp, ref, and src match
            matched_rows = score_df[
                (score_df['mt'] == h) & 
                (score_df['ref'] == r) & 
                (score_df['src'] == s)
            ]
            
            # If matches found, print or store the metric values
            if not matched_rows.empty:
                filtered_score_df.append(matched_rows[METAMETRICS_SCORE].mean(axis=0).to_list())
            else:
                logger.error(f"[ERROR] Translation not found")
        
        filtered_score_df = pd.DataFrame(filtered_score_df, columns=[METAMETRICS_SCORE])
        segment_scores[sysname] = filtered_score_df[METAMETRICS_SCORE]
        system_scores[sysname] = [filtered_score_df.mean()]

    return segment_scores, system_scores

def evaluate_wmt23(score_df, output_path, **kwargs):
    from mt_metrics_eval import meta_info
    from mt_metrics_eval import data
    from mt_metrics_eval import tasks

    evs_dict = {('wmt23', lp): data.EvalSet('wmt23', lp, True) for lp in WMT23_LPS}
    
    for lp in WMT23_LPS:
        logger.debug(f"Evaluating for language pair: {lp}")
        evs = evs_dict[('wmt23', lp)]

        for refname, ref in evs.all_refs.items():
            seg_scores, sys_scores = NewMetric(score_df, evs.src, ref, evs.sys_outputs)
            evs.AddMetric(METAMETRICS_SCORE, {refname}, 'sys', sys_scores, replace=True)
            evs.AddMetric(METAMETRICS_SCORE, {refname}, 'seg', seg_scores, replace=True)

    # Add new metric to the primary lists, so it will get picked up when tasks get
    # run with primary=True (avoiding having to evaluate all contrastive
    # submissions as well).

    for evs in evs_dict.values():
        evs.SetPrimaryMetrics(evs.primary_metrics | {METAMETRICS_SCORE})

    # We turn off significance testing.
    wmt23_tasks, wts = tasks.WMT23(WMT23_LPS, k=0)
    new_results = wmt23_tasks.Run(eval_set_dict=evs_dict)
    avg_corrs = new_results.AverageCorrs(wts)
    table = new_results.Table(
        metrics=list(avg_corrs),
        initial_column=avg_corrs,
        initial_column_header='avg-corr',
        attr_list=['lang', 'level', 'corr_fcn'],
        nicknames={'KendallWithTiesOpt': 'acc-t'},
        fmt='text',
        baselines_metainfo=meta_info.WMT23)

    with open(os.path.join(output_path, "wmt23_result.txt"), "w") as f:
        f.write(str(table))


def evaluate_wmt24(score_df, output_path, **kwargs):
    evs_dict = {('wmt24', lp): data.EvalSet('wmt24', lp, True) for lp in WMT24_LPS}
    
    for lp in WMT24_LPS:
        logger.debug(f"Evaluating for language pair: {lp}")
        evs = evs_dict[('wmt24', lp)]

        for refname, ref in evs.all_refs.items():
            seg_scores, sys_scores = NewMetric(score_df, evs.src, ref, evs.sys_outputs)
            evs.AddMetric(METAMETRICS_SCORE, {refname}, 'sys', sys_scores, replace=True)
            evs.AddMetric(METAMETRICS_SCORE, {refname}, 'seg', seg_scores, replace=True)

    # Add new metric to the primary lists, so it will get picked up when tasks get
    # run with primary=True (avoiding having to evaluate all contrastive
    # submissions as well).

    for evs in evs_dict.values():
        evs.SetPrimaryMetrics(evs.primary_metrics | {METAMETRICS_SCORE})

    # We turn off significance testing.
    wmt24_tasks, wts = tasks.WMT24(WMT24_LPS, k=0)
    new_results = wmt24_tasks.Run(eval_set_dict=evs_dict)
    avg_corrs = new_results.AverageCorrs(wts)
    table = new_results.Table(
        metrics=list(avg_corrs),
        initial_column=avg_corrs,
        initial_column_header='avg-corr',
        attr_list=['lang', 'level', 'corr_fcn'],
        nicknames={'KendallWithTiesOpt': 'acc-t'},
        fmt='text',
        baselines_metainfo=meta_info.WMT24)

    with open(os.path.join(output_path, "wmt24_result.txt"), "w") as f:
        f.write(str(table))
