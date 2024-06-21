# pylint: disable=W0703,C0321,C0103,C0301
import json
import argparse
import math
from collections import defaultdict
import numpy as np


parser = argparse.ArgumentParser(description='Process argument')
parser.add_argument('--correlator_str', default='kendall', help='which correlation metric to use')
parser.add_argument('--only_extractive', action='store_true', \
    help='only include extractive methods in calculations')
parser.add_argument('--annotators_str', default='expert_annotations', \
    help='string specificying whether to use expert annotations or turker annotations')
parser.add_argument('--only_abstractive', action='store_true', \
    help='only include abstractive methods in calculations')
parser.add_argument('--subset', default=11, \
    help='how many references used to calculate metric scores for correlation calculations')
parser.add_argument('--input_file', \
    default="model_annotations.aligned.scored.jsonl", \
    help="jsonl file with annotations and metric scores")
args = parser.parse_args()

assert not (args.only_extractive and args.only_abstractive)
if args.correlator_str == "pearson":
    from scipy.stats import pearsonr as correlator
else:
    from scipy.stats import kendalltau as correlator

tmp_summ_ids = {'M15', 'M20', 'M0', 'M10', 'M23_C4', 'M23_dynamicmix', 'M2', \
    'M5', 'M17', 'M11', 'M9', 'M22', 'M12', 'M1', 'M8', 'M13', 'M14'}
extractive_ids = ["M0", "M1", "M2", "M5"]
abstractive_ids = list(tmp_summ_ids - set(extractive_ids))
summ_ids = set()

sorted_keys = ['rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_f_score', \
    'rouge_4_f_score', 'rouge_l_f_score', 'rouge_su*_f_score', \
    'rouge_w_1.2_f_score', 'rouge_we_1_f', 'rouge_we_2_f', 'rouge_we_3_f', \
    's3_pyr', 's3_resp', 'bert_score_precision', 'bert_score_recall', \
    'bert_score_f1', 'mover_score', 'sentence_movers_glove_sms', 'summaqa_avg_fscore', \
    'blanc', 'supert', 'bleu', 'chrf', 'cider', \
    'meteor', 'summary_length', 'percentage_novel_1-gram', \
    'percentage_novel_2-gram', 'percentage_novel_3-gram', \
    'percentage_repeated_1-gram_in_summ', 'percentage_repeated_2-gram_in_summ', \
    'percentage_repeated_3-gram_in_summ', 'coverage', 'compression', 'density']
table_names = ['ROUGE-1 ', 'ROUGE-2 ', 'ROUGE-3  ', 'ROUGE-4 ', 'ROUGE-L  ', \
    'ROUGE-su* ', 'ROUGE-w  ', 'ROUGE-we-1 ', \
    'ROUGE-we-2 ', 'ROUGE-we-3  ', '$S^3$-pyr ', '$S^3$-resp  ', \
    'BertScore-p ', 'BertScore-r ', 'BertScore-f  ', 'MoverScore ', \
    'SMS  ', 'SummaQA\\^ ', 'BLANC', 'SuPERT', 'BLEU  ', 'CHRF  ', 'CIDEr  ', \
    'METEOR  ', 'Length\\^  ', 'Novel unigram\\^ ', \
    'Novel bi-gram\\^ ', 'Novel tri-gram\\^  ', 'Repeated unigram\\^ ', \
    'Repeated bi-gram\\^ ', 'Repeated tri-gram\\^  ', \
    'Stats-coverage\\^ ', 'Stats-compression\\^ ', 'Stats-density\\^ ']

article2humanscores = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
article2systemscores = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

metrics = set()
articles = set()
with open(args.input_file) as inputf:
    for line_count, line in enumerate(inputf):
        # if line_count >= 1600:
        #     break
        data = json.loads(line)
        curid = data['id']
        summ_id = data['model_id']
        if args.only_extractive and summ_id not in extractive_ids:
            continue
        if args.only_abstractive and summ_id not in abstractive_ids:
            continue

        summ_ids.add(summ_id)
        articles.add(curid)

        annotations = data[args.annotators_str]
        coh = [x["coherence"] for x in annotations]
        con = [x["consistency"] for x in annotations]
        flu = [x["fluency"] for x in annotations]
        rel = [x["relevance"] for x in annotations]
        #annotations_mean = np.mean(annotations, axis=0).tolist()
        article2humanscores[curid][summ_id]["coherence"] = np.mean(coh)
        article2humanscores[curid][summ_id]["consistency"] = np.mean(con)
        article2humanscores[curid][summ_id]["fluency"] = np.mean(flu)
        article2humanscores[curid][summ_id]["relevance"] = np.mean(rel)

        scores = data[f'metric_scores_{args.subset}']
        for key1, val1 in scores.items():
            if key1 == "id":
                continue
            # supert returned a list of length 1
            if key1 == "supert":
                article2systemscores[curid][summ_id][key1] = val1[0]
                metrics.add(key1)
            elif key1 == "rouge":
                for key2, val2 in scores["rouge"].items():
                    article2systemscores[curid][summ_id][key2] = val2
                    metrics.add(key2)
            else:
                article2systemscores[curid][summ_id][key1] = val1
                metrics.add(key1)

summ_ids = list(summ_ids)
summ_ids = sorted(summ_ids, key=lambda x: int("".join([i for i in x if i.isdigit()])))
metric2table = {}
sorted_metrics = sorted(list(metrics))
articles = list(articles)
max_coherence, max_consistency, max_fluency, max_relevance = [], [], [], []
for metric in sorted_metrics:
    if metric == "id":
        continue
    coherence_scores, consistency_scores, fluency_scores, relevance_scores = [], [], [], []
    metric_scores = []
    for summ_id in summ_ids:
        cur_metric = []
        cur_coherence, cur_consistency, cur_fluency, cur_relevance = [], [], [], []
        for article in articles:
            try:
                cur_metric.append(article2systemscores[article][summ_id][metric])
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()
            cur_coherence.append(article2humanscores[article][summ_id]["coherence"])
            cur_consistency.append(article2humanscores[article][summ_id]["consistency"])
            cur_fluency.append(article2humanscores[article][summ_id]["fluency"])
            cur_relevance.append(article2humanscores[article][summ_id]["relevance"])

        metric_scores.append(np.mean(cur_metric))

        coherence_scores.append(np.mean(cur_coherence))
        consistency_scores.append(np.mean(cur_consistency))
        fluency_scores.append(np.mean(cur_fluency))
        relevance_scores.append(np.mean(cur_relevance))

    coherence_corr = correlator(coherence_scores, metric_scores)[0]
    consistency_corr = correlator(consistency_scores, metric_scores)[0]
    fluency_corr = correlator(fluency_scores, metric_scores)[0]
    relevance_corr = correlator(relevance_scores, metric_scores)[0]

    if not math.isnan(coherence_corr):
        pass
    else:
        import pdb;pdb.set_trace()
    if not math.isnan(consistency_corr):
        pass
    else:
        import pdb;pdb.set_trace()
    if not math.isnan(fluency_corr):
        pass
    else:
        import pdb;pdb.set_trace()
    if not math.isnan(relevance_corr):
        pass
    else:
        import pdb;pdb.set_trace()

    coherence_corrs_final = format(round(coherence_corr, 4), ".4f")
    consistency_corrs_final = format(round(consistency_corr, 4), ".4f")
    fluency_corrs_final = format(round(fluency_corr, 4), ".4f")
    relevance_corrs_final = format(round(relevance_corr, 4), ".4f")

    if metric in sorted_keys:
        max_coherence.append(coherence_corrs_final)
        max_consistency.append(consistency_corrs_final)
        max_fluency.append(fluency_corrs_final)
        max_relevance.append(relevance_corrs_final)

    corr_list = [coherence_corrs_final, consistency_corrs_final, fluency_corrs_final, relevance_corrs_final]

    metric2table[metric] = corr_list

# get the max so we can make it bold in the table
max_coherence_ = sorted(max_coherence, key=lambda x: abs(float(x)), reverse=True)[:5]
max_consistency_ = sorted(max_consistency, key=lambda x: abs(float(x)), reverse=True)[:5]
max_fluency_ = sorted(max_fluency, key=lambda x: abs(float(x)), reverse=True)[:5]
max_relevance_ = sorted(max_relevance, key=lambda x: abs(float(x)), reverse=True)[:5]


# Output to table format
key2name = {sorted_keys[x]: table_names[x] for x in range(len(sorted_keys))}
print("==============================================")
print("Table 2")
for key, table_name in zip(sorted_keys, table_names):
    if key == "id":
        continue
    try:
        corrs = metric2table[key]
        # if corrs[0] in max_coherence_:
        #     corrs[0] = "\\textbf{" + corrs[0] + "}"
        # if corrs[1] in max_consistency_:
        #     corrs[1] = "\\textbf{" + corrs[1] + "}"
        # if corrs[2] in max_fluency_:
        #     corrs[2] = "\\textbf{" + corrs[2] + "}"
        # if corrs[3] in max_relevance_:
        #     corrs[3] = "\\textbf{" + corrs[3] + "}"
        corr_list_str = " & ".join(corrs)
        out_str = table_name +  " & " + corr_list_str + "  \\\\ \\hline"
        print(out_str)
    except Exception as e:
        continue

print("==============================================")
groups = [['rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_f_score'], \
    ['rouge_4_f_score', 'rouge_l_f_score'], ['rouge_su*_f_score', 'rouge_w_1.2_f_score'], \
    ['rouge_we_1_f', 'rouge_we_2_f', 'rouge_we_3_f'], ['s3_pyr', 's3_resp'], \
    ['bert_score_precision', 'bert_score_recall', 'bert_score_f1'], ['mover_score'], \
    ['sentence_movers_glove_sms'], ['summaqa_avg_fscore'], ['blanc'], ['supert'], ['bleu'], \
    ['chrf'], ['cider'], ['meteor'], ['summary_length'], \
    ['percentage_novel_1-gram', 'percentage_novel_2-gram', 'percentage_novel_3-gram'], \
    ['percentage_repeated_1-gram_in_summ', 'percentage_repeated_2-gram_in_summ', \
    'percentage_repeated_3-gram_in_summ'], \
    ['coverage', 'compression', 'density']]
group_names = [' ROUGE-1/2/3/ ', '  ROUGE-4/L ', '  ROUGE-su*/w ', '  ROUGE-we (1/2/3) ', \
    '  S3 (pyr/resp) ', '  BertScore (p/r/f) ', '  MoverScore ', '  SMS  ', '  SummaQA\\^ ', \
    ' BLANC ', ' SuPERT ', ' BLEU  ', ' CHRF  ', ' CIDEr  ', ' METEOR  ', '  Length\\^  ', \
    '  Novel n-gram (1/2/3)\\^ ', '   Repeated n-gram (1/2/3)\\^ ', '  Stats (cov/comp/den)\\^ ']
for group, group_name in zip(groups, group_names):
    # in the cases where a metric only uses a single reference or the source,
    #  key won't be in metric2table, so copy from other table
    try:
        coherence, consistency, fluency, relevance = [], [], [], []
        for key in group:
            corrs = metric2table[key]
            # if corrs[0] in max_coherence_:
            #     corrs[0] = "\\textbf{" + corrs[0] + "}"
            # if corrs[1] in max_consistency_:
            #     corrs[1] = "\\textbf{" + corrs[1] + "}"
            # if corrs[2] in max_fluency_:
            #     corrs[2] = "\\textbf{" + corrs[2] + "}"
            # if corrs[3] in max_relevance_:
            #     corrs[3] = "\\textbf{" + corrs[3] + "}"
            coherence.append(corrs[0])
            consistency.append(corrs[1])
            fluency.append(corrs[2])
            relevance.append(corrs[3])
        group_coherence = "/".join(coherence)
        group_consistency = "/".join(consistency)
        group_fluency = "/".join(fluency)
        group_relevance = "/".join(relevance)
        group_all = [group_coherence, group_consistency, group_fluency, group_relevance]
        group_all_str = group_name + " & " +  " & ".join(group_all) + " \\\\ \\hline"
        print(group_all_str)
    except Exception as e:
        continue
