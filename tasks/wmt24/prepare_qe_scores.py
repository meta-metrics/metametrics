from meta_metrics import MetaMetrics

import argparse
import os
import random
import pandas as pd
import numpy as np
from typing import Dict, List
 

SRC_PATH = "wmt24-metrics-inputs/metrics_inputs/txt/{}/sources/{}.{}.src.{}"
REFA_PATH = "wmt24-metrics-inputs/metrics_inputs/txt/{}/references/{}.{}.ref.refA.{}"
REFB_PATH = "wmt24-metrics-inputs/metrics_inputs/txt/{}/references/{}.{}.ref.refB.{}"
CS_REF_PATH = "wmt24-metrics-inputs/metrics_inputs/txt/{}/references/{}.{}.ref.ref1.{}"
SYSTEM_FOLDER = "wmt24-metrics-inputs/metrics_inputs/txt/{}/system_outputs/"
METADATA_PATH = "wmt24-metrics-inputs/metrics_inputs/txt/{}/metadata/{}.tsv"
METADATA_PATH_challenge_bioMQM = "wmt24-metrics-inputs/metrics_inputs/txt/{}/metadata/{}.{}.docID.csv" 


LANGUAGE_PAIRS = ['cs-uk',
 'en-cs',
 'en-de',
 'en-es',
 'en-hi',
 'en-is',
 'en-ja',
 'en-ru',
 'en-uk',
 'en-zh',
 'ja-zh']
 
METADATA_LANGUAGES = LANGUAGE_PAIRS

CHALLENGE_SETS = [
 'challenge_AfriMTE',
 'challenge_MSLC24-A',
 'challenge_MSLC24-B',
 'challenge_bioMQM',
 'challenge_dfki']
 
 
CHALLENGE_SETS_LPS = {'challenge_AfriMTE': ['ary-fr',
  'en-arz',
  'en-fr',
  'en-hau',
  'en-ibo',
  'en-kik',
  'en-luo',
  'en-som',
  'en-swh',
  'en-twi',
  'en-xho',
  'en-yor',
  'yor-en'],
 'challenge_MSLC24-A': ['en-de', 'en-es', 'ja-zh'],
 'challenge_MSLC24-B': ['en-de', 'en-es', 'ja-zh'],
 'challenge_bioMQM': ['de-en',
  'en-de',
  'en-es',
  'en-fr',
  'en-ru',
  'en-zh',
  'es-en',
  'fr-en',
  'pt-en',
  'ru-en',
  'zh-en'],
 'challenge_dfki': ['en-de', 'en-ru']}

# Loading comet model just once.
'''from comet import download_model, load_from_checkpoint

model_path = download_model("wmt21-comet-qe-mqm")
model = load_from_checkpoint(model_path)
'''

def segment_level_meta_metrics_scoring(sources, candidates):
    metrics_configs = [
        
    ]
    weights = []
    metric = MetaMetrics([metrics_configs], weights=weights)
    return metric.score(candidates, references=None, sources=sources)

#TODO: Change the function below and add your metric in order to score the translations provided
# Segment-level scoring function
def segment_level_scoring(samples: Dict[str, List[str]], metric: str):
    """ Function that takes source, translations and references along with a metric and returns
    segment level scores.
    
    :param samples: Dictionary with 'src', 'mt', 'ref' keys containing source sentences, translations and 
        references respectively.
    :param metric: String with the metric name. 
        If 'BLEU' runs sentence_bleu from sacrebleu. 
        If chrF runs chrF from sacrebleu    
    """
    if metric == "COMET-QE":
        data = [{"src": s, "mt": m} for s, m in zip(samples["src"], samples["mt"])]
        scores, _ = model.predict(data, batch_size=8, gpus=1)
    elif metric == "meta_metrics":
        sources = [s for s in samples["src"]]
        candidates = [s for s in samples["mt"]]
        scores = segment_level_meta_metrics_scoring(sources, candidates)
    elif metric == 'random':
        scores = np.random.random(size = len(samples["src"]))
    else:
        raise Exception(f"{metric} segment_scoring is not implemented!!")

    return scores


#TODO: Change the function below and add your metric in order to score the systems provided
# System-level scoring function
def system_level_scoring(samples: Dict[str, List[str]], metric: str, scores=List[float]):
    """ Function that takes source, translations and references along with a metric and returns
    system level scores.
    
    :param samples: Dictionary with 'src', 'mt', 'ref' keys containing source sentences, translations and 
        references respectively.
    :param metric: String with the metric name. 
        If 'BLEU' runs sentence_bleu from sacrebleu. 
        If chrF runs chrF from sacrebleu
    :param scores: List with segment level scores coming from the segment_level_scoring function.  
        Change this function if your metric DOES NOT use a simple average across segment level scores   
    """
    return sum(scores)/len(scores)



def read_data(testset_name: str, language_pair: str):
    src_lang, trg_lang = language_pair.split("-")
    testset_type = 'challengesets2024' if testset_name.startswith('challenge') else testset_name
    
    sources = [s.strip() for s in open(SRC_PATH.format(testset_type, testset_name, language_pair, src_lang)).readlines()]
    references = {}

    if os.path.isfile(REFA_PATH.format(testset_type, testset_name, language_pair, trg_lang)):
        references["refA"] = [s.strip() for s in open(REFA_PATH.format(testset_type, testset_name, language_pair, trg_lang)).readlines()]
        assert len(references["refA"]) == len(sources)

    if os.path.isfile(REFB_PATH.format(testset_type, testset_name, language_pair, trg_lang)):
        references["refB"] = [s.strip() for s in open(REFB_PATH.format(testset_type, testset_name, language_pair, trg_lang)).readlines()]
        assert len(references["refB"]) == len(sources)
        
    if os.path.isfile(CS_REF_PATH.format(testset_type, testset_name, language_pair, trg_lang)):
        references["ref1"] = [s.strip() for s in open(CS_REF_PATH.format(testset_type, testset_name, language_pair, trg_lang)).readlines()]
        assert len(references["ref1"]) == len(sources)
        
    lp_systems = [
        (SYSTEM_FOLDER.format(testset_type) +s,  ".".join(s.split(".")[3:-1]))
        for s in os.listdir(SYSTEM_FOLDER.format(testset_type) ) if language_pair in s and testset_name in s
    ] 

    system_outputs = {}
    for system_path, system_name in lp_systems:
        if "ref" in system_name:
            continue
        system_outputs[system_name] = [s.strip() for s in open(system_path).readlines()]
        assert len(system_outputs[system_name]) == len(sources)

    
    if  testset_type == 'generaltest2024':
        metadata = [s.strip().split() for s in open(METADATA_PATH.format(testset_type, language_pair)).readlines()]
        assert len(metadata) == len(sources)
    elif testset_name == 'challenge_bioMQM':
        metadata = [('all', s.strip()) for s in open(METADATA_PATH_challenge_bioMQM.format(testset_type, testset_name, language_pair)).readlines()]
        assert len(metadata) == len(sources)
    else:
        metadata = None
    
    return sources, references, system_outputs, metadata


def segment_scores(source, references, system_outputs, metadata, language_pair, metric_name, testset="generaltest2024"):
    segment_scores = []
    system_scores = []
    all_domains = set()
    
    for hyp in system_outputs:
        print (f"Scoring {testset}-{language_pair} system {hyp} with src:")
        samples = {"src": source, "mt": system_outputs[hyp]}
        scores = segment_level_scoring(samples, metric_name)
        assert len(scores) == len(system_outputs[hyp])
        # Save Segment Scores
        for i in range(len(source)):
            if metadata is not None:
                domain = metadata[i][0]
                all_domains.add(domain)
                document = metadata[i][1]
            else:
                domain = "all"
                document = "-"
                
            segment_scores.append({
                "METRIC": metric_name,
                "LANG-PAIR": language_pair,
                "TESTSET": testset,
                "DOMAIN": domain,
                "DOCUMENT": document,
                "REFERENCE": "src",
                "SYSTEM_ID": hyp,
                "SEGMENT_ID": i+1,
                "SEGMENT_SCORE": scores[i]
            })

        # Compute and save System scores for all domains.
        system_score = system_level_scoring(samples, metric_name, scores)            
        system_scores.append({
            "METRIC": metric_name,
            "LANG-PAIR": language_pair,
            "TESTSET": testset,
            "DOMAIN": "all",
            "REFERENCE": "src",
            "SYSTEM_ID": hyp,
            "SYSTEM_LEVEL_SCORE": system_score
        })

        # Compute and save System scores for each domain.
        if metadata is not None:
            for domain in all_domains:
                domain_idx = [i for i in range(len(metadata)) if metadata[i][0] == domain]
                domain_src = [source[idx] for idx in domain_idx]
                domain_hyp  = [system_outputs[hyp][idx] for idx in domain_idx]
                domain_scores  = [scores[idx] for idx in domain_idx]
                domain_samples = {"src": domain_src, "mt": domain_hyp}
                system_score = system_level_scoring(domain_samples, metric_name, domain_scores)
                system_scores.append({
                    "METRIC": metric_name,
                    "LANG-PAIR": language_pair,
                    "TESTSET": testset,
                    "DOMAIN": domain,
                    "REFERENCE": "src",
                    "SYSTEM_ID": hyp,
                    "SYSTEM_LEVEL_SCORE": system_score
                })
                
    for ref in references.keys():
        print (f"Scoring {testset}-{language_pair} system {ref} with src:")
        samples = {"src": source, "mt": references[ref]}

        # Compute and Save Segment Scores
        scores = segment_level_scoring(samples, metric_name)
        for i in range(len(source)):
                    
            if metadata is not None:
                domain = metadata[i][0]
                document = metadata[i][1]
            else:
                domain = "all"
                document = "-"
                    
            segment_scores.append({
                "METRIC": metric_name,
                "LANG-PAIR": language_pair,
                "TESTSET": testset,
                "DOMAIN": domain,
                "DOCUMENT": document,
                "REFERENCE": "src",
                "SYSTEM_ID": ref,
                "SEGMENT_ID": i+1,
                "SEGMENT_SCORE": scores[i]
            })

        # Compute and save System scores for all domains.
        system_score = system_level_scoring(samples, metric_name, scores)           
        system_scores.append({
            "METRIC": metric_name,
            "LANG-PAIR": language_pair,
            "TESTSET": testset,
            "DOMAIN": "all",
            "REFERENCE": "src",
            "SYSTEM_ID": ref,
            "SYSTEM_LEVEL_SCORE": system_score
        })
                
        # Compute and save System scores for each domain.
        if metadata is not None:
            for domain in all_domains:
                domain_idx = [i for i in range(len(metadata)) if metadata[i][0] == domain]
                domain_src = [source[idx] for idx in domain_idx]
                domain_hyp  = [references[ref][idx] for idx in domain_idx]
                domain_scores  = [scores[idx] for idx in domain_idx]
                domain_samples = {"src": domain_src, "mt": domain_hyp}
                system_score = system_level_scoring(domain_samples, metric_name, domain_scores)
                system_scores.append({
                    "METRIC": metric_name,
                    "LANG-PAIR": language_pair,
                    "TESTSET": testset,
                    "DOMAIN": domain,
                    "REFERENCE": "src",
                    "SYSTEM_ID": ref,
                    "SYSTEM_LEVEL_SCORE": system_score
                })
        
    return pd.DataFrame(segment_scores), pd.DataFrame(system_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scores Newstest2020 segments."
    )
    parser.add_argument(
        "--baseline",
        help="Metric to run.",
        type=str,
    )
    args = parser.parse_args()
    segment_data, system_data = [], []

    
            
    for challengeset_name, lps in CHALLENGE_SETS_LPS.items():
        for lp in lps:
            source, references, system_outputs, metadata = read_data(challengeset_name, lp)
            #print('todo', lp, challengeset_name, system_outputs.keys())
            segments, systems = segment_scores(source, references, system_outputs, metadata, lp, args.baseline, testset=challengeset_name)
            segment_data.append(segments)
            system_data.append(systems)

    for lp in LANGUAGE_PAIRS:
        source, references, system_outputs, metadata = read_data('generaltest2024', lp)
        segments, systems = segment_scores(source, references, system_outputs, metadata, lp, args.baseline)
        segment_data.append(segments)
        system_data.append(systems)
    
    segment_data = pd.concat(segment_data, ignore_index=True)
    segment_data.to_csv("scores/qe-as-a-metric/{}.seg.score".format(args.baseline), index=False, header=False, sep="\t")
    
    system_data = pd.concat(system_data, ignore_index=True)
    system_data.to_csv("scores/qe-as-a-metric/{}.sys.score".format(args.baseline), index=False, header=False, sep="\t")

