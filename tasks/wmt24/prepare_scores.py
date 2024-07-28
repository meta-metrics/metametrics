from meta_metrics import MetaMetrics

import argparse
import os

import pandas as pd
from sacrebleu import corpus_bleu, corpus_chrf, sentence_bleu, sentence_chrf
import numpy as np
from tqdm import tqdm
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
    if metric == "chrF":
        scores = run_sentence_chrf(samples["mt"], samples["ref"])
        
    elif metric == "BLEU":
        scores = run_sentence_bleu(samples["mt"], samples["ref"])
        
    elif metric == "random":
        scores = np.random.random(size = len(samples["ref"]))
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
    if metric == "chrF":
        system_score = corpus_chrf(samples["mt"], [samples["ref"]]).score

    elif metric == "BLEU":
        system_score = corpus_bleu(samples["mt"], [samples["ref"],]).score
            
    else:
        system_score = sum(scores)/len(scores)

    return system_score


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
 
def corpus_meta_metrics(candidates: list, references: list) -> list:
    scores = run_sentence_meta_metrics(candidates, references)
    avg = np.mean(np.array(scores))
    return float(avg)

def run_sentence_meta_metrics(candidates: list, references: list) -> list:
    metrics_configs = [
        
    ]
    weights = []
    metric = MetaMetrics([metrics_configs], weights=weights)
    return metric.score(candidates, references, sources=None)
 
def run_sentence_bleu(candidates: list, references: list) -> list:
    """ Runs sentence BLEU from Sacrebleu. """
    assert len(candidates) == len(references)
    bleu_scores = []
    for i in tqdm(range(len(candidates)), desc="Running BLEU..."):
        bleu_scores.append(sentence_bleu(candidates[i], [references[i]]).score)
    return bleu_scores


def run_sentence_chrf(candidates: list, references: list) -> list:
    """ Runs sentence chrF from Sacrebleu. """
    assert len(candidates) == len(references)
    chrf_scores = []
    for i in tqdm(range(len(candidates)), desc="Running chrF..."):
        chrf_scores.append(
            sentence_chrf(hypothesis=candidates[i], references=[references[i]]).score
        )
    return chrf_scores


def segment_scores(source, references, system_outputs, metadata, language_pair, metric_name, testset="generaltest2024"):
    segment_scores = []
    system_scores = []
    all_domains = set()
    for ref in references:
        for hyp in system_outputs:
            print (f"Scoring {testset}-{language_pair} system {hyp} with {ref}:")
            samples = {"src": source, "mt": system_outputs[hyp], "ref": references[ref]}
            scores = segment_level_scoring(samples, metric_name)
            assert len(scores) == len(references[ref])
            assert len(references[ref]) == len(system_outputs[hyp])
            assert len(system_outputs[hyp]) == len(source)
            
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
                    "REFERENCE": ref,
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
                "REFERENCE": ref,
                "SYSTEM_ID": hyp,
                "SYSTEM_LEVEL_SCORE": system_score
            })

            # Compute and save System scores for each domain.
            if metadata is not None:
                for domain in all_domains:
                    domain_idx = [i for i in range(len(metadata)) if metadata[i][0] == domain]
                    domain_src = [source[idx] for idx in domain_idx]
                    domain_ref = [references[ref][idx] for idx in domain_idx]
                    domain_hyp  = [system_outputs[hyp][idx] for idx in domain_idx]
                    domain_scores  = [scores[idx] for idx in domain_idx]
                    domain_samples = {"src": domain_src, "mt": domain_hyp, "ref": domain_ref}
                    system_score = system_level_scoring(domain_samples, metric_name, domain_scores)
                    system_scores.append({
                        "METRIC": metric_name,
                        "LANG-PAIR": language_pair,
                        "TESTSET": testset,
                        "DOMAIN": domain,
                        "REFERENCE": ref,
                        "SYSTEM_ID": hyp,
                        "SYSTEM_LEVEL_SCORE": system_score
                    })
                
        for alt_ref in references.keys():
            if ref != alt_ref:
                print (f"Scoring {testset}-{language_pair} system {alt_ref} with {ref}:")
                samples = {"src": source, "mt": references[alt_ref], "ref": references[ref]}
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
                        "REFERENCE": ref,
                        "SYSTEM_ID": alt_ref,
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
                    "REFERENCE": ref,
                    "SYSTEM_ID": alt_ref,
                    "SYSTEM_LEVEL_SCORE": system_score
                })
                
                # Compute and save System scores for each domain.
                if metadata is not None:
                    for domain in all_domains:
                        domain_idx = [i for i in range(len(metadata)) if metadata[i][0] == domain]
                        domain_src = [source[idx] for idx in domain_idx]
                        domain_ref = [references[ref][idx] for idx in domain_idx]
                        domain_hyp  = [references[alt_ref][idx] for idx in domain_idx]
                        domain_scores  = [scores[idx] for idx in domain_idx]
                        domain_samples = {"src": domain_src, "mt": domain_hyp, "ref": domain_ref}
                        system_score = system_level_scoring(domain_samples, metric_name, domain_scores)
                        system_scores.append({
                            "METRIC": metric_name,
                            "LANG-PAIR": language_pair,
                            "TESTSET": testset,
                            "DOMAIN": domain,
                            "REFERENCE": ref,
                            "SYSTEM_ID": alt_ref,
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
            segments, systems = segment_scores(source, references, system_outputs, metadata, lp, args.baseline, testset=challengeset_name)
            segment_data.append(segments)
            system_data.append(systems)

    for lp in LANGUAGE_PAIRS:
        source, references, system_outputs, metadata = read_data('generaltest2024', lp)
        segments, systems = segment_scores(source, references, system_outputs, metadata, lp, args.baseline)
        segment_data.append(segments)
        system_data.append(systems)
    
    segment_data = pd.concat(segment_data, ignore_index=True)
    segment_data.to_csv("scores/{}.seg.score".format(args.baseline), index=False, header=False, sep="\t")
    
    system_data = pd.concat(system_data, ignore_index=True)
    system_data.to_csv("scores/{}.sys.score".format(args.baseline), index=False, header=False, sep="\t")

