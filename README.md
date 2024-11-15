# MetaMetrics V0.0.1
[MetaMetrics](https://github.com/meta-metrics/metametrics) is a tuneable, easily extensible, and explainable metric for human evaluation alignment in generation tasks.
The repository is the open-source implementation for MetaMetrics: Calibrating Metrics For Generation Tasks Using Human Preferences https://arxiv.org/pdf/2410.02381. We will release the code soon.

## Supported Tasks
The current version supports the following tasks:
1. Question Answering
2. Machine Translation
3. Image Captioning
4. Text Summarization
5. Reward Modeling

You can clone and adapt the codes to support other generation tasks.

## Supported Metrics
The current version supports the following metrics:
1. BLEU
2. BARTScore
3. BERTScore
4. BLEURT20
5. chrF
6. comet
7. MetricX
8. METEOR
9. ROUGE
10. ROUGEWE
11. SummaQA
12. YiSi
13. GEMBA_MQM
14. ClipScore
15. ArmoRM

## Installation Guide
Requires `Python 3.10+`
```
PENDING
```

## How To Use
Example use-case with MetaMetrics library:
```
CODE PENDING
```

## How To Extend New Metrics
Extending MetaMetrics to support other metrics is done by creating a Subclass of `metametrics.metrics.base_metric.BaseMetric` (Text Only Metric)
or `metametrics.metrics.base_metric.VisionToTextBaseMetric` (Vision to Text Metric)
and placing the file in `metametrics/src/metametrics/metrics/`.


The new metric must contain the following functions:
1. `NewMetric.score(self, predictions: List[str], references: Union[None,List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]`.


Checklist To Integrate Custom Metrics:
1. [ ] `metametrics/src/metametrics/metrics/__init__.py` | Import your new metric
2. [ ] `metametrics/src/metametrics/metrics/__init__.py` | Extend the `__all__` variable
3. [ ] `metametrics/src/metametrics/metametrics.py` | Import your metric
4. [ ] `metametrics/src/metametrics/metametrics.py` | Update variable `MetaMetrics.normalization_config`
5. [ ] `metametrics/src/metametrics/metametrics.py` | update function `MetaMetrics.get_metric()`