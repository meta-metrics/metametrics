import unittest

from .test_util import *

from meta_metrics.metrics.bert_score_metric import BERTScoreMetric

class TestBaseMetric(unittest.TestCase):
    def test_score(self):
        metric = BERTScoreMetric(model_name="microsoft/deberta-xlarge-mnli", model_metric="f1")
        result = metric.score(PREDICTIONS, REFERENCES)
        expected = [0.8979293704032898, 0.8278171420097351, 0.7889761924743652]
        
        self.assertAlmostEqual(result, expected, delta=0.0005)

if __name__ == '__main__':
    unittest.main()
