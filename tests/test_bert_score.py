import unittest
import numpy as np

from .test_util import *

from meta_metrics.metrics.bert_score_metric import BERTScoreMetric

class TestBERTScoreMetric(unittest.TestCase):
    def test_score(self):
        metric = BERTScoreMetric(model_name="microsoft/deberta-xlarge-mnli", model_metric="f1")
        result = metric.score(MT_PREDICTIONS, MT_REFERENCES)
        expected = [0.8979293704032898, 0.8278171420097351, 0.7889761924743652]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)

if __name__ == '__main__':
    unittest.main()
