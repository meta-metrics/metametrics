import unittest
import numpy as np

from .test_util import *

from meta_metrics.metrics.summaqa_metric import SummaQAMetric

class TestSummaQAMetric(unittest.TestCase):
    def test_score_f1(self):
        metric = SummaQAMetric(model_metric="f1")
        result = metric.score(predictions=[SUMM_PRED1, SUMM_PRED2], sources=[SUMM_ARTICLE, SUMM_ARTICLE])
        expected = [0.21497435897435896, 0.0]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)
    
    def test_score_prob(self):
        metric = SummaQAMetric(model_metric="prob")
        result = metric.score(predictions=[SUMM_PRED1, SUMM_PRED2], sources=[SUMM_ARTICLE, SUMM_ARTICLE])
        expected = [0.1116120220720768, 0.020100885266438127]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)

if __name__ == '__main__':
    unittest.main()
