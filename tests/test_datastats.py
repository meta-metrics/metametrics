import unittest
import numpy as np

from .test_util import *

from meta_metrics.metrics.datastats_metric import DataStatsMetric

class TestDataStatsMetric(unittest.TestCase):
    def test_score_density(self):
        metric = DataStatsMetric(stats_type="density")
        result = metric.score(predictions=[SUMM_PRED1, SUMM_PRED2], sources=[SUMM_ARTICLE, SUMM_ARTICLE])
        expected = [3.4375, 0.0]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)
    
    def test_score_repeated_1gram(self):
        metric = DataStatsMetric(stats_type="repeated_1-gram")
        result = metric.score(predictions=[SUMM_PRED1, SUMM_PRED2], sources=[SUMM_ARTICLE, SUMM_ARTICLE])
        expected = [0.25, 0.3333333333333333]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)
            
    def test_score_repeated_2gram(self):
        metric = DataStatsMetric(stats_type="repeated_2-gram")
        result = metric.score(predictions=[SUMM_PRED1, SUMM_PRED2], sources=[SUMM_ARTICLE, SUMM_ARTICLE])
        expected = [0.15384615384615385, 0.0]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)

if __name__ == '__main__':
    unittest.main()