import unittest
import numpy as np

from .test_util import *

from meta_metrics.metrics.bart_score_metric import BARTScoreMetric

class TestBARTScoreMetric(unittest.TestCase):
    def test_score(self):
        metric = BARTScoreMetric()
        result = metric.score(MT_PREDICTIONS, MT_REFERENCES)
        expected = [0.8979293704032898, 0.8278171420097351, 0.7889761924743652]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)

if __name__ == '__main__':
    unittest.main()
