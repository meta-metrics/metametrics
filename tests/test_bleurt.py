import unittest
import numpy as np

from .test_util import *

from meta_metrics.metrics.bleurt20_metric import BLEURT20Metric

class TestBLEURT20Metric(unittest.TestCase):
    def test_score(self):
        metric = BLEURT20Metric()
        result = metric.score(MT_PREDICTIONS, MT_REFERENCES)
        expected = [0.8208704590797424, 0.7630288004875183, 0.5910331010818481]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)

if __name__ == '__main__':
    unittest.main()
