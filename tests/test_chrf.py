import unittest
import numpy as np

from .test_util import *

from meta_metrics.metrics.chrf_metric import chrFMetric

class TestchrFMetric(unittest.TestCase):
    def test_score(self):
        metric = chrFMetric(word_order=2, eps_smoothing=True) # Using chrF2++
        result = metric.score(MT_PREDICTIONS, MT_REFERENCES)
        expected = [61.823242741315454, 46.78990844988301, 28.56850981104568]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)

if __name__ == '__main__':
    unittest.main()
