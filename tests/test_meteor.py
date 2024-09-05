import unittest
import numpy as np

from .test_util import *

from meta_metrics.metrics.meteor_metric import METEORMetric

class TestMETEORMetric(unittest.TestCase):
    def test_score(self):
        metric = METEORMetric()
        result = metric.score(MT_PREDICTIONS, MT_REFERENCES)
        expected = [0.7471655328798186, 0.5830583058305829, 0.3583386479591837]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)

if __name__ == '__main__':
    unittest.main()
