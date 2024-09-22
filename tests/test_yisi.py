import unittest
import numpy as np

from .test_util import *

from meta_metrics.metrics.yisi_metric import YiSiMetric

class TestYisiMetric(unittest.TestCase):
    def test_score(self):
        metric = YiSiMetric()
        result = metric.score(MT_PREDICTIONS, MT_REFERENCES)
        expected = [0.8144895406879522, 0.7592438764413839, 0.6555026936563987]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)

if __name__ == '__main__':
    unittest.main()
