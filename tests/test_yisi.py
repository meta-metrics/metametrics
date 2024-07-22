import unittest

from test_util import *

from meta_metrics.metrics.yisi_metric import YiSiMetric

class TestBaseMetric(unittest.TestCase):
    def test_score(self):
        metric = YiSiMetric()
        result = metric.score(PREDICTIONS, REFERENCES)
        expected = [0.8144895406879522, 0.7592438764413839, 0.6555026936563987]
        
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
