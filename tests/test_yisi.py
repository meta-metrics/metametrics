import unittest

from test_util import *

from meta_metrics.metrics.yisi_metric import YiSiMetric

class TestBaseMetric(unittest.TestCase):
    def test_score(self):
        metric = YiSiMetric()
        result = metric.score(PREDICTIONS, REFERENCES)
        expected = [0.752079, 0.593061, 0.406926]
        
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
