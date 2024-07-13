import unittest

from test_util import *

from meta_metrics.metrics.bleurt20_metric import BLEURT20Metric

class TestBaseMetric(unittest.TestCase):
    def test_score(self):
        metric = BLEURT20Metric()
        result = metric.score(PREDICTIONS, REFERENCES)
        expected = [0.8208704590797424, 0.7630288004875183, 0.5910331010818481]
        
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
