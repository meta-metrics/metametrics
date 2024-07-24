import unittest

from .test_util import *

from meta_metrics.metrics.gemba_metric import GEMBA_MQM

class TestBaseMetric(unittest.TestCase):
    def test_score(self):
        metric = GEMBA_MQM()
        result = metric.score(PREDICTIONS, REFERENCES)
        expected = [0.8144895406879522, 0.7592438764413839, 0.6555026936563987]
        
        self.assertAlmostEqual(result, expected, delta=0.0005)

if __name__ == '__main__':
    unittest.main()