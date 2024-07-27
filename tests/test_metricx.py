import unittest

from .test_util import *

from meta_metrics.metrics.metricx_metric import MetricXMetric

class TestBaseMetric(unittest.TestCase):
    def test_score(self):
        metric = MetricXMetric(is_qe=False, tokenizer_name="google/mt5-xl", model_name="google/metricx-23-xl-v2p0",
                               batch_size=8, max_input_length=1024)
        result = metric.score(PREDICTIONS, REFERENCES)
        # expected = [0.8144895406879522, 0.7592438764413839, 0.6555026936563987]
        
        self.assertAlmostEqual(result, expected, delta=0.0005)

if __name__ == '__main__':
    unittest.main()