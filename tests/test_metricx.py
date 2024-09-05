import unittest
import numpy as np

from .test_util import *

from meta_metrics.metrics.metricx_metric import MetricXMetric

class TestMetricXMetric(unittest.TestCase):
    def test_score(self):
        metric = MetricXMetric(is_qe=False, tokenizer_name="google/mt5-xl", model_name="google/metricx-23-xl-v2p0",
                               batch_size=8, max_input_length=1024)
        result = metric.score(MT_PREDICTIONS, MT_REFERENCES)
        # expected = [0.8144895406879522, 0.7592438764413839, 0.6555026936563987]
        # np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)

if __name__ == '__main__':
    unittest.main()