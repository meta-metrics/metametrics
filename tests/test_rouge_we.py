import unittest
import numpy as np

from .test_util import *

from meta_metrics.metrics.rouge_we_metric import ROUGEWEMetric

class TestROUGEWEMetric(unittest.TestCase):
    def test_score_rouge_we1(self):
        metric = ROUGEWEMetric()
        result = metric.score(predictions=MT_PREDICTIONS, references=MT_REFERENCES)
        expected = [0.0, 0.04878, 0.04494]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)

if __name__ == '__main__':
    unittest.main()