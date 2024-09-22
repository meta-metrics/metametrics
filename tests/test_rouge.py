import unittest
import numpy as np

from .test_util import *

from meta_metrics.metrics.rouge_metric import ROUGEMetric

class TestROUGEMetric(unittest.TestCase):
    def test_score_rouge1(self):
        metric = ROUGEMetric(rouge_type='rouge1')
        result = metric.score(predictions=MT_PREDICTIONS, references=MT_REFERENCES)
        expected = [0.75, 0.5263157894736842, 0.42424242424242425]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)
                
    def test_score_rouge3(self):
        metric = ROUGEMetric(rouge_type='rouge3')
        result = metric.score(predictions=MT_PREDICTIONS, references=MT_REFERENCES)
        expected = [0.16666666666666666, 0.26666666666666666, 0.0]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)
                
    def test_score_rouge4(self):
        metric = ROUGEMetric(rouge_type='rouge4')
        result = metric.score(predictions=MT_PREDICTIONS, references=MT_REFERENCES)
        expected = [0.0, 0.15384615384615383, 0.0]
        np.testing.assert_almost_equal(np.array(result), np.array(expected), 5)

if __name__ == '__main__':
    unittest.main()
