import unittest

from test_util import *

from meta_metrics.metrics.bert_score_metric import BERTScoreMetric

class TestBaseMetric(unittest.TestCase):
    def test_score(self):
        metric = BERTScoreMetric(None)
        pass

if __name__ == '__main__':
    unittest.main()
