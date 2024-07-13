import unittest

from test_util import *

from meta_metrics.metrics.comet_metric import COMETMetric

class TestBaseMetric(unittest.TestCase):
    def test_score_comet(self):
        metric = COMETMetric(comet_model="Unbabel/wmt22-comet-da") 
        result = metric.score(PREDICTIONS, REFERENCES, SOURCES)
        expected = [0.9187837839126587, 0.8417137861251831, 0.7982953786849976]
        
        self.assertEqual(result, expected)
        
    def test_score_xcomet(self):
        metric = COMETMetric(comet_model="Unbabel/XCOMET-XXL") 
        result = metric.score(PREDICTIONS, REFERENCES, SOURCES)
        expected = [0.9615516662597656, 0.8727560043334961, 0.9395455121994019]
        
        self.assertEqual(result, expected)
    
    def test_score_cometkiwi(self):
        metric = COMETMetric(comet_model="Unbabel/wmt22-cometkiwi-da") 
        result = metric.score(PREDICTIONS, REFERENCES, SOURCES)
        expected = [0.83762127161026, 0.783673882484436, 0.8230524063110352]
        
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
