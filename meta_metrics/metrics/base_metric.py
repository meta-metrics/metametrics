from typing import List, Tuple

class BaseMetric:
    def score(self, predictions:List[str], references:List[str]) -> List[float]:
        raise NotImplementedError()