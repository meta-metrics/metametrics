from typing import List, Tuple

class BaseMetric:
    def score(self, predictions:List[str], references:List[str], sources:List[str]=None) -> List[float]:
        raise NotImplementedError()