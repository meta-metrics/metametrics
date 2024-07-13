from typing import List, Union

class BaseMetric:
    def score(self, predictions: List[str], references: List[str], sources: Union[None, List[str]]=None) -> List[float]:
        raise NotImplementedError()