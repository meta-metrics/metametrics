from typing import List, Union

class BaseMetric:
    def score(self, predictions: List[str], references: Union[None, List[str]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        raise NotImplementedError()