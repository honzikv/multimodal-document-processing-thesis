from abc import abstractmethod
from typing import NamedTuple, List, Dict, Any


class ClassifierPreprocessor:
    """
    Base class for all classifier preprocessors
    """

    @abstractmethod
    def preprocess(self, tokens: List[str], bboxes: List[tuple], image):
        raise NotImplementedError('This method must be implemented by a subclass')
