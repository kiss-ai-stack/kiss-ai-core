from abc import ABC, abstractmethod
from typing import Dict, List


class VectorDBAbc(ABC):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def push(self, documents: List[str], metadata_list: List[Dict]=None):
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 4):
        pass
