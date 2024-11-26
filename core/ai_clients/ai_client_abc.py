from abc import ABC, abstractmethod
from typing import List, Optional


class AIClientAbc(ABC):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def instance(self):
        pass

    @abstractmethod
    def embedding_function(self, embedding_model):
        pass

    @abstractmethod
    def generate_answer(self, query: str, chunks: List[str] = None, temperature: Optional[float] = 0.7) -> str:
        pass
