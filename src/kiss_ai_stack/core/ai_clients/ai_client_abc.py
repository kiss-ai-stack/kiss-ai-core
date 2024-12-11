from abc import ABC, abstractmethod
from typing import List, Optional


class AIClientAbc(ABC):
    """
    Abstract base class for AI client implementations.

    This class defines the necessary methods that all AI clients must implement
    to ensure consistent behavior across different AI providers.
    """

    @abstractmethod
    def initialize(self):
        """
        Initialize the AI client.

        This method should handle any required setup, such as authentication or
        configuration, to prepare the client for usage.
        """
        pass

    @abstractmethod
    def instance(self):
        """
        Get the underlying AI client instance.

        Returns:
            The raw client object from the underlying AI provider.
        """
        pass

    @abstractmethod
    def generate_answer(self, query: str, chunks: List[str] = None, temperature: Optional[float] = 0.7) -> str:
        """
        Generate an answer for the given query.

        Args:
            query (str): The input query or prompt to process.
            chunks (List[str], optional): Contextual chunks to guide the response, if applicable.
            temperature (float, optional): The randomness of the response. Defaults to 0.7.

        Returns:
            str: The generated response from the AI client.
        """
        pass

    @abstractmethod
    def destroy(self):
        """
        Close AI client's connection.
        """
        pass
