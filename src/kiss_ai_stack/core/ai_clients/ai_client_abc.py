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
        Initialize the AI client object.

        This method should handle any required setup, such as authentication, configuration,
        or any other preparatory tasks to prepare the client for usage.

        :raises Exception: If there is an error during initialization.
        """
        pass

    @abstractmethod
    def instance(self):
        """
        Get the underlying AI client instance.

        This method should return the raw client object from the underlying AI provider.

        :return: The raw AI client instance.
        :rtype: object
        """
        pass

    @abstractmethod
    async def generate_answer(self, query: str, chunks: List[str] = None, temperature: Optional[float] = 0.7) -> str:
        """
        Asynchronously generate an answer for the given query.

        This method should process the input query and, optionally, the provided context
        to generate a response. The behavior may vary depending on the tool kind (e.g., RAG or prompt-based).

        :param query: The input query or prompt to process.
        :type query: str
        :param chunks: Contextual chunks to guide the response, if applicable (default is None).
        :type chunks: List[str], optional
        :param temperature: The randomness of the response, controlling creativity (default is 0.7).
        :type temperature: float, optional

        :return: The generated response from the AI client.
        :rtype: str

        :raises Exception: If there is an error generating the answer.
        """
        pass

    @abstractmethod
    async def destroy(self):
        """
        Asynchronously close the AI client's connection.

        This method should handle the cleanup process, such as releasing resources
        or closing any open connections, to ensure proper shutdown of the client.

        :raises Exception: If there is an error during destruction.
        """
        pass
