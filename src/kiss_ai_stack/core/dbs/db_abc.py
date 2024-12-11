from abc import ABC, abstractmethod
from typing import Dict, List

from kiss_ai_stack.core.models.enums.ai_client_vendor import AIClientVendor


class VectorDBAbc(ABC):
    """
    Abstract base class for vector database implementations.

    This class defines the interface for initializing the database,
    pushing documents with optional metadata, and retrieving results based on a query.
    """

    @abstractmethod
    def initialize(self, embedding_api_key: str, embedding_model: str, ai_vendor: AIClientVendor):
        """
        Initialize the vector database.

        This method should handle the setup of the database connection
        and prepare any necessary configurations for use.

        Args:
            ai_vendor(str): AI provider, ex: openai.
            embedding_model(str): Embedding generation model.
            embedding_api_key(AIClientVendor): AI's API key for embeddings generation.
        """
        pass

    @abstractmethod
    def push(self, documents: List[str], metadata_list: List[Dict] = None):
        """
        Add documents and optional metadata to the vector database.

        Args:
            documents (List[str]): A list of document texts to store in the database.
            metadata_list (List[Dict], optional): A list of metadata dictionaries corresponding
                to each document. Defaults to None.

        Returns:
            List[str]: A list of unique identifiers for the added documents.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 4):
        """
        Retrieve the top-k documents relevant to the query.

        Args:
            query (str): The query text to search for in the vector database.
            k (int): The number of top results to retrieve. Defaults to 4.

        Returns:
            dict: A dictionary containing the retrieved documents and associated metadata.
        """
        pass

    @abstractmethod
    def destroy(self):
        """
        Removes the collection in Vector Db.
        """
        pass
