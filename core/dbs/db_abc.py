from abc import ABC, abstractmethod
from typing import Dict, List


class VectorDBAbc(ABC):
    """
    Abstract base class for vector database implementations.

    This class defines the interface for initializing the database,
    pushing documents with optional metadata, and retrieving results based on a query.
    """

    @abstractmethod
    def initialize(self):
        """
        Initialize the vector database.

        This method should handle the setup of the database connection
        and prepare any necessary configurations for use.
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
