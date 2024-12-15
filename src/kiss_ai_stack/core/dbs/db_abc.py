from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from kiss_ai_stack.core.models.enums.ai_client_vendor import AIClientVendor


class VectorDBAbc(ABC):
    """
    Abstract base class for vector database implementations.

    This class defines the interface for initializing the database,
    pushing documents with optional metadata, and retrieving results based on a query.
    """

    @abstractmethod
    async def initialize(self, embedding_api_key: str, embedding_model: str, ai_vendor: AIClientVendor, tenant: Optional[str]=None):
        """
        Initializes the vector database.

        Sets up the database connection and prepares any necessary configurations
        for use, based on the provided AI vendor, embedding model, and API key.

        :param embedding_api_key: The API key for the embedding service.
                                  This key is used to authenticate requests to the service.
        :param embedding_model: The embedding model to be used for generating embeddings.
        :param ai_vendor: The AI vendor for the embedding service (e.g., OpenAI).
        :param tenant: Preferably user's unique Id

        :return: None
        :raises Exception: If any error occurs during the initialization of the database.
        """
        pass

    @abstractmethod
    async def push(self, documents: List[str], metadata_list: List[Dict] = None):
        """
        Adds documents and optional metadata to the vector database.

        Stores the provided documents in the database, optionally associating each
        document with metadata.

        :param documents: A list of document texts to store in the database.
        :param metadata_list: An optional list of metadata dictionaries corresponding to
                              each document. Defaults to None.

        :return: A list of unique identifiers for the added documents.
        :raises Exception: If any error occurs while adding the documents to the database.
        """
        pass

    @abstractmethod
    async def retrieve(self, query: str, k: int = 4):
        """
        Retrieves the top-k documents relevant to the query.

        Searches the database for documents most relevant to the provided query and
        returns the top-k results.

        :param query: The query text to search for in the vector database.
        :param k: The number of top results to retrieve. Defaults to 4.

        :return: A dictionary containing the retrieved documents and their associated metadata.
        :raises Exception: If any error occurs during the retrieval of documents.
        """
        pass

    @abstractmethod
    async def destroy(self):
        """
        Removes the collection in the vector database.

        Deletes the entire collection of documents and metadata from the database.

        :return: None
        :raises Exception: If any error occurs during the removal of the collection.
        """
        pass
