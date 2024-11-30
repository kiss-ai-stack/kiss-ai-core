from typing import List, Dict

import chromadb
from chromadb.api.client import Client
from chromadb.api.models.Collection import Collection

from core.dbs.db_abc import VectorDBAbc
from core.models.config import VectorDBProperties
from core.utilities.logger import LOG


class ChromaVectorDB(VectorDBAbc):
    """
    ChromaDB-based implementation of the VectorDBAbc interface.

    This class provides methods for initializing a ChromaDB client, 
    adding documents with optional metadata, and retrieving results 
    using a specified embedding function.
    """

    def __init__(self, collection_name: str, embedding_function, properties: VectorDBProperties):
        """
        Initialize the ChromaVectorDB instance.

        Args:
            collection_name (str): The name of the collection to be created or accessed in ChromaDB.
            embedding_function: The embedding function to be used for storing and querying documents.
            properties (VectorDBProperties): Configuration properties for connecting to ChromaDB.
        """
        self.__collection_name = collection_name
        self.__embedding_function = embedding_function
        self.__properties = properties
        self.__client: Client | None = None
        self.__collection: Collection | None = None

        LOG.debug(f'ChromaVectorDB :: Initialized ChromaVectorDB with collection name \'{self.__collection_name}\'.')

    def initialize(self):
        """
        Initialize the ChromaDB client and collection.

        This method sets up the ChromaDB client and ensures the specified 
        collection exists, creating it if necessary.
        """
        LOG.info('ChromaVectorDB :: Initializing ChromaDB client...')
        try:
            self.__client = chromadb.Client()
            self.__collection = self.__client.get_or_create_collection(
                name=self.__collection_name,
                embedding_function=self.__embedding_function
            )
            LOG.info(
                f'ChromaVectorDB :: ChromaDB client initialized successfully. Collection \'{self.__collection_name}\' is ready.')
        except Exception as e:
            LOG.error(f'ChromaVectorDB :: Error initializing ChromaDB client')
            raise e

    def push(self, documents: List[str], metadata_list: List[Dict] = None) -> List[str]:
        """
        Add documents and optional metadata to the ChromaDB collection.

        Args:
            documents (List[str]): A list of document texts to add to the collection.
            metadata_list (List[Dict], optional): A list of metadata dictionaries 
                corresponding to each document. If None, empty metadata dictionaries 
                will be used. Defaults to None.

        Returns:
            List[str]: A list of unique identifiers for the added documents.
        """
        LOG.info(f'ChromaVectorDB :: Pushing {len(documents)} documents to collection \'{self.__collection_name}\'.')
        try:
            id_count = self.__collection.count()
            ids = [str(i) for i in range(id_count, id_count + len(documents))]
            self.__collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadata_list if metadata_list else [{}] * len(documents)
            )
            LOG.debug(f'ChromaVectorDB :: Documents pushed successfully with IDs: ****')
            return ids
        except Exception as e:
            LOG.error(f'ChromaVectorDB :: Error pushing documents to collection \'{self.__collection_name}\'')
            raise e

    def retrieve(self, query: str, k: int = 10) -> dict:
        """
        Retrieve the top-k documents relevant to the given query.

        Args:
            query (str): The query text to search for in the collection.
            k (int): The number of top results to retrieve. Defaults to 10.

        Returns:
            dict: A dictionary containing the retrieved documents and their metadata.
        """
        LOG.info(f'ChromaVectorDB :: Retrieving top {k} results from collection \'{self.__collection_name}\'.')
        try:
            results = self.__collection.query(
                query_texts=[query],
                n_results=k
            )
            LOG.debug(f'ChromaVectorDB :: Retrieve operation successful.')
            return results
        except Exception as e:
            LOG.error(f'ChromaVectorDB :: Error retrieving results from collection \'{self.__collection_name}\'')
            raise e
