from typing import List, Dict

from kiss_ai_stack.core.models.config.db import VectorDBProperties
from kiss_ai_stack.core.models.enums.ai_client_vendor import AIClientVendor

try:
    import chromadb
    from chromadb.api.client import Client
    from chromadb.api.models.Collection import Collection
    from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
except ImportError:
    chromadb = None
    raise ImportError('ChromaDB is not installed. Please install \'chromadb\' python package `pip install chromadb`.')

from kiss_ai_stack.core.dbs.db_abc import VectorDBAbc
from kiss_ai_stack.core.utilities.logger import LOG


class ChromaVectorDB(VectorDBAbc):
    """
    ChromaDB-based implementation of the VectorDBAbc interface.

    This class provides methods for initializing a ChromaDB client, 
    adding documents with optional metadata, and retrieving results 
    using a specified embedding function.
    """

    def __init__(self,
                 collection_name: str,
                 properties: VectorDBProperties):
        """
        Initialize the ChromaVectorDB instance.

        Args:
            collection_name (str): The name of the collection to be created or accessed in ChromaDB.
            properties (VectorDBProperties): Configuration properties for connecting to ChromaDB.
        """
        self.__collection_name = collection_name
        self.__embedding_function = None
        self.__properties = properties
        self.__client: Client | None = None
        self.__collection: Collection | None = None

        LOG.debug(f'ChromaVectorDB :: Initialized ChromaVectorDB with collection name \'{self.__collection_name}\'.')

    def __init_embedding_function(self, embedding_api_key: str, embedding_model: str, ai_vendor: AIClientVendor):
        """
        Initialize an embedding function for the specified model.
        """
        LOG.info(f'ChromaVectorDB :: creating embedding function for, {ai_vendor}: {embedding_model}')
        match ai_vendor:
            case AIClientVendor.OPENAI:
                self.__embedding_function = OpenAIEmbeddingFunction(
                    api_key=embedding_api_key,
                    model_name=embedding_model
                )

    def initialize(self, embedding_api_key: str, embedding_model: str, ai_vendor: AIClientVendor):
        """
        Initialize the ChromaDB client and collection.

        This method sets up the ChromaDB client and ensures the specified 
        collection exists, creating it if necessary.

        Args:
            ai_vendor(str): AI provider, ex: openai.
            embedding_model(str): Embedding generation model.
            embedding_api_key(AIClientVendor): AI's API key for embeddings generation.
        """
        LOG.info('ChromaVectorDB :: Initializing ChromaDB client...')
        try:
            self.__client = chromadb.Client()
            self.__init_embedding_function(
                embedding_api_key=embedding_api_key,
                embedding_model=embedding_model,
                ai_vendor=ai_vendor
            )
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

    def destroy(self):
        """
        Completely delete the current ChromaDB collection.

        Raises:
            Exception: If deletion fails or collection is already None
        """
        try:
            if self.__collection is None:
                LOG.warning(f'ChromaVectorDB :: No collection \'{self.__collection_name}\' exists to delete.')
                return
            doc_count = self.__collection.count()
            LOG.info(
                f'ChromaVectorDB :: Attempting to delete collection \'{self.__collection_name}\' with {doc_count} documents.')

            self.__client.delete_collection(name=self.__collection_name)
            self.__collection = None
            self.__embedding_function = None
            LOG.info(f'ChromaVectorDB :: Collection \'{self.__collection_name}\' successfully deleted.')
        except Exception as e:
            LOG.error(f'ChromaVectorDB :: Failed to delete collection \'{self.__collection_name}\': {e}')
            raise e
