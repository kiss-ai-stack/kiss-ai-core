from typing import List, Dict, Optional, Tuple

from kiss_ai_stack.core.config import VECTOR_DB
from kiss_ai_stack.core.dbs.db_abc import VectorDBAbc
from kiss_ai_stack.core.models.config.vdb_props import VectorDBProperties
from kiss_ai_stack.core.models.enums.ai_client_vendor import AIClientVendor
from kiss_ai_stack.core.models.enums.db_kind import VectorDBKind
from kiss_ai_stack.core.models.enums.db_vendor import VectorDBVendor
from kiss_ai_stack.core.utilities import install_package
from kiss_ai_stack.core.utilities.logger import LOG


class ChromaVectorDB(VectorDBAbc):
    """
    Async-compatible ChromaDB implementation of the VectorDBAbc interface.
    """

    def __init__(self, collection_name: str, properties: VectorDBProperties):
        """
        Initialize the ChromaVectorDB instance.

        :param collection_name: The name of the collection to be created or accessed in ChromaDB.
        :param properties: Configuration properties for connecting to ChromaDB.
        """
        self.__collection_name = collection_name
        self.__embedding_function = None
        self.__properties = properties
        self.__client: Optional['AsyncHttpClient'] = None
        self.__admin_client: Optional['AdminClient'] = None
        self.__collection = None

        LOG.debug(f'ChromaVectorDB :: ChromaVectorDB initialized with collection name: \'{self.__collection_name}\'')

    def __validate_enum(self, value, enum_type, param_name: str):
        if value not in enum_type:
            raise ValueError(
                f'ChromaVectorDB :: Invalid {param_name}: {value}. Must be one of: {[e.name for e in enum_type]}')

    async def __ensure_database(self, tenant: Optional[str], database: Optional[str]) -> Optional[str]:
        from chromadb.errors import NotFoundError

        try:
            await self.__admin_client.get_database(name=database, tenant=tenant)
        except NotFoundError:
            LOG.warning(f'ChromaVectorDB :: Database :: {tenant} : {database} not fount, attempting to create')
            await self.__admin_client.create_database(name=database, tenant=tenant)
            LOG.warning(f'ChromaVectorDB :: Database :: {tenant} : {database}  created')
        finally:
            return database

    async def __ensure_tenant(self, tenant: str) -> Optional[Tuple]:
        from chromadb.errors import NotFoundError

        database = f'{tenant}_docs'
        try:
            await self.__admin_client.get_tenant(tenant)
        except NotFoundError:
            LOG.warning(f'ChromaVectorDB :: Tenant :: {tenant} not fount, attempting to create')
            await self.__admin_client.create_tenant(tenant)
            LOG.warning(f'ChromaVectorDB :: Tenant :: {tenant} created')
        finally:
            database = await self.__ensure_database(tenant=tenant, database=database)
            return tenant, database

    def __initialize_embedding_function(self, embedding_api_key: str, embedding_model: str, ai_vendor: AIClientVendor):
        """
        Initialize an embedding function for the specified model.

        :param embedding_api_key: The API key for embedding service.
        :param embedding_model: The model name to be used for embeddings.
        :param ai_vendor: The vendor providing the embedding service.
        """
        self.__validate_enum(ai_vendor, AIClientVendor, 'ai_vendor')
        LOG.info(f'ChromaVectorDB :: Creating embedding function for {ai_vendor}: {embedding_model}')

        if ai_vendor == AIClientVendor.OPENAI:
            from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

            self.__embedding_function = OpenAIEmbeddingFunction(
                api_key=embedding_api_key,
                model_name=embedding_model
            )
        else:
            raise NotImplementedError(f'ChromaVectorDB :: Unsupported embedding function type: {ai_vendor}')

    async def __initialize_client(self, tenant: Optional[str] = None):
        """
        Initialize the ChromaDB client based on the properties' configuration.
        :param tenant: Preferably user's unique Id.
        """
        try:
            from chromadb import AsyncHttpClient
            from chromadb import AdminClient, Settings
        except ImportError:
            package_name = VECTOR_DB[VectorDBVendor.CHROMA]
            LOG.warning(f'ChromaVectorDB :: ChromaDB is not installed. Attempting to auto-install {package_name}.')
            install_package(package_name)
            from chromadb import AsyncHttpClient
            from chromadb import AdminClient, Settings

        if self.__properties.kind == VectorDBKind.REMOTE:
            self.__client = await AsyncHttpClient(
                host=self.__properties.host,
                port=self.__properties.port,
                ssl=self.__properties.secure
            )
            self.__admin_client = AdminClient(settings=self.__client.get_settings())

            database = 'default_database'
            if tenant:
                tenant, database = await self.__ensure_tenant(tenant)
            else:
                tenant = 'default_tenant'
            await self.__client.set_tenant(tenant=tenant, database=database)
        else:
            raise ValueError(f'ChromaVectorDB :: Only \'REMOTE\' kind is supported for ChromaDB.')

    async def initialize(self, embedding_api_key: str, embedding_model: str, ai_vendor: AIClientVendor,
                         tenant: Optional[str] = None):
        """
        Initialize the ChromaDB client and collection asynchronously.

        :param embedding_api_key: API key for embeddings generation.
        :param embedding_model: Embedding model to use.
        :param ai_vendor: The AI provider (e.g., OpenAI) for embeddings generation.
        :param tenant: Preferably user's unique Id.

        :raises Exception: If the initialization fails for any reason.
        """
        LOG.info('ChromaVectorDB :: Initializing ChromaDB client...')

        try:
            await self.__initialize_client(tenant=tenant)
            self.__initialize_embedding_function(
                embedding_api_key=embedding_api_key,
                embedding_model=embedding_model,
                ai_vendor=ai_vendor
            )
            self.__collection = await self.__client.get_or_create_collection(
                name=self.__collection_name,
                embedding_function=self.__embedding_function,
            )

            LOG.info(
                f'ChromaVectorDB :: ChromaDB client initialized successfully. Collection \'{self.__collection_name}\' is ready.')

        except Exception as e:
            LOG.error(f'ChromaVectorDB :: Error initializing ChromaDB client: {e}')
            raise

    async def push(self, documents: List[str], metadata_list: Optional[List[Dict]] = None) -> List[str]:
        """
        Add documents and optional metadata to the ChromaDB collection asynchronously.

        :param documents: A list of document texts to add to the collection.
        :param metadata_list: A list of metadata dictionaries corresponding to each document.

        :returns: A list of unique identifiers for the added documents.
        """
        if not documents:
            raise ValueError('ChromaVectorDB :: No documents provided to push.')

        LOG.info(f'ChromaVectorDB :: Pushing {len(documents)} documents to collection \'{self.__collection_name}\'.')

        try:
            id_count = await self.__collection.count()
            ids = [str(i) for i in range(id_count, id_count + len(documents))]

            await self.__collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadata_list or [{}] * len(documents)
            )

            LOG.debug('ChromaVectorDB :: Documents pushed successfully.')
            return ids

        except Exception as e:
            LOG.error(f'ChromaVectorDB :: Error pushing documents: {e}')
            raise

    async def retrieve(self, query: str, k: int = 10) -> dict:
        """
        Retrieve the top-k documents relevant to the given query asynchronously.

        :param query: The query text to search for in the collection.
        :param k: The number of top results to retrieve. Defaults to 10.

        :returns: A dictionary containing the retrieved documents and their metadata.
        """
        if not query:
            raise ValueError('ChromaVectorDB :: Query string is empty.')

        LOG.info(f'ChromaVectorDB :: Retrieving top {k} results from collection \'{self.__collection_name}\'.')

        try:
            results = await self.__collection.query(
                query_texts=[query],
                n_results=k
            )
            LOG.debug('ChromaVectorDB :: Retrieve operation successful.')
            return results

        except Exception as e:
            LOG.error(f'ChromaVectorDB :: Error retrieving results: {e}')
            raise

    async def destroy(self):
        """
        Completely delete the current ChromaDB collection asynchronously.
        """
        if not self.__collection:
            LOG.warning(f'ChromaVectorDB :: No collection \'{self.__collection_name}\' exists to delete.')
            return

        try:
            doc_count = await self.__collection.count()
            LOG.info(f'ChromaVectorDB :: Deleting collection \'{self.__collection_name}\' with {doc_count} documents.')

            await self.__client.delete_collection(name=self.__collection_name)
            self.__collection = None
            self.__embedding_function = None

            LOG.info(f'ChromaVectorDB :: Collection \'{self.__collection_name}\' successfully deleted.')

        except Exception as e:
            LOG.error(f'ChromaVectorDB :: Failed to delete collection: {e}')
            raise
