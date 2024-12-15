from kiss_ai_stack.core.dbs.db_abc import VectorDBAbc
from kiss_ai_stack.core.dbs.vendors.chroma_db import ChromaVectorDB
from kiss_ai_stack.core.models.config.db import VectorDBProperties
from kiss_ai_stack.core.models.enums.db_vendor import VectorDBVendor


class VectorDBFactory:
    """
    A factory class for creating instances of vector database clients.

    This class provides a static method to initialize and return an appropriate
    vector database implementation based on the provided configuration.

    Supported Vendors:
    - ChromaDB
    """

    @staticmethod
    def get_vector_db(
            collection_name: str,
            properties: VectorDBProperties
    ) -> VectorDBAbc | None:
        """
        Creates and returns an instance of a vector database client.

        This method checks the `provider` attribute in the `properties` to
        determine the vector database vendor and initializes the corresponding
        database client.

        :param collection_name: The name of the collection to be accessed or created.
                                 This will be used when initializing the vector database.
        :param properties: Configuration properties for the vector database client.
                           This includes the database vendor and other settings.

        :return: An instance of a vector database client that implements the `VectorDBAbc` interface,
                 or None if no matching provider is found.

        :raises ValueError: If an unsupported database vendor is specified in the properties.

        Example:
            ```python
            properties = VectorDBProperties(provider=VectorDBVendor.CHROMA, ...)
            vector_db = VectorDBFactory.get_vector_db(
                collection_name="my_collection",
                properties=properties
            )
            ```
        """
        match properties.provider:
            case VectorDBVendor.CHROMA:
                return ChromaVectorDB(
                    collection_name=collection_name,
                    properties=properties
                )
        return None
