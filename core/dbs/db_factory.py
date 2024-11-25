from core.dbs.db_abc import VectorDBAbc
from core.dbs.vendors.chroma_db import ChromaVectorDB
from core.models.config import VectorDBProperties
from core.models.enums.db_vendor import VectorDBVendor


class VectorDBFactory:

    @staticmethod
    def get_vector_db(
            embedding_function,
            collection_name,
            properties: VectorDBProperties
    ) -> VectorDBAbc | None:
        match properties.kind:
            case VectorDBVendor.CHROMA:
                return ChromaVectorDB(
                    collection_name=collection_name,
                    embedding_function= embedding_function,
                    properties=properties
                )
        return None
