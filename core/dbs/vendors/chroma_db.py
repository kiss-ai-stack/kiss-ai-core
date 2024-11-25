from typing import List, Dict

import chromadb
from chromadb.api.client import Client
from chromadb.api.models.Collection import Collection

from core.dbs.db_abc import VectorDBAbc
from core.models.config import VectorDBProperties


class ChromaVectorDB(VectorDBAbc):

    def __init__(self, collection_name: str, embedding_function, properties: VectorDBProperties):
        self.__collection_name = collection_name
        self.__embedding_function = embedding_function
        self.__properties = properties
        self.__client: Client | None = None
        self.__collection: Collection | None = None

    def initialize(self):
        self.__client = chromadb.Client()
        self.__collection = self.__client.get_or_create_collection(
            name=self.__collection_name,
            embedding_function=self.__embedding_function
        )

    def push(self, documents: List[str], metadata_list: List[Dict] = None) -> List[str]:
        id_count = self.__collection.count()
        ids = [str(i) for i in range(id_count, id_count + len(documents))]
        self.__collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadata_list if metadata_list else [{}] * len(documents)
        )
        return ids

    def retrieve(self, query: str, k: int = 4) -> dict:
        return self.__collection.query(
            query_texts=[query],
            n_results=k
        )
