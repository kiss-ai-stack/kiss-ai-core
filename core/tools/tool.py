from typing import Dict

from core.ai_clients.ai_client_abc import AIClientAbc
from core.dbs.db_abc import VectorDBAbc
from core.models.config import ToolProperties
from core.models.core.rag_response import RagResponse


class Tool:

    def __init__(
            self,
            tool_properties: ToolProperties,
            ai_client: AIClientAbc,
            vector_db: VectorDBAbc
    ):
        self.__tool_properties = tool_properties
        self.__ai_client = ai_client
        self.__vector_db = vector_db

    def store_docs(self, documents):
        self.__vector_db.push(documents=documents)

    def process_rag_pipeline(self, query: str, k: int = 4) -> RagResponse:
        chunk_result = self.__vector_db.retrieve(query, k)
        chunks = chunk_result['documents'][0]
        answer = self.__ai_client.generate_answer(query, chunks)
        return RagResponse(
            answer=answer,
            docs=chunks,
            metadata=chunk_result['metadata'],
            distances=chunk_result['distances']
        )

    def process_prompt(self, query):
        return