from typing import Dict

from core.ai_clients.ai_client_abc import AIClientAbc
from core.dbs.db_abc import VectorDBAbc
from core.models.core.rag_response import ToolResponse
from core.models.enums import ToolKind


class Tool:

    def __init__(
            self,
            tool_kind: ToolKind,
            ai_client: AIClientAbc,
            vector_db: VectorDBAbc = None
    ):
        self.__tool_kind = tool_kind
        self.__ai_client = ai_client
        self.__vector_db = vector_db

    def store_docs(self, documents):
        if self.__vector_db:
            return self.__vector_db.push(documents=documents)
        else:
            raise IOError('Vector DB has not being initialized.')


    def process_query(self, query):
        tool_response = None

        if self.__tool_kind == ToolKind.RAG:
            chunk_result = self.__vector_db.retrieve(query)
            chunks = chunk_result['documents'][0]
            answer = self.__ai_client.generate_answer(query, chunks)
            tool_response = ToolResponse(
                answer=answer,
                docs=chunks,
                metadata=chunk_result['metadata'],
                distances=chunk_result['distances']
            )
        else:
            tool_response = ToolResponse(answer=self.__ai_client.generate_answer(query=query))

        return tool_response
