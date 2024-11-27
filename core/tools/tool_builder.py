from core.ai_clients.ai_client_factory import AIClientFactory
from core.dbs.db_factory import VectorDBFactory
from core.models.config import ToolProperties, VectorDBProperties
from core.models.enums import ToolKind
from core.tools.tool import Tool


class ToolBuilder:

    @staticmethod
    def build_tool(tool_properties: ToolProperties, vector_db_properties: VectorDBProperties) -> Tool:
        ai_client = AIClientFactory.get_ai_client(tool_properties.ai_client, tool_properties.kind)
        ai_client.initialize()

        if tool_properties.kind == ToolKind.RAG:
            return Tool(
                tool_kind=tool_properties.kind,
                ai_client=ai_client,
                vector_db=VectorDBFactory.get_vector_db(
                    collection_name=f'{tool_properties.name}_collection',
                    embedding_function=ai_client.embedding_function(tool_properties.embeddings),
                    properties=vector_db_properties
                )
            )
        else:
            return Tool(
                tool_kind=tool_properties.kind,
                ai_client=ai_client
            )
