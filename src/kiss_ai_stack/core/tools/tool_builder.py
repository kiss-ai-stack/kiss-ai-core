from kiss_ai_stack.core.ai_clients.ai_client_factory import AIClientFactory
from kiss_ai_stack.core.dbs.db_factory import VectorDBFactory
from kiss_ai_stack.core.models.config.db import VectorDBProperties
from kiss_ai_stack.core.models.config.tool import ToolProperties
from kiss_ai_stack.core.models.enums.tool_kind import ToolKind
from kiss_ai_stack.core.tools.tool import Tool
from kiss_ai_stack.core.utilities.logger import LOG


class ToolBuilder:
    """
    A builder class responsible for constructing instances of `Tool` based on provided configuration properties.

    The `ToolBuilder` class abstracts the process of initializing and configuring tools, including their associated AI clients
    and optional vector databases. This class is used to build tools based on specified `ToolProperties` and `VectorDBProperties`.
    """

    @staticmethod
    def build_tool(
            tool_properties: ToolProperties,
            vector_db_properties: VectorDBProperties
    ) -> Tool:
        """
        Builds a `Tool` instance based on the provided tool and vector DB properties.

        This method uses the `AIClientFactory` to initialize the AI client and the `VectorDBFactory` to initialize
        the vector database (if the tool kind requires it). Once initialized, it returns a `Tool` instance with
        the appropriate configuration.

        Args:
            tool_properties (ToolProperties): The properties defining the tool, including AI client and tool kind.
            vector_db_properties (VectorDBProperties): The properties for the vector database if the tool requires one.

        Returns:
            Tool: A configured `Tool` instance, either with or without a vector database, depending on the tool kind.
        """
        try:
            LOG.info(f'Tool Builder :: Building {tool_properties.name}, kind: {tool_properties.kind}')
            ai_client = AIClientFactory.get_ai_client(
                tool_properties.ai_client, tool_properties.kind
            )
            ai_client.initialize()

            if tool_properties.kind == ToolKind.RAG:
                LOG.info(f'Tool Builder :: Initializing Vector DB for {tool_properties.name}')
                vector_db = VectorDBFactory.get_vector_db(
                    collection_name=f'{tool_properties.name}_collection',
                    properties=vector_db_properties
                )
                vector_db.initialize(
                    embedding_api_key=tool_properties.ai_client.api_key,
                    embedding_model=tool_properties.embeddings,
                    ai_vendor=tool_properties.ai_client.provider)
                LOG.info(f'Tool Builder :: Tool {tool_properties.name} built successfully with RAG capabilities.')
                return Tool(
                    tool_kind=tool_properties.kind,
                    ai_client=ai_client,
                    vector_db=vector_db
                )
            else:
                LOG.info(f'Tool Builder :: Tool {tool_properties.name} built successfully without RAG capabilities.')
                return Tool(
                    tool_kind=tool_properties.kind,
                    ai_client=ai_client
                )
        except Exception as e:
            LOG.error(f'Tool Builder :: Error occurred while building the tool {tool_properties.name}: {str(e)}')
            raise e
