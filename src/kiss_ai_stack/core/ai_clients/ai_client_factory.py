from kiss_ai_stack.core.ai_clients.ai_client_abc import AIClientAbc
from kiss_ai_stack.core.ai_clients.vendors.openai_client import OpenAIClient
from kiss_ai_stack.core.models.config.ai_client import AIClientProperties
from kiss_ai_stack.core.models.enums.ai_client_vendor import AIClientVendor
from kiss_ai_stack.core.models.enums.tool_kind import ToolKind


class AIClientFactory:
    """
    Factory class to create AI client instances.

    This class provides a static method to create instances of AI clients based on
    the provided configuration properties and tool kind. It simplifies the process
    of selecting the correct AI client implementation for the given provider.
    """

    @staticmethod
    def get_ai_client(properties: AIClientProperties, tool_kind: ToolKind) -> AIClientAbc | None:
        """
        Retrieve an AI client instance based on the provider specified in properties.

        Args:
            properties (AIClientProperties): The configuration properties for the AI client,
                including the provider type and API key.
            tool_kind (ToolKind): The type of tool to be used with the AI client,
                such as PROMPT or RAG.

        Returns:
            AIClientAbc | None: An instance of the appropriate AI client implementation if the
                provider is supported, or None if the provider is not recognized.

        Example:
            properties = AIClientProperties(provider=AIClientVendor.OPENAI, api_key='your_api_key')
            tool_kind = ToolKind.PROMPT
            ai_client = AIClientFactory.get_ai_client(properties, tool_kind)
        """
        match properties.provider:
            case AIClientVendor.OPENAI:
                return OpenAIClient(properties, tool_kind)
        return None
