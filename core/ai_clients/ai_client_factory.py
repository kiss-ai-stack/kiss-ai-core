from core.ai_clients.ai_client_abc import AIClientAbc
from core.ai_clients.vendors.openai_client import OpenAIClient
from core.models.config import AIClientProperties
from core.models.enums.ai_client_vendor import AIClientVendor


class AIClientFactory:

    @staticmethod
    def get_ai_client(properties: AIClientProperties) -> AIClientAbc | None:
        match properties.provider:
            case AIClientVendor.OPENAI:
                return OpenAIClient(properties)
        return None
