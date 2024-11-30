from pydantic import BaseModel

from kiss_ai_stack.core.models.enums.ai_client_vendor import AIClientVendor


class AIClientProperties(BaseModel):
    provider: AIClientVendor
    model: str
    api_key: str

    class Config:
        str_min_length = 1
        str_strip_whitespace = True
