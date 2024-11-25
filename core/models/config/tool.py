from pydantic import BaseModel

from core.models.config.ai_client import AIClientProperties
from core.models.enums import ToolKind


class ToolProperties(BaseModel):
    name: str
    role: str
    kind: ToolKind
    ai: AIClientProperties
    embeddings: str

    class Config:
        str_min_length = 1
        str_strip_whitespace = True
