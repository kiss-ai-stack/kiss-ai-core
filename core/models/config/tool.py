from pydantic import BaseModel

from core.models.config.ai import AIProperties
from core.models.enums import ToolKind


class ToolProperties(BaseModel):
    name: str
    role: str
    kind: ToolKind
    ai: AIProperties

    class Config:
        str_min_length = 1
        str_strip_whitespace = True
