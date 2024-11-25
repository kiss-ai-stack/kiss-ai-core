from typing import List

from pydantic import BaseModel

from core.models.config.db import VectorDBProperties
from core.models.config.tool import ToolProperties


class AgentProperties(BaseModel):
    classifier: ToolProperties
    tools: List[ToolProperties]
    vector_db: VectorDBProperties

    class Config:
        str_min_length = 1
        str_strip_whitespace = True
