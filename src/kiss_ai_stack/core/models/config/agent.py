from typing import List

from pydantic import BaseModel

from kiss_ai_stack.core.models.config.db import VectorDBProperties
from kiss_ai_stack.core.models.config.tool import ToolProperties


class AgentProperties(BaseModel):
    decision_maker: ToolProperties
    tools: List[ToolProperties]
    vector_db: VectorDBProperties

    class Config:
        str_min_length = 1
        str_strip_whitespace = True
