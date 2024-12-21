from typing import List

from pydantic import BaseModel

from kiss_ai_stack.core.models.config.vdb_props import VectorDBProperties
from kiss_ai_stack.core.models.config.tool_props import ToolProperties


class StackProperties(BaseModel):
    decision_maker: ToolProperties
    tools: List[ToolProperties]
    vector_db: VectorDBProperties

    class Config:
        str_min_length = 1
        str_strip_whitespace = True
