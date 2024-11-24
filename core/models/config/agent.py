from typing import List

from pydantic import BaseModel

from core.models.config.store import StoreProperties
from core.models.config.tool import ToolProperties


class AgentProperties(BaseModel):
    tools: List[ToolProperties]
    store: StoreProperties

    class Config:
        str_min_length = 1
        str_strip_whitespace = True
