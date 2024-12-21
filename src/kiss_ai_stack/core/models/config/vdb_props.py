from typing import Optional

from pydantic import BaseModel

from kiss_ai_stack.core.models.enums.db_kind import VectorDBKind


class VectorDBProperties(BaseModel):
    provider: str
    kind: VectorDBKind
    path: Optional[str] = None
    host: Optional[str]
    port: Optional[int]
    secure: Optional[bool] = True

    class Config:
        str_min_length = 1
        str_strip_whitespace = True
        populate_by_name = True
