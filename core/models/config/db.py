from typing import Optional

from pydantic import BaseModel

from core.models.enums import VectorDBKind


class VectorDBProperties(BaseModel):
    provider: str
    kind: VectorDBKind
    path: Optional[str] = None

    class Config:
        str_min_length = 1
        str_strip_whitespace = True
        populate_by_name = True
