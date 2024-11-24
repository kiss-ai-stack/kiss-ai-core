from typing import Optional

from pydantic import BaseModel

from core.models.enums import StoreKind


class StoreProperties(BaseModel):
    provider: str
    kind: StoreKind
    path: Optional[str] = None

    class Config:
        str_min_length = 1
        str_strip_whitespace = True
        populate_by_name = True
