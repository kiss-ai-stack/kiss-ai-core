from pydantic import BaseModel


class AIProperties(BaseModel):
    provider: str
    model: str
    host: str

    class Config:
        str_min_length = 1
        str_strip_whitespace = True
